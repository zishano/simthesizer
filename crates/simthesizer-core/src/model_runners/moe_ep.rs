use std::any::Any;
use std::collections::HashSet;
use std::path::Path;

use anyhow::{anyhow, bail, Result};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::Deserialize;

use crate::compute_sim::ComputeSimulator;
use crate::engine::Graph;
use crate::model::Model;
use crate::model_runner::ModelRunner;
use crate::module::{LogicalHandler, Module};
use crate::network_sim::NetworkSimulator;
use crate::types::{
    ComputeContext, Layer, NetworkDeviceID, NodeID, ParallelStrategy, RequestState,
};

const DEFAULT_SEED: u64 = 42;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MoeExpertParallelRunner {
    ep: u32,
    link_bw: f64,
    link_latency: f64,
    compute_sim: Box<dyn ComputeSimulator>,
    network_sim: Box<dyn NetworkSimulator>,

    #[serde(skip_deserializing, default)]
    net_gpu_device_ids: Vec<NetworkDeviceID>,

    #[serde(skip_deserializing, default)]
    block_start_time: Option<f64>,

    #[serde(skip_deserializing, default)]
    rng: Option<StdRng>,
}

crate::register_model_runner!(MoeExpertParallelRunner, "moe_ep");

#[derive(Debug)]
enum LogicalPayload {
    BlockStart,
    FastForward { remaining_blocks: u32 },
    Barrier,
}

fn parse_payload(payload: Box<dyn Any>) -> Result<LogicalPayload> {
    payload
        .downcast::<LogicalPayload>()
        .map(|payload| *payload)
        .map_err(|_| anyhow!("moe_ep runner received an unexpected logical payload"))
}

#[derive(Debug, Clone)]
struct MoeRouting {
    remote_tokens: Vec<Vec<u64>>,
    local_tokens: Vec<u64>,
    activated_experts: Vec<u64>,
}

impl MoeExpertParallelRunner {
    fn moe_expert_owner(ep: u32, num_experts: u64, expert_id: u64) -> usize {
        ((expert_id * u64::from(ep)) / num_experts).min(u64::from(ep - 1)) as usize
    }

    fn rng(&mut self) -> &mut StdRng {
        self.rng
            .get_or_insert_with(|| StdRng::seed_from_u64(DEFAULT_SEED))
    }

    fn random_below(&mut self, upper: u64) -> u64 {
        debug_assert!(upper > 0);
        self.rng().gen_range(0..upper)
    }

    fn device_id(&self, rank: u32) -> NetworkDeviceID {
        self.net_gpu_device_ids[rank as usize]
    }

    fn total_tokens(batch: &[RequestState]) -> u64 {
        batch.iter().map(|request| u64::from(request.q_len)).sum()
    }

    fn synthetic_batch(tokens: u64) -> Result<Vec<RequestState>> {
        let q_len = u32::try_from(tokens)
            .map_err(|_| anyhow!("simulated token count {tokens} exceeds u32"))?;
        Ok(vec![RequestState {
            q_len,
            kv_len: 0,
            lm_head_len: 0,
        }])
    }

    fn source_tokens(&self, total_tokens: u64) -> Vec<u64> {
        let width = u64::from(self.ep);
        let base = total_tokens / width;
        let remainder = total_tokens % width;
        (0..self.ep)
            .map(|rank| base + u64::from(rank < remainder as u32))
            .collect()
    }

    fn source_tokens_from_instance_batches(
        &self,
        instance_batches: &[Vec<RequestState>],
    ) -> Vec<u64> {
        instance_batches
            .iter()
            .map(|instance_batch| Self::total_tokens(instance_batch))
            .collect()
    }

    fn sample_moe_routing(
        &mut self,
        total_tokens: u64,
        num_experts: u64,
        num_experts_per_token: u32,
        num_shared_experts_per_token: u32,
    ) -> MoeRouting {
        self.sample_moe_routing_from_source_tokens(
            self.source_tokens(total_tokens),
            num_experts,
            num_experts_per_token,
            num_shared_experts_per_token,
        )
    }

    fn sample_moe_routing_from_source_tokens(
        &mut self,
        source_tokens: Vec<u64>,
        num_experts: u64,
        num_experts_per_token: u32,
        num_shared_experts_per_token: u32,
    ) -> MoeRouting {
        let mut remote_tokens = vec![vec![0_u64; self.ep as usize]; self.ep as usize];
        let mut local_tokens = vec![0_u64; self.ep as usize];
        let mut activated_experts = vec![HashSet::new(); self.ep as usize];

        for src_rank in 0..self.ep {
            for _ in 0..source_tokens[src_rank as usize] {
                let mut selected = Vec::with_capacity(num_experts_per_token as usize);
                let mut local_owners = HashSet::new();
                if num_shared_experts_per_token > 0 {
                    local_owners.insert(src_rank as usize);
                }
                while selected.len() < num_experts_per_token as usize {
                    let expert_id = self.random_below(num_experts);
                    if selected.contains(&expert_id) {
                        continue;
                    }
                    selected.push(expert_id);
                }
                for expert_id in selected {
                    let owner = Self::moe_expert_owner(self.ep, num_experts, expert_id);
                    activated_experts[owner].insert(expert_id);
                    local_owners.insert(owner);
                }
                for owner in local_owners {
                    local_tokens[owner] += 1;
                    if owner != src_rank as usize {
                        remote_tokens[src_rank as usize][owner] += 1;
                    }
                }
            }
        }

        let activated_experts = activated_experts
            .iter()
            .enumerate()
            .map(|(rank, experts)| {
                experts.len() as u64
                    + u64::from(source_tokens[rank] > 0) * u64::from(num_shared_experts_per_token)
            })
            .collect();

        MoeRouting {
            remote_tokens,
            local_tokens,
            activated_experts,
        }
    }

    fn add_barrier_nodes(&self, mut nodes: Vec<NodeID>, graph: &mut Graph) -> Result<NodeID> {
        if nodes.is_empty() {
            bail!("parallel barrier requires at least one dependency");
        }

        nodes.sort_unstable();
        nodes.dedup();
        if let [only] = nodes.as_slice() {
            return Ok(*only);
        }

        let module = self as *const Self as *mut Self as *mut dyn LogicalHandler;
        let barrier = graph.add_logical_node(module, LogicalPayload::Barrier);
        for node in nodes {
            graph.add_edge(node, barrier)?;
        }
        Ok(barrier)
    }

    fn add_compute_layer(
        &self,
        previous: NodeID,
        layer: &Layer,
        batch: &[RequestState],
        context: ComputeContext,
        graph: &mut Graph,
    ) -> Result<NodeID> {
        let node = graph.add_compute_node(
            self.compute_sim_ptr(),
            layer.clone(),
            batch.to_vec(),
            context,
        );
        graph.add_edge(previous, node)?;
        Ok(node)
    }

    fn add_parallel_barrier(
        &self,
        previous: NodeID,
        nodes: Vec<NodeID>,
        graph: &mut Graph,
    ) -> Result<NodeID> {
        if nodes.is_empty() {
            return Ok(previous);
        }
        let mut dependencies = vec![previous];
        dependencies.extend(nodes);
        self.add_barrier_nodes(dependencies, graph)
    }

    fn add_ep_collective(
        &self,
        previous: NodeID,
        remote_tokens: &[Vec<u64>],
        bytes_per_token: u64,
        combine: bool,
        graph: &mut Graph,
    ) -> Result<NodeID> {
        if self.ep <= 1 || bytes_per_token == 0 {
            return Ok(previous);
        }

        let mut nodes = Vec::new();
        let network_sim = self.network_sim_ptr();
        for src_rank in 0..self.ep {
            for dst_rank in 0..self.ep {
                if src_rank == dst_rank {
                    continue;
                }
                let tokens = if combine {
                    remote_tokens[dst_rank as usize][src_rank as usize]
                } else {
                    remote_tokens[src_rank as usize][dst_rank as usize]
                };
                if tokens == 0 {
                    continue;
                }
                let node = graph.add_network_node(
                    network_sim,
                    self.device_id(src_rank),
                    self.device_id(dst_rank),
                    tokens.saturating_mul(bytes_per_token),
                );
                graph.add_edge(previous, node)?;
                nodes.push(node);
            }
        }

        self.add_parallel_barrier(previous, nodes, graph)
    }

    fn add_standard_layer(
        &self,
        previous: NodeID,
        layer: &Layer,
        batch: &[RequestState],
        graph: &mut Graph,
    ) -> Result<NodeID> {
        match layer {
            Layer::AllReduce { .. } => {
                bail!("moe_ep does not support all-reduce layers because tp is fixed to 1")
            }
            Layer::Moe { .. } => bail!("moe layers must be handled by moe_ep"),
            _ => self.add_compute_layer(previous, layer, batch, ComputeContext::default(), graph),
        }
    }

    fn add_standard_layer_multi(
        &self,
        previous: &[NodeID],
        layer: &Layer,
        instance_batches: &[Vec<RequestState>],
        graph: &mut Graph,
    ) -> Result<Vec<NodeID>> {
        let mut next = previous.to_vec();
        for rank in 0..self.ep as usize {
            if instance_batches[rank].is_empty() {
                continue;
            }
            next[rank] =
                self.add_standard_layer(previous[rank], layer, &instance_batches[rank], graph)?;
        }
        Ok(next)
    }

    fn add_ep_collective_multi(
        &self,
        previous: &[NodeID],
        remote_tokens: &[Vec<u64>],
        bytes_per_token: u64,
        combine: bool,
        graph: &mut Graph,
    ) -> Result<NodeID> {
        if previous.len() != self.ep as usize {
            bail!(
                "moe_ep received {} predecessor nodes for ep={}",
                previous.len(),
                self.ep
            );
        }

        let mut dependencies = previous.to_vec();
        if self.ep > 1 && bytes_per_token > 0 {
            let network_sim = self.network_sim_ptr();
            for src_rank in 0..self.ep {
                for dst_rank in 0..self.ep {
                    if src_rank == dst_rank {
                        continue;
                    }
                    let tokens = if combine {
                        remote_tokens[dst_rank as usize][src_rank as usize]
                    } else {
                        remote_tokens[src_rank as usize][dst_rank as usize]
                    };
                    if tokens == 0 {
                        continue;
                    }
                    let node = graph.add_network_node(
                        network_sim,
                        self.device_id(src_rank),
                        self.device_id(dst_rank),
                        tokens.saturating_mul(bytes_per_token),
                    );
                    graph.add_edge(previous[src_rank as usize], node)?;
                    dependencies.push(node);
                }
            }
        }

        self.add_barrier_nodes(dependencies, graph)
    }

    fn add_moe_layer(
        &mut self,
        previous: NodeID,
        layer: &Layer,
        batch: &[RequestState],
        graph: &mut Graph,
    ) -> Result<NodeID> {
        let Layer::Moe {
            num_experts,
            num_experts_per_token,
            num_shared_experts_per_token,
            dispatch_bytes_per_token,
            combine_bytes_per_token,
            ..
        } = layer
        else {
            bail!("moe layer handler received a non-moe layer");
        };

        let routing = self.sample_moe_routing(
            Self::total_tokens(batch),
            *num_experts,
            *num_experts_per_token,
            *num_shared_experts_per_token,
        );

        let last_node = self.add_ep_collective(
            previous,
            &routing.remote_tokens,
            *dispatch_bytes_per_token,
            false,
            graph,
        )?;
        let mut compute_nodes = Vec::new();
        for rank in 0..self.ep as usize {
            if routing.local_tokens[rank] == 0 {
                continue;
            }
            let local_batch = Self::synthetic_batch(routing.local_tokens[rank])?;
            let compute = self.add_compute_layer(
                last_node,
                layer,
                &local_batch,
                ComputeContext {
                    moe_activated_experts: Some(routing.activated_experts[rank]),
                },
                graph,
            )?;
            compute_nodes.push(compute);
        }
        let last_node = self.add_parallel_barrier(last_node, compute_nodes, graph)?;
        self.add_ep_collective(
            last_node,
            &routing.remote_tokens,
            *combine_bytes_per_token,
            true,
            graph,
        )
    }

    fn add_moe_layer_multi(
        &mut self,
        previous: &[NodeID],
        layer: &Layer,
        instance_batches: &[Vec<RequestState>],
        graph: &mut Graph,
    ) -> Result<Vec<NodeID>> {
        let Layer::Moe {
            num_experts,
            num_experts_per_token,
            num_shared_experts_per_token,
            dispatch_bytes_per_token,
            combine_bytes_per_token,
            ..
        } = layer
        else {
            bail!("moe layer handler received a non-moe layer");
        };

        let routing = self.sample_moe_routing_from_source_tokens(
            self.source_tokens_from_instance_batches(instance_batches),
            *num_experts,
            *num_experts_per_token,
            *num_shared_experts_per_token,
        );

        let dispatch_done = self.add_ep_collective_multi(
            previous,
            &routing.remote_tokens,
            *dispatch_bytes_per_token,
            false,
            graph,
        )?;
        let mut compute_nodes = Vec::new();
        for rank in 0..self.ep as usize {
            if routing.local_tokens[rank] == 0 {
                continue;
            }
            let local_batch = Self::synthetic_batch(routing.local_tokens[rank])?;
            let compute = self.add_compute_layer(
                dispatch_done,
                layer,
                &local_batch,
                ComputeContext {
                    moe_activated_experts: Some(routing.activated_experts[rank]),
                },
                graph,
            )?;
            compute_nodes.push(compute);
        }
        let compute_done = self.add_parallel_barrier(dispatch_done, compute_nodes, graph)?;
        let combine_done = self.add_ep_collective_multi(
            &vec![compute_done; self.ep as usize],
            &routing.remote_tokens,
            *combine_bytes_per_token,
            true,
            graph,
        )?;
        Ok(vec![combine_done; self.ep as usize])
    }

    fn add_layer(
        &mut self,
        previous: NodeID,
        layer: &Layer,
        batch: &[RequestState],
        graph: &mut Graph,
    ) -> Result<NodeID> {
        match layer {
            Layer::Moe { .. } => self.add_moe_layer(previous, layer, batch, graph),
            _ => self.add_standard_layer(previous, layer, batch, graph),
        }
    }

    fn add_layer_multi(
        &mut self,
        previous: &[NodeID],
        layer: &Layer,
        instance_batches: &[Vec<RequestState>],
        graph: &mut Graph,
    ) -> Result<Vec<NodeID>> {
        match layer {
            Layer::Moe { .. } => self.add_moe_layer_multi(previous, layer, instance_batches, graph),
            _ => self.add_standard_layer_multi(previous, layer, instance_batches, graph),
        }
    }

    fn compute_sim_ptr(&self) -> *mut dyn ComputeSimulator {
        self.compute_sim.as_ref() as *const dyn ComputeSimulator as *mut dyn ComputeSimulator
    }

    fn network_sim_ptr(&self) -> *mut dyn NetworkSimulator {
        self.network_sim.as_ref() as *const dyn NetworkSimulator as *mut dyn NetworkSimulator
    }
}

impl Module for MoeExpertParallelRunner {
    fn resolve_path(&mut self, base_dir: &Path) -> Result<()> {
        self.compute_sim.resolve_path(base_dir)?;
        self.network_sim.resolve_path(base_dir)?;
        Ok(())
    }
}

impl LogicalHandler for MoeExpertParallelRunner {
    fn describe(&self, payload: &dyn Any) -> String {
        if let Some(value) = payload.downcast_ref::<LogicalPayload>() {
            format!("{}::{value:?}", std::any::type_name::<Self>())
        } else {
            std::any::type_name::<Self>().to_owned()
        }
    }

    fn handle(
        &mut self,
        payload: Box<dyn Any>,
        now: f64,
        _current: NodeID,
        _graph: &mut Graph,
    ) -> Result<f64> {
        match parse_payload(payload)? {
            LogicalPayload::BlockStart => {
                if self.block_start_time.replace(now).is_some() {
                    bail!("moe_ep block start time was already recorded");
                }
                Ok(0.0)
            }
            LogicalPayload::FastForward { remaining_blocks } => {
                let block_start_time = self
                    .block_start_time
                    .take()
                    .ok_or_else(|| anyhow!("moe_ep fast forward is missing a block start"))?;
                let block_latency = now - block_start_time;
                if !block_latency.is_finite() || block_latency < 0.0 {
                    bail!("moe_ep computed invalid block latency {block_latency}");
                }
                Ok(block_latency * f64::from(remaining_blocks))
            }
            LogicalPayload::Barrier => Ok(0.0),
        }
    }
}

impl ModelRunner for MoeExpertParallelRunner {
    fn parallel_strategy(&self) -> ParallelStrategy {
        ParallelStrategy {
            tp: 1,
            pp: 1,
            ep: self.ep,
        }
    }

    fn init(&mut self, model: &dyn Model) -> Result<()> {
        if self.ep == 0 {
            bail!("ep must be positive");
        }
        for layer in model
            .pre_block_layers()
            .iter()
            .chain(model.block_layers())
            .chain(model.post_block_layers())
        {
            if let Layer::Moe {
                num_experts,
                num_experts_per_token,
                ..
            } = layer
            {
                if u64::from(*num_experts_per_token) > *num_experts {
                    bail!("MoE top-k exceeds the number of experts");
                }
            }
        }

        self.rng = Some(StdRng::seed_from_u64(DEFAULT_SEED));
        self.compute_sim.init(self.parallel_strategy())?;

        self.net_gpu_device_ids.clear();
        for _ in 0..self.ep {
            self.net_gpu_device_ids.push(self.network_sim.add_device());
        }

        if self.ep > 1 {
            for src_rank in 0..self.ep {
                for dst_rank in 0..self.ep {
                    if src_rank == dst_rank {
                        continue;
                    }
                    self.network_sim.add_link(
                        self.device_id(src_rank),
                        self.device_id(dst_rank),
                        self.link_bw,
                        self.link_latency,
                    )?;
                }
            }
        }

        Ok(())
    }

    fn add_iteration(
        &mut self,
        model: &dyn Model,
        batch: &[RequestState],
        current: NodeID,
        graph: &mut Graph,
    ) -> Result<NodeID> {
        let module = self as *mut Self as *mut dyn LogicalHandler;

        let mut last_node = current;
        for layer in model.pre_block_layers() {
            last_node = self.add_layer(last_node, layer, batch, graph)?;
        }

        let block_start = graph.add_logical_node(module, LogicalPayload::BlockStart);
        graph.add_edge(last_node, block_start)?;

        last_node = block_start;
        for layer in model.block_layers() {
            last_node = self.add_layer(last_node, layer, batch, graph)?;
        }

        let fast_forward = graph.add_logical_node(
            module,
            LogicalPayload::FastForward {
                remaining_blocks: model.num_blocks().saturating_sub(1),
            },
        );
        graph.add_edge(last_node, fast_forward)?;

        last_node = fast_forward;
        for layer in model.post_block_layers() {
            last_node = self.add_layer(last_node, layer, batch, graph)?;
        }

        Ok(last_node)
    }

    fn add_multi_iteration(
        &mut self,
        model: &dyn Model,
        instance_batches: &[Vec<RequestState>],
        current: NodeID,
        graph: &mut Graph,
    ) -> Result<NodeID> {
        if instance_batches.len() != self.ep as usize {
            bail!(
                "moe_ep expected {} instance batches, got {}",
                self.ep,
                instance_batches.len()
            );
        }

        let module = self as *mut Self as *mut dyn LogicalHandler;

        let mut last_nodes = vec![current; self.ep as usize];
        for layer in model.pre_block_layers() {
            last_nodes = self.add_layer_multi(&last_nodes, layer, instance_batches, graph)?;
        }

        let pre_block_done = self.add_barrier_nodes(last_nodes, graph)?;
        let block_start = graph.add_logical_node(module, LogicalPayload::BlockStart);
        graph.add_edge(pre_block_done, block_start)?;

        last_nodes = vec![block_start; self.ep as usize];
        for layer in model.block_layers() {
            last_nodes = self.add_layer_multi(&last_nodes, layer, instance_batches, graph)?;
        }

        let block_done = self.add_barrier_nodes(last_nodes, graph)?;
        let fast_forward = graph.add_logical_node(
            module,
            LogicalPayload::FastForward {
                remaining_blocks: model.num_blocks().saturating_sub(1),
            },
        );
        graph.add_edge(block_done, fast_forward)?;

        last_nodes = vec![fast_forward; self.ep as usize];
        for layer in model.post_block_layers() {
            last_nodes = self.add_layer_multi(&last_nodes, layer, instance_batches, graph)?;
        }

        self.add_barrier_nodes(last_nodes, graph)
    }
}
