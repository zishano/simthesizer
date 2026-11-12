use std::any::Any;
use std::path::Path;

use anyhow::{anyhow, bail, Result};
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

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RingModelRunner {
    tp: u32,
    link_bw: f64,
    link_latency: f64,
    compute_sim: Box<dyn ComputeSimulator>,
    network_sim: Box<dyn NetworkSimulator>,

    #[serde(skip_deserializing, default)]
    net_gpu_device_ids: Vec<NetworkDeviceID>,

    #[serde(skip_deserializing, default)]
    block_start_time: Option<f64>,
}

crate::register_model_runner!(RingModelRunner, "ring");

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
        .map_err(|_| anyhow!("ring runner received an unexpected logical payload"))
}

impl RingModelRunner {
    fn all_reduce(&self, previous: NodeID, total_bytes: u64, graph: &mut Graph) -> Result<NodeID> {
        if self.tp <= 1 || total_bytes == 0 {
            return Ok(previous);
        }

        let module = self as *const Self as *mut Self as *mut dyn LogicalHandler;
        let network_sim = self.network_sim_ptr();
        let chunk_bytes = total_bytes.div_ceil(u64::from(self.tp));

        let mut last_node = previous;
        for _ in 0..(2 * (self.tp - 1)) {
            let mut nodes = Vec::with_capacity(self.tp as usize);
            for src in 0..self.tp {
                let node = graph.add_network_node(
                    network_sim,
                    self.net_gpu_device_ids[src as usize],
                    self.net_gpu_device_ids[(src + 1) as usize % self.tp as usize],
                    chunk_bytes,
                );
                graph.add_edge(last_node, node)?;
                nodes.push(node);
            }

            let barrier = graph.add_logical_node(module, LogicalPayload::Barrier);
            for node in nodes {
                graph.add_edge(node, barrier)?;
            }
            last_node = barrier;
        }

        Ok(last_node)
    }

    fn add_layer(
        &self,
        previous: NodeID,
        layer: &Layer,
        batch: &[RequestState],
        graph: &mut Graph,
    ) -> Result<NodeID> {
        match layer {
            Layer::AllReduce {
                bytes_per_token, ..
            } => self.all_reduce(
                previous,
                batch
                    .iter()
                    .map(|request| u64::from(request.q_len))
                    .sum::<u64>()
                    .saturating_mul(*bytes_per_token),
                graph,
            ),
            _ => {
                let node = graph.add_compute_node(
                    self.compute_sim_ptr(),
                    layer.clone(),
                    batch.to_vec(),
                    ComputeContext::default(),
                );
                graph.add_edge(previous, node)?;
                Ok(node)
            }
        }
    }

    fn compute_sim_ptr(&self) -> *mut dyn ComputeSimulator {
        self.compute_sim.as_ref() as *const dyn ComputeSimulator as *mut dyn ComputeSimulator
    }

    fn network_sim_ptr(&self) -> *mut dyn NetworkSimulator {
        self.network_sim.as_ref() as *const dyn NetworkSimulator as *mut dyn NetworkSimulator
    }
}

impl Module for RingModelRunner {
    fn resolve_path(&mut self, base_dir: &Path) -> Result<()> {
        self.compute_sim.resolve_path(base_dir)?;
        self.network_sim.resolve_path(base_dir)?;
        Ok(())
    }
}

impl LogicalHandler for RingModelRunner {
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
                    bail!("ring block start time was already recorded");
                }
                Ok(0.0)
            }
            LogicalPayload::FastForward { remaining_blocks } => {
                let block_start_time = self
                    .block_start_time
                    .take()
                    .ok_or_else(|| anyhow!("ring fast forward is missing a block start"))?;
                let block_latency = now - block_start_time;
                if !block_latency.is_finite() || block_latency < 0.0 {
                    bail!("ring computed invalid block latency {block_latency}");
                }
                Ok(block_latency * f64::from(remaining_blocks))
            }
            LogicalPayload::Barrier => Ok(0.0),
        }
    }
}

impl ModelRunner for RingModelRunner {
    fn parallel_strategy(&self) -> ParallelStrategy {
        ParallelStrategy {
            tp: self.tp,
            pp: 1,
            ep: 1,
        }
    }

    fn init(&mut self, model: &dyn Model) -> Result<()> {
        let _ = model;
        if self.tp == 0 {
            bail!("tp must be positive");
        }
        let parallel = self.parallel_strategy();
        self.compute_sim.init(parallel)?;

        self.net_gpu_device_ids.clear();
        for _ in 0..self.tp {
            self.net_gpu_device_ids.push(self.network_sim.add_device());
        }

        if self.tp > 1 {
            for rank in 0..self.net_gpu_device_ids.len() {
                let next_rank = (rank + 1) % self.net_gpu_device_ids.len();
                self.network_sim.add_link(
                    self.net_gpu_device_ids[rank],
                    self.net_gpu_device_ids[next_rank],
                    self.link_bw,
                    self.link_latency,
                )?;
                self.network_sim.add_link(
                    self.net_gpu_device_ids[next_rank],
                    self.net_gpu_device_ids[rank],
                    self.link_bw,
                    self.link_latency,
                )?;
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
}
