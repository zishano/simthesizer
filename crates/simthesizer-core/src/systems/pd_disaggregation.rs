use std::any::Any;
use std::cell::RefCell;
use std::collections::HashMap;
use std::path::Path;
use std::rc::Rc;

use anyhow::{anyhow, bail, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

use crate::engine::Graph;
use crate::model::Model;
use crate::module::{LogicalHandler, Module};
use crate::network_sim::NetworkSimulator;
use crate::policies::{
    InstanceId, PolicyConfig, PolicyRuntime, PolicyStatistics, RouteAction, RouteObservation,
    SharedPolicy, Stage,
};
use crate::system::{System, SystemEvent};
use crate::types::{
    NetworkDeviceID, NodeID, ParallelStrategy, Request, RequestResult, ResultSubRequest, SubRequest,
};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PdDisaggregationSystem {
    #[serde(default)]
    shared_execution: bool,
    #[serde(default)]
    allow_gpu_sharing: bool,
    #[serde(default = "layer_handoff")]
    kv_handoff: String,
    prefill_instances: Vec<Box<dyn System>>,
    decode_instances: Vec<Box<dyn System>>,
    model: Box<dyn Model>,
    tp: u32,
    network_sim: Box<dyn NetworkSimulator>,
    #[serde(default)]
    link_bw: Option<f64>,
    #[serde(default)]
    link_latency: Option<f64>,
    #[serde(default)]
    kv_topology: Option<KvTopology>,

    #[serde(default)]
    policy: Option<PolicyConfig>,
    #[serde(skip)]
    policy_runtime: Option<SharedPolicy>,
    #[serde(skip)]
    policy_graph_bound: bool,
    #[serde(skip)]
    incoming: HashMap<u64, IncomingRequest>,
    #[serde(skip)]
    decode_reserved: Vec<usize>,

    #[serde(skip_deserializing, default)]
    prefill_device_ids: Vec<NetworkDeviceID>,

    #[serde(skip_deserializing, default)]
    decode_device_ids: Vec<NetworkDeviceID>,

    #[serde(skip_deserializing, default)]
    next_prefill_instance: usize,

    #[serde(skip_deserializing, default)]
    next_decode_instance: usize,

    #[serde(skip_deserializing, default)]
    pending_requests: HashMap<u64, PendingRequest>,

    #[serde(skip_deserializing, default)]
    results: Vec<RequestResult>,
}

/// Device and link indices refer to insertion order in the parent network simulator.
#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct KvTopology {
    devices: Vec<String>,
    links: Vec<KvLink>,
    routes: Vec<KvRoute>,
    shards: u32,
    prefill_gpus: Vec<Vec<usize>>,
    decode_gpus: Vec<Vec<usize>>,
}
#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct KvLink {
    #[serde(default)]
    resource: Option<String>,
    src: usize,
    dst: usize,
    bandwidth: f64,
    latency: f64,
}

fn layer_handoff() -> String {
    "layer".into()
}
#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct KvRoute {
    src: usize,
    dst: usize,
    links: Vec<usize>,
}

impl KvTopology {
    fn validate(&self, p: usize, d: usize, allow_gpu_sharing: bool) -> Result<()> {
        if self.shards == 0 || self.prefill_gpus.len() != p || self.decode_gpus.len() != d {
            bail!("KV topology pool dimensions or shard count invalid");
        }
        let mut names = HashSet::new();
        if self
            .devices
            .iter()
            .any(|name| name.is_empty() || !names.insert(name))
        {
            bail!("KV topology device names must be unique and nonempty");
        }
        let mut allocated = HashSet::new();
        for group in self.prefill_gpus.iter().chain(&self.decode_gpus) {
            let mut local = HashSet::new();
            if group.is_empty() || self.shards as usize % group.len() != 0 {
                bail!("KV shards must divide evenly across each TP group");
            }
            for &gpu in group {
                if gpu >= self.devices.len()
                    || !local.insert(gpu)
                    || (!allocated.insert(gpu) && !allow_gpu_sharing)
                {
                    bail!("KV topology allocations must be disjoint valid device IDs");
                }
            }
        }
        for link in &self.links {
            if link.src >= self.devices.len()
                || link.dst >= self.devices.len()
                || link.src == link.dst
            {
                bail!("KV topology link endpoint invalid");
            }
        }
        let mut pairs = HashSet::new();
        for route in &self.routes {
            if route.src >= self.devices.len()
                || route.dst >= self.devices.len()
                || !pairs.insert((route.src, route.dst))
            {
                bail!("KV topology route endpoints invalid or duplicated");
            }
        }
        for &src in self.prefill_gpus.iter().flatten() {
            for &dst in self.decode_gpus.iter().flatten() {
                if !pairs.contains(&(src, dst)) {
                    bail!("missing KV topology route");
                }
            }
        }
        Ok(())
    }

    fn slices(&self, prefill: usize, decode: usize) -> Vec<(usize, usize, u32, u32)> {
        let source = &self.prefill_gpus[prefill];
        let target = &self.decode_gpus[decode];
        let source_width = self.shards / source.len() as u32;
        let target_width = self.shards / target.len() as u32;
        let mut slices = Vec::new();
        for (p, &src) in source.iter().enumerate() {
            for (d, &dst) in target.iter().enumerate() {
                let start = (p as u32 * source_width).max(d as u32 * target_width);
                let end = ((p as u32 + 1) * source_width).min((d as u32 + 1) * target_width);
                if start < end {
                    slices.push((src, dst, start, end));
                }
            }
        }
        slices
    }
}

crate::register_system!(PdDisaggregationSystem, "pd_disaggregation");

#[derive(Debug)]
struct PendingRequest {
    original: Request,
    prefill_request: Request,
    prefill_instance: usize,
    prefill_result: Option<(RequestResult, f64)>,
    decode_request: Option<Request>,
    decode_instance: Option<usize>,
    sink: Option<*mut dyn LogicalHandler>,
}

struct IncomingRequest {
    request: Request,
    sink: Option<*mut dyn LogicalHandler>,
}

struct LlmTokenData<'a> {
    input_tokens: u32,
    output_tokens: u32,
    input_token_ids: Option<&'a [u32]>,
    output_token_ids: Option<&'a [u32]>,
}

#[derive(Debug)]
enum LogicalPayload {
    RequestArrival {
        request_id: u64,
    },
    TransferComplete {
        request_id: u64,
    },
    KvSlice {
        request_id: u64,
        head_start: u32,
        head_end: u32,
    },
}

enum HandlerPayload {
    Internal(LogicalPayload),
    Event(SystemEvent),
}

fn parse_payload(payload: Box<dyn Any>) -> Result<HandlerPayload> {
    match payload.downcast::<LogicalPayload>() {
        Ok(payload) => Ok(HandlerPayload::Internal(*payload)),
        Err(payload) => match payload.downcast::<SystemEvent>() {
            Ok(payload) => Ok(HandlerPayload::Event(*payload)),
            Err(_) => Err(anyhow!(
                "pd_disaggregation received an unexpected logical payload"
            )),
        },
    }
}

impl PendingRequest {
    fn new(
        original: Request,
        prefill_request: Request,
        prefill_instance: usize,
        sink: Option<*mut dyn LogicalHandler>,
    ) -> Self {
        Self {
            original,
            prefill_request,
            prefill_instance,
            prefill_result: None,
            decode_request: None,
            decode_instance: None,
            sink,
        }
    }
}

impl PdDisaggregationSystem {
    fn single_llm_sub_request(request: &Request) -> Result<LlmTokenData<'_>> {
        let [sub_request] = request.sub_requests.as_slice() else {
            bail!("pd_disaggregation only supports requests with a single llm sub-request");
        };
        let SubRequest::Llm {
            input_tokens,
            output_tokens,
            input_token_ids,
            output_token_ids,
            ..
        } = sub_request
        else {
            bail!("pd_disaggregation only supports llm sub-requests");
        };
        Ok(LlmTokenData {
            input_tokens: *input_tokens,
            output_tokens: *output_tokens,
            input_token_ids: input_token_ids.as_deref(),
            output_token_ids: output_token_ids.as_deref(),
        })
    }

    fn single_llm_tokens(request: &Request) -> Result<(u32, u32)> {
        let tokens = Self::single_llm_sub_request(request)?;
        Ok((tokens.input_tokens, tokens.output_tokens))
    }

    fn build_llm_request(
        original: &Request,
        input_tokens: u32,
        output_tokens: u32,
        kv_tokens: u32,
        input_token_ids: Option<Vec<u32>>,
        output_token_ids: Option<Vec<u32>>,
    ) -> Request {
        Request {
            request_id: original.request_id,
            arrival_time: original.arrival_time,
            initial: vec![0],
            sub_requests: vec![SubRequest::Llm {
                input_tokens,
                output_tokens,
                known_tokens: Some(input_tokens),
                kv_tokens: Some(kv_tokens),
                next: Vec::new(),
                interval: None,
                model: None,
                input_token_ids,
                output_token_ids,
            }],
        }
    }

    fn build_prefill_request(request: &Request) -> Result<Request> {
        let tokens = Self::single_llm_sub_request(request)?;
        let mut prefill = Self::build_llm_request(
            request,
            tokens.input_tokens,
            tokens.output_tokens.min(1),
            0,
            tokens.input_token_ids.map(<[u32]>::to_vec),
            tokens.output_token_ids.map(<[u32]>::to_vec),
        );
        if let SubRequest::Llm {
            interval, model, ..
        } = &mut prefill.sub_requests[0]
        {
            *interval = request.sub_requests[0].interval();
            *model = request.sub_requests[0].model().map(str::to_owned);
        }
        Ok(prefill)
    }

    fn build_decode_request(
        original: &Request,
        prefill_request: &Request,
    ) -> Result<Option<Request>> {
        let tokens = Self::single_llm_sub_request(original)?;
        let carried_tokens = prefill_request
            .total_input_tokens()
            .saturating_add(prefill_request.total_output_tokens());
        let remaining_output = tokens
            .output_tokens
            .saturating_sub(prefill_request.total_output_tokens());
        if remaining_output == 0 {
            return Ok(None);
        }

        let carried_output = prefill_request.total_output_tokens() as usize;
        let input_token_ids = tokens.input_token_ids.map(|input_token_ids| {
            let mut combined = input_token_ids.to_vec();
            if let Some(output_token_ids) = tokens.output_token_ids {
                combined.extend(output_token_ids.iter().copied().take(carried_output));
            }
            combined
        });
        let output_token_ids = tokens.output_token_ids.map(|output_token_ids| {
            output_token_ids
                .iter()
                .copied()
                .skip(carried_output)
                .collect()
        });

        Ok(Some(Self::build_llm_request(
            original,
            carried_tokens,
            remaining_output,
            prefill_request.total_input_tokens(),
            input_token_ids,
            output_token_ids,
        )))
    }

    fn choose_prefill_instance(&mut self) -> Result<usize> {
        if self.prefill_instances.is_empty() {
            bail!("pd_disaggregation requires at least one prefill instance");
        }
        let index = self.next_prefill_instance % self.prefill_instances.len();
        self.next_prefill_instance = (index + 1) % self.prefill_instances.len();
        Ok(index)
    }

    fn choose_decode_instance(&mut self) -> Result<usize> {
        if self.decode_instances.is_empty() {
            bail!("pd_disaggregation requires at least one decode instance");
        }
        let index = self.next_decode_instance % self.decode_instances.len();
        self.next_decode_instance = (index + 1) % self.decode_instances.len();
        Ok(index)
    }

    fn merge_result(
        original: &Request,
        (prefill_result, prefill_finished_at): (RequestResult, f64),
        decode_result: Option<(RequestResult, f64)>,
    ) -> Result<RequestResult> {
        let (input_tokens, output_tokens) = Self::single_llm_tokens(original)?;
        let ResultSubRequest::Llm {
            token_latency: mut combined_token_latency,
            ..
        } = prefill_result
            .sub_requests
            .into_iter()
            .next()
            .ok_or_else(|| anyhow!("pd_disaggregation is missing prefill result sub-request"))?
        else {
            bail!("pd_disaggregation prefill result must contain one llm sub-request");
        };

        if let Some((decode_result, decode_finished_at)) = decode_result {
            let ResultSubRequest::Llm {
                token_latency: mut decode_token_latency,
                ..
            } = decode_result
                .sub_requests
                .into_iter()
                .next()
                .ok_or_else(|| anyhow!("pd_disaggregation is missing decode result sub-request"))?
            else {
                bail!("pd_disaggregation decode result must contain one llm sub-request");
            };
            // Recover the first decode token's time from completion and later ITLs.
            let later_latency: f64 = decode_token_latency.iter().skip(1).sum();
            if let Some(first) = decode_token_latency.first_mut() {
                *first = (decode_finished_at - prefill_finished_at - later_latency).max(0.0);
            }
            combined_token_latency.extend(decode_token_latency);
        }
        if combined_token_latency.len() != output_tokens as usize {
            bail!("pd_disaggregation completed with an incorrect output token count");
        }

        Ok(RequestResult {
            request_id: original.request_id,
            arrival_time: original.arrival_time,
            initial: vec![0],
            sub_requests: vec![ResultSubRequest::Llm {
                input_tokens,
                output_tokens,
                next: Vec::new(),
                token_latency: combined_token_latency,
                interval: original.sub_requests[0].interval(),
                model: original.sub_requests[0].model().map(str::to_owned),
            }],
        })
    }

    fn emit_final_result(
        &mut self,
        pending: PendingRequest,
        decode_result: Option<(RequestResult, f64)>,
        current: NodeID,
        graph: &mut Graph,
    ) -> Result<()> {
        let prefill_result = pending.prefill_result.ok_or_else(|| {
            anyhow!(
                "pd_disaggregation is missing prefill result for request {}",
                pending.original.request_id
            )
        })?;
        let result = Self::merge_result(&pending.original, prefill_result, decode_result)?;
        if let Some(sink) = pending.sink {
            let event = graph.add_logical_node(
                sink,
                SystemEvent::RequestCompleted {
                    request: pending.original,
                    result,
                },
            );
            graph.add_edge(current, event)?;
        } else {
            self.results.push(result);
        }
        Ok(())
    }

    fn handle_child_completion(
        &mut self,
        request: Request,
        result: RequestResult,
        now: f64,
        current: NodeID,
        graph: &mut Graph,
    ) -> Result<f64> {
        let request_id = request.request_id;
        let prefill_request = self
            .pending_requests
            .get(&request_id)
            .map(|pending| pending.prefill_request.clone())
            .ok_or_else(|| anyhow!("pd_disaggregation is missing request {request_id}"))?;

        if request == prefill_request {
            let decode_request = {
                let pending = self
                    .pending_requests
                    .get(&request_id)
                    .ok_or_else(|| anyhow!("pd_disaggregation is missing request {request_id}"))?;
                match &pending.decode_request {
                    Some(request) => Some(request.clone()),
                    None => {
                        Self::build_decode_request(&pending.original, &pending.prefill_request)?
                    }
                }
            };

            if self.shared_execution {
                graph.kv_event(crate::kv::Event::PrefillDone(request_id), current)?;
                if let Some(decode_request) = decode_request {
                    let pending = self.pending_requests.get_mut(&request_id).unwrap();
                    pending.prefill_result = Some((result, now));
                    pending.decode_request = Some(decode_request.clone());
                    let index = pending.decode_instance.unwrap();
                    if self.policy_runtime.is_some() {
                        self.decode_reserved[index] -= 1;
                    }
                    let sink = self as *mut Self as *mut dyn LogicalHandler;
                    self.decode_instances[index].add_request_arrival(
                        decode_request,
                        graph,
                        Some(current),
                        Some(sink),
                    )?;
                    return Ok(0.0);
                }
            }

            if let Some(decode_request) = decode_request {
                let decode_instance = self.choose_decode_instance()?;
                let prefill_instance = {
                    let pending = self.pending_requests.get(&request_id).ok_or_else(|| {
                        anyhow!("pd_disaggregation is missing request {request_id}")
                    })?;
                    pending.prefill_instance
                };

                let pending = self
                    .pending_requests
                    .get_mut(&request_id)
                    .ok_or_else(|| anyhow!("pd_disaggregation is missing request {request_id}"))?;
                pending.prefill_result = Some((result, now));
                pending.decode_request = Some(decode_request.clone());
                pending.decode_instance = Some(decode_instance);

                let transfer_tokens = prefill_request.total_input_tokens();
                let transfer_complete = graph.add_logical_node(
                    self as *mut Self as *mut dyn LogicalHandler,
                    LogicalPayload::TransferComplete { request_id },
                );
                if let Some(topology) = &self.kv_topology {
                    let slices = topology.slices(prefill_instance, decode_instance);
                    let bytes_per_head =
                        self.model.kv_bytes_per_token_per_device() / u64::from(topology.shards);
                    for (src, dst, head_start, head_end) in slices {
                        let bytes = u64::from(transfer_tokens)
                            .checked_mul(bytes_per_head)
                            .and_then(|v| v.checked_mul(u64::from(head_end - head_start)))
                            .ok_or_else(|| anyhow!("KV transfer size overflow"))?;
                        let marker = graph.add_logical_node(
                            self as *mut Self as *mut dyn LogicalHandler,
                            LogicalPayload::KvSlice {
                                request_id,
                                head_start,
                                head_end,
                            },
                        );
                        graph.add_edge(current, marker)?;
                        let transfer = graph.add_network_node(
                            self.network_sim_ptr(),
                            src as u64,
                            dst as u64,
                            bytes,
                        );
                        graph.add_edge(marker, transfer)?;
                        graph.add_edge(transfer, transfer_complete)?;
                    }
                } else {
                    let transfer_bytes = u64::from(transfer_tokens)
                        .saturating_mul(self.model.kv_bytes_per_token_per_device());
                    let transfer = graph.add_network_node(
                        self.network_sim_ptr(),
                        self.prefill_device_ids[prefill_instance],
                        self.decode_device_ids[decode_instance],
                        transfer_bytes,
                    );
                    graph.add_edge(current, transfer)?;
                    graph.add_edge(transfer, transfer_complete)?;
                }
                return Ok(0.0);
            }

            let pending = self
                .pending_requests
                .remove(&request_id)
                .ok_or_else(|| anyhow!("pd_disaggregation is missing request {request_id}"))?;
            let mut pending = pending;
            pending.prefill_result = Some((result, now));
            self.emit_final_result(pending, None, current, graph)?;
            return Ok(0.0);
        }

        let expected_decode_request = self
            .pending_requests
            .get(&request_id)
            .and_then(|pending| pending.decode_request.clone())
            .ok_or_else(|| {
                anyhow!("pd_disaggregation received unexpected completion for request {request_id}")
            })?;
        if request != expected_decode_request {
            bail!(
                "pd_disaggregation received mismatched completion payload for request {request_id}"
            );
        }

        if self.shared_execution {
            graph.kv_event(crate::kv::Event::DecodeDone(request_id), current)?;
        }

        let pending = self
            .pending_requests
            .remove(&request_id)
            .ok_or_else(|| anyhow!("pd_disaggregation is missing request {request_id}"))?;
        self.emit_final_result(pending, Some((result, now)), current, graph)?;
        Ok(0.0)
    }

    fn handle_transfer_complete(
        &mut self,
        request_id: u64,
        current: NodeID,
        graph: &mut Graph,
    ) -> Result<f64> {
        let (decode_request, decode_instance) = {
            let pending = self
                .pending_requests
                .get(&request_id)
                .ok_or_else(|| anyhow!("pd_disaggregation is missing request {request_id}"))?;
            (
                pending.decode_request.clone().ok_or_else(|| {
                    anyhow!("pd_disaggregation is missing decode request for request {request_id}")
                })?,
                pending.decode_instance.ok_or_else(|| {
                    anyhow!("pd_disaggregation is missing decode instance for request {request_id}")
                })?,
            )
        };

        let sink = self as *mut Self as *mut dyn LogicalHandler;
        self.decode_instances[decode_instance].add_request_arrival(
            decode_request,
            graph,
            Some(current),
            Some(sink),
        )?;
        Ok(0.0)
    }

    fn network_sim_ptr(&self) -> *mut dyn NetworkSimulator {
        self.network_sim.as_ref() as *const dyn NetworkSimulator as *mut dyn NetworkSimulator
    }
}

impl LogicalHandler for PdDisaggregationSystem {
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
        current: NodeID,
        graph: &mut Graph,
    ) -> Result<f64> {
        match parse_payload(payload)? {
            HandlerPayload::Internal(LogicalPayload::RequestArrival { request_id }) => {
                let incoming = self
                    .incoming
                    .remove(&request_id)
                    .ok_or_else(|| anyhow!("missing policy arrival {request_id}"))?;
                let policy = self
                    .policy_runtime
                    .as_ref()
                    .ok_or_else(|| anyhow!("policy arrival without policy"))?
                    .clone();
                let route = if policy.borrow().is_reference() {
                    policy.borrow_mut().record_reference_route();
                    None
                } else {
                    let prefill = self
                        .prefill_instances
                        .iter()
                        .map(|s| s.instance_load())
                        .collect::<Vec<_>>();
                    let decode = self
                        .decode_instances
                        .iter()
                        .enumerate()
                        .map(|(i, s)| {
                            let mut load = s.instance_load();
                            load.waiting_requests += self.decode_reserved[i];
                            load.waiting_tokens += self.decode_reserved[i] as u64;
                            load
                        })
                        .collect::<Vec<_>>();
                    let (input_tokens, _) = Self::single_llm_tokens(&incoming.request)?;
                    Some(policy.borrow_mut().route(&RouteObservation {
                        now,
                        request_id,
                        input_tokens,
                        prefill: &prefill,
                        decode: &decode,
                    })?)
                };
                self.enroll_request(incoming.request, graph, Some(current), incoming.sink, route)?;
                Ok(0.0)
            }
            HandlerPayload::Internal(LogicalPayload::KvSlice {
                request_id,
                head_start,
                head_end,
            }) => {
                let _ = (request_id, head_start, head_end); // Metadata-only event; the network node owns cost.
                Ok(0.0)
            }
            HandlerPayload::Internal(LogicalPayload::TransferComplete { request_id }) => {
                self.handle_transfer_complete(request_id, current, graph)
            }
            HandlerPayload::Event(SystemEvent::RequestCompleted { request, result }) => {
                self.handle_child_completion(request, result, now, current, graph)
            }
        }
    }
}

impl Module for PdDisaggregationSystem {
    fn observe(&self) -> serde_json::Value {
        serde_json::json!({
            "prefill_instances": self.prefill_instances.iter().map(|s| s.observe()).collect::<Vec<_>>(),
            "decode_instances": self.decode_instances.iter().map(|s| s.observe()).collect::<Vec<_>>(),
        })
    }

    fn resolve_path(&mut self, base_dir: &Path) -> Result<()> {
        for instance in &mut self.prefill_instances {
            instance.resolve_path(base_dir)?;
        }
        for instance in &mut self.decode_instances {
            instance.resolve_path(base_dir)?;
        }
        self.model.resolve_path(base_dir)?;
        self.network_sim.resolve_path(base_dir)?;
        Ok(())
    }
}

impl System for PdDisaggregationSystem {
    fn policy_statistics(&self) -> Option<PolicyStatistics> {
        self.policy_runtime
            .as_ref()
            .map(|p| p.borrow().statistics())
    }

    fn prepare_graph(&mut self, graph: &mut Graph) -> Result<()> {
        if !self.policy_graph_bound {
            if let Some(policy) = &self.policy_runtime {
                graph.kv.configure_policy(policy.clone())?;
                self.policy_graph_bound = true;
            }
        }
        for instance in self
            .prefill_instances
            .iter_mut()
            .chain(&mut self.decode_instances)
        {
            instance.prepare_graph(graph)?;
        }
        Ok(())
    }
    fn completed_requests(&self) -> Option<usize> {
        Some(self.results.len())
    }

    fn instance_load(&self) -> crate::types::InstanceLoad {
        self.prefill_instances
            .iter()
            .chain(&self.decode_instances)
            .map(|system| system.instance_load())
            .sum()
    }

    fn init(&mut self) -> Result<()> {
        anyhow::ensure!(
            self.policy.is_none() || self.shared_execution,
            "serving policy requires shared P/D execution"
        );
        anyhow::ensure!(
            !self.allow_gpu_sharing || self.shared_execution,
            "GPU sharing requires shared execution"
        );
        anyhow::ensure!(
            matches!(self.kv_handoff.as_str(), "layer" | "whole"),
            "kv_handoff must be layer or whole"
        );
        for instance in self.prefill_instances.iter().chain(&self.decode_instances) {
            anyhow::ensure!(
                instance.shared_execution() == self.shared_execution,
                "P/D parent and all runners must use the same execution mode"
            );
        }
        if self.shared_execution {
            let topology = self
                .kv_topology
                .as_ref()
                .ok_or_else(|| anyhow!("shared Rooflang requires a physical KV topology"))?;
            anyhow::ensure!(
                topology
                    .links
                    .iter()
                    .all(|l| l.resource.as_ref().is_some_and(|s| !s.is_empty())),
                "shared KV links require physical resource IDs"
            );
        }
        if self.prefill_instances.is_empty() {
            bail!("pd_disaggregation requires at least one prefill instance");
        }
        if self.decode_instances.is_empty() {
            bail!("pd_disaggregation requires at least one decode instance");
        }

        self.next_prefill_instance = 0;
        self.next_decode_instance = 0;
        self.pending_requests.clear();
        self.results.clear();
        self.prefill_device_ids.clear();
        self.decode_device_ids.clear();
        self.incoming.clear();
        self.decode_reserved = vec![0; self.decode_instances.len()];
        self.policy_graph_bound = false;
        if let Some(config) = &self.policy {
            let policy = Rc::new(RefCell::new(PolicyRuntime::new(config.clone())?));
            for (stage, instances) in [
                (Stage::Prefill, &mut self.prefill_instances),
                (Stage::Decode, &mut self.decode_instances),
            ] {
                for (index, instance) in instances.iter_mut().enumerate() {
                    instance.configure_policy(policy.clone(), InstanceId { stage, index })?;
                }
            }
            self.policy_runtime = Some(policy);
        }
        self.model.init(ParallelStrategy {
            tp: self.tp,
            pp: 1,
            ep: 1,
        })?;

        if let Some(topology) = &self.kv_topology {
            if self.tp != 1 || self.link_bw.is_some() || self.link_latency.is_some() {
                bail!("KV topology requires parent TP=1 and replaces manual KV links");
            }
            topology.validate(
                self.prefill_instances.len(),
                self.decode_instances.len(),
                self.allow_gpu_sharing,
            )?;
            if self.model.kv_bytes_per_token_per_device() % u64::from(topology.shards) != 0 {
                bail!("KV bytes must divide evenly into shards");
            }
            for index in 0..topology.devices.len() {
                if self.network_sim.add_device() != index as u64 {
                    bail!("KV network device index mismatch");
                }
            }
            for link in &topology.links {
                self.network_sim.add_link(
                    link.src as u64,
                    link.dst as u64,
                    link.bandwidth,
                    link.latency,
                )?;
            }
            for route in &topology.routes {
                self.network_sim
                    .set_route(route.src as u64, route.dst as u64, &route.links)?;
            }
        } else {
            let bw = self.link_bw.ok_or_else(|| anyhow!("missing link_bw"))?;
            let latency = self
                .link_latency
                .ok_or_else(|| anyhow!("missing link_latency"))?;
            for _ in &self.prefill_instances {
                self.prefill_device_ids.push(self.network_sim.add_device());
            }
            for _ in &self.decode_instances {
                self.decode_device_ids.push(self.network_sim.add_device());
            }
            for &src in &self.prefill_device_ids {
                for &dst in &self.decode_device_ids {
                    self.network_sim.add_link(src, dst, bw, latency)?;
                }
            }
        }
        for instance in self
            .prefill_instances
            .iter_mut()
            .chain(&mut self.decode_instances)
        {
            instance.init()?;
        }

        Ok(())
    }

    fn add_request_arrival(
        &mut self,
        request: Request,
        graph: &mut Graph,
        parent: Option<NodeID>,
        sink: Option<*mut dyn LogicalHandler>,
    ) -> Result<()> {
        request.validate()?;
        if request
            .sub_requests
            .iter()
            .any(|sub| sub.known_tokens().is_some() || sub.kv_tokens().is_some())
        {
            bail!("pd_disaggregation does not support external seeded progress");
        }
        if self.pending_requests.contains_key(&request.request_id)
            || self.incoming.contains_key(&request.request_id)
        {
            bail!(
                "pd_disaggregation received duplicate request id {}",
                request.request_id
            );
        }

        if self.policy_runtime.is_some() {
            Self::single_llm_tokens(&request)?;
            let request_id = request.request_id;
            let arrival_time = request.arrival_time;
            self.incoming
                .insert(request_id, IncomingRequest { request, sink });
            let arrival = graph.add_logical_node_at(
                self as *mut Self as *mut dyn LogicalHandler,
                LogicalPayload::RequestArrival { request_id },
                arrival_time,
            );
            if let Some(parent) = parent {
                graph.add_edge(parent, arrival)?;
            }
            return Ok(());
        }

        self.enroll_request(request, graph, parent, sink, None)
    }

    fn into_results(self: Box<Self>) -> Result<Vec<RequestResult>> {
        anyhow::ensure!(
            self.incoming.is_empty(),
            "pd_disaggregation finished before all policy arrivals"
        );
        if !self.pending_requests.is_empty() {
            bail!(
                "pd_disaggregation finished with {} incomplete requests",
                self.pending_requests.len()
            );
        }
        Ok(self.results)
    }
}

impl PdDisaggregationSystem {
    fn enroll_request(
        &mut self,
        request: Request,
        graph: &mut Graph,
        parent: Option<NodeID>,
        sink: Option<*mut dyn LogicalHandler>,
        route: Option<RouteAction>,
    ) -> Result<()> {
        let prefill_instance = match route {
            Some(route) => route.prefill,
            None => self.choose_prefill_instance()?,
        };
        let request_id = request.request_id;
        let prefill_request = Self::build_prefill_request(&request)?;
        let mut pending =
            PendingRequest::new(request, prefill_request.clone(), prefill_instance, sink);
        if self.shared_execution {
            self.prepare_graph(graph)?;
            let (input, output) = Self::single_llm_tokens(&pending.original)?;
            anyhow::ensure!(input > 0, "shared KV requires nonempty input");
            let decode = if output > 1 {
                Some(match route {
                    Some(route) => route.decode,
                    None => self.choose_decode_instance()?,
                })
            } else {
                None
            };
            pending.decode_instance = decode;
            if self.policy_runtime.is_some() {
                if let Some(index) = decode {
                    self.decode_reserved[index] += 1;
                }
            }
            let topology = self.kv_topology.as_ref().unwrap();
            let mut routes = Vec::new();
            if let Some(decode) = decode {
                for (src, dst, head_start, head_end) in topology.slices(prefill_instance, decode) {
                    let path = topology
                        .routes
                        .iter()
                        .find(|r| r.src == src && r.dst == dst)
                        .unwrap();
                    let mut seconds_per_byte = std::collections::BTreeMap::new();
                    let mut latency = 0.0;
                    for &i in &path.links {
                        let link = &topology.links[i];
                        seconds_per_byte
                            .insert(link.resource.clone().unwrap(), 1.0 / link.bandwidth);
                        latency += link.latency;
                    }
                    routes.push(crate::kv::Route {
                        src: topology.devices[src].clone(),
                        dst: topology.devices[dst].clone(),
                        head_start,
                        head_end,
                        seconds_per_byte,
                        latency,
                    });
                }
            }
            let layers = self.model.num_blocks();
            let full = self.model.kv_bytes_per_token_per_device();
            anyhow::ensure!(
                full % u64::from(layers * topology.shards) == 0,
                "nonintegral per-layer/head KV size"
            );
            graph.kv.plans.insert(
                request_id,
                crate::kv::Plan {
                    source: topology.prefill_gpus[prefill_instance]
                        .iter()
                        .map(|&i| topology.devices[i].clone())
                        .collect(),
                    target: decode
                        .map(|d| {
                            topology.decode_gpus[d]
                                .iter()
                                .map(|&i| topology.devices[i].clone())
                                .collect()
                        })
                        .unwrap_or_default(),
                    routes,
                    layers,
                    heads: topology.shards,
                    bytes_per_token_head_layer: full / u64::from(layers * topology.shards),
                    input_tokens: input,
                    max_decode_tokens: if output > 1 {
                        input
                            .checked_add(output - 1)
                            .ok_or_else(|| anyhow!("KV token count overflow"))?
                    } else {
                        0
                    },
                    layer_handoff: self.kv_handoff == "layer",
                    prefill_done: false,
                    copies: Default::default(),
                },
            );
        }
        self.pending_requests.insert(request_id, pending);

        let sink = self as *mut Self as *mut dyn LogicalHandler;
        if let Err(err) = self.prefill_instances[prefill_instance].add_request_arrival(
            prefill_request,
            graph,
            parent,
            Some(sink),
        ) {
            self.pending_requests.remove(&request_id);
            return Err(err);
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{InstanceLoad, Layer};

    struct RouteModel;
    impl Module for RouteModel {}
    impl Model for RouteModel {
        fn init(&mut self, _: ParallelStrategy) -> Result<()> {
            Ok(())
        }
        fn weight_bytes_per_device(&self) -> u64 {
            0
        }
        fn kv_bytes_per_token_per_device(&self) -> u64 {
            2
        }
        fn pre_block_layers(&self) -> &[Layer] {
            &[]
        }
        fn block_layers(&self) -> &[Layer] {
            &[]
        }
        fn post_block_layers(&self) -> &[Layer] {
            &[]
        }
        fn num_blocks(&self) -> u32 {
            1
        }
    }

    type Assignments = Rc<RefCell<Vec<(u64, usize, f64)>>>;
    struct RouteChild {
        index: usize,
        assignments: Assignments,
        queued: HashMap<u64, u32>,
    }
    struct Finish {
        request: Request,
        sink: *mut dyn LogicalHandler,
    }
    impl Module for RouteChild {}
    impl LogicalHandler for RouteChild {
        fn handle(
            &mut self,
            payload: Box<dyn Any>,
            now: f64,
            current: NodeID,
            graph: &mut Graph,
        ) -> Result<f64> {
            let Finish { request, sink } = *payload.downcast::<Finish>().unwrap();
            let tokens = self.queued.remove(&request.request_id).unwrap();
            graph.kv_event(
                crate::kv::Event::Produce {
                    request: request.request_id,
                    prefill: true,
                    layer: 0,
                    rank: 0,
                    start: 0,
                    end: tokens,
                },
                current,
            )?;
            let result = RequestResult {
                request_id: request.request_id,
                arrival_time: request.arrival_time,
                initial: vec![0],
                sub_requests: vec![ResultSubRequest::Llm {
                    input_tokens: tokens,
                    output_tokens: 1,
                    token_latency: vec![now - request.arrival_time],
                    next: vec![],
                    interval: None,
                    model: None,
                }],
            };
            let done =
                graph.add_logical_node(sink, SystemEvent::RequestCompleted { request, result });
            graph.add_edge(current, done)?;
            Ok(0.0)
        }
    }
    impl System for RouteChild {
        fn configure_policy(&mut self, _: SharedPolicy, _: InstanceId) -> Result<()> {
            Ok(())
        }
        fn shared_execution(&self) -> bool {
            true
        }
        fn init(&mut self) -> Result<()> {
            Ok(())
        }
        fn instance_load(&self) -> InstanceLoad {
            InstanceLoad {
                waiting_requests: self.queued.len(),
                waiting_tokens: self.queued.values().map(|&v| u64::from(v)).sum(),
                ..Default::default()
            }
        }
        fn add_request_arrival(
            &mut self,
            request: Request,
            graph: &mut Graph,
            parent: Option<NodeID>,
            sink: Option<*mut dyn LogicalHandler>,
        ) -> Result<()> {
            self.assignments
                .borrow_mut()
                .push((request.request_id, self.index, graph.now()));
            let tokens = request.total_input_tokens();
            self.queued.insert(request.request_id, tokens);
            let finish = graph.add_logical_node_at(
                self as *mut dyn LogicalHandler,
                Finish {
                    request,
                    sink: sink.unwrap(),
                },
                graph.now() + f64::from(tokens),
            );
            if let Some(parent) = parent {
                graph.add_edge(parent, finish)?;
            }
            Ok(())
        }
        fn into_results(self: Box<Self>) -> Result<Vec<RequestResult>> {
            anyhow::ensure!(self.queued.is_empty(), "route fixture has unfinished work");
            Ok(vec![])
        }
    }

    fn routed_run(future_arrival: f64) -> Vec<(u64, usize, f64)> {
        let assignments = Assignments::default();
        let children = || {
            (0..2)
                .map(|index| {
                    Box::new(RouteChild {
                        index,
                        assignments: assignments.clone(),
                        queued: HashMap::new(),
                    }) as Box<dyn System>
                })
                .collect()
        };
        let mut system = Box::new(PdDisaggregationSystem {
            shared_execution: true,
            allow_gpu_sharing: false,
            kv_handoff: "whole".into(),
            prefill_instances: children(),
            decode_instances: children(),
            model: Box::new(RouteModel),
            tp: 1,
            network_sim: serde_json::from_value(serde_json::json!({"kind":"flow"})).unwrap(),
            link_bw: None,
            link_latency: None,
            kv_topology: Some(KvTopology {
                devices: vec!["p0".into(), "p1".into(), "d0".into(), "d1".into()],
                shards: 1,
                prefill_gpus: vec![vec![0], vec![1]],
                decode_gpus: vec![vec![2], vec![3]],
                links: (0..4)
                    .map(|i| KvLink {
                        resource: Some(format!("link{i}")),
                        src: i / 2,
                        dst: i % 2 + 2,
                        bandwidth: 100.0,
                        latency: 0.0,
                    })
                    .collect(),
                routes: (0..4)
                    .map(|i| KvRoute {
                        src: i / 2,
                        dst: i % 2 + 2,
                        links: vec![i],
                    })
                    .collect(),
            }),
            policy: Some(crate::policies::test_config()),
            policy_runtime: None,
            policy_graph_bound: false,
            incoming: HashMap::new(),
            decode_reserved: Vec::new(),
            prefill_device_ids: Vec::new(),
            decode_device_ids: Vec::new(),
            next_prefill_instance: 0,
            next_decode_instance: 0,
            pending_requests: HashMap::new(),
            results: Vec::new(),
        });
        system.init().unwrap();
        system
            .policy_runtime
            .as_ref()
            .unwrap()
            .borrow_mut()
            .set_route_for_test(|observation, _, _| {
                let least = |loads: &[InstanceLoad]| {
                    loads
                        .iter()
                        .enumerate()
                        .min_by_key(|(i, load)| (load.total_tokens(), *i))
                        .unwrap()
                        .0
                };
                RouteAction {
                    prefill: least(observation.prefill),
                    decode: least(observation.decode),
                }
            });
        let mut graph = Graph::new();
        for (id, arrival_time) in [1.0, 1.5, future_arrival].into_iter().enumerate() {
            let request = Request {
                request_id: id as u64,
                arrival_time,
                initial: vec![0],
                sub_requests: vec![SubRequest::Llm {
                    input_tokens: 4,
                    output_tokens: 1,
                    known_tokens: None,
                    kv_tokens: None,
                    next: vec![],
                    interval: None,
                    model: None,
                    input_token_ids: None,
                    output_token_ids: None,
                }],
            };
            system
                .add_request_arrival(request, &mut graph, None, None)
                .unwrap();
        }
        assert!(
            assignments.borrow().is_empty(),
            "routing must not inspect load while seeding future requests"
        );
        assert_eq!(system.policy_statistics().unwrap().route_decisions, 0);
        let (results, statistics) = crate::Engine::new(graph, system)
            .run_with_summary(|_, _| Ok(()))
            .unwrap();
        assert_eq!(results.len(), 3);
        assert_eq!(statistics.unwrap().route_decisions, 3);
        let collected = assignments.borrow().clone();
        collected
    }

    #[test]
    fn pd_policy_routes_at_arrival_using_current_load_and_not_future_requests() {
        let early_future = routed_run(10.0);
        let late_future = routed_run(20.0);
        assert_eq!(&early_future[..2], &[(0, 0, 1.0), (1, 1, 1.5)]);
        assert_eq!(&early_future[..2], &late_future[..2]);
        assert_eq!(early_future[2], (2, 0, 10.0));
        assert_eq!(late_future[2], (2, 0, 20.0));
    }

    #[test]
    fn decode_handoff_excludes_sampled_token_from_kv() {
        let original = Request {
            request_id: 1,
            arrival_time: 2.0,
            initial: vec![0],
            sub_requests: vec![SubRequest::Llm {
                input_tokens: 16,
                output_tokens: 3,
                known_tokens: Some(16),
                kv_tokens: Some(0),
                next: vec![],
                interval: None,
                model: None,
                input_token_ids: Some((0..16).collect()),
                output_token_ids: Some(vec![20, 21, 22]),
            }],
        };
        let prefill = PdDisaggregationSystem::build_prefill_request(&original).unwrap();
        let decode = PdDisaggregationSystem::build_decode_request(&original, &prefill)
            .unwrap()
            .unwrap();
        assert_eq!(decode.total_input_tokens(), 17);
        assert_eq!(decode.total_output_tokens(), 2);
        assert_eq!(decode.sub_requests[0].known_tokens(), Some(17));
        assert_eq!(decode.sub_requests[0].kv_tokens(), Some(16));
        assert_eq!(decode.request_id, original.request_id);
        assert_eq!(decode.arrival_time, original.arrival_time);
        let tokens = PdDisaggregationSystem::single_llm_sub_request(&decode).unwrap();
        let expected_input = (0..16).chain([20]).collect::<Vec<_>>();
        assert_eq!(tokens.input_token_ids, Some(expected_input.as_slice()));
        assert_eq!(tokens.output_token_ids, Some([21, 22].as_slice()));
    }
}
