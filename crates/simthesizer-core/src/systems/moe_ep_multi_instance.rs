use std::any::Any;
use std::collections::{HashMap, HashSet};
use std::path::Path;

use anyhow::{anyhow, bail, Result};
use serde::Deserialize;

use crate::engine::Graph;
use crate::model::Model;
use crate::model_runner::ModelRunner;
use crate::module::{LogicalHandler, Module};
use crate::router::Router;
use crate::scheduler::Scheduler;
use crate::system::{pending_llm_load, System, SystemEvent};
use crate::types::{
    InstanceLoad, NodeID, ParallelStrategy, ReadySubRequest, Request, RequestResult,
    ResultSubRequest, SubRequest, SubRequestCompletion,
};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MoeEpMultiInstanceSystem {
    model_runner: Box<dyn ModelRunner>,
    model: Box<dyn Model>,
    schedulers: Vec<Box<dyn Scheduler>>,
    #[serde(default = "default_router")]
    router: Box<dyn Router>,

    #[serde(skip_deserializing, default)]
    results: Vec<RequestResult>,

    #[serde(skip_deserializing, default)]
    pending_requests: HashMap<u64, Request>,

    #[serde(skip_deserializing, default)]
    request_sinks: HashMap<u64, *mut dyn LogicalHandler>,

    #[serde(skip_deserializing, default)]
    active_requests: HashMap<u64, ActiveRequest>,

    #[serde(skip_deserializing, default)]
    pending_completion_results: HashMap<NodeID, SubRequestCompletion>,

    #[serde(skip_deserializing, default)]
    wave_running: bool,
}

struct ActiveRequest {
    pending_roots: HashSet<usize>,
    request: Request,
    instance_index: usize,
    completed_results: Vec<Option<SubRequestCompletion>>,
}

impl ActiveRequest {
    fn new(request: Request, instance_index: usize) -> Self {
        let len = request.sub_requests.len();
        Self {
            pending_roots: request.initial.iter().copied().collect(),
            request,
            instance_index,
            completed_results: vec![None; len],
        }
    }

    fn all_complete(&self) -> bool {
        self.completed_results.iter().all(Option::is_some)
    }
}

crate::register_system!(MoeEpMultiInstanceSystem, "moe_ep_multi_instance");

fn default_router() -> Box<dyn Router> {
    Box::new(crate::routers::round_robin::RoundRobinRouter::default())
}

#[derive(Debug)]
enum LogicalPayload {
    RequestArrival {
        request_id: u64,
    },
    SubRequestArrival {
        request_id: u64,
        subrequest_index: usize,
    },
    SubRequestComplete {
        request_id: u64,
        subrequest_index: usize,
    },
    LlmEnqueue {
        request_id: u64,
        subrequest_index: usize,
        completion_node_id: NodeID,
    },
    LlmSchedule,
    LlmBatchComplete,
    ToolCallRun {
        duration: f64,
    },
}

fn parse_payload(payload: Box<dyn Any>) -> Result<LogicalPayload> {
    payload
        .downcast::<LogicalPayload>()
        .map(|payload| *payload)
        .map_err(|_| anyhow!("moe_ep_multi_instance received an unexpected logical payload"))
}

impl MoeEpMultiInstanceSystem {
    fn rank_loads(&self) -> Vec<InstanceLoad> {
        let mut loads: Vec<_> = self.schedulers.iter().map(|s| s.instance_load()).collect();
        for active in self.active_requests.values() {
            let index = active.instance_index;
            loads[index] = [
                loads[index],
                pending_llm_load(&active.request, active.pending_roots.iter().copied()),
            ]
            .into_iter()
            .sum();
        }
        loads
    }

    fn detached_completion_node(&mut self, current: NodeID, graph: &mut Graph) -> Result<NodeID> {
        let detached = graph.pop_edges(current)?;
        match detached.as_slice() {
            [completion_node_id] => Ok(*completion_node_id),
            _ => bail!(
                "sub-request arrival node {current} detached {} children, expected exactly one completion node",
                detached.len()
            ),
        }
    }

    fn queue_completion(
        &mut self,
        completion_node_id: NodeID,
        completion: SubRequestCompletion,
    ) -> Result<()> {
        if completion.completion_node_id != completion_node_id {
            bail!(
                "completion node mismatch: payload carries {} but queued under {}",
                completion.completion_node_id,
                completion_node_id
            );
        }
        if self
            .pending_completion_results
            .insert(completion_node_id, completion)
            .is_some()
        {
            bail!("completion node {completion_node_id} was queued more than once");
        }
        Ok(())
    }

    fn instance_index(&self, request_id: u64) -> Result<usize> {
        self.active_requests
            .get(&request_id)
            .map(|active| active.instance_index)
            .ok_or_else(|| anyhow!("unknown active request {request_id}"))
    }

    fn finalize_completed_request(
        &mut self,
        request_id: u64,
        current: NodeID,
        graph: &mut Graph,
    ) -> Result<()> {
        let Some(active) = self.active_requests.get(&request_id) else {
            bail!("request {request_id} disappeared before finalization");
        };
        if !active.all_complete() {
            return Ok(());
        }

        let active = self
            .active_requests
            .remove(&request_id)
            .ok_or_else(|| anyhow!("request {request_id} disappeared during finalization"))?;
        let request = active.request;
        let completed_results = active.completed_results;
        let sub_requests = request
            .sub_requests
            .iter()
            .enumerate()
            .map(|(index, sub_request)| {
                let completion = completed_results
                    .get(index)
                    .and_then(Option::as_ref)
                    .ok_or_else(|| {
                        anyhow!("request {request_id} missing completion for sub-request {index}")
                    })?;
                Ok(match sub_request {
                    SubRequest::Llm {
                        input_tokens,
                        output_tokens,
                        next,
                        interval,
                        model,
                        ..
                    } => ResultSubRequest::Llm {
                        input_tokens: *input_tokens,
                        output_tokens: *output_tokens,
                        next: next.clone(),
                        token_latency: completion.token_latency.clone(),
                        interval: *interval,
                        model: model.clone(),
                    },
                    SubRequest::ToolCall {
                        input_tokens,
                        output_tokens,
                        next,
                        duration,
                        interval,
                    } => ResultSubRequest::ToolCall {
                        input_tokens: *input_tokens,
                        output_tokens: *output_tokens,
                        next: next.clone(),
                        duration: duration.unwrap_or(0.0),
                        interval: *interval,
                    },
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let result = RequestResult {
            request_id,
            arrival_time: request.arrival_time,
            initial: request.initial.clone(),
            sub_requests,
        };
        if let Some(sink) = self.request_sinks.remove(&request_id) {
            let event =
                graph.add_logical_node(sink, SystemEvent::RequestCompleted { request, result });
            graph.add_edge(current, event)?;
        } else {
            self.results.push(result);
        }
        Ok(())
    }
}

impl LogicalHandler for MoeEpMultiInstanceSystem {
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
        let module = self as *mut Self as *mut dyn LogicalHandler;
        match parse_payload(payload)? {
            LogicalPayload::RequestArrival { request_id } => {
                let pending = self.pending_requests.remove(&request_id).ok_or_else(|| {
                    anyhow!("moe_ep_multi_instance is missing request {request_id} during arrival")
                })?;
                let loads = self.rank_loads();
                let instance_index = self.router.select_instance(&pending, &loads)?;
                if instance_index >= self.schedulers.len() {
                    bail!("router selected invalid instance index {instance_index}");
                }
                self.active_requests
                    .insert(request_id, ActiveRequest::new(pending, instance_index));
                Ok(0.0)
            }
            LogicalPayload::SubRequestArrival {
                request_id,
                subrequest_index,
            } => {
                let completion_node_id = self.detached_completion_node(current, graph)?;
                let sub_request = {
                    let active = self
                        .active_requests
                        .get(&request_id)
                        .ok_or_else(|| anyhow!("unknown active request {request_id}"))?;
                    active
                        .request
                        .sub_requests
                        .get(subrequest_index)
                        .ok_or_else(|| anyhow!("unknown subrequest index {subrequest_index}"))?
                        .clone()
                };

                let interval = sub_request.interval().unwrap_or(0.0);

                match sub_request {
                    SubRequest::Llm { .. } => {
                        let enqueue = graph.add_logical_node(
                            module,
                            LogicalPayload::LlmEnqueue {
                                request_id,
                                subrequest_index,
                                completion_node_id,
                            },
                        );
                        graph.add_edge(current, enqueue)?;
                    }
                    SubRequest::ToolCall { duration, .. } => {
                        self.queue_completion(
                            completion_node_id,
                            SubRequestCompletion {
                                request_id,
                                subrequest_index,
                                completion_node_id,
                                token_latency: Vec::new(),
                            },
                        )?;
                        let run = graph.add_logical_node(
                            module,
                            LogicalPayload::ToolCallRun {
                                duration: duration.unwrap_or(0.0),
                            },
                        );
                        graph.add_edge(current, run)?;
                        graph.add_edge(run, completion_node_id)?;
                    }
                }

                Ok(interval)
            }
            LogicalPayload::SubRequestComplete {
                request_id,
                subrequest_index,
            } => {
                let completion = self
                    .pending_completion_results
                    .remove(&current)
                    .ok_or_else(|| anyhow!("completion node {current} has no queued result"))?;
                if completion.request_id != request_id
                    || completion.subrequest_index != subrequest_index
                    || completion.completion_node_id != current
                {
                    bail!("completion node {current} received mismatched completion payload");
                }

                let active = self
                    .active_requests
                    .get_mut(&request_id)
                    .ok_or_else(|| anyhow!("completion for unknown request {request_id}"))?;
                if active.completed_results[subrequest_index]
                    .replace(completion)
                    .is_some()
                {
                    bail!(
                        "request {request_id} sub-request {subrequest_index} completed more than once"
                    );
                }

                self.finalize_completed_request(request_id, current, graph)?;
                Ok(0.0)
            }
            LogicalPayload::LlmEnqueue {
                request_id,
                subrequest_index,
                completion_node_id,
            } => {
                let instance_index = self.instance_index(request_id)?;
                let (
                    input_tokens,
                    output_tokens,
                    input_token_ids,
                    output_token_ids,
                    known_tokens,
                    kv_tokens,
                ) = {
                    let active = self
                        .active_requests
                        .get_mut(&request_id)
                        .ok_or_else(|| anyhow!("unknown active request {request_id}"))?;
                    active.pending_roots.remove(&subrequest_index);
                    let sub_request = active
                        .request
                        .sub_requests
                        .get(subrequest_index)
                        .ok_or_else(|| anyhow!("unknown subrequest index {subrequest_index}"))?;
                    let SubRequest::Llm {
                        input_tokens,
                        output_tokens,
                        known_tokens,
                        kv_tokens,
                        input_token_ids,
                        output_token_ids,
                        ..
                    } = sub_request
                    else {
                        bail!("LlmEnqueue received non-Llm sub-request");
                    };
                    let known_tokens = known_tokens.unwrap_or(*input_tokens);
                    let kv_tokens = kv_tokens.unwrap_or(0);
                    (
                        *input_tokens,
                        *output_tokens,
                        input_token_ids.clone(),
                        output_token_ids.clone(),
                        known_tokens,
                        kv_tokens,
                    )
                };
                self.schedulers[instance_index].enqueue_sub_request(ReadySubRequest {
                    request_id,
                    subrequest_index,
                    completion_node_id,
                    arrival_time: now,
                    input_tokens,
                    output_tokens,
                    known_tokens,
                    kv_tokens,
                    input_token_ids,
                    output_token_ids,
                });
                let schedule = graph.add_logical_node(module, LogicalPayload::LlmSchedule);
                graph.add_edge(current, schedule)?;
                Ok(0.0)
            }
            LogicalPayload::LlmSchedule => {
                if self.wave_running {
                    return Ok(0.0);
                }

                let mut instance_batches = Vec::with_capacity(self.schedulers.len());
                let mut has_work = false;
                for scheduler in &mut self.schedulers {
                    let batch = scheduler.schedule()?;
                    has_work |= !batch.is_empty();
                    instance_batches.push(batch);
                }
                if !has_work {
                    return Ok(0.0);
                }

                self.wave_running = true;
                let last_node = self.model_runner.add_multi_iteration(
                    self.model.as_ref(),
                    &instance_batches,
                    current,
                    graph,
                )?;
                let complete = graph.add_logical_node(module, LogicalPayload::LlmBatchComplete);
                graph.add_edge(last_node, complete)?;
                Ok(0.0)
            }
            LogicalPayload::LlmBatchComplete => {
                self.wave_running = false;
                let mut completions = Vec::new();
                for scheduler in &mut self.schedulers {
                    completions.extend(scheduler.done_iteration(now));
                }
                for completion in completions {
                    let completion_node_id = completion.completion_node_id;
                    self.queue_completion(completion_node_id, completion)?;
                    graph.add_edge(current, completion_node_id)?;
                }

                let schedule = graph.add_logical_node(module, LogicalPayload::LlmSchedule);
                graph.add_edge(current, schedule)?;
                Ok(0.0)
            }
            LogicalPayload::ToolCallRun { duration } => Ok(duration),
        }
    }
}

impl Module for MoeEpMultiInstanceSystem {
    fn observe(&self) -> serde_json::Value {
        serde_json::json!({
            "schedulers": self.schedulers.iter().map(|s| s.observe()).collect::<Vec<_>>()
        })
    }

    fn resolve_path(&mut self, base_dir: &Path) -> Result<()> {
        self.model_runner.resolve_path(base_dir)?;
        self.model.resolve_path(base_dir)?;
        self.router.resolve_path(base_dir)?;
        for scheduler in &mut self.schedulers {
            scheduler.resolve_path(base_dir)?;
        }
        Ok(())
    }
}

impl System for MoeEpMultiInstanceSystem {
    fn instance_load(&self) -> InstanceLoad {
        self.rank_loads()
            .into_iter()
            .chain(
                self.pending_requests
                    .values()
                    .map(|request| pending_llm_load(request, request.initial.iter().copied())),
            )
            .sum()
    }

    fn init(&mut self) -> Result<()> {
        if self.schedulers.is_empty() {
            bail!("moe_ep_multi_instance requires at least one scheduler");
        }

        let parallel = self.model_runner.parallel_strategy();
        if parallel.ep as usize != self.schedulers.len() {
            bail!(
                "moe_ep_multi_instance requires exactly one scheduler per EP/DP instance: ep={} schedulers={}",
                parallel.ep,
                self.schedulers.len()
            );
        }

        self.model.init(parallel)?;
        let local_parallel = ParallelStrategy {
            tp: parallel.tp,
            pp: parallel.pp,
            ep: 1,
        };
        for scheduler in &mut self.schedulers {
            scheduler.init(
                self.model.weight_bytes_per_device(),
                self.model.kv_bytes_per_request_per_device(),
                self.model.kv_bytes_per_token_per_device(),
                local_parallel,
            )?;
        }
        self.wave_running = false;
        self.model_runner.init(self.model.as_ref())
    }

    fn add_request_arrival(
        &mut self,
        request: Request,
        graph: &mut Graph,
        parent: Option<NodeID>,
        sink: Option<*mut dyn LogicalHandler>,
    ) -> Result<()> {
        request.validate()?;
        if self.pending_requests.contains_key(&request.request_id)
            || self.active_requests.contains_key(&request.request_id)
        {
            return Err(anyhow!(
                "moe_ep_multi_instance received duplicate request id {}",
                request.request_id
            ));
        }

        let module = self as *mut Self as *mut dyn LogicalHandler;
        let request_id = request.request_id;
        let arrival_time = request.arrival_time;
        let initial = request.initial.clone();
        let successors = request
            .sub_requests
            .iter()
            .map(|sub_request| sub_request.next().to_vec())
            .collect::<Vec<_>>();
        self.pending_requests.insert(request_id, request);
        if let Some(sink) = sink {
            self.request_sinks.insert(request_id, sink);
        } else {
            self.request_sinks.remove(&request_id);
        }

        let request_arrival = graph.add_logical_node_at(
            module,
            LogicalPayload::RequestArrival { request_id },
            arrival_time,
        );
        if let Some(parent) = parent {
            graph.add_edge(parent, request_arrival)?;
        }

        let mut arrival_nodes = Vec::with_capacity(successors.len());
        let mut completion_nodes = Vec::with_capacity(successors.len());
        for subrequest_index in 0..successors.len() {
            arrival_nodes.push(graph.add_logical_node(
                module,
                LogicalPayload::SubRequestArrival {
                    request_id,
                    subrequest_index,
                },
            ));
            completion_nodes.push(graph.add_logical_node(
                module,
                LogicalPayload::SubRequestComplete {
                    request_id,
                    subrequest_index,
                },
            ));
        }

        for subrequest_index in initial {
            graph.add_edge(request_arrival, arrival_nodes[subrequest_index])?;
        }
        for (subrequest_index, next_indices) in successors.into_iter().enumerate() {
            graph.add_edge(
                arrival_nodes[subrequest_index],
                completion_nodes[subrequest_index],
            )?;
            for next_index in next_indices {
                graph.add_edge(
                    completion_nodes[subrequest_index],
                    arrival_nodes[next_index],
                )?;
            }
        }

        Ok(())
    }

    fn into_results(self: Box<Self>) -> Result<Vec<RequestResult>> {
        if !self.pending_requests.is_empty() {
            bail!(
                "moe_ep_multi_instance finished with {} requests still awaiting arrival",
                self.pending_requests.len()
            );
        }
        if !self.active_requests.is_empty() {
            bail!(
                "moe_ep_multi_instance finished with {} active requests still incomplete",
                self.active_requests.len()
            );
        }
        if !self.pending_completion_results.is_empty() {
            bail!(
                "moe_ep_multi_instance finished with {} queued completion results still pending",
                self.pending_completion_results.len()
            );
        }
        Ok(self.results)
    }
}
