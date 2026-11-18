use std::any::Any;
use std::collections::{HashMap, HashSet};
use std::path::Path;

use anyhow::{anyhow, bail, Result};
use serde::Deserialize;

use crate::engine::Graph;
use crate::model::Model;
use crate::model_runner::ModelRunner;
use crate::module::{LogicalHandler, Module};
use crate::policies::{BatchAction, BatchObservation, InstanceId, SharedPolicy};
use crate::scheduler::Scheduler;
use crate::system::{pending_llm_load, System, SystemEvent};
use crate::types::{
    InstanceLoad, NodeID, ReadySubRequest, Request, RequestResult, ResultSubRequest, SubRequest,
    SubRequestCompletion,
};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SingleInstanceSystem {
    model_runner: Box<dyn ModelRunner>,
    model: Box<dyn Model>,
    scheduler: Box<dyn Scheduler>,

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

    #[serde(skip)]
    policy: Option<(SharedPolicy, InstanceId)>,

    #[serde(skip)]
    policy_wakeup: Option<NodeID>,
}

struct ActiveRequest {
    pending_roots: HashSet<usize>,
    request: Request,
    completed_results: Vec<Option<SubRequestCompletion>>,
}

impl ActiveRequest {
    fn new(request: Request) -> Self {
        Self {
            pending_roots: request.initial.iter().copied().collect(),
            completed_results: vec![None; request.sub_requests.len()],
            request,
        }
    }

    fn all_complete(&self) -> bool {
        self.completed_results.iter().all(Option::is_some)
    }
}

crate::register_system!(SingleInstanceSystem, "single_instance");

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
    PolicyWake,
    LlmBatchComplete,
    ToolCallRun {
        duration: f64,
    },
}

fn parse_payload(payload: Box<dyn Any>) -> Result<LogicalPayload> {
    payload
        .downcast::<LogicalPayload>()
        .map(|payload| *payload)
        .map_err(|_| anyhow!("single_instance received an unexpected logical payload"))
}

impl SingleInstanceSystem {
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

impl LogicalHandler for SingleInstanceSystem {
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
                let request = self.pending_requests.remove(&request_id).ok_or_else(|| {
                    anyhow!("single_instance is missing request {request_id} during arrival")
                })?;
                self.active_requests
                    .insert(request_id, ActiveRequest::new(request));
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
                self.scheduler.enqueue_sub_request(ReadySubRequest {
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
                if self.policy_wakeup.is_some() {
                    return Ok(0.0);
                }
                let batch_requests = if let Some((policy, instance)) = &self.policy {
                    if policy.borrow().is_reference() {
                        let batch = self.scheduler.schedule()?;
                        if !batch.is_empty() {
                            policy.borrow_mut().record_reference_batch();
                        }
                        batch
                    } else {
                        let Some(requests) = self.scheduler.policy_queue()? else {
                            return Ok(0.0);
                        };
                        let (default_token_budget, default_max_batch) =
                            self.scheduler.policy_defaults()?;
                        let observation = BatchObservation {
                            now,
                            instance: *instance,
                            requests: &requests,
                            default_token_budget,
                            default_max_batch,
                            memory_limit_bytes: self.scheduler.memory_limit(),
                        };
                        let action = policy.borrow_mut().batch(&observation)?;
                        match action {
                            BatchAction::Run(plan) => self.scheduler.schedule_batch(&plan.items)?,
                            BatchAction::Wait { until } => {
                                let wake = graph.add_logical_node_at(
                                    module,
                                    LogicalPayload::PolicyWake,
                                    until,
                                );
                                graph.add_edge(current, wake)?;
                                self.policy_wakeup = Some(wake);
                                return Ok(0.0);
                            }
                        }
                    }
                } else {
                    self.scheduler.schedule()?
                };
                if batch_requests.is_empty() {
                    return Ok(0.0);
                }
                let last_node = self.model_runner.add_scheduled_iteration(
                    self.model.as_ref(),
                    &batch_requests,
                    &self.scheduler.batch_requests(),
                    current,
                    graph,
                )?;

                let complete = graph.add_logical_node(module, LogicalPayload::LlmBatchComplete);
                graph.add_edge(last_node, complete)?;

                Ok(0.0)
            }
            LogicalPayload::PolicyWake => {
                anyhow::ensure!(
                    self.policy_wakeup == Some(current),
                    "unexpected policy wakeup"
                );
                self.policy_wakeup = None;
                let schedule = graph.add_logical_node(module, LogicalPayload::LlmSchedule);
                graph.add_edge(current, schedule)?;
                Ok(0.0)
            }
            LogicalPayload::LlmBatchComplete => {
                for completion in self.scheduler.done_iteration(now) {
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

impl Module for SingleInstanceSystem {
    fn observe(&self) -> serde_json::Value {
        serde_json::json!({"scheduler": self.scheduler.observe()})
    }

    fn resolve_path(&mut self, base_dir: &Path) -> Result<()> {
        self.model_runner.resolve_path(base_dir)?;
        self.model.resolve_path(base_dir)?;
        self.scheduler.resolve_path(base_dir)?;
        Ok(())
    }
}

impl System for SingleInstanceSystem {
    fn configure_policy(&mut self, policy: SharedPolicy, instance: InstanceId) -> Result<()> {
        anyhow::ensure!(
            self.model_runner.shared_execution(),
            "serving policy requires shared execution"
        );
        anyhow::ensure!(
            self.policy.is_none(),
            "policy already configured for instance"
        );
        self.policy = Some((policy, instance));
        Ok(())
    }

    fn shared_execution(&self) -> bool {
        self.model_runner.shared_execution()
    }
    fn prepare_graph(&mut self, graph: &mut Graph) -> Result<()> {
        self.model_runner
            .prepare_graph(graph, self.scheduler.memory_limit())
    }
    fn completed_requests(&self) -> Option<usize> {
        Some(self.results.len())
    }

    fn instance_load(&self) -> InstanceLoad {
        std::iter::once(self.scheduler.instance_load())
            .chain(
                self.pending_requests
                    .values()
                    .map(|request| pending_llm_load(request, request.initial.iter().copied())),
            )
            .chain(self.active_requests.values().map(|active| {
                pending_llm_load(&active.request, active.pending_roots.iter().copied())
            }))
            .sum()
    }

    fn init(&mut self) -> Result<()> {
        let parallel = self.model_runner.parallel_strategy();
        self.model.init(parallel)?;
        self.scheduler.init(
            self.model.scheduler_model_weight_bytes(),
            self.model.scheduler_kv_bytes_per_request(),
            self.model.scheduler_kv_bytes_per_token(),
            parallel,
        )?;
        if self.model_runner.shared_execution() {
            self.scheduler.use_shared_memory()?;
        }
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
                "single_instance received duplicate request id {}",
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
                "single_instance finished with {} requests still awaiting arrival",
                self.pending_requests.len()
            );
        }
        if !self.active_requests.is_empty() {
            bail!(
                "single_instance finished with {} active requests still incomplete",
                self.active_requests.len()
            );
        }
        if !self.pending_completion_results.is_empty() {
            bail!(
                "single_instance finished with {} queued completion results still pending",
                self.pending_completion_results.len()
            );
        }
        Ok(self.results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::policies::{reference_batch, test_config, PolicyRuntime, Stage};
    use crate::schedulers::chunked_prefill::ChunkedPrefillScheduler;
    use crate::types::{Layer, ParallelStrategy, RequestState};
    use std::cell::RefCell;
    use std::rc::Rc;

    struct FixedModel;
    impl Module for FixedModel {}
    impl Model for FixedModel {
        fn init(&mut self, _: ParallelStrategy) -> Result<()> {
            Ok(())
        }
        fn weight_bytes_per_device(&self) -> u64 {
            0
        }
        fn kv_bytes_per_token_per_device(&self) -> u64 {
            1
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
            0
        }
    }

    struct FixedRunner {
        starts: Rc<RefCell<Vec<(f64, Vec<u64>)>>>,
    }
    impl Module for FixedRunner {}
    impl LogicalHandler for FixedRunner {
        fn handle(
            &mut self,
            payload: Box<dyn Any>,
            now: f64,
            _: NodeID,
            _: &mut Graph,
        ) -> Result<f64> {
            self.starts
                .borrow_mut()
                .push((now, *payload.downcast::<Vec<u64>>().unwrap()));
            Ok(0.25)
        }
    }
    impl ModelRunner for FixedRunner {
        fn shared_execution(&self) -> bool {
            true
        }
        fn parallel_strategy(&self) -> ParallelStrategy {
            ParallelStrategy {
                tp: 1,
                pp: 1,
                ep: 1,
            }
        }
        fn add_iteration(
            &mut self,
            _: &dyn Model,
            _: &[RequestState],
            _: NodeID,
            _: &mut Graph,
        ) -> Result<NodeID> {
            bail!("fixture requires explicit request identities")
        }
        fn add_scheduled_iteration(
            &mut self,
            _: &dyn Model,
            _: &[RequestState],
            requests: &[crate::kv::BatchRequest],
            current: NodeID,
            graph: &mut Graph,
        ) -> Result<NodeID> {
            let node = graph.add_logical_node(
                self as *mut dyn LogicalHandler,
                requests.iter().map(|r| r.request_id).collect::<Vec<_>>(),
            );
            graph.add_edge(current, node)?;
            Ok(node)
        }
    }

    fn request(id: u64, arrival_time: f64) -> Request {
        Request {
            request_id: id,
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
        }
    }

    #[test]
    fn policy_wait_wakes_without_arrivals_and_arrivals_cannot_bypass_it() {
        for second_arrival in [None, Some(0.25)] {
            let starts = Rc::new(RefCell::new(Vec::new()));
            let scheduler: ChunkedPrefillScheduler = toml::from_str("block_size=4\nmax_batch=4\nmax_num_batched_tokens=8\ndevice_mem=1024\ndevice_mem_utilization=1.0\nenable_prefix_caching=false").unwrap();
            let mut system = Box::new(SingleInstanceSystem {
                model: Box::new(FixedModel),
                model_runner: Box::new(FixedRunner {
                    starts: starts.clone(),
                }),
                scheduler: Box::new(scheduler),
                results: Vec::new(),
                pending_requests: HashMap::new(),
                request_sinks: HashMap::new(),
                active_requests: HashMap::new(),
                pending_completion_results: HashMap::new(),
                policy: None,
                policy_wakeup: None,
            });
            let mut runtime = PolicyRuntime::new(test_config()).unwrap();
            runtime.set_batch_for_test(|observation, context, memory| {
                if memory[0] == 0 {
                    memory[0] = 1;
                    BatchAction::Wait {
                        until: observation.now + 0.5,
                    }
                } else {
                    reference_batch(observation, context, memory)
                }
            });
            let policy = Rc::new(RefCell::new(runtime));
            system
                .configure_policy(
                    policy.clone(),
                    InstanceId {
                        stage: Stage::Prefill,
                        index: 0,
                    },
                )
                .unwrap();
            system.init().unwrap();
            let mut graph = Graph::new();
            system
                .add_request_arrival(request(0, 0.0), &mut graph, None, None)
                .unwrap();
            if let Some(at) = second_arrival {
                system
                    .add_request_arrival(request(1, at), &mut graph, None, None)
                    .unwrap();
            }
            let results = crate::Engine::new(graph, system).run().unwrap();
            assert_eq!(starts.borrow().len(), 1);
            assert_eq!(starts.borrow()[0].0, 0.5);
            assert_eq!(
                starts.borrow()[0].1.len(),
                if second_arrival.is_some() { 2 } else { 1 }
            );
            for result in results {
                let ResultSubRequest::Llm { token_latency, .. } = &result.sub_requests[0] else {
                    unreachable!()
                };
                assert_eq!(token_latency, &vec![0.75 - result.arrival_time]);
            }
            assert_eq!(policy.borrow().statistics().waits, 1);
            assert_eq!(policy.borrow().statistics().batch_decisions, 2);
        }
    }
}
