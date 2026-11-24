//! Rooflang whole-iteration and shared operator bridges (JSONL protocol v1).
use crate::types::{NodeID, ParallelStrategy, RequestState};
use crate::wall_profile::{self, scope, Section};
use crate::{Graph, LogicalHandler, Model, ModelRunner, Module};
use anyhow::{bail, ensure, Context, Result};
use serde::Deserialize;
use serde_json::{json, Value};
use std::any::Any;
use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};

#[derive(Clone, Copy, Default, Deserialize, PartialEq, Eq)]
enum GraphModel {
    #[default]
    #[serde(rename = "llama")]
    Llama,
    #[serde(rename = "qwen3_235b_a22b")]
    Qwen3,
}

fn one() -> u32 {
    1
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RooflangRunner {
    #[serde(default)]
    shared_execution: bool,
    #[serde(default)]
    stage: Option<String>,
    #[serde(skip)]
    resolved_hardware: Value,
    #[serde(skip)]
    weight_bytes: u64,
    #[serde(skip)]
    round: u64,
    tp: u32,
    #[serde(default = "one")]
    ep: u32,
    #[serde(default = "one")]
    expert_tp: u32,
    #[serde(default)]
    attention_groups: Vec<Vec<usize>>,
    #[serde(default)]
    graph_model: GraphModel,
    #[serde(default)]
    routing_seed: u64,
    #[serde(skip)]
    layers: u32,
    /// Stable identity in a disjoint instance pool; also joins outer and inner traces.
    #[serde(default)]
    instance_id: Option<String>,
    /// Executable and arguments. No shell expansion is performed.
    command: Vec<String>,
    hardware: Value,
    #[serde(skip)]
    worker: Option<Worker>,
    #[serde(skip)]
    hardware_label: String,
}
struct Worker {
    child: Child,
    input: ChildStdin,
    output: BufReader<ChildStdout>,
}

type WorkerSections = std::collections::BTreeMap<String, wall_profile::WorkerSection>;

#[derive(Deserialize)]
struct WorkerTiming {
    sections: WorkerSections,
}

trait WorkerResponse: serde::de::DeserializeOwned {
    fn error(&self) -> Option<&Value>;
    fn take_timing(&mut self) -> Result<Option<WorkerSections>>;
}

impl WorkerResponse for Value {
    fn error(&self) -> Option<&Value> {
        self.get("error")
    }

    fn take_timing(&mut self) -> Result<Option<WorkerSections>> {
        self.get_mut("wall_profile")
            .and_then(|value| value.get_mut("sections"))
            .map(|sections| {
                serde_json::from_value(sections.take()).context("invalid worker wall profile")
            })
            .transpose()
    }
}

#[derive(Deserialize)]
struct GraphResponse {
    operations: Option<Vec<WireOperation>>,
    operation_rows: Option<Vec<OperationRow>>,
    #[serde(default)]
    work_templates: Vec<crate::resources::Work>,
    workspace: Option<std::collections::BTreeMap<String, u64>>,
    error: Option<Value>,
    wall_profile: Option<WorkerTiming>,
}

// Parents, work template, layer, KV rank, batch rows. The row index is the node ID.
type OperationRow = (
    Vec<usize>,
    usize,
    Option<u32>,
    Option<usize>,
    Option<Vec<usize>>,
);

impl GraphResponse {
    fn take_operations(&mut self, instance: Option<&str>) -> Result<Vec<Operation>> {
        let templates: Vec<_> = std::mem::take(&mut self.work_templates)
            .into_iter()
            .map(|work| scope_work(work, instance))
            .collect();
        match (self.operations.take(), self.operation_rows.take()) {
            (Some(operations), None) => operations
                .into_iter()
                .map(|operation| operation.resolve(&templates, instance))
                .collect(),
            (None, Some(rows)) => rows
                .into_iter()
                .enumerate()
                .map(|(id, (parents, work_id, layer, kv_rank, batch_indices))| {
                    WireOperation {
                        id,
                        parents,
                        layer,
                        kv_rank,
                        batch_indices,
                        work: None,
                        work_id: Some(work_id),
                    }
                    .resolve(&templates, instance)
                })
                .collect(),
            _ => bail!("Rooflang response requires exactly one of operations or operation_rows"),
        }
    }
}

#[derive(Deserialize)]
struct WireOperation {
    id: usize,
    parents: Vec<usize>,
    layer: Option<u32>,
    kv_rank: Option<usize>,
    batch_indices: Option<Vec<usize>>,
    work: Option<crate::resources::Work>,
    work_id: Option<usize>,
}

impl WireOperation {
    fn resolve(
        self,
        templates: &[crate::resources::SharedWork],
        instance: Option<&str>,
    ) -> Result<Operation> {
        let work = match (self.work, self.work_id) {
            (Some(work), None) => scope_work(work, instance),
            (None, Some(id)) => templates
                .get(id)
                .context("invalid Rooflang work template index")?
                .clone(),
            _ => bail!("Rooflang operation requires exactly one of work or work_id"),
        };
        Ok(Operation {
            id: self.id,
            parents: self.parents,
            layer: self.layer,
            kv_rank: self.kv_rank,
            batch_indices: self.batch_indices,
            work,
        })
    }
}

fn scope_work(
    mut work: crate::resources::Work,
    instance: Option<&str>,
) -> crate::resources::SharedWork {
    if let Some(instance) = instance {
        for slot in &mut work.slots {
            *slot = format!("{instance}:{slot}");
        }
    }
    crate::resources::SharedWork::new(work)
}

impl WorkerResponse for GraphResponse {
    fn error(&self) -> Option<&Value> {
        self.error.as_ref()
    }

    fn take_timing(&mut self) -> Result<Option<WorkerSections>> {
        Ok(self.wall_profile.take().map(|timing| timing.sections))
    }
}
impl Drop for Worker {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}
impl Worker {
    fn request<R: WorkerResponse>(&mut self, mut value: Value) -> Result<R> {
        let _wall = scope(Section::WorkerRoundtrip);
        if wall_profile::enabled() {
            value["profile_wall"] = json!(true);
        }
        serde_json::to_writer(&mut self.input, &value)?;
        self.input.write_all(b"\n")?;
        self.input.flush()?;
        let mut line = String::new();
        ensure!(
            self.output.read_line(&mut line)? > 0,
            "Rooflang worker exited without a response"
        );
        let mut response: R = {
            let _wall = scope(Section::WorkerJsonDecode);
            serde_json::from_str(&line).context("invalid Rooflang response")?
        };
        if wall_profile::enabled() {
            if let Some(sections) = response.take_timing()? {
                ensure!(
                    sections
                        .values()
                        .all(|s| s.seconds.is_finite() && s.seconds >= 0.0),
                    "invalid worker wall-profile duration"
                );
                wall_profile::add_worker_sections(sections);
            }
        }
        if let Some(error) = response.error() {
            bail!("Rooflang: {error}");
        }
        Ok(response)
    }
}
crate::register_model_runner!(RooflangRunner, "rooflang");
impl Module for RooflangRunner {}
impl LogicalHandler for RooflangRunner {
    fn describe(&self, payload: &dyn Any) -> String {
        let identity = self
            .instance_id
            .as_ref()
            .map_or_else(String::new, |id| format!("instance_id: {}, ", json!(id)));
        format!(
            "RooflangIteration {{ {}hardware: {}, duration_s: {:?} }}",
            identity,
            self.hardware_label,
            payload.downcast_ref::<f64>()
        )
    }
    fn handle(&mut self, payload: Box<dyn Any>, _: f64, _: NodeID, _: &mut Graph) -> Result<f64> {
        payload
            .downcast::<f64>()
            .map(|v| *v)
            .map_err(|_| anyhow::anyhow!("invalid Rooflang iteration payload"))
    }
}
impl ModelRunner for RooflangRunner {
    fn shared_execution(&self) -> bool {
        self.shared_execution
    }
    fn prepare_graph(&mut self, graph: &mut Graph, memory_limit: Option<u64>) -> Result<()> {
        if !self.shared_execution {
            return Ok(());
        }
        let mut weights = std::collections::BTreeMap::new();
        for d in self.resolved_hardware["devices"]
            .as_array()
            .context("missing hardware devices")?
        {
            let name = d["name"].as_str().context("missing GPU name")?.to_owned();
            let capacity = (d["memory_capacity_gb"]
                .as_f64()
                .context("missing memory capacity")?
                * 1e9) as u64;
            graph.resources.capacity(
                name.clone(),
                capacity
                    .min(memory_limit.context("shared execution requires scheduler memory limit")?),
            )?;
            graph.kv.hbm_seconds_per_byte.insert(
                name.clone(),
                1.0 / (d["memory_bandwidth_gbs"]
                    .as_f64()
                    .context("missing HBM bandwidth")?
                    * 1e9),
            );
            weights.insert(name, self.weight_bytes);
        }
        graph.resources.resident(crate::resources::Allocation {
            id: format!(
                "weights:{}",
                self.instance_id
                    .as_deref()
                    .context("shared execution requires instance_id")?
            ),
            bytes: weights,
        })
    }

    fn add_scheduled_iteration(
        &mut self,
        model: &dyn Model,
        batch: &[RequestState],
        requests: &[crate::kv::BatchRequest],
        current: NodeID,
        graph: &mut Graph,
    ) -> Result<NodeID> {
        if !self.shared_execution {
            return self.add_iteration(model, batch, current, graph);
        }
        self.add_shared_iteration(batch, requests, current, graph)
    }
    fn parallel_strategy(&self) -> ParallelStrategy {
        ParallelStrategy {
            tp: self.tp,
            pp: 1,
            ep: self.ep,
        }
    }
    fn init_model(&self, model: &mut dyn Model) -> Result<()> {
        self.validate_configuration()?;
        if self.graph_model == GraphModel::Qwen3 {
            model.init_with_expert_tp(self.parallel_strategy(), self.expert_tp)
        } else {
            model.init(self.parallel_strategy())
        }
    }
    fn init(&mut self, model: &dyn Model) -> Result<()> {
        self.validate_configuration()?;
        if self.graph_model == GraphModel::Qwen3 && self.attention_groups.is_empty() {
            // Compatibility for old TP1 configurations; new configs provide the layout.
            self.attention_groups = (0..self.ep as usize).map(|rank| vec![rank]).collect();
        }
        self.weight_bytes = model.weight_bytes_per_device();
        self.layers = model.num_blocks();
        ensure!(!self.command.is_empty(), "Rooflang command is empty");
        let mut child = Command::new(&self.command[0])
            .args(&self.command[1..])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .context("starting Rooflang worker")?;
        let input = child.stdin.take().context("worker stdin")?;
        let output = BufReader::new(child.stdout.take().context("worker stdout")?);
        let mut worker = Worker {
            child,
            input,
            output,
        };
        let mut init = json!({"op":"init", "version":1,"tp":self.tp,"hardware":self.hardware,"instance_id":self.instance_id,"shared_execution":self.shared_execution,
            "memory":{"weight_bytes_per_device":model.weight_bytes_per_device(),"kv_bytes_per_token_per_device":model.kv_bytes_per_token_per_device()},
            "model":{"pre":model.pre_block_layers(),"block":model.block_layers(),"post":model.post_block_layers(),"layers":model.num_blocks()}});
        if self.graph_model == GraphModel::Qwen3 {
            init["model_kind"] = json!("qwen3_235b_a22b");
            init["ep"] = json!(self.ep);
            init["expert_tp"] = json!(self.expert_tp);
            init["attention_groups"] = json!(self.attention_groups);
            init["routing_seed"] = json!(self.routing_seed);
        }
        let response: Value = worker.request(init)?;
        ensure!(
            response["ready"] == true,
            "Rooflang worker did not initialize"
        );
        let hardware = &response["hardware"];
        if self.graph_model == GraphModel::Qwen3 {
            ensure!(
                hardware["devices"].as_array().map(Vec::len) == Some(self.ep as usize),
                "Qwen Rooflang hardware must contain one device per EP rank"
            );
        }
        self.resolved_hardware = hardware.clone();
        self.hardware_label = format!(
            "{} x {}",
            hardware["gpu_count"]
                .as_u64()
                .context("missing GPU count")?,
            hardware["gpu_model"]
                .as_str()
                .context("missing GPU model")?
        );
        eprintln!(
            "Rooflang hardware: {} (TP={})",
            self.hardware_label, self.tp
        );
        self.worker = Some(worker);
        Ok(())
    }
    fn add_iteration(
        &mut self,
        _: &dyn Model,
        batch: &[RequestState],
        current: NodeID,
        graph: &mut Graph,
    ) -> Result<NodeID> {
        let _wall = scope(Section::GraphBuild);
        ensure!(!batch.is_empty(), "empty Rooflang batch");
        let response: Value = self.worker.as_mut().context("Rooflang runner is not initialized")?.request(json!({"op":"iteration", "batch":batch.iter().map(|r|json!({"q_len":r.q_len,"kv_len":r.kv_len,"lm_head_len":r.lm_head_len})).collect::<Vec<_>>()}))?;
        let us = response["duration_us"]
            .as_f64()
            .context("missing Rooflang duration_us")?;
        ensure!(
            us.is_finite() && us > 0.0,
            "invalid Rooflang duration: {us}"
        );
        let node = graph.add_logical_node(self as *mut dyn LogicalHandler, us * 1e-6);
        graph.add_edge(current, node)?;
        Ok(node)
    }
    fn add_multi_iteration(
        &mut self,
        _: &dyn Model,
        _: &[Vec<RequestState>],
        _: NodeID,
        _: &mut Graph,
    ) -> Result<NodeID> {
        bail!("Rooflang backend requires exclusive resources and single-instance execution")
    }
}

#[derive(Deserialize)]
struct Operation {
    id: usize,
    parents: Vec<usize>,
    layer: Option<u32>,
    kv_rank: Option<usize>,
    #[serde(default)]
    batch_indices: Option<Vec<usize>>,
    work: crate::resources::SharedWork,
}

impl RooflangRunner {
    fn validate_configuration(&self) -> Result<()> {
        if self.shared_execution {
            ensure!(
                matches!(self.stage.as_deref(), Some("p" | "d")),
                "shared Rooflang requires stage p or d"
            );
        }
        match self.graph_model {
            GraphModel::Llama => {
                ensure!(self.ep == 1, "Rooflang Llama requires EP 1");
                ensure!(
                    self.expert_tp == 1 && self.attention_groups.is_empty(),
                    "Llama does not accept expert TP or attention-DP groups"
                );
                ensure!(
                    self.tp > 0 && self.tp <= 8 && 8 % self.tp == 0,
                    "Rooflang Llama supports TP 1, 2, 4, 8"
                );
            }
            GraphModel::Qwen3 => {
                ensure!(
                    self.shared_execution,
                    "Qwen Rooflang requires shared execution"
                );
                ensure!(
                    matches!(self.ep, 1 | 2 | 4 | 8),
                    "Qwen Rooflang supports EP 1, 2, 4, 8"
                );
                ensure!(
                    matches!(self.tp, 1 | 2 | 4) && self.ep.is_multiple_of(self.tp),
                    "Qwen Rooflang requires Attention TP 1, 2, 4 dividing EP"
                );
                ensure!(self.expert_tp == 1, "Qwen Rooflang requires expert_tp=1");
                if self.attention_groups.is_empty() {
                    ensure!(self.tp == 1, "Qwen TP>1 requires explicit attention_groups");
                } else {
                    ensure!(
                        self.attention_groups.len() == (self.ep / self.tp) as usize
                            && self
                                .attention_groups
                                .iter()
                                .all(|g| g.len() == self.tp as usize)
                            && self
                                .attention_groups
                                .iter()
                                .flatten()
                                .copied()
                                .eq(0..self.ep as usize),
                        "Qwen attention_groups must partition EP ranks into consecutive TP groups"
                    );
                }
            }
        }
        Ok(())
    }

    fn validate_qwen_batch(
        &self,
        requests: &[crate::kv::BatchRequest],
        graph: &Graph,
    ) -> Result<()> {
        let devices = self.resolved_hardware["devices"]
            .as_array()
            .context("missing Qwen hardware devices")?;
        let prefill = self.stage.as_deref() == Some("p");
        for request in requests {
            let plan = graph
                .kv
                .plans
                .get(&request.request_id)
                .context("missing Qwen KV plan")?;
            let owners = if prefill { &plan.source } else { &plan.target };
            let group = self
                .attention_groups
                .get((request.request_id % u64::from(self.ep / self.tp)) as usize)
                .context("missing normalized Qwen attention group")?;
            ensure!(
                owners.len() == self.tp as usize
                    && owners.iter().enumerate().all(|(rank, owner)| {
                        devices.get(group[rank]).and_then(|d| d["name"].as_str())
                            == Some(owner.as_str())
                    }),
                "Qwen KV plan must use the request's complete Attention TP group"
            );
            ensure!(
                plan.heads == 4
                    && plan.layers == self.layers
                    && plan.bytes_per_token_head_layer == 512,
                "Qwen KV plan must preserve all four complete KV heads per layer"
            );
        }
        Ok(())
    }

    fn validate_qwen_operations(&self, operations: &[Operation], batch_len: usize) -> Result<()> {
        let mut coverage = std::collections::BTreeSet::new();
        let mut ids = (!operations.iter().enumerate().all(|(id, op)| id == op.id))
            .then(std::collections::BTreeSet::new);
        for op in operations {
            if let Some(ids) = &mut ids {
                ensure!(ids.insert(op.id), "duplicate Rooflang operation ID");
            }
            match (op.layer, op.kv_rank, &op.batch_indices) {
                (Some(layer), Some(rank), Some(indices)) => {
                    ensure!(
                        layer < self.layers && rank < self.tp as usize && !indices.is_empty(),
                        "invalid Qwen attention layer, KV rank or empty sub-batch"
                    );
                    for &index in indices {
                        ensure!(
                            index < batch_len && coverage.insert((layer, index, rank)),
                            "duplicate or out-of-range Qwen attention batch index"
                        );
                    }
                }
                (None, None, None) => {}
                _ => {
                    bail!("Qwen attention requires layer, group-local KV rank and batch_indices")
                }
            }
        }
        ensure!(
            coverage.len() == batch_len * self.layers as usize * self.tp as usize,
            "Qwen attention must cover every request and KV rank exactly once per layer"
        );
        for op in operations {
            ensure!(
                op.parents.iter().all(|parent| {
                    ids.as_ref()
                        .map_or(*parent < operations.len(), |ids| ids.contains(parent))
                        && *parent != op.id
                }),
                "invalid Qwen operation dependency"
            );
        }
        Ok(())
    }

    fn add_shared_iteration(
        &mut self,
        batch: &[RequestState],
        requests: &[crate::kv::BatchRequest],
        current: NodeID,
        graph: &mut Graph,
    ) -> Result<NodeID> {
        use crate::resources::{Allocation, Work};
        let _wall = scope(Section::GraphBuild);
        ensure!(
            !batch.is_empty() && batch.len() == requests.len(),
            "shared batch identities missing"
        );
        let mut request = json!({"op":"graph", "details":graph.tracing_enabled(), "batch":batch.iter().map(|r|json!({"q_len":r.q_len,"kv_len":r.kv_len,"lm_head_len":r.lm_head_len})).collect::<Vec<_>>()});
        if self.graph_model == GraphModel::Qwen3 {
            self.validate_qwen_batch(requests, graph)?;
            request["requests"] = json!(requests.iter().map(|r| r.request_id).collect::<Vec<_>>());
            request["compact_work"] = json!(!graph.tracing_enabled());
            request["operation_rows"] = json!(!graph.tracing_enabled());
        }
        let mut response: GraphResponse = self
            .worker
            .as_mut()
            .context("worker not initialized")?
            .request(request)?;
        let instance = self
            .instance_id
            .clone()
            .context("missing instance identity")?;
        let operations = response.take_operations(Some(&instance))?;
        if self.graph_model == GraphModel::Qwen3 {
            self.validate_qwen_operations(&operations, batch.len())?;
        }
        let prefill = self.stage.as_deref() == Some("p");
        let workspace = format!("workspace:{instance}:{}", self.round);
        let mut reservations = Vec::new();
        for r in requests {
            reservations.extend(graph.kv.reservations(r.request_id, prefill)?);
        }
        reservations.push(Allocation {
            id: workspace.clone(),
            bytes: response.workspace.context("missing Rooflang workspace")?,
        });
        let begin = graph.add_resource_node(Work { reserve:reservations, details:json!({"name":"RooflangRoundBegin","instance_id":instance,"round":self.round,"requests":requests,"batch":batch.iter().map(|r|json!({"q_len":r.q_len,"kv_len":r.kv_len,"lm_head_len":r.lm_head_len})).collect::<Vec<_>>()}), ..Default::default() });
        graph.add_edge(current, begin)?;
        let end = graph.add_resource_node(Work {
            release: vec![workspace],
            details: json!({"name":"RooflangRoundEnd","instance_id":instance,"round":self.round}),
            ..Default::default()
        });
        let mut ids = (!operations.iter().enumerate().all(|(id, op)| id == op.id))
            .then(|| std::collections::HashMap::with_capacity(operations.len()));
        let mut endpoints = Vec::with_capacity(operations.len());
        let mut dependencies = Vec::with_capacity(operations.len());
        for mut op in operations {
            if graph.tracing_enabled() {
                let mut work = op.work.into_work();
                work.details["instance_id"] = json!(instance);
                work.details["round"] = json!(self.round);
                work.details["operator_id"] = json!(op.id);
                op.work = crate::resources::SharedWork::new(work);
            }
            let work = graph.add_shared_resource_node(op.work);
            let mut entry = work;
            let mut exit = work;
            if let (Some(layer), Some(rank)) = (op.layer, op.kv_rank) {
                (entry, exit) = match op.batch_indices.as_deref() {
                    Some(indices) => attach_attention_kv(
                        indices.iter().map(|&i| (&requests[i], &batch[i])),
                        prefill,
                        layer,
                        rank,
                        work,
                        graph,
                    )?,
                    None => attach_attention_kv(
                        requests.iter().zip(batch),
                        prefill,
                        layer,
                        rank,
                        work,
                        graph,
                    )?,
                };
            }
            if let Some(ids) = &mut ids {
                ids.insert(op.id, endpoints.len());
            }
            endpoints.push((entry, exit));
            dependencies.push((op.id, op.parents));
            graph.add_edge(exit, end)?;
        }
        let endpoint = |id| endpoints[ids.as_ref().map_or(id, |ids| ids[&id])];
        for (id, parents) in dependencies {
            let entry = endpoint(id).0;
            if parents.is_empty() {
                graph.add_edge(begin, entry)?;
            }
            for parent in parents {
                graph.add_edge(endpoint(parent).1, entry)?;
            }
        }
        self.round += 1;
        Ok(end)
    }
}

// The iterator selects the real attention sub-batch; KV owns readiness and publication.
fn attach_attention_kv<'a>(
    requests: impl Iterator<Item = (&'a crate::kv::BatchRequest, &'a RequestState)>,
    prefill: bool,
    layer: u32,
    rank: usize,
    work: NodeID,
    graph: &mut Graph,
) -> Result<(NodeID, NodeID)> {
    use crate::kv::Event;
    let mut entry = work;
    let mut exit = work;
    for (r, b) in requests {
        let read = graph.add_kv_node(Event::Read {
            request: r.request_id,
            prefill,
            layer,
            rank,
            tokens: b.kv_len,
        });
        graph.add_edge(read, entry)?;
        entry = read;
        if !prefill {
            if graph.kv.controlled() {
                if graph.kv.ready_tokens(r.request_id, false, layer, rank) < b.kv_len {
                    let gate = match graph.kv.read_gate(r.request_id, layer, rank, b.kv_len) {
                        Some(gate) => gate,
                        None => {
                            let gate = graph.add_condition();
                            graph
                                .kv
                                .register_read_gate(r.request_id, layer, rank, b.kv_len, gate);
                            gate
                        }
                    };
                    graph.add_edge(gate, read)?;
                }
            } else {
                for dependency in graph.kv.read_dependencies(r.request_id, layer, rank) {
                    if graph.is_pending(dependency) {
                        graph.add_edge(dependency, read)?;
                    }
                }
            }
        }
        let produce = graph.add_kv_node(Event::Produce {
            request: r.request_id,
            prefill,
            layer,
            rank,
            start: b.kv_len,
            end: b.kv_len + b.q_len,
        });
        graph.add_edge(exit, produce)?;
        exit = produce;
    }
    Ok((entry, exit))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kv::{BatchRequest, Plan};
    use crate::resources::Work;

    #[test]
    fn typed_graph_response_preserves_numeric_demands_and_details() {
        let json = r#"{"operations":[{"id":0,"parents":[],"work":{
            "demands":[{"latency":5e-7,"resources":{"hbm":1.2345678901234567e-5,"link":0.00000000000000000001}}],
            "details":{"bytes":18446744073709551615,"label":"copy"}
        }}],"workspace":{"gpu":18446744073709551615},"wall_profile":{"sections":{"graph_build_other":{"calls":1,"seconds":0.001}}}}"#;
        let mut value: Value = serde_json::from_str(json).unwrap();
        let old: Vec<Operation> = serde_json::from_value(value["operations"].take()).unwrap();
        let mut response: GraphResponse = serde_json::from_str(json).unwrap();
        let timings = response.take_timing().unwrap().unwrap();
        assert_eq!(timings["graph_build_other"].calls, 1);
        assert!(response.error().is_none());
        let operations = response.take_operations(None).unwrap();
        assert_eq!(
            serde_json::to_value(&old[0].work).unwrap(),
            serde_json::to_value(&operations[0].work).unwrap()
        );
        assert_eq!(operations[0].id, old[0].id);
        assert_eq!(operations[0].parents, old[0].parents);
        assert_eq!(response.workspace.unwrap()["gpu"], u64::MAX);
    }

    #[test]
    fn compact_work_shares_immutable_templates_and_scopes_slots_once() {
        let response = json!({
            "operation_rows": [[[], 0, null, null, null], [[0], 0, null, null, null]],
            "work_templates": [{"slots": ["compute:0"]}]
        });
        let mut p: GraphResponse = serde_json::from_value(response.clone()).unwrap();
        let mut d: GraphResponse = serde_json::from_value(response).unwrap();
        let p = p.take_operations(Some("p0")).unwrap();
        let d = d.take_operations(Some("d0")).unwrap();
        assert!(std::ptr::eq(&*p[0].work, &*p[1].work));
        assert_eq!(p[0].work.slots[0], "p0:compute:0");
        assert_eq!(d[0].work.slots[0], "d0:compute:0");
        assert!(!std::ptr::eq(&*p[0].work, &*d[0].work));
        let template = scope_work(Work::default(), None);
        let parse = |fields: &str| {
            serde_json::from_str::<WireOperation>(&format!("{{\"id\":0,\"parents\":[],{fields}}}"))
                .unwrap()
        };
        assert!(parse("\"work_id\":1")
            .resolve(std::slice::from_ref(&template), None)
            .is_err());
        assert!(parse("\"work\":{},\"work_id\":0")
            .resolve(std::slice::from_ref(&template), None)
            .is_err());
        assert!(parse("\"work\":null").resolve(&[], None).is_err());
        assert!(parse("\"work\":{}").resolve(&[], None).is_ok());
    }

    #[test]
    fn operation_rows_preserve_ids_parents_kv_and_work_bits() {
        let templates = json!([{
            "slots": ["compute:0"],
            "demands": [{"latency": 5e-7, "resources": {"hbm": 1.2345678901234567e-5}}]
        }]);
        let mut rows: GraphResponse = serde_json::from_value(json!({
            "operation_rows": [[[], 0, null, null, null], [[0], 0, 7, 1, [2, 0]]],
            "work_templates": templates
        }))
        .unwrap();
        let mut objects: GraphResponse = serde_json::from_value(json!({
            "operations": [
                {"id":0,"parents":[],"work_id":0},
                {"id":1,"parents":[0],"work_id":0,"layer":7,"kv_rank":1,"batch_indices":[2,0]}
            ],
            "work_templates": templates
        }))
        .unwrap();
        let actual = rows.take_operations(None).unwrap();
        let expected = objects.take_operations(None).unwrap();
        for (actual, expected) in actual.iter().zip(&expected) {
            assert_eq!(actual.id, expected.id);
            assert_eq!(actual.parents, expected.parents);
            assert_eq!(actual.layer, expected.layer);
            assert_eq!(actual.kv_rank, expected.kv_rank);
            assert_eq!(actual.batch_indices, expected.batch_indices);
            assert_eq!(
                serde_json::to_string(&actual.work).unwrap(),
                serde_json::to_string(&expected.work).unwrap()
            );
        }
        assert_eq!(actual.len(), 2);
    }

    #[test]
    fn operation_rows_reject_ambiguous_formats_invalid_indices_and_malformed_rows() {
        for value in [
            json!({}),
            json!({"operations":[],"operation_rows":[]}),
            json!({"operation_rows":[[[],0,null,null,null]]}),
            json!({"operation_rows":[[[],1,null,null,null]],"work_templates":[{}]}),
        ] {
            let mut response: GraphResponse = serde_json::from_value(value).unwrap();
            assert!(response.take_operations(None).is_err());
        }
        for row in [
            json!([[], 0, null, null]),
            json!([[], 0, null, null, null, null]),
            json!([0, 0, null, null, null]),
            json!([[], "0", null, null, null]),
            json!([[], -1, null, null, null]),
            json!([[], 0, null, null, 0]),
        ] {
            assert!(serde_json::from_value::<GraphResponse>(json!({
                "operation_rows":[row],"work_templates":[{}]
            }))
            .is_err());
        }
    }

    fn qwen(ep: u32) -> RooflangRunner {
        serde_json::from_value(json!({
            "tp": 1, "ep": ep, "graph_model": "qwen3_235b_a22b",
            "attention_groups": (0..ep).map(|rank| vec![rank]).collect::<Vec<_>>(),
            "shared_execution": true, "stage": "p", "instance_id": "p0",
            "command": [], "hardware": {}
        }))
        .unwrap()
    }

    fn plan(device: &str, tokens: u32) -> Plan {
        Plan {
            source: vec![device.into()],
            target: vec![],
            routes: vec![],
            layers: 94,
            heads: 4,
            bytes_per_token_head_layer: 512,
            input_tokens: tokens,
            max_decode_tokens: 0,
            layer_handoff: true,
            prefill_done: false,
            copies: Default::default(),
        }
    }

    #[test]
    fn graph_selection_preserves_llama_defaults_and_rejects_unsupported_modes() {
        let mut llama: RooflangRunner = serde_json::from_value(json!({
            "tp": 1, "command": [], "hardware": {}
        }))
        .unwrap();
        assert_eq!(
            llama.parallel_strategy(),
            ParallelStrategy {
                tp: 1,
                pp: 1,
                ep: 1
            }
        );
        assert!(llama.graph_model == GraphModel::Llama);
        for tp in [1, 2, 4, 8] {
            llama.tp = tp;
            llama.validate_configuration().unwrap();
        }
        llama.ep = 2;
        assert!(llama.validate_configuration().is_err());
        for ep in [1, 2, 4, 8] {
            let runner = qwen(ep);
            runner.validate_configuration().unwrap();
            assert_eq!(runner.parallel_strategy().ep, ep);
        }
        for ep in [0, 3, 16] {
            assert!(qwen(ep).validate_configuration().is_err());
        }
        let mut runner = qwen(8);
        runner.tp = 2;
        assert!(runner.validate_configuration().is_err());
        runner.attention_groups = vec![vec![0, 1], vec![2, 3], vec![4, 5], vec![6, 7]];
        runner.validate_configuration().unwrap();
        runner.expert_tp = 2;
        assert!(runner.validate_configuration().is_err());
        runner.expert_tp = 1;
        runner.tp = 1;
        runner.shared_execution = false;
        assert!(runner.validate_configuration().is_err());
        assert!(serde_json::from_value::<RooflangRunner>(json!({
            "tp": 1, "graph_model": "qwen3_moe", "command": [], "hardware": {}
        }))
        .is_err());
    }

    #[test]
    fn qwen_attention_requires_exact_layer_and_request_coverage() {
        let mut runner = qwen(8);
        runner.layers = 2;
        let operations = || {
            (0..4)
                .map(|id| Operation {
                    id,
                    parents: vec![],
                    layer: Some((id / 2) as u32),
                    kv_rank: Some(0),
                    batch_indices: Some(if id % 2 == 0 { vec![0, 2] } else { vec![1] }),
                    work: crate::resources::SharedWork::new(Work::default()),
                })
                .collect::<Vec<_>>()
        };
        runner.validate_qwen_operations(&operations(), 3).unwrap();
        let mut invalid = operations();
        invalid.pop();
        assert!(runner.validate_qwen_operations(&invalid, 3).is_err());
        for indices in [None, Some(vec![]), Some(vec![0, 0]), Some(vec![0, 3])] {
            let mut invalid = operations();
            invalid[0].batch_indices = indices;
            assert!(runner.validate_qwen_operations(&invalid, 3).is_err());
        }
        let mut invalid = operations();
        invalid[0].kv_rank = Some(7);
        assert!(runner.validate_qwen_operations(&invalid, 3).is_err());
        let mut invalid = operations();
        invalid[0].layer = Some(2);
        assert!(runner.validate_qwen_operations(&invalid, 3).is_err());
        let mut invalid = operations();
        invalid[0].parents.push(100);
        assert!(runner.validate_qwen_operations(&invalid, 3).is_err());
        let mut invalid = operations();
        invalid[0].id = 1;
        assert!(runner.validate_qwen_operations(&invalid, 3).is_err());
        let mut invalid = operations();
        invalid[0].parents.push(0);
        assert!(runner.validate_qwen_operations(&invalid, 3).is_err());
        let mut sparse = operations();
        for op in &mut sparse {
            op.id += 10;
        }
        sparse[1].parents.push(10);
        runner.validate_qwen_operations(&sparse, 3).unwrap();
        sparse[1].parents.push(0);
        assert!(runner.validate_qwen_operations(&sparse, 3).is_err());
    }

    #[test]
    fn qwen_attention_tp_requires_every_kv_rank_per_request_and_layer() {
        let mut runner = qwen(8);
        runner.tp = 2;
        runner.layers = 2;
        let operations = || {
            (0..4)
                .map(|id| Operation {
                    id,
                    parents: vec![],
                    layer: Some((id / 2) as u32),
                    kv_rank: Some(id % 2),
                    batch_indices: Some(vec![0, 1]),
                    work: crate::resources::SharedWork::new(Work::default()),
                })
                .collect::<Vec<_>>()
        };
        runner.validate_qwen_operations(&operations(), 2).unwrap();
        let mut invalid = operations();
        invalid[1].kv_rank = Some(0);
        assert!(runner.validate_qwen_operations(&invalid, 2).is_err());
        let mut invalid = operations();
        invalid[1].batch_indices = Some(vec![0]);
        assert!(runner.validate_qwen_operations(&invalid, 2).is_err());
        let mut invalid = operations();
        invalid[0].kv_rank = Some(2);
        assert!(runner.validate_qwen_operations(&invalid, 2).is_err());
    }

    #[test]
    fn attention_tp2_reserves_two_kv_shards_without_halving_expert_weights() {
        let mut runner = qwen(8);
        runner.tp = 2;
        runner.attention_groups = vec![vec![0, 1], vec![2, 3], vec![4, 5], vec![6, 7]];
        runner.layers = 94;
        let mut model: Box<dyn Model> = toml::from_str("kind = 'qwen3_235b_a22b'").unwrap();
        runner.init_model(model.as_mut()).unwrap();
        runner.weight_bytes = model.weight_bytes_per_device();
        runner.resolved_hardware = json!({"devices": (0..8).map(|i| json!({
            "name": format!("g{i}"), "memory_capacity_gb": 144.0,
            "memory_bandwidth_gbs": 4800.0
        })).collect::<Vec<_>>()});
        let mut graph = Graph::new();
        runner
            .prepare_graph(&mut graph, Some(144_000_000_000))
            .unwrap();
        for memory in graph.resources.policy_memory() {
            assert_eq!(memory.allocated_bytes, 65_443_748_864);
        }
        let mut ownership = plan("g6", 5);
        ownership.source.push("g7".into());
        graph.kv.plans.insert(7, ownership);
        let requests = [BatchRequest {
            request_id: 7,
            input_tokens: 5,
            output_tokens: 1,
        }];
        runner.validate_qwen_batch(&requests, &graph).unwrap();
        let reservation = graph.kv.reservations(7, true).unwrap();
        assert_eq!(reservation[0].bytes.len(), 2);
        assert_eq!(reservation[0].bytes["g6"], 481_280);
        assert_eq!(reservation[0].bytes["g7"], 481_280);
        graph.kv.plans.get_mut(&7).unwrap().source.swap(0, 1);
        assert!(runner.validate_qwen_batch(&requests, &graph).is_err());
        graph.kv.plans.get_mut(&7).unwrap().source = vec!["g0".into(), "g1".into()];
        assert!(runner.validate_qwen_batch(&requests, &graph).is_err());
    }

    #[test]
    fn ep8_keeps_complete_kv_on_one_owner_and_allocates_per_rank_weights() {
        let mut runner = qwen(8);
        runner.layers = 94;
        let mut model: Box<dyn Model> = toml::from_str("kind = 'qwen3_235b_a22b'").unwrap();
        model.init(runner.parallel_strategy()).unwrap();
        runner.weight_bytes = model.weight_bytes_per_device();
        runner.resolved_hardware = json!({"devices": (0..8).map(|i| json!({
            "name": format!("g{i}"), "memory_capacity_gb": 144.0,
            "memory_bandwidth_gbs": 4800.0
        })).collect::<Vec<_>>()});
        let mut graph = Graph::new();
        runner
            .prepare_graph(&mut graph, Some(144_000_000_000))
            .unwrap();
        for memory in graph.resources.policy_memory() {
            assert_eq!(memory.allocated_bytes, 72_768_576_512);
        }
        assert_eq!(graph.resources.policy_memory().len(), 8);
        graph.kv.plans.insert(7, plan("g7", 5));
        let requests = [BatchRequest {
            request_id: 7,
            input_tokens: 5,
            output_tokens: 1,
        }];
        runner.validate_qwen_batch(&requests, &graph).unwrap();
        let reservation = graph.kv.reservations(7, true).unwrap();
        assert_eq!(reservation[0].bytes.len(), 1);
        assert_eq!(reservation[0].bytes["g7"], 5 * 192_512);
        graph.kv.plans.get_mut(&7).unwrap().source = vec!["g0".into()];
        assert!(runner.validate_qwen_batch(&requests, &graph).is_err());
        graph.kv.plans.get_mut(&7).unwrap().source = vec!["g7".into(), "g0".into()];
        assert!(runner.validate_qwen_batch(&requests, &graph).is_err());
        graph.kv.plans.get_mut(&7).unwrap().source = vec!["g7".into()];
        graph.kv.plans.get_mut(&7).unwrap().heads = 1;
        assert!(runner.validate_qwen_batch(&requests, &graph).is_err());
    }

    struct CheckSubset;
    impl Module for CheckSubset {}
    impl LogicalHandler for CheckSubset {
        fn handle(
            &mut self,
            payload: Box<dyn Any>,
            _: f64,
            current: NodeID,
            graph: &mut Graph,
        ) -> Result<f64> {
            let (selected, tokens, excluded) = *payload.downcast::<(u64, u32, u64)>().unwrap();
            assert_eq!(graph.kv.ready_tokens(selected, true, 0, 0), tokens);
            assert_eq!(graph.kv.ready_tokens(excluded, true, 0, 0), 0);
            graph.kv_event(crate::kv::Event::PrefillDone(selected), current)?;
            Ok(0.0)
        }
    }
    impl crate::System for CheckSubset {
        fn init(&mut self) -> Result<()> {
            Ok(())
        }
        fn instance_load(&self) -> crate::types::InstanceLoad {
            Default::default()
        }
        fn add_request_arrival(
            &mut self,
            _: crate::types::Request,
            _: &mut Graph,
            _: Option<NodeID>,
            _: Option<*mut dyn LogicalHandler>,
        ) -> Result<()> {
            unreachable!()
        }
        fn into_results(self: Box<Self>) -> Result<Vec<crate::types::RequestResult>> {
            Ok(vec![])
        }
    }

    #[test]
    fn native_attention_subsets_publish_only_their_own_kv_and_drain() {
        let requests = [
            BatchRequest {
                request_id: 7,
                input_tokens: 2,
                output_tokens: 1,
            },
            BatchRequest {
                request_id: 8,
                input_tokens: 3,
                output_tokens: 1,
            },
        ];
        let batch = [
            RequestState {
                q_len: 2,
                kv_len: 0,
                lm_head_len: 1,
            },
            RequestState {
                q_len: 3,
                kv_len: 0,
                lm_head_len: 1,
            },
        ];
        let mut graph = Graph::new();
        graph.kv.plans.insert(7, plan("g7", 2));
        graph.kv.plans.insert(8, plan("g0", 3));
        let mut system = Box::new(CheckSubset);
        let mut previous = None;
        for (index, excluded) in [(1, 7_u64), (0, 8)] {
            let work = graph.add_resource_node(Work::default());
            let (entry, exit) = attach_attention_kv(
                std::iter::once((&requests[index], &batch[index])),
                true,
                0,
                0,
                work,
                &mut graph,
            )
            .unwrap();
            if let Some(previous) = previous {
                graph.add_edge(previous, entry).unwrap();
            }
            let check = graph.add_logical_node(
                system.as_mut() as *mut dyn LogicalHandler,
                (requests[index].request_id, batch[index].q_len, excluded),
            );
            graph.add_edge(exit, check).unwrap();
            previous = Some(check);
        }
        crate::Engine::new(graph, system).run().unwrap();
    }
}
