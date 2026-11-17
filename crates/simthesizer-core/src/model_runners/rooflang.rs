//! Rooflang whole-iteration and shared operator bridges (JSONL protocol v1).
use crate::types::{NodeID, ParallelStrategy, RequestState};
use crate::{Graph, LogicalHandler, Model, ModelRunner, Module};
use anyhow::{bail, ensure, Context, Result};
use serde::Deserialize;
use serde_json::{json, Value};
use std::any::Any;
use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};

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
impl Drop for Worker {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}
impl Worker {
    fn request(&mut self, value: Value) -> Result<Value> {
        serde_json::to_writer(&mut self.input, &value)?;
        self.input.write_all(b"\n")?;
        self.input.flush()?;
        let mut line = String::new();
        ensure!(
            self.output.read_line(&mut line)? > 0,
            "Rooflang worker exited without a response"
        );
        let response: Value = serde_json::from_str(&line).context("invalid Rooflang response")?;
        if let Some(error) = response.get("error") {
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
            ep: 1,
        }
    }
    fn init(&mut self, model: &dyn Model) -> Result<()> {
        if self.shared_execution {
            ensure!(
                matches!(self.stage.as_deref(), Some("p" | "d")),
                "shared Rooflang requires stage p or d"
            );
        }
        self.weight_bytes = model.weight_bytes_per_device();
        ensure!(
            self.tp > 0 && self.tp <= 8 && 8 % self.tp == 0,
            "Rooflang Llama supports TP 1, 2, 4, 8"
        );
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
        let response = worker.request(json!({"op":"init", "version":1,"tp":self.tp,"hardware":self.hardware,"instance_id":self.instance_id,"shared_execution":self.shared_execution,
            "memory":{"weight_bytes_per_device":model.weight_bytes_per_device(),"kv_bytes_per_token_per_device":model.kv_bytes_per_token_per_device()},
            "model":{"pre":model.pre_block_layers(),"block":model.block_layers(),"post":model.post_block_layers(),"layers":model.num_blocks()}}))?;
        ensure!(
            response["ready"] == true,
            "Rooflang worker did not initialize"
        );
        let hardware = &response["hardware"];
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
        ensure!(!batch.is_empty(), "empty Rooflang batch");
        let response = self.worker.as_mut().context("Rooflang runner is not initialized")?.request(json!({"op":"iteration", "batch":batch.iter().map(|r|json!({"q_len":r.q_len,"kv_len":r.kv_len,"lm_head_len":r.lm_head_len})).collect::<Vec<_>>()}))?;
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
    work: crate::resources::Work,
}

impl RooflangRunner {
    fn add_shared_iteration(
        &mut self,
        batch: &[RequestState],
        requests: &[crate::kv::BatchRequest],
        current: NodeID,
        graph: &mut Graph,
    ) -> Result<NodeID> {
        use crate::kv::Event;
        use crate::resources::{Allocation, Work};
        ensure!(
            !batch.is_empty() && batch.len() == requests.len(),
            "shared batch identities missing"
        );
        let mut response = self.worker.as_mut().context("worker not initialized")?.request(json!({"op":"graph", "details":graph.tracing_enabled(), "batch":batch.iter().map(|r|json!({"q_len":r.q_len,"kv_len":r.kv_len,"lm_head_len":r.lm_head_len})).collect::<Vec<_>>()}))?;
        let operations: Vec<Operation> = serde_json::from_value(response["operations"].take())?;
        let prefill = self.stage.as_deref() == Some("p");
        let instance = self
            .instance_id
            .clone()
            .context("missing instance identity")?;
        let workspace = format!("workspace:{instance}:{}", self.round);
        let mut reservations = Vec::new();
        for r in requests {
            reservations.extend(graph.kv.reservations(r.request_id, prefill)?);
        }
        reservations.push(Allocation {
            id: workspace.clone(),
            bytes: serde_json::from_value(response["workspace"].take())?,
        });
        let begin = graph.add_resource_node(Work { reserve:reservations, details:json!({"name":"RooflangRoundBegin","instance_id":instance,"round":self.round,"requests":requests,"batch":batch.iter().map(|r|json!({"q_len":r.q_len,"kv_len":r.kv_len,"lm_head_len":r.lm_head_len})).collect::<Vec<_>>()}), ..Default::default() });
        graph.add_edge(current, begin)?;
        let end = graph.add_resource_node(Work {
            release: vec![workspace],
            details: json!({"name":"RooflangRoundEnd","instance_id":instance,"round":self.round}),
            ..Default::default()
        });
        let mut entries = std::collections::BTreeMap::new();
        let mut exits = std::collections::BTreeMap::new();
        let mut dependencies = Vec::new();
        for mut op in operations {
            for slot in &mut op.work.slots {
                *slot = format!("{instance}:{slot}");
            }
            if graph.tracing_enabled() {
                op.work.details["instance_id"] = json!(instance);
                op.work.details["round"] = json!(self.round);
                op.work.details["operator_id"] = json!(op.id);
            }
            let work = graph.add_resource_node(op.work);
            let mut entry = work;
            let mut exit = work;
            if let (Some(layer), Some(rank)) = (op.layer, op.kv_rank) {
                // Chain per-request checks and publications around a batched attention.
                for (r, b) in requests.iter().zip(batch) {
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
                        for dependency in graph.kv.read_dependencies(r.request_id, layer, rank) {
                            if graph.is_pending(dependency) {
                                graph.add_edge(dependency, read)?;
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
            }
            entries.insert(op.id, entry);
            exits.insert(op.id, exit);
            dependencies.push((op.id, op.parents));
            graph.add_edge(exit, end)?;
        }
        for (id, parents) in dependencies {
            if parents.is_empty() {
                graph.add_edge(begin, entries[&id])?;
            }
            for parent in parents {
                graph.add_edge(exits[&parent], entries[&id])?;
            }
        }
        self.round += 1;
        Ok(end)
    }
}
