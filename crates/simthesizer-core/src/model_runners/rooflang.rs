//! Exclusive-resource, whole-iteration Rooflang bridge (JSONL protocol v1).
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
    tp: u32,
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
        format!(
            "RooflangIteration {{ hardware: {}, duration_s: {:?} }}",
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
    fn parallel_strategy(&self) -> ParallelStrategy {
        ParallelStrategy {
            tp: self.tp,
            pp: 1,
            ep: 1,
        }
    }
    fn init(&mut self, model: &dyn Model) -> Result<()> {
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
        let response = worker.request(json!({"op":"init", "version":1,"tp":self.tp,"hardware":self.hardware,
            "memory":{"weight_bytes_per_device":model.weight_bytes_per_device(),"kv_bytes_per_token_per_device":model.kv_bytes_per_token_per_device()},
            "model":{"pre":model.pre_block_layers(),"block":model.block_layers(),"post":model.post_block_layers(),"layers":model.num_blocks()}}))?;
        ensure!(
            response["ready"] == true,
            "Rooflang worker did not initialize"
        );
        let hardware = &response["hardware"];
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
