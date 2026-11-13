//! Python model definitions implementing the same layer/metadata contract as Rust models.
use std::io::Write;
use std::process::{Command, Stdio};

use anyhow::{ensure, Context, Result};
use serde::Deserialize;
use serde_json::json;

use crate::models::Model;
use crate::module::Module;
use crate::types::{Layer, ParallelStrategy};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PythonModel {
    command: Vec<String>,
    #[serde(skip)]
    definition: Definition,
}

#[derive(Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct Definition {
    version: u32,
    pre: Vec<Layer>,
    block: Vec<Layer>,
    post: Vec<Layer>,
    layers: u32,
    weight_bytes_per_device: u64,
    kv_bytes_per_token_per_device: u64,
}

crate::register_model!(PythonModel, "python");
impl Module for PythonModel {}

impl Model for PythonModel {
    fn init(&mut self, parallel: ParallelStrategy) -> Result<()> {
        ensure!(!self.command.is_empty(), "Python model command is empty");
        let mut child = Command::new(&self.command[0])
            .args(&self.command[1..])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .context("starting Python model definition")?;
        let request = json!({"version": 1, "parallel": {
            "tp": parallel.tp, "pp": parallel.pp, "ep": parallel.ep
        }});
        let write_result = (|| -> Result<()> {
            let mut input = child.stdin.take().context("Python model stdin")?;
            serde_json::to_writer(&mut input, &request)?;
            input.flush()?;
            Ok(())
        })();
        if let Err(error) = write_result {
            let _ = child.kill();
            let _ = child.wait();
            return Err(error);
        }
        let output = child.wait_with_output().context("reading Python model")?;
        ensure!(
            output.status.success(),
            "Python model definition failed: {}",
            output.status
        );
        let definition: Definition =
            serde_json::from_slice(&output.stdout).context("invalid Python model definition")?;
        ensure!(definition.version == 1, "unsupported Python model protocol");
        ensure!(
            definition.layers > 0 && !definition.block.is_empty(),
            "empty Python model"
        );
        ensure!(
            definition.weight_bytes_per_device > 0,
            "Python model weights must be positive"
        );
        ensure!(
            definition.kv_bytes_per_token_per_device > 0,
            "Python model KV bytes must be positive"
        );
        self.definition = definition;
        Ok(())
    }

    fn weight_bytes_per_device(&self) -> u64 {
        self.definition.weight_bytes_per_device
    }
    fn kv_bytes_per_token_per_device(&self) -> u64 {
        self.definition.kv_bytes_per_token_per_device
    }
    fn pre_block_layers(&self) -> &[Layer] {
        &self.definition.pre
    }
    fn block_layers(&self) -> &[Layer] {
        &self.definition.block
    }
    fn post_block_layers(&self) -> &[Layer] {
        &self.definition.post
    }
    fn num_blocks(&self) -> u32 {
        self.definition.layers
    }
}
