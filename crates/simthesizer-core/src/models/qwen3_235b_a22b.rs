//! Fixed checkpoint parameters; execution remains owned by Qwen3MoeModel.
use serde::Deserialize;
use serde_json::Value;

use crate::model::{Model, ModelRegistration};
use crate::models::qwen3_moe::Qwen3MoeModel;
use crate::registry::{strip_kind, RegistryError};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct FixedConfig {}

fn deserialize(value: Value) -> Result<Box<dyn Model>, RegistryError> {
    serde_json::from_value::<FixedConfig>(strip_kind(value))
        .map_err(|error| RegistryError::deserialize("qwen3_235b_a22b", error))?;
    let model: Qwen3MoeModel = toml::from_str(include_str!("qwen3_235b_a22b.toml"))
        .map_err(|error| RegistryError::deserialize("qwen3_235b_a22b", error))?;
    Ok(Box::new(model))
}

inventory::submit! {
    ModelRegistration {
        kind: "qwen3_235b_a22b",
        deserialize,
    }
}
