use anyhow::Result;

use crate::module::Module;
use crate::types::{Layer, ParallelStrategy};

pub trait Model: Module {
    fn init(&mut self, parallel: ParallelStrategy) -> Result<()>;
    fn init_with_expert_tp(&mut self, _parallel: ParallelStrategy, _expert_tp: u32) -> Result<()> {
        anyhow::bail!("model does not support independent expert tensor parallelism")
    }
    fn weight_bytes_per_device(&self) -> u64;
    fn kv_bytes_per_request_per_device(&self) -> u64 {
        0
    }
    fn kv_bytes_per_token_per_device(&self) -> u64;
    fn scheduler_model_weight_bytes(&self) -> u64 {
        self.weight_bytes_per_device()
    }
    fn scheduler_kv_bytes_per_request(&self) -> u64 {
        0
    }
    fn scheduler_kv_bytes_per_token(&self) -> u64 {
        self.kv_bytes_per_token_per_device()
    }
    fn pre_block_layers(&self) -> &[Layer];
    fn block_layers(&self) -> &[Layer];
    fn post_block_layers(&self) -> &[Layer];
    fn num_blocks(&self) -> u32;
}

crate::declare_registry! {
    registration = ModelRegistration,
    trait = crate::model::Model,
    deserialize_fn = deserialize_model_impl,
    registry_name = "model",
}

crate::impl_registry_deserialize! {
    trait = crate::model::Model,
    deserialize_fn = crate::model::deserialize_model_impl,
}

#[macro_export]
macro_rules! register_model {
    ($ty:ty, $kind:expr $(,)?) => {
        $crate::submit_registration!(
            registration = $crate::model::ModelRegistration,
            trait = $crate::model::Model,
            type = $ty,
            kind = $kind,
            registry_name = "model",
        );
    };
}
