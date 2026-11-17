use anyhow::Result;

use crate::module::Module;
use crate::types::{
    InstanceLoad, ParallelStrategy, ReadySubRequest, RequestState, SubRequestCompletion,
};

pub trait Scheduler: Module {
    fn use_shared_memory(&mut self) -> Result<()> {
        anyhow::bail!("scheduler does not support shared memory admission")
    }
    fn memory_limit(&self) -> Option<u64> {
        None
    }
    fn batch_requests(&self) -> Vec<crate::kv::BatchRequest> {
        Vec::new()
    }
    fn init(
        &mut self,
        model_weight_bytes_per_device: u64,
        kv_bytes_per_request_per_device: u64,
        kv_bytes_per_token_per_device: u64,
        parallel: ParallelStrategy,
    ) -> Result<()> {
        let _ = (
            model_weight_bytes_per_device,
            kv_bytes_per_request_per_device,
            kv_bytes_per_token_per_device,
            parallel,
        );
        Ok(())
    }

    fn schedule(&mut self) -> Result<Vec<RequestState>>;

    fn enqueue_sub_request(&mut self, sub_request: ReadySubRequest);

    fn done_iteration(&mut self, now: f64) -> Vec<SubRequestCompletion>;

    fn instance_load(&self) -> InstanceLoad {
        InstanceLoad::default()
    }
}

crate::declare_registry! {
    registration = SchedulerRegistration,
    trait = crate::scheduler::Scheduler,
    deserialize_fn = deserialize_scheduler_impl,
    registry_name = "scheduler",
}

crate::impl_registry_deserialize! {
    trait = crate::scheduler::Scheduler,
    deserialize_fn = crate::scheduler::deserialize_scheduler_impl,
}

#[macro_export]
macro_rules! register_scheduler {
    ($ty:ty, $kind:expr $(,)?) => {
        $crate::submit_registration!(
            registration = $crate::scheduler::SchedulerRegistration,
            trait = $crate::scheduler::Scheduler,
            type = $ty,
            kind = $kind,
            registry_name = "scheduler",
        );
    };
}
