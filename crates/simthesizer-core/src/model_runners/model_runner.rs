use anyhow::Result;

use crate::engine::Graph;
use crate::model::Model;
use crate::module::{LogicalHandler, Module};
use crate::types::{NodeID, ParallelStrategy, RequestState};

pub trait ModelRunner: Module + LogicalHandler {
    fn shared_execution(&self) -> bool {
        false
    }
    fn prepare_graph(&mut self, _graph: &mut Graph, _memory_limit: Option<u64>) -> Result<()> {
        Ok(())
    }
    fn add_scheduled_iteration(
        &mut self,
        model: &dyn Model,
        batch: &[RequestState],
        _requests: &[crate::kv::BatchRequest],
        current: NodeID,
        graph: &mut Graph,
    ) -> Result<NodeID> {
        self.add_iteration(model, batch, current, graph)
    }
    fn parallel_strategy(&self) -> ParallelStrategy;

    fn init(&mut self, _model: &dyn Model) -> Result<()> {
        Ok(())
    }

    fn add_iteration(
        &mut self,
        model: &dyn Model,
        batch: &[RequestState],
        current: NodeID,
        graph: &mut Graph,
    ) -> Result<NodeID>;

    fn add_multi_iteration(
        &mut self,
        model: &dyn Model,
        instance_batches: &[Vec<RequestState>],
        current: NodeID,
        graph: &mut Graph,
    ) -> Result<NodeID> {
        let batch = instance_batches
            .iter()
            .flat_map(|instance_batch| instance_batch.iter().copied())
            .collect::<Vec<_>>();
        self.add_iteration(model, &batch, current, graph)
    }
}

crate::declare_registry! {
    registration = ModelRunnerRegistration,
    trait = crate::model_runner::ModelRunner,
    deserialize_fn = deserialize_model_runner_impl,
    registry_name = "model_runner",
}

crate::impl_registry_deserialize! {
    trait = crate::model_runner::ModelRunner,
    deserialize_fn = crate::model_runner::deserialize_model_runner_impl,
}

#[macro_export]
macro_rules! register_model_runner {
    ($ty:ty, $kind:expr $(,)?) => {
        $crate::submit_registration!(
            registration = $crate::model_runner::ModelRunnerRegistration,
            trait = $crate::model_runner::ModelRunner,
            type = $ty,
            kind = $kind,
            registry_name = "model_runner",
        );
    };
}
