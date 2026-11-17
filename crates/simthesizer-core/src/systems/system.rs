use anyhow::Result;

use crate::engine::Graph;
use crate::module::{LogicalHandler, Module};
use crate::types::{InstanceLoad, NodeID, Request, RequestResult, SubRequest};

pub enum SystemEvent {
    RequestCompleted {
        request: Request,
        result: RequestResult,
    },
}

pub trait System: Module + LogicalHandler {
    fn shared_execution(&self) -> bool {
        false
    }
    fn prepare_graph(&mut self, _graph: &mut Graph) -> Result<()> {
        Ok(())
    }
    fn init(&mut self) -> Result<()>;

    /// Completed root results retained by this system, excluding forwarded child results.
    fn completed_requests(&self) -> Option<usize> {
        None
    }

    /// Scheduler work plus assigned initial LLM calls awaiting enqueue.
    fn instance_load(&self) -> InstanceLoad;

    fn add_request_arrival(
        &mut self,
        request: Request,
        graph: &mut Graph,
        parent: Option<NodeID>,
        sink: Option<*mut dyn LogicalHandler>,
    ) -> Result<()>;

    fn into_results(self: Box<Self>) -> Result<Vec<RequestResult>>;
}

pub(super) fn pending_llm_load(
    request: &Request,
    indices: impl IntoIterator<Item = usize>,
) -> InstanceLoad {
    indices
        .into_iter()
        .filter_map(|index| match &request.sub_requests[index] {
            SubRequest::Llm {
                input_tokens,
                known_tokens,
                kv_tokens,
                ..
            } => Some(InstanceLoad {
                waiting_requests: 1,
                waiting_tokens: u64::from(
                    known_tokens
                        .unwrap_or(*input_tokens)
                        .saturating_sub(kv_tokens.unwrap_or(0))
                        .max(1),
                ),
                ..Default::default()
            }),
            _ => None,
        })
        .sum()
}

crate::declare_registry! {
    registration = SystemRegistration,
    trait = crate::system::System,
    deserialize_fn = deserialize_system_impl,
    registry_name = "system",
}

crate::impl_registry_deserialize! {
    trait = crate::system::System,
    deserialize_fn = crate::system::deserialize_system_impl,
}

#[macro_export]
macro_rules! register_system {
    ($ty:ty, $kind:expr $(,)?) => {
        $crate::submit_registration!(
            registration = $crate::system::SystemRegistration,
            trait = $crate::system::System,
            type = $ty,
            kind = $kind,
            registry_name = "system",
        );
    };
}
