//! Candidate-owned functions. Keep persistent heuristic state in `memory` only.
//! Observations contain arrived work; cost, dependencies, and progress stay in core.
use super::{
    BatchAction, BatchObservation, KvAction, KvObservation, PolicyContext, RouteAction,
    RouteObservation,
};

pub fn route(
    observation: &RouteObservation<'_>,
    context: &PolicyContext<'_>,
    memory: &mut [u8],
) -> RouteAction {
    super::reference_route(observation, context, memory)
}

pub fn batch(
    observation: &BatchObservation<'_>,
    context: &PolicyContext<'_>,
    memory: &mut [u8],
) -> BatchAction {
    super::reference_batch(observation, context, memory)
}

pub fn kv(
    observation: &KvObservation<'_>,
    context: &PolicyContext<'_>,
    memory: &mut [u8],
) -> KvAction {
    super::reference_kv(observation, context, memory)
}
