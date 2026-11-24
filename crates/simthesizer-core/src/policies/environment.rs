/// Borrowed causal state. Constructing this view performs no scans or sampling.
/// Queries allocate summaries only when a candidate explicitly calls them.
/// Safe queries remain available from the candidate's actual sibling position:
///
/// ```
/// # pub use simthesizer_core::{kv, resources};
/// # mod policies {
/// #     mod environment {
/// #         include!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/policies/environment.rs"));
/// #     }
/// #     pub use environment::EnvironmentObservation;
/// #     mod candidate {
/// pub fn inspect(view: super::EnvironmentObservation<'_>) {
///     let _ = view.pressure();
///     let _ = view.memory();
///     let _ = view.registered_reads();
/// }
/// #     }
/// # }
/// # fn main() {}
/// ```
///
/// The backing owners remain private even to a sibling candidate module:
///
/// ```compile_fail,E0616
/// # pub use simthesizer_core::{kv, resources};
/// # mod policies {
/// #     mod environment {
/// #         include!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/policies/environment.rs"));
/// #     }
/// #     pub use environment::EnvironmentObservation;
/// #     mod candidate {
/// fn peek(view: super::EnvironmentObservation<'_>) {
///     let _ = &view.kv.plans;
///     let _ = view.resources.snapshot();
/// }
/// #     }
/// # }
/// # fn main() {}
/// ```
#[derive(Clone, Copy)]
pub struct EnvironmentObservation<'a> {
    resources: &'a crate::resources::Resources,
    kv: &'a crate::kv::KvRuntime,
}

impl<'a> EnvironmentObservation<'a> {
    pub(crate) fn new(
        resources: &'a crate::resources::Resources,
        kv: &'a crate::kv::KvRuntime,
    ) -> Self {
        Self { resources, kv }
    }

    /// Remaining full-capacity service seconds, not predicted latency/utilization.
    pub fn pressure(&self) -> Vec<crate::resources::ResourcePressure> {
        self.resources.policy_pressure()
    }

    /// Allocated capacity includes the existing known_trace KV reservations.
    pub fn memory(&self) -> Vec<crate::resources::MemoryUsage> {
        self.resources.policy_memory()
    }

    /// Unsatisfied reads registered for expanded batches, not current stalls.
    pub fn registered_reads(&self) -> Vec<crate::kv::KvReadRequirement> {
        self.kv.policy_reads()
    }

    pub(super) fn registered_read_sample(
        &self,
        limit: usize,
    ) -> (usize, Vec<crate::kv::KvReadRequirement>) {
        self.kv.policy_read_sample(limit)
    }

    pub fn ready_tokens(&self, request: u64, layer: u32, rank: usize) -> u32 {
        self.kv.ready_tokens(request, false, layer, rank)
    }
}
