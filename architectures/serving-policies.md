# Serving policies

Shared Rooflang P/D optionally accepts one `system.policy`. The P/D parent owns
the runtime and shares it with its instances and KV domain. Without this setting,
the existing scheduling and eager KV paths remain active.

The runtime exposes three deterministic Rust decisions: route an arrived request,
select an explicit batch, and submit existing KV intents. Policies receive current
observations and bounded trial-local memory, without mutable graph/resource state.
The executor validates identities, budgets, token progress and future wake times.
Batch completion commits by stable completion-node ID, including reordered batches.

Candidate observations include a lazy read-only environment view. Resource and
KV owners derive remaining service demand, allocated capacity and registered
unmet reads only on explicit queries. Service demand is not utilization; reads
of an expanded batch are not necessarily current stalls. Request plans and mutable
owners stay private; allocated capacity retains the `known_trace` reservation
assumption. Optional decision samples are bounded and disabled
by default; the reference path remains unchanged. See the
[search guide](../agents/find_pareto_frontier/SEARCH.md) for seeds and diagnostics.

Optional offline tools render composable scheduling candidates and their controls.
The search planner derives case coverage and promotion suggestions from checked
receipts and complete native results. It neither runs trials nor changes the
evaluator's budget, metrics or verification boundary. Generated recipes are
starting hypotheses; arbitrary legal candidate programs remain supported.

Waiting uses registered logical events. A full KV window waits for ACK rather than
polling. Pending transfers pin their source until ACK; sending does not pin twice.
Decode reads wait on held conditions indexed by request/layer/rank/token coverage.
ACK merges ready intervals and releases only satisfied conditions. Unreleased
conditions participate in the engine's drain check. Resource demands still own all
compute, HBM and network costs.

Reference mode retains FCFS and eager transfer behavior. Candidate observation and
validation occur only at decision events; no additional Python or model invocation
is introduced. Aggregate callback time is wall-clock evidence, not simulated GPU
latency; it excludes observation construction and action validation. The existing
`known_trace` KV reservation and zero simulated scheduling cost remain assumptions.

The frozen evaluator owns workloads, hardware, SLO metrics, trial identities and
timeouts. It runs one simulation per case, preserves results and metric plots, and
reuses only matching completed evidence. Pareto groups compare identical cases.
Independent host replay rebuilds candidates before marking covered cases verified;
it establishes reproducibility, not hardware accuracy or unseen-workload quality.

See [the experiment contract](../agents/find_pareto_frontier/DESIGN.md) and
[commands](../agents/find_pareto_frontier/LAUNCHER.md).
