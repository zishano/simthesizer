# Serving policies

Shared Rooflang P/D optionally accepts one `system.policy`. The P/D parent owns
the runtime and shares it with its instances and KV domain. Without this setting,
the existing scheduling and eager KV paths remain active.

The runtime exposes three deterministic Rust decisions: route an arrived request,
select an explicit batch, and submit existing KV intents. Policies receive current
observations and bounded trial-local memory, without mutable graph/resource state.
The executor validates identities, budgets, token progress and future wake times.
Batch completion commits by stable completion-node ID, including reordered batches.

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
