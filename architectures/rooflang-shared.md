# Shared Rooflang execution

## Entry and ownership

Run `bash run_rooflang_shared.sh`, or add `--pd --execution shared` to
`scripts/inspect_rooflang.py`. `--kv-handoff layer|whole` selects early layer
copies or copies after P completion. Existing scripts/default inspection retain
the exclusive bridge. Llama 3.1 8B supports FP16, TP 1/2/4/8 and PP=EP=1.
`run_qwen3_235b_shared.sh` selects the separate Qwen3-235B-A22B preset with
Attention TP 1/2/4 dividing EP 1/2/4/8, expert TP=1 and PP=1.
Both use single-LLM P/D requests. Qwen defaults to Attention TP2/EP8 and two
complete replicas per P/D pool: 32 H200 GPUs total.

The worker's `graph` request returns dependencies, devices, stream ordering,
tensors and service demands. Rooflang owns cost formulas. A bounded per-worker
cache retains stream ordering and workspace for each query/logits-row shape.
Resident KV is separate; all service costs are repriced for every batch. A new
shape and selected audited rounds run the isolated reference; its timestamps
do not drive shared execution. `--reference-every-round` retains the diagnostic
reference path. Native resource nodes execute the actual graph. The scheduler
commits tokens after all batch operators finish. IPC consumes no simulated time.

The graph retains one physical resource domain and one boxed KV handler.
The engine owns time; the domain owns capacity/rates; the KV handler owns
identities/readiness/references; PD owns routing and the first-token boundary.
The scheduler owns token progress and batching, with explicit batch identities.

## Qwen expert parallelism

The fixed `qwen3_235b_a22b.toml` supplies both native memory metadata and Python
graph dimensions. Its 94 layers include 64 query heads, four KV heads, 128
experts and eight selections per token. BF16 storage uses two bytes and the
existing FP16-equivalent analytical hardware rate; there is no measured Qwen
profile or learned routing trace.

Each instance is a synchronized EP pool with one scheduler. Its EP GPUs form
EP/TP consecutive Attention TP groups; request ID modulo the group count assigns
each request's group for its lifetime. The shared scheduler's batch and token
limits apply to the complete instance. Instances are independent complete model
replicas, separate from the Attention groups inside each replica.
Expert ranges are contiguous across physical ranks. Deterministic seeded
uniform routing chooses eight distinct experts per token/layer. Expert work
depends on actual routed token counts, including remote requests and idle
Attention ranks. Dispatch/combine have explicit graph dependencies and use the
same fabric and endpoint HBM resources as P/D KV copies. EP groups may span
connected preset clusters; they need point-to-point routes, not a collective
aggregate-bandwidth scope. A round commits only after every participating rank
completes. Expert tensor splitting, independent asynchronous rank schedulers
and PP are not implemented.

Routing is keyed by seed, request identity, token position and layer, independent
of batching. Its fixed top-eight implementation preserves the original seeded
selection order and rejection draws; it does not reuse routes across tokens.

Each request's P and D KV plans contain their respective Attention TP group,
partitioning all four KV heads across its ranks. Logical KV remains 192512 bytes
per token; TP2 owns 96256 bytes per rank. P/D may use different TP and EP degrees.
Explicit groups drive the runner, native KV ownership and report topology;
legacy flat topology inputs normalize to this representation at load time.
Resident weights are allocated on every EP rank, even if no request attends there; unfit GPU
configurations fail. EP1/2 support does not imply the full model fits on H200.

Attention shards QKV heads and output-projection inputs; explicit gather,
reduction and broadcast restore the full hidden state. Each group leader alone
routes and dispatches tokens, then broadcasts the completed MoE residual.
Experts retain their full intermediate dimension regardless of Attention TP.
Embedding weights remain replicated with leader lookup/broadcast; LM-head
weights shard the vocabulary, gather logits to the leader and sample once.
Thus TP activation copies never multiply expert assignments. Temporary buffers,
reduction arithmetic and transfer HBM accesses count toward shared resources.

Qwen rebuilds route-dependent operations and workspace each round rather than
using Llama's shape cache. Bounded caches reuse exact Rooflang prices for the same
physical device, tensor shapes, FLOPs and read fractions, or the same copy
endpoints and byte count. They never reuse routing, batch dependencies or
contention results. Metrics-only responses can intern identical work descriptions
within one round; native lowering preserves every operator, dependency and KV
binding. Repeated expert shapes reuse their three work templates while retaining
three independent nodes per expert. Attention work is likewise reused within a
round; every layer keeps independent nodes, dependencies and KV bindings. Detailed
reports retain the expanded representation.
Metrics-only Qwen requests can also negotiate fixed operation rows: parents,
work-template ID, layer, KV rank and batch indices. Row position supplies the
operator ID. Native lowering preserves every node and validates dependency and
KV coverage; legacy object responses remain supported.
Native nodes share immutable work templates, with stream names scoped once per
instance/template. Each node retains independent progress, setup latency and
resource rates. Detailed report metadata remains specific to each operator.
Audits contain native plans and execution times;
`--reference-every-round` is rejected because this path has no independent
isolated Rooflang replay. Report checks establish consistency, not GPU accuracy.

Dispatch sends one activation plus 128 two-byte routing scores per token and
distinct destination rank: 8448 bytes. Each owner weights and reduces its expert
outputs before returning one 8192-byte partial per token. The source sums owner
partials. Local copies charge HBM reads/writes without fabric traffic. Top-k
pricing uses the existing softmax convention, log2(8) comparisons per expert,
and 15 operations to renormalize eight selected probabilities. These are
explicit analytical conventions. Workspace is a conservative, route-dependent
arena reused across synchronized layers, not a measured CUDA allocation peak.

## Resources

Compute and HBM use physical GPU IDs. Directed fabrics use endpoint, edge and
direction IDs reconstructed identically in every worker and KV route.
Instance streams have separate IDs. Each work item has concurrent components;
a component's progress is limited by every resource it uses. Normalized
full-speed fractions use progressive filling. Rates change on arrivals and
component completions. A slow copy consumes its actual HBM demand, rather than
blindly halving another kernel's HBM rate.

The resource domain indexes occupied streams and allocated bytes, rescanning
waiting work only when admission conditions change. Immutable demand coefficients
are validated and normalized at the template's first submission to its resource
domain, then reused by its nodes. Prepared templates belong to that domain;
allocation admission and release still occur independently for every node.
Rate calculation uses compact tables in the
original arithmetic order; independent resources use the same full-speed formula.
Unchanged resource state reuses its next event time; every admission and actual
time advance invalidates that boundary. The engine transfers each work description once while
retaining dependencies and observation flags. Every operator, KV event and
contention boundary still executes.

Compute and local memory service overlap. Network setup precedes bandwidth
service and consumes no bandwidth. KV copies use the same fabrics as collectives
and source-read/destination-write HBM. Same-GPU copies charge both HBM accesses.
Llama collective service retains Rooflang's aggregate-bandwidth/route-union
abstraction; multi-participant costs omit separate HBM traffic. Qwen Attention
TP uses explicit point-to-point transfers and reduction work, including HBM.
Neither path is a packet-level or measured NCCL algorithm model.

Default GPU allocations are disjoint. `--allow-gpu-sharing` permits explicit
overlap between `--prefill-gpu-ids` and `--decode-gpu-ids` in shared mode. A TP
group cannot repeat a device. Colocated instances share compute/HBM/fabric;
resident weights remain separate allocations. Shared devices require the same
physical memory limit. Automatic weight sharing is not assumed.

## KV and admission

D is chosen when a request is assigned to P. Before its first iteration, an
atomic reservation covers prompt KV on P and the maximum requested context on
D: input + output - 1 tokens. The sampled final token has no KV. Reservations
include all heads/layers and physical rank placement. Resident weights and
iteration workspace count against the same capacity. Past KV is removed from
the isolated workspace estimate; append staging buffers remain conservatively
included. `device_mem_utilization` limits capacity. Storage uses dense intervals,
not block rounding.

Temporarily insufficient capacity waits for releases. A batch that never fits
fails; FCFS batches are not automatically split to cure capacity deadlocks.
Whole-request reservations trade utilization for transfer-safe admission.
Adaptive reservation is a separate future policy.

A block identifies request, layer, token/head intervals, location and source
copy. Attention completion publishes newly written KV. Layer mode starts copies
immediately; whole mode waits for P completion. P retains a producer reference
until prefill finishes and a pin per copy until acknowledgement. Source storage
is released only after both conditions hold. Receivers become ready at actual
copy completion.

D begins only after P emits the first token. Each attention depends on the KV
copies needed by its layer/rank and validates readable token/head coverage.
Other layers may still be transferring. Decode KV grows after each attention;
all D blocks/reservations release after request completion. Ready intervals are
indexed by request, stage, layer, rank and head, and merged without bridging
gaps. Pending copy dependencies are removed on acknowledgement. Consecutive
decode appends share one block per layer/rank; trace `extend` events retain its
growth. Source blocks and copies retain distinct identities and pins. Zero/one-output
requests have no D allocation or copy.

Prefix caching is rejected and scheduler KV preemption is disabled in shared
mode. Cache sharing/eviction, swap, compression and dynamic migration are not
implemented. Optional causal policy decisions and autonomous search are described
in [serving policies](serving-policies.md).

## Evidence and observations

Inspection defaults to metrics and plots: `result.jsonl`, `metrics.json`,
`metrics.png`, `throughput.png`, configuration, workload, runtime receipt and small logs.
The standalone throughput plot places each run at (mean tokens/s/user, tokens/s/GPU),
normalizing GPU throughput by all allocated P/D devices. Both plots use request
results; GPU counts come from the saved configuration. It saves no historical events, audits,
CSV or HTML. The serving simulation and runtime correctness checks remain active.
Metrics cannot reconstruct an unsaved execution history.

`--reports` exports actual native events to `rounds.csv`, `operator-costs.csv`,
`kv-transfers.csv`, `kv-lifecycle.csv`, `memory-usage.csv` and `shared-check.json`.
Reserved KV capacity is distinct from live KV. Operator columns separate base
demands from elapsed time under contention. Selected detailed rounds include
plans and Perfetto traces in global microseconds; all round summaries remain.
An isolated reference duration is present only when that round was actually
pre-executed. CSV and selected Perfetto exports stream events, retaining only
unfinished graph state and live KV. The raw trace remains complete and can be large.

`--full-graphs` also enables reports. Its `execution-graph.html` joins the complete native
history with batches, scheduler snapshots, memory and KV events. It provides
module replay, per-instance/GPU operator and communication timelines, and
clickable dependency neighborhoods. Node filtering never samples the graph.
Playback groups simultaneous timestamps; `outer.html` retains exact sequence
replay and removed edges. Standalone rendering uses
`python scripts/draw_shared.py RUN_DIRECTORY` and does not rerun the simulator.
Full HTML reconstruction still loads the history; use it for bounded diagnostics.

The module-flow tab groups request arrival/completion, P/D scheduling, round
control, compute/TP, source/receiver KV and copies. It shares the global cursor;
concurrent work highlights concurrently, while instantaneous control events and
waiting states remain distinct. KV progress counts acknowledged bytes, not elapsed
time. Source retention, decode extension and release use native lifecycle events.
Control ownership follows same-instance dependency components anchored by round
metadata; ambiguous or missing ownership stays explicitly unassigned. Indexed
replay supports backward seeks without retaining full state snapshots per moment.
Module selection filters native nodes and opens their existing details. The module
event list explains every occurrence, including future events and scheduling checks
that produce no round. Counts and statuses share the global cursor; native dispatch
sequence orders simultaneous starts without implying extra dependencies.

Checks cover dependencies, stream exclusion, service lower bounds, capacity,
source lifetime, byte conservation and drain. Runtime enforces KV readiness.
Diagnostics cover isolated TP, slow-link overlap, reshards, capacity waits,
colocated instances and export on/off equivalence. These are consistency checks,
not measured GPU accuracy. Analytical costs and workspace bounds require
measurement before predicted speedups can be interpreted as hardware results.
