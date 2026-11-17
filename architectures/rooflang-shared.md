# Shared Rooflang execution

## Entry and ownership

Run `bash run_rooflang_shared.sh`, or add `--pd --execution shared` to
`scripts/inspect_rooflang.py`. `--kv-handoff layer|whole` selects early layer
copies or copies after P completion. Existing scripts/default inspection retain
the exclusive bridge. Both modes support FP16 Llama 3.1 8B, TP 1/2/4/8,
PP=EP=1, and single-LLM P/D requests.

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

## Resources

Compute and HBM use physical GPU IDs. Directed fabrics use endpoint, edge and
direction IDs reconstructed identically in every worker and KV route.
Instance streams have separate IDs. Each work item has concurrent components;
a component's progress is limited by every resource it uses. Normalized
full-speed fractions use progressive filling. Rates change on arrivals and
component completions. A slow copy consumes its actual HBM demand, rather than
blindly halving another kernel's HBM rate.

Compute and local memory service overlap. Network setup precedes bandwidth
service and consumes no bandwidth. KV copies use the same fabrics as collectives
and source-read/destination-write HBM. Same-GPU copies charge both HBM accesses.
Collective service retains Rooflang's aggregate-bandwidth/route-union abstraction;
it is not a packet or NCCL algorithm model. Multi-participant collective costs
still omit separate HBM traffic.

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
implemented. No agent policy or search loop is included.

## Evidence and observations

Inspection defaults to metrics and their plot: `result.jsonl`, `metrics.json`,
`metrics.png`, configuration, workload, runtime receipt and small logs. The plot
uses request results only. It saves no historical events, audits,
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
Module selection filters native nodes and opens their existing details.

Checks cover dependencies, stream exclusion, service lower bounds, capacity,
source lifetime, byte conservation and drain. Runtime enforces KV readiness.
Diagnostics cover isolated TP, slow-link overlap, reshards, capacity waits,
colocated instances and export on/off equivalence. These are consistency checks,
not measured GPU accuracy. Analytical costs and workspace bounds require
measurement before predicted speedups can be interpreted as hardware results.
