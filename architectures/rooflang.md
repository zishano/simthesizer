# Rooflang integration

Two execution modes share the validated Llama graph and hardware presets.
`exclusive` retains the whole-iteration bridge described below. `shared`
lowers operators into native events with cross-instance resources and persistent
KV; see [shared execution](rooflang-shared.md). Agent search is separate.

## Source and model entry points

Required Rooflang sources are copied into `scripts/rooflang/`; see its
`UPSTREAM.md` for revision and scope, and `LICENSE.txt` for the MIT license.
The graph, hardware, placement, kernels and simulator are local source files,
not an external checkout, installed Rooflang package or profiling table.
NetworkX is a regular locked project dependency. Upstream files are preserved
unchanged and excluded from local Ruff rewrites.

Both model entry points implement the existing Rust `Model` contract:

- `system.model.kind = "llama3_1"`: the existing Rust Llama definition.
- `system.model.kind = "python"`: run a configured command once at model init.
  The supplied `scripts/rooflang_model.py --module rooflang_bridge.llama`
  calls the Python module's `describe(parallel)` function. The response contains
  version 1, pre/block/post Layer lists, layer count, per-device weight bytes
  and KV bytes per token. Rust deserializes it into the same Layer types used
  by native models, so the scheduler receives the Python definition's metadata.

Commands run without a shell and must emit only JSON on stdout. The Python
model protocol currently describes token-KV models; fixed per-request KV and
custom scheduler-memory overrides are not exposed. Nonzero exit status,
malformed responses, unsupported versions and empty definitions fail at init.
The Rooflang backend checks both layer structure and memory metadata against
its supported Llama graph. It does not silently accept arbitrary model graphs.
Python and Rust definitions are tested for identical plans, metadata, per-round
costs and token timings at TP 1/2/4/8. The Python module also defines the Rooflang
graph; the Rust path is lowered to that matching graph after validation.

## Exclusive-mode contract and mapping

Supported scope: FP16 Llama 3.1 8B, each serving instance with exclusive devices
for computation. Scheduler admission, token commits, preemption and KV ownership
remain in Simthesizer. `RooflangRunner` sends the exact per-request query, cached
KV and logits lengths to a persistent Python worker. The worker validates the
32-layer plan, builds and validates one complete Rooflang graph template,
binds each batch to it, runs a fresh DES, and returns microseconds. The runner converts once to seconds and emits one logical
iteration node before the existing batch-completion callback. Wall-clock IPC
and Python runtime do not enter simulated time.

The CLI permits a single-instance root or a P/D root containing independent
single-instance Rooflang leaves. Other nested/multi-instance topologies remain
rejected. Topology-backed P/D leaves reference disjoint device IDs in a shared
hardware specification. Workers reconstruct that specification independently;
there is no shared persistent Rooflang runtime. Leaves may overlap in engine
time, while rounds on one leaf remain serialized. Legacy manual-link scenarios
retain independent virtual allocations even when preset names coincide.

## Native P/D composition (exclusive mode)

P/D uses the existing `PdDisaggregationSystem`: round-robin P and D pools,
per-leaf schedulers/KV accounting, a native network node gating decode release,
and merged token latencies. It supports single-LLM requests only. Each leaf has
a unique nonempty runner `instance_id`, its own Python worker and audit directory.
The identity is included in outer iteration labels and the worker init record;
analysis selects that instance before joining round summaries or checking time.
Old single-instance audits without an identity remain readable.

Parent and children use matching model definitions. In topology mode, the parent
initializes the model at TP=1 to obtain full KV bytes; each leaf independently
uses TP=1/2/4/8. Contiguous KV-head intervals determine rank-to-rank slices.
One native network node carries each nonempty interval intersection, and a
barrier releases D only after every slice arrives. Total bytes equal input
tokens × full-model KV bytes per token, regardless of P/D TP. The newly sampled
token has no KV until D processes it. First decode ITL includes transfer and
D queueing. P private KV is still released at child completion; sender staging
residency and receiver reservations are not modeled.

The inspection composer owns deployment normalization: per-side model, instance
count, TP, batch/token limits, and ordered GPU IDs. Pool GPU count must equal
instances × TP. PP/EP values other than 1 fail explicitly. Device capacity comes
from the allocated GPU's local memory. One TP group must stay within one preset
cluster. Default same-model deployment uses a shared upstream preset: RTX
Ethernet scope rounded up to eight GPUs, or H200/B300 node count rounded up to
eight GPUs per node. Custom `--hardware-topology JSON` specifies `clusters`
(id, gpu_model, optional n_nodes/RTX eth_scope) and `links` (src/dst component
names, bandwidth_gbs, latency_us). Bridges are symmetric full-duplex fabrics.
Mixed models require explicit connectivity; missing paths fail before execution.
Optional `--prefill-gpu-ids` and `--decode-gpu-ids` override automatic allocation.

The composer exports directed physical links and fixed Rooflang routes to the
native flow simulator. Link indices retain shared resource identity across
ranks and instances; pinning routes avoids the native widest-path policy choosing
a different path. KV traffic shares these links. Paths containing half-duplex
fabrics are currently rejected. Worker-internal TP/HBM operations do not contend
with this native KV network. No separate source/destination memory-copy cost is
added. Preset costs and diagnostic bridges are not measured performance claims.

```sh
./run_rooflang_pd.sh
# Or: one TP4 prefill instance and two TP2 decode instances, eight GPUs total.
uv run --locked python scripts/inspect_rooflang.py --pd \
  --prefill-gpu-model rtx6000d --prefill-gpu-count 4 --prefill-tp 4 \
  --decode-gpu-model rtx6000d --decode-gpu-count 4 --decode-tp 2 \
  --decode-instances 2
```

`--prefill-max-batch`, `--decode-max-batch`, `--prefill-token-budget` and
`--decode-token-budget` independently control schedulers. Global `--tp`,
`--gpu-model` and `--token-budget` remain defaults when side-specific options
are omitted. `pd-mixed-diagnostic.json` demonstrates an explicitly synthetic
10 GB/s, 3 us bridge between RTX and H200 clusters; it is a schema example,
not a measured or standard physical interconnect.

With `--reports`, inspection retains the global outer trace and per-instance audits, costs,
timing reports and graphs under `instances/p0/`, `instances/d0/`, etc.
`kv-transfers.csv` contains request ID, rank endpoints, head interval [start,end),
bytes, path/link IDs and engine start/finish times. `kv-handoffs.csv` contains
total bytes and KV waiting time until decode release (excludes D queue/compute).
The report shows all allocations, including idle leaves. `hardware-topology.json`
and `kv-topology.json` retain the source specification and lowered network.
`--trace-rounds` applies to each leaf's local zero-based rounds. Idle leaves get
zero-round reports without graphs or cost CSVs.

Explicit `--kv-link-bw` and `--kv-link-latency` retain legacy single-shard
manual-link diagnostics; they cannot be combined with independent P/D settings.
Legacy scenarios require matching TP and preserve their original cost semantics.

R1: Llama shape identity -> worker init validates all pre/block/post layers;
unsupported models fail. R2: ragged batch -> sum queries for GEMMs, sum causal
query-key pairs per request for attention; no mean context or dense padding.
R3: token progress -> original scheduler completion semantics; validate chunked
prefill emits only at the final chunk and decode advances one token per round.
R4: time -> Rooflang microseconds to engine seconds once; sum round durations
against token latency. R5: TP -> ranks 1/2/4/8, sharded projections and KV heads,
explicit collectives and slower-link sensitivity. R6: isolation -> reject
multi-instance CLI use; retain existing simulator tests.

## Cost and memory assumptions

Reference: Rooflang commit 5cf122bb260fb53dd85688cd88001627e33b018c, paper v2
sections 2.2-2.6; Llama shape from the existing `Llama31Model`. This is an
analytical FP16 prototype, not a measured vLLM performance replacement.

- H=4096, intermediate=14336, Q/KV heads=32/8, head dim=128,
  vocabulary=128256, blocks=32. All blocks are explicitly expanded.
- GEMM FLOPs=2MNK; RMSNorm=4 per element (Rooflang convention).
  Causal attention counts each request's q*k + q*(q+1)/2 pairs;
  QK and AV cost 4*head_dim per pair/head plus 5 for softmax.
  RoPE uses 3 FLOPs per Q/K element, SiLU+multiply 6 per intermediate
  element, sampling 5 per vocabulary element. These are analytical operation
  conventions, not measured constants or numerical implementation proofs.
- Flash-style attention reads cached K/V once per iteration; score matrices
  are not materialized. KV append bytes are explicit outputs kept to the
  iteration boundary. QKV read/write, residual additions and selection copies
  are represented explicitly. Kernel launch, host/tokenization, fragmentation,
  allocator overhead and implementation-specific fusion are not modeled.
- TP uses vocabulary-sharded embedding (uniform shard-hit assumption), an
  embedding AllReduce, two AllReduces per block, sharded LM head, and logits
  AllGather before replicated sampling: 66 collectives for TP>1. This differs
  deliberately from the older Ring runner's 64 block-only collectives.
- Rooflang uses its aggregate collective model on explicitly configured
  preset fabrics, not a measured NCCL implementation or packet-level ring.
- Weights and existing KV are resident tensors in each fresh execution;
  activation lifetimes follow data edges and residual forks. KV append is
  retained until the iteration sink. Scheduler KV allocation remains rounded
  to blocks and includes all admitted requests; Rooflang models exact active
  batch bytes. Its peak memory is not a replacement for scheduler admission or
  a report of all idle retained KV. An active-wave OOM aborts the run.
- Example configs select `gpu_model = "rtx6000d"`, `gpu_count = 2`, TP=2.
  `rtx6000d`, `h200` and `b300` directly instantiate the copied upstream cluster classes;
  no rates are overridden. RTX 6000D uses 148 FP16 TFLOP/s, 84 GB GDDR7,
  1398 GB/s local memory and 100 GB/s Ethernet; H200 uses 989.5 FP16 TFLOP/s,
  144 GB HBM3e, 4800 GB/s local memory and 450 GB/s NVLink ports, as encoded
  upstream. B300 uses 2250 FP16 TFLOP/s, 288 GB HBM3e, 7750 GB/s local
  memory and 900 GB/s NVLink ports. These are Rooflang preset assumptions, not independently calibrated
  measurements. RTX 6000D is not the original RTX PRO 6000 profiling target.
- All three upstream classes build an 8-GPU node with supporting CPU, memory and
  switch infrastructure. `gpu_count` allocates the first N GPUs of that topology
  and must equal TP (1/2/4/8). Unallocated GPUs execute no kernels; full topology
  is preserved, including the original collective bandwidth override. Logs,
  outer iteration labels and the explorer show model and allocated count;
  the hardware view separately reports total topology count.
- Scheduler `device_mem` must match the selected per-card preset capacity:
  84000000000 bytes for RTX 6000D, 144000000000 for H200, or
  288000000000 for B300. The example uses
  the former. Admission and model state remain owned by the original scheduler.
- Raw numeric hardware inputs remain supported only for diagnostic sensitivity
  tests and replaying earlier synthetic audits.

## Running

Run from this repository; no sibling checkout or PYTHONPATH is needed:

```sh
uv sync --locked
cargo run --release --locked -- \
  --config examples/scenarios/rooflang-llama3_1_8b/simthesizer.toml \
  --workload examples/workloads/realistic.jsonl --output-jsonl output/rooflang.jsonl
```

Use `examples/scenarios/rooflang-llama3_1_8b/python-model.toml` instead to
select the Python definition. Both configs run the bundled worker through
`uv run --locked python`. Set both `tp` and `gpu_count` to change the allocated
card count; TP>1 enables communication. Set `gpu_model = "h200"` and adjust
scheduler `device_mem` to select the H200 preset. Add `--audit-dir PATH` to the worker
command for a fresh directory containing init metadata, exact batches, per-round
summaries, and per-kernel Perfetto traces. Export the existing engine graph/state
traces to correlate round durations and token commits. The outer graph has one
Rooflang iteration node; detailed compute/communication lives in the worker trace.

The worker is launched once per runner, uses versioned JSONL without a shell,
and is terminated/reaped when the runner drops. Missing dependencies, unsupported
models, worker failures and invalid durations fail explicitly. No latency results are cached; every round still runs the complete DES.

The existing scheduler reports `lm_head_len=1` for every scheduled request,
including intermediate prefill chunks. The backend preserves that cost input;
only the scheduler commits tokens when prefill is finished. This prototype does
not infer a different logits mask or claim to reproduce vLLM chunk masking.

## Graph template reuse

Each worker owns one template for its initialized model, TP and hardware.
Kernel and tensor identities, data edges, device/stream placement and hardware
are built and validated once. The same construction formulas can either produce
concrete values (`build` / `simulate`, the rebuild reference) or record batch
bindings (`LlamaTemplate`). No second set of per-operator cost formulas exists.

Each iteration validates the batch schema and updates four aggregate inputs:
query tokens, logits rows, cached KV tokens, and the sum of per-request causal
query-key pairs. Bindings update existing tensor shapes (including residual
views and collective outputs), Work FLOPs, collective byte counts and embedding
read fractions. Static weights and tensor-to-memory references remain fixed.
The graph topology does not depend on batch cardinality in this Llama lowering.

A new, unchanged upstream Simulator is created for each execution; event queues,
resource shares, tensor lifetimes and memory accounting cannot leak across rounds,
including after OOM. This is graph reuse, not persistent physical KV allocation,
latency memoization, or concurrent shared-resource execution. Template results
reference mutable kernels: consume/export their traces before binding another
batch. The worker does this before reading the next request. Do not mutate the
owned graph/placement externally or use a template concurrently.

Add `--rebuild-graph` to the worker command for differential checks and timing
comparisons. It preserves the original per-round construction path. Offline
DAG export still uses this fresh-build path and checks the recorded duration.

## Validation commands

```sh
cargo test --workspace --locked
cargo build --locked
uv run --locked python -m unittest discover -s tests -v
```

Rooflang tests run in the normal project environment without optional skips: ragged causal work, TP collectives
and link sensitivity, hardware units and OOM, chunked-prefill/token timing,
mixed batches/model rejection, independent P/D handoff and topology validation. Tests use the
compiled debug CLI; set `SIMTHESIZER_BIN` for the integration cases when needed.

## Whole-run graph explorer

With both `--output-graph-trace` and worker `--audit-dir` enabled, export an
offline HTML view and complete JSON graph data:

```sh
uv run --locked python scripts/draw_rooflang.py \
  --trace output/rooflang-full-graph/outer.jsonl \
  --audit-dir output/rooflang-full-graph/audit \
  --output output/rooflang-full-graph/index.html
```

The outer view replays recorded graph mutations. Each iteration links to a
complete inner graph, with layer filtering, tensor/weight details, hardware
links and optional stream-order edges. Inner graphs are reconstructed from
recorded batches using the current bundled builder; the exporter checks both
reconstructed and outer durations against the audit and rejects mismatches.
Use the source revision that produced the audit. This is an offline view,
not a change to execution or an assertion that inner nodes exist in the outer
engine. Graph dependencies and observed stream order are shown separately.


## Timing coordination inspection

`--audit-trace-rounds 0 1 100 1000` on the worker retains summaries for every
iteration but exports detailed Perfetto events only at selected zero-based
indices. Without this option all audited rounds get full traces. Then run:

```sh
uv run --locked python scripts/check_rooflang_timing.py \
  --trace output/outer.jsonl --audit-dir output/audit --output-dir output/check
```

The checker verifies the selected instance's outer duration conversions and
exclusive iteration order, then checks sampled operator costs against fresh graph execution,
stream serialization and graph dependencies. Its max(compute,memory,network)
check assumes this backend's exclusive devices and stream0; it is not a generic
contention checker. The CSV reports base compute, local-memory and network costs
separately from elapsed time. Per-device collective trace entries represent one
shared collective; summing all GPU lanes double-counts concurrent work.

Upstream runtime assumptions remain visible: local tensor reads/writes contribute
to memory time and overlap compute; network time includes route latency plus
transfer and overlaps local work within a kernel. Current collectives block all
participant stream0 lanes. Multi-participant collectives do not separately price
HBM traffic, and local memory fabric alpha is not added to each tensor access.
Weights and prior KV are resident, so no initial host-to-device loading is priced.
These are modeling boundaries, not evidence of calibrated GPU accuracy.

## One-command inspection

Run `uv run --locked python scripts/inspect_rooflang.py` from the repository.
The default saves request results, metrics, `metrics.png`, configuration, workload
and small runtime/plot logs in a new directory under `output/`; no history is collected.
The metric plot reads request results only and does not require `--reports`.
Use `--gpu-model h200 --tp 4 --model-entry python` to select hardware and entry.
Use `--workload examples/workloads/realistic.jsonl` for a full workload.
`--reports` enables native history, per-round audits, CSVs, timing checks and
the report index. `--full-graphs` also enables complete HTML graphs.
`--trace-rounds` selects detailed audited rounds when reports are enabled;
all round summaries remain saved in that mode. Shared execution caches plans;
`--reference-every-round` enables the slower diagnostic reference path.
`--reference PATH` adds a comparison with matching measured results.
`--output-dir PATH` requires a new directory to prevent mixing runs.
