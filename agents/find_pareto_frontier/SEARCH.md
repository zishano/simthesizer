# Dynamic policy search

The default experiment and its workload are unchanged. New task instructions
apply to newly frozen runs; resume uses the original frozen source and task.

Search decisions that change with causal state: Prefill chunks, Decode fairness,
KV concurrency/order, combined P/D backpressure, and routing with multiple
instances. Record strategy-family coverage, hypotheses, candidate IDs and exact
completed cases in `artifacts/search-coverage.json`. Static sweeps are screening,
not evidence that a dynamic rule helps. Compare winners with component ablations.
Use only observations advertised by the frozen runtime. Remaining resource work
is not measured GPU utilization; registered KV read gates are not a critical-path
oracle. Missing signals/mechanisms belong in proposals, not fabricated scores.

## Executable search catalog and planner

Newly frozen runs include `search_space.py` and `search_plan.py`. From the agent's
container, generate an opt-in starter catalog in a new artifact directory:

```bash
python /workspace/simthesizer/agents/find_pareto_frontier/search_space.py \
  --output-dir /workspace/artifacts/catalog-v1
python /workspace/simthesizer/agents/find_pareto_frontier/search_plan.py \
  --manifest /workspace/task/manifest.json \
  --catalog /workspace/artifacts/catalog-v1/catalog.json \
  --output /workspace/artifacts/search-plan.json
```

The catalog contains present-load routing, aged-short Prefill, Decode fairness,
bounded microbatch waiting, pressure-dependent chunks, KV demand/aging priority,
and hysteretic KV concurrency. It includes individual components, combinations,
leave-one-out ablations and static chunk controls. `--recipe recipe.json --expand`
generates a customized combination and its controls. Recipes enable boolean
component names; parameters and units are defined in `DEFAULT_RECIPE` and the
library `examples/policies/composable.rs`. These are executable hypotheses,
not measured improvements. Waiting uses existing timers; arrivals do not wake
them early. Global resource pressure is not a per-instance utilization estimate.

The planner validates catalog source identities, candidate snapshots, frozen
cases, trial identities and result artifacts, and recomputes completion and
scores. It reports attempts, failures, complete/missing cases, matched controls,
remaining budget and one suggested case. Copy that suggestion's source files
into the writable candidate directory, replacing its previous Rust files, then
run its `evaluate_argv` through the trusted evaluator. Refresh as needed after
new evidence. Each refresh rechecks the entire ledger and records
`planning_wall_seconds`; this analysis and extra evaluations consume host time.
The planner runs no trials and changes no frontier or budget.

Suggestions screen uncovered applicable families, complete promising training
cases, promote training Pareto candidates to available validation cases, and
fill matching controls. Routing recipes are skipped when neither pool offers
multiple instances. No unfinished or rejected receipt counts as completed.
An open attempt suspends suggestions pending inspection. Rejected evidence
withholds the budget estimate and suggestions until its cause is resolved.
Failed candidate/case pairs are listed for deliberate retry after diagnosis;
suggestions advance to unattempted pairs instead of repeating the same failure.
Authored hypotheses belong in separate notes, never in place of result evidence.
Checked receipts remain provisional: only external clean replay verifies results. Missing
validation is reported explicitly, not treated as passed.

The catalog is a starting set, not the search boundary. Generate new parameters,
combine components, or write arbitrary legal Rust policies when it is exhausted.
Uncatalogued completed candidates remain visible in the report. Preserve exact
source IDs and causal hypotheses for these candidates too. An empty suggestion
is not a new stopping rule; the frozen external budgets still govern search.
Older runs retain their original frozen tools and task instructions.

## Causal observations and diagnostics

`RouteObservation`, `BatchObservation` and `KvObservation` expose `environment`.
The borrowed view computes `pressure()`, `memory()`, `registered_reads()` and
`ready_tokens(request, layer, rank)` only when called. No new per-event sampling
runs by default. Pressure is remaining full-capacity service seconds, separated
into active, setup and waiting demands. Memory is allocated capacity, including
the existing idealized `known_trace` reservations. Read requirements cover
already-expanded batches; they do not identify which layer is currently stalled.
Pressure covers submitted resource work; counts are demand components, not
requests or kernels. Resource/device names are global, without a P/D instance
mapping. Queries scan current demands, allocations or KV gates; reuse a returned
summary within a callback instead of querying inside sorting comparisons.

The executable seed `examples/policies/adaptive.rs` combines queue-age ordering,
queue-dependent Prefill budgets and fabric-pressure-dependent KV windows. Copy
it into the frozen candidate directory to evaluate; the default candidate stays
unchanged. This is an example hypothesis, not a promoted optimization. Compare
component ablations and report both simulated metrics and host execution cost.

Optional compact samples in `policy-stats.json`:

```toml
[artifacts.policy_diagnostics]
max_samples = 32
every_decisions = 1000
```

Omit this table for the original path. Samples precede candidate actions, start
at decision zero, and stop at the configured bound (maximum 128). Each retains
at most 64 registered reads plus their total count. Reference mode keeps its
existing bypass and produces no samples. This sample-count limit is separate
from the older graph `diagnostic_budget_mib` option; no graph capture is enabled.
Resource/device summaries scale with topology, so the limit is not a byte cap.
The runtime advertises `causal-environment-v1` and `bounded-policy-samples-v1`
through `--policy-capabilities`. Sampling may cost host time and is explicit in
the frozen trial identity; it never changes simulated work or dependencies.

## Optional synthetic workloads

Generate inputs before preparing an experiment, using a new directory:

```bash
.venv/bin/python agents/find_pareto_frontier/generate_workloads.py \
  --output-dir output/search-inputs-v1 \
  --num-requests 64 --arrival-interval-s 0.005 \
  --train-seed 10 --validation-seed 11
```

This creates train/validation JSONL pairs for four profiles and `manifest.json`
with parameters, generator and file hashes. All records contain one LLM call.

| Profile | Arrival and prompt shape |
|---|---|
| sustained | Jittered periodic arrivals, short prompts |
| burst | Simultaneous groups, short prompts; group interval = base interval × group size |
| mixed | Jittered periodic arrivals, alternating short/long prompts |
| context | Jittered periodic arrivals, long prompts |

`--help` lists prompt/output ranges, context bound, burst size and jitter. Match
`--max-context-tokens` to the model. Inputs omit token IDs and do not exercise
prefix caching. Names describe controlled shapes, not measured distributions,
guaranteed saturation or independent evidence. Short examples only test behavior;
capacity claims need longer arrivals and queue-growth evidence.

Copy the existing experiment to a separate TOML and add selected workload pairs:

```toml
[[workloads]]
id = "mixed-train"
path = "output/search-inputs-v1/mixed-train.jsonl"
split = "train"

[[workloads]]
id = "mixed-validation"
path = "output/search-inputs-v1/mixed-validation.jsonl"
split = "validation"
```

For a copy saved as `agents/find_pareto_frontier/experiment-expanded.toml`:

```bash
EXPERIMENT=agents/find_pareto_frontier/experiment-expanded.toml ./run_rooflang_baseline.sh
./run_rooflang_agent.sh --resume RUN_DIR
```

Use the baseline command's printed `RUN_DIR`; do not resume an older experiment
to change its inputs. Rebuild the environment image if its input check requests it.

Start with a small selected matrix. The evaluator expands each workload by all
arrival multipliers; baseline cases also consume the frozen budget. Screening
and promotion use existing complete case evaluations, never truncated requests.
Adding cases increases experiment time explicitly; generation adds no simulator
runtime work. Keep the original input as a regression case.

Validation inputs are visible to search. Different generation seeds do not make
them unseen data or guarantee different contents with fixed lengths/zero jitter.
The current external verifier clean-replays frozen cases; it does not implement
holdout evaluation or establish real-hardware accuracy.

## Runtime regression checks

Keep the previous release executable before changing simulator code. Compare it
with the new executable using identical config, workload and Python workers:

```bash
.venv/bin/python tools/benchmark_simulator.py \
  --baseline PATH_TO_OLD_BINARY --candidate target/release/simthesizer \
  --case original CONFIG.toml WORKLOAD.jsonl \
  --output output/runtime-comparison --pairs 5 --max-rss-regression 0.05
```

The tool warms up, alternates execution order, validates complete native results,
requires byte-identical results and checks the median paired wall-time ratio
(default regression limit 5%). It records sampled process-tree RSS, commands,
hashes and failures. RSS sums shared pages and can miss peaks between samples;
it is not an exact OS high-water mark. Use sufficiently long cases, keep competing
builds/runs idle, and inspect per-pair values before attributing noise to code.
Use `--identity` for additional unchanged worker/dependency files. Enabled
diagnostics and changed policies need separately reported host costs; the default
path must retain its original behavior. This check measures simulator execution,
not the hardware serving speed predicted by it.
