# Validation Architecture

## Purpose

The simulator and vLLM replay emit the same result JSONL. `scripts/compare.py`
reports serving-metric errors; `scripts/draw.py` plots the measurements. Both use
`scripts/results.py` for parsing, DAG timing, and metric definitions.

## Experiments

`devenv shell` supplies Rust, Python, and uv. Python's version comes from
`.python-version`; dependencies come from `pyproject.toml` and `uv.lock`.
Shell entry runs `uv sync --locked`. The default environment includes Matplotlib
and Ruff. `devenv --profile vllm shell` also installs the `vllm` dependency group
and supplies CUDA build tools and host driver libraries for profiling and replay.
Scripts use this project environment without inline dependency declarations.

Examples use `examples/scenarios/<count>x<device>--<model>--<workload>/`.
Device names use lowercase words separated by hyphens; model and workload names
retain their file stems. The GPU count identifies a single-GPU setup without a
separate `single-gpu` label. Add a topology or precision suffix only to distinguish
otherwise identical scenarios. `baseline.toml` holds native vLLM engine arguments;
`simthesizer.toml` holds the simulator tree. Workloads live in `examples/workloads/`
and profiles in `examples/data/<device>/`.

For the bundled RTX PRO 6000 Blackwell Server example:

```sh
scenario_dir=examples/scenarios/1xrtx-pro-6000--llama3_1_8b--realistic
cargo run --release -- --config "$scenario_dir/simthesizer.toml" \
  --workload examples/workloads/realistic.jsonl --output-jsonl output/realistic-sim.jsonl
CUDA_VISIBLE_DEVICES=0 uv run scripts/baseline.py --config "$scenario_dir/baseline.toml" \
  --workload examples/workloads/realistic.jsonl --output-jsonl output/realistic-baseline.jsonl
```

The baseline accepts `--model` for a local model directory. Its GPU and serving
settings must match the profile. These examples use FP16 eager execution, TP1,
and no prefix caching. Copied workloads retain their source token IDs: ShareGPT's
IDs exceed Llama's vocabulary, and agentic traces need a DAG-capable replay runner.
They are not interchangeable baseline inputs.

## Runners

- **Simthesizer** is the CLI.
- **Baseline** (`scripts/baseline.py`) drives vLLM with the configured weights,
  replays each request's token IDs at its arrival time with EOS ignored, and
  records client-visible per-token latency. The example uses dummy weights.
  Warmup batches of one and `max_num_seqs` requests finish before replay begins.
  TTFT starts at the scheduled arrival; coalesced emissions divide their interval
  evenly among tokens. Every request must emit its requested token count.
  A metadata sidecar records hardware, versions, engine arguments, and workload hash.

The replay supports single-LLM-call workloads only. Analysis supports full DAGs,
including parallel branches, joins, intervals, and tools; those workloads still
need a DAG-capable reference runner.

## Comparison and figures

Pass explicit files as `[LABEL=]PATH`; an omitted label defaults to the file stem.
The first file is the reference. Any number of candidates can follow. Paths have
no required directory layout, and neither tool discovers files implicitly.

```sh
python scripts/compare.py real=output/realistic-baseline.jsonl \
  sim=output/realistic-sim.jsonl --output output/comparison.json
uv run scripts/draw.py real=output/realistic-baseline.jsonl \
  sim=output/realistic-sim.jsonl --output output/comparison.svg
uv run scripts/draw.py real=output/realistic-baseline.jsonl \
  sim=output/realistic-sim.jsonl --view scatter --metrics ttft tpot e2e \
  --output output/latency-scatter.pdf
```

Comparison uses only the Python standard library. Drawing declares its
Matplotlib dependency in `pyproject.toml`. Run both tools inside `devenv shell`.
`--metrics` selects metrics in display order;
comparison also accepts `--percentiles` (default: 50, 95, 99). Its stdout table and
optional JSON contain the same values, with explicit units and sample counts.
Mean values are always reported, including one value per run for scalar metrics.

Drawing supports latency CDFs or paired scatter plots, plus `throughput` and
`cumulative` time-series panels. A single run can be drawn without a reference;
scatter requires at least two. `--window-sec` controls throughput bins only.
`--title`, `--columns`, and `--size WIDTH HEIGHT` control the figure layout.
The output extension selects PNG, SVG, or PDF.

## Metrics

For an LLM call with latency array `L` and `N = len(L)`:

| Metric | Definition | Sample weighting |
| --- | --- | --- |
| `ttft` | `L[0]`; call enqueue to first token, including queueing | One per call with output |
| `tpot` | `sum(L[1:]) / (N - 1)` | One per call with at least two tokens |
| `itl` | Each entry of `L[1:]` | One per inter-token interval |
| `e2e` | `sum(L)` | One per call with output |
| `normalized_e2e` | `sum(L) / N` | One per call with output; seconds/token |
| `request_e2e` | Last DAG node completion minus request arrival | One per request; includes tools and intervals |
| `makespan` | Last DAG completion minus first request arrival | One per run |
| `output_throughput` | All LLM output tokens divided by makespan | One per run |
| `request_throughput` | Completed request DAGs divided by makespan | One per run |
| `tokens_per_user` | `(N - 1) / sum(L[1:])`; excludes TTFT | One per LLM call with at least two tokens and finite positive rate |
| `tokens_per_gpu` | `output_throughput / allocated physical GPU count` | One per run with known GPU allocation |

TTFT and LLM E2E start when the call becomes ready after its interval. TPOT
weights calls equally; ITL weights token intervals equally. Zero-output calls
have no latency samples and contribute no execution time to reconstructed DAGs.
Throughput includes arrival gaps, tool time, and drain time; it measures this
finite workload, not saturation capacity. Zero-duration runs have undefined rates.

For `tokens_per_user`, a user means one LLM call, not an account spanning calls.
Rates are computed before averaging; reciprocal mean TPOT is a different quantity.
GPU allocation comes from `config.toml` beside the result. Explicit runner GPU IDs
are deduplicated across P/D and instances; unused topology devices are excluded.
Independent legacy allocations without IDs use summed hardware `gpu_count`.
Missing or ambiguous GPU metadata leaves the rate undefined. Inspection summaries
include eligible user sample count and the GPU denominator. `metrics.png` retains
the six latency/time-series panels. A separate `throughput.png` plots one point
per run: X is mean `tokens_per_user`, Y is whole-run `tokens_per_gpu`. These are
deployment summaries, not individual request points. Multiple completed runs can
be plotted together, including different arrival rates; no curve is inferred from
a single run. The GPU mean uses makespan, not padded throughput bins.

Refresh a saved run without simulation:

```sh
python scripts/draw.py RUN/result.jsonl --output RUN/metrics.png --metrics-json RUN/metrics.json
python scripts/draw_throughput.py RUN/result.jsonl --output RUN/throughput.png
```

For each latency metric, report the reference and prediction's mean and selected
percentiles, plus signed relative error `100 * (prediction - reference) / reference`.
Percentiles use linear interpolation at `(n - 1) * p / 100`. Negative error means
underprediction. Paired samples additionally report:

- MAE: `mean(abs(prediction - reference))`, in the metric's units.
- Bias: `mean(prediction - reference)`, in the metric's units.
- MAPE: `100 * mean(abs(prediction - reference) / reference)`.
- WAPE: `100 * sum(abs(prediction - reference)) / sum(reference)`.

MAPE gives small and large latencies equal weight and can amplify small reference
values. WAPE weights their absolute contribution; MAE retains the physical scale.
Pairs match request IDs, sub-request indices, and token indices. Distribution
summaries alone can miss errors that swap fast and slow requests.

Zero denominators and empty samples produce `null` in JSON and `n/a` in text.
MAPE excludes zero-reference samples and reports its eligible count; MAE and WAPE
retain their absolute errors. Nothing is clipped or replaced with an epsilon.

The choice follows [Vidur's latency-error evaluation, §7.2](https://www.microsoft.com/en-us/research/uploads/prod/2024/05/vidur_mlsys24.pdf)
and [vLLM's TTFT, TPOT, ITL, and E2E reporting](https://docs.vllm.ai/en/latest/cli/bench/serve/).
Vidur evaluates output-length-normalized E2E and tail latency; this tool also
reports paired errors to expose request-level inaccuracies. DTW is excluded:
warping the time axis can conceal queueing and completion-time errors.

## Validation boundary

Agent review and decision recording follow the [agent workflow](agent-workflow.md).
`tools/validate.py` separately pairs and compares supplied engine CSV bundles
with declared clocks, semantics, and input identities. Its `run-sim` command
collects native JSONL observations and creates a comparison bundle; see
[observability](observability.md) and the [tool guide](../.agents/tools/validation.md).
Engine observations must not be inferred from request token timings.

All compared or overlaid runs must contain the same request IDs, arrivals, DAG
edges, intervals, token counts, and tool durations. Optional models and token
traces are compared when both files contain them; simulator results omit token
traces. Hardware and serving settings remain the experiment owner's responsibility.
Matching exported fields cannot prove that omitted token traces were identical.

Malformed DAGs, duplicate IDs, invalid timings, and missing or extra token
latencies fail before a report is written. Truncated runs are unsupported.
The loader cannot detect a request missing from every supplied file.

Time-series panels use a common origin at the first request arrival, without
shifting individual curves. Throughput counts all LLM output tokens, including
the first, in fixed-width `[left, right)` bins and pads shorter runs with zeros.
The final bin retains the full window width. Cumulative output needs no bins.

## Evolution Rules

- A runner joins by emitting result JSONL in the shared schema.
- Compare runs with the same completion policy; every runner finishes every
  request.
- Bundled scenarios and profiles describe the same machine. Changing one
  without the other pairs latencies with the wrong capacity.
