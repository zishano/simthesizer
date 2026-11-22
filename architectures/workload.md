# Workload and Result Architecture

## Purpose

Workload JSONL is the only simulator input and the contract shared by every
runner. Result JSONL is the shared output. Both hold one request DAG per row.

## Request DAG

- A row is one request: `request_id`, `arrival_time`, `initial`, and non-empty
  `sub_requests`.
- A sub-request is an `llm` or a `tool_call` node; `next` indices link them.
- `initial` lists the roots explicitly and must equal the indegree-zero set
  implied by `next`. Root activation is a validated input fact, not inferred.
- `interval` delays a node's start after all predecessors finish, modeling
  think time and invocation overhead.
- A `tool_call` occupies wall-clock time (`duration`) and no device.
- Unknown fields are rejected; legacy flat rows are not accepted.
- Cycles and invalid indices are rejected. Arrival times, intervals, and tool
  durations must be finite and nonnegative. LLM input plus output must fit `u32`.

## Context Authority

Each `llm` sub-request is self-contained. `input_tokens` is the full prompt of
that step, and optional `input_token_ids` / `output_token_ids` are that step's
own token trace. Runtime never derives context or KV lineage from predecessor
completions; reuse across steps is found by the prefix cache from token IDs
alone. Request-level token counts do not exist.

Optional seeded progress (`known_tokens`, `kv_tokens`) is set together on an
`llm` sub-request so that a system can start a synthetic child mid-generation,
as disaggregated decode does. Committed workloads omit it, which means
`known_tokens = input_tokens` and `kv_tokens = 0`.

`model` records provenance and never affects simulation.

## Results

A result row repeats the request's DAG, arrival, token counts, intervals, and
tool durations, and adds one `token_latency` array per `llm` sub-request.
Entry 0 is the time from entering the scheduler to the first token; later
entries are inter-token latencies. Keeping both in one array prevents them from
drifting apart.

`scripts/results.py` reconstructs absolute token times by walking the DAG from
`arrival_time`. A node starts after every predecessor completes, plus its own
`interval`; tool durations also advance the timeline. Rows are matched by
`request_id`, LLM calls by sub-request index, and ITLs by output-token index.

## Boundaries

- The Rust types are the schema authority. Python analysis validates the fields
  it reads and requires every LLM's latency count to equal `output_tokens`.
- Raw agent traces are converted to canonical rows offline. Runtime accepts
  only canonical rows.
- JSON timing values preserve their nearest binary64 value through Rust/Python
  round trips; result validation compares input metadata exactly.

## Evolution Rules

- Convert a new trace source offline; never teach runtime a second format.
- A field that changes simulation belongs on the sub-request it affects.
- Change the schema together with both languages, bundled workloads, and
  analysis.
