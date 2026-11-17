# Scheduling Architecture

## Purpose

A `Scheduler` owns the ready LLM sub-requests of one memory domain: which run
in the next iteration, how many tokens each processes, and what KV each holds.
It is the only component that commits token progress. Runners receive batch
shapes and, for shared execution, matching request identities. A `Router` owns
whole-request placement across instances.

## Contract

- Initialization supplies weight bytes, per-request state bytes, per-token KV
  bytes, and the parallel strategy.
- A ready sub-request is enqueued with its completion node, which the scheduler
  carries back opaquely.
- Scheduling returns one batch of `(q_len, kv_len, lm_head_len)`, or nothing
  while an iteration is in flight or no work is ready.
- Completing an iteration returns finished sub-requests with their
  `token_latency`.
- A load snapshot exposes waiting and running requests and tokens.

## Chunked Prefill

`chunked_prefill` models vLLM-style continuous batching with chunked prefill.

- Requests are served first-come, first-served. An iteration admits them in
  order until `max_batch`, `max_num_batched_tokens`, or memory runs out. Each
  takes its remaining known tokens up to the remaining budget, and at least one
  token, which is a decode step.
- Usable memory per GPU is `device_mem × device_mem_utilization`. It holds weights,
  prefix-cache blocks, block-rounded private KV, and per-request state such as
  Mamba SSM state. A request is admitted only if its footprint after the
  iteration fits.
- Queued requests that hold KV but no longer fit are preempted: their KV is
  dropped and recomputed later.
- A token is emitted when an iteration brings KV up to the known tokens,
  including the final prefill chunk. The sampled token has no KV until the next
  iteration.
- On completion a sub-request releases its KV. A successor step prefills its
  full context again, except for blocks the prefix cache still holds.
- If not even the first queued request fits, the run fails.
- `max_batch` and `max_num_batched_tokens` are per rank and scale with `ep`.

## Prefix Cache

The cache and block-memory rules here describe legacy scheduling. Shared
Rooflang keeps FCFS batching/token budgets but delegates physical memory
admission to the execution domain, disables preemption, and rejects enabled
prefix caching. Its persistent KV is dense token/head/layer intervals; see
[shared execution](rooflang-shared.md). A scheduled batch may wait for its
atomic capacity reservation; scheduler admission does not imply GPU execution.

- One cache per scheduler, never shared across instances, ranks, or
  disaggregated pools.
- It is block-granular and hash-chained over token IDs, so it needs token
  traces. A hit is the longest chain of full blocks.
- Requests retain block hashes through preemption and extend them as traced
  tokens become relevant. Hash identity does not imply resident or computed KV.
- Requests reference the blocks they use. Unreferenced blocks are kept in LRU
  order and evicted only under memory pressure. Releasing a prefix makes its
  later blocks evictable first.
- After each iteration, a request publishes its newly completed full blocks,
  output tokens included, so prefixes become shareable while it still runs.
  Growth updates only the new shared blocks.
- A cache hit leaves at least one token to recompute for logits.
- Cached blocks count against device memory.

## Routing

- A router picks one destination from per-instance load snapshots. It never
  splits a request or touches scheduler state.
- `round_robin` ignores load. `moe_load_balancing` picks the lowest queued plus
  running tokens, then requests, rotating ties.
- Placement runs at simulated arrival after parent dependencies. System load
  includes scheduler work and assigned initial LLM calls awaiting enqueue;
  reservations move into scheduler ownership on enqueue. This makes same-time
  arrivals account for earlier assignments. Tool calls contribute no device load.

## Evolution Rules

- Model memory by allocating and reclaiming real blocks, not with capacity or
  hit-rate factors.
- Tool calls never reach a scheduler.
- Cache state never crosses a scheduler boundary; a shared cache tier needs an
  explicit owner above the schedulers.
