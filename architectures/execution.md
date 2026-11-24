# Model Execution Architecture

## Purpose

A `Model` describes what one iteration computes. A `ModelRunner` decides how it
executes across devices. Together they expand one scheduled batch into compute
and network nodes.

## Model

- Initialized with a parallel strategy, a model exposes a layer plan:
  pre-block layers, one representative block with its repeat count, and
  post-block layers. The plan is already sharded for that strategy.
- Layers are named semantic descriptors with shape metadata: embedding, layer
  norm, GEMM, RoPE, attention, Mamba, activation, sampler, MoE, and an
  all-reduce marker for TP synchronization. MoE layers carry their
  dispatch and combine bytes per token.
- Identical decoder blocks fold into one block and a count. A heterogeneous
  stack, such as hybrid Mamba/attention/MoE, flattens into one block with a
  count of one.
- A model reports weight bytes per device, KV bytes per token, and per-request
  state bytes. One scheduler spanning several EP ranks averages request KV/state
  costs across ranks; resident weights remain the full per-device amount.
- Built-ins: `llama3_1`, `qwen3_moe`, `nemotron_h`.

`qwen3_235b_a22b` is a fixed Qwen3 MoE preset. Its separate
[`parameter file`](../crates/simthesizer-core/src/models/qwen3_235b_a22b.toml)
pins the checkpoint source and defines 94 MoE blocks, 64/4 Q/KV heads of
dimension 128, and 128 experts with top-8 routing and no shared expert.
Select it without parameter overrides:

```toml
[system.model]
kind = "qwen3_235b_a22b"
```

Custom dimensions still use `qwen3_moe`. Native profile-based runners use TP=1.
Profile-based EP runners
require a compatible Qwen3 performance profile; none is supplied for 235B.
The fixed 235B preset additionally supports analytical shared Rooflang Attention
TP1/2/4 with complete-expert EP1/2/4/8, where TP divides EP. Expert TP is 1.
See [the shared contract](rooflang-shared.md) and `run_qwen3_235b_shared.sh`.

Llama TP partitions query heads evenly. KV heads partition when there are at
least as many heads as ranks; otherwise each rank holds one replicated KV head.
Invalid partitions are rejected. Projection shapes, weights, and KV memory use
the same per-rank head counts.

## Model Runner

A runner owns its parallel layout, device topology, and compute and network
simulators.

- **`ring`** is tensor parallelism. Each all-reduce marker becomes a ring
  all-reduce: `2(tp − 1)` steps, each of `tp` concurrent transfers carrying
  `1/tp` of the payload, separated by barriers. Links form a bidirectional
  ring.
- **`moe_ep`** is expert parallelism with `tp = 1` over a full mesh. Each MoE
  layer becomes routing, an all-to-all dispatch, one compute node per rank, a
  barrier, and an all-to-all combine. It rejects all-reduce markers.

These profile-based runners do not combine TP and EP. Shared Rooflang Qwen
groups Attention TP ranks inside a shared EP pool; its explicit graph models
each collective. Pipeline parallelism is not modeled.

For `moe_ep_multi_instance`, non-MoE layers run per rank on that rank's own
batch. Ranks join at MoE collectives and block boundaries, then split again.

## Block Fast-Forward

A runner simulates the representative block once, then one logical node adds
`(count − 1)` times that block's measured latency. Repeated blocks are assumed
to cost the same, communication included, so graph size per iteration scales
with block width rather than depth.

## MoE Routing

Expert choice is sampled uniformly from a seeded generator. Experts map to
ranks in contiguous ranges, and shared experts stay on the token's own rank.
Each rank's compute is priced by its local token count and its locally
activated experts, never by a union across ranks.

Dispatch sends one payload per token to each distinct remote owner selected
by that sample. Experts on the same rank share the payload; local and shared
experts need no transfer. Combine reverses those transfers after local reduction.

## Evolution Rules

- A new model implements `Model` from its real configuration; never alias a
  similar one.
- Layer names must match the rows of the model's profile.
- A new parallelism axis is a new runner. Collectives remain point-to-point
  transfers joined by barriers.
