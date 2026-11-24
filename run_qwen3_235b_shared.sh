#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

# 空路径：输入 5 / 输出 3 tokens。每个副本 8 卡；P/D 各 2 副本，共 32 张 H200。
WORKLOAD=""
P_GPU="h200"
P_INSTANCES=2
P_TP=2
P_EP=8
P_EXPERT_TP=1
P_TOKEN_BUDGET=128

D_GPU="h200"
D_INSTANCES=2
D_TP=2
D_EP=8
D_EXPERT_TP=1
D_TOKEN_BUDGET=32
ROUTING_SEED=0
KV_HANDOFF="whole"
TOPOLOGY=""
TRACE_ROUNDS=(0 1 2)
REPORTS=0
FULL_GRAPHS=0

args=(--model qwen3_235b_a22b --pd --execution shared --tp 1
  --routing-seed "$ROUTING_SEED" --kv-handoff "$KV_HANDOFF"
  --prefill-gpu-model "$P_GPU" --prefill-instances "$P_INSTANCES" --prefill-ep "$P_EP"
  --prefill-tp "$P_TP" --prefill-expert-tp "$P_EXPERT_TP"
  --prefill-token-budget "$P_TOKEN_BUDGET"
  --decode-gpu-model "$D_GPU" --decode-instances "$D_INSTANCES" --decode-ep "$D_EP"
  --decode-tp "$D_TP" --decode-expert-tp "$D_EXPERT_TP"
  --decode-token-budget "$D_TOKEN_BUDGET" --trace-rounds "${TRACE_ROUNDS[@]}")
[[ -z "$WORKLOAD" ]] || args+=(--workload "$WORKLOAD")
[[ -z "$TOPOLOGY" ]] || args+=(--hardware-topology "$TOPOLOGY")
[[ "$REPORTS" != 1 ]] || args+=(--reports)
[[ "$FULL_GRAPHS" != 1 ]] || args+=(--full-graphs)
uv run --locked python scripts/inspect_rooflang.py "${args[@]}" "$@"
