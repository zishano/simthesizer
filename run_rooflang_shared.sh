#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

# 空路径运行小请求；真实负载可设为 examples/workloads/realistic.jsonl。
WORKLOAD=""
P_GPU="h200"
P_INSTANCES=1
P_TP=2
P_TOKEN_BUDGET=128
D_GPU="h200"
D_INSTANCES=1
D_TP=2
D_TOKEN_BUDGET=32
KV_HANDOFF="whole"  # layer：逐层提前传输；whole：P 完成后传输。
TOPOLOGY=""        # 异型号 GPU 需要显式硬件图。
TRACE_ROUNDS=(0 1 2)
REPORTS=0         # 1 保存历史、审计和 CSV；默认保存指标和指标图。
FULL_GRAPHS=0     # 1 同时开启报告和完整 HTML 图。

args=(--pd --execution shared --kv-handoff "$KV_HANDOFF"
  --prefill-gpu-model "$P_GPU" --prefill-instances "$P_INSTANCES" --prefill-tp "$P_TP"
  --prefill-token-budget "$P_TOKEN_BUDGET"
  --decode-gpu-model "$D_GPU" --decode-instances "$D_INSTANCES" --decode-tp "$D_TP"
  --decode-token-budget "$D_TOKEN_BUDGET" --trace-rounds "${TRACE_ROUNDS[@]}")
[[ -z "$WORKLOAD" ]] || args+=(--workload "$WORKLOAD")
[[ -z "$TOPOLOGY" ]] || args+=(--hardware-topology "$TOPOLOGY")
[[ "$REPORTS" != 1 ]] || args+=(--reports)
[[ "$FULL_GRAPHS" != 1 ]] || args+=(--full-graphs)
uv run --locked python scripts/inspect_rooflang.py "${args[@]}" "$@"
