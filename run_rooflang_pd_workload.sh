#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

# 内置 100 个单 LLM 请求；输入、输出长度和到达时间由文件决定。
# P/D 当前要求每条请求只有一个 LLM 节点，不支持多节点 agent DAG。
WORKLOAD="examples/workloads/realistic.jsonl"
MODEL_ENTRY=rust

# P 池：4 张 H200，1 个 TP4 实例。GPU_COUNT 是整个池的卡数。
P_GPU=h200
P_GPU_COUNT=4
P_INSTANCES=1
P_TP=4
P_PP=1
P_EP=1
P_MAX_BATCH=32
P_TOKEN_BUDGET=2048

# D 池：4 张 H200，2 个 TP2 实例；每实例独立调度。
# 常规 decode 每个请求每轮处理 1 个 token，32 的预算可容纳 32 个请求。
# 此预算不限制请求的输出长度，输出长度由 workload 的 output_tokens 决定。
D_GPU=h200
D_GPU_COUNT=4
D_INSTANCES=2
D_TP=2
D_PP=1
D_EP=1
D_MAX_BATCH=32
D_TOKEN_BUDGET=32

# 同型号默认共用预设硬件图；异型号需提供含显式连接的拓扑 JSON。
HARDWARE_TOPOLOGY=""
# 留空按时间创建输出目录；指定时必须是尚不存在的新目录。
OUTPUT_DIR=""
# 保留所有轮次汇总，只保存以下各实例本地轮次的详细 trace（从 0 开始）。
TRACE_ROUNDS=(0 1 2 100 1000 4000)
# 设为 1 会重建所有轮次的大型 HTML 图，长 workload 可能耗时较多。
FULL_GRAPHS=0

if [[ ! -f "$WORKLOAD" ]]; then
  echo "Workload 不存在：$WORKLOAD" >&2
  exit 1
fi
args=(--workload "$WORKLOAD" --model-entry "$MODEL_ENTRY")
[[ -z "$HARDWARE_TOPOLOGY" ]] || args+=(--hardware-topology "$HARDWARE_TOPOLOGY")
[[ -z "$OUTPUT_DIR" ]] || args+=(--output-dir "$OUTPUT_DIR")
[[ "$FULL_GRAPHS" != 1 ]] || args+=(--full-graphs)

uv run --locked python scripts/inspect_rooflang.py --pd \
  --prefill-gpu-model "$P_GPU" --prefill-gpu-count "$P_GPU_COUNT" \
  --prefill-instances "$P_INSTANCES" --prefill-tp "$P_TP" \
  --prefill-pp "$P_PP" --prefill-ep "$P_EP" \
  --prefill-max-batch "$P_MAX_BATCH" --prefill-token-budget "$P_TOKEN_BUDGET" \
  --decode-gpu-model "$D_GPU" --decode-gpu-count "$D_GPU_COUNT" \
  --decode-instances "$D_INSTANCES" --decode-tp "$D_TP" \
  --decode-pp "$D_PP" --decode-ep "$D_EP" \
  --decode-max-batch "$D_MAX_BATCH" --decode-token-budget "$D_TOKEN_BUDGET" \
  --trace-rounds "${TRACE_ROUNDS[@]}" \
  "${args[@]}" "$@"
