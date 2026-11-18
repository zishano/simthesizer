#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

# 使用仓库已有的 100 条请求，保留文件中的到达时间和输入、输出长度。
# P/D 要求每条请求只有一个 LLM 节点。
WORKLOAD="examples/workloads/realistic.jsonl"

# P 池：1 个 TP2 实例，共 2 张 H200。
P_GPU="h200"
P_INSTANCES=1
P_TP=4
P_MAX_BATCH=32
P_TOKEN_BUDGET=128

# D 池：1 个 TP2 实例，共 2 张 H200。
# 每轮 token 预算不限制请求的总输出长度。
D_GPU="h200"
D_INSTANCES=1
D_TP=4
D_MAX_BATCH=32
D_TOKEN_BUDGET=32

KV_HANDOFF="whole"  # layer：逐层提前传输；whole：P 完成后传输。
TOPOLOGY=""        # 同型号自动使用预设硬件图；异型号须填写拓扑 JSON。
OUTPUT_DIR=""      # 留空按时间命名；指定时必须是尚不存在的新目录。

# 默认保存指标及指标图、逐 token 结果和复现配置；1 开启历史、审计和 CSV。
REPORTS=0
# 报告开启时生效；每个实例独立编号。
TRACE_ROUNDS=(0 1 2 100 1000 4000)
# 1 同时开启报告和完整 HTML 图；长负载的 HTML 会很大。
FULL_GRAPHS=0

args=(--pd --execution shared --workload "$WORKLOAD"
  --kv-handoff "$KV_HANDOFF"
  --prefill-gpu-model "$P_GPU" --prefill-instances "$P_INSTANCES" --prefill-tp "$P_TP"
  --prefill-max-batch "$P_MAX_BATCH" --prefill-token-budget "$P_TOKEN_BUDGET"
  --decode-gpu-model "$D_GPU" --decode-instances "$D_INSTANCES" --decode-tp "$D_TP"
  --decode-max-batch "$D_MAX_BATCH" --decode-token-budget "$D_TOKEN_BUDGET"
  --trace-rounds "${TRACE_ROUNDS[@]}")
[[ -z "$TOPOLOGY" ]] || args+=(--hardware-topology "$TOPOLOGY")
[[ -z "$OUTPUT_DIR" ]] || args+=(--output-dir "$OUTPUT_DIR")
[[ "$REPORTS" != 1 ]] || args+=(--reports)
[[ "$FULL_GRAPHS" != 1 ]] || args+=(--full-graphs)

uv run --locked python scripts/inspect_rooflang.py "${args[@]}" "$@"
