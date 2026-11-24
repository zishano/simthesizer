#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

# 从 realistic.jsonl 随机抽取 10 条（种子 42），保留原始顺序、ID、到达时间和内容。
# P/D 要求每条请求只有一个 LLM 节点；专家路由仍采用固定种子的模拟分布。
# WORKLOAD="examples/workloads/realistic-10-seed42.jsonl"
WORKLOAD="examples/workloads/realistic.jsonl"

# P 池：2 个完整副本，每副本 EP8、Attention TP2，共 16 张 H200。
P_GPU="h200"
P_INSTANCES=2
P_TP=2
P_EP=8
P_EXPERT_TP=1
P_MAX_BATCH=32
P_TOKEN_BUDGET=128

# D 池：同样 16 张 H200；P/D 合计 32 张模拟 GPU。
# batch/token 预算按每个完整副本计算，不限制请求的总输出长度。
D_GPU="h200"
D_INSTANCES=2
D_TP=2
D_EP=8
D_EXPERT_TP=1
D_MAX_BATCH=32
D_TOKEN_BUDGET=32

ROUTING_SEED=0
KV_HANDOFF="whole"  # layer：逐层提前传输；whole：P 完成后传输。
TOPOLOGY=""        # 同型号使用预设硬件图；异型号须填写拓扑 JSON。
OUTPUT_DIR=""      # 留空按时间命名；指定时必须是尚不存在的新目录。

# 默认保存逐 token 结果、指标、指标图和复现配置；1 开启历史、审计和 CSV。
REPORTS=0
WALL_PROFILE=0                   # 累计实际耗时，不开启历史；--no-wall-profile 可关闭。
TRACE_ROUNDS=(0 1 2 100 1000 4000)  # 报告开启时生效，每个实例独立编号。
FULL_GRAPHS=0                     # 1 同时开启报告和完整 HTML 图。

if [[ ! -f "$WORKLOAD" ]]; then
  echo "Workload 不存在：$WORKLOAD" >&2
  exit 1
fi

args=(--model qwen3_235b_a22b --pd --execution shared --tp 1 --workload "$WORKLOAD"
  --routing-seed "$ROUTING_SEED" --kv-handoff "$KV_HANDOFF"
  --prefill-gpu-model "$P_GPU" --prefill-instances "$P_INSTANCES"
  --prefill-tp "$P_TP" --prefill-ep "$P_EP" --prefill-expert-tp "$P_EXPERT_TP"
  --prefill-max-batch "$P_MAX_BATCH" --prefill-token-budget "$P_TOKEN_BUDGET"
  --decode-gpu-model "$D_GPU" --decode-instances "$D_INSTANCES"
  --decode-tp "$D_TP" --decode-ep "$D_EP" --decode-expert-tp "$D_EXPERT_TP"
  --decode-max-batch "$D_MAX_BATCH" --decode-token-budget "$D_TOKEN_BUDGET"
  --trace-rounds "${TRACE_ROUNDS[@]}")
[[ -z "$TOPOLOGY" ]] || args+=(--hardware-topology "$TOPOLOGY")
[[ -z "$OUTPUT_DIR" ]] || args+=(--output-dir "$OUTPUT_DIR")
[[ "$REPORTS" != 1 ]] || args+=(--reports)
[[ "$WALL_PROFILE" != 1 ]] || args+=(--wall-profile)
[[ "$FULL_GRAPHS" != 1 ]] || args+=(--full-graphs)

uv run --locked python scripts/inspect_rooflang.py "${args[@]}" "$@"
