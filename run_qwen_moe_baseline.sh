#!/usr/bin/env bash
set -euo pipefail

project_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
agent_dir="$project_dir/agents/qwen_moe_optimization"
EXPERIMENT="${EXPERIMENT:-$agent_dir/experiment.toml}"
RESULTS_DIR="${RESULTS_DIR:-}"
AGENT_NETWORK="${AGENT_NETWORK:-host}"

if [[ ${1:-} == --help || ${1:-} == -h ]]; then
    cat <<'EOF'
用法：./run_qwen_moe_baseline.sh [--resume RUN_DIR] [控制器参数]
  默认              新建冻结 Qwen 实验，在本地完成并封存全部基线案例
  --resume RUN_DIR   继续同一基线，只补缺失案例；完成后仅校验

参数在 agents/qwen_moe_optimization/experiment.toml 中维护。
不启动 Docker，不读取 API 凭据，不调用 agent。
完成后：./run_qwen_moe_agent.sh --resume RUN_DIR
EOF
    exit 0
fi

cd -- "$project_dir"
if [[ ${1:-} == --resume ]]; then
    exec bash "$agent_dir/run-agent.sh" --baseline-only "$@"
fi

args=(--baseline-only --experiment "$EXPERIMENT" --network "$AGENT_NETWORK")
if [[ -n "$RESULTS_DIR" ]]; then
    args+=(--runs-root "$RESULTS_DIR")
fi
exec bash "$agent_dir/run-agent.sh" "${args[@]}" "$@"
