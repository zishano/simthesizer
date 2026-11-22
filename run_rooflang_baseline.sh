#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
agent_dir="$script_dir/agents/find_pareto_frontier"

# 与搜索使用同一个实验配置；不读取 API 凭据，不启动 Docker。
EXPERIMENT="${EXPERIMENT:-$agent_dir/experiment.toml}"
RESULTS_DIR="${RESULTS_DIR:-}"          # 留空使用实验的 artifacts.runs_root。
AGENT_NETWORK="${AGENT_NETWORK:-host}" # 冻结后续搜索容器的网络模式。
# SIM_AGENT_PYTHON 可指定解释器，默认使用项目 .venv/bin/python。

if [[ ${1:-} == --help || ${1:-} == -h ]]; then
    cat <<'EOF'
用法：./run_rooflang_baseline.sh [--resume RUN_DIR] [底层启动器参数]
  默认              新建冻结实验，在本地完成全部基线并封存结果
  --resume RUN_DIR   继续本地基线，只补缺失案例；已封存时仅校验

参数在脚本开头，也可通过 EXPERIMENT、RESULTS_DIR、AGENT_NETWORK 覆盖。
本地 Python、Rust 和相关依赖版本须与实验/镜像兼容。
不启动 Docker，不读取 agent.env，不调用模型 API。
完成后按输出的目录启动：./run_rooflang_agent.sh --resume RUN_DIR
EOF
    exit 0
fi

cd -- "$script_dir"
if [[ ${1:-} == --resume ]]; then
    exec bash "$agent_dir/run-agent.sh" --baseline-only "$@"
fi

args=(--baseline-only --experiment "$EXPERIMENT" --network "$AGENT_NETWORK")
if [[ -n "$RESULTS_DIR" ]]; then
    args+=(--runs-root "$RESULTS_DIR")
fi
exec bash "$agent_dir/run-agent.sh" "${args[@]}" "$@"
