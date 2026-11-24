#!/usr/bin/env bash
set -euo pipefail

project_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
agent_dir="$project_dir/agents/qwen_moe_optimization"

# 实验参数只在 EXPERIMENT 中维护；凭据沿用已有配置，不复制进运行副本。
EXPERIMENT="${EXPERIMENT:-$agent_dir/experiment.toml}"
RUNTIME_ENV="${RUNTIME_ENV:-$project_dir/agents/find_pareto_frontier/agent.env}"
RESULTS_DIR="${RESULTS_DIR:-}"
AGENT_NETWORK="${AGENT_NETWORK:-host}"
BUILDKIT_IMAGE="${BUILDKIT_IMAGE:-moby/buildkit:buildx-stable-1}"
BUILD_PROXY="${BUILD_PROXY-socks5h://127.0.0.1:17890}"
BUILD_FALLBACK_PROXY="${BUILD_FALLBACK_PROXY-http://10.250.0.3:7890}"
BUILD_IMAGE="${BUILD_IMAGE:-1}"

if [[ -n "${TMP_DOCKER_CONFIG:-}" ]]; then
    export DOCKER_CONFIG="$TMP_DOCKER_CONFIG"
fi

usage() {
    cat <<'EOF'
用法：./run_qwen_moe_agent.sh [模式] [控制器参数]
  默认 / --run       新建冻结实验，本地完成基线后启动 Docker 机制与策略搜索
  --check            静态检查配置与路径，不运行 Docker、仿真或 API
  --prepare-only     仅冻结实验副本
  --build-only       仅构建共用环境镜像
  --resume RUN_DIR   继续同一实验，复用已封存基线和剩余预算
  --help             显示帮助

实验：agents/qwen_moe_optimization/experiment.toml
结果：agents/qwen_moe_optimization/runs/YYYYMMDD-HHMMSS/
凭据：默认复用 agents/find_pareto_frontier/agent.env；可用 RUNTIME_ENV 覆盖。
仅跑本地基线：./run_qwen_moe_baseline.sh
已有匹配镜像：BUILD_IMAGE=0 ./run_qwen_moe_agent.sh
首次使用更新后的 launcher 需要构建匹配镜像；恢复时不隐式构建。
构建使用专用 host 网络 builder；不修改 Docker 全局配置。
EOF
}

mode="${1:---run}"
if (( $# > 0 )); then
    shift
fi
case "$mode" in
    --help|-h) usage; exit 0 ;;
    --run|--check|--prepare-only|--build-only|--resume) ;;
    *) printf 'error: 未知模式 %s；使用 --help 查看用法。\n' "$mode" >&2; exit 2 ;;
esac

cd -- "$project_dir"
case "$mode" in
    --build-only)
        exec bash "$agent_dir/run-agent.sh" --build-only --buildkit-image "$BUILDKIT_IMAGE" \
            --build-proxy "$BUILD_PROXY" --build-fallback-proxy "$BUILD_FALLBACK_PROXY" "$@"
        ;;
    --resume)
        # 恢复只使用冻结配置，不传当前实验或 --build。
        exec bash "$agent_dir/run-agent.sh" --env-file "$RUNTIME_ENV" --resume "$@"
        ;;
esac

args=(--experiment "$EXPERIMENT")
if [[ -n "$RESULTS_DIR" ]]; then
    args+=(--runs-root "$RESULTS_DIR")
fi
case "$mode" in
    --run)
        args+=(--env-file "$RUNTIME_ENV" --network "$AGENT_NETWORK")
        case "$BUILD_IMAGE" in
            1) args+=(--build --buildkit-image "$BUILDKIT_IMAGE" \
                --build-proxy "$BUILD_PROXY" --build-fallback-proxy "$BUILD_FALLBACK_PROXY") ;;
            0) ;;
            *) echo 'error: BUILD_IMAGE 只能是 0 或 1。' >&2; exit 2 ;;
        esac
        ;;
    --prepare-only) args+=(--prepare-only --network "$AGENT_NETWORK") ;;
    --check) args+=(--check) ;;
esac
exec bash "$agent_dir/run-agent.sh" "${args[@]}" "$@"
