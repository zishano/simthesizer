#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
agent_dir="$script_dir/agents/find_pareto_frontier"

# 常用启动参数；可直接修改，也可通过同名环境变量覆盖。
# 模型、GPU、负载、SLO、搜索预算统一写在 EXPERIMENT 中。
EXPERIMENT="${EXPERIMENT:-$agent_dir/experiment.toml}"
RUNTIME_ENV="${RUNTIME_ENV:-$agent_dir/agent.env}"
RESULTS_DIR="${RESULTS_DIR:-}"          # 留空沿用实验配置的 artifacts.runs_root。
AGENT_NETWORK="${AGENT_NETWORK:-host}" # 与宿主共享网络，访问本地代理/API。
BUILDKIT_IMAGE="${BUILDKIT_IMAGE:-moby/buildkit:buildx-stable-1}"
BUILD_PROXY="${BUILD_PROXY-socks5h://127.0.0.1:17890}"
BUILD_FALLBACK_PROXY="${BUILD_FALLBACK_PROXY-http://10.250.0.3:7890}"
BUILD_IMAGE="${BUILD_IMAGE:-1}"         # 新运行前构建镜像；已有匹配镜像可设为 0。

if [[ -n "${TMP_DOCKER_CONFIG:-}" ]]; then
    export DOCKER_CONFIG="$TMP_DOCKER_CONFIG"
fi

usage() {
    cat <<'EOF'
用法：./run_rooflang_agent.sh [模式] [底层启动器参数]
  默认 / --run       按实验配置构建镜像并启动
  --build-only       仅构建镜像，不需要实验配置或凭据
  --check            检查实验配置与缺失能力，不启动 Docker
  --prepare-only     保存实验副本，不启动 Docker
  --resume RUN_DIR   恢复已有实验，沿用冻结配置及剩余预算
  --help             显示此帮助

已提供 experiment.toml：沿用 shared workload 脚本的 P/D H200 TP4、100 请求。
运行前可调整该文件中的初始 SLO、搜索范围与预算。
首次配置凭据（检查/准备不需要）：
  cp -n agents/find_pareto_frontier/agent.env.example agents/find_pareto_frontier/agent.env
凭据只填入 agent.env，不写入实验配置。
相对路径按项目根目录解释。额外参数跟在模式之后，例如 --run --network host。
构建优先使用本地 SOCKS5H，网络失败后尝试备用 HTTP 代理一次。
BUILD_PROXY/BUILD_FALLBACK_PROXY 可覆盖；备用值设为空可关闭重试。
构建使用项目专用的 host 网络 builder；不修改 Docker 全局配置。
EOF
}

mode="${1:---run}"
if (( $# > 0 )); then
    shift
fi
case "$mode" in
    --help|-h) usage; exit 0 ;;
    --run|--build-only|--check|--prepare-only|--resume) ;;
    *) printf 'error: 未知模式 %s；使用 --help 查看用法。\n' "$mode" >&2; exit 2 ;;
esac

# 从其他目录调用时，场景、负载与输出路径也按项目根目录解析。
cd -- "$script_dir"
case "$mode" in
    --build-only)
        exec bash "$agent_dir/run-agent.sh" --build-only --buildkit-image "$BUILDKIT_IMAGE" \
            --build-proxy "$BUILD_PROXY" --build-fallback-proxy "$BUILD_FALLBACK_PROXY" "$@"
        ;;
    --resume)
        # 不传入当前实验、镜像或 --build，避免覆盖冻结实验。
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
    --prepare-only)
        args+=(--prepare-only --network "$AGENT_NETWORK")
        ;;
    --check)
        args+=("$mode")
        ;;
esac
exec bash "$agent_dir/run-agent.sh" "${args[@]}" "$@"
