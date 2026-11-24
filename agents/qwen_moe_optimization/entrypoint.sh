#!/usr/bin/env bash
set -euo pipefail
umask 077

mode="${1:-research}"
if (( $# > 0 )); then
    shift
fi
[[ -r /workspace/task/manifest.json ]] || {
    echo 'error: a frozen Qwen task is required' >&2
    exit 2
}
case "$mode" in
    research)
        [[ $# == 0 ]] || { echo 'error: research takes no arguments' >&2; exit 2; }
        for directory in /workspace/worktree /workspace/queue /workspace/notes /workspace/cache; do
            [[ -w "$directory" ]] || {
                printf 'error: writable directory required: %s\n' "$directory" >&2
                exit 2
            }
        done
        [[ -r /workspace/task/AGENTS.md && -r /workspace/artifacts ]] || {
            echo 'error: frozen task and baseline artifacts are required' >&2
            exit 2
        }
        ;;
    trial)
        [[ -w /workspace/cache ]] || {
            echo 'error: writable trial cache is required' >&2
            exit 2
        }
        ;;
    *) printf 'error: unsupported entrypoint mode: %s\n' "$mode" >&2; exit 2 ;;
esac

mkdir -p /home/agent/.codex /home/agent/.config /home/agent/.local/share \
    /home/agent/.local/state /workspace/cache/cargo /workspace/cache/matplotlib
if [[ ! -f /workspace/cache/cargo/.seeded ]]; then
    cp -R /opt/cargo-seed/registry /workspace/cache/cargo/
    touch /workspace/cache/cargo/.seeded
fi

task_root=/workspace/simthesizer/agents/qwen_moe_optimization
case "$mode" in
    research) exec python "$task_root/runtime.py" --inside ;;
    trial) exec python "$task_root/runner.py" --inside "$@" ;;
esac
