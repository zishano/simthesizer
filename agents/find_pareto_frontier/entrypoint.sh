#!/usr/bin/env bash
set -euo pipefail
umask 077

if [[ ${1:-} == --check-env ]]; then
    python_check='
import sys, importlib.util, importlib.metadata
assert sys.version_info[:2] == (3, 14), sys.version
assert sys.prefix == "/opt/venv", sys.executable
import matplotlib, networkx
assert importlib.util.find_spec("vllm") is None
print(sys.executable, sys.version)
for name in ("networkx", "matplotlib"):
    print(name, importlib.metadata.version(name))
'
    # Codex commands may use login shells, whose /etc/profile resets PATH.
    for mode in -c -lc; do
        echo "Checking bash $mode"
        bash "$mode" 'set -euo pipefail; python -c "$1"; cargo --version; uv --version; codex --version' \
            sim-agent-check "$python_check"
    done
    exit 0
fi
[[ $# == 0 ]] || { echo 'error: unsupported entrypoint arguments' >&2; exit 2; }
[[ -r /workspace/task/manifest.json && -r /workspace/task/AGENTS.md ]] || {
    echo 'error: use run-agent.sh to prepare and mount a frozen task' >&2
    exit 2
}
[[ -w /workspace/artifacts && -w /workspace/cache ]] || {
    echo 'error: artifacts/cache must be writable by the container UID' >&2
    exit 2
}
mkdir -p /home/agent/.codex /home/agent/.config /home/agent/.local/share \
    /home/agent/.local/state /workspace/cache/cargo /workspace/cache/matplotlib
if [[ ! -f /workspace/cache/cargo/.seeded ]]; then
    cp -R /opt/cargo-seed/registry /workspace/cache/cargo/
    touch /workspace/cache/cargo/.seeded
fi
# No CLI update, dependency sync, simulation, or API request before preflight.
exec python /opt/sim-agent/launcher.py --inside-container
