#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
project_dir=$(cd -- "$script_dir/../.." && pwd)
python_bin=${SIM_AGENT_PYTHON:-$project_dir/.venv/bin/python}
if [[ ! -x "$python_bin" ]]; then
    python_bin=$(command -v python3) || {
        echo 'error: Python 3.11+ is required; set SIM_AGENT_PYTHON.' >&2
        exit 2
    }
fi
if ! "$python_bin" -c 'import tomllib' 2>/dev/null; then
    echo 'error: launcher requires Python 3.11+; set SIM_AGENT_PYTHON to its executable.' >&2
    exit 2
fi
# Parsing TOML and building argv in Python avoids shell eval and quoting hazards.
exec "$python_bin" "$script_dir/launcher.py" "$@"
