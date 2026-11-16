#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

exec .venv/bin/python scripts/draw_graph.py \
  output/dag/graph.jsonl \
  --workload examples/workloads/realistic.jsonl \
  --output output/dag/graph.html
