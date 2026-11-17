#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

# Each count is the pool total: instances * TP. PP/EP currently require 1.
P_GPU=h200
P_GPU_COUNT=4
P_INSTANCES=1
P_TP=4
P_PP=1
P_EP=1
P_MAX_BATCH=32
P_TOKEN_BUDGET=3
D_GPU=h200
D_GPU_COUNT=4
D_INSTANCES=2
D_TP=2
D_PP=1
D_EP=1
D_MAX_BATCH=32
D_TOKEN_BUDGET=3

# Empty uses the five-input / three-output token diagnostic workload.
WORKLOAD=""
# Empty uses a common preset topology. Mixed models require explicit bridge links.
HARDWARE_TOPOLOGY=""
args=()
[[ -z "$WORKLOAD" ]] || args+=(--workload "$WORKLOAD")
[[ -z "$HARDWARE_TOPOLOGY" ]] || args+=(--hardware-topology "$HARDWARE_TOPOLOGY")
uv run --locked python scripts/inspect_rooflang.py --pd \
  --prefill-gpu-model "$P_GPU" --prefill-gpu-count "$P_GPU_COUNT" \
  --prefill-instances "$P_INSTANCES" --prefill-tp "$P_TP" --prefill-pp "$P_PP" --prefill-ep "$P_EP" \
  --prefill-max-batch "$P_MAX_BATCH" --prefill-token-budget "$P_TOKEN_BUDGET" \
  --decode-gpu-model "$D_GPU" --decode-gpu-count "$D_GPU_COUNT" \
  --decode-instances "$D_INSTANCES" --decode-tp "$D_TP" --decode-pp "$D_PP" --decode-ep "$D_EP" \
  --decode-max-batch "$D_MAX_BATCH" --decode-token-budget "$D_TOKEN_BUDGET" \
  "${args[@]}" "$@"
