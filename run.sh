cargo run --release --locked -- \
  --config examples/scenarios/1xrtx-pro-6000--llama3_1_8b--realistic/simthesizer.toml \
  --workload examples/workloads/realistic.jsonl \
  --output-jsonl output/dag/sim.jsonl \
  --output-graph-trace output/dag/graph.jsonl \
  --output-engine-trace output/dag/state.jsonl