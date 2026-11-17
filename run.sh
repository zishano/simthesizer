# 一条命令运行演示：
# 自动生成：
# - 逐 token 时间与指标汇总
# - 延迟、吞吐量图片
# - 外层动态 DAG、完整两层图
# - 算子计算／显存／通信耗时 CSV
# - 时间协调检查报告
# - 汇总入口 index.html
# uv run --locked python scripts/inspect_rooflang.py

# 真实负载
uv run --locked python scripts/inspect_rooflang.py \
  --workload examples/workloads/realistic.jsonl \
  --gpu-model h200 \
  --tp 2