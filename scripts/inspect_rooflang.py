"""Run Rooflang simulation and collect metrics, graphs and timing checks."""

import argparse
import html
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from results import load_run, measurements, metric_unit, summarize

ROOT = Path(__file__).resolve().parents[1]


def run(command, log):
    print(f"运行：{' '.join(map(str, command))}", flush=True)
    with log.open("w") as stream:
        result = subprocess.run(
            command, check=False, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT
        )
    if result.returncode:
        raise RuntimeError(f"命令失败，日志：{log}\n{log.read_text()[-4000:]}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workload", type=Path, help="默认：输入 5 / 输出 3 tokens 的小请求"
    )
    parser.add_argument("--output-dir", type=Path, help="必须是不存在的新目录")
    parser.add_argument(
        "--gpu-model", choices=["rtx6000d", "h200", "b300"], default="rtx6000d"
    )
    parser.add_argument("--tp", type=int, choices=[1, 2, 4, 8], default=2)
    parser.add_argument("--model-entry", choices=["rust", "python"], default="rust")
    parser.add_argument(
        "--token-budget", type=int, help="默认：演示 3；外部工作负载 2048"
    )
    parser.add_argument(
        "--full-graphs",
        action="store_true",
        help="外部工作负载也生成全部轮次的两层图，可能很大",
    )
    parser.add_argument(
        "--trace-rounds",
        type=int,
        nargs="+",
        help="只导出这些轮次的详细时间线（从 0 开始）",
    )
    parser.add_argument(
        "--reference", type=Path, help="可选：匹配工作负载的真实结果 JSONL"
    )
    args = parser.parse_args()
    if args.token_budget is not None and args.token_budget <= 0:
        parser.error("--token-budget must be positive")
    if args.trace_rounds is not None and any(i < 0 for i in args.trace_rounds):
        parser.error("--trace-rounds must be nonnegative")
    workload = args.workload.resolve() if args.workload else None
    reference = args.reference.resolve() if args.reference else None
    for path in (workload, reference):
        if path is not None and not path.is_file():
            parser.error(f"文件不存在：{path}")
    if args.output_dir:
        output = args.output_dir.resolve()
        output.mkdir(parents=True, exist_ok=False)
    else:
        (ROOT / "output").mkdir(exist_ok=True)
        output = Path(tempfile.mkdtemp(prefix="rooflang-inspect-", dir=ROOT / "output"))
    print(f"输出目录：{output}", flush=True)
    if workload is None:
        workload = output / "workload.jsonl"
        workload.write_text(
            json.dumps(
                {
                    "request_id": 0,
                    "arrival_time": 1.0,
                    "initial": [0],
                    "sub_requests": [
                        {
                            "kind": "llm",
                            "input_tokens": 5,
                            "output_tokens": 3,
                            "next": [],
                        }
                    ],
                }
            )
            + "\n"
        )
    else:
        # Keep the exact input alongside the outputs for replay.
        (output / "workload.jsonl").write_bytes(workload.read_bytes())
        workload = output / "workload.jsonl"
    template = (
        "python-model.toml" if args.model_entry == "python" else "simthesizer.toml"
    )
    config = (ROOT / "examples/scenarios/rooflang-llama3_1_8b" / template).read_text()
    worker = [
        sys.executable,
        str(ROOT / "scripts/rooflang_worker.py"),
        "--audit-dir",
        str(output / "audit"),
    ]
    selected = args.trace_rounds
    if selected is None and args.workload:
        selected = [0, 1, 2, 100, 1000, 4000]
    if selected is not None:
        worker += ["--audit-trace-rounds", *map(str, selected)]
    config = config.replace(
        'command = ["uv", "run", "--locked", "python", "scripts/rooflang_worker.py"]',
        "command = " + json.dumps(worker),
    )
    config = config.replace("tp = 2", f"tp = {args.tp}").replace(
        "gpu_count = 2", f"gpu_count = {args.tp}"
    )
    config = config.replace('gpu_model = "rtx6000d"', f'gpu_model = "{args.gpu_model}"')
    capacity = {"rtx6000d": 84000000000, "h200": 144000000000, "b300": 288000000000}[
        args.gpu_model
    ]
    config = config.replace("device_mem = 84000000000", f"device_mem = {capacity}")
    budget = args.token_budget or (2048 if args.workload else 3)
    config = config.replace(
        "max_num_batched_tokens = 2048", f"max_num_batched_tokens = {budget}"
    )
    (output / "config.toml").write_text(config)
    started = time.perf_counter()
    run(
        [
            "cargo",
            "run",
            "--release",
            "--locked",
            "--",
            "--config",
            str(output / "config.toml"),
            "--workload",
            str(workload),
            "--output-jsonl",
            str(output / "result.jsonl"),
            "--output-graph-trace",
            str(output / "outer.jsonl"),
            "--output-engine-trace",
            str(output / "state.jsonl"),
        ],
        output / "simulation.log",
    )
    wall = time.perf_counter() - started
    samples = measurements(load_run(str(output / "result.jsonl")))
    metrics = {
        name: {"unit": metric_unit(name), **summarize(values, [50, 95, 99])}
        for name, values in samples.items()
    }
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    (output / "execution.json").write_text(
        json.dumps(
            {
                "simulation_command_wall_seconds": wall,
                "note": "Includes cargo startup/build; excludes subsequent analysis and graph reconstruction.",
            },
            indent=2,
        )
        + "\n"
    )

    def script(name, *arguments):
        run(
            [sys.executable, str(ROOT / "scripts" / name), *map(str, arguments)],
            output / (name + ".log"),
        )

    script("draw.py", output / "result.jsonl", "--output", output / "metrics.png")
    script(
        "check_rooflang_timing.py",
        "--trace",
        output / "outer.jsonl",
        "--audit-dir",
        output / "audit",
        "--output-dir",
        output / "check",
    )
    graphs = not args.workload or args.full_graphs
    if graphs:
        script(
            "draw_graph.py", output / "outer.jsonl", "--output", output / "outer.html"
        )
        script(
            "draw_rooflang.py",
            "--trace",
            output / "outer.jsonl",
            "--audit-dir",
            output / "audit",
            "--output",
            output / "full-dag.html",
        )
    if reference:
        script(
            "compare.py",
            f"real={reference}",
            f"sim={output / 'result.jsonl'}",
            "--output",
            output / "comparison.json",
        )
    links = [
        ("result.jsonl", "逐 token 时间（秒）"),
        ("metrics.json", "指标汇总"),
        ("outer.jsonl", "外层图原始事件"),
        ("state.jsonl", "系统状态"),
        ("check/timing-check.json", "时间协调检查"),
        ("check/operator-costs.csv", "算子计算 / 显存 / 通信耗时"),
        ("audit/", "批次审计和 Perfetto 时间线"),
        ("execution.json", "命令现实耗时"),
    ]
    if graphs:
        links += [("outer.html", "外层动态图"), ("full-dag.html", "完整两层图")]
    if reference:
        links.append(("comparison.json", "实测对比"))
    rows = "".join(
        f"<tr><td>{html.escape(name)}</td><td>{m['mean']}</td><td>{m['unit']}</td></tr>"
        for name, m in metrics.items()
    )
    anchors = "".join(f'<li><a href="{path}">{label}</a></li>' for path, label in links)
    (
        output / "index.html"
    ).write_text(f"""<!doctype html><html lang="zh"><meta charset="utf-8">
<title>Rooflang 仿真观测</title><style>body{{font:16px sans-serif;max-width:1100px;margin:40px auto;padding:20px}}
td,th{{padding:8px 16px;text-align:left;border-bottom:1px solid #ddd}}img{{max-width:100%}}li{{margin:10px}}</style>
<h1>Llama 3.1 8B 仿真观测</h1><p>{args.gpu_model} × {args.tp}；TP={args.tp}；模型入口：{args.model_entry}</p>
<p>指标为仿真预测。模拟时间与程序现实耗时不同。内层图通过批次重建并核对审计耗时；原始时间线位于 audit。</p>
<p>仿真命令现实耗时（含 Cargo 启动/构建，不含后续分析）：{wall:.3f} 秒。</p>
<p>{"已生成全部轮次图。" if graphs else "未生成大型 HTML 图；使用 --full-graphs 可启用。原始外层事件已保存。"}</p>
<ul>{anchors}</ul><table><tr><th>指标</th><th>均值 / 整体值</th><th>单位</th></tr>{rows}</table>
<img src="metrics.png" alt="延迟与吞吐量图"><p>时间检查仅验证当前独占资源后端的一致性，不验证真实硬件精度。</p></html>""")
    print(f"\n完成。浏览器打开：{output / 'index.html'}", flush=True)


if __name__ == "__main__":
    main()
