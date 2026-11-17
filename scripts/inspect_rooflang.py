"""Run Rooflang simulation; save metrics and plots, with detailed reports on request."""

import argparse
import html
import json
import math
import subprocess
import sys
import time
import tomllib
from datetime import datetime
from pathlib import Path

from results import load_run, measurements, metric_unit, summarize
from rooflang_bridge.configuration import pd_config, toml_document, topology_pd_config
from rooflang_bridge.pd_report import export_transfers

ROOT = Path(__file__).resolve().parents[1]


def run(command, log):
    print(f"运行：{' '.join(map(str, command))}", flush=True)
    started = time.perf_counter()
    shown = False
    tty = sys.stdout.isatty()
    with (
        log.open("w") as stream,
        subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        ) as process,
    ):
        try:
            for line in process.stdout:
                stream.write(line)
                if line.startswith("SIMTHESIZER_PROGRESS "):
                    stream.flush()
                    progress = json.loads(line.removeprefix("SIMTHESIZER_PROGRESS "))
                    completed, total = progress["completed"], progress["total"]
                    percent = completed * 100 // total if total else 100
                    filled = completed * 30 // total if total else 30
                    bar = "#" * filled + "-" * (30 - filled)
                    message = (
                        f"仿真进度 [{bar}] {percent:3d}% | 已完成 {completed}/{total} 个请求"
                        f" | 已用 {time.perf_counter() - started:.1f}s"
                        f" | 仿真时间 {progress['time_s']:.3f}s"
                    )
                    print(
                        ("\r" if tty else "") + message,
                        end="" if tty else "\n",
                        flush=True,
                    )
                    shown = True
        finally:
            if shown and tty:
                print(flush=True)
        returncode = process.wait()
    if returncode:
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
    parser.add_argument("--pd", action="store_true", help="独立 GPU 的 P/D 实例池")
    parser.add_argument(
        "--execution",
        choices=["exclusive", "shared"],
        default="exclusive",
        help="shared：原生算子事件、共享资源及持久 KV",
    )
    parser.add_argument("--kv-handoff", choices=["layer", "whole"], default="layer")
    parser.add_argument(
        "--allow-gpu-sharing",
        action="store_true",
        help="shared 模式允许显式 GPU IDs 重叠，独立实例权重分别占内存",
    )
    parser.add_argument("--prefill-instances", type=int, default=1)
    parser.add_argument("--decode-instances", type=int, default=1)
    parser.add_argument(
        "--kv-link-bw",
        type=float,
        help="旧版单分片诊断模式带宽，bytes/s；默认从硬件图导出",
    )
    parser.add_argument(
        "--kv-link-latency", type=float, help="旧版单分片诊断模式延迟，秒"
    )
    parser.add_argument(
        "--hardware-topology", type=Path, help="共享硬件图 JSON；异型号须提供显式连接"
    )
    for side in ("prefill", "decode"):
        parser.add_argument(f"--{side}-gpu-model", choices=["rtx6000d", "h200", "b300"])
        parser.add_argument(
            f"--{side}-gpu-count", type=int, help="整个池的 GPU 数，须等于实例数 × TP"
        )
        parser.add_argument(
            f"--{side}-gpu-ids",
            nargs="+",
            help="硬件图中的 GPU 名字，按实例和 TP rank 排列",
        )
        parser.add_argument(f"--{side}-tp", type=int, choices=[1, 2, 4, 8])
        parser.add_argument(
            f"--{side}-pp", type=int, choices=[1], default=1, help="当前仅支持 PP=1"
        )
        parser.add_argument(
            f"--{side}-ep", type=int, choices=[1], default=1, help="当前仅支持 EP=1"
        )
        parser.add_argument(f"--{side}-max-batch", type=int)
        parser.add_argument(f"--{side}-token-budget", type=int)
    parser.add_argument("--model-entry", choices=["rust", "python"], default="rust")
    parser.add_argument(
        "--token-budget", type=int, help="默认：演示 3；外部工作负载 2048"
    )
    parser.add_argument(
        "--reports",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="额外保存历史事件、轮次审计和 CSV；默认保存指标、指标图与运行配置",
    )
    parser.add_argument(
        "--full-graphs",
        action="store_true",
        help="启用 --reports，并额外生成完整 HTML 图，可能很大",
    )
    parser.add_argument(
        "--reference-every-round",
        action="store_true",
        help="诊断：shared 每轮执行 Python 独占预演，默认复用执行计划",
    )
    parser.add_argument(
        "--trace-rounds",
        type=int,
        nargs="+",
        help="启用报告时，只导出这些轮次的详细时间线（从 0 开始）",
    )
    parser.add_argument(
        "--reference", type=Path, help="可选：匹配工作负载的真实结果 JSONL"
    )
    args = parser.parse_args()
    reports = args.reports or args.full_graphs
    if args.prefill_instances < 1 or args.decode_instances < 1:
        parser.error("P/D 实例数必须为正整数")
    manual_link = args.kv_link_bw is not None or args.kv_link_latency is not None
    if args.execution == "shared" and (not args.pd or manual_link):
        parser.error("shared 执行需要 --pd 和硬件图拓扑")
    if args.allow_gpu_sharing and args.execution != "shared":
        parser.error("GPU 共享需要 --execution shared")
    if (
        args.pd
        and manual_link
        and (
            args.kv_link_bw is None
            or not math.isfinite(args.kv_link_bw)
            or args.kv_link_bw <= 0
            or args.kv_link_latency is None
            or not math.isfinite(args.kv_link_latency)
            or args.kv_link_latency < 0
        )
    ):
        parser.error(
            "手动链路需要同时指定正数 --kv-link-bw 和非负 --kv-link-latency（均须有限）"
        )
    if not args.pd and (
        args.kv_link_bw is not None
        or args.kv_link_latency is not None
        or args.prefill_instances != 1
        or args.decode_instances != 1
    ):
        parser.error("P/D 参数需要同时指定 --pd")
    separate = args.hardware_topology is not None or any(
        getattr(args, f"{side}_{field}") is not None
        for side in ("prefill", "decode")
        for field in (
            "gpu_model",
            "gpu_count",
            "gpu_ids",
            "tp",
            "max_batch",
            "token_budget",
        )
    )
    if separate and (not args.pd or manual_link):
        parser.error("独立 P/D 参数需要 --pd，且不能使用旧版手动 KV 链路")
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
        name = f"rooflang-inspect-{datetime.now():%Y%m%d-%H%M%S}"
        suffix = 0
        while True:
            output = ROOT / "output" / (name if suffix == 0 else f"{name}-{suffix}")
            try:
                output.mkdir(exist_ok=False)
                break
            except FileExistsError:
                suffix += 1
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
    ]
    if reports:
        worker += ["--audit-dir", str(output / "audit")]
    if args.reference_every_round:
        worker += ["--reference-every-round"]
    selected = args.trace_rounds
    if selected is None and args.workload:
        selected = [0, 1, 2, 100, 1000, 4000]
    if reports and selected is not None:
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
    instances = [("", output / "audit", output)]
    if args.pd:
        leaf = tomllib.loads(config)["system"]
        if manual_link:
            document, instances = pd_config(
                leaf,
                output,
                args.prefill_instances,
                args.decode_instances,
                args.kv_link_bw,
                args.kv_link_latency,
            )
        else:
            pools = []
            for side in ("prefill", "decode"):
                pool = {
                    "gpu_model": getattr(args, f"{side}_gpu_model") or args.gpu_model,
                    "instances": getattr(args, f"{side}_instances"),
                    "tp": getattr(args, f"{side}_tp") or args.tp,
                    "pp": getattr(args, f"{side}_pp"),
                    "ep": getattr(args, f"{side}_ep"),
                }
                for argument, key in [
                    ("gpu_count", "gpu_count"),
                    ("gpu_ids", "gpu_ids"),
                    ("max_batch", "max_batch"),
                    ("token_budget", "max_num_batched_tokens"),
                ]:
                    value = getattr(args, f"{side}_{argument}")
                    if value is not None:
                        pool[key] = value
                pools.append(pool)
            try:
                topology = (
                    json.loads(args.hardware_topology.read_text())
                    if args.hardware_topology
                    else None
                )
                document, instances = topology_pd_config(
                    leaf,
                    output,
                    *pools,
                    topology=topology,
                    shared_execution=args.execution == "shared",
                    kv_handoff=args.kv_handoff,
                    allow_gpu_sharing=args.allow_gpu_sharing,
                )
            except (ValueError, KeyError, OSError) as error:
                parser.error(str(error))
        config = toml_document(document)
    (output / "config.toml").write_text(config)
    started = time.perf_counter()
    print("仿真准备中（编译和实例初始化），随后显示完整请求完成进度……", flush=True)
    command = [
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
        "--progress",
    ]
    if reports:
        command += [
            "--output-graph-trace",
            str(output / "outer.jsonl"),
            "--output-engine-trace",
            str(output / "state.jsonl"),
        ]
    run(command, output / "simulation.log")
    wall = time.perf_counter() - started
    print("请求仿真完成，正在汇总指标……", flush=True)
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
                "reports": reports,
                "full_graphs": args.full_graphs,
                "reference_every_round": args.reference_every_round,
                "note": "Includes cargo startup/build; excludes subsequent analysis and graph reconstruction.",
            },
            indent=2,
        )
        + "\n"
    )

    def script(name, *arguments, tag=""):
        run(
            [sys.executable, str(ROOT / "scripts" / name), *map(str, arguments)],
            output / (tag + name + ".log"),
        )

    script("draw.py", output / "result.jsonl", "--output", output / "metrics.png")
    if not reports:
        if reference:
            script(
                "compare.py",
                f"real={reference}",
                f"sim={output / 'result.jsonl'}",
                "--output",
                output / "comparison.json",
            )
        print(
            f"\n完成。指标图：{output / 'metrics.png'}\n指标：{output / 'metrics.json'}\n逐 token 结果：{output / 'result.jsonl'}",
            flush=True,
        )
        return
    graphs = args.full_graphs
    instance_links = []
    kv_report = ""
    if args.execution == "shared":
        from rooflang_bridge.shared_report import export_shared

        instance_links, kv_report = export_shared(output, instances)
        (output / "hardware-topology.json").write_text(
            json.dumps(
                document["system"]["prefill_instances"][0]["model_runner"]["hardware"][
                    "topology"
                ],
                indent=2,
            )
        )
        (output / "kv-topology.json").write_text(
            json.dumps(document["system"]["kv_topology"], indent=2)
        )
        instance_links += [
            ("hardware-topology.json", "共享硬件图"),
            ("kv-topology.json", "KV 路由与物理资源 ID"),
        ]
    for identity, audit, directory in [] if args.execution == "shared" else instances:
        script(
            "check_rooflang_timing.py",
            "--trace",
            output / "outer.jsonl",
            "--audit-dir",
            audit,
            "--output-dir",
            directory / "check",
            tag=identity,
        )
        report = json.loads((directory / "check/timing-check.json").read_text())
        prefix = directory.relative_to(output).as_posix()
        for path, label in [
            ("check/timing-check.json", "时间协调检查"),
            ("audit/", "批次审计和 Perfetto 时间线"),
        ]:
            instance_links.append(
                (f"{prefix}/{path}", f"{identity or '实例'} · {label}")
            )
        if (directory / "check/operator-costs.csv").exists():
            instance_links.append(
                (
                    f"{prefix}/check/operator-costs.csv",
                    f"{identity or '实例'} · 算子成本 CSV",
                )
            )
        if graphs and report["rounds_checked"]:
            script(
                "draw_rooflang.py",
                "--trace",
                output / "outer.jsonl",
                "--audit-dir",
                audit,
                "--output",
                directory / "full-dag.html",
                tag=identity,
            )
            instance_links.append(
                (f"{prefix}/full-dag.html", f"{identity or '实例'} · 完整两层图")
            )
    if args.pd and args.execution != "shared":
        kv_report = export_transfers(output, document["system"])
        instance_links.append(("kv-transfers.csv", "P→D KV 分片与路径、时间"))
        if not manual_link:
            instance_links += [
                ("kv-handoffs.csv", "每请求 KV 等待与 decode 释放时间"),
                ("hardware-topology.json", "共享硬件图配置"),
                ("kv-topology.json", "KV 设备、链路和路由"),
            ]
    if graphs:
        script(
            "draw_graph.py", output / "outer.jsonl", "--output", output / "outer.html"
        )
        if args.execution == "shared":
            script("draw_shared.py", output)
            instance_links.insert(
                0,
                ("execution-graph.html", "完整调度与计算图（时间线 / KV / 依赖回放）"),
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
        ("execution.json", "命令现实耗时"),
    ] + instance_links
    if graphs:
        links += [
            (
                "outer.html",
                "原始动态图（全部节点与依赖历史）"
                if args.execution == "shared"
                else "外层动态图",
            )
        ]
    if reference:
        links.append(("comparison.json", "实测对比"))
    rows = "".join(
        f"<tr><td>{html.escape(name)}</td><td>{m['mean']}</td><td>{m['unit']}</td></tr>"
        for name, m in metrics.items()
    )
    anchors = "".join(f'<li><a href="{path}">{label}</a></li>' for path, label in links)
    deployment = f"{args.gpu_model}，单实例 TP={args.tp}"
    if args.pd:
        descriptions = []
        for side, title in [("prefill_instances", "P"), ("decode_instances", "D")]:
            leaves = document["system"][side]
            runner, scheduler = leaves[0]["model_runner"], leaves[0]["scheduler"]
            descriptions.append(
                f"{title}：{runner['hardware']['gpu_model']}，{len(leaves)} 个实例，"
                f"TP={runner['tp']}，PP=1，EP=1，共 {len(leaves) * runner['tp']} 张 GPU，"
                f"max_batch={scheduler['max_batch']}，token_budget={scheduler['max_num_batched_tokens']}"
            )
        deployment = "；".join(descriptions)
        deployment += (
            "。KV：旧版单分片手动链路。"
            if manual_link
            else "。KV：硬件图路径，多分片共享链路。"
        )
    (
        output / "index.html"
    ).write_text(f"""<!doctype html><html lang="zh"><head><meta charset="utf-8">
<title>Rooflang 仿真观测</title><style>body{{font:16px sans-serif;max-width:1100px;margin:40px auto;padding:20px}}
td,th{{padding:8px 16px;text-align:left;border-bottom:1px solid #ddd}}img{{max-width:100%}}li{{margin:10px}}</style></head><body>
<h1>Llama 3.1 8B 仿真观测</h1><p>{deployment} 模型入口：{args.model_entry}</p>
<p>指标为仿真预测。模拟时间与程序现实耗时不同。{"共享模式直接记录原生算子与 KV 事件；详细时间线采用全局时钟。KV 交接模式：" + args.kv_handoff if args.execution == "shared" else "内层图通过批次重建并核对审计耗时；原始时间线位于 audit。"}</p>
<p>仿真命令现实耗时（含 Cargo 启动/构建，不含后续分析）：{wall:.3f} 秒。</p>
<p>{"已生成全部轮次图。" if graphs else "未生成大型 HTML 图；使用 --full-graphs 可启用。原始外层事件已保存。"}</p>
<ul>{anchors}</ul>{kv_report}<table><tr><th>指标</th><th>均值 / 整体值</th><th>单位</th></tr>{rows}</table>
<img src="metrics.png" alt="延迟与吞吐量图"><p>{"共享模式检查依赖、资源、KV 生命周期一致性；TP 与 KV 使用相同物理链路及 HBM 资源。" if args.execution == "shared" else "时间检查仅验证当前独占资源后端的一致性；KV 共享链路竞争不包含算子内部 TP 通信。"}这些检查不验证真实硬件精度。</p></body></html>""")
    print(f"\n完成。浏览器打开：{output / 'index.html'}", flush=True)


if __name__ == "__main__":
    main()
