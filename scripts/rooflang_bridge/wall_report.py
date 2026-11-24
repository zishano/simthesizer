"""Bounded host wall-time reports; child rows partition, never add to, their parent."""

import csv
import io
import json
import math
import time
from pathlib import Path

LABELS = {
    "prepare": "准备配置与负载",
    "simulation": "仿真命令（含启动/构建）",
    "metrics": "汇总指标与保存结果",
    "plots": "生成指标图",
    "reports": "详细报告与完整图",
    "config_load": "Rust 配置读取",
    "system_init": "实例初始化（扣除 worker 调用）",
    "workload_load": "负载读取与请求初始化",
    "engine_other": "事件队列与引擎其他工作",
    "resource_admit": "资源准入与显存分配",
    "resource_rates": "资源速率与下个完成时刻",
    "resource_advance": "资源服务推进与完成",
    "event_dispatch": "事件回调、调度与 KV 管理",
    "graph_build": "Rust 图解析、校验与依赖挂接",
    "worker_roundtrip": "worker 调用（包含下面的 Python 耗时）",
    "worker_json_decode": "Rust 解析 worker JSON",
    "result_write": "原生结果写入",
    "request_decode": "Python 解析请求",
    "worker_init": "Python 导入、硬件与模型初始化",
    "graph_build_other": "Python 建图与其他处理",
    "expert_routing": "MoE 专家路由与计数",
    "operator_pricing": "算子计算/HBM 成本计算",
    "audit_write": "Python 审计输出",
    "response_encode": "Python 响应 JSON 编码",
}


def _seconds(value):
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError("wall-time measurements must be finite and nonnegative")
    return value


def _residual(parent, children):
    rest = parent - children
    if rest < -(1e-6 + parent * 1e-6):
        raise ValueError("wall-time child measurements exceed their parent")
    return max(0.0, rest)


def build_rows(phases, native):
    """Flatten a partition tree. Percentages always use the direct parent."""
    rows = []
    total = sum(_seconds(value) for value in phases.values())

    def add(key, label, parent, seconds, denominator, calls=None):
        rows.append(
            {
                "key": key,
                "label": label,
                "parent": parent,
                "seconds": _seconds(seconds),
                "percent_of_parent": 100 * seconds / denominator
                if denominator
                else 0.0,
                "calls": calls,
            }
        )

    for phase, seconds in phases.items():
        add(f"phase:{phase}", LABELS.get(phase, phase), None, seconds, total)
        if phase != "simulation" or native is None:
            continue
        wall = _seconds(native["wall_seconds"])
        add(
            "native:total",
            "原生仿真进程（含 Python 等待）",
            "phase:simulation",
            wall,
            seconds,
        )
        sections = native.get("sections", {})
        accounted = 0.0
        for name, section in sections.items():
            elapsed = _seconds(section["exclusive_seconds"])
            accounted += elapsed
            add(
                f"native:{name}",
                LABELS.get(name, name),
                "native:total",
                elapsed,
                wall,
                section["calls"],
            )
            if name != "worker_roundtrip":
                continue
            worker_total = 0.0
            for worker_name, entry in native.get("worker_sections", {}).items():
                measured = _seconds(entry["seconds"])
                worker_total += measured
                add(
                    f"worker:{worker_name}",
                    LABELS.get(worker_name, worker_name),
                    "native:worker_roundtrip",
                    measured,
                    elapsed,
                    entry["calls"],
                )
            add(
                "worker:unattributed",
                "worker 启动、管道等待及其他未细分",
                "native:worker_roundtrip",
                _residual(elapsed, worker_total),
                elapsed,
            )
        add(
            "native:unattributed",
            "原生其他未细分与计时开销",
            "native:total",
            _residual(wall, accounted),
            wall,
        )
        add(
            "native:outside",
            "构建/启动、进程收尾与外层监控等",
            "phase:simulation",
            _residual(seconds, wall),
            seconds,
        )
    return rows


def render_table(document):
    lines = [
        f"实际耗时分解（{document['status']}，共 {document['wall_seconds']:.3f}s）",
        "阶段 | 秒 | 占父项 | 次数",
        "--- | ---: | ---: | ---:",
    ]
    depths = {}
    for row in document["rows"]:
        depth = 0 if row["parent"] is None else depths[row["parent"]] + 1
        depths[row["key"]] = depth
        lines.append(
            f"{'　' * depth}{row['label']} | {row['seconds']:.3f} | "
            f"{row['percent_of_parent']:.1f}% | {row['calls'] if row['calls'] is not None else '—'}"
        )
    lines.append(
        "子项包含在父项内，不能重复相加；计时为主机经过时间，包含等待，不是仿真时间。"
    )
    return "\n".join(lines) + "\n"


class WallReport:
    def __init__(self, clock=time.perf_counter):
        self.clock = clock
        self.enabled = False
        self.phases = {}
        self.active = None
        self.progress = None
        self.native = None
        self.status = "running"

    def enable(self, output, *, started=None):
        self.output = Path(output)
        self.enabled = True
        self.active = "prepare"
        self.phase_started = self.clock() if started is None else started

    def phase(self, name):
        if not self.enabled:
            return
        now = self.clock()
        if self.active is not None:
            self.phases[self.active] = (
                self.phases.get(self.active, 0.0) + now - self.phase_started
            )
        self.active = name
        self.phase_started = now

    def observe(self, progress):
        if not self.enabled:
            return
        self.progress = {
            key: value for key, value in progress.items() if key != "wall_profile"
        }
        if "wall_profile" in progress:
            self.native = progress["wall_profile"]
        self._write()

    def live_summary(self):
        if not self.enabled or self.native is None:
            return ""
        worker = self.native.get("worker_sections", {})
        graph = sum(
            worker.get(name, {}).get("seconds", 0)
            for name in ("graph_build_other", "expert_routing", "operator_pricing")
        )
        sections = self.native.get("sections", {})
        engine = sum(
            sections.get(name, {}).get("exclusive_seconds", 0)
            for name in (
                "engine_other",
                "resource_admit",
                "resource_rates",
                "resource_advance",
                "event_dispatch",
            )
        )
        return f" | 主机建图 {graph:.1f}s · 事件推进 {engine:.1f}s"

    def _document(self):
        phases = dict(self.phases)
        if self.active is not None:
            phases[self.active] = (
                phases.get(self.active, 0.0) + self.clock() - self.phase_started
            )
        return {
            "schema_version": 1,
            "clock": "host_wall_seconds",
            "status": self.status,
            "wall_seconds": sum(phases.values()),
            "phases": phases,
            "progress": self.progress,
            "native": self.native,
            "rows": build_rows(phases, self.native),
            "notes": [
                "Rows partition their direct parent; never sum parents and children together.",
                "Wall time includes OS scheduling and blocking; it is not CPU time or simulated GPU time.",
                "Live snapshots update on completed-request progress, not at every operator.",
                "Outside-native time includes build/startup/teardown, monitor IO and unmeasured wrapper work; it is not a measured compilation time.",
                "Worker residual includes startup, pipe waits and unmeasured work; it is not measured transport-only cost.",
                "Final report serialization/write is outside the final measured interval.",
            ],
        }

    def _write(self):
        document = self._document()
        table = render_table(document)
        stream = io.StringIO(newline="")
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "key",
                "label",
                "parent",
                "seconds",
                "percent_of_parent",
                "calls",
            ],
        )
        writer.writeheader()
        writer.writerows(document["rows"])
        for name, content in (
            (
                "wall-profile.json",
                json.dumps(document, ensure_ascii=False, indent=2) + "\n",
            ),
            ("wall-profile.csv", stream.getvalue()),
            ("wall-profile.txt", table),
        ):
            path = self.output / name
            temporary = path.with_suffix(path.suffix + ".tmp")
            temporary.write_text(content, encoding="utf-8")
            temporary.replace(path)
        return table

    def finish(self, success):
        if not self.enabled:
            return
        self.phase(None)
        path = self.output / "wall-profile-native.json"
        if path.exists():
            self.native = json.loads(path.read_text())
        self.status = "complete" if success else "failed"
        print("\n" + self._write(), flush=True)
        print(
            f"耗时报告：{self.output / 'wall-profile.txt'}（另存 CSV / JSON）",
            flush=True,
        )
