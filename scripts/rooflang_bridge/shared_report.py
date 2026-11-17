"""Stream native shared execution reports, retaining only unfinished state."""

import csv
import json
import math
import tempfile
from collections import Counter
from contextlib import ExitStack
from pathlib import Path


class CsvRows:
    def __init__(self, stream):
        self.stream, self.writer = stream, None

    def write(self, row):
        if self.writer is None:
            self.writer = csv.DictWriter(self.stream, fieldnames=list(row))
            self.writer.writeheader()
        self.writer.writerow(row)


class RoundTrace:
    def __init__(self, path):
        self.path = path
        self.stream = path.open("w")
        self.stream.write('{"traceEvents":[')
        self.first = True

    def write(self, op):
        for device in op["devices"].split(";"):
            if not self.first:
                self.stream.write(",")
            self.first = False
            json.dump(
                {
                    "ph": "X",
                    "name": op["operator"],
                    "pid": device,
                    "tid": f"{op['instance_id']}/stream0",
                    "ts": op["start_s"] * 1e6,
                    "dur": op["duration_s"] * 1e6,
                    "args": op,
                },
                self.stream,
            )

    def close(self):
        if not self.stream.closed:
            self.stream.write(
                '],"otherData":{"clock":"global microseconds","execution":"shared"}}'
            )
            self.stream.close()


def export_shared(output, instances):
    audits = {identity: audit for identity, audit, _ in instances}
    nodes, rounds, streams = {}, {}, {}
    live, pending, prefill_done = {}, {}, set()
    counts = Counter()
    last_seq = last_node = -1
    complete = False
    # Publish CSVs only after the complete trace passes validation.
    with tempfile.TemporaryDirectory(prefix=".shared-report-", dir=output) as directory:
        staging = Path(directory)
        with ExitStack() as files:
            active_traces = set()
            files.callback(lambda: [trace.close() for trace in active_traces])
            tables = {
                name: CsvRows(
                    files.enter_context((staging / f"{name}.csv").open("w", newline=""))
                )
                for name in (
                    "operator-costs",
                    "kv-transfers",
                    "kv-lifecycle",
                    "memory-usage",
                    "rounds",
                )
            }
            with (output / "outer.jsonl").open() as source:
                for line in source:
                    event = json.loads(line)
                    seq, kind = event["seq"], event["event"]
                    if seq <= last_seq or complete:
                        raise ValueError("invalid trace sequence")
                    if last_seq == -1 and (
                        seq != 0 or kind != "header" or event.get("version") != 1
                    ):
                        raise ValueError("expected graph trace version 1")
                    last_seq = seq
                    if kind == "node_create":
                        identity = event["node"]
                        if identity <= last_node:
                            raise ValueError("duplicate or non-monotone node ID")
                        last_node = identity
                        nodes[identity] = {
                            "work": event["details"].get("work"),
                            "children": Counter(),
                            "indegree": 0,
                            "required_start": 0,
                            "queued": None,
                        }
                        counts["peak_live_nodes"] = max(
                            counts["peak_live_nodes"], len(nodes)
                        )
                    elif kind in ("edge_add", "edge_remove"):
                        a, b = event["parent"], event["child"]
                        if a not in nodes or b not in nodes:
                            raise ValueError("edge references unknown node")
                        delta = 1 if kind == "edge_add" else -1
                        if delta < 0 and not nodes[a]["children"][b]:
                            raise ValueError("removing unknown edge")
                        nodes[a]["children"][b] += delta
                        nodes[b]["indegree"] += delta
                        if not nodes[a]["children"][b]:
                            del nodes[a]["children"][b]
                    elif kind == "dispatch":
                        node = nodes[event["node"]]
                        if node["queued"] is None:
                            node["queued"] = event["time_s"]
                    elif kind == "node_complete":
                        identity = event["node"]
                        node = nodes.pop(identity)
                        start, finish = event["start_s"], event["finish_s"]
                        if (
                            not all(
                                math.isfinite(t) and t >= 0 for t in (start, finish)
                            )
                            or finish < start
                        ):
                            raise ValueError("invalid shared resource time")
                        if node["indegree"] or start + 1e-12 < node["required_start"]:
                            raise ValueError(f"dependency violated: {identity}")
                        for child, multiplicity in node["children"].items():
                            nodes[child]["indegree"] -= multiplicity
                            nodes[child]["required_start"] = max(
                                nodes[child]["required_start"], finish
                            )
                        work = node["work"]
                        if work is None:
                            continue
                        base = max(
                            (
                                d.get("latency", 0)
                                + max(d["resources"].values(), default=0)
                                for d in work["demands"]
                            ),
                            default=0,
                        )
                        if finish - start + 1e-12 < base:
                            raise ValueError(
                                f"resource service shorter than its isolated bound: {identity}"
                            )
                        for slot in work["slots"]:
                            if start < streams.get(slot, 0) - 1e-12:
                                raise ValueError(f"stream overlap: {slot}")
                            streams[slot] = max(streams.get(slot, 0), finish)
                        d = work["details"]
                        name = d["name"]
                        if name == "kv_transfer":
                            tables["kv-transfers"].write(
                                {
                                    "node": identity,
                                    **d,
                                    "start_s": start,
                                    "finish_s": finish,
                                    "duration_s": finish - start,
                                    "resources": json.dumps(
                                        work["demands"][0]["resources"], sort_keys=True
                                    ),
                                }
                            )
                            counts["kv_copies"] += 1
                            continue
                        key = (d["instance_id"], d["round"])
                        if name == "RooflangRoundBegin":
                            audit = audits[key[0]]
                            trace = None
                            if (audit / f"{key[1]:05d}.plan.json").exists():
                                trace = RoundTrace(
                                    staging / f"{key[0]}-{key[1]}.trace.json"
                                )
                                active_traces.add(trace)
                            rounds[key] = {
                                "start": start,
                                "queued": node["queued"],
                                "trace": trace,
                            }
                        elif name == "RooflangRoundEnd":
                            r = rounds.pop(key)
                            audit = audits[key[0]]
                            saved_path = audit / f"{key[1]:05d}.json"
                            saved = json.loads(saved_path.read_text())
                            saved.update(
                                duration_us=(finish - r["start"]) * 1e6,
                                global_start_s=r["start"],
                                global_finish_s=finish,
                                admission_wait_s=r["start"] - r["queued"],
                            )
                            saved_path.write_text(json.dumps(saved, indent=2))
                            isolated = (saved.get("isolated_reference") or {}).get(
                                "duration_us"
                            )
                            tables["rounds"].write(
                                {
                                    "instance_id": key[0],
                                    "round": key[1],
                                    "start_s": r["start"],
                                    "finish_s": finish,
                                    "admission_wait_s": r["start"] - r["queued"],
                                    "duration_s": finish - r["start"],
                                    "isolated_reference_s": isolated * 1e-6
                                    if isolated is not None
                                    else None,
                                }
                            )
                            if r["trace"]:
                                r["trace"].close()
                                active_traces.remove(r["trace"])
                                r["trace"].path.replace(
                                    audit / f"{key[1]:05d}.trace.json"
                                )
                            counts["rounds_checked"] += 1
                        else:
                            op = {
                                "node": identity,
                                "instance_id": key[0],
                                "round": key[1],
                                "operator": name,
                                "devices": ";".join(d["devices"]),
                                "start_s": start,
                                "finish_s": finish,
                                "duration_s": finish - start,
                                "compute_us": d["compute_us"],
                                "memory_us": d["memory_us"],
                                "network_us": d["network_us"],
                                "resources": json.dumps(
                                    work["demands"], sort_keys=True
                                ),
                            }
                            tables["operator-costs"].write(op)
                            counts["operators"] += 1
                            if rounds[key]["trace"]:
                                rounds[key]["trace"].write(op)
                    elif kind == "memory_state":
                        state = event["state"]
                        for device, capacity in state["capacity_bytes"].items():
                            groups = {"weights": 0, "kv": 0, "workspace": 0}
                            for allocation, value in state["allocations"].items():
                                groups[allocation.split(":", 1)[0]] += value[
                                    "bytes"
                                ].get(device, 0)
                            reserved = sum(groups.values())
                            if reserved > capacity:
                                raise ValueError("physical memory capacity exceeded")
                            tables["memory-usage"].write(
                                {
                                    "time_s": event["time_s"],
                                    "device": device,
                                    "capacity_bytes": capacity,
                                    "reserved_bytes": reserved,
                                    **groups,
                                }
                            )
                    elif kind == "kv_prefill_done":
                        prefill_done.add(event["request_id"])
                    elif kind == "kv_reservation_release":
                        if event["allocation"].endswith(":p"):
                            prefill_done.discard(int(event["allocation"].split(":")[1]))
                    elif kind == "kv":
                        b, transition = event["block"], event["transition"]
                        bid, src = b["id"], b["source"]
                        tables["kv-lifecycle"].write(
                            {"time_s": event["time_s"], "transition": transition, **b}
                        )
                        if transition in ("produce", "send"):
                            if bid in live:
                                raise ValueError("duplicate KV allocation")
                            live[bid] = b
                        if transition == "send":
                            if src not in live:
                                raise ValueError("copy from released KV")
                            pending.setdefault(src, set()).add(bid)
                            counts["kv_sent_bytes"] += b["bytes"]
                        elif transition == "receive":
                            if src not in live or bid not in pending.get(src, ()):
                                raise ValueError("invalid KV receive")
                            pending[src].remove(bid)
                            if not pending[src]:
                                del pending[src]
                            live[bid] = b
                            counts["kv_received_bytes"] += b["bytes"]
                        elif transition == "extend":
                            previous = live.get(bid)
                            if (
                                previous is None
                                or src is not None
                                or b["prefill"]
                                or b["state"] != "ready"
                                or b["token_end"] <= previous["token_end"]
                                or b["bytes"] <= previous["bytes"]
                                or any(
                                    b[k] != previous[k]
                                    for k in (
                                        "request",
                                        "layer",
                                        "device",
                                        "token_start",
                                        "head_start",
                                        "head_end",
                                    )
                                )
                            ):
                                raise ValueError("invalid KV append")
                            live[bid] = b
                        elif transition == "release":
                            if bid not in live or pending.get(bid):
                                raise ValueError(
                                    "KV released before copy acknowledgement"
                                )
                            if b["prefill"] and b["request"] not in prefill_done:
                                raise ValueError(
                                    "KV released before producer completion"
                                )
                            del live[bid]
                        counts["peak_live_kv_blocks"] = max(
                            counts["peak_live_kv_blocks"], len(live)
                        )
                    elif kind == "run_complete":
                        complete = True
            if not complete or nodes or rounds:
                raise ValueError(
                    "trace has no successful run_complete or contains unresolved nodes/rounds"
                )
            if (
                live
                or pending
                or counts["kv_sent_bytes"] != counts["kv_received_bytes"]
            ):
                raise ValueError(
                    "KV lifecycle did not drain or transferred bytes differ"
                )
        for path in staging.glob("*.csv"):
            path.replace(output / path.name)

    report = {
        "execution": "shared",
        **{
            k: counts[k]
            for k in (
                "rounds_checked",
                "operators",
                "kv_copies",
                "kv_sent_bytes",
                "kv_received_bytes",
            )
        },
        "live_kv_after_run": 0,
        "checks": [
            "native graph dependencies",
            "resource lower bounds",
            "stream exclusion",
            "KV source lifetime",
            "KV transfer byte conservation",
            "KV release/drain",
        ],
        "streaming": {k: counts[k] for k in ("peak_live_nodes", "peak_live_kv_blocks")},
        "limits": "Analytical resource model; not a measured accuracy claim. Layer reads also checked during native execution.",
    }
    (output / "shared-check.json").write_text(json.dumps(report, indent=2))
    links = [
        ("shared-check.json", "共享执行一致性检查"),
        ("rounds.csv", "所有轮次与显存准入等待"),
        ("operator-costs.csv", "算子实际开始、结束及成本"),
        ("kv-transfers.csv", "KV 传输及共享资源"),
        ("kv-lifecycle.csv", "KV 生成、传输、接收与释放"),
        ("memory-usage.csv", "物理显存：权重、KV 预留与工作区"),
    ]
    return links, (
        f"<p>共享资源执行：{counts['operators']} 个算子，{counts['kv_copies']} 次 KV 分片传输；"
        "运行结束后活动 KV 为 0。详细轮次的 Perfetto 时间线位于各实例 audit 目录。</p>"
    )
