"""Check outer/inner timing alignment and export audited compute/memory/network costs.

Checks target the current exclusive-resource, single-stream Llama backend.
Detailed audits may cover a subset of rounds; all round summaries are required.
"""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

from draw_graph import load_trace
from rooflang.runtime.simulator import Simulator
from rooflang_bridge.llama import build


def close(a, b, context, *, tolerance=1e-8):
    if not math.isclose(a, b, rel_tol=1e-10, abs_tol=tolerance):
        raise ValueError(f"{context}: {a} != {b}")


def check(trace_path, audit_dir):
    outer = load_trace(trace_path)
    iterations = sorted(
        (
            n
            for n in outer["nodes"]
            if n["details"]["label"].startswith("RooflangIteration")
        ),
        key=lambda n: n["dispatches"][0][0],
    )
    config = json.loads((audit_dir / "init.json").read_text())
    paths = sorted(p for p in audit_dir.glob("[0-9]*.json") if ".trace." not in p.name)
    if len(paths) != len(iterations):
        raise ValueError("audit and outer iteration counts differ")
    rows, sampled = [], []
    previous_finish = -math.inf
    max_outer_error = 0.0
    for index, (path, node) in enumerate(zip(paths, iterations, strict=True)):
        audit = json.loads(path.read_text())
        elapsed = node["finish"] - node["start"]
        close(
            elapsed, audit["duration_us"] / 1e6, f"round {index} units", tolerance=1e-12
        )
        max_outer_error = max(
            max_outer_error, abs(elapsed - audit["duration_us"] / 1e6)
        )
        if node["start"] < previous_finish - 1e-12:
            raise ValueError("exclusive outer iterations overlap")
        previous_finish = node["finish"]
        detailed = path.with_suffix(".trace.json")
        if not detailed.exists():
            continue
        data = json.loads(detailed.read_text())
        events = [e for e in data["traceEvents"] if e.get("ph") == "X"]
        close(
            data["otherData"]["total_time_us"], audit["duration_us"], "trace duration"
        )
        close(
            max(e["ts"] + e["dur"] for e in events), audit["duration_us"], "last event"
        )
        streams = defaultdict(list)
        round_rows = []
        for e in events:
            costs = e["args"]
            ct, mt, nt = (
                costs[k]
                for k in ("compute_time_us", "memory_time_us", "network_time_us")
            )
            values = [e["ts"], e["dur"], ct, mt, nt]
            if any(not math.isfinite(v) or v < 0 for v in values):
                raise ValueError("invalid event time")
            # Current placement has one stream per device, no competing instances.
            close(e["dur"], max(ct, mt, nt), f"{e['name']} bottleneck")
            if e["tid"] != "stream0":
                raise ValueError("checker expects the exclusive stream0 backend")
            streams[e["pid"]].append(e)
            row = {
                "round": index,
                "operator": e["name"],
                "device": e["pid"],
                "start_us": e["ts"],
                "duration_us": e["dur"],
                "global_start_s": node["start"] + e["ts"] / 1e6,
                "global_finish_s": node["start"] + (e["ts"] + e["dur"]) / 1e6,
                "compute_us": ct,
                "memory_us": mt,
                "network_us": nt,
                "input_bytes": costs["input_bytes"],
                "weight_read_bytes": costs["weight_bytes"],
                "output_bytes": costs["output_bytes"],
                "bound": costs["bound"],
            }
            rows.append(row)
            round_rows.append(row)
        lane_sums = {}
        for device, lane in streams.items():
            end = 0.0
            for event in sorted(lane, key=lambda e: (e["ts"], e["dur"])):
                if event["ts"] < end - 1e-8:
                    raise ValueError(f"stream overlaps on {device}")
                end = event["ts"] + event["dur"]
            lane_sums[device] = sum(e["dur"] for e in lane)
        graph, placement, hardware, _ = build(
            audit["batch"], config["tp"], config["hardware"]
        )
        rebuilt = Simulator(graph, placement, hardware).run()
        expected = sorted(
            (
                e.kernel.name,
                e.device.name,
                e.start_us,
                e.end_us - e.start_us,
                e.kernel.flops,
                e.compute_time_us,
                e.memory_time_us,
                e.network_time_us,
            )
            for e in rebuilt.trace
        )
        actual = sorted(
            (
                e["name"],
                e["pid"],
                e["ts"],
                e["dur"],
                e["args"]["flops"],
                e["args"]["compute_time_us"],
                e["args"]["memory_time_us"],
                e["args"]["network_time_us"],
            )
            for e in events
        )
        if len(expected) != len(actual):
            raise ValueError("rebuilt and audited trace counts differ")
        for a, b in zip(expected, actual, strict=True):
            if a[:2] != b[:2]:
                raise ValueError("rebuilt and audited operator identities differ")
            for x, y in zip(a[2:], b[2:], strict=True):
                close(x, y, f"rebuilt {a[0]}")
        times = defaultdict(list)
        for e in rebuilt.trace:
            times[e.kernel].append((e.start_us, e.end_us))
        for a, b in graph._dag.edges:
            if max(t[1] for t in times[a]) > min(t[0] for t in times[b]) + 1e-8:
                raise ValueError("graph dependency violated")
        sampled.append(
            {
                "round": index,
                "batch": audit["batch"],
                "duration_us": audit["duration_us"],
                "device_events": len(events),
                "stream_busy_us": lane_sums,
                "all_device_duration_sum_us": sum(lane_sums.values()),
            }
        )
    return {
        "rounds_checked": len(iterations),
        "detailed_rounds": sampled,
        "max_outer_conversion_error_s": max_outer_error,
        "checks": [
            "outer/inner units",
            "exclusive outer order",
            "no stream overlap",
            "duration=max(compute,memory,network) in this scope",
            "fresh graph matches audited operator costs/times",
            "graph dependencies",
        ],
    }, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report, rows = check(args.trace, args.audit_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "timing-check.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    if rows:
        with (args.output_dir / "operator-costs.csv").open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
