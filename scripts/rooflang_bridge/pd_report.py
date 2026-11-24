"""Join native KV events to hardware paths without reconstructing timing."""

import csv
import html
import json
import re

from draw_graph import load_trace


def export_transfers(output, system):
    outer = load_trace(output / "outer.jsonl")
    nodes = {n["id"]: n for n in outer["nodes"]}
    incoming = {}
    for edge in outer["edges"]:
        incoming.setdefault(edge["child"], []).append(nodes[edge["parent"]])
    topology = system.get("kv_topology")
    routes = {(r["src"], r["dst"]): r for r in topology["routes"]} if topology else {}
    owners = {}
    if topology:
        for side, prefix in [("prefill", "p"), ("decode", "d")]:
            replicas = topology.get(f"{side}_gpus")
            if replicas is None:
                replicas = [
                    [gpu for group in groups for gpu in group]
                    for groups in topology[f"{side}_groups"]
                ]
            for index, group in enumerate(replicas):
                owners.update({gpu: f"{prefix}{index}" for gpu in group})
        (output / "kv-topology.json").write_text(json.dumps(topology, indent=2) + "\n")
        spec = system["prefill_instances"][0]["model_runner"]["hardware"]["topology"]
        (output / "hardware-topology.json").write_text(
            json.dumps(spec, indent=2) + "\n"
        )
    rows = []
    for node in nodes.values():
        if node["details"]["kind"] != "network":
            continue
        detail = node["details"]
        src, dst = detail["src"], detail["dst"]
        start, finish = node["dispatches"][0][1], node["finish"]
        row = dict(
            node=node["id"],
            src=src,
            dst=dst,
            bytes=detail["bytes"],
            start_s=start,
            finish_s=finish,
            duration_s=finish - start,
        )
        if topology:
            marker = next(
                n for n in incoming[node["id"]] if "::KvSlice" in n["details"]["label"]
            )
            fields = dict(
                re.findall(
                    r"(request_id|head_start|head_end): (\d+)",
                    marker["details"]["label"],
                )
            )
            path = routes[src, dst]["links"]
            links = [topology["links"][i] for i in path]
            row.update(
                **{k: int(v) for k, v in fields.items()},
                source_instance=owners[src],
                target_instance=owners[dst],
                source_gpu=topology["devices"][src],
                target_gpu=topology["devices"][dst],
                path=" -> ".join(
                    topology["devices"][i] for i in [src] + [e["dst"] for e in links]
                ),
                link_ids=" ".join(map(str, path)),
                bottleneck_bytes_s=min(e["bandwidth"] for e in links),
                path_latency_s=sum(e["latency"] for e in links),
            )
        rows.append(row)
    columns = ["node", "src", "dst", "bytes", "start_s", "finish_s", "duration_s"]
    if topology:
        columns += [
            "request_id",
            "head_start",
            "head_end",
            "source_instance",
            "target_instance",
            "source_gpu",
            "target_gpu",
            "path",
            "link_ids",
            "bottleneck_bytes_s",
            "path_latency_s",
        ]
    with (output / "kv-transfers.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    handoffs = []
    if topology:
        by_request = {}
        for row in rows:
            by_request.setdefault(row["request_id"], []).append(row)
        for request_id, slices in sorted(by_request.items()):
            start = min(r["start_s"] for r in slices)
            end = max(r["finish_s"] for r in slices)
            handoffs.append(
                dict(
                    request_id=request_id,
                    source_instance=slices[0]["source_instance"],
                    target_instance=slices[0]["target_instance"],
                    slices=len(slices),
                    bytes=sum(r["bytes"] for r in slices),
                    start_s=start,
                    decode_release_s=end,
                    kv_wait_s=end - start,
                )
            )
        with (output / "kv-handoffs.csv").open("w") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=[
                    "request_id",
                    "source_instance",
                    "target_instance",
                    "slices",
                    "bytes",
                    "start_s",
                    "decode_release_s",
                    "kv_wait_s",
                ],
            )
            writer.writeheader()
            writer.writerows(handoffs)

    def table(records, fields):
        return (
            '<div style="overflow-x:auto"><table><tr>'
            + "".join(f"<th>{html.escape(k)}</th>" for k in fields)
            + "</tr>"
            + "".join(
                "<tr>"
                + "".join(f"<td>{html.escape(str(row[k]))}</td>" for k in fields)
                + "</tr>"
                for row in records[:200]
            )
            + "</table></div>"
        )

    allocations = []
    for pool in ("prefill_instances", "decode_instances"):
        for leaf in system[pool]:
            runner, scheduler = leaf["model_runner"], leaf["scheduler"]
            allocations.append(
                dict(
                    instance=runner["instance_id"],
                    gpu_model=runner["hardware"]["gpu_model"],
                    tp=runner["tp"],
                    gpu_ids=", ".join(runner["hardware"].get("gpu_ids", [])),
                    max_batch=scheduler["max_batch"],
                    token_budget=scheduler["max_num_batched_tokens"],
                )
            )
    return (
        "<h2>P/D 设备分配</h2>"
        + table(
            allocations,
            ["instance", "gpu_model", "tp", "gpu_ids", "max_batch", "token_budget"],
        )
        + "<h2>P→D KV 交接</h2><p>时间单位：秒；head 范围为左闭右开。表格最多显示 200 条，完整记录见 CSV。KV 等待不含 D 排队和计算。</p>"
        + (
            table(
                handoffs,
                [
                    "request_id",
                    "source_instance",
                    "target_instance",
                    "bytes",
                    "kv_wait_s",
                    "decode_release_s",
                ],
            )
            if topology
            else ""
        )
        + table(
            rows,
            (
                [
                    "request_id",
                    "source_gpu",
                    "target_gpu",
                    "head_start",
                    "head_end",
                    "bytes",
                    "start_s",
                    "finish_s",
                    "path",
                ]
                if topology
                else columns
            ),
        )
    )
