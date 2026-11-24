"""Render a complete shared Rooflang run without replaying simulation costs."""

import argparse
import hashlib
import json
import os
import re
import tomllib
from pathlib import Path

from draw_graph import load_trace
from rooflang_bridge.module_flow import annotate_modules


def read_rows(path):
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def work_details(node):
    return node["details"].get("work", {}).get("details", {})


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def build_view(directory):
    config = tomllib.loads((directory / "config.toml").read_text())
    system = config["system"]
    if not system.get("shared_execution"):
        raise ValueError("execution graph requires shared Rooflang execution")
    graph = load_trace(directory / "outer.jsonl")
    by_id = {n["id"]: n for n in graph["nodes"]}
    instances, devices = [], []
    for field, side in [("prefill_instances", "p"), ("decode_instances", "d")]:
        for leaf in system[field]:
            runner = leaf["model_runner"]
            hardware = runner["hardware"]
            instance = {
                "id": runner["instance_id"],
                "side": side,
                "gpu_model": hardware["gpu_model"],
                "devices": hardware["gpu_ids"],
                "tp": runner["tp"],
                "ep": runner.get("ep", 1),
                "scheduler": leaf["scheduler"],
            }
            instances.append(instance)
            for device in instance["devices"]:
                if device not in devices:
                    devices.append(device)
    instance_map = {i["id"]: i for i in instances}
    rounds, contexts = {}, {}
    for node in graph["nodes"]:
        details = work_details(node)
        if details.get("name") not in ("RooflangRoundBegin", "RooflangRoundEnd"):
            continue
        key = f"{details['instance_id']}/{details['round']}"
        row = rounds.setdefault(
            key,
            {
                "key": key,
                "instance": details["instance_id"],
                "round": details["round"],
                "side": instance_map[details["instance_id"]]["side"],
            },
        )
        if details["name"] == "RooflangRoundBegin":
            row.update(
                begin=node["id"],
                start=node["start"],
                queued=node["dispatches"][0][1],
                requests=[r["request_id"] for r in details["requests"]],
                batch=details["batch"],
            )
            contexts[node["cause"]] = key
        else:
            row.update(end=node["id"], finish=node["finish"])

    lifecycle, memory = [], []
    with (directory / "outer.jsonl").open() as stream:
        for line in stream:
            event = json.loads(line)
            if event["event"] == "kv":
                lifecycle.append(event)
            elif event["event"] == "memory_state":
                memory.append(event)
    blocks = {e["block"]["id"]: e["block"] for e in lifecycle}
    for node in graph["nodes"]:
        details = work_details(node)
        name = details.get("name") or node["details"]["label"].rsplit("::", 1)[-1]
        key = (
            f"{details['instance_id']}/{details['round']}"
            if "instance_id" in details
            else contexts.get(node["id"], contexts.get(node["cause"]))
        )
        if key:
            contexts[node["id"]] = key
        context = rounds.get(key, {})
        instance = context.get("instance")
        requests = context.get("requests", [])
        layer = re.search(r"(?:layer[ :]*)(\d+)", name)
        view = {
            "name": name,
            "category": "schedule",
            "module": context.get("side", "schedule"),
            "instance": instance,
            "round": context.get("round"),
            "layer": int(layer[1]) if layer else None,
            "requests": requests,
            "devices": details.get("devices", []),
        }
        if details.get("name") == "kv_transfer":
            view.update(
                name=f"KV L{details['layer']} {details['token_start']}:{details['token_end']}",
                category="kv",
                module="kv",
                instance=None,
                round=None,
                layer=details["layer"],
                requests=[details["request_id"]],
                devices=[details["src"], details["dst"]],
            )
        elif details.get("name", "").startswith("RooflangRound"):
            view["category"] = "boundary"
        elif node["details"]["kind"] == "resource":
            view["category"] = "tp" if details.get("network_us", 0) else "operator"
        else:
            match = re.search(r"\brequest(?:_id)?: (\d+)", name)
            if match:
                view["requests"] = [int(match[1])]
            if name.startswith("KV "):
                view["category"] = "kv_state"
                ack = re.search(r"Ack \{ source: (\d+)", name)
                if ack:
                    block = blocks[int(ack[1])]
                    view.update(
                        requests=[block["request"]],
                        layer=block["layer"],
                        module="kv",
                        instance=None,
                        round=None,
                    )
        work = node["details"].get("work", {})
        view["base_s"] = max(
            (
                d.get("latency", 0) + max(d["resources"].values(), default=0)
                for d in work.get("demands", [])
            ),
            default=0,
        )
        node["view"] = view

    # Labels without instance metadata stay unassigned. Full native payloads
    # and dependency edges remain the authority for these control nodes.
    requests = []
    for result in read_rows(directory / "result.jsonl"):
        call = result["sub_requests"][0]
        times, elapsed = [], result["arrival_time"]
        for latency in call["token_latency"]:
            elapsed += latency
            times.append(elapsed)
        requests.append(
            {
                "id": result["request_id"],
                "arrival": result["arrival_time"],
                "input": call["input_tokens"],
                "output": call["output_tokens"],
                "tokens": times,
                "finish": elapsed,
            }
        )
    clock = []
    for seq, time, _, _ in graph["changes"]:
        if clock and clock[-1][0] == time:
            clock[-1][1] = seq
        else:
            clock.append([time, seq])
    origin = min((r["arrival"] for r in requests), default=0)
    finish = max((n["finish"] for n in by_id.values()), default=origin)
    times = sorted(
        {origin, finish}
        | {t for n in by_id.values() for t in (n["start"], n["finish"]) if t >= origin}
    )
    model = dict(system["model"])
    if "num_hidden_layers" not in model:
        audit = directory / "instances" / instances[0]["id"] / "audit/init.json"
        model["num_hidden_layers"] = json.loads(audit.read_text())["model"]["layers"]
    data = {
        "nodes": graph["nodes"],
        "edges": graph["edges"],
        "clock": clock,
        "times": times,
        "instances": instances,
        "devices": devices,
        "rounds": list(rounds.values()),
        "requests": requests,
        "states": read_rows(directory / "state.jsonl"),
        "kv": lifecycle,
        "memory": memory,
        "meta": {
            "directory": str(directory.resolve()),
            "mode": system["kv_handoff"],
            "model": model,
            "origin": origin,
            "finish": finish,
            "workload": read_rows(directory / "workload.jsonl"),
        },
    }
    annotate_modules(data)
    return data


def render(directory, output):
    data = build_view(directory)
    prefix = Path(os.path.relpath(directory, output.parent)).as_posix()
    data["links"] = {
        name: f"{prefix}/{name}"
        for name in (
            "index.html",
            "outer.html",
            "outer.jsonl",
            "state.jsonl",
            "config.toml",
            "workload.jsonl",
            "result.jsonl",
            "rounds.csv",
            "operator-costs.csv",
            "kv-transfers.csv",
            "kv-lifecycle.csv",
            "memory-usage.csv",
            "shared-check.json",
        )
        if (directory / name).exists()
    }
    encoded = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace(
        "<", "\\u003c"
    )
    template = Path(__file__).with_name("rooflang_bridge") / "shared_view.html"
    with output.open("x") as stream:
        script = template.with_name("module_flow.js").read_text()
        stream.write(
            template.read_text()
            .replace("__MODULE_FLOW__", script)
            .replace("__DATA__", encoded)
        )
    receipt = {
        "nodes": len(data["nodes"]),
        "edges": len(data["edges"]),
        "rounds": len(data["rounds"]),
        "kv_events": len(data["kv"]),
        "input_sha256": {
            name: digest(directory / name)
            for name in (
                "outer.jsonl",
                "state.jsonl",
                "config.toml",
                "workload.jsonl",
                "result.jsonl",
            )
        },
    }
    output.with_suffix(".json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="shared inspection output directory")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.run / "execution-graph.html"
    receipt = render(args.run, output)
    print(f"{output}: {receipt['nodes']} nodes, {receipt['edges']} historical edges")


if __name__ == "__main__":
    main()
