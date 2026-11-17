"""Compose independent Rooflang leaves using the native P/D system schema."""

import copy
import json


def toml_document(config):
    lines = []

    def emit(table, path, array=False):
        if path:
            brackets = ("[[", "]]") if array else ("[", "]")
            lines.append(brackets[0] + path + brackets[1])
        for key, value in table.items():
            if not isinstance(value, dict) and not (
                isinstance(value, list) and value and isinstance(value[0], dict)
            ):
                lines.append(f"{key} = {json.dumps(value)}")
        lines.append("")
        for key, value in table.items():
            child = f"{path}.{key}" if path else key
            if isinstance(value, dict):
                emit(value, child)
            elif isinstance(value, list) and value and isinstance(value[0], dict):
                for item in value:
                    emit(item, child, True)

    emit(config, "")
    return "\n".join(lines)


def pd_config(leaf, output, prefill_count, decode_count, bandwidth, latency):
    runner = leaf["model_runner"]
    parent = {
        "kind": "pd_disaggregation",
        "tp": runner["tp"],
        "link_bw": bandwidth,
        "link_latency": latency,
        "network_sim": {"kind": "flow"},
        "model": copy.deepcopy(leaf["model"]),
    }
    instances = []
    for pool, prefix, count in [
        ("prefill_instances", "p", prefill_count),
        ("decode_instances", "d", decode_count),
    ]:
        parent[pool] = []
        for index in range(count):
            identity = f"{prefix}{index}"
            directory = output / "instances" / identity
            child = copy.deepcopy(leaf)
            child_runner = child["model_runner"]
            child_runner["instance_id"] = identity
            command = child_runner["command"]
            if "--audit-dir" in command:
                command[command.index("--audit-dir") + 1] = str(directory / "audit")
            parent[pool].append(child)
            instances.append((identity, directory / "audit", directory))
    return {"system": parent}, instances


def topology_pd_config(
    leaf,
    output,
    prefill,
    decode,
    topology=None,
    *,
    shared_execution=False,
    kv_handoff="layer",
    allow_gpu_sharing=False,
):
    """Compose independent pools; counts refer to total pool GPUs, not per replica."""
    from rooflang_bridge.llama import KVH
    from rooflang_bridge.topology import (
        build_topology,
        default_topology,
        device_order,
        kv_topology,
    )

    if allow_gpu_sharing and not shared_execution:
        raise ValueError("GPU sharing requires shared execution")
    pools = [prefill, decode]
    for pool in pools:
        if pool["tp"] not in (1, 2, 4, 8) or pool["instances"] < 1:
            raise ValueError("P/D requires positive instances and TP=1/2/4/8")
        if pool.get("pp", 1) != 1 or pool.get("ep", 1) != 1:
            raise ValueError("Rooflang currently supports PP=1 and EP=1 only")
        expected = pool["instances"] * pool["tp"]
        if pool.get("gpu_count", expected) != expected:
            raise ValueError("pool gpu_count must equal instances * TP (PP=1)")
    if topology is None:
        if prefill["gpu_model"] != decode["gpu_model"]:
            raise ValueError(
                "different GPU models require --hardware-topology with explicit bridge links"
            )
        topology = default_topology(
            prefill["gpu_model"], sum(p["instances"] * p["tp"] for p in pools)
        )
    hw = build_topology(topology)
    available = sorted((d for d in hw.nodes if d.kind == "gpu"), key=device_order)
    used, groups, instances = set(), [], []
    parent = {
        "kind": "pd_disaggregation",
        "shared_execution": shared_execution,
        "allow_gpu_sharing": allow_gpu_sharing,
        "kv_handoff": kv_handoff,
        "tp": 1,
        "model": copy.deepcopy(leaf["model"]),
        "network_sim": {"kind": "flow"},
    }
    for key, prefix, pool in zip(
        ["prefill_instances", "decode_instances"], ["p", "d"], pools
    ):
        count = pool["instances"] * pool["tp"]
        names = pool.get("gpu_ids")
        if names is None:
            names = [
                d.name
                for d in available
                if d.name not in used and hw.models[d] == pool["gpu_model"]
            ][:count]
        if len(names) != count or (
            not allow_gpu_sharing
            and (len(set(names)) != count or used.intersection(names))
        ):
            raise ValueError("pool requires enough disjoint GPU IDs matching its count")
        lookup = {d.name: d for d in available}
        if any(
            n not in lookup or hw.models[lookup[n]] != pool["gpu_model"] for n in names
        ):
            raise ValueError("pool GPU IDs do not match GPU model")
        used.update(names)
        parent[key], allocated = [], []
        for index in range(pool["instances"]):
            identity = f"{prefix}{index}"
            directory = output / "instances" / identity
            child = copy.deepcopy(leaf)
            runner = child["model_runner"]
            ids = names[index * pool["tp"] : (index + 1) * pool["tp"]]
            if len(set(ids)) != len(ids):
                raise ValueError("one TP group cannot repeat a GPU")
            gpu = [lookup[n] for n in ids]
            hw.find_aggregate_bandwidth(
                gpu
            )  # Validate the collective scope before spawning workers.
            capacities = {hw.find_local_memory(d).capacity_gb for d in gpu}
            if len(capacities) != 1:
                raise ValueError("one TP group requires equal GPU memory capacities")
            runner.update(
                instance_id=identity,
                tp=pool["tp"],
                hardware={
                    "gpu_model": pool["gpu_model"],
                    "gpu_count": len(ids),
                    "gpu_ids": ids,
                    "topology": topology,
                },
            )
            if shared_execution:
                runner.update(shared_execution=True, stage=prefix)
            command = runner["command"]
            if "--audit-dir" in command:
                command[command.index("--audit-dir") + 1] = str(directory / "audit")
            child["scheduler"]["device_mem"] = round(next(iter(capacities)) * 1e9)
            for parameter in ["max_batch", "max_num_batched_tokens"]:
                if parameter in pool:
                    if pool[parameter] <= 0:
                        raise ValueError(f"{parameter} must be positive")
                    child["scheduler"][parameter] = pool[parameter]
            parent[key].append(child)
            allocated.append(ids)
            instances.append((identity, directory / "audit", directory))
        groups.append(allocated)
    parent["kv_topology"] = kv_topology(hw, *groups, shards=KVH)
    return {"system": parent}, instances
