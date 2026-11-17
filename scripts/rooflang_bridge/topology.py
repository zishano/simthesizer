"""Shared preset topology, disjoint allocations and native KV routes."""

import copy
import math
import re

from rooflang.language.graph import FabricEdge, HardwareGraph
from rooflang.programs.presets.b300 import B300Cluster
from rooflang.programs.presets.h200 import H200Cluster
from rooflang.programs.presets.rtx6000d import RTX6000DCluster

MODELS = {
    "rtx6000d": "NVIDIA RTX 6000D",
    "h200": "NVIDIA H200 SXM",
    "b300": "NVIDIA B300 SXM",
}


class DeploymentHardware(HardwareGraph):
    def __init__(self):
        super().__init__()
        self.owners = {}
        self.models = {}

    def find_aggregate_bandwidth(self, devices):
        owners = {self.owners[d] for d in devices}
        if len(owners) != 1:
            raise ValueError("one TP group must belong to one preset cluster")
        # Preset aggregate rules identify nodes from their original device names.
        local = [copy.copy(d) for d in devices]
        for device in local:
            device.name = device.name.split("/", 1)[1]
        return next(iter(owners)).find_aggregate_bandwidth(local)


def device_order(device):
    return [
        int(part) if part.isdigit() else part
        for part in re.split(r"(\d+)", device.name)
    ]


def default_topology(model, count):
    cluster = {"id": "cluster", "gpu_model": model}
    if model == "rtx6000d":
        cluster["eth_scope"] = max(8, math.ceil(count / 8) * 8)
    else:
        cluster["n_nodes"] = math.ceil(count / 8)
    return {"clusters": [cluster], "links": []}


def build_topology(spec):
    if set(spec) != {"clusters", "links"} or not spec["clusters"]:
        raise ValueError("topology requires nonempty clusters and links")
    hw = DeploymentHardware()
    for config in spec["clusters"]:
        if set(config) - {"id", "gpu_model", "n_nodes", "eth_scope"}:
            raise ValueError("unknown topology cluster parameter")
        identity, model = config["id"], config["gpu_model"]
        if not isinstance(identity, str) or not identity or "/" in identity:
            raise ValueError("cluster id must be nonempty and contain no slash")
        nodes = config.get("n_nodes", 1)
        if type(nodes) is not int or nodes < 1:
            raise ValueError("n_nodes must be positive")
        if model == "rtx6000d":
            preset = RTX6000DCluster(
                eth_scope=config.get("eth_scope", 8), n_nodes=nodes
            )
        elif model in ("h200", "b300"):
            if "eth_scope" in config:
                raise ValueError("eth_scope applies only to RTX 6000D")
            preset = (H200Cluster if model == "h200" else B300Cluster)(n_nodes=nodes)
        else:
            raise ValueError(f"unsupported GPU model: {model}")
        for node in sorted(preset.nodes, key=lambda n: n.name):
            node.name = identity + "/" + node.name
            hw.add_node(node)
            hw.owners[node] = preset
            hw.models[node] = model
        for _, _, data in preset._graph.edges(data=True):
            for edge in data["fabrics"]:
                hw.add_edge(edge)
    names = {n.name: n for n in hw.nodes}
    for link in spec["links"]:
        if set(link) != {"src", "dst", "bandwidth_gbs", "latency_us"}:
            raise ValueError("bridge requires src, dst, bandwidth_gbs, latency_us")
        bw, latency = link["bandwidth_gbs"], link["latency_us"]
        if (
            not math.isfinite(bw)
            or bw <= 0
            or not math.isfinite(latency)
            or latency < 0
        ):
            raise ValueError("invalid bridge bandwidth or latency")
        if link["src"] == link["dst"]:
            raise ValueError("bridge endpoints must differ")
        hw.add_edge(
            FabricEdge(
                name="bridge",
                src=names[link["src"]],
                dst=names[link["dst"]],
                src_to_dst_bandwidth_gbs=bw,
                dst_to_src_bandwidth_gbs=bw,
                is_full_duplex=True,
                alpha_us=latency,
            )
        )
    return hw


def kv_topology(hw, prefill, decode, shards):
    """Export only traversed directed resources; fixed routes preserve Rooflang routing."""
    from rooflang_bridge.execution import fabric_resource

    devices = sorted(hw.nodes, key=lambda n: n.name)
    ids = {d.name: i for i, d in enumerate(devices)}
    links, routes, link_ids = [], [], {}
    routed = set()
    for source in (d for group in prefill for d in group):
        for target in (d for group in decode for d in group):
            if (source, target) in routed:
                continue
            routed.add((source, target))
            path = hw.find_fabric_path_directed(
                devices[ids[source]], devices[ids[target]]
            )
            hops = []
            for edge, direction in path:
                if not edge.is_full_duplex:
                    raise ValueError("KV paths currently require full-duplex fabrics")
                src, dst = (
                    (edge.src, edge.dst) if direction == "fwd" else (edge.dst, edge.src)
                )
                key = (id(edge), direction)
                if key not in link_ids:
                    link_ids[key] = len(links)
                    bw = (
                        edge.src_to_dst_bandwidth_gbs
                        if direction == "fwd"
                        else edge.dst_to_src_bandwidth_gbs
                    )
                    links.append(
                        {
                            "src": ids[src.name],
                            "dst": ids[dst.name],
                            "bandwidth": bw * 1e9,
                            "latency": edge.alpha_us * 1e-6,
                            "resource": fabric_resource(edge, direction),
                        }
                    )
                hops.append(link_ids[key])
            routes.append({"src": ids[source], "dst": ids[target], "links": hops})
    return {
        "devices": [d.name for d in devices],
        "links": links,
        "routes": routes,
        "shards": shards,
        "prefill_gpus": [[ids[n] for n in group] for group in prefill],
        "decode_gpus": [[ids[n] for n in group] for group in decode],
    }
