"""Select original Rooflang hardware presets and allocate GPUs for one TP group."""

import math

from rooflang.language.graph import FabricEdge, HardwareGraph
from rooflang.language.hardware.component import Compute, Memory
from rooflang.programs.presets.b300 import B300Cluster
from rooflang.programs.presets.h200 import H200Cluster
from rooflang.programs.presets.rtx6000d import RTX6000DCluster


def build_hardware(config, tp):
    if "gpu_model" not in config:
        # Retained for old audit replay and diagnostic sensitivity tests only.
        hw, gpu, mem = _synthetic(config, tp)
        return hw, gpu, mem, _describe(hw, gpu, mem, "Synthetic", tp)
    if set(config) != {"gpu_model", "gpu_count"}:
        raise ValueError("preset hardware requires only gpu_model and gpu_count")
    count = config["gpu_count"]
    if type(count) is not int or count != tp or count not in (1, 2, 4, 8):
        raise ValueError("gpu_count must equal TP and be one of 1, 2, 4, 8")
    model = config["gpu_model"]
    if model == "rtx6000d":
        hw = RTX6000DCluster(eth_scope=8)
        label = "NVIDIA RTX 6000D"
    elif model == "h200":
        hw = H200Cluster()
        label = "NVIDIA H200 SXM"
    elif model == "b300":
        hw = B300Cluster()
        label = "NVIDIA B300 SXM"
    else:
        raise ValueError(f"unsupported Rooflang GPU preset: {model}")
    all_gpus = sorted((n for n in hw.nodes if n.kind == "gpu"), key=lambda n: n.name)
    gpu = all_gpus[:count]
    mem = [hw.find_local_memory(d) for d in gpu]
    return hw, gpu, mem, _describe(hw, gpu, mem, label, len(all_gpus))


def _describe(hw, gpu, mem, label, topology_count):
    return {
        "gpu_model": label,
        "gpu_count": len(gpu),
        "topology_gpu_count": topology_count,
        "devices": [
            {
                "name": d.name,
                "tflops": d.tflops,
                "memory": m.name,
                "memory_capacity_gb": m.capacity_gb,
                "memory_bandwidth_gbs": sum(
                    e.src_to_dst_bandwidth_gbs
                    if e.src is d
                    else e.dst_to_src_bandwidth_gbs
                    for e in hw._graph[d][m]["fabrics"]
                ),
            }
            for d, m in zip(gpu, mem)
        ],
    }


def _synthetic(hardware, tp):
    required = {
        "tflops_fp16",
        "memory_bandwidth_gbs",
        "memory_capacity_gb",
        "link_bandwidth_gbs",
        "link_latency_us",
    }
    if set(hardware) != required or any(
        not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0
        for v in hardware.values()
    ):
        raise ValueError("invalid hardware schema")
    if any(hardware[k] <= 0 for k in required - {"link_latency_us"}):
        raise ValueError("hardware capacities and rates must be positive")
    hw = HardwareGraph()
    gpu = []
    mem = []
    for rank in range(tp):
        dev = Compute(name=f"gpu{rank}", tflops={"fp16": hardware["tflops_fp16"]})
        ram = Memory(name=f"hbm{rank}", capacity_gb=hardware["memory_capacity_gb"])
        hw.add_node(dev)
        hw.add_node(ram)
        gpu.append(dev)
        mem.append(ram)
        bw = hardware["memory_bandwidth_gbs"]
        hw.add_edge(
            FabricEdge(
                name=f"hbm{rank}",
                src=dev,
                dst=ram,
                src_to_dst_bandwidth_gbs=bw,
                dst_to_src_bandwidth_gbs=bw,
                is_full_duplex=False,
            )
        )
    for a in range(tp):
        for b in range(a + 1, tp):
            bw = hardware["link_bandwidth_gbs"]
            hw.add_edge(
                FabricEdge(
                    name=f"link{a}-{b}",
                    src=gpu[a],
                    dst=gpu[b],
                    src_to_dst_bandwidth_gbs=bw,
                    dst_to_src_bandwidth_gbs=bw,
                    is_full_duplex=True,
                    alpha_us=hardware["link_latency_us"],
                )
            )

    return hw, gpu, mem
