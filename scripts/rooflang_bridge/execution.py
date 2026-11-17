"""Lower Rooflang dependencies and service demands, never predicted timestamps."""

import re
from collections import OrderedDict

from rooflang.runtime.simulator import Simulator


def fabric_resource(edge, direction):
    # Endpoints, edge name and direction survive independent worker construction.
    direction = direction if edge.is_full_duplex else "shared"
    return f"fabric:{edge.src.name}|{edge.dst.name}|{edge.name}|{direction}"


class ExecutionPlan:
    """Bounded workspace/order cache for the validated single-stream Llama graph.

    Changing resident KV changes attention service, not tensor lifetimes: each
    rank's cost-bearing kernels form a chain joined by collective barriers.
    Residual forks/views are zero-work aliases. Workspace therefore depends on
    query/logits rows; past KV is accounted separately by the native ledger.
    Costs and tensor dimensions are rebound and repriced on every call.
    """

    def __init__(self, template):
        self.template = template
        self.pricing = Simulator(template.graph, template.placement, template.hardware)
        self.kernels = list(template.graph.topological_sort())
        self.ids = {k: i for i, k in enumerate(self.kernels)}
        self.cache = OrderedDict()
        self.reference_runs = 0

    def bind(self, batch, reference):
        from rooflang_bridge.llama import describe

        template = self.template
        shape = template.bind(batch)
        key = (shape["tokens"], shape["logits_rows"])
        memory = describe({"tp": shape["tp"], "pp": 1, "ep": 1})
        resident = (
            memory["weight_bytes_per_device"]
            + sum(r["kv_len"] for r in batch) * memory["kv_bytes_per_token_per_device"]
        )
        cached = self.cache.get(key)
        summary = None
        if cached is None or reference:
            result, summary = template.simulate(batch)
            self.reference_runs += 1
            predecessors = {
                k: set(template.graph._dag.predecessors(k)) for k in self.kernels
            }
            previous = {}
            for entry in result.trace:
                slot = (entry.device.name, entry.stream)
                if slot in previous and previous[slot] is not entry.kernel:
                    predecessors[entry.kernel].add(previous[slot])
                previous[slot] = entry.kernel
            parents = [
                sorted(self.ids[p] for p in predecessors[k]) for k in self.kernels
            ]
            workspace = {
                d["name"]: max(
                    0, round(summary["peak_memory_bytes"][d["memory"]]) - resident
                )
                for d in summary["hardware"]["devices"]
            }
            plan = (parents, workspace)
            if cached is not None and cached != plan:
                raise ValueError(
                    "Llama workspace/stream order changed for a cached batch shape"
                )
            self.cache[key] = plan
            if len(self.cache) > 64:
                self.cache.popitem(last=False)
        self.cache.move_to_end(key)
        parents, workspace = self.cache[key]
        for device in shape["hardware"]["devices"]:
            # A cache hit must not bypass the reference runtime's capacity check.
            dev = next(d for d in template.hardware.nodes if d.name == device["name"])
            capacity = template.hardware.find_local_memory(dev).capacity_gb * 1e9
            if resident + workspace[device["name"]] > capacity:
                raise ValueError(
                    f"OOM on {device['memory']}: resident KV and workspace exceed capacity"
                )
        return parents, workspace, summary


def export_execution(template, batch, *, reference=False, details=True):
    plan = getattr(template, "_execution_plan", None)
    if plan is None:
        plan = template._execution_plan = ExecutionPlan(template)
    parents, workspace, summary = plan.bind(batch, reference)
    pricing, kernels, ids = plan.pricing, plan.kernels, plan.ids
    operations = []
    for k in kernels:
        dev, stream, cap, participants = pricing._resolve(k)
        if cap != 1:
            raise ValueError("shared bridge currently requires full-capacity kernels")
        ct, mt, alpha, xfer, links = pricing._base_times(k, dev, participants)
        demands = []
        if ct:
            demands.append({"resources": {f"compute:{dev.name}": ct * 1e-6}})
        if mt:
            demands.append({"resources": {f"hbm:{dev.name}": mt * 1e-6}})
        if alpha or xfer:
            costs = {
                fabric_resource(edge, direction): (n / (bw * 1e9) if n else xfer * 1e-6)
                for (edge, direction), (n, bw) in links.items()
                if n or xfer
            }
            demands.append({"resources": costs, "latency": alpha * 1e-6})
        match = re.fullmatch(r"layer(\d+)\.rank(\d+)\.attention", k.name)
        devices = [d.name for d in participants] or [dev.name]
        operations.append(
            {
                "id": ids[k],
                "parents": parents[ids[k]],
                "layer": int(match[1]) if match else None,
                "kv_rank": int(match[2]) if match else None,
                "work": {
                    "demands": demands,
                    "slots": [f"stream:{d}:{stream}" for d in devices],
                    "details": {
                        "name": k.name,
                        "devices": devices,
                        "stream": stream,
                        "compute_us": ct,
                        "memory_us": mt,
                        "network_us": alpha + xfer,
                        "flops": float(k.flops),
                        "input_bytes": float(k.input_bytes),
                        "weight_bytes": float(k.loaded_weight_bytes),
                        "output_bytes": float(k.output_bytes),
                        "inputs": {
                            n: [float(x) for x in t.shape] for n, t in k.inputs.items()
                        },
                        "outputs": {
                            n: [float(x) for x in t.shape] for n, t in k.outputs.items()
                        },
                    }
                    if details
                    else {},
                },
            }
        )
    return {"operations": operations, "workspace": workspace, "reference": summary}
