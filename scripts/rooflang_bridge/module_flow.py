"""Assign native shared events to display modules without changing their history."""

import re

CONTROLS = {
    "RequestArrival",
    "SubRequestArrival",
    "SubRequestComplete",
    "LlmEnqueue",
    "LlmSchedule",
    "LlmBatchComplete",
}


def annotate_modules(data):
    nodes = {n["id"]: n for n in data["nodes"]}
    instances = {i["id"]: i for i in data["instances"]}
    requests = {r["id"]: r for r in data["requests"]}
    roles = {i: n["view"]["name"].split(" {", 1)[0] for i, n in nodes.items()}
    controls = {i for i, role in roles.items() if role in CONTROLS}
    roots = {i: i for i in controls}

    def root(i):
        while roots[i] != i:
            roots[i] = roots[roots[i]]
            i = roots[i]
        return i

    def join(a, b):
        if a in controls and b in controls:
            roots[root(a)] = root(b)

    # These edges stay within SingleInstanceSystem. Do not propagate through
    # the PD callback: it creates the next stage's arrival nodes.
    for edge in data["edges"]:
        join(edge["parent"], edge["child"])
    for i in controls:
        join(i, nodes[i]["cause"])
    owners, schedule_rounds = {}, {}
    for row in data["rounds"]:
        cause = nodes[row["begin"]]["cause"]
        if cause in controls:
            owners.setdefault(root(cause), set()).add(row["instance"])
            schedule_rounds[cause] = row.get("round")
    for i in controls:
        candidates = owners.get(root(i), set())
        view = nodes[i]["view"]
        owner = next(iter(candidates)) if len(candidates) == 1 else None
        if owner != view["instance"] or roles[i] in {
            "RequestArrival",
            "SubRequestArrival",
            "SubRequestComplete",
            "LlmEnqueue",
        }:
            view["round"] = None
        view["instance"] = owner
        view["module"] = instances[owner]["side"] if owner else "schedule"
        if roles[i] == "LlmSchedule":
            view["round"] = schedule_rounds.get(i)

    blocks = {e["block"]["id"]: e["block"] for e in data["kv"]}
    incoming = {}
    for edge in data["edges"]:
        incoming.setdefault(edge["child"], []).append(nodes[edge["parent"]])
    for i, node in nodes.items():
        view, role = node["view"], roles[i]
        side = instances.get(view["instance"], {}).get("side")
        flow, event = "unassigned", role
        if view["category"] == "kv":
            flow, event = "transfer", "copy"
        elif view["category"] == "kv_state":
            name = view["name"]
            match = re.search(r"\b(Read|Produce|Ack)\s*\{", name)
            event = match[1] if match else "KV"
            if event == "Ack":
                source = re.search(r"\bsource: (\d+)", name)
                block = blocks.get(int(source[1])) if source else None
                if block:
                    view.update(requests=[block["request"]], layer=block["layer"])
                flow = "d_kv"
                # Acknowledgement is owned by KV, not by its creator's P round.
                view.update(instance=None, round=None, module="kv")
            else:
                prefill = re.search(r"\bprefill: (true|false)", name)
                kv_side = ("p" if prefill[1] == "true" else "d") if prefill else side
                flow = f"{kv_side}_kv" if kv_side else "unassigned"
            if event == "Read" and flow == "d_kv":
                parents = incoming.get(i, [])
                copies = [
                    p for p in parents if re.search(r"\bAck\s*\{", p["view"]["name"])
                ]
                if copies:
                    ready = max(
                        [node["time"], node["release"]]
                        + [p["finish"] for p in parents if p not in copies]
                    )
                    if ready < node["start"]:
                        view["kv_wait"] = [ready, node["start"]]
        elif view["category"] == "boundary" and side:
            flow = f"{side}_control"
        elif view["category"] in ("operator", "tp") and side:
            flow = f"{side}_compute"
        elif i in controls and side:
            if role == "RequestArrival" and side == "p":
                flow = "arrivals"
            elif role in ("LlmBatchComplete", "SubRequestComplete"):
                flow = f"{side}_control"
            else:
                flow = f"{side}_schedule"
        elif role == "PdDisaggregationSystem":
            parent = nodes.get(node["cause"])
            if parent and roles[parent["id"]] == "SubRequestComplete":
                previous = parent["view"]
                view.update(
                    instance=previous["instance"],
                    round=None,
                    requests=previous["requests"],
                )
                side = instances.get(previous["instance"], {}).get("side")
                has_decode = any(
                    requests.get(r, {}).get("output", 0) > 1 for r in view["requests"]
                )
                if side == "d" or (side == "p" and not has_decode):
                    flow, event = "complete", "RequestComplete"
                elif side == "p":
                    flow, event = "p_control", "Handoff"
                view["module"] = side or "schedule"
        view["flow"] = flow
        view["flow_event"] = event
