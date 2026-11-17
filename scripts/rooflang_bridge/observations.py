"""Join an instance's worker audit to its outer iteration nodes."""

import json
import re


def iteration_nodes(outer, config):
    identity = config.get("instance_id")
    result = []
    for node in outer["nodes"]:
        label = node["details"]["label"]
        if not label.startswith("RooflangIteration"):
            continue
        match = re.search(r'instance_id: ("(?:\\.|[^"\\])*")', label)
        node_identity = json.loads(match[1]) if match else None
        if node_identity == identity:
            result.append(node)
    return sorted(result, key=lambda n: n["dispatches"][0][0])
