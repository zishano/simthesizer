"""Real Rooflang regression for causal serving policies and deferred KV copies."""

import json
import re
import sys
import unittest
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_rooflang_shared as shared
from results import load_run, load_workload, validate_complete
from rooflang_bridge.llama import KVH, describe


def policy(kind):
    return {
        "kind": kind,
        "seed": 0,
        "limits": {
            "p_token_budget_bounds": [1, 3],
            "d_token_budget_bounds": [1, 3],
            "p_batch_bounds": [1, 2],
            "d_batch_bounds": [1, 2],
            "kv_inflight_bounds": [1, 1],
            "max_wait_s": 0.05,
            "state_limit_bytes": 64,
        },
        "slo": {"ttft_limit_s": 5.0, "tpot_limit_s": 0.05},
    }


class ServingPolicyTests(unittest.TestCase):
    run_case = shared.SharedExecutionTests.run_case

    def assert_causal_trace(self, root):
        nodes, live, pending = {}, {}, {}
        active, inflight, prefill_done = set(), set(), set()
        ready = defaultdict(set)
        peak_inflight = peak_pending = reads = received_bytes = 0
        completed = False
        memory = None
        read_label = re.compile(
            r"KV Some\(Read \{ request: (\d+), prefill: (true|false), "
            r"layer: (\d+), rank: (\d+), tokens: (\d+) \}\)"
        )
        with (root / "outer.jsonl").open() as source:
            for line in source:
                event = json.loads(line)
                kind = event["event"]
                self.assertFalse(completed, "events after run_complete")
                if kind == "node_create":
                    self.assertNotIn(event["node"], nodes)
                    nodes[event["node"]] = event["details"]
                elif kind == "resource_start":
                    self.assertNotIn(event["node"], active)
                    active.add(event["node"])
                elif kind == "node_complete":
                    identity = event["node"]
                    details = nodes.pop(identity)
                    if details["kind"] == "resource":
                        active.remove(identity)
                    match = read_label.fullmatch(details["label"])
                    if match:
                        request, prefill, layer, rank, tokens = match.groups()
                        request, layer, rank, tokens = map(
                            int, (request, layer, rank, tokens)
                        )
                        prefill = prefill == "true"
                        width = KVH // (2 if prefill else 1)
                        for head in range(rank * width, (rank + 1) * width):
                            self.assertTrue(
                                set(range(tokens)).issubset(
                                    ready[request, prefill, layer, head]
                                ),
                                f"read before KV ready: {details['label']}",
                            )
                        reads += 1
                    elif details["label"].startswith("KV Some(Read"):
                        self.fail(f"unrecognized KV read: {details['label']}")
                elif kind == "kv_prefill_done":
                    prefill_done.add(event["request_id"])
                elif kind == "memory_state":
                    memory = event["state"]
                elif kind == "kv":
                    block, transition = event["block"], event["transition"]
                    identity, src = block["id"], block["source"]
                    if transition in ("produce", "pending_send"):
                        self.assertNotIn(identity, live)
                        live[identity] = block
                    if transition == "pending_send":
                        self.assertIn(src, live, "pending copy lost its source pin")
                        pending[identity] = src
                        peak_pending = max(peak_pending, len(pending))
                    elif transition == "send":
                        self.assertEqual(pending[identity], src)
                        self.assertIn(src, live)
                        self.assertNotIn(identity, inflight)
                        inflight.add(identity)
                        peak_inflight = max(peak_inflight, len(inflight))
                    elif transition == "receive":
                        self.assertIn(src, live, "source released before ACK")
                        self.assertEqual(pending.pop(identity), src)
                        inflight.remove(identity)
                        received_bytes += block["bytes"]
                    elif transition == "release":
                        self.assertNotIn(identity, pending.values())
                        self.assertNotIn(identity, pending)
                        if block["prefill"]:
                            self.assertIn(block["request"], prefill_done)
                        del live[identity]
                    if transition in ("produce", "receive", "extend"):
                        for head in range(block["head_start"], block["head_end"]):
                            ready[
                                block["request"], block["prefill"], block["layer"], head
                            ].update(range(block["token_start"], block["token_end"]))
                elif kind == "run_complete":
                    completed = True
        self.assertTrue(completed)
        self.assertGreater(reads, 0)
        self.assertEqual(peak_inflight, 1)
        self.assertGreater(peak_pending, 1, "case did not exercise deferred copies")
        self.assertFalse(nodes or active or live or pending or inflight)
        self.assertIsNotNone(memory)
        self.assertEqual(memory["active"], 0)
        self.assertEqual(memory["waiting"], 0)
        self.assertTrue(all(k.startswith("weights:") for k in memory["allocations"]))
        metadata = describe({"tp": 1, "pp": 1, "ep": 1})
        expected_bytes = 2 * 5 * metadata["kv_bytes_per_token_per_device"]
        self.assertEqual(received_bytes, expected_bytes)
        report = json.loads((root / "shared-check.json").read_text())
        self.assertEqual(report["kv_sent_bytes"], expected_bytes)
        self.assertEqual(report["kv_received_bytes"], expected_bytes)
        self.assertEqual(report["live_kv_after_run"], 0)

    def test_cross_tp_chunked_policies_complete_and_preserve_causality(self):
        for handoff in ("layer", "whole"):
            baseline = None
            for kind in (None, "reference", "candidate"):
                with self.subTest(handoff=handoff, policy=kind):
                    root, result = self.run_case(
                        f"policy-{handoff}-{kind or 'disabled'}",
                        p=2,
                        d=1,
                        count=2,
                        outputs=3,
                        budget=3,
                        slow=True,
                        handoff=handoff,
                        trace=kind == "candidate",
                        mutate=(
                            (
                                lambda system, kind=kind: system.update(
                                    policy=policy(kind)
                                )
                            )
                            if kind
                            else None
                        ),
                    )
                    self.assertEqual(result.returncode, 0, result.stderr)
                    run = load_run(str(root / "result.jsonl"))
                    complete = validate_complete(
                        run, load_workload(root / "workload.jsonl")
                    )
                    self.assertEqual(complete["completed_requests"], 2)
                    self.assertEqual(complete["output_tokens"], 6)
                    if kind is None:
                        baseline = run.requests
                    elif kind == "reference":
                        self.assertEqual(run.requests, baseline)
                    else:
                        self.assert_causal_trace(root)


if __name__ == "__main__":
    unittest.main()
