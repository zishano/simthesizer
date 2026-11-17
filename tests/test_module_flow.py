"""Module ownership must not cross P/D callbacks or invent an instance."""

import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from rooflang_bridge.module_flow import annotate_modules


def node(i, name, cause=None, category="schedule", instance=None, requests=None):
    return {
        "id": i,
        "cause": cause,
        "time": 0,
        "start": 1,
        "finish": 1,
        "release": 0,
        "view": {
            "name": name,
            "category": category,
            "instance": instance,
            "round": 0 if instance else None,
            "module": "schedule",
            "requests": requests or [],
            "layer": None,
        },
    }


class ModuleFlowTests(unittest.TestCase):
    def fixture(self):
        data = {
            "nodes": [],
            "edges": [],
            "rounds": [],
            "kv": [],
            "instances": [
                {"id": "p0", "side": "p"},
                {"id": "p1", "side": "p"},
                {"id": "d0", "side": "d"},
            ],
            "requests": [{"id": 0, "output": 3}, {"id": 1, "output": 1}],
        }
        for offset, owner, request, cause in [
            (0, "p0", 0, None),
            (10, "d0", 0, 9),
            (20, "p1", 1, None),
        ]:
            names = [
                "RequestArrival",
                "SubRequestArrival",
                "SubRequestComplete",
                "LlmEnqueue",
                "LlmSchedule",
                "RooflangRoundBegin",
                "RooflangRoundEnd",
                "LlmBatchComplete",
                "LlmSchedule",
                "PdDisaggregationSystem",
            ]
            for j, name in enumerate(names):
                parent = {
                    0: cause,
                    1: cause,
                    2: cause,
                    3: offset + 1,
                    4: offset + 3,
                    5: offset + 4,
                    6: offset + 4,
                    7: offset + 4,
                    8: offset + 7,
                    9: offset + 2,
                }[j]
                data["nodes"].append(
                    node(
                        offset + j,
                        name,
                        parent,
                        "boundary" if j in (5, 6) else "schedule",
                        owner if j in (5, 6) else None,
                        [request] if j in (0, 1, 2, 3) else [],
                    )
                )
            for a, b in [
                (0, 1),
                (1, 2),
                (1, 3),
                (3, 4),
                (4, 5),
                (6, 7),
                (7, 2),
                (7, 8),
            ]:
                data["edges"].append({"parent": offset + a, "child": offset + b})
            data["rounds"].append({"begin": offset + 5, "instance": owner})
        # The cross-stage dependency must not connect the control components.
        data["edges"].append({"parent": 9, "child": 10})
        return data

    def test_three_instances_and_no_decode_request(self):
        data = self.fixture()
        before = copy.deepcopy(data)
        annotate_modules(data)
        by_id = {n["id"]: n for n in data["nodes"]}
        for base, owner in [(0, "p0"), (10, "d0"), (20, "p1")]:
            for i in (0, 1, 2, 3, 4, 7, 8):
                self.assertEqual(by_id[base + i]["view"]["instance"], owner)
        self.assertEqual(by_id[9]["view"]["flow"], "p_control")
        self.assertEqual(by_id[19]["view"]["flow"], "complete")
        self.assertEqual(by_id[29]["view"]["flow"], "complete")
        self.assertEqual(by_id[10]["view"]["flow"], "d_schedule")
        self.assertEqual(data["edges"], before["edges"])
        for old, new in zip(before["nodes"], data["nodes"]):
            self.assertEqual(
                {k: v for k, v in old.items() if k != "view"},
                {k: v for k, v in new.items() if k != "view"},
            )

    def test_conflicting_or_missing_ownership_is_explicit(self):
        data = self.fixture()
        data["rounds"].append({"begin": 5, "instance": "d0"})
        data["nodes"].append(node(30, "LlmSchedule"))
        annotate_modules(data)
        for i in (0, 4, 7, 30):
            view = data["nodes"][i]["view"]
            self.assertIsNone(view["instance"])
            self.assertEqual(view["flow"], "unassigned")

    def test_ack_versions_and_kv_wait_exclude_compute_dependencies(self):
        for ack in (
            "KV Some(Ack { source: 7, target: 8 })",
            "KV Some(Ack { request: 0, source: 7, target: 8 })",
        ):
            with self.subTest(ack=ack):
                data = self.fixture()
                producer = node(
                    30, "layer0.rank0.projection", category="operator", instance="d0"
                )
                producer.update(start=1, finish=3)
                copied = node(31, ack, category="kv_state", instance="p0")
                copied.update(start=5, finish=5)
                read = node(
                    32,
                    "KV Some(Read { request: 0, prefill: false, layer: 0, rank: 0, tokens: 5 })",
                    category="kv_state",
                    instance="d0",
                )
                read.update(start=5, finish=5)
                data["nodes"] += [producer, copied, read]
                data["edges"] += [
                    {"parent": 30, "child": 32},
                    {"parent": 31, "child": 32},
                ]
                data["kv"] = [{"block": {"id": 7, "request": 0, "layer": 0}}]
                annotate_modules(data)
                self.assertEqual(copied["view"]["flow"], "d_kv")
                self.assertIsNone(copied["view"]["instance"])
                self.assertEqual(read["view"]["kv_wait"], [3, 5])


if __name__ == "__main__":
    unittest.main()
