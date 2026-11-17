"""Observation checks: native coverage, time boundaries, and safe offline export."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from draw_graph import load_trace
from draw_shared import build_view, digest, render
from rooflang_bridge.configuration import toml_document


class SharedGraphTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        system = {
            "shared_execution": True,
            "kv_handoff": "layer",
            "model": {"num_hidden_layers": 1},
        }
        for side, device in [("prefill", "gpu0"), ("decode", "gpu1")]:
            identity = side[0] + "0"
            system[side + "_instances"] = [
                {
                    "model_runner": {
                        "instance_id": identity,
                        "tp": 1,
                        "hardware": {"gpu_model": "h200", "gpu_ids": [device]},
                    },
                    "scheduler": {"max_num_batched_tokens": 2},
                }
            ]
        (self.root / "config.toml").write_text(toml_document({"system": system}))
        self.bad_name = 'layer0.rank0.attention </script><script>alert("x")</script>'
        events = [{"event": "header", "version": 1, "time_s": 0}]
        completions = []
        for index, (instance, round_id, start) in enumerate(
            [("p0", 0, 1.0), ("d0", 0, 1.4), ("d0", 1, 1.8)]
        ):
            schedule, begin, end, op = [index * 4 + i for i in range(4)]
            metadata = {"instance_id": instance, "round": round_id}
            common = {"demands": [], "reserve": [], "release": [], "slots": []}
            details = [
                {"kind": "logical", "label": "LlmSchedule"},
                {
                    "kind": "resource",
                    "label": "RooflangRoundBegin",
                    "work": {
                        **common,
                        "details": {
                            **metadata,
                            "name": "RooflangRoundBegin",
                            "requests": [{"request_id": 0}],
                            "batch": [{"q_len": 2, "kv_len": 0, "lm_head_len": 1}],
                        },
                    },
                },
                {
                    "kind": "resource",
                    "label": "RooflangRoundEnd",
                    "work": {
                        **common,
                        "details": {**metadata, "name": "RooflangRoundEnd"},
                    },
                },
                {
                    "kind": "resource",
                    "label": self.bad_name,
                    "work": {
                        **common,
                        "demands": [
                            {"resources": {"compute:gpu0": 0.2}},
                            {"resources": {"hbm:gpu0": 0.1}},
                        ],
                        "details": {
                            **metadata,
                            "name": self.bad_name,
                            "devices": ["gpu0" if instance == "p0" else "gpu1"],
                            "network_us": 0,
                        },
                    },
                },
            ]
            for node_id, payload in zip([schedule, begin, end, op], details):
                events.append(
                    {
                        "event": "node_create",
                        "time_s": 0,
                        "node": node_id,
                        "cause": None if node_id == schedule else schedule,
                        "release_s": 0,
                        "details": payload,
                    }
                )
            for parent, child, event in [
                (schedule, op, "edge_add"),
                (schedule, op, "edge_remove"),
                (schedule, begin, "edge_add"),
                (begin, op, "edge_add"),
                (op, end, "edge_add"),
            ]:
                events.append(
                    {"event": event, "time_s": 0, "parent": parent, "child": child}
                )
            for node_id, ready, actual_start, finish in [
                (schedule, start, start, start),
                (begin, start, start, start),
                (op, start + 0.1, start + 0.2, start + 0.4),
                (end, start + 0.4, start + 0.4, start + 0.4),
            ]:
                completions.append(
                    {"event": "dispatch", "time_s": ready, "node": node_id}
                )
                if node_id != schedule:
                    completions.append(
                        {
                            "event": "resource_start",
                            "time_s": actual_start,
                            "node": node_id,
                        }
                    )
                completions.append(
                    {
                        "event": "node_complete",
                        "time_s": finish,
                        "node": node_id,
                        "start_s": actual_start,
                        "finish_s": finish,
                    }
                )
        events.extend(sorted(completions, key=lambda e: e["time_s"]))
        events.append({"event": "run_complete", "time_s": 2.2})
        (self.root / "outer.jsonl").write_text(
            "".join(json.dumps(dict(e, seq=i)) + "\n" for i, e in enumerate(events))
        )
        workload = {
            "request_id": 0,
            "arrival_time": 1.0,
            "initial": [0],
            "sub_requests": [{"kind": "llm", "input_tokens": 2, "output_tokens": 3}],
        }
        (self.root / "workload.jsonl").write_text(json.dumps(workload) + "\n")
        workload["sub_requests"][0]["token_latency"] = [0.4, 0.4, 0.4]
        (self.root / "result.jsonl").write_text(json.dumps(workload) + "\n")
        (self.root / "state.jsonl").write_text('{"time_s":0,"state":{}}\n')

    def test_complete_history_includes_every_round_and_removed_edges(self):
        raw = load_trace(self.root / "outer.jsonl")
        data = build_view(self.root)
        self.assertEqual(data["edges"], raw["edges"])
        self.assertEqual(
            [{k: v for k, v in n.items() if k != "view"} for n in data["nodes"]],
            raw["nodes"],
        )
        self.assertEqual([r["key"] for r in data["rounds"]], ["p0/0", "d0/0", "d0/1"])
        self.assertEqual(data["nodes"][11]["view"]["instance"], "d0")
        self.assertEqual(data["nodes"][11]["view"]["round"], 1)

    def test_resource_start_is_distinct_from_dispatch_and_base_costs_overlap(self):
        node = build_view(self.root)["nodes"][3]
        self.assertAlmostEqual(node["start"], 1.2)
        self.assertAlmostEqual(node["start"] - node["dispatches"][0][1], 0.1)
        self.assertAlmostEqual(node["finish"] - node["start"], 0.2)
        self.assertAlmostEqual(node["view"]["base_s"], 0.2)

    def test_offline_render_preserves_inputs_and_escapes_embedded_labels(self):
        sources = list(self.root.iterdir())
        before = {p.name: digest(p) for p in sources}
        output = self.root / "execution-graph.html"
        receipt = render(self.root, output)
        self.assertEqual(before, {p.name: digest(p) for p in sources})
        self.assertEqual(receipt["input_sha256"], before)
        document = output.read_text()
        payload = document.split('<script id="data" type="application/json">', 1)[1]
        payload = payload.split("</script>", 1)[0]
        embedded = json.loads(payload)
        self.assertEqual(embedded["nodes"][3]["view"]["name"], self.bad_name)
        self.assertNotIn(self.bad_name, document)
        self.assertEqual(receipt["nodes"], 12)
        with self.assertRaises(FileExistsError):
            render(self.root, output)


if __name__ == "__main__":
    unittest.main()
