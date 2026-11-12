import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "draw_graph", Path(__file__).resolve().parents[1] / "scripts/draw_graph.py"
)
draw_graph = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(draw_graph)


class GraphReplayTests(unittest.TestCase):
    def load(self, rows):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "trace.jsonl"
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            return draw_graph.load_trace(path)

    def rows(self):
        events = [
            {"event": "header", "version": 1},
            {
                "event": "node_create",
                "node": 0,
                "cause": None,
                "release_s": 0,
                "details": {"kind": "logical", "label": "parent"},
            },
            {
                "event": "node_create",
                "node": 1,
                "cause": None,
                "release_s": 0,
                "details": {"kind": "compute", "label": "child"},
            },
            {"event": "edge_add", "parent": 0, "child": 1},
            {"event": "edge_remove", "parent": 0, "child": 1},
            {"event": "edge_add", "parent": 0, "child": 1},
            {"event": "dispatch", "node": 0},
            {"event": "node_complete", "node": 0, "start_s": 0, "finish_s": 1},
            {"event": "dispatch", "node": 1},
            {"event": "node_complete", "node": 1, "start_s": 1, "finish_s": 2},
            {"event": "run_complete"},
        ]
        return [dict(row, seq=i, time_s=0) for i, row in enumerate(events)]

    def test_readded_dependencies_and_future_finish_are_retained(self):
        data = self.load(self.rows())
        self.assertEqual([e["removed"] for e in data["edges"]], [4, None])
        self.assertEqual(data["nodes"][0]["finish"], 1)
        self.assertEqual(data["changes"][7][1], 0)
        self.assertEqual(data["nodes"][1]["dispatches"], [[8, 0]])

    def test_incomplete_or_inconsistent_history_is_rejected(self):
        rows = self.rows()
        with self.assertRaisesRegex(ValueError, "run_complete"):
            self.load(rows[:-1])
        rows[5]["child"] = 99
        with self.assertRaisesRegex(ValueError, "unknown node"):
            self.load(rows)
