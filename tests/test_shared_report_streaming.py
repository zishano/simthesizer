"""History length must not determine exporter memory; invalid traces still fail."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from rooflang_bridge.shared_report import export_shared


class StreamingReportTests(unittest.TestCase):
    def export(self, events):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        with (root / "outer.jsonl").open("w") as stream:
            for seq, event in enumerate(events):
                stream.write(json.dumps({"seq": seq, "time_s": seq, **event}) + "\n")
        export_shared(root, [])
        return json.loads((root / "shared-check.json").read_text())

    def history(self, count, bad_dependency=False):
        yield {"event": "header", "version": 1}
        for i in range(count):
            for node in (2 * i, 2 * i + 1):
                yield {
                    "event": "node_create",
                    "node": node,
                    "details": {"kind": "logical"},
                }
            yield {"event": "edge_add", "parent": 2 * i, "child": 2 * i + 1}
            yield {
                "event": "node_complete",
                "node": 2 * i,
                "start_s": i,
                "finish_s": i + 1,
            }
            yield {
                "event": "node_complete",
                "node": 2 * i + 1,
                "start_s": i if bad_dependency else i + 1,
                "finish_s": i + 1,
            }
        yield {"event": "run_complete"}

    def test_history_is_released_after_completion(self):
        for count in (1, 2000):
            self.assertEqual(
                self.export(self.history(count))["streaming"]["peak_live_nodes"], 2
            )

    def test_dependency_failure_is_not_hidden_by_releasing_parents(self):
        with self.assertRaisesRegex(ValueError, "dependency violated"):
            self.export(self.history(2, bad_dependency=True))

    def test_missing_run_complete_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "successful run_complete"):
            self.export(list(self.history(1))[:-1])

    def test_released_source_cannot_be_received(self):
        source = {"id": 1, "source": None, "prefill": True, "request": 0, "bytes": 8}
        target = {**source, "id": 2, "source": 1, "prefill": False}
        events = [
            {"event": "header", "version": 1},
            {"event": "kv", "transition": "produce", "block": source},
            {"event": "kv", "transition": "send", "block": target},
            {"event": "kv_prefill_done", "request_id": 0},
            {"event": "kv", "transition": "release", "block": source},
            {"event": "run_complete"},
        ]
        with self.assertRaisesRegex(ValueError, "before copy acknowledgement"):
            self.export(events)
