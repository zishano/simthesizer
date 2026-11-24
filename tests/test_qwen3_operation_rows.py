"""Dense wire rows preserve every operation, dependency and attention sub-batch."""

import json
import tempfile
import unittest
from pathlib import Path

from test_qwen3_compact_work import BATCH, HW, QwenTemplate, init, worker


def expand_rows(plan):
    operations = []
    for index, (parents, work_id, layer, kv_rank, batch_indices) in enumerate(
        plan["operation_rows"]
    ):
        operation = {
            "id": index,
            "parents": parents,
            "layer": layer,
            "kv_rank": kv_rank,
            "work": plan["work_templates"][work_id],
        }
        if batch_indices is not None:
            operation["batch_indices"] = batch_indices
        operations.append(operation)
    return {
        "operations" if key == "operation_rows" else key: (
            operations if key == "operation_rows" else value
        )
        for key, value in plan.items()
        if key != "work_templates"
    }


class OperationRowsTests(unittest.TestCase):
    def test_expansion_is_byte_exact_across_layouts_ragged_batches_and_routes(self):
        for ep in (1, 2, 4, 8):
            for tp in (tp for tp in (1, 2, 4) if ep % tp == 0):
                template = QwenTemplate(tp, ep, HW)
                for batch, identities, routing in (
                    (BATCH, [3], None),
                    (
                        [{"q_len": 3, "kv_len": 0, "lm_head_len": 1}, *BATCH],
                        [5, 11],
                        None,
                    ),
                    (
                        [{"q_len": 2, "kv_len": 17, "lm_head_len": 1}],
                        [2],
                        [list(range(8))] * 2,
                    ),
                    (
                        [{"q_len": 2, "kv_len": 17, "lm_head_len": 1}],
                        [2],
                        [[16 * i for i in range(8)]] * 2,
                    ),
                ):
                    with self.subTest(tp=tp, ep=ep, batch=batch, routing=routing):
                        plain = template.export(
                            batch, identities, details=False, routing=routing
                        )
                        rows = template.export(
                            batch,
                            identities,
                            details=False,
                            routing=routing,
                            operation_rows=True,
                        )
                        self.assertNotIn("operations", rows)
                        self.assertEqual(
                            json.dumps(expand_rows(rows)), json.dumps(plain)
                        )
                        self.assertTrue(
                            all(len(row) == 5 for row in rows["operation_rows"])
                        )

    def test_worker_negotiates_only_boolean_true_and_recovers_after_error(self):
        graph = {"op": "graph", "batch": BATCH, "requests": [0], "details": False}
        _, plain, rows, compact_rows, disabled, failed, recovered = worker(
            [
                init(),
                graph,
                graph | {"operation_rows": True},
                graph | {"operation_rows": True, "compact_work": True},
                graph | {"operation_rows": "true"},
                graph | {"operation_rows": True, "requests": []},
                graph | {"operation_rows": True},
            ]
        )
        self.assertEqual(expand_rows(rows), plain)
        self.assertEqual(compact_rows, rows)
        self.assertEqual(disabled, plain)
        self.assertIn("error", failed)
        self.assertEqual(recovered, rows)

    def test_reports_audit_and_llama_keep_legacy_objects(self):
        graph = {"op": "graph", "batch": BATCH, "requests": [0]}
        _, detailed, requested = worker(
            [init(), graph, graph | {"operation_rows": True, "compact_work": True}]
        )
        with tempfile.TemporaryDirectory() as directory:
            _, audited = worker(
                [init(), graph | {"details": False, "operation_rows": True}],
                "--audit-dir",
                str(Path(directory) / "audit"),
            )
        self.assertEqual(requested, detailed)
        self.assertEqual(audited, detailed)
        self.assertNotIn("operation_rows", requested)
        _, _, plain, requested = worker(
            [init(False), graph, graph, graph | {"operation_rows": True}]
        )
        self.assertEqual(requested, plain)
        self.assertNotIn("operation_rows", requested)


if __name__ == "__main__":
    unittest.main()
