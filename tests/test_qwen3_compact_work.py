"""Work interning compresses the wire representation, never the execution DAG."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from rooflang_bridge.llama import describe as describe_llama
from rooflang_bridge.qwen3 import QwenTemplate, describe

HW = {
    "tflops_fp16": 100,
    "memory_bandwidth_gbs": 1000,
    "memory_capacity_gb": 1000,
    "link_bandwidth_gbs": 100,
    "link_latency_us": 1,
}
BATCH = [{"q_len": 1, "kv_len": 17, "lm_head_len": 1}]


def expand(plan):
    templates = plan["work_templates"]
    expanded = {key: value for key, value in plan.items() if key != "work_templates"}
    expanded["operations"] = [
        {
            ("work" if key == "work_id" else key): (
                templates[value] if key == "work_id" else value
            )
            for key, value in operation.items()
        }
        for operation in plan["operations"]
    ]
    return expanded


def init(qwen=True):
    model = (describe if qwen else describe_llama)(
        {"tp": 1, "ep": 8 if qwen else 1, "pp": 1}
    )
    return {
        "op": "init",
        "version": 1,
        "tp": 1,
        "hardware": HW,
        "model": {key: model[key] for key in ("pre", "block", "post", "layers")},
        "memory": {
            key: model[key]
            for key in ("weight_bytes_per_device", "kv_bytes_per_token_per_device")
        },
        **(
            {"model_kind": "qwen3_235b_a22b", "ep": 8, "routing_seed": 0}
            if qwen
            else {}
        ),
    }


def worker(requests, *args):
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/rooflang_worker.py"), *args],
        input="\n".join(map(json.dumps, requests)) + "\n",
        text=True,
        capture_output=True,
        check=True,
        timeout=30,
    )
    return [json.loads(line) for line in result.stdout.splitlines()]


class CompactWorkTests(unittest.TestCase):
    def test_expert_chains_keep_each_layer_and_rank_with_changing_routes(self):
        batch = [{"q_len": 2, "kv_len": 17, "lm_head_len": 1}]
        for tp, ep in (
            (1, 1),
            (1, 2),
            (1, 4),
            (1, 8),
            (2, 2),
            (2, 4),
            (2, 8),
            (4, 4),
            (4, 8),
        ):
            template = QwenTemplate(tp, ep, HW)
            for routing in (
                [list(range(8))] * 2,
                [[16 * rank for rank in range(8)]] * 2,
            ):
                with self.subTest(tp=tp, ep=ep, routing=routing):
                    plain = template.export(batch, [3], details=False, routing=routing)
                    compact = template.export(
                        batch, [3], details=False, routing=routing, compact_work=True
                    )
                    self.assertEqual(json.dumps(expand(compact)), json.dumps(plain))

    def test_expansion_is_byte_exact_for_ragged_shapes_and_attention_tp(self):
        for tp in (1, 2, 4):
            template = QwenTemplate(tp, 8, HW)
            for batch, requests in (
                (BATCH, [0]),
                ([{"q_len": 3, "kv_len": 7, "lm_head_len": 1}, *BATCH], [3, 5]),
            ):
                with self.subTest(tp=tp, batch=batch):
                    plain = template.export(batch, requests, details=False)
                    compact = template.export(
                        batch, requests, details=False, compact_work=True
                    )
                    self.assertEqual(json.dumps(expand(compact)), json.dumps(plain))
                    self.assertLess(
                        len(compact["work_templates"]), len(compact["operations"]) // 10
                    )
                    self.assertTrue(
                        all("work" not in op for op in compact["operations"])
                    )

    def test_worker_negotiation_errors_and_llama_fallback(self):
        graph = {"op": "graph", "batch": BATCH, "requests": [0], "details": False}
        ready, plain, compact, failed, recovered, disabled = worker(
            [
                init(),
                graph,
                graph | {"compact_work": True},
                graph | {"compact_work": True, "requests": []},
                graph | {"compact_work": True},
                graph | {"compact_work": "true"},
            ]
        )
        self.assertTrue(ready["ready"])
        self.assertEqual(expand(compact), plain)
        self.assertIn("error", failed)
        self.assertEqual(recovered, compact)
        self.assertEqual(disabled, plain)
        _, _, plain, unsupported = worker(
            [init(False), graph, graph, graph | {"compact_work": True}]
        )
        self.assertEqual(unsupported, plain)
        self.assertNotIn("work_templates", unsupported)

    def test_reports_always_keep_per_operator_metadata(self):
        graph = {"op": "graph", "batch": BATCH, "requests": [0], "compact_work": True}
        with tempfile.TemporaryDirectory() as directory:
            _, compact_requested = worker(
                [init(), graph | {"details": False}],
                "--audit-dir",
                str(Path(directory) / "audit"),
            )
            _, detailed = worker([init(), graph])
        self.assertEqual(compact_requested, detailed)
        self.assertNotIn("work_templates", detailed)
        self.assertTrue(
            all("name" in op["work"]["details"] for op in detailed["operations"])
        )


if __name__ == "__main__":
    unittest.main()
