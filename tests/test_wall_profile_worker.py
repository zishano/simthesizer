"""Worker wall clocks are optional observations, never simulation inputs."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from rooflang_bridge.llama import describe as describe_llama
from rooflang_bridge.qwen3 import QwenTemplate, describe
from rooflang_bridge.wall_profile import (
    WallProfile,
    encode_reply,
    request_profile,
    scope,
)

HW = {
    "tflops_fp16": 100,
    "memory_bandwidth_gbs": 1000,
    "memory_capacity_gb": 1000,
    "link_bandwidth_gbs": 100,
    "link_latency_us": 1,
}
BATCH = [{"q_len": 1, "kv_len": 0, "lm_head_len": 1}]


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


class WallProfileTests(unittest.TestCase):
    def test_nested_sections_are_exclusive_even_on_failure(self):
        def fail():
            raise ValueError("failed")

        profile = WallProfile()
        with (
            patch(
                "rooflang_bridge.wall_profile.perf_counter", side_effect=[0, 2, 5, 9]
            ),
            profile.scope("graph"),
            self.assertRaisesRegex(ValueError, "failed"),
        ):
            profile.call("routing", fail)
        self.assertEqual(
            profile.sections,
            {
                "graph": {"calls": 1, "seconds": 6.0},
                "routing": {"calls": 1, "seconds": 3.0},
            },
        )

    def test_disabled_decode_and_encode_never_read_wall_clock(self):
        with patch(
            "rooflang_bridge.wall_profile.perf_counter", side_effect=AssertionError
        ):
            for request in ({"op": "graph"}, {"op": "graph", "profile_wall": False}):
                line = json.dumps(request)
                profile = request_profile(line)
                self.assertIsNone(profile)
                with scope(profile, "request_decode"):
                    self.assertEqual(json.loads(line), request)
                self.assertEqual(encode_reply(request, profile), line)

    def test_response_is_encoded_once_and_telemetry_is_not_simulation_data(self):
        reply = {"operations": [{"id": 0, "cost": 1.2345678901234567}]}
        profile = WallProfile()
        with (
            patch("rooflang_bridge.wall_profile.json.dumps", wraps=json.dumps) as dumps,
            patch("rooflang_bridge.wall_profile.perf_counter", side_effect=[1, 4]),
        ):
            payload = json.loads(encode_reply(reply, profile))
        self.assertEqual(dumps.call_count, 2)  # large reply once, small telemetry once
        self.assertIs(dumps.call_args_list[0].args[0], reply)
        self.assertEqual(
            payload.pop("wall_profile"),
            {"sections": {"response_encode": {"calls": 1, "seconds": 3.0}}},
        )
        self.assertEqual(payload, reply)
        self.assertNotIn("wall_profile", reply)

    def test_qwen_pricing_and_routes_preserve_exact_graph(self):
        template = QwenTemplate(1, 8, HW)
        with patch(
            "rooflang_bridge.wall_profile.perf_counter", side_effect=AssertionError
        ):
            original = template.export(BATCH, [0], details=False)
        profile = WallProfile()
        with profile.scope("graph_build_other"):
            measured = template.export(BATCH, [0], details=False, wall_profile=profile)
        self.assertEqual(measured, original)
        self.assertEqual(profile.sections["expert_routing"]["calls"], 94)
        self.assertGreater(profile.sections["operator_pricing"]["calls"], 94)

    def test_each_rpc_can_toggle_profile_without_changing_qwen_or_llama(self):
        graph = {"op": "graph", "batch": BATCH, "requests": [0], "details": False}
        for qwen in (True, False):
            with self.subTest(qwen=qwen):
                baseline_init, baseline_graph, baseline_next = worker(
                    [init(qwen), graph, graph]
                )
                ready, measured, disabled, failure = worker(
                    [
                        init(qwen) | {"profile_wall": True},
                        graph | {"profile_wall": True},
                        graph,
                        graph | {"batch": [], "profile_wall": True},
                    ]
                )
                init_stats = ready.pop("wall_profile")["sections"]
                graph_stats = measured.pop("wall_profile")["sections"]
                error_stats = failure.pop("wall_profile")["sections"]
                self.assertEqual(ready, baseline_init)
                self.assertEqual(measured, baseline_graph)
                self.assertEqual(disabled, baseline_next)
                self.assertIn("error", failure)
                self.assertIn("worker_init", init_stats)
                self.assertNotIn("worker_init", graph_stats)
                self.assertIn("graph_build_other", graph_stats)
                self.assertNotIn("expert_routing", error_stats)
                self.assertEqual("expert_routing" in graph_stats, qwen)
                for stats in (init_stats, graph_stats, error_stats):
                    self.assertIn("request_decode", stats)
                    self.assertIn("response_encode", stats)
                    for values in stats.values():
                        self.assertEqual(set(values), {"calls", "seconds"})
                        self.assertGreaterEqual(values["calls"], 1)
                        self.assertGreaterEqual(values["seconds"], 0)

    def test_audit_write_is_counted_separately_and_written_plan_has_no_clocks(self):
        with tempfile.TemporaryDirectory() as temp:
            audit = Path(temp) / "audit"
            ready, plan = worker(
                [
                    init() | {"profile_wall": True},
                    {
                        "op": "graph",
                        "batch": BATCH,
                        "requests": [0],
                        "profile_wall": True,
                    },
                ],
                "--audit-dir",
                str(audit),
            )
            for reply in (ready, plan):
                self.assertEqual(
                    reply["wall_profile"]["sections"]["audit_write"]["calls"], 1
                )
            stored = json.loads((audit / "00000.plan.json").read_text())
            self.assertNotIn("wall_profile", stored)
            self.assertEqual(stored["operations"], plan["operations"])


if __name__ == "__main__":
    unittest.main()
