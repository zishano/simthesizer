"""Native P/D handoff with independent Rooflang workers (synthetic link inputs)."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import tomllib
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from check_rooflang_timing import check
from draw_graph import load_trace
from draw_rooflang import assemble
from rooflang_bridge.configuration import pd_config, toml_document
from rooflang_bridge.observations import iteration_nodes


class RooflangPdTests(unittest.TestCase):
    def run_case(self, *, tp=2, count=1, pools=1, outputs=3, bw=1e9, change=None):
        temporary = tempfile.TemporaryDirectory(prefix="rooflang-pd-")
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        leaf = tomllib.loads(
            (
                ROOT / "examples/scenarios/rooflang-llama3_1_8b/simthesizer.toml"
            ).read_text()
        )["system"]
        runner = leaf["model_runner"]
        runner.update(
            tp=tp,
            command=[
                sys.executable,
                str(ROOT / "scripts/rooflang_worker.py"),
                "--audit-dir",
                "unused",
                "--audit-trace-rounds",
                "0",
            ],
        )
        runner["hardware"]["gpu_count"] = tp
        leaf["scheduler"].update(max_batch=1, max_num_batched_tokens=3)
        config, instances = pd_config(leaf, root, pools, pools, bw, 1e-6)
        if change:
            change(config)
        (root / "config.toml").write_text(toml_document(config))
        requests = [
            {
                "request_id": i,
                "arrival_time": 1.0,
                "initial": [0],
                "sub_requests": [
                    {
                        "kind": "llm",
                        "input_tokens": 5,
                        "output_tokens": outputs,
                        "next": [],
                    }
                ],
            }
            for i in range(count)
        ]
        (root / "workload.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in requests)
        )
        binary = os.environ.get(
            "SIMTHESIZER_BIN", str(ROOT / "target/debug/simthesizer")
        )
        result = subprocess.run(
            [
                binary,
                "--config",
                str(root / "config.toml"),
                "--workload",
                str(root / "workload.jsonl"),
                "--output-jsonl",
                str(root / "result.jsonl"),
                "--output-graph-trace",
                str(root / "outer.jsonl"),
                "--output-engine-trace",
                str(root / "state.jsonl"),
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=60,
        )
        return root, instances, result

    def test_kv_handoff_units_and_link_sensitivity(self):
        runs = []
        for bandwidth in [1e9, 1e7]:
            root, instances, result = self.run_case(bw=bandwidth)
            self.assertEqual(result.returncode, 0, result.stderr)
            outer = load_trace(root / "outer.jsonl")
            transfers = [n for n in outer["nodes"] if n["details"]["kind"] == "network"]
            self.assertEqual(len(transfers), 1)
            transfer = transfers[0]
            # FP16, K+V, 32 layers, 4 local KV heads at TP2, head width 128.
            expected_bytes = 5 * 2 * 32 * 4 * 128 * 2
            self.assertEqual(transfer["details"]["bytes"], expected_bytes)
            self.assertAlmostEqual(
                transfer["finish"] - transfer["dispatches"][0][1],
                1e-6 + expected_bytes / bandwidth,
                places=12,
            )
            for identity, audit, _ in instances:
                report, rows = check(root / "outer.jsonl", audit)
                self.assertEqual(report["instance_id"], identity)
                self.assertEqual(report["rounds_checked"], 2)
                self.assertEqual([r["round"] for r in report["detailed_rounds"]], [0])
                self.assertTrue(rows)
                self.assertTrue(all(r["instance_id"] == identity for r in rows))
            d_audit = instances[1][1]
            first = json.loads((d_audit / "00000.json").read_text())
            self.assertEqual(
                first["batch"], [{"q_len": 1, "kv_len": 5, "lm_head_len": 1}]
            )
            d_nodes = iteration_nodes(
                outer, json.loads((d_audit / "init.json").read_text())
            )
            self.assertGreaterEqual(d_nodes[0]["dispatches"][0][1], transfer["finish"])
            graph = assemble(root / "outer.jsonl", d_audit)
            self.assertEqual(
                [r["outer_node"] for r in graph["rounds"]], [n["id"] for n in d_nodes]
            )
            latencies = json.loads((root / "result.jsonl").read_text())["sub_requests"][
                0
            ]["token_latency"]
            self.assertEqual(len(latencies), 3)
            runs.append(latencies)
        self.assertAlmostEqual(runs[0][0], runs[1][0], places=12)
        self.assertAlmostEqual(
            runs[1][1] - runs[0][1], 327680 * (1 / 1e7 - 1 / 1e9), places=12
        )
        self.assertAlmostEqual(runs[0][2], runs[1][2], places=12)

    def test_pools_route_independently_and_overlap(self):
        root, instances, result = self.run_case(count=4, pools=2)
        self.assertEqual(result.returncode, 0, result.stderr)
        outer = load_trace(root / "outer.jsonl")
        starts = {}
        for identity, audit, _ in instances:
            report, _ = check(root / "outer.jsonl", audit)
            self.assertEqual(report["rounds_checked"], 4)
            nodes = iteration_nodes(
                outer, json.loads((audit / "init.json").read_text())
            )
            starts[identity] = nodes[0]["dispatches"][0][1]
        self.assertEqual(starts["p0"], starts["p1"])
        self.assertEqual(starts["d0"], starts["d1"])
        transfers = [n for n in outer["nodes"] if n["details"]["kind"] == "network"]
        self.assertEqual(len(transfers), 4)
        self.assertEqual({n["details"]["src"] for n in transfers}, {0, 1})
        self.assertEqual({n["details"]["dst"] for n in transfers}, {2, 3})
        states = [
            json.loads(line)["state"]
            for line in (root / "state.jsonl").read_text().splitlines()
        ]
        self.assertTrue(
            any(
                any(
                    s["scheduler"]["running_requests"] for s in row["prefill_instances"]
                )
                and any(
                    s["scheduler"]["running_requests"] for s in row["decode_instances"]
                )
                for row in states
            )
        )

    def test_one_output_skips_transfer_and_decode_for_supported_tp(self):
        for tp in [1, 2, 4, 8]:
            with self.subTest(tp=tp):
                root, instances, result = self.run_case(tp=tp, outputs=1)
                self.assertEqual(result.returncode, 0, result.stderr)
                outer = load_trace(root / "outer.jsonl")
                self.assertFalse(
                    any(n["details"]["kind"] == "network" for n in outer["nodes"])
                )
                report, rows = check(root / "outer.jsonl", instances[1][1])
                self.assertEqual(report["rounds_checked"], 0)
                self.assertEqual(rows, [])

    def test_decode_handoff_for_other_tp_degrees(self):
        for tp in [1, 4, 8]:
            with self.subTest(tp=tp):
                root, instances, result = self.run_case(tp=tp)
                self.assertEqual(result.returncode, 0, result.stderr)
                report, rows = check(root / "outer.jsonl", instances[1][1])
                self.assertEqual(report["rounds_checked"], 2)
                self.assertEqual(len({row["device"] for row in rows}), tp)
                networks = [
                    n
                    for n in load_trace(root / "outer.jsonl")["nodes"]
                    if n["details"]["kind"] == "network"
                ]
                self.assertEqual(networks[0]["details"]["bytes"], 5 * 131072 // tp)

    def test_invalid_pool_identity_and_redistribution_fail_before_workers(self):
        for field, value, expected in [
            ("instance_id", "p0", "unique"),
            ("tp", 1, "TP must match"),
        ]:
            root, _, result = self.run_case(
                change=lambda config: config["system"]["decode_instances"][0][
                    "model_runner"
                ].update({field: value})
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(expected, result.stderr)
            self.assertFalse((root / "instances").exists())
