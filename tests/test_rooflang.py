"""In-tree Rooflang tests: uv run --locked python -m unittest discover -s tests."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import rooflang
from rooflang.language.kernels.comm import AllGather, AllReduce
from rooflang_bridge.llama import build, describe, simulate

HW = {
    "tflops_fp16": 100,
    "memory_bandwidth_gbs": 1000,
    "memory_capacity_gb": 96,
    "link_bandwidth_gbs": 100,
    "link_latency_us": 1,
}


def batch(*pairs):
    return [{"q_len": q, "kv_len": k, "lm_head_len": 1} for q, k in pairs]


class RooflangGraphTests(unittest.TestCase):
    def test_ragged_causal_work_does_not_use_mean_context(self):
        g, _, _, summary = build(batch((4, 0), (1, 16)), 1, HW)
        self.assertEqual(summary["causal_pairs"], 27)
        self.assertEqual(summary["tokens"], 5)
        self.assertEqual(summary["logits_rows"], 2)
        attention = [
            k for k in g.kernels if getattr(k, "name", "").endswith(".attention")
        ]
        self.assertEqual(len(attention), 32)
        self.assertEqual(sum(k.flops for k in attention), 32 * 32 * (4 * 128 + 5) * 27)
        _, _, _, other = build(batch((1, 0), (4, 16)), 1, HW)
        self.assertEqual(other["causal_pairs"], 75)
        self.assertNotEqual(summary["causal_pairs"], other["causal_pairs"])

    def test_tp_collectives_and_link_sensitivity(self):
        g, _, _, _ = build(batch((8, 0)), 2, HW)
        reductions = [k for k in g.kernels if isinstance(k, AllReduce)]
        self.assertEqual(len(reductions), 65)
        self.assertTrue(all(k.total_bytes == 8 * 4096 * 2 for k in reductions))
        gathers = [k for k in g.kernels if isinstance(k, AllGather)]
        self.assertEqual(len(gathers), 1)
        self.assertEqual(gathers[0].total_bytes, 128256 * 2)
        fast, fast_summary = simulate(batch((8, 0)), 2, HW)
        slow, _ = simulate(batch((8, 0)), 2, {**HW, "link_bandwidth_gbs": 1})
        self.assertGreater(slow.total_time_us, fast.total_time_us)
        self.assertEqual(fast_summary["collective_count"], 66)
        # All collective participants have identical completion times.
        for k in {e.kernel for e in fast.trace if isinstance(e.kernel, AllReduce)}:
            self.assertEqual(len({e.end_us for e in fast.trace if e.kernel is k}), 1)

    def test_hardware_units_and_invalid_input(self):
        a, _ = simulate(batch((8, 0)), 1, HW)
        b, _ = simulate(
            batch((8, 0)), 1, {**HW, "tflops_fp16": 200, "memory_bandwidth_gbs": 2000}
        )
        self.assertAlmostEqual(a.total_time_us, 2 * b.total_time_us, places=6)
        from rooflang.runtime.simulator import OOMError

        with self.assertRaises(OOMError):
            simulate(batch((8, 0)), 1, {**HW, "memory_capacity_gb": 1})
        with self.assertRaises(ValueError):
            build(batch((0, 1)), 1, HW)
        with self.assertRaises(ValueError):
            build(batch((1, 1)), 3, HW)

    def test_original_hardware_presets_and_card_allocation(self):
        from rooflang.programs.presets.h200 import H200Cluster
        from rooflang.programs.presets.rtx6000d import RTX6000DCluster
        from rooflang_bridge.hardware import build_hardware

        for model, cls, flops, capacity, bandwidth, collective_bw in [
            ("rtx6000d", RTX6000DCluster, 148.0, 84.0, 1398.0, 100.0),
            ("h200", H200Cluster, 989.5, 144.0, 4800.0, 450.0),
        ]:
            config = {"gpu_model": model, "gpu_count": 2}
            hw, gpu, mem, info = build_hardware(config, 2)
            self.assertIsInstance(hw, cls)
            self.assertEqual(len(gpu), 2)
            self.assertEqual(info["topology_gpu_count"], 8)
            self.assertTrue(all(d.tflops["fp16"] == flops for d in gpu))
            self.assertTrue(all(m.capacity_gb == capacity for m in mem))
            self.assertEqual(info["devices"][0]["memory_bandwidth_gbs"], bandwidth)
            self.assertEqual(hw.find_aggregate_bandwidth(gpu), collective_bw)
            result, summary = simulate(batch((3, 0)), 2, config)
            self.assertGreater(result.total_time_us, 0)
            self.assertEqual(summary["hardware"], info)
            self.assertEqual(
                {t.device.name for t in result.trace}, {d.name for d in gpu}
            )
        for config in [
            {"gpu_model": "rtx6000d", "gpu_count": 1},
            {"gpu_model": "unknown", "gpu_count": 2},
        ]:
            with self.assertRaises(ValueError):
                build_hardware(config, 2)


class RooflangIntegrationTests(unittest.TestCase):
    def run_case(
        self,
        rows,
        tp=1,
        budget=3,
        broken_model=False,
        python_model=False,
        model_command=None,
    ):
        root = Path(tempfile.mkdtemp(prefix="rooflang-test-"))
        self.addCleanup(lambda: __import__("shutil").rmtree(root))
        source = (
            ROOT / "examples/scenarios/rooflang-llama3_1_8b/simthesizer.toml"
        ).read_text()
        command = [
            sys.executable,
            str(ROOT / "scripts/rooflang_worker.py"),
            "--audit-dir",
            str(root / "audit"),
        ]
        source = "\n".join(
            "command = " + json.dumps(command) if l.startswith("command =") else l
            for l in source.splitlines()
        )
        source = (
            source.replace("tp = 2", f"tp = {tp}")
            .replace("gpu_count = 2", f"gpu_count = {tp}")
            .replace(
                "max_num_batched_tokens = 2048", f"max_num_batched_tokens = {budget}"
            )
        )
        if python_model or model_command is not None:
            model_command = (
                model_command
                if model_command is not None
                else [
                    sys.executable,
                    str(ROOT / "scripts/rooflang_model.py"),
                    "--module",
                    "rooflang_bridge.llama",
                ]
            )
            source = source.split("[system.model]", 1)[0] + (
                '[system.model]\nkind = "python"\ncommand = '
                + json.dumps(model_command)
                + "\n"
            )
        if broken_model:
            source = source.replace("num_hidden_layers = 32", "num_hidden_layers = 31")
        (root / "config.toml").write_text(source)
        (root / "workload.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in rows)
        )
        binary = Path(
            os.environ.get("SIMTHESIZER_BIN", str(ROOT / "target/debug/simthesizer"))
        )
        result = subprocess.run(
            [
                str(binary),
                "--config",
                str(root / "config.toml"),
                "--workload",
                str(root / "workload.jsonl"),
                "--output-jsonl",
                str(root / "result.jsonl"),
                "--output-graph-trace",
                str(root / "graph.jsonl"),
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )
        return root, result

    def request(self, i, inputs, outputs):
        return {
            "request_id": i,
            "arrival_time": 1.0,
            "initial": [0],
            "sub_requests": [
                {
                    "kind": "llm",
                    "input_tokens": inputs,
                    "output_tokens": outputs,
                    "next": [],
                }
            ],
        }

    def test_chunked_prefill_commits_and_microseconds_conversion(self):
        for tp in [1, 2]:
            root, run = self.run_case([self.request(0, 5, 3)], tp)
            self.assertEqual(run.returncode, 0, run.stderr)
            rounds = [
                json.loads(p.read_text())
                for p in sorted((root / "audit").glob("[0-9]*.json"))
                if ".trace." not in p.name
            ]
            self.assertEqual(
                [(r["batch"][0]["q_len"], r["batch"][0]["kv_len"]) for r in rounds],
                [(3, 0), (2, 3), (1, 5), (1, 6)],
            )
            values = json.loads((root / "result.jsonl").read_text())["sub_requests"][0][
                "token_latency"
            ]
            self.assertEqual(len(values), 3)
            expected = [
                (rounds[0]["duration_us"] + rounds[1]["duration_us"]) * 1e-6,
                rounds[2]["duration_us"] * 1e-6,
                rounds[3]["duration_us"] * 1e-6,
            ]
            for a, b in zip(values, expected):
                self.assertAlmostEqual(a, b, places=12)
            self.assertTrue(
                all(r["collective_count"] == (0 if tp == 1 else 66) for r in rounds)
            )
            graph = [
                json.loads(l) for l in (root / "graph.jsonl").read_text().splitlines()
            ]
            labels = [r.get("details", {}).get("label", "") for r in graph]
            self.assertEqual(sum("RooflangIteration" in s for s in labels), 4)
            self.assertTrue(
                all(
                    f"{tp} x NVIDIA RTX 6000D" in label
                    for label in labels
                    if "RooflangIteration" in label
                )
            )

    def test_mixed_batch_and_model_rejection(self):
        root, run = self.run_case([self.request(0, 2, 3), self.request(1, 4, 2)])
        self.assertEqual(run.returncode, 0, run.stderr)
        rounds = [
            json.loads(p.read_text())
            for p in sorted((root / "audit").glob("[0-9]*.json"))
            if ".trace." not in p.name
        ]
        self.assertTrue(
            any(
                any(x["q_len"] == 1 and x["kv_len"] > 0 for x in r["batch"])
                and any(x["q_len"] > 1 for x in r["batch"])
                for r in rounds
            )
        )
        rows = [json.loads(s) for s in (root / "result.jsonl").read_text().splitlines()]
        self.assertEqual(
            [len(r["sub_requests"][0]["token_latency"]) for r in rows], [3, 2]
        )
        root, run = self.run_case([self.request(0, 2, 3)], broken_model=True)
        self.assertNotEqual(run.returncode, 0)
        self.assertIn("exactly FP16 Llama", run.stderr)
        self.assertFalse((root / "result.jsonl").exists())

    def test_in_tree_source_and_python_rust_model_equivalence(self):
        self.assertEqual(
            Path(rooflang.__file__).resolve().parent, ROOT / "scripts/rooflang"
        )
        rows = [self.request(0, 2, 3), self.request(1, 4, 2)]
        for tp in (1, 2, 4, 8):
            rust, run = self.run_case(rows, tp)
            self.assertEqual(run.returncode, 0, run.stderr)
            python, run = self.run_case(rows, tp, python_model=True)
            self.assertEqual(run.returncode, 0, run.stderr)
            for path in (rust / "audit").glob("*.json"):
                if ".trace." not in path.name:
                    self.assertEqual(
                        json.loads(path.read_text()),
                        json.loads((python / "audit" / path.name).read_text()),
                    )
            self.assertEqual(
                (rust / "result.jsonl").read_text(),
                (python / "result.jsonl").read_text(),
            )

    def test_python_model_protocol_and_memory_mismatch_fail(self):
        descriptor = describe({"tp": 1, "pp": 1, "ep": 1})
        cases = [
            ("print('not json')", "invalid Python model definition"),
            ("raise SystemExit(3)", "Python model definition failed"),
        ]
        for changes, error in [
            ({"version": 2}, "unsupported Python model protocol"),
            ({"layers": 0}, "empty Python model"),
            (
                {"weight_bytes_per_device": descriptor["weight_bytes_per_device"] + 1},
                "model memory metadata does not match Llama graph",
            ),
        ]:
            cases.append(
                ("print(" + repr(json.dumps({**descriptor, **changes})) + ")", error)
            )
        for code, error in cases:
            with self.subTest(error=error):
                root, run = self.run_case(
                    [self.request(0, 2, 1)], model_command=[sys.executable, "-c", code]
                )
                self.assertNotEqual(run.returncode, 0)
                self.assertIn(error, run.stderr)
                self.assertFalse((root / "result.jsonl").exists())

    def test_multi_instance_is_rejected_before_worker_launch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.toml").write_text(
                'system = { kind = "multi_instance", instances = [{ model_runner = { kind = "rooflang" } }] }'
            )
            (root / "workload.jsonl").write_text("")
            result = subprocess.run(
                [
                    str(ROOT / "target/debug/simthesizer"),
                    "--config",
                    str(root / "config.toml"),
                    "--workload",
                    str(root / "workload.jsonl"),
                    "--output-jsonl",
                    str(root / "out.jsonl"),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("single_instance root", result.stderr)
