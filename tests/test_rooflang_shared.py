"""Causal diagnostics for the shared bridge; costs are analytical, not measured."""

import csv
import json
import os
import subprocess
import sys
import tempfile
import tomllib
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from rooflang_bridge.configuration import toml_document, topology_pd_config
from rooflang_bridge.llama import describe
from rooflang_bridge.shared_report import export_shared


class SharedExecutionTests(unittest.TestCase):
    def run_case(
        self,
        name,
        *,
        p=1,
        d=1,
        outputs=3,
        count=1,
        pools=1,
        handoff="layer",
        slow=False,
        mutate=None,
        trace=True,
        budget=128,
        colocated=False,
        policy_stats=False,
    ):
        evidence = os.environ.get("SHARED_TEST_OUTPUT")
        if evidence:
            root = Path(evidence).resolve() / name
            root.mkdir(parents=True, exist_ok=False)
        else:
            tmp = tempfile.TemporaryDirectory(prefix="rooflang-shared-")
            self.addCleanup(tmp.cleanup)
            root = Path(tmp.name)
        leaf = tomllib.loads(
            (
                ROOT / "examples/scenarios/rooflang-llama3_1_8b/simthesizer.toml"
            ).read_text()
        )["system"]
        leaf["model_runner"]["command"] = [
            sys.executable,
            str(ROOT / "scripts/rooflang_worker.py"),
            "--audit-dir",
            "unused",
            "--audit-trace-rounds",
            "0",
        ]
        leaf["scheduler"].update(max_batch=1, max_num_batched_tokens=budget)
        topology = None
        if slow:
            topology = {
                "clusters": [
                    {"id": "p", "gpu_model": "h200"},
                    {"id": "d", "gpu_model": "h200"},
                ],
                "links": [
                    {
                        "src": "p/n0-nvswitch",
                        "dst": "d/n0-nvswitch",
                        "bandwidth_gbs": 0.01,
                        "latency_us": 3,
                    }
                ],
            }
        prefill = {"gpu_model": "h200", "instances": pools, "tp": p}
        decode = {"gpu_model": "h200", "instances": 1, "tp": d}
        if slow:
            prefill["gpu_ids"] = [f"p/n0-nvidia-h200-sxm-{i}" for i in range(p * pools)]
            decode["gpu_ids"] = [f"d/n0-nvidia-h200-sxm-{i}" for i in range(d)]
        if colocated:
            prefill["gpu_ids"] = ["cluster/n0-nvidia-h200-sxm-0"]
            decode["gpu_ids"] = ["cluster/n0-nvidia-h200-sxm-0"]
        config, instances = topology_pd_config(
            leaf,
            root,
            prefill,
            decode,
            topology,
            shared_execution=True,
            kv_handoff=handoff,
            allow_gpu_sharing=colocated,
        )
        if mutate:
            mutate(config["system"])
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
        command = [
            binary,
            "--config",
            str(root / "config.toml"),
            "--workload",
            str(root / "workload.jsonl"),
            "--output-jsonl",
            str(root / "result.jsonl"),
        ]
        if trace:
            command += [
                "--output-graph-trace",
                str(root / "outer.jsonl"),
                "--output-engine-trace",
                str(root / "state.jsonl"),
            ]
        if policy_stats:
            command += ["--output-policy-stats", str(root / "policy-stats.json")]
        result = subprocess.run(
            command, cwd=ROOT, capture_output=True, text=True, timeout=180, check=False
        )
        (root / "run.log").write_text(result.stdout + result.stderr)
        (root / "command.json").write_text(json.dumps(command))
        if result.returncode == 0 and trace:
            export_shared(root, instances)
        return root, result

    def test_isolated_tp_has_same_operator_service_as_rooflang(self):
        for tp in (1, 2, 4, 8):
            with self.subTest(tp=tp):
                root, result = self.run_case(f"isolated-tp{tp}", p=tp, d=1, outputs=1)
                self.assertEqual(result.returncode, 0, result.stderr)
                report = json.loads((root / "shared-check.json").read_text())
                self.assertEqual(report["kv_copies"], 0)
                rounds = list(
                    csv.DictReader((root / "rounds.csv").read_text().splitlines())
                )
                self.assertEqual(len(rounds), 1)
                self.assertAlmostEqual(
                    float(rounds[0]["duration_s"]),
                    float(rounds[0]["isolated_reference_s"]),
                    places=10,
                )

    def test_layer_handoff_overlaps_and_decode_waits_per_layer(self):
        runs = {}
        for mode in ("layer", "whole"):
            root, result = self.run_case(mode, slow=True, handoff=mode)
            self.assertEqual(result.returncode, 0, result.stderr)
            rounds = list(
                csv.DictReader((root / "rounds.csv").read_text().splitlines())
            )
            p_end = float(
                next(r["finish_s"] for r in rounds if r["instance_id"] == "p0")
            )
            copies = list(
                csv.DictReader((root / "kv-transfers.csv").read_text().splitlines())
            )
            self.assertEqual(sum(int(r["bytes"]) for r in copies), 655360)
            first_send = min(float(r["start_s"]) for r in copies)
            if mode == "layer":
                self.assertLess(first_send, p_end)
            else:
                self.assertGreaterEqual(first_send, p_end)
            ops = list(
                csv.DictReader((root / "operator-costs.csv").read_text().splitlines())
            )
            attention = next(
                r
                for r in ops
                if r["instance_id"] == "d0"
                and r["round"] == "0"
                and r["operator"] == "layer0.rank0.attention"
            )
            first_receive = max(
                float(r["finish_s"]) for r in copies if r["layer"] == "0"
            )
            self.assertGreaterEqual(
                float(attention["start_s"]) + 1e-12, max(p_end, first_receive)
            )
            if mode == "layer":
                self.assertLess(
                    float(attention["start_s"]),
                    max(float(r["finish_s"]) for r in copies),
                )
            runs[mode] = sum(
                json.loads((root / "result.jsonl").read_text())["sub_requests"][0][
                    "token_latency"
                ]
            )
        self.assertLess(runs["layer"], runs["whole"])

    def test_cross_tp_chunked_multi_request_lifecycles(self):
        for p, d in ((4, 2), (2, 4)):
            root, result = self.run_case(
                f"chunks-{p}-{d}", p=p, d=d, count=3, pools=2, budget=3, slow=True
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads((root / "shared-check.json").read_text())
            self.assertEqual(report["kv_sent_bytes"], 3 * 655360)
            self.assertEqual(report["live_kv_after_run"], 0)
            transfers = list(
                csv.DictReader((root / "kv-transfers.csv").read_text().splitlines())
            )
            operators = list(
                csv.DictReader((root / "operator-costs.csv").read_text().splitlines())
            )
            # Collective and KV demand IDs must actually intersect on physical ports.
            keys = {
                key
                for t in transfers
                for key in json.loads(t["resources"])
                if key.startswith("fabric:")
            }
            collective = {
                key
                for op in operators
                if "allreduce" in op["operator"]
                for demand in json.loads(op["resources"])
                for key in demand["resources"]
            }
            self.assertTrue(keys & collective)

    def test_receive_capacity_blocks_next_prefill_until_decode_releases(self):
        def constrain(system):
            scheduler = system["decode_instances"][0]["scheduler"]
            scheduler.update(
                device_mem=describe({"tp": 1, "pp": 1, "ep": 1})[
                    "weight_bytes_per_device"
                ]
                + 1_400_000,
                device_mem_utilization=1.0,
            )

        root, result = self.run_case("capacity", count=2, pools=2, mutate=constrain)
        self.assertEqual(result.returncode, 0, result.stderr)
        rounds = list(csv.DictReader((root / "rounds.csv").read_text().splitlines()))
        second = next(r for r in rounds if r["instance_id"] == "p1")
        self.assertGreater(float(second["admission_wait_s"]), 0)
        first_decode = [
            r for r in rounds if r["instance_id"] == "d0" and int(r["round"]) < 2
        ]
        self.assertGreaterEqual(
            float(second["start_s"]), max(float(r["finish_s"]) for r in first_decode)
        )

    def test_trace_does_not_change_request_results(self):
        on, a = self.run_case("trace-on", count=2, budget=3)
        off, b = self.run_case("trace-off", count=2, budget=3, trace=False)
        self.assertEqual(a.returncode, 0, a.stderr)
        self.assertEqual(b.returncode, 0, b.stderr)
        self.assertEqual(
            (on / "result.jsonl").read_bytes(), (off / "result.jsonl").read_bytes()
        )

    def test_colocated_instances_share_compute_and_hbm(self):
        root, result = self.run_case("colocated", count=2, colocated=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        rows = list(
            csv.DictReader((root / "operator-costs.csv").read_text().splitlines())
        )
        p = [r for r in rows if r["instance_id"] == "p0" and r["round"] == "1"]
        d = [r for r in rows if r["instance_id"] == "d0"]
        self.assertTrue(
            any(
                min(float(a["finish_s"]), float(b["finish_s"]))
                > max(float(a["start_s"]), float(b["start_s"]))
                for a in p
                for b in d
            )
        )
        self.assertTrue(
            any(
                float(r["duration_s"]) * 1e6
                > max(float(r[k]) for k in ("compute_us", "memory_us", "network_us"))
                * 1.1
                for r in p
                if float(r["duration_s"]) > 0
            )
        )

    def test_unsupported_cache_and_insufficient_capacity_fail(self):
        def cache(s):
            s["prefill_instances"][0]["scheduler"]["enable_prefix_caching"] = True

        _, result = self.run_case("reject-cache", mutate=cache)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("prefix caching", result.stderr)

        def oom(s):
            s["decode_instances"][0]["scheduler"].update(
                device_mem=1, device_mem_utilization=1.0
            )

        root, result = self.run_case("reject-oom", mutate=oom)
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse((root / "result.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
