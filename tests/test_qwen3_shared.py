"""Native Qwen EP/P-D checks; analytical diagnostics, not measured GPU accuracy."""

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
from results import physical_gpu_count
from rooflang_bridge.configuration import toml_document, topology_pd_config
from rooflang_bridge.shared_report import export_shared


def leaf_config():
    leaf = tomllib.loads(
        (ROOT / "examples/scenarios/rooflang-llama3_1_8b/simthesizer.toml").read_text()
    )["system"]
    leaf["model"] = {"kind": "qwen3_235b_a22b"}
    leaf["model_runner"].update(
        graph_model="qwen3_235b_a22b",
        routing_seed=7,
        command=[sys.executable, str(ROOT / "scripts/rooflang_worker.py")],
    )
    return leaf


class QwenSharedTests(unittest.TestCase):
    def case(
        self,
        name,
        *,
        p=8,
        d=8,
        handoff="whole",
        outputs=(3,),
        budget=128,
        trace=True,
        gpu="h200",
        slow=False,
        ptp=1,
        dtp=1,
        p_replicas=1,
        d_replicas=1,
        request_ids=None,
    ):
        evidence = os.environ.get("QWEN_TEST_OUTPUT")
        if evidence:
            root = Path(evidence).resolve() / name
            root.mkdir(parents=True, exist_ok=False)
        else:
            temporary = tempfile.TemporaryDirectory(prefix="qwen-ep-")
            self.addCleanup(temporary.cleanup)
            root = Path(temporary.name)
        leaf = leaf_config()
        leaf["scheduler"].update(max_batch=8, max_num_batched_tokens=budget)
        if trace:
            leaf["model_runner"]["command"] += [
                "--audit-dir",
                "unused",
                "--audit-trace-rounds",
                "0",
            ]
        prefill = {"gpu_model": gpu, "instances": p_replicas, "tp": ptp, "ep": p}
        decode = {"gpu_model": gpu, "instances": d_replicas, "tp": dtp, "ep": d}
        topology = None
        if slow:
            topology = {
                "clusters": [{"id": side, "gpu_model": gpu} for side in ("p", "d")],
                "links": [
                    {
                        "src": "p/n0-nvswitch",
                        "dst": "d/n0-nvswitch",
                        "bandwidth_gbs": 0.01,
                        "latency_us": 3,
                    }
                ],
            }
            for pool, side, count in ((prefill, "p", p), (decode, "d", d)):
                pool["gpu_ids"] = [
                    f"{side}/n0-nvidia-h200-sxm-{rank}" for rank in range(count)
                ]
        config, instances = topology_pd_config(
            leaf,
            root,
            prefill,
            decode,
            topology,
            shared_execution=True,
            kv_handoff=handoff,
        )
        (root / "config.toml").write_text(toml_document(config))
        workload = [
            {
                "request_id": rank + 7 if request_ids is None else request_ids[rank],
                "arrival_time": 1.0,
                "initial": [0],
                "sub_requests": [
                    {"kind": "llm", "input_tokens": 5, "output_tokens": n, "next": []}
                ],
            }
            for rank, n in enumerate(outputs)
        ]
        (root / "workload.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in workload)
        )
        command = [
            os.environ.get("SIMTHESIZER_BIN", str(ROOT / "target/debug/simthesizer")),
            "--config",
            str(root / "config.toml"),
            "--workload",
            str(root / "workload.jsonl"),
            "--output-jsonl",
            str(root / "result.jsonl"),
        ]
        if trace:
            command += ["--output-graph-trace", str(root / "outer.jsonl")]
        result = subprocess.run(
            command, cwd=ROOT, capture_output=True, text=True, timeout=240, check=False
        )
        (root / "run.log").write_text(result.stdout + result.stderr)
        (root / "command.json").write_text(json.dumps(command))
        if result.returncode == 0 and trace:
            export_shared(root, instances)
        return root, result, config

    def test_group_counts_and_rejected_combinations(self):
        for ep in (1, 2, 4, 8):
            for tp in (1, 2, 4):
                if tp > ep:
                    continue
                p = {"gpu_model": "h200", "instances": 2, "tp": tp, "ep": ep}
                d = {"gpu_model": "h200", "instances": 1, "tp": tp, "ep": ep}
                config, _ = topology_pd_config(
                    leaf_config(), Path("unused"), p, d, shared_execution=True
                )
                self.assertEqual(physical_gpu_count(config), 3 * ep)
                topology = config["system"]["kv_topology"]
                self.assertEqual(topology["shards"], 4)
                self.assertEqual(
                    [len(group) for group in topology["prefill_groups"][0]],
                    [tp] * (ep // tp),
                )
                devices = [
                    topology["devices"][device]
                    for side in ("prefill", "decode")
                    for replica in topology[f"{side}_groups"]
                    for group in replica
                    for device in group
                ]
                self.assertEqual(len(set(devices)), 3 * ep)
                self.assertEqual(
                    devices,
                    [
                        device
                        for side in ("prefill", "decode")
                        for replica in config["system"][f"{side}_instances"]
                        for device in replica["model_runner"]["hardware"]["gpu_ids"]
                    ],
                )
        for change in (
            {"tp": 8},
            {"tp": 4, "ep": 2},
            {"ep": 3},
            {"expert_tp": 2},
            {"pp": 2},
            {"gpu_count": 1},
        ):
            p = {"gpu_model": "h200", "instances": 1, "tp": 1, "ep": 8}
            with self.assertRaises(ValueError):
                topology_pd_config(
                    leaf_config(),
                    Path("unused"),
                    p,
                    dict(p, **change),
                    shared_execution=True,
                )

    def test_ep1_h200_cannot_hide_resident_weights(self):
        root, result, _ = self.case("oom-ep1", p=1, d=1, trace=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse((root / "result.jsonl").exists())
        self.assertRegex(result.stderr.lower(), "memory|capacity|weight|fit")

    def test_ep_group_can_span_connected_preset_clusters(self):
        from rooflang_bridge.qwen3 import QwenTemplate

        topology = {
            "clusters": [{"id": name, "gpu_model": "h200"} for name in ("a", "b")],
            "links": [
                {
                    "src": "a/n0-nvswitch",
                    "dst": "b/n0-nvswitch",
                    "bandwidth_gbs": 50,
                    "latency_us": 3,
                }
            ],
        }
        names = lambda side, first: [
            f"{side}/n0-nvidia-h200-sxm-{i}" for i in range(first, first + 4)
        ]
        p = {
            "gpu_model": "h200",
            "instances": 1,
            "tp": 1,
            "ep": 8,
            "gpu_ids": names("a", 0) + names("b", 0),
        }
        d = dict(p, gpu_ids=names("a", 4) + names("b", 4))
        config, _ = topology_pd_config(
            leaf_config(), Path("unused"), p, d, topology, shared_execution=True
        )
        hardware = config["system"]["prefill_instances"][0]["model_runner"]["hardware"]
        graph = QwenTemplate(1, 8, hardware).export(
            [{"q_len": 1, "kv_len": 0, "lm_head_len": 1}],
            [0],
            routing=[[64, 65, 66, 67, 68, 69, 70, 71]],
        )
        copy = next(
            op
            for op in graph["operations"]
            if op["work"]["details"]["name"] == "layer0.dispatch.0-4"
        )
        self.assertTrue(
            any(
                "bridge" in resource
                for resource in copy["work"]["demands"][0]["resources"]
            )
        )
        with self.assertRaisesRegex(ValueError, "No path"):
            topology_pd_config(
                leaf_config(),
                Path("unused"),
                p,
                d,
                dict(topology, links=[]),
                shared_execution=True,
            )

    def test_unequal_ep_chunking_and_kv_ownership(self):
        root, result, config = self.case(
            "unequal-chunks", p=4, d=8, budget=3, outputs=(3, 1, 0)
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        rows = [
            json.loads(line)
            for line in (root / "result.jsonl").read_text().splitlines()
        ]
        self.assertEqual(
            [len(r["sub_requests"][0]["token_latency"]) for r in rows], [3, 1, 0]
        )
        report = json.loads((root / "shared-check.json").read_text())
        self.assertEqual(
            report["kv_sent_bytes"], 962560
        )  # 5 * 94 * 4 * 128 * K/V * BF16
        self.assertEqual(report["kv_received_bytes"], 962560)
        self.assertEqual(report["live_kv_after_run"], 0)
        topology = config["system"]["kv_topology"]
        lifecycle = list(
            csv.DictReader((root / "kv-lifecycle.csv").read_text().splitlines())
        )
        for row in lifecycle:
            self.assertEqual((int(row["head_start"]), int(row["head_end"])), (0, 4))
            is_prefill = row["prefill"] == "True"
            groups = topology["prefill_groups" if is_prefill else "decode_groups"][0]
            expected = topology["devices"][groups[int(row["request"]) % len(groups)][0]]
            self.assertEqual(row["device"], expected)
        self.assertEqual(physical_gpu_count(config), 12)

    def test_handoff_dependencies_and_export_equivalence(self):
        timings = {}
        for mode in ("whole", "layer"):
            root, result, _ = self.case("slow-" + mode, handoff=mode, slow=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            rounds = list(
                csv.DictReader((root / "rounds.csv").read_text().splitlines())
            )
            p_end = float(
                next(row["finish_s"] for row in rounds if row["instance_id"] == "p0")
            )
            copies = list(
                csv.DictReader((root / "kv-transfers.csv").read_text().splitlines())
            )
            first_send = min(float(row["start_s"]) for row in copies)
            if mode == "layer":
                self.assertLess(first_send, p_end)
            else:
                self.assertGreaterEqual(first_send, p_end)
            ops = list(
                csv.DictReader((root / "operator-costs.csv").read_text().splitlines())
            )
            attention = next(
                row
                for row in ops
                if row["instance_id"] == "d0"
                and row["round"] == "0"
                and row["operator"] == "layer0.rank7.attention"
            )
            received = max(
                float(row["finish_s"]) for row in copies if row["layer"] == "0"
            )
            self.assertGreaterEqual(
                float(attention["start_s"]) + 1e-12, max(p_end, received)
            )
            timings[mode] = sum(
                json.loads((root / "result.jsonl").read_text())["sub_requests"][0][
                    "token_latency"
                ]
            )
            if mode == "whole":
                plain, run, _ = self.case(
                    "slow-whole-no-export", handoff=mode, slow=True, trace=False
                )
                self.assertEqual(run.returncode, 0, run.stderr)
                self.assertEqual(
                    (plain / "result.jsonl").read_bytes(),
                    (root / "result.jsonl").read_bytes(),
                )
        self.assertLess(timings["layer"], timings["whole"])

    def test_ep2_b300_prefill(self):
        _, result, config = self.case(
            "ep2-b300", p=2, d=2, gpu="b300", outputs=(1,), trace=False
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(physical_gpu_count(config), 4)

    def test_attention_tp_reshards_chunked_kv_without_duplicating_heads(self):
        for ptp, dtp in ((1, 2), (2, 1), (2, 4)):
            with self.subTest(ptp=ptp, dtp=dtp):
                root, result, config = self.case(
                    f"tp{ptp}-tp{dtp}-chunks",
                    ptp=ptp,
                    dtp=dtp,
                    budget=3,
                    outputs=(3, 1, 0),
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                rows = [
                    json.loads(line)
                    for line in (root / "result.jsonl").read_text().splitlines()
                ]
                self.assertEqual(
                    [len(row["sub_requests"][0]["token_latency"]) for row in rows],
                    [3, 1, 0],
                )
                report = json.loads((root / "shared-check.json").read_text())
                self.assertEqual(report["kv_sent_bytes"], 962560)
                self.assertEqual(report["kv_received_bytes"], 962560)
                self.assertEqual(report["live_kv_after_run"], 0)
                topology = config["system"]["kv_topology"]
                received_cells = set()
                for row in csv.DictReader(
                    (root / "kv-lifecycle.csv").read_text().splitlines()
                ):
                    side = "prefill" if row["prefill"] == "True" else "decode"
                    groups = topology[f"{side}_groups"][0]
                    group = groups[int(row["request"]) % len(groups)]
                    names = [topology["devices"][device] for device in group]
                    self.assertIn(row["device"], names)
                    rank = names.index(row["device"])
                    width = 4 // len(group)
                    start, end = int(row["head_start"]), int(row["head_end"])
                    self.assertLess(start, end)
                    self.assertGreaterEqual(start, rank * width)
                    self.assertLessEqual(end, (rank + 1) * width)
                    if row["transition"] == "receive":
                        for token in range(
                            int(row["token_start"]), int(row["token_end"])
                        ):
                            for head in range(start, end):
                                cell = (
                                    int(row["request"]),
                                    int(row["layer"]),
                                    token,
                                    head,
                                )
                                self.assertNotIn(cell, received_cells)
                                received_cells.add(cell)
                self.assertEqual(
                    received_cells,
                    {
                        (7, layer, token, head)
                        for layer in range(94)
                        for token in range(5)
                        for head in range(4)
                    },
                )

    def test_tp2_ep8_two_replicas_use_32_gpus_and_all_attention_groups(self):
        root, result, config = self.case(
            "tp2-ep8-dp2",
            ptp=2,
            dtp=2,
            p_replicas=2,
            d_replicas=2,
            outputs=(3,) * 8,
            request_ids=(0, 4, 1, 5, 2, 6, 3, 7),
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(physical_gpu_count(config), 32)
        rounds = list(csv.DictReader((root / "rounds.csv").read_text().splitlines()))
        self.assertEqual(
            {row["instance_id"] for row in rounds}, {"p0", "p1", "d0", "d1"}
        )
        ops = list(
            csv.DictReader((root / "operator-costs.csv").read_text().splitlines())
        )
        for identity in ("p0", "p1", "d0", "d1"):
            names = {row["operator"] for row in ops if row["instance_id"] == identity}
            self.assertTrue(
                all(f"layer0.rank{rank}.attention" in names for rank in range(8))
            )
        report = json.loads((root / "shared-check.json").read_text())
        self.assertEqual(report["kv_sent_bytes"], 8 * 962560)
        self.assertEqual(report["kv_received_bytes"], 8 * 962560)
        self.assertEqual(report["live_kv_after_run"], 0)

    def test_tp2_handoff_gates_each_shard_and_reports_preserve_results(self):
        timings = {}
        for handoff in ("whole", "layer"):
            root, result, _ = self.case(
                f"tp2-slow-{handoff}", ptp=2, dtp=2, handoff=handoff, slow=True
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            rounds = list(
                csv.DictReader((root / "rounds.csv").read_text().splitlines())
            )
            p_end = float(
                next(row["finish_s"] for row in rounds if row["instance_id"] == "p0")
            )
            copies = list(
                csv.DictReader((root / "kv-transfers.csv").read_text().splitlines())
            )
            first_send = min(float(row["start_s"]) for row in copies)
            if handoff == "layer":
                self.assertLess(first_send, p_end)
            else:
                self.assertGreaterEqual(first_send, p_end)
            ops = list(
                csv.DictReader((root / "operator-costs.csv").read_text().splitlines())
            )
            for rank in (6, 7):
                op = next(
                    row
                    for row in ops
                    if row["instance_id"] == "d0"
                    and row["round"] == "0"
                    and row["operator"] == f"layer0.rank{rank}.attention"
                )
                received = max(
                    float(row["finish_s"])
                    for row in copies
                    if row["layer"] == "0" and row["dst"] == op["devices"]
                )
                self.assertGreaterEqual(
                    float(op["start_s"]) + 1e-12, max(received, p_end)
                )
            timings[handoff] = sum(
                json.loads((root / "result.jsonl").read_text())["sub_requests"][0][
                    "token_latency"
                ]
            )
            if handoff == "whole":
                plain, run, _ = self.case(
                    "tp2-slow-whole-plain", ptp=2, dtp=2, slow=True, trace=False
                )
                self.assertEqual(run.returncode, 0, run.stderr)
                self.assertEqual(
                    (root / "result.jsonl").read_bytes(),
                    (plain / "result.jsonl").read_bytes(),
                )
        self.assertLess(timings["layer"], timings["whole"])


if __name__ == "__main__":
    unittest.main()
