"""Reference-guided checks for physical allocations and cross-TP KV handoff."""

import csv
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from draw_graph import load_trace
from rooflang_bridge.configuration import topology_pd_config
from rooflang_bridge.hardware import build_hardware
from rooflang_bridge.observations import iteration_nodes
from rooflang_bridge.pd_report import export_transfers
from rooflang_bridge.topology import build_topology, default_topology
import test_rooflang_pd


class RooflangTopologyTests(unittest.TestCase):
    def run_case(
        self,
        p,
        d,
        *,
        count=1,
        outputs=3,
        topology=None,
        p_model="rtx6000d",
        d_model="rtx6000d",
        mutate=None,
        p_instances=1,
        d_instances=1,
    ):
        def configure(config):
            leaf = config["system"]["prefill_instances"][0]
            root = Path(leaf["model_runner"]["command"][3]).parents[2]
            document, _ = topology_pd_config(
                leaf,
                root,
                {
                    "gpu_model": p_model,
                    "instances": p_instances,
                    "tp": p,
                    "max_num_batched_tokens": 3,
                },
                {
                    "gpu_model": d_model,
                    "instances": d_instances,
                    "tp": d,
                    "max_num_batched_tokens": 2,
                },
                topology=topology,
            )
            config.clear()
            config.update(document)
            if mutate:
                mutate(config)

        return test_rooflang_pd.RooflangPdTests.run_case(
            self, tp=p, count=count, outputs=outputs, change=configure
        )

    def test_all_tp_pairs_conserve_heads_bytes_and_gate_decode(self):
        for p in (1, 2, 4, 8):
            for d in (1, 2, 4, 8):
                with self.subTest(prefill=p, decode=d):
                    root, instances, result = self.run_case(p, d)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    outer = load_trace(root / "outer.jsonl")
                    networks = [
                        n for n in outer["nodes"] if n["details"]["kind"] == "network"
                    ]
                    self.assertEqual(len(networks), max(p, d))
                    # 5 tokens × K,V × 32 layers × 8 heads × 128 dimensions × FP16.
                    self.assertEqual(
                        sum(n["details"]["bytes"] for n in networks), 655360
                    )
                    self.assertEqual(
                        {n["details"]["bytes"] for n in networks}, {655360 // max(p, d)}
                    )
                    for field, ranks in [("src", p), ("dst", d)]:
                        totals = {}
                        for n in networks:
                            gpu = n["details"][field]
                            totals[gpu] = totals.get(gpu, 0) + n["details"]["bytes"]
                        self.assertEqual(len(totals), ranks)
                        self.assertEqual(set(totals.values()), {655360 // ranks})
                    expected = 655360 / min(p, d) / 100e9 + 2e-6
                    for n in networks:
                        self.assertAlmostEqual(
                            n["finish"] - n["dispatches"][0][1], expected, places=11
                        )
                    config = json.loads((instances[1][1] / "init.json").read_text())
                    d_nodes = iteration_nodes(outer, config)
                    self.assertGreaterEqual(
                        d_nodes[0]["dispatches"][0][1],
                        max(n["finish"] for n in networks),
                    )
                    self.assertEqual(config["tp"], d)
                    p_init = json.loads((instances[0][1] / "init.json").read_text())
                    self.assertTrue(
                        set(p_init["hardware"]["gpu_ids"]).isdisjoint(
                            config["hardware"]["gpu_ids"]
                        )
                    )
                    import tomllib

                    system = tomllib.loads((root / "config.toml").read_text())["system"]
                    export_transfers(root, system)
                    rows = list(
                        csv.DictReader(
                            (root / "kv-transfers.csv").read_text().splitlines()
                        )
                    )
                    self.assertEqual(
                        sum(int(r["head_end"]) - int(r["head_start"]) for r in rows), 8
                    )
                    handoff = next(
                        csv.DictReader(
                            (root / "kv-handoffs.csv").read_text().splitlines()
                        )
                    )
                    self.assertAlmostEqual(
                        float(handoff["kv_wait_s"]), expected, places=11
                    )
                    self.assertEqual(
                        system["decode_instances"][0]["scheduler"][
                            "max_num_batched_tokens"
                        ],
                        2,
                    )

    def test_one_token_skips_all_transfers(self):
        root, instances, result = self.run_case(4, 2, outputs=1)
        self.assertEqual(result.returncode, 0, result.stderr)
        outer = load_trace(root / "outer.jsonl")
        self.assertFalse(any(n["details"]["kind"] == "network" for n in outer["nodes"]))
        self.assertFalse(
            iteration_nodes(
                outer, json.loads((instances[1][1] / "init.json").read_text())
            )
        )

    def test_mixed_hardware_explicit_bridge_and_bandwidth_sensitivity(self):
        latencies = []
        for bw in (10, 1):
            spec = {
                "clusters": [
                    {"id": "p", "gpu_model": "rtx6000d"},
                    {"id": "d", "gpu_model": "h200"},
                ],
                "links": [
                    {
                        "src": "p/n0-eth-supernode-switch",
                        "dst": "d/n0-nvswitch",
                        "bandwidth_gbs": bw,
                        "latency_us": 3,
                    }
                ],
            }
            root, instances, result = self.run_case(2, 2, topology=spec, d_model="h200")
            self.assertEqual(result.returncode, 0, result.stderr)
            outer = load_trace(root / "outer.jsonl")
            networks = [n for n in outer["nodes"] if n["details"]["kind"] == "network"]
            for n in networks:
                # Both rank transfers share the bridge; path latency is 1+3+0.5 us.
                self.assertAlmostEqual(
                    n["finish"] - n["dispatches"][0][1],
                    655360 / (bw * 1e9) + 4.5e-6,
                    places=11,
                )
            latencies.append(
                json.loads((root / "result.jsonl").read_text())["sub_requests"][0][
                    "token_latency"
                ]
            )
            config = json.loads((instances[1][1] / "init.json").read_text())
            _, _, memories, _ = build_hardware(config["hardware"], 2)
            self.assertEqual({m.capacity_gb for m in memories}, {144})
        self.assertAlmostEqual(latencies[0][0], latencies[1][0], places=11)
        self.assertAlmostEqual(
            latencies[1][1] - latencies[0][1], 655360 * (1 / 1e9 - 1 / 10e9), places=11
        )
        self.assertAlmostEqual(latencies[0][2], latencies[1][2], places=11)

    def test_two_instances_contend_on_one_physical_bridge(self):
        spec = {
            "clusters": [
                {"id": "p", "gpu_model": "rtx6000d"},
                {"id": "d", "gpu_model": "h200"},
            ],
            "links": [
                {
                    "src": "p/n0-eth-supernode-switch",
                    "dst": "d/n0-nvswitch",
                    "bandwidth_gbs": 1,
                    "latency_us": 3,
                }
            ],
        }
        root, _, result = self.run_case(
            2, 2, count=2, topology=spec, d_model="h200", p_instances=2, d_instances=2
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        outer = load_trace(root / "outer.jsonl")
        flows = [n for n in outer["nodes"] if n["details"]["kind"] == "network"]
        self.assertEqual(len(flows), 4)
        self.assertEqual(len({n["dispatches"][0][1] for n in flows}), 1)
        for n in flows:
            self.assertAlmostEqual(
                n["finish"] - n["dispatches"][0][1],
                2 * 655360 / 1e9 + 4.5e-6,
                places=11,
            )

    def test_invalid_config_fails_before_workers(self):
        mutations = [
            lambda c: c["system"]["decode_instances"][0]["model_runner"].update(tp=4),
            lambda c: c["system"]["kv_topology"]["decode_gpus"][0].__setitem__(
                0, c["system"]["kv_topology"]["prefill_gpus"][0][0]
            ),
            lambda c: c["system"]["kv_topology"]["routes"][0].update(links=[]),
        ]
        for mutate in mutations:
            root, _, result = self.run_case(4, 2, mutate=mutate)
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse((root / "instances").exists())

    def test_cluster_names_do_not_change_cross_node_bandwidth(self):
        hw = build_topology(
            {
                "clusters": [{"id": "rack-a", "gpu_model": "h200", "n_nodes": 2}],
                "links": [],
            }
        )
        devices = {n.name: n for n in hw.nodes}
        self.assertEqual(
            hw.find_aggregate_bandwidth(
                [
                    devices["rack-a/n0-nvidia-h200-sxm-0"],
                    devices["rack-a/n1-nvidia-h200-sxm-0"],
                ]
            ),
            50,
        )

    def test_topology_and_capacity_validation(self):
        import tomllib

        leaf = tomllib.loads(
            (
                ROOT / "examples/scenarios/rooflang-llama3_1_8b/simthesizer.toml"
            ).read_text()
        )["system"]
        leaf["model_runner"]["command"] += ["--audit-dir", "unused"]
        p = {"gpu_model": "rtx6000d", "instances": 1, "tp": 4}
        d = {"gpu_model": "rtx6000d", "instances": 1, "tp": 2}
        for changed in [dict(d, gpu_count=4), dict(d, pp=2), dict(d, gpu_model="h200")]:
            with self.assertRaises(ValueError):
                topology_pd_config(leaf, Path("unused"), p, changed)
        disconnected = {
            "clusters": [
                {"id": "p", "gpu_model": "rtx6000d"},
                {"id": "d", "gpu_model": "h200"},
            ],
            "links": [],
        }
        with self.assertRaisesRegex(ValueError, "No path"):
            topology_pd_config(
                leaf, Path("unused"), p, dict(d, gpu_model="h200"), disconnected
            )
        for model in ("rtx6000d", "h200", "b300"):
            hw = build_topology(default_topology(model, 16))
            self.assertEqual(sum(n.kind == "gpu" for n in hw.nodes), 16)


if __name__ == "__main__":
    unittest.main()
