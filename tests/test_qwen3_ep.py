"""Independent invariants for the analytical Qwen shared EP workload model."""

import copy
import json
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from rooflang_bridge.qwen3 import (
    QwenTemplate,
    describe,
    route,
    token_experts,
    validate_model,
)

HW = {
    "tflops_fp16": 100,
    "memory_bandwidth_gbs": 1000,
    "memory_capacity_gb": 1000,
    "link_bandwidth_gbs": 100,
    "link_latency_us": 1,
}


def batch(*shapes):
    return [{"q_len": q, "kv_len": kv, "lm_head_len": 1} for q, kv in shapes]


def named(plan, name):
    return next(
        op for op in plan["operations"] if op["work"]["details"].get("name") == name
    )


def init(ep=8, tp=1):
    model = describe({"tp": tp, "pp": 1, "ep": ep})
    return {
        "op": "init",
        "version": 1,
        "tp": tp,
        "ep": ep,
        "expert_tp": 1,
        "attention_groups": [
            list(range(start, start + tp)) for start in range(0, ep, tp)
        ],
        "model_kind": "qwen3_235b_a22b",
        "routing_seed": 0,
        "hardware": HW,
        "model": {key: model[key] for key in ("pre", "block", "post", "layers")},
        "memory": {
            key: model[key]
            for key in ("weight_bytes_per_device", "kv_bytes_per_token_per_device")
        },
    }


class QwenGraphTests(unittest.TestCase):
    def test_native_metadata_distinguishes_qwen_projection_and_residency(self):
        request = init()
        self.assertEqual(request["model"]["layers"], 94)
        self.assertEqual(
            request["model"]["block"][1]["Gemm"],
            {"name": "qkv_projection", "m": 4096, "n": 9216},
        )
        self.assertEqual(request["model"]["block"][5]["Gemm"]["m"], 8192)
        self.assertEqual(
            request["memory"],
            {
                "weight_bytes_per_device": 72_768_576_512,
                "kv_bytes_per_token_per_device": 192_512,
            },
        )
        validate_model(request["model"], request["memory"], 1, 8)
        changed = copy.deepcopy(request["model"])
        changed["block"][1]["Gemm"]["n"] = 5120
        with self.assertRaisesRegex(ValueError, "exact.*layer plan"):
            validate_model(changed, request["memory"], 1, 8)
        with self.assertRaisesRegex(ValueError, "memory metadata"):
            validate_model(
                request["model"],
                request["memory"] | {"kv_bytes_per_token_per_device": 24064},
                1,
                8,
            )

    def test_ragged_attention_has_one_owner_and_all_heads_in_every_layer(self):
        plan = QwenTemplate(1, 8, HW).export(batch((5, 0), (1, 16)), [7, 8])
        attention = [op for op in plan["operations"] if op["layer"] is not None]
        self.assertEqual(len(attention), 188)
        for layer in range(94):
            rows = [op for op in attention if op["layer"] == layer]
            self.assertEqual(
                sorted(i for op in rows for i in op["batch_indices"]), [0, 1]
            )
            self.assertTrue(all(op["kv_rank"] == 0 for op in rows))
        short = named(plan, "layer0.rank7.attention")["work"]["details"]
        self.assertEqual(short["flops"], 496_320)
        self.assertEqual(short["outputs"], {"y": (5, 8192), "kv_append": (5, 1024)})
        context = named(plan, "layer0.rank0.attention")["work"]["details"]
        self.assertEqual(context["flops"], 562_496)
        self.assertEqual(context["weight_bytes"], 32_768)
        qkv = named(plan, "layer0.rank7.qkv")["work"]["details"]
        self.assertEqual(qkv["weights"], {"w": (4096, 9216)})
        self.assertEqual(
            named(plan, "layer0.rank7.topk")["work"]["details"]["flops"], 5195
        )

    def test_every_supported_ep_keeps_attention_owned_and_experts_partitioned(self):
        for ep in (1, 2, 4, 8):
            with self.subTest(ep=ep):
                plan = QwenTemplate(1, ep, HW).export(batch((1, 0)), [7])
                attention = [op for op in plan["operations"] if op["layer"] is not None]
                self.assertEqual(len(attention), 94)
                self.assertTrue(
                    all(
                        op["work"]["details"]["devices"] == [f"gpu{7 % ep}"]
                        for op in attention
                    )
                )
                self.assertTrue(all(op["batch_indices"] == [0] for op in attention))
                for operation in plan["operations"]:
                    detail = operation["work"]["details"]
                    if "expert" in detail:
                        self.assertEqual(
                            detail["devices"], [f"gpu{detail['expert'] // (128 // ep)}"]
                        )
                self.assertEqual(
                    describe({"tp": 1, "ep": ep, "pp": 1})[
                        "kv_bytes_per_token_per_device"
                    ],
                    192_512,
                )
                if ep == 1:
                    self.assertTrue(
                        all(
                            row["dispatch_network_bytes"]
                            == row["combine_network_bytes"]
                            == 0
                            for row in plan["summary"]["routing_layers"]
                        )
                    )

    def test_routing_is_batch_order_and_chunk_invariant_with_distinct_top8(self):
        full = route(batch((4, 10), (2, 0)), [3, 9], 8, 4, 19)
        reordered = route(batch((2, 0), (4, 10)), [9, 3], 8, 4, 19)
        self.assertEqual(full, reordered)
        first = route(batch((2, 10)), [3], 8, 4, 19)
        second = route(batch((2, 12)), [3], 8, 4, 19)
        whole = route(batch((4, 10)), [3], 8, 4, 19)
        self.assertEqual(
            [a + b for a, b in zip(first["expert_tokens"], second["expert_tokens"])],
            whole["expert_tokens"],
        )
        self.assertEqual(sum(full["expert_tokens"]), 48)
        for token in range(128):
            experts = token_experts(3, token, 4, 19)
            self.assertEqual(len(set(experts)), 8)
            self.assertTrue(all(0 <= e < 128 for e in experts))
        self.assertNotEqual(token_experts(3, 10, 4, 19), token_experts(3, 10, 4, 20))

    def test_dispatch_deduplicates_rank_and_combines_after_local_reduction(self):
        template = QwenTemplate(1, 8, HW)
        skew = template.export(batch((1, 0)), [7], routing=[list(range(8))])
        spread = template.export(
            batch((1, 0)), [7], routing=[[0, 1, 2, 3, 16, 17, 18, 19]]
        )
        skew_stats = skew["summary"]["routing_layers"][0]
        self.assertEqual(skew_stats["dispatch_network_bytes"], 8448)
        self.assertEqual(skew_stats["combine_network_bytes"], 8192)
        spread_stats = spread["summary"]["routing_layers"][0]
        self.assertEqual(spread_stats["dispatch_network_bytes"], 16_896)
        self.assertEqual(spread_stats["combine_network_bytes"], 16_384)
        reduce = named(skew, "layer0.expert_reduce.0-7")
        self.assertEqual(reduce["work"]["details"]["flops"], 61_440)
        for expert in range(8):
            self.assertIn(
                named(skew, f"layer0.expert{expert}.down")["id"], reduce["parents"]
            )
        combine = named(skew, "layer0.combine.0-7")
        self.assertEqual(combine["parents"], [reduce["id"]])
        self.assertEqual(
            named(spread, "layer0.rank7.combine_accumulate")["work"]["details"][
                "flops"
            ],
            4096,
        )

    def test_active_experts_have_separate_gemms_and_weights_read_once(self):
        plan = QwenTemplate(1, 8, HW).export(
            batch((2, 0)), [7], routing=[list(range(8))] * 2
        )
        details = [op["work"]["details"] for op in plan["operations"]]
        first = [
            row
            for row in details
            if row["name"].startswith("layer0.expert") and "expert_tokens" in row
        ]
        self.assertEqual(len(first), 24)
        self.assertEqual(sum(row["weight_bytes"] for row in first), 301_989_888)
        self.assertTrue(all(row["expert_tokens"] == 2 for row in first))
        self.assertEqual(
            named(plan, "layer0.expert0.gate_up")["work"]["details"]["flops"],
            50_331_648,
        )
        self.assertEqual(
            named(plan, "layer0.expert0.down")["work"]["details"]["flops"], 25_165_824
        )
        self.assertEqual(
            named(plan, "layer0.expert0.silu_mul")["work"]["details"]["flops"], 18_432
        )

    def test_local_dispatch_uses_hbm_but_no_fabric_and_remote_directions_match(self):
        template = QwenTemplate(1, 8, HW)
        local = template.export(batch((1, 0)), [0], routing=[list(range(8))])
        remote = template.export(batch((1, 0)), [7], routing=[list(range(8))])
        work = named(local, "layer0.dispatch.0-0")["work"]
        self.assertEqual(set(work["demands"][0]["resources"]), {"hbm:gpu0"})
        self.assertAlmostEqual(
            work["demands"][0]["resources"]["hbm:gpu0"], 16_896 / 1e12
        )
        dispatch = named(remote, "layer0.dispatch.7-0")["work"]["demands"][0]
        self.assertEqual(
            set(dispatch["resources"]),
            {"hbm:gpu7", "hbm:gpu0", "fabric:gpu0|gpu7|link0-7|rev"},
        )
        self.assertAlmostEqual(dispatch["latency"], 1e-6)
        combine = named(remote, "layer0.combine.0-7")["work"]["demands"][0]
        self.assertIn("fabric:gpu0|gpu7|link0-7|fwd", combine["resources"])

    def test_link_and_memory_sensitivity_affect_their_own_service(self):
        def generate(hardware):
            return QwenTemplate(1, 8, hardware).export(
                batch((1, 0)), [7], routing=[list(range(8))]
            )

        fast = generate(HW)
        slow_link = generate(HW | {"link_bandwidth_gbs": 10})
        slow_hbm = generate(HW | {"memory_bandwidth_gbs": 100})
        names = ("layer0.dispatch.7-0", "layer0.expert0.gate_up")
        for name in names:
            a = named(fast, name)["work"]["details"]
            link = named(slow_link, name)["work"]["details"]
            hbm = named(slow_hbm, name)["work"]["details"]
            self.assertEqual(a["compute_us"], link["compute_us"])
            self.assertEqual(a["compute_us"], hbm["compute_us"])
            self.assertEqual(a["memory_us"], link["memory_us"])
            self.assertAlmostEqual(hbm["memory_us"], a["memory_us"] * 10)
        resource = "fabric:gpu0|gpu7|link0-7|rev"
        a = named(fast, names[0])["work"]["demands"][0]["resources"][resource]
        b = named(slow_link, names[0])["work"]["demands"][0]["resources"][resource]
        self.assertAlmostEqual(b, 10 * a)

    def test_workspace_tracks_skew_and_capacity_includes_past_kv(self):
        template = QwenTemplate(1, 8, HW)
        skew = template.export(batch((2, 0)), [7], routing=[list(range(8))] * 2)
        spread = template.export(
            batch((2, 0)), [7], routing=[list(range(0, 128, 16))] * 2
        )
        self.assertGreater(skew["workspace"]["gpu0"], spread["workspace"]["gpu0"])
        self.assertEqual(set(skew["workspace"]), {f"gpu{i}" for i in range(8)})
        with self.assertRaisesRegex(ValueError, "resident weights"):
            QwenTemplate(1, 1, HW | {"memory_capacity_gb": 96})
        tight = QwenTemplate(1, 8, HW | {"memory_capacity_gb": 72.769})
        with self.assertRaisesRegex(ValueError, "resident KV and workspace"):
            tight.export(batch((1, 100)), [7])

    def test_graph_drains_all_work_and_rebinding_never_reuses_routes(self):
        template = QwenTemplate(1, 4, HW)
        first = template.export(batch((1, 0)), [3])
        changed = template.export(batch((2, 9)), [4])
        repeated = template.export(batch((1, 0)), [3])
        self.assertEqual(first, repeated)
        self.assertEqual(changed["summary"]["tokens_per_rank"], [2, 0, 0, 0])
        for plan in (first, changed):
            operations = plan["operations"]
            visited = set()
            pending = [len(operations) - 1]
            while pending:
                current = pending.pop()
                if current in visited:
                    continue
                visited.add(current)
                parents = operations[current]["parents"]
                self.assertTrue(all(parent < current for parent in parents))
                pending.extend(parents)
            self.assertEqual(len(visited), len(operations))
            self.assertTrue(
                all(
                    {"devices", "compute_us", "memory_us", "network_us"}
                    <= op["work"]["details"].keys()
                    for op in operations
                )
            )
        compact = template.export(batch((1, 0)), [3], details=False)
        for old, small in zip(first["operations"], compact["operations"]):
            self.assertEqual(old | {"work": old["work"] | {"details": {}}}, small)
        self.assertEqual(first["workspace"], compact["workspace"])

    def test_unsupported_parallel_batch_and_routing_are_rejected(self):
        for tp, ep in ((8, 8), (2, 1), (4, 2), (1, 3), (True, 1), (1, False)):
            with self.assertRaises(ValueError):
                QwenTemplate(tp, ep, HW)
        template = QwenTemplate(1, 2, HW)
        for rows, requests, routing in (
            (batch((1, 0)), [], None),
            (batch((1, 0), (1, 0)), [3, 3], None),
            (batch((1, 0)), [True], None),
            (batch((1, 0)), [0], [[0] * 8]),
            (batch((1, 0)), [0], [list(range(7)) + [128]]),
            (batch((0, 0)), [0], None),
        ):
            with self.assertRaises(ValueError):
                template.export(rows, requests, routing=routing)

    def test_worker_protocol_uses_qwen_and_refuses_unsupported_reference(self):
        graph = {
            "op": "graph",
            "batch": batch((1, 0)),
            "requests": [7],
            "details": False,
        }
        command = [sys.executable, str(ROOT / "scripts/rooflang_worker.py")]
        process = subprocess.run(
            command,
            input="\n".join(
                map(
                    json.dumps,
                    [init(), graph, {"op": "iteration", "batch": batch((1, 0))}],
                )
            ),
            text=True,
            capture_output=True,
            check=True,
            timeout=30,
        )
        ready, result, unsupported = [
            json.loads(line) for line in process.stdout.splitlines()
        ]
        self.assertTrue(ready["ready"])
        self.assertEqual(ready["hardware"]["gpu_count"], 8)
        self.assertEqual(
            sum(op["layer"] is not None for op in result["operations"]), 94
        )
        self.assertIsNone(result["reference"])
        self.assertIn("requires shared execution", unsupported["error"])
        process = subprocess.run(
            command + ["--reference-every-round"],
            input=json.dumps(init()) + "\n",
            text=True,
            capture_output=True,
            check=True,
            timeout=30,
        )
        self.assertIn("unsupported", json.loads(process.stdout)["error"])


class QwenAttentionTpTests(unittest.TestCase):
    def test_metadata_shards_attention_and_lm_head_and_retains_full_experts(self):
        for tp in (1, 2, 4):
            with self.subTest(tp=tp):
                request = init(tp=tp)
                block = request["model"]["block"]
                self.assertEqual(block[1]["Gemm"]["n"], 9216 // tp)
                self.assertEqual(block[4]["Attention"]["num_q_heads"], 64 // tp)
                self.assertEqual(block[4]["Attention"]["num_kv_heads"], 4 // tp)
                self.assertEqual(block[5]["Gemm"]["m"], 8192 // tp)
                self.assertEqual(sum("AllReduce" in row for row in block), tp > 1)
                self.assertEqual(request["model"]["post"][1]["Gemm"]["n"], 151936 // tp)
                # Independent resident inventory: embeddings replicated; LM head and
                # attention matrices sharded; router, norms and 16 experts complete.
                per_layer = (
                    (4096 * 9216 + 8192 * 4096) // tp
                    + 2 * 128
                    + 2 * 4096
                    + 4096 * 128
                    + 16 * 3 * 4096 * 1536
                )
                expected_weights = 2 * (
                    151936 * 4096 + 151936 * 4096 // tp + 4096 + 94 * per_layer
                )
                self.assertEqual(
                    request["memory"]["weight_bytes_per_device"], expected_weights
                )
                self.assertEqual(
                    request["memory"]["kv_bytes_per_token_per_device"], 192512 // tp
                )
                validate_model(request["model"], request["memory"], tp, 8)

    def test_all_valid_layouts_cover_each_request_head_shard_once(self):
        for tp, ep in ((2, 2), (2, 4), (2, 8), (4, 4), (4, 8)):
            with self.subTest(tp=tp, ep=ep):
                plan = QwenTemplate(tp, ep, HW).export(batch((5, 0), (1, 16)), [7, 8])
                attention = [op for op in plan["operations"] if op["layer"] is not None]
                for layer in range(94):
                    rows = [op for op in attention if op["layer"] == layer]
                    self.assertEqual(
                        sorted(
                            (index, op["kv_rank"])
                            for op in rows
                            for index in op["batch_indices"]
                        ),
                        [(index, rank) for index in range(2) for rank in range(tp)],
                    )
                    self.assertEqual(
                        sum(op["work"]["details"]["flops"] for op in rows), 1_058_816
                    )
                    self.assertEqual(
                        sum(op["work"]["details"]["weight_bytes"] for op in rows), 32768
                    )
                    for op in rows:
                        for index in op["batch_indices"]:
                            owner = ([7, 8][index] % (ep // tp)) * tp + op["kv_rank"]
                            self.assertEqual(
                                op["work"]["details"]["devices"], [f"gpu{owner}"]
                            )

    def test_projection_expert_and_lm_work_are_conserved_across_tp(self):
        rows, identities, routing = batch((2, 0), (1, 4)), [7, 8], [list(range(8))] * 3
        totals = []
        for tp in (1, 2, 4):
            plan = QwenTemplate(tp, 8, HW).export(rows, identities, routing=routing)
            details = [op["work"]["details"] for op in plan["operations"]]
            selected = [
                row
                for row in details
                if row["name"].endswith(
                    (
                        ".qkv",
                        ".qk_norm",
                        ".rope",
                        ".attention",
                        ".o_projection",
                        ".lm_head",
                    )
                )
                or "expert_tokens" in row
            ]
            totals.append(
                (
                    sum(row["flops"] for row in selected),
                    sum(
                        row["weight_bytes"]
                        for row in selected
                        if "expert_tokens" in row
                    ),
                )
            )
            self.assertEqual(
                sum(row["flops"] for row in details if row["name"].endswith(".router")),
                94 * 3 * 2 * 4096 * 128,
            )
            self.assertTrue(
                all(
                    sum(row["expert_tokens"]) == 24
                    for row in plan["summary"]["routing_layers"]
                )
            )
            expert = named(plan, "layer0.expert0.gate_up")["work"]["details"]
            self.assertEqual(expert["weights"], {"w": (4096, 3072)})
        self.assertEqual(totals, [totals[0]] * 3)

    def test_tp_reduction_charges_arithmetic_memory_and_both_network_directions(self):
        plan = QwenTemplate(2, 8, HW).export(
            batch((1, 0)), [3], routing=[list(range(96, 104))]
        )
        gather = named(plan, "layer0.tp6.gather.7-6")
        reduce = named(plan, "layer0.tp6.reduce")
        broadcast = named(plan, "layer0.tp6.broadcast.6-7")
        self.assertEqual(
            gather["parents"], [named(plan, "layer0.rank7.o_projection")["id"]]
        )
        self.assertIn(gather["id"], reduce["parents"])
        self.assertIn(named(plan, "layer0.rank6.o_projection")["id"], reduce["parents"])
        self.assertEqual(reduce["work"]["details"]["flops"], 4096)
        self.assertEqual(reduce["work"]["details"]["input_bytes"], 16384)
        self.assertEqual(reduce["work"]["details"]["output_bytes"], 8192)
        self.assertAlmostEqual(reduce["work"]["details"]["memory_us"], 24576 / 1e6)
        for operation, direction in ((gather, "rev"), (broadcast, "fwd")):
            demand = operation["work"]["demands"][0]
            self.assertEqual(
                set(demand["resources"]),
                {"hbm:gpu6", "hbm:gpu7", f"fabric:gpu6|gpu7|link6-7|{direction}"},
            )
            self.assertEqual(operation["work"]["details"]["communication_bytes"], 8192)
        self.assertEqual(broadcast["parents"], [reduce["id"]])
        self.assertIn(
            broadcast["id"], named(plan, "layer0.rank7.attention_residual")["parents"]
        )
        self.assertEqual(
            named(plan, "layer0.moe_broadcast.6-7")["parents"],
            [named(plan, "layer0.rank6.moe_residual")["id"]],
        )

    def test_leader_routes_once_to_local_peer_and_remote_experts(self):
        template = QwenTemplate(2, 8, HW)
        for owner, experts in (
            (6, list(range(96, 104))),
            (7, list(range(112, 120))),
            (0, list(range(8))),
        ):
            with self.subTest(owner=owner):
                plan = template.export(batch((1, 0)), [3], routing=[experts])
                stats = plan["summary"]["routing_layers"][0]
                self.assertEqual(
                    stats["dispatch_network_bytes"], 8448 if owner != 6 else 0
                )
                self.assertEqual(
                    stats["combine_network_bytes"], 8192 if owner != 6 else 0
                )
                self.assertEqual(sum(map(sum, stats["rank_tokens"])), 1)
                self.assertEqual(stats["rank_tokens"][6][owner], 1)
                first = [
                    op["work"]["details"]
                    for op in plan["operations"]
                    if op["work"]["details"]["name"].startswith("layer0.")
                ]
                self.assertEqual(
                    sum(row["name"].endswith(".router") for row in first), 1
                )
                self.assertEqual(sum("expert_tokens" in row for row in first), 24)
                self.assertTrue(
                    all(
                        row["expert_tokens"] == 1
                        for row in first
                        if "expert_tokens" in row
                    )
                )
                self.assertEqual(
                    named(plan, f"layer0.dispatch.6-{owner}")["work"]["details"][
                        "token_vectors"
                    ],
                    1,
                )

    def test_embedding_broadcast_and_full_logits_gather_precede_sampling(self):
        plan = QwenTemplate(2, 8, HW).export(
            batch((1, 0)), [3], routing=[list(range(8))]
        )
        embedding = named(plan, "embedding")
        self.assertEqual(embedding["work"]["details"]["devices"], ["gpu6"])
        self.assertEqual(embedding["work"]["details"]["weight_bytes"], 8192)
        self.assertEqual(
            named(plan, "embedding_broadcast.6-7")["parents"], [embedding["id"]]
        )
        for rank in (6, 7):
            self.assertEqual(
                named(plan, f"rank{rank}.lm_head")["work"]["details"]["weights"],
                {"w": (4096, 75968)},
            )
        gathered = named(plan, "logits_gather.7-6")
        self.assertEqual(gathered["work"]["details"]["communication_bytes"], 151936)
        self.assertEqual(gathered["parents"], [named(plan, "rank7.lm_head")["id"]])
        sampler = named(plan, "rank6.sampler")
        self.assertEqual(sampler["work"]["details"]["inputs"], {"x": (1, 151936)})
        self.assertEqual(sampler["work"]["details"]["flops"], 759680)
        self.assertEqual(
            sampler["parents"],
            sorted([gathered["id"], named(plan, "rank6.lm_head")["id"]]),
        )
        self.assertEqual(
            sum(
                op["work"]["details"]["name"].endswith(".sampler")
                for op in plan["operations"]
            ),
            1,
        )
        self.assertGreaterEqual(plan["workspace"]["gpu6"], 2 * (3 * 4096 + 151936 + 1))
        self.assertGreaterEqual(plan["workspace"]["gpu7"], 2 * (3 * 4096 + 75968 + 1))

    def test_details_do_not_change_dependencies_cost_or_workspace(self):
        template = QwenTemplate(4, 8, HW)
        full = template.export(batch((2, 3), (1, 8)), [1, 2])
        compact = template.export(batch((2, 3), (1, 8)), [1, 2], details=False)
        self.assertEqual(full["workspace"], compact["workspace"])
        self.assertEqual(len(full["operations"]), len(compact["operations"]))
        for large, small in zip(full["operations"], compact["operations"]):
            self.assertEqual(large | {"work": large["work"] | {"details": {}}}, small)
        visited, pending = set(), [len(full["operations"]) - 1]
        while pending:
            current = pending.pop()
            if current in visited:
                continue
            visited.add(current)
            parents = full["operations"][current]["parents"]
            self.assertTrue(all(parent < current for parent in parents))
            pending.extend(parents)
        self.assertEqual(len(visited), len(full["operations"]))

    def test_tp_routing_preserves_token_identity_across_reordering_and_chunks(self):
        for tp in (2, 4):
            whole = route(batch((4, 10), (2, 0)), [3, 9], 8, 4, 19, tp=tp)
            reordered = route(batch((2, 0), (4, 10)), [9, 3], 8, 4, 19, tp=tp)
            self.assertEqual(whole, reordered)
            self.assertTrue(
                all(
                    not any(row)
                    for rank, row in enumerate(whole["rank_tokens"])
                    if rank % tp
                )
            )
            first = route(batch((2, 10)), [3], 8, 4, 19, tp=tp)
            second = route(batch((2, 12)), [3], 8, 4, 19, tp=tp)
            unchunked = route(batch((4, 10)), [3], 8, 4, 19, tp=tp)
            self.assertEqual(
                [
                    [a + b for a, b in zip(left, right)]
                    for left, right in zip(first["rank_tokens"], second["rank_tokens"])
                ],
                unchunked["rank_tokens"],
            )

    def test_workspace_accounts_for_reduction_receives_and_sharded_past_kv(self):
        count = 64
        plan = QwenTemplate(4, 8, HW).export(
            batch((count, 0)), [1], routing=[list(range(8))] * count
        )
        self.assertEqual(
            plan["workspace"]["gpu4"] - plan["workspace"]["gpu5"],
            2 * count * 4096 * 3,
        )
        resident = describe({"tp": 2, "ep": 8, "pp": 1})["weight_bytes_per_device"]
        template = QwenTemplate(
            2, 8, HW | {"memory_capacity_gb": (resident + 1_000_000) / 1e9}
        )
        template.export(batch((1, 0)), [3], routing=[list(range(8))])
        with self.assertRaisesRegex(ValueError, "resident KV and workspace"):
            template.export(batch((1, 100)), [3], routing=[list(range(8))])

    def test_invalid_layout_and_expert_tp_are_rejected(self):
        for groups in (
            [],
            [[0, 1], [2, 3]],
            [[1, 0], [2, 3], [4, 5], [6, 7]],
            [[False, 1], [2, 3], [4, 5], [6, 7]],
        ):
            with self.assertRaisesRegex(ValueError, "attention_groups"):
                QwenTemplate(2, 8, HW, attention_groups=groups)
        for expert_tp in (2, True, 0):
            with self.assertRaisesRegex(ValueError, "expert TP"):
                QwenTemplate(2, 8, HW, expert_tp=expert_tp)
            with self.assertRaisesRegex(ValueError, "expert TP"):
                describe({"tp": 2, "ep": 8, "pp": 1, "expert_tp": expert_tp})

    def test_worker_accepts_layout_and_rejects_missing_or_inconsistent_groups(self):
        request = init(tp=2)
        graph = {
            "op": "graph",
            "batch": batch((1, 0)),
            "requests": [3],
            "details": False,
        }
        missing = {
            key: value for key, value in request.items() if key != "attention_groups"
        }
        invalid = request | {
            "attention_groups": [[0], [1], [2], [3], [4], [5], [6], [7]]
        }
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts/rooflang_worker.py")],
            input="\n".join(
                map(
                    json.dumps,
                    [request, graph, missing, invalid, request | {"expert_tp": 2}],
                )
            ),
            text=True,
            capture_output=True,
            check=True,
            timeout=30,
        )
        ready, plan, *failures = [
            json.loads(line) for line in result.stdout.splitlines()
        ]
        self.assertTrue(ready["ready"])
        self.assertEqual(sum(op["layer"] is not None for op in plan["operations"]), 188)
        self.assertTrue(all("error" in failure for failure in failures))


if __name__ == "__main__":
    unittest.main()
