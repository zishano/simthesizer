"""Research objectives retain throughput and latency tradeoffs."""

import copy
import unittest

import objectives


class ObjectivesTests(unittest.TestCase):
    def setUp(self):
        self.metrics = {
            "ttft": {"p99": 0.2},
            "tpot": {"p99": 0.05, "mean": 0.03},
            "slo": {"goodput_per_gpu": 12, "tpot_applicable": True},
            "tokens_per_gpu": {"mean": 20, "gpu_count": 32},
            "tokens_per_user": {"mean": 60, "samples": 2},
            "makespan": {"mean": 50},
        }

    def row(self, name, **point):
        return {
            "event": "finished",
            "status": "VALID_FEASIBLE",
            "group_id": "real-m1.0-s0",
            "candidate_id": name,
            "tpot_applicable": True,
            "objectives": {**objectives.from_metrics(self.metrics), **point},
        }

    def names(self, rows):
        return [row["candidate_id"] for row in objectives.pareto(rows)["real-m1.0-s0"]]

    def test_extracts_five_host_metrics_without_reconstructing_user_rate(self):
        before = copy.deepcopy(self.metrics)
        self.assertEqual(
            objectives.from_metrics(self.metrics),
            {
                "ttft_p99": 0.2,
                "tpot_p99": 0.05,
                "goodput_per_gpu": 12,
                "output_tokens_per_gpu": 20,
                "tokens_per_user": 60,
            },
        )
        self.assertEqual(self.metrics, before)
        self.metrics["makespan"]["mean"] = 500
        self.assertEqual(
            objectives.from_metrics(self.metrics), objectives.from_metrics(before)
        )

    def test_missing_or_single_output_metrics_remain_undefined(self):
        self.assertEqual(
            objectives.from_metrics({}), dict.fromkeys(objectives.METRIC_DIRECTIONS)
        )
        self.metrics["slo"]["tpot_applicable"] = False
        point = objectives.from_metrics(self.metrics)
        self.assertIsNone(point["tpot_p99"])
        self.assertIsNone(point["tokens_per_user"])
        self.assertEqual(point["output_tokens_per_gpu"], 20)
        self.metrics["slo"]["tpot_applicable"] = True
        self.metrics["tokens_per_user"]["samples"] = 0
        self.assertIsNone(objectives.from_metrics(self.metrics)["tokens_per_user"])

    def test_nonfinite_boolean_and_negative_values_cannot_be_ranked(self):
        for value in (float("nan"), float("inf"), -1, True, "20"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                objectives.from_metrics({"tokens_per_gpu": {"mean": value}})
            with self.subTest(value=value), self.assertRaises(ValueError):
                objectives.pareto([self.row("bad", output_tokens_per_gpu=value)])

    def test_equal_old_objectives_do_not_hide_total_throughput_or_user_rate(self):
        for field in ("output_tokens_per_gpu", "tokens_per_user"):
            with self.subTest(field=field):
                rows = [self.row("a"), self.row("b", **{field: 100})]
                self.assertEqual(self.names(rows), ["b"])

    def test_latency_and_user_speed_costs_preserve_tradeoffs(self):
        base = self.row("base")
        for point in (
            {"output_tokens_per_gpu": 30, "tokens_per_user": 50},
            {"output_tokens_per_gpu": 30, "ttft_p99": 0.3},
            {"goodput_per_gpu": 13, "tpot_p99": 0.06},
        ):
            with self.subTest(point=point):
                self.assertEqual(
                    self.names([base, self.row("tradeoff", **point)]),
                    ["base", "tradeoff"],
                )
        winner = self.row(
            "winner",
            ttft_p99=0.1,
            tpot_p99=0.04,
            goodput_per_gpu=13,
            output_tokens_per_gpu=30,
            tokens_per_user=70,
        )
        self.assertEqual(self.names([base, winner]), ["winner"])

    def test_incomplete_points_cannot_dominate_or_merge(self):
        rows = [
            self.row("complete"),
            self.row("missing-a", tokens_per_user=None, output_tokens_per_gpu=100),
            self.row("missing-b", tokens_per_user=None, output_tokens_per_gpu=100),
        ]
        frontier = objectives.pareto(rows)["real-m1.0-s0"]
        self.assertEqual(
            [row["candidate_id"] for row in frontier],
            ["complete", "missing-a", "missing-b"],
        )
        self.assertTrue(all(row["equivalent_candidates"] == [] for row in frontier))

    def test_single_output_case_uses_only_applicable_coordinates(self):
        rows = [self.row("a"), self.row("b", output_tokens_per_gpu=30)]
        for row in rows:
            row["tpot_applicable"] = False
            row["objectives"]["tpot_p99"] = None
            row["objectives"]["tokens_per_user"] = None
        self.assertEqual(self.names(rows), ["b"])
        rows[0]["tpot_applicable"] = True
        with self.assertRaisesRegex(ValueError, "inconsistent TPOT"):
            objectives.pareto(rows)
        rows[0]["tpot_applicable"] = None
        with self.assertRaisesRegex(ValueError, "boolean"):
            objectives.pareto(rows)

    def test_groups_latest_receipts_equivalence_and_feasibility_match_archive_schema(
        self,
    ):
        other = {**self.row("other"), "group_id": "real-m2.0-s0"}
        rows = [
            self.row("a", tokens_per_user=30),
            self.row("b"),
            self.row("a"),
            other,
            {**self.row("invalid"), "status": "INVALID"},
            {**self.row("infeasible"), "status": "VALID_INFEASIBLE"},
            {**self.row("pending"), "event": "started"},
        ]
        before = copy.deepcopy(rows)
        result = objectives.pareto(rows)
        self.assertEqual(set(result), {"real-m1.0-s0", "real-m2.0-s0"})
        self.assertEqual(len(result["real-m1.0-s0"]), 1)
        representative = result["real-m1.0-s0"][0]
        self.assertEqual(representative["candidate_id"], "a")
        self.assertEqual(representative["equivalent_candidates"], ["b"])
        self.assertEqual(representative["verification"], "EXPLORATORY")
        self.assertEqual(rows, before)
        self.assertEqual(objectives.pareto([]), {})


if __name__ == "__main__":
    unittest.main()
