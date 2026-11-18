import copy
import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from compare import compare
from draw_throughput import draw as draw_throughput
from results import (
    Run,
    load_run,
    load_runs,
    match_runs,
    measurements,
    paired_errors,
    parse_request,
    physical_gpu_count,
    relative_error,
    summarize,
    summarize_run,
    throughput,
    timeline,
)


def llm(latencies, **overrides):
    return {
        "kind": "llm",
        "input_tokens": 4,
        "output_tokens": len(latencies),
        "next": [],
        "token_latency": latencies,
        **overrides,
    }


def request(nodes=None, request_id=0, arrival=0, initial=None):
    return {
        "request_id": request_id,
        "arrival_time": arrival,
        "initial": [0] if initial is None else initial,
        "sub_requests": [llm([1, 2, 3])] if nodes is None else nodes,
    }


def run(*rows, label="test"):
    return Run(
        label,
        Path(f"{label}.jsonl"),
        {row["request_id"]: parse_request(copy.deepcopy(row)) for row in rows},
    )


class MetricsTests(unittest.TestCase):
    def test_call_and_token_weighting_and_single_token_response(self):
        result = measurements(
            run(
                request(),
                request([llm([4, 5])], request_id=1),
                request([llm([7])], request_id=2),
            )
        )
        self.assertEqual(list(result["ttft"].values()), [1, 4, 7])
        self.assertEqual(list(result["e2e"].values()), [6, 9, 7])
        self.assertEqual(list(result["normalized_e2e"].values()), [2, 4.5, 7])
        self.assertEqual(list(result["tpot"].values()), [2.5, 5])
        self.assertEqual(list(result["itl"].values()), [2, 3, 5])
        self.assertEqual(summarize(result["tpot"], [50])["mean"], 3.75)
        self.assertAlmostEqual(summarize(result["itl"], [50])["mean"], 10 / 3)
        self.assertEqual(result["makespan"], {(): 9})
        self.assertEqual(result["output_throughput"], {(): 6 / 9})
        self.assertEqual(result["request_throughput"], {(): 3 / 9})
        self.assertEqual(result["tokens_per_user"], {(0, 0): 0.4, (1, 0): 0.2})
        self.assertAlmostEqual(summarize(result["tokens_per_user"], [50])["mean"], 0.3)
        self.assertEqual(result["tokens_per_gpu"], {})

    def test_gpu_rate_includes_arrival_gaps_and_undefined_user_rates(self):
        result = run(
            request([llm([1, 0, 0])], arrival=10),
            request([llm([1])], request_id=1, arrival=17),
        )
        result.gpu_count = 4
        metrics = summarize_run(result)
        # Four output tokens over eight seconds on four GPUs.
        self.assertEqual(metrics["tokens_per_gpu"]["mean"], 0.125)
        self.assertEqual(metrics["tokens_per_gpu"]["gpu_count"], 4)
        self.assertEqual(metrics["tokens_per_gpu"]["unit"], "token/s/GPU")
        self.assertIsNone(metrics["tokens_per_user"]["mean"])
        self.assertEqual(metrics["tokens_per_user"]["samples"], 0)
        result = run(request([llm([0, 0])]))
        result.gpu_count = 4
        metrics = summarize_run(result)
        self.assertIsNone(metrics["tokens_per_gpu"]["mean"])
        json.dumps(metrics, allow_nan=False)

    def test_dag_intervals_fork_join_and_tool_tail(self):
        tool = {
            "kind": "tool_call",
            "input_tokens": 0,
            "output_tokens": 0,
            "duration": 4,
            "interval": 1,
            "next": [3],
        }
        tail = {**tool, "duration": 2, "interval": 0, "next": []}
        row = request(
            [
                llm([1, 1], interval=2, next=[1, 2]),
                llm([2], interval=3, next=[3]),
                tool,
                llm([1, 2], interval=2, next=[4]),
                tail,
            ],
            arrival=10,
        )
        result = run(row)
        starts, ends = timeline(result.requests[0])
        self.assertEqual(starts, [12, 17, 15, 21, 24])
        self.assertEqual(ends, [14, 19, 19, 24, 26])
        self.assertEqual(list(result.token_times()), [13, 14, 19, 22, 24])
        measured = measurements(result)
        self.assertEqual(measured["request_e2e"], {(0,): 16})
        self.assertEqual(measured["ttft"], {(0, 0): 1, (0, 1): 2, (0, 3): 1})
        self.assertEqual(measured["makespan"], {(): 16})
        self.assertEqual(measured["output_throughput"], {(): 5 / 16})

    def test_multiple_roots_order_is_irrelevant(self):
        result = run(request([llm([1]), llm([2])], initial=[1, 0]))
        self.assertEqual(measurements(result)["request_e2e"], {(0,): 2})

    def test_identity_and_scaled_predictions(self):
        reference = run(request(), label="real")
        candidate = run(request([llm([2, 4, 6])]), label="sim")
        report = compare(
            [reference, candidate],
            ["ttft", "tpot", "e2e", "makespan", "output_throughput"],
            [50, 95],
        )
        for row in report["metrics"]:
            expected = -50 if row["metric"] == "output_throughput" else 100
            self.assertEqual(row["relative_error_pct"]["mean"], expected)
            if row["paired"] is not None:
                self.assertEqual(row["paired"]["mape_pct"], abs(expected))
        identity = compare([reference, reference], ["ttft", "tpot", "e2e"], [50, 95])
        for row in identity["metrics"]:
            self.assertEqual(row["paired"]["mae"], 0)
            self.assertEqual(row["paired"]["mape_pct"], 0)
            self.assertEqual(set(row["relative_error_pct"].values()), {0})

    def test_pairing_exposes_swapped_latencies_with_identical_distributions(self):
        reference = {(1,): 1, (2,): 3}
        candidate = {(2,): 1, (1,): 3}
        self.assertEqual(summarize(reference, [50, 95]), summarize(candidate, [50, 95]))
        errors = paired_errors(reference, candidate)
        self.assertEqual(errors["mae"], 2)
        self.assertEqual(errors["bias"], 0)
        self.assertEqual(errors["wape_pct"], 100)
        self.assertAlmostEqual(errors["mape_pct"], 400 / 3)

    def test_zero_denominators_are_explicit(self):
        errors = paired_errors({0: 0, 1: 2}, {0: 5, 1: 1})
        self.assertEqual(
            errors,
            {
                "count": 2,
                "mae": 3,
                "bias": 2,
                "mape_pct": 50,
                "mape_count": 1,
                "wape_pct": 300,
            },
        )
        self.assertIsNone(relative_error(0, 0))
        self.assertIsNone(relative_error(0, 3))
        self.assertIsNone(paired_errors({0: 0}, {0: 3})["wape_pct"])
        self.assertEqual(paired_errors({}, {})["count"], 0)
        self.assertEqual(summarize({}, [95]), {"mean": None, "p95": None})
        result = measurements(run(request([llm([0])], arrival=10)))
        self.assertEqual(result["tpot"], {})
        self.assertEqual(result["itl"], {})
        self.assertEqual(result["output_throughput"], {})
        report = compare(
            [
                run(request([llm([0])]), label="real"),
                run(request([llm([1])]), label="sim"),
            ],
            ["ttft", "output_throughput"],
            [50],
        )
        self.assertIsNone(report["metrics"][1]["relative_error_pct"]["mean"])
        self.assertEqual(report["metrics"][1]["prediction"]["mean"], 1)

    def test_percentile_uses_linear_interpolation(self):
        self.assertEqual(
            summarize({0: 1, 1: 3}, [0, 50, 95, 100]),
            {"mean": 2, "p0": 1, "p50": 2, "p95": 2.9, "p100": 3},
        )

    def test_throughput_preserves_tokens_boundaries_and_idle_time(self):
        result = run(
            request([llm([0, 0.5, 0.5])], arrival=10),
            request([llm([0.5])], request_id=1, arrival=12),
        )
        edges, rates = throughput(result, 1, 4)
        self.assertEqual(edges, [0, 1, 2, 3, 4, 5])
        self.assertEqual(rates, [2, 1, 1, 0, 0])
        self.assertEqual(sum(rates), 4)
        self.assertEqual(measurements(result)["makespan"], {(): 2.5})


class GpuAllocationTests(unittest.TestCase):
    def test_allocated_ids_are_deduplicated_across_pd_pools(self):
        config = {
            "system": {
                "tp": 1,
                "prefill_instances": [
                    {
                        "model_runner": {
                            "hardware": {
                                "gpu_count": 2,
                                "gpu_ids": ["g0", "g1"],
                                "topology": {"gpu_count": 8},
                            }
                        }
                    }
                ],
                "decode_instances": [
                    {
                        "model_runner": {
                            "hardware": {
                                "gpu_count": 2,
                                "gpu_ids": ["g2", "g3"],
                            }
                        }
                    }
                ],
            }
        }
        self.assertEqual(physical_gpu_count(config), 4)
        config["system"]["allow_gpu_sharing"] = True
        config["system"]["decode_instances"][0]["model_runner"]["hardware"][
            "gpu_ids"
        ] = ["g1", "g2"]
        self.assertEqual(physical_gpu_count(config), 3)

    def test_legacy_counts_and_unknown_allocations(self):
        leaf = {"model_runner": {"hardware": {"gpu_count": 2}}}
        self.assertEqual(physical_gpu_count({"system": leaf}), 2)
        config = {"system": {"prefill_instances": [leaf], "decode_instances": [leaf]}}
        self.assertEqual(physical_gpu_count(config), 4)
        self.assertIsNone(physical_gpu_count({"system": {"model_runner": {"tp": 2}}}))
        config["system"]["allow_gpu_sharing"] = True
        self.assertIsNone(physical_gpu_count(config))
        config["system"]["allow_gpu_sharing"] = False
        config["system"]["decode_instances"] = [
            {
                "model_runner": {
                    "hardware": {
                        "gpu_count": 2,
                        "gpu_ids": ["g0", "g1"],
                    }
                }
            }
        ]
        self.assertIsNone(physical_gpu_count(config))


class ValidationTests(unittest.TestCase):
    def test_rejects_incomplete_or_invalid_timings(self):
        for node in [
            llm([1], output_tokens=2),
            llm([1, 2], output_tokens=1),
            llm([math.nan]),
            llm([math.inf]),
            llm([-1]),
            llm([True]),
            llm([1], interval=-1),
        ]:
            with self.subTest(node=node), self.assertRaises(ValueError):
                run(request([node]))

    def test_rejects_invalid_dags(self):
        rows = [
            request([llm([1], next=[1])]),
            request([llm([1], next=[0])], initial=[]),
            request([llm([1]), llm([1], next=[1])]),
            request([llm([1]), llm([1])], initial=[0]),
            request([llm([1], next=[1, 1]), llm([1])]),
            request(initial=[0, 0]),
            request(nodes=[]),
        ]
        for row in rows:
            with self.subTest(row=row), self.assertRaises(ValueError):
                run(row)

    def test_matching_checks_workload_but_tolerates_absent_trace(self):
        reference = run(request([llm([1], input_token_ids=[1, 2, 3, 4])]))
        match_runs(reference, run(request([llm([2])]), label="candidate"))
        for row in [
            request([llm([2])], request_id=1),
            request([llm([2])], arrival=1),
            request([llm([2], interval=1)]),
            request([llm([2], input_tokens=5)]),
            request([llm([2], input_token_ids=[4, 3, 2, 1])]),
            request([llm([2, 2])]),
        ]:
            with self.subTest(row=row), self.assertRaises(ValueError):
                match_runs(reference, run(row, label="candidate"))

    def test_loader_reports_line_and_duplicate_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.jsonl"
            path.write_text(json.dumps(request()) + "\n" + json.dumps(request()))
            with self.assertRaisesRegex(ValueError, r":2: duplicate request_id"):
                load_run(str(path))
            path.write_text("{bad json}\n")
            with self.assertRaisesRegex(ValueError, r":1:"):
                load_run(str(path))
            path.write_text("\n")
            with self.assertRaisesRegex(ValueError, "empty result"):
                load_run(str(path))


class CliTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.reference = self.root / "real.jsonl"
        self.candidate = self.root / "sim.jsonl"
        self.reference.write_text(json.dumps(request()) + "\n")
        self.candidate.write_text(json.dumps(request([llm([2, 4, 6])])) + "\n")

    def command(self, script, *arguments):
        return subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / script),
                str(self.reference),
                str(self.candidate),
                *map(str, arguments),
            ],
            capture_output=True,
            text=True,
            check=False,
        )

    def test_compare_report_and_multiple_candidates(self):
        output = self.root / "report.json"
        process = self.command(
            "compare.py",
            f"repeat={self.reference}",
            "--output",
            output,
            "--metrics",
            "ttft",
            "tpot",
            "--percentiles",
            "50",
            "99",
        )
        self.assertEqual(process.returncode, 0, process.stderr)
        report = json.loads(output.read_text())
        self.assertEqual(len(report["metrics"]), 4)
        self.assertEqual(report["metrics"][0]["paired"]["mape_pct"], 100)
        self.assertIn("MAPE %", process.stdout)
        self.assertIn("p99", process.stdout)

    def test_compare_failure_does_not_create_report(self):
        output = self.root / "report.json"
        self.candidate.write_text(json.dumps(request([llm([1])])) + "\n")
        process = self.command("compare.py", "--output", output)
        self.assertNotEqual(process.returncode, 0)
        self.assertIn("output_tokens differs", process.stderr)
        self.assertFalse(output.exists())

    def test_output_cannot_replace_input(self):
        before = self.reference.read_bytes()
        for script in ("compare.py", "draw.py"):
            process = self.command(script, "--output", self.reference)
            self.assertNotEqual(process.returncode, 0)
            self.assertIn("overwrite", process.stderr)
            self.assertEqual(self.reference.read_bytes(), before)

    def test_rejects_duplicate_labels(self):
        with self.assertRaisesRegex(ValueError, "duplicate run labels"):
            load_runs([str(self.reference), str(self.reference)])

    def test_draw_cdf_scatter_and_time_series(self):
        try:
            import matplotlib  # noqa: F401
        except ImportError:
            self.skipTest("matplotlib is required for drawing")
        for view, suffix in (("cdf", "svg"), ("scatter", "png")):
            output = self.root / f"{view}.{suffix}"
            process = self.command(
                "draw.py",
                "--output",
                output,
                "--view",
                view,
                "--metrics",
                "ttft",
                "tpot",
                "throughput",
                "cumulative",
            )
            self.assertEqual(process.returncode, 0, process.stderr)
            self.assertGreater(output.stat().st_size, 1000)
        svg = (self.root / "cdf.svg").read_text()
        self.assertIn("Time since first arrival", svg)
        self.assertIn("TTFT", svg)

    def test_summary_and_separate_throughput_plot(self):
        (self.root / "config.toml").write_text(
            "[[system.prefill_instances]]\n"
            "[system.prefill_instances.model_runner.hardware]\n"
            'gpu_count = 2\ngpu_ids = ["g0", "g1"]\n'
            "[[system.decode_instances]]\n"
            "[system.decode_instances.model_runner.hardware]\n"
            'gpu_count = 2\ngpu_ids = ["g2", "g3"]\n'
        )
        output = self.root / "metrics.svg"
        summary = self.root / "metrics.json"
        process = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/draw.py"),
                str(self.reference),
                "--output",
                str(output),
                "--metrics-json",
                str(summary),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(process.returncode, 0, process.stderr)
        metrics = json.loads(summary.read_text())
        self.assertEqual(metrics["tokens_per_user"]["mean"], 0.4)
        self.assertEqual(metrics["tokens_per_user"]["samples"], 1)
        self.assertEqual(metrics["tokens_per_gpu"]["mean"], 0.125)
        self.assertEqual(metrics["tokens_per_gpu"]["gpu_count"], 4)
        svg = output.read_text()
        for label in ("TTFT", "Output throughput"):
            self.assertIn(label, svg)
        self.assertNotIn("Tokens/s/user", svg)
        self.assertNotIn("Tokens/s/GPU", svg)
        self.candidate.write_text(
            json.dumps(request([llm([2, 4, 6])], arrival=7)) + "\n"
        )
        xy = self.root / "throughput.svg"
        process = self.command("draw_throughput.py", "--output", xy)
        self.assertEqual(process.returncode, 0, process.stderr)
        svg = xy.read_text()
        for label in ("Tokens/s/user", "Tokens/s/GPU", "4 GPUs", "excluding TTFT"):
            self.assertIn(label, svg)
        self.assertNotIn("CDF", svg)
        self.assertNotIn("Time since first arrival", svg)

    def test_throughput_has_one_point_per_deployment(self):
        from matplotlib.collections import PathCollection

        first = run(request(), request([llm([4, 5])], request_id=1), label="first")
        second = run(
            request([llm([2, 4, 6])]),
            request([llm([8, 10])], request_id=1),
            label="second",
        )
        first.gpu_count = second.gpu_count = 4
        with patch("matplotlib.figure.Figure.savefig", autospec=True) as save:
            draw_throughput([first, second], self.root / "throughput.png")
        figure = save.call_args.args[0]
        self.assertEqual(len(figure.axes), 1)
        ax = figure.axes[0]
        points = [
            item.get_offsets()
            for item in ax.collections
            if isinstance(item, PathCollection)
        ]
        self.assertEqual([len(point) for point in points], [1, 1])
        for actual, expected in zip(points, [(0.3, 5 / 36), (0.15, 5 / 72)]):
            self.assertAlmostEqual(actual[0][0], expected[0])
            self.assertAlmostEqual(actual[0][1], expected[1])
        self.assertEqual(ax.get_xlabel(), "Tokens/s/user")
        self.assertEqual(ax.get_ylabel(), "Tokens/s/GPU")

    def test_throughput_leaves_undefined_rates_unplotted(self):
        output = self.root / "throughput.svg"
        for known_gpu in (False, True):
            if known_gpu:
                (self.root / "config.toml").write_text(
                    "[system.model_runner.hardware]\ngpu_count = 4\n"
                )
                self.reference.write_text(json.dumps(request([llm([1, 0, 0])])) + "\n")
            process = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/draw_throughput.py"),
                    str(self.reference),
                    "--output",
                    str(output),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(process.returncode, 0, process.stderr)
            svg = output.read_text()
            self.assertIn("No eligible deployment point", svg)
            self.assertNotIn("PathCollection", svg)


if __name__ == "__main__":
    unittest.main()
