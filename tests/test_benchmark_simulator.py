"""Exercise measurement, exact-output gating and paired ordering using fake CLIs."""

import importlib.util
import json
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "benchmark_simulator", ROOT / "tools/benchmark_simulator.py"
)
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


class BenchmarkTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="sim-benchmark-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.config = self.root / "config.toml"
        self.workload = self.root / "workload.jsonl"
        self.config.write_text("fixture = true\n")
        self.workload.write_text(
            json.dumps(
                {
                    "request_id": 0,
                    "arrival_time": 1.0,
                    "initial": [0],
                    "sub_requests": [
                        {
                            "kind": "llm",
                            "input_tokens": 3,
                            "output_tokens": 2,
                            "next": [],
                        }
                    ],
                }
            )
            + "\n"
        )

    def executable(
        self, name, *, latency=1, delay=0.03, child=False, linger=False, invalid=None
    ):
        path = self.root / name
        path.write_text(
            f"#!{sys.executable}\n"
            "import argparse, json, pathlib, subprocess, sys, time\n"
            "p=argparse.ArgumentParser()\n"
            "for key in ('config','workload','output-jsonl','output-policy-stats'): p.add_argument('--'+key)\n"
            "a=p.parse_args()\n"
            + (
                "child=subprocess.Popen([sys.executable, '-c', 'import time, signal; signal.signal(signal.SIGTERM, signal.SIG_IGN); x=bytearray(16000000); time.sleep(30)'])\n"
                if linger
                else "child=subprocess.Popen([sys.executable, '-c', 'import time; x=bytearray(16000000); time.sleep(0.15)'])\n"
                if child
                else ""
            )
            + (
                f"pathlib.Path({str(self.root / 'child.pid')!r}).write_text(str(child.pid))\n"
                if child or linger
                else ""
            )
            + f"time.sleep({delay!r})\n"
            + ("child.wait()\n" if child and not linger else "")
            + "rows=[json.loads(line) for line in pathlib.Path(a.workload).read_text().splitlines()]\n"
            + f"for row in rows:\n for node in row['sub_requests']: node['token_latency']=[{latency!r}]*node['output_tokens']\n"
            + (
                "rows=[]\n"
                if invalid == "empty"
                else "rows[0]['sub_requests'][0]['token_latency']=[]\n"
                if invalid == "partial"
                else ""
            )
            + "pathlib.Path(a.output_jsonl).write_text(''.join(json.dumps(row)+'\\n' for row in rows))\n"
            "print('fixture completed')\n"
        )
        path.chmod(0o755)
        return path

    def arguments(self, baseline, candidate, *extra):
        return bench.parser().parse_args(
            [
                "--baseline",
                str(baseline),
                "--candidate",
                str(candidate),
                "--case",
                "fixture",
                str(self.config),
                str(self.workload),
                "--output",
                str(self.root / "output"),
                "--cwd",
                str(self.root),
                "--pairs",
                "3",
                "--warmups",
                "1",
                "--max-wall-regression",
                "10",
                "--sample-interval",
                "0.005",
                *extra,
            ]
        )

    def test_sequential_alternation_warmup_and_tree_samples(self):
        executable = self.executable("native", child=True)
        report = bench.benchmark(self.arguments(executable, executable))
        self.assertEqual(report["status"], "passed", report)
        measured = [r for r in report["records"] if not r["warmup"]]
        self.assertEqual(
            [r["variant"] for r in measured],
            ["baseline", "candidate", "candidate", "baseline", "baseline", "candidate"],
        )
        self.assertEqual(len(report["records"]), 8)
        self.assertEqual(report["summary"]["fixture"]["pairs"], 3)
        for record in report["records"]:
            directory = Path(record["directory"])
            samples = [
                json.loads(line)
                for line in (directory / "rss.jsonl").read_text().splitlines()
            ]
            self.assertEqual(len(samples), record["rss_samples"])
            self.assertGreater(
                record["sampled_process_tree_peak_rss_bytes"], 20_000_000
            )
            self.assertIn("fixture completed", (directory / "run.log").read_text())
        self.assertIn(str(executable), report["identities"])

    def test_output_mismatch_fails_and_keeps_both_results(self):
        report = bench.benchmark(
            self.arguments(self.executable("old"), self.executable("new", latency=2))
        )
        self.assertEqual(report["status"], "failed")
        self.assertIn("byte-exact result mismatch", report["failures"][0])
        self.assertEqual(len(report["records"]), 2)
        self.assertNotEqual(*(r["result_sha256"] for r in report["records"]))

    def test_timeout_keeps_failure_measurement(self):
        executable = self.executable("slow", delay=30)
        report = bench.benchmark(
            self.arguments(executable, executable, "--timeout", "0.05")
        )
        self.assertEqual(report["status"], "failed")
        self.assertIn("exceeded", report["failures"][0])
        saved = json.loads(
            next((self.root / "output").glob("*/measurement.json")).read_text()
        )
        self.assertEqual(saved["status"], "failed")
        self.assertLess(saved["wall_s"], 5)
        self.assertIsNotNone(saved["exit_code"])

    def test_regression_gates_ignore_warmup_and_use_paired_median(self):
        records = []
        for pair, ratio in enumerate((1.01, 1.08, 1.09)):
            for variant, wall in (("baseline", 10), ("candidate", 10 * ratio)):
                records.append(
                    {
                        "case": "one",
                        "pair": pair,
                        "warmup": False,
                        "variant": variant,
                        "wall_s": wall,
                        "sampled_process_tree_peak_rss_bytes": 100
                        if variant == "baseline"
                        else 110,
                    }
                )
        records.append(
            {
                "case": "one",
                "pair": 0,
                "warmup": True,
                "variant": "candidate",
                "wall_s": 10000,
            }
        )
        summary, failures = bench.summarize(records, ["one"], 0.05, 0.05)
        self.assertAlmostEqual(summary["one"]["median_paired_wall_ratio"], 1.08)
        self.assertEqual(len(failures), 2)
        self.assertFalse(bench.summarize(records, ["one"], 0.1, None)[1])

    def test_spawn_failure_keeps_measurement(self):
        missing = self.root / "missing"
        directory = self.root / "failed-run"
        with self.assertRaises(FileNotFoundError):
            bench.measure(
                missing, self.config, self.workload, directory, self.root, 1, 0.02
            )
        saved = json.loads((directory / "measurement.json").read_text())
        self.assertEqual(saved["status"], "failed")
        self.assertIsNone(saved["exit_code"])

    def test_equal_empty_and_partial_outputs_cannot_pass(self):
        for invalid in ("empty", "partial"):
            with self.subTest(invalid=invalid):
                native = self.executable(invalid, invalid=invalid)
                args = self.arguments(native, native)
                args.output = str(self.root / f"output-{invalid}")
                report = bench.benchmark(args)
                self.assertEqual(report["status"], "failed")
                self.assertEqual(report["records"], [])

    def assert_child_stopped(self):
        identity = int((self.root / "child.pid").read_text())
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            try:
                state = (
                    Path(f"/proc/{identity}/stat")
                    .read_text()
                    .rsplit(")", 1)[1]
                    .split()[0]
                )
            except FileNotFoundError:
                return
            if state == "Z":
                return
            time.sleep(0.01)
        self.fail(f"child {identity} survived cleanup")

    def test_successful_parent_with_live_worker_fails_and_cleans_up(self):
        native = self.executable("survivor", linger=True)
        report = bench.benchmark(self.arguments(native, native))
        self.assertEqual(report["status"], "failed")
        self.assertIn("surviving descendants", report["failures"][0])
        self.assert_child_stopped()

    def test_sigterm_marks_incomplete_and_stops_native_tree(self):
        native = self.executable("long", linger=True, delay=30)
        args = self.arguments(native, native)
        command = [
            sys.executable,
            str(ROOT / "tools/benchmark_simulator.py"),
            "--baseline",
            args.baseline,
            "--candidate",
            args.candidate,
            "--case",
            *args.case[0],
            "--output",
            args.output,
            "--cwd",
            args.cwd,
        ]
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        try:
            deadline = time.monotonic() + 5
            while (
                not (self.root / "child.pid").exists() and time.monotonic() < deadline
            ):
                if process.poll() is not None:
                    self.fail(process.communicate())
                time.sleep(0.01)
            self.assertTrue((self.root / "child.pid").exists())
            process.terminate()
            process.communicate(timeout=10)
            self.assertNotEqual(process.returncode, 0)
            report = json.loads((Path(args.output) / "report.json").read_text())
            self.assertEqual(report["status"], "incomplete")
            self.assertIn("SIGTERM", report["failures"][0])
            self.assert_child_stopped()
        finally:
            if process.poll() is None:
                process.kill()
            process.communicate()

    def test_rss_gate_requires_positive_samples_from_every_pair_member(self):
        records = [
            {
                "case": "one",
                "pair": pair,
                "warmup": False,
                "variant": variant,
                "wall_s": 1,
                "sampled_process_tree_peak_rss_bytes": 0
                if pair == 1 and variant == "candidate"
                else 100,
            }
            for pair in range(3)
            for variant in ("baseline", "candidate")
        ]
        summary, failures = bench.summarize(records, ["one"], 0.05, 0.05)
        self.assertEqual(len(summary["one"]["paired_sampled_rss_ratios"]), 2)
        self.assertEqual(len(failures), 1)

    def test_rejects_path_unsafe_case_name(self):
        executable = self.executable("native")
        args = self.arguments(executable, executable)
        args.case[0][0] = "../outside"
        with self.assertRaisesRegex(ValueError, "path-safe"):
            bench.benchmark(args)
        self.assertFalse((self.root / "output").exists())


if __name__ == "__main__":
    unittest.main()
