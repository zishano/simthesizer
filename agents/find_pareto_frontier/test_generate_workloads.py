import contextlib
import hashlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import baseline_store
import generate_workloads as generator

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from results import load_workload


def parameters(**changes):
    return {
        "num_requests": 12,
        "arrival_interval_s": 0.01,
        "arrival_start_s": 0.0,
        "arrival_jitter": 0.25,
        "burst_size": 4,
        "short_input_tokens": [1, 3],
        "long_input_tokens": [8, 12],
        "output_tokens": [1, 4],
        "max_context_tokens": 16,
        "train_seed": 17,
        "validation_seed": 18,
        "profiles": list(generator.PROFILES),
        **changes,
    }


class WorkloadGenerationTests(unittest.TestCase):
    def test_native_schema_bounds_and_file_identities_for_every_profile_split(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "suite"
            manifest = generator.write_suite(root, parameters())
            self.assertEqual(len(manifest["workloads"]), 8)
            self.assertTrue(manifest["assumptions"]["synthetic"])
            self.assertFalse(manifest["assumptions"]["holdout"])
            self.assertEqual(
                json.loads((root / "manifest.json").read_bytes()), manifest
            )
            for entry in manifest["workloads"]:
                path = root / entry["path"]
                self.assertEqual(
                    hashlib.sha256(path.read_bytes()).hexdigest(), entry["sha256"]
                )
                rows = load_workload(path)
                self.assertEqual(len(rows), 12)
                self.assertEqual([row["request_id"] for row in rows], list(range(12)))
                arrivals = [row["arrival_time"] for row in rows]
                self.assertEqual(arrivals, sorted(arrivals))
                self.assertEqual(arrivals[0], entry["first_arrival_s"])
                self.assertEqual(arrivals[-1], entry["last_arrival_s"])
                for row in rows:
                    self.assertEqual(row["initial"], [0])
                    self.assertEqual(len(row["sub_requests"]), 1)
                    llm = row["sub_requests"][0]
                    self.assertEqual(llm["kind"], "llm")
                    self.assertEqual(llm["next"], [])
                    self.assertGreater(llm["input_tokens"], 0)
                    self.assertGreater(llm["output_tokens"], 0)
                    self.assertLessEqual(llm["input_tokens"] + llm["output_tokens"], 16)

    def test_reproduction_is_independent_of_output_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            roots = [Path(directory) / name for name in ("first", "second")]
            for root in roots:
                generator.write_suite(root, parameters())
            for path in roots[0].iterdir():
                self.assertEqual(path.read_bytes(), (roots[1] / path.name).read_bytes())

    def test_distinct_seed_inputs_and_profile_selection_isolation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "all"
            generator.write_suite(root, parameters())
            one = Path(directory) / "one"
            generator.write_suite(one, parameters(profiles=["mixed"]))
            for profile in generator.PROFILES:
                self.assertNotEqual(
                    (root / f"{profile}-train.jsonl").read_bytes(),
                    (root / f"{profile}-validation.jsonl").read_bytes(),
                )
            for split in ("train", "validation"):
                self.assertEqual(
                    (root / f"mixed-{split}.jsonl").read_bytes(),
                    (one / f"mixed-{split}.jsonl").read_bytes(),
                )

    def test_burst_group_timing_preserves_ids_and_mean_offered_rate(self):
        settings = parameters(num_requests=10, arrival_jitter=0.0, arrival_start_s=2.0)
        rows = list(generator.generate(settings, "burst", 17))
        self.assertEqual([row["request_id"] for row in rows], list(range(10)))
        self.assertEqual(
            [row["arrival_time"] for row in rows],
            [2.0] * 4 + [2.04] * 4 + [2.08] * 2,
        )

    def test_length_profiles_exercise_distinct_context_pressure(self):
        settings = parameters(short_input_tokens=[2, 2], long_input_tokens=[10, 10])
        for profile, expected in (
            ("sustained", [2] * 12),
            ("burst", [2] * 12),
            ("mixed", [2, 10] * 6),
            ("context", [10] * 12),
        ):
            with self.subTest(profile=profile):
                rows = generator.generate(settings, profile, 17)
                self.assertEqual(
                    [row["sub_requests"][0]["input_tokens"] for row in rows], expected
                )

    def test_invalid_parameters_fail_before_creating_output(self):
        invalid = (
            {"num_requests": 0},
            {"num_requests": True},
            {"arrival_interval_s": 0},
            {"arrival_interval_s": float("inf")},
            {"arrival_start_s": -1},
            {"arrival_jitter": float("nan")},
            {"arrival_jitter": 1},
            {"burst_size": 0},
            {"short_input_tokens": [3, 1]},
            {"long_input_tokens": [3, 12]},
            {"output_tokens": [0, 1]},
            {"max_context_tokens": 15},
            {"max_context_tokens": 2**32},
            {
                "long_input_tokens": [2**32, 2**32],
                "max_context_tokens": 2**32 + 4,
            },
            {"validation_seed": 17},
            {"train_seed": False},
            {"profiles": []},
            {"profiles": ["mixed", "mixed"]},
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "invalid"
            for changes in invalid:
                with self.subTest(changes=changes), self.assertRaises(ValueError):
                    generator.write_suite(root, parameters(**changes))
                self.assertFalse(root.exists())

    def test_existing_inputs_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "suite"
            generator.write_suite(root, parameters())
            before = {path.name: path.read_bytes() for path in root.iterdir()}
            with self.assertRaises(FileExistsError):
                generator.write_suite(root, parameters(train_seed=19))
            self.assertEqual(
                before, {path.name: path.read_bytes() for path in root.iterdir()}
            )

    def test_unrepresentable_arrival_gap_cannot_publish_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "suite"
            with self.assertRaisesRegex(ValueError, "floating-point"):
                generator.write_suite(root, parameters(arrival_start_s=1e100))
            self.assertFalse((root / "manifest.json").exists())

    def test_generated_splits_use_existing_case_expansion(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = generator.write_suite(Path(directory) / "suite", parameters())
            cases = baseline_store.cases(
                {
                    "experiment": {
                        "workloads": manifest["workloads"],
                        "loads": {"arrival_rate_multipliers": [1.0, 2.0]},
                        "search": {"seed": 9},
                    }
                }
            )
            self.assertEqual(len(cases), 16)
            self.assertEqual(len({case["id"] for case in cases}), 16)
            self.assertEqual({case["split"] for case in cases}, {"train", "validation"})

    def test_cli_is_standalone_and_requires_explicit_scale(self):
        command = [sys.executable, str(Path(generator.__file__).resolve())]
        help_result = subprocess.run(
            command + ["--help"], capture_output=True, text=True, check=False
        )
        self.assertEqual(help_result.returncode, 0, help_result.stderr)
        self.assertIn("--num-requests", help_result.stdout)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "cli"
            with contextlib.redirect_stdout(io.StringIO()):
                generator.main(
                    [
                        "--output-dir",
                        str(root),
                        "--num-requests",
                        "3",
                        "--arrival-interval-s",
                        "0.1",
                        "--profiles",
                        "mixed",
                    ]
                )
            manifest = json.loads((root / "manifest.json").read_text())
            self.assertEqual(len(manifest["workloads"]), 2)
            self.assertEqual(manifest["parameters"]["num_requests"], 3)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            generator.main(["--output-dir", "unused"])


if __name__ == "__main__":
    unittest.main()
