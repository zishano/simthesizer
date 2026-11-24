"""Pressure inputs and their frozen preparation; no simulation or API calls."""

import contextlib
import io
import json
import subprocess
import sys
import tempfile
import tomllib
import unittest
from pathlib import Path
from unittest.mock import patch

import workloads

sys.path.insert(0, str(workloads.ROOT / "agents/find_pareto_frontier"))
import launcher


class PressureSuiteTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="qwen-pressure-")
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)

    def generate(self, name="suite"):
        path = workloads.generate(self.base / name)
        return path.parent, tomllib.loads(path.read_text())

    def test_determinism_and_independent_seed(self):
        first, _ = self.generate("one")
        second, _ = self.generate("two")
        for name in ["suite.json", "anchor-realistic.jsonl"] + [
            profile + ".jsonl" for profile in workloads.PROFILES
        ]:
            self.assertEqual((first / name).read_bytes(), (second / name).read_bytes())
        for profile in workloads.PROFILES:
            self.assertNotEqual(
                workloads.synthetic_rows(profile, 24, 0),
                workloads.synthetic_rows(profile, 24, 1),
            )

    def test_native_complete_inputs_and_visible_splits(self):
        root, config = self.generate()
        manifest = json.loads((root / "suite.json").read_text())
        self.assertFalse(manifest["hidden_holdout"])
        self.assertEqual(manifest["visibility"], "visible")
        self.assertEqual(len(config["workloads"]), 4)
        for row in manifest["workloads"]:
            path = root / row["path"]
            records = workloads.load_workload(path)
            self.assertEqual(row["sha256"], workloads.sha256(path.read_bytes()))
            self.assertEqual(row["requests"], len(records))
            expected = 100 if row["origin"] == "canonical_anchor" else 24
            self.assertEqual(len(records), expected)
            self.assertEqual(row["split"], "train" if expected == 100 else "validation")
            self.assertTrue(all(r["initial"] == [0] for r in records))
            self.assertTrue(all(len(r["sub_requests"]) == 1 for r in records))
            for request in records:
                node = request["sub_requests"][0]
                self.assertGreater(node["input_tokens"], 0)
                self.assertGreater(node["output_tokens"], 0)
                self.assertEqual(node["next"], [])
                self.assertNotIn("token_latency", node)
            self.assertEqual(
                row["output_tokens"],
                sum(r["sub_requests"][0]["output_tokens"] for r in records),
            )
        arrivals = [
            row["arrival_time"]
            for row in workloads.load_workload(root / "bursty.jsonl")
        ]
        self.assertEqual(len(set(arrivals)), 4)

    def test_anchor_preserved_and_only_workloads_assets_changed(self):
        root, config = self.generate()
        self.assertEqual(
            (root / "anchor-realistic.jsonl").read_bytes(),
            (workloads.ROOT / workloads.ANCHOR).read_bytes(),
        )
        original = tomllib.loads((workloads.HERE / "experiment.toml").read_text())
        config["workloads"] = original["workloads"]
        self.assertEqual(config["inputs"]["assets"][-1], str(root / "suite.json"))
        config["inputs"]["assets"].pop()
        self.assertEqual(config, original)

    def test_generated_paths_prepare_and_freeze_manifest(self):
        root, config = self.generate()
        self.assertEqual(launcher.config_issues(config), [])
        for row in config["workloads"]:
            self.assertTrue(launcher.input_file(workloads.ROOT, row["path"]).is_file())
        with (
            patch.object(launcher, "source_files", return_value=[]),
            patch.object(launcher, "build_identity", return_value={}),
            patch.object(launcher, "runtime_gaps", return_value=[]),
            patch.object(launcher, "render_task", return_value="fixture\n"),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            run = launcher.prepare(
                workloads.ROOT,
                config,
                self.base / "runs",
                {},
                "none",
                profile="qwen_moe",
            )
        frozen = json.loads((run / "task/manifest.json").read_text())
        for row in frozen["experiment"]["workloads"]:
            path = run / Path(row["path"]).relative_to("/workspace")
            self.assertEqual(
                path.read_bytes(), (root / (row["id"] + ".jsonl")).read_bytes()
            )
            self.assertIn(
                str(path.relative_to(run / "inputs")), frozen["inputs_sha256"]
            )
        assets = frozen["experiment"]["inputs"]["assets"]
        suite = run / Path(assets[-1]).relative_to("/workspace")
        self.assertEqual(suite.read_bytes(), (root / "suite.json").read_bytes())

    def test_existing_directory_and_symlink_are_not_overwritten(self):
        root, _ = self.generate()
        before = {p.name: p.read_bytes() for p in root.iterdir()}
        with self.assertRaises(FileExistsError):
            workloads.generate(root)
        self.assertEqual(before, {p.name: p.read_bytes() for p in root.iterdir()})
        empty = self.base / "empty"
        empty.mkdir()
        with self.assertRaises(FileExistsError):
            workloads.generate(empty)
        alias = self.base / "alias"
        alias.symlink_to(root, target_is_directory=True)
        with self.assertRaises(FileExistsError):
            workloads.generate(alias)

    def test_invalid_counts_seeds_and_truncated_anchor(self):
        for count, seed in [(0, 0), (129, 0), (True, 0), (24, -1), (24, 2**64)]:
            with self.subTest(count=count, seed=seed), self.assertRaises(ValueError):
                workloads.synthetic_rows("bursty", count, seed)
        shortened = self.base / "shortened.jsonl"
        shortened.write_text(
            (workloads.ROOT / workloads.ANCHOR).read_text().splitlines()[0] + "\n"
        )
        with (
            patch.object(workloads, "ANCHOR", shortened),
            self.assertRaisesRegex(ValueError, "100"),
        ):
            workloads.generate(self.base / "invalid")
        self.assertFalse((self.base / "invalid").exists())

    def test_cli_from_other_directory_with_spaces(self):
        command = [
            sys.executable,
            str(workloads.HERE / "workloads.py"),
            "--output",
            "directory with spaces",
        ]
        result = subprocess.run(
            command, cwd=self.base, capture_output=True, text=True, check=True
        )
        config_path = self.base / "directory with spaces/experiment.toml"
        self.assertEqual(result.stdout.strip(), str(config_path))
        config = tomllib.loads(config_path.read_text())
        self.assertTrue(all(Path(row["path"]).is_file() for row in config["workloads"]))
        repeated = subprocess.run(
            command, cwd=self.base, capture_output=True, text=True, check=False
        )
        self.assertEqual(repeated.returncode, 2)
        self.assertIn("File exists", repeated.stderr)


if __name__ == "__main__":
    unittest.main()
