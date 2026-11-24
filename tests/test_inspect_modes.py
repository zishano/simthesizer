"""Default metric plots must not collect history or change token timings."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class InspectModesTests(unittest.TestCase):
    def test_default_reports_graphs_and_reference_are_equivalent(self):
        with tempfile.TemporaryDirectory() as directory:
            outputs = []
            for name, flags in (
                ("metrics", []),
                ("reports", ["--reports"]),
                ("graphs", ["--full-graphs"]),
                ("reference", ["--reference-every-round"]),
                ("profile", ["--wall-profile"]),
                ("profile_reports", ["--wall-profile", "--reports"]),
            ):
                root = Path(directory) / name
                result = subprocess.run(
                    [
                        sys.executable,
                        "scripts/inspect_rooflang.py",
                        "--pd",
                        "--execution",
                        "shared",
                        "--gpu-model",
                        "h200",
                        "--tp",
                        "2",
                        "--output-dir",
                        str(root),
                        *flags,
                    ],
                    cwd=ROOT,
                    text=True,
                    capture_output=True,
                    timeout=120,
                    check=False,
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn("100%", result.stdout)
                self.assertTrue(
                    (root / "metrics.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
                )
                self.assertTrue(
                    (root / "throughput.png")
                    .read_bytes()
                    .startswith(b"\x89PNG\r\n\x1a\n")
                )
                outputs.append(
                    (
                        (root / "result.jsonl").read_bytes(),
                        json.loads((root / "metrics.json").read_text()),
                    )
                )
                if name in ("metrics", "reference", "profile"):
                    self.assertEqual(
                        {p.name for p in root.iterdir()},
                        {
                            "config.toml",
                            "workload.jsonl",
                            "simulation.log",
                            "result.jsonl",
                            "metrics.json",
                            "metrics.png",
                            "throughput.png",
                            "draw.py.log",
                            "draw_throughput.py.log",
                            "execution.json",
                        }
                        | (
                            {
                                "wall-profile-native.json",
                                "wall-profile.json",
                                "wall-profile.csv",
                                "wall-profile.txt",
                            }
                            if name == "profile"
                            else set()
                        ),
                    )
                    self.assertNotIn("--audit-dir", (root / "config.toml").read_text())
                else:
                    for path in (
                        "outer.jsonl",
                        "state.jsonl",
                        "operator-costs.csv",
                        "shared-check.json",
                    ):
                        self.assertTrue((root / path).is_file(), path)
                    self.assertEqual(
                        (root / "execution-graph.html").exists(), name == "graphs"
                    )
                if name.startswith("profile"):
                    profile = json.loads((root / "wall-profile.json").read_text())
                    self.assertEqual(profile["status"], "complete")
                    self.assertEqual(profile["progress"]["completed"], 1)
                    for stage in ("prepare", "simulation", "metrics", "plots"):
                        self.assertGreater(profile["phases"][stage], 0)
                    rows = profile["rows"]
                    for parent in rows:
                        children = [r for r in rows if r["parent"] == parent["key"]]
                        if children:
                            self.assertAlmostEqual(
                                sum(r["seconds"] for r in children), parent["seconds"]
                            )
                    if name == "profile_reports":
                        self.assertGreater(profile["phases"]["reports"], 0)
                        self.assertIn(
                            "wall-profile.txt", (root / "index.html").read_text()
                        )
            self.assertTrue(all(value == outputs[0] for value in outputs))
