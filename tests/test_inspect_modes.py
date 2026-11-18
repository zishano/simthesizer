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
                if name in ("metrics", "reference"):
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
                        },
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
            self.assertTrue(all(value == outputs[0] for value in outputs))
