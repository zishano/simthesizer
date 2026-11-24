"""Host wall accounting is a hierarchy, never simulated latency or CPU time."""

import contextlib
import csv
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from rooflang_bridge.wall_report import WallReport, build_rows


def native_profile(wall=8.0):
    return {
        "schema_version": 1,
        "wall_seconds": wall,
        "sections": {
            "engine_dispatch": {
                "calls": 12,
                "inclusive_seconds": 3.0,
                "exclusive_seconds": 1.0,
            },
            "worker_roundtrip": {
                "calls": 3,
                "inclusive_seconds": 5.0,
                "exclusive_seconds": 4.0,
            },
            "worker_encode": {
                "calls": 3,
                "inclusive_seconds": 0.3,
                "exclusive_seconds": 0.3,
            },
            "worker_decode": {
                "calls": 3,
                "inclusive_seconds": 0.7,
                "exclusive_seconds": 0.7,
            },
            "resources": {
                "calls": 12,
                "inclusive_seconds": 1.0,
                "exclusive_seconds": 1.0,
            },
            "initialization": {
                "calls": 1,
                "inclusive_seconds": 0.5,
                "exclusive_seconds": 0.5,
            },
        },
        "worker_sections": {
            "routing": {"calls": 2, "seconds": 1.0},
            "build": {"calls": 2, "seconds": 1.5},
            "response_encode": {"calls": 3, "seconds": 0.5},
        },
    }


class WallRowsTests(unittest.TestCase):
    def test_nested_work_is_not_added_twice(self):
        phases = {
            "prepare": 2.0,
            "simulation": 10.0,
            "metrics": 1.0,
            "plots": 3.0,
            "reports": 4.0,
        }
        rows = build_rows(phases, native_profile())
        by_key = {row["key"]: row for row in rows}
        self.assertEqual(len(by_key), len(rows))
        self.assertEqual(sum(r["seconds"] for r in rows if r["parent"] is None), 20)
        self.assertEqual(by_key["native:outside"]["seconds"], 2)
        self.assertAlmostEqual(by_key["native:unattributed"]["seconds"], 0.5)
        self.assertEqual(by_key["native:worker_roundtrip"]["seconds"], 4)
        self.assertEqual(by_key["worker:unattributed"]["seconds"], 1)
        self.assertEqual(by_key["worker:routing"]["calls"], 2)
        self.assertEqual(by_key["native:worker_roundtrip"]["calls"], 3)
        self.assertEqual(by_key["native:total"]["percent_of_parent"], 80)
        self.assertEqual(by_key["worker:routing"]["percent_of_parent"], 25)
        for parent in (
            "phase:simulation",
            "native:total",
            "native:worker_roundtrip",
        ):
            self.assertAlmostEqual(
                sum(r["seconds"] for r in rows if r["parent"] == parent),
                by_key[parent]["seconds"],
                msg=parent,
            )

    def test_absent_native_profile_does_not_invent_measurements(self):
        rows = build_rows({"prepare": 1.0, "simulation": 2.0}, None)
        self.assertEqual(
            {r["key"] for r in rows}, {"phase:prepare", "phase:simulation"}
        )
        self.assertTrue(all(r["parent"] is None for r in rows))

    def test_empty_native_sections_remain_unattributed(self):
        rows = build_rows(
            {"simulation": 3.0},
            {"schema_version": 1, "wall_seconds": 2.0, "sections": {}},
        )
        by_key = {r["key"]: r for r in rows}
        self.assertEqual(by_key["native:unattributed"]["seconds"], 2)
        self.assertEqual(by_key["native:outside"]["seconds"], 1)

    def test_inconsistent_or_invalid_times_are_not_silently_hidden(self):
        for change in ("native_overflow", "worker_overflow", "negative", "nan"):
            with self.subTest(change=change):
                native = native_profile()
                if change == "native_overflow":
                    native["sections"]["resources"]["exclusive_seconds"] = 9.0
                elif change == "worker_overflow":
                    native["worker_sections"]["routing"]["seconds"] = 9.0
                elif change == "negative":
                    native["sections"]["resources"]["exclusive_seconds"] = -1.0
                else:
                    native["wall_seconds"] = float("nan")
                with self.assertRaises(ValueError):
                    build_rows({"simulation": 10.0}, native)

    def test_roundoff_and_zero_duration_do_not_create_negative_percentages(self):
        native = {
            "schema_version": 1,
            "wall_seconds": 1.0,
            "sections": {
                "resources": {
                    "calls": 1,
                    "inclusive_seconds": 1.0 + 1e-8,
                    "exclusive_seconds": 1.0 + 1e-8,
                }
            },
        }
        rows = build_rows({"prepare": 0.0, "simulation": 1.0}, native)
        for row in rows:
            self.assertGreaterEqual(row["seconds"], 0)
            self.assertGreaterEqual(row["percent_of_parent"], 0)


class Clock:
    def __init__(self):
        self.now = 0.0
        self.calls = 0

    def __call__(self):
        self.calls += 1
        return self.now

    def advance(self, seconds):
        self.now += seconds


class WallReportTests(unittest.TestCase):
    def test_disabled_monitor_does_not_read_clock(self):
        clock = Clock()
        report = WallReport(clock=clock)
        report.phase("simulation")
        report.observe({"completed": 1, "total": 2, "time_s": 0.1})
        with contextlib.redirect_stdout(io.StringIO()):
            report.finish(True)
        self.assertEqual(clock.calls, 0)

    def test_running_snapshot_and_final_report_include_active_phase(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            clock = Clock()
            report = WallReport(clock=clock)
            report.enable(root)
            clock.advance(2)
            report.phase("simulation")
            clock.advance(3)
            native = {
                "schema_version": 1,
                "wall_seconds": 2.0,
                "sections": {},
                "worker_sections": {},
            }
            report.observe(
                {
                    "completed": 1,
                    "total": 2,
                    "time_s": 0.1,
                    "wall_profile": native,
                }
            )
            snapshot = json.loads((root / "wall-profile.json").read_text())
            self.assertEqual(snapshot["schema_version"], 1)
            self.assertEqual(snapshot["status"], "running")
            self.assertEqual(snapshot["phases"]["prepare"], 2)
            self.assertEqual(snapshot["phases"]["simulation"], 3)
            clock.advance(1)
            report.phase("metrics")
            clock.advance(0.5)
            report.phase("plots")
            clock.advance(0.2)
            # The final native file is newer than the last progress snapshot.
            native["wall_seconds"] = 3.5
            (root / "wall-profile-native.json").write_text(json.dumps(native))
            with contextlib.redirect_stdout(io.StringIO()):
                report.finish(True)
            final = json.loads((root / "wall-profile.json").read_text())
            self.assertEqual(final["status"], "complete")
            self.assertAlmostEqual(sum(final["phases"].values()), 6.7)
            self.assertEqual(final["phases"]["simulation"], 4)
            self.assertEqual(final["phases"]["metrics"], 0.5)
            by_key = {r["key"]: r for r in final["rows"]}
            self.assertEqual(by_key["native:total"]["seconds"], 3.5)
            self.assertEqual(by_key["native:outside"]["seconds"], 0.5)
            with (root / "wall-profile.csv").open() as stream:
                csv_rows = list(csv.DictReader(stream))
            self.assertEqual({r["key"] for r in csv_rows}, set(by_key))
            self.assertTrue((root / "wall-profile.txt").read_text().strip())

    def test_failure_keeps_partial_timing_without_claiming_completion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            clock = Clock()
            report = WallReport(clock=clock)
            clock.advance(1)
            report.enable(root, started=0.0)
            clock.advance(1)
            report.phase("simulation")
            clock.advance(3)
            with contextlib.redirect_stdout(io.StringIO()):
                report.finish(False)
            result = json.loads((root / "wall-profile.json").read_text())
            self.assertEqual(result["status"], "failed")
            self.assertEqual(result["phases"]["prepare"], 2)
            self.assertEqual(result["phases"]["simulation"], 3)


if __name__ == "__main__":
    unittest.main()
