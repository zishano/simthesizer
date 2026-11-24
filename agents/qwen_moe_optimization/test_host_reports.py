"""Exercise report wiring with the real frozen result parser and host scorer."""

import copy
import json
import shutil
import unittest

import contract
import control
import runner
import test_runner


class HostReportTests(unittest.TestCase):
    def setUp(self):
        fixture = test_runner.JobTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        fixture.evidence()
        self.fixture = fixture
        self.result = runner.collect_job(
            fixture.run, fixture.job, fixture.meta, fixture.case["id"]
        )
        self.assertEqual(self.result["status"], "EVALUATED")
        destination = fixture.run / "artifacts/baseline/cases/fixture"
        shutil.copytree(fixture.job / "collected", destination)
        self.record = {
            **self.result,
            "case": fixture.case,
            "directory": "baseline/cases/fixture",
            "event": "finished",
            "status": "VALID_FEASIBLE",
        }
        self.baseline = {"records": [self.record]}

    def test_baseline_diagnostics_loads_real_result_parser(self):
        control.baseline_diagnostics(
            self.fixture.run, self.fixture.manifest, self.baseline
        )
        report = json.loads(
            (self.fixture.run / "artifacts/baseline-diagnostics.json").read_text()
        )
        row = report["records"][0]
        self.assertEqual(row["case_id"], self.fixture.case["id"])
        self.assertEqual(row["evidence_sha256"], self.result["evidence_sha256"])
        self.assertEqual(row["objectives"]["tokens_per_user"], 100)
        self.assertIn("window", row["diagnostics"])
        self.assertEqual(self.result["objectives"]["tokens_per_user"], 100)
        reference = contract.resolve_evidence(
            self.fixture.run,
            self.fixture.manifest,
            {
                "kind": "baseline",
                "id": self.fixture.case["id"],
                "observation": "fixture tail",
            },
            self.fixture.case["id"],
            [],
            self.baseline,
        )
        self.assertEqual(reference["objectives"], row["objectives"])

    def test_frontier_uses_host_total_throughput_and_user_speed(self):
        improved = copy.deepcopy(self.result)
        improved["metrics"]["tokens_per_gpu"]["mean"] *= 2
        improved["metrics"]["tokens_per_user"]["mean"] *= 2
        improved.update(
            id="a" * 32,
            event="finished",
            candidate_id="b" * 64,
            policy_kind="candidate",
            case_id=self.fixture.case["id"],
        )
        control.append(self.fixture.run, improved)
        control.update_frontier(self.fixture.run, self.fixture.manifest, self.baseline)
        report = json.loads((self.fixture.run / "artifacts/frontier.json").read_text())
        frontier = report["groups"][self.fixture.case["id"]]
        self.assertEqual(len(frontier), 1)
        self.assertEqual(frontier[0]["candidate_id"], "b" * 64 + ":candidate")
        self.assertEqual(frontier[0]["objectives"]["tokens_per_user"], 200)
        self.assertEqual(frontier[0]["verification"], "EXPLORATORY")
        # Legacy archive ignores the additional objectives and merges equal points.
        legacy = copy.deepcopy(self.fixture.manifest)
        legacy["experiment"]["research"].pop("workflow")
        control.update_frontier(self.fixture.run, legacy, self.baseline)
        report = json.loads((self.fixture.run / "artifacts/frontier.json").read_text())
        frontier = report["groups"][self.fixture.case["id"]]
        self.assertEqual(len(frontier[0]["equivalent_candidates"]), 1)


if __name__ == "__main__":
    unittest.main()
