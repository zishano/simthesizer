"""Distinguishing timing examples and host-only diagnostic collection."""

import copy
import json
import shutil
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import diagnostics
import runner
import test_runner


def request(request_id, arrival, latencies, input_tokens=10):
    return {
        "request_id": request_id,
        "arrival_time": arrival,
        "initial": [0],
        "sub_requests": [
            {
                "kind": "llm",
                "input_tokens": input_tokens,
                "output_tokens": len(latencies),
                "token_latency": latencies,
                "next": [],
            }
        ],
    }


def measured(*rows, finish=None):
    return SimpleNamespace(
        requests={row["request_id"]: row for row in rows}, service_finish_s=finish
    )


def metrics(**policy):
    return {
        "slo": {"limits": {"ttft_limit_s": 0.5, "tpot_limit_s": 1.0}},
        "policy": policy,
    }


class DiagnosticTests(unittest.TestCase):
    def test_tail_request_differs_from_last_arrival_and_native_tail_is_separate(self):
        # Request 7 emits at 1, 3, 5; request 9 arrives at 2 and emits at 2.1, 2.2.
        run = measured(request(7, 0, [1, 2, 2]), request(9, 2, [0.1, 0.1]), finish=6)
        summary = diagnostics.summarize_diagnostics(run, metrics())
        self.assertEqual(
            summary["window"],
            {
                "origin_s": 0,
                "last_arrival_s": 2,
                "arrival_span_s": 2,
                "last_output_s": 5,
                "native_finish_s": 6,
                "finish_s": 6,
                "duration_s": 6,
                "drain_s": 4,
                "native_after_last_output_s": 1,
            },
        )
        tail = summary["last_completing_requests"][0]
        self.assertEqual(tail["request_id"], 7)
        self.assertEqual((tail["ttft_s"], tail["first_s"], tail["last_s"]), (1, 1, 5))
        self.assertEqual((tail["post_first_span_s"], tail["tpot_s"]), (4, 2))
        self.assertEqual(summary["slo_failure_count"], 1)
        self.assertEqual(summary["slo_failures"][0]["failed_metrics"], ["ttft", "tpot"])

    def test_longest_request_is_not_necessarily_the_last_completion(self):
        run = measured(request(1, 0, [1, 4]), request(2, 10, [1]), finish=11)
        summary = diagnostics.summarize_diagnostics(run, metrics())
        self.assertEqual(summary["longest_requests"][0]["request_id"], 1)
        self.assertEqual(summary["last_completing_requests"][0]["request_id"], 2)
        self.assertIsNone(summary["last_completing_requests"][0]["tpot_s"])
        self.assertEqual(summary["slo_failures"][0]["failed_metrics"], ["ttft", "tpot"])
        self.assertEqual(summary["slo_failures"][1]["failed_metrics"], ["ttft"])

    def test_missing_native_end_and_policy_do_not_become_zero_or_utilization(self):
        run = measured(request(1, 4, [0.2, 0.3]), request(2, 4.1, []))
        summary = diagnostics.summarize_diagnostics(run, {})
        self.assertEqual(summary["window"]["last_output_s"], 4.5)
        self.assertIsNone(summary["window"]["finish_s"])
        self.assertIsNone(summary["window"]["duration_s"])
        self.assertIsNone(summary["window"]["native_finish_s"])
        self.assertIsNone(summary["window"]["native_after_last_output_s"])
        self.assertIsNone(summary["slo_failure_count"])
        self.assertEqual(summary["zero_output_requests"], 1)
        self.assertEqual(summary["timed_request_count"], 1)
        self.assertIn("native_service_finish", summary["missing_evidence"])
        self.assertIn("policy_samples", summary["missing_evidence"])
        self.assertIn("zero_output_request_completion", summary["missing_evidence"])
        self.assertIn("gpu_utilization", summary["missing_evidence"])
        self.assertIn("resource_causal_attribution", summary["missing_evidence"])
        self.assertIn("not utilization", " ".join(summary["limitations"]))

    def test_unsupported_dag_does_not_claim_a_complete_output_window(self):
        row = request(1, 0, [0.1])
        row["sub_requests"].append(row["sub_requests"][0].copy())
        summary = diagnostics.summarize_diagnostics(measured(row), metrics())
        self.assertEqual(summary["unsupported_requests"], 1)
        self.assertIsNone(summary["window"]["finish_s"])
        self.assertIsNone(summary["slo_failure_count"])

    def test_bounded_observations_preserve_counts_and_do_not_mutate_inputs(self):
        sample = {
            "decision": 3,
            "time_s": 0.5,
            "kind": "batch",
            "pressure": [
                {"resource": f"gpu{i}", "waiting_service_s": 3.0} for i in range(20)
            ],
            "memory": [
                {"device": str(i), "allocated_bytes": 5, "capacity_bytes": 10}
                for i in range(20)
            ],
            "registered_read_count": 25,
            "registered_reads": [{"request_id": i, "layer": 1} for i in range(20)],
            "unrelated": {"unbounded": list(range(100))},
        }
        data = metrics(batch_decisions=500, samples=[sample] * 20)
        run = measured(*(request(i, i, [1, 2]) for i in range(20)), finish=22)
        before = copy.deepcopy((run.requests, data))
        summary = diagnostics.summarize_diagnostics(run, data, limit=8)
        for name in ("last_completing_requests", "longest_requests", "slo_failures"):
            self.assertEqual(len(summary[name]), 8)
        self.assertEqual(summary["slo_failure_count"], 20)
        policy = summary["policy"]
        self.assertEqual(policy["sample_count"], 20)
        self.assertTrue(policy["samples_truncated"])
        self.assertEqual(len(policy["samples"]), 8)
        observation = policy["samples"][0]
        self.assertEqual(observation["registered_read_count"], 25)
        for name in diagnostics.OBSERVATIONS:
            self.assertEqual(len(observation[name]), 8)
            self.assertEqual(observation[name + "_count"], 20)
        self.assertNotIn("unrelated", observation)
        self.assertEqual(before, (run.requests, data))
        json.dumps(summary, allow_nan=False)
        for limit in (0, 17, True):
            with self.assertRaises(ValueError):
                diagnostics.summarize_diagnostics(run, data, limit)


class CollectionTests(unittest.TestCase):
    def fixture(self, workflow=None):
        fixture = test_runner.JobTests()
        self.addCleanup(fixture.doCleanups)
        fixture.setUp()
        settings = fixture.manifest["experiment"]["research"]
        settings.pop("workflow", None)
        if workflow:
            settings["workflow"] = workflow
        runner.launcher.save(fixture.run / "task/manifest.json", fixture.manifest)
        fixture.evidence()
        return fixture

    def test_new_workflow_hashes_host_diagnostics_and_relocates_with_evidence(self):
        fixture = self.fixture("bottleneck-v1")
        outcome = runner.collect_job(
            fixture.run, fixture.job, fixture.meta, fixture.case["id"]
        )
        self.assertEqual(outcome["status"], "EVALUATED", outcome)
        relative = Path(outcome["diagnostics_file"])
        self.assertFalse(relative.is_absolute())
        path = Path(outcome["evidence_directory"]) / relative
        self.assertEqual(
            outcome["evidence_sha256"][str(relative)], runner.launcher.digest(path)
        )
        summary = json.loads(path.read_text())
        self.assertEqual(summary["last_completing_requests"][0]["request_id"], 7)
        self.assertEqual(summary["window"]["last_output_s"], 0.51)
        self.assertGreaterEqual(outcome["diagnostics_wall_seconds"], 0)
        receipt = runner.read_json(fixture.job / "collected", "host-receipt.json")
        self.assertEqual(receipt["evidence_sha256"], outcome["evidence_sha256"])
        archived = fixture.run / "archived-evidence"
        shutil.copytree(Path(outcome["evidence_directory"]), archived)
        self.assertEqual(
            runner.launcher.digest(archived / relative),
            outcome["evidence_sha256"][str(relative)],
        )

    def test_old_workflow_does_not_call_diagnostics(self):
        fixture = self.fixture()
        with patch.object(
            diagnostics, "summarize_diagnostics", side_effect=AssertionError
        ):
            outcome = runner.collect_job(
                fixture.run, fixture.job, fixture.meta, fixture.case["id"]
            )
        self.assertEqual(outcome["status"], "EVALUATED", outcome)
        self.assertNotIn("diagnostics_file", outcome)
        self.assertNotIn("diagnostics.json", outcome["evidence_sha256"])
        self.assertFalse((fixture.job / "collected/diagnostics.json").exists())


if __name__ == "__main__":
    unittest.main()
