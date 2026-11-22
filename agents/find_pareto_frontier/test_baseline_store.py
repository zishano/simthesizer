import copy
import json
import shutil
import subprocess
import sys
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import archive
import baseline_store
import evaluate
import launcher
import test_evaluate
import verify


class LocalBaselineTests(unittest.TestCase):
    def setUp(self):
        test_evaluate.EvaluationTests.setUp(self)
        manifest = self.experiment.manifest
        manifest["baseline_mode"] = "local"
        launcher.save(self.experiment.manifest_path, manifest)
        (self.run / "task/AGENTS.md").write_text(
            launcher.render_task(self.run / "source", manifest)
        )
        self.experiment = evaluate.Experiment(self.experiment.manifest_path)

    @staticmethod
    def process(command, **_kwargs):
        if "--output-jsonl" in command:
            workload = Path(command[command.index("--workload") + 1])
            result = Path(command[command.index("--output-jsonl") + 1])
            rows = [json.loads(line) for line in workload.read_text().splitlines()]
            for row in rows:
                row["sub_requests"][0]["token_latency"] = [0.02] * row["sub_requests"][
                    0
                ]["output_tokens"]
            result.write_text("".join(json.dumps(row) + "\n" for row in rows))
            Path(command[command.index("--output-policy-stats") + 1]).write_text("{}")
        else:
            Path(command[-1]).write_bytes(b"fixture plot")

    def prepare(self):
        with (
            patch.object(evaluate, "in_container", return_value=False),
            patch.object(
                self.experiment,
                "build",
                return_value=(Path("/fake/simthesizer"), {"fixture": True}),
            ),
            patch.object(self.experiment, "capabilities", return_value={}),
            patch.object(evaluate, "run_process", side_effect=self.process) as process,
        ):
            index = self.experiment.prepare_baseline()
        self.assertEqual(process.call_count, 6)
        return index

    def rewrite_index(self, index):
        path = self.run / "artifacts/baseline/index.json"
        launcher.save(path, index)
        launcher.save(
            self.run / "task/baseline.json",
            {
                "schema_version": 1,
                "manifest_sha256": evaluate.identity(self.experiment.manifest),
                "index_sha256": launcher.digest(path),
            },
        )

    def test_sealed_reuse_survives_compiler_banner_and_path_changes_without_execution(
        self,
    ):
        index = self.prepare()
        original_ledger = self.experiment.book.read_bytes()
        with (
            patch.object(sys, "version", sys.version + " alternate compiler banner"),
            patch.object(evaluate, "in_container", return_value=False),
            patch.object(
                evaluate, "run_process", side_effect=AssertionError("must not execute")
            ),
            patch.object(
                evaluate.Experiment,
                "build",
                side_effect=AssertionError("must not build"),
            ),
        ):
            other = evaluate.Experiment(self.experiment.manifest_path)
            self.assertEqual(other.check_baseline(), index)
            self.assertEqual(other.prepare_baseline(), index)
            reference = baseline_store.reference(other.manifest)
            for case in other.cases():
                self.assertTrue(other.evaluate(reference, case, force=True)["reused"])
        self.assertEqual(self.experiment.book.read_bytes(), original_ledger)
        destination = self.run / "relocated"
        destination.mkdir()
        for name in ("source", "inputs", "baseline-policy", "task", "artifacts"):
            shutil.copytree(self.run / name, destination / name)
        relocated = evaluate.Experiment(destination / "task/manifest.json")
        self.assertEqual(relocated.check_baseline(), index)

    def test_missing_seal_refuses_candidate_and_reference_before_build(self):
        with patch.object(
            self.experiment, "build", side_effect=AssertionError("must not build")
        ):
            for baseline in (False, True):
                candidate = self.experiment.freeze_candidate(baseline=baseline)
                with self.assertRaisesRegex(ValueError, "baseline evidence missing"):
                    self.experiment.evaluate(candidate, self.experiment.cases()[0])
        with (
            patch.object(evaluate, "in_container", return_value=True),
            self.assertRaisesRegex(ValueError, "on the host"),
        ):
            self.experiment.prepare_baseline()

    def test_all_eight_artifacts_are_required_and_tamper_is_rejected(self):
        index = self.prepare()
        row = index["records"][0]
        directory = self.experiment.artifacts / row["directory"]
        for name in baseline_store.EVIDENCE:
            with self.subTest(evidence=name):
                path = directory / name
                original = path.read_bytes()
                path.write_bytes(original + b"tampered")
                with self.assertRaisesRegex(ValueError, "baseline evidence changed"):
                    baseline_store.validate(self.run, self.experiment.manifest)
                path.write_bytes(original)
        missing = directory / "metrics.png"
        missing.unlink()
        with self.assertRaisesRegex(ValueError, "baseline evidence missing"):
            baseline_store.validate(self.run, self.experiment.manifest)

    def test_manifest_input_slo_version_and_package_changes_are_rejected(self):
        self.prepare()
        changed = copy.deepcopy(self.experiment.manifest)
        changed["experiment"]["slo"]["ttft_limit_s"] += 1
        with self.assertRaisesRegex(ValueError, "anchor or manifest"):
            baseline_store.validate(self.run, changed)
        for name in ("python", "machine", "implementation", "system", "packages"):
            with self.subTest(environment=name):
                old = self.experiment.python_identity[name]
                self.experiment.python_identity[name] = "changed"
                with self.assertRaisesRegex(ValueError, "result environment differs"):
                    self.experiment.check_baseline()
                self.experiment.python_identity[name] = old
        (self.run / "inputs/workload").write_text("changed")
        with self.assertRaisesRegex(ValueError, "frozen workload changed"):
            baseline_store.validate(self.run, self.experiment.manifest)

    def test_missing_case_partial_receipt_and_path_escape_are_rejected(self):
        index = self.prepare()
        changed = copy.deepcopy(index)
        changed["records"].pop()
        self.rewrite_index(changed)
        with self.assertRaisesRegex(ValueError, "case matrix"):
            baseline_store.validate(self.run, self.experiment.manifest)
        for status in ("INCOMPLETE", "INVALID"):
            changed = copy.deepcopy(index)
            changed["records"][0]["status"] = status
            self.rewrite_index(changed)
            with self.assertRaisesRegex(ValueError, "valid reference"):
                baseline_store.validate(self.run, self.experiment.manifest)
        changed = copy.deepcopy(index)
        changed["records"][0]["directory"] = "baseline/cases/../../trials"
        self.rewrite_index(changed)
        with self.assertRaisesRegex(ValueError, "escapes"):
            baseline_store.validate(self.run, self.experiment.manifest)

    def test_infeasible_complete_baseline_is_accepted_and_accounted_once(self):
        manifest = self.experiment.manifest
        manifest["experiment"]["slo"]["ttft_limit_s"] = 0.001
        launcher.save(self.experiment.manifest_path, manifest)
        (self.run / "task/AGENTS.md").write_text(
            launcher.render_task(self.run / "source", manifest)
        )
        self.experiment = evaluate.Experiment(self.experiment.manifest_path)
        index = self.prepare()
        self.assertTrue(
            all(row["status"] == "VALID_INFEASIBLE" for row in index["records"])
        )
        self.assertEqual(self.experiment.accounting(), index["accounting"])
        self.assertEqual(len(self.experiment.records()), 2)
        self.assertEqual(archive.pareto(self.experiment.records()), {})
        candidate = self.experiment.freeze_candidate()
        with patch.object(
            self.experiment, "build", side_effect=RuntimeError("fixture failure")
        ):
            row = self.experiment.evaluate(candidate, self.experiment.cases()[0])
        self.assertEqual(row["status"], "INVALID")
        self.assertEqual(self.experiment.accounting()["attempts"], 3)
        # Removing mutable local reference rows cannot recover their sealed budget.
        events = [
            row
            for row in evaluate.ledger(self.experiment.book)
            if row["kind"] != "reference"
        ]
        self.experiment.book.write_text(
            "".join(json.dumps(row) + "\n" for row in events)
        )
        self.assertEqual(self.experiment.accounting()["attempts"], 3)
        with self.assertRaisesRegex(ValueError, "budget exhausted"):
            self.experiment.evaluate(candidate, self.experiment.cases()[1])

    def test_failed_baseline_attempt_is_sealed_and_not_repeated(self):
        self.experiment.preparing_baseline = True
        reference = self.experiment.freeze_candidate(baseline=True)
        with patch.object(
            self.experiment, "build", side_effect=RuntimeError("fixture failure")
        ):
            failed = self.experiment.evaluate(reference, self.experiment.cases()[0])
        self.experiment.preparing_baseline = False
        self.assertEqual(failed["status"], "INVALID")
        index = self.prepare()
        self.assertEqual(index["accounting"]["attempts"], 3)
        self.assertEqual(
            index["accounting"]["charges"][0]["finished"]["status"], "INVALID"
        )
        self.assertEqual(self.experiment.accounting(), index["accounting"])

    def test_interrupt_stops_before_next_case_and_retains_receipt(self):
        with (
            patch.object(evaluate, "in_container", return_value=False),
            patch.object(
                self.experiment,
                "build",
                return_value=(Path("/fake/simthesizer"), {"fixture": True}),
            ),
            patch.object(self.experiment, "capabilities", return_value={}),
            patch.object(
                evaluate, "run_process", side_effect=KeyboardInterrupt
            ) as process,
            self.assertRaisesRegex(ValueError, "incomplete or invalid"),
        ):
            self.experiment.prepare_baseline()
        self.assertEqual(process.call_count, 1)
        events = evaluate.ledger(self.experiment.book)
        self.assertEqual(len(events), 2)
        self.assertEqual(events[-1]["status"], "INCOMPLETE")
        self.assertFalse((self.run / "task/baseline.json").exists())

    def test_candidate_shares_group_and_external_replay_reads_sealed_reference(self):
        index = self.prepare()
        candidate = self.experiment.freeze_candidate()
        with (
            patch.object(
                evaluate.Experiment,
                "build",
                return_value=(Path("/fake/simthesizer"), {"fixture": True}),
            ),
            patch.object(evaluate.Experiment, "capabilities", return_value={}),
            patch.object(evaluate, "run_process", side_effect=self.process),
        ):
            candidate_row = self.experiment.evaluate(
                candidate, self.experiment.cases()[0]
            )
            _, replay = verify.replay_candidate(
                self.experiment, index["reference"]["id"], ()
            )
        self.assertEqual(candidate_row["group_id"], index["records"][0]["group_id"])
        self.assertEqual(replay["status"], "VERIFIED", replay)
        rows = [row for row in self.experiment.records() if row["event"] == "finished"]
        self.assertEqual(len(rows), 3)
        group = archive.pareto(rows)[candidate_row["group_id"]]
        self.assertEqual(len(group), 1)
        self.assertEqual(len(group[0]["equivalent_candidates"]), 1)

    def test_canonical_result_identity_but_separate_container_build_cache(self):
        candidate = self.experiment.freeze_candidate()
        case = self.experiment.cases()[0]
        trial = self.experiment.trial_identity(
            candidate, case, self.experiment.derived(case)
        )
        targets = []

        def compile_fixture(_command, *, env, **_kwargs):
            targets.append(env["CARGO_TARGET_DIR"])
            binary = Path(env["CARGO_TARGET_DIR"]) / "release/simthesizer"
            binary.parent.mkdir(parents=True, exist_ok=True)
            binary.write_bytes(b"native fixture")

        with (
            patch.object(
                subprocess,
                "check_output",
                side_effect=lambda command, **_: command[0] + " fixture",
            ),
            patch.object(evaluate, "run_process", side_effect=compile_fixture),
        ):
            self.experiment.build_identity["domain"] = "host"
            host, _ = self.experiment.build(candidate, time.monotonic() + 10)
            self.experiment.build_identity["domain"] = "container"
            container, _ = self.experiment.build(candidate, time.monotonic() + 10)
        self.assertNotEqual(host, container)
        self.assertEqual(len(set(targets)), 2)
        self.assertEqual(
            trial,
            self.experiment.trial_identity(
                candidate, case, self.experiment.derived(case)
            ),
        )


if __name__ == "__main__":
    unittest.main()
