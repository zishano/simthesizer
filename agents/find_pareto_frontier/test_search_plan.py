import copy
import io
import json
import shutil
import unittest
from pathlib import Path
from unittest.mock import patch

import baseline_store
import evaluate
import launcher
import search_plan
import search_space
import test_evaluate as fixtures


class SearchPlanTests(unittest.TestCase):
    def setUp(self):
        fixtures.EvaluationTests.setUp(self)
        manifest = evaluate.read_json(self.experiment.manifest_path)
        manifest["experiment"]["search"].update(max_trials=100, max_wall_seconds=600)
        manifest["experiment"]["workloads"].append(
            {
                "id": "validation",
                "path": "/workspace/inputs/workload",
                "split": "validation",
            }
        )
        for path in (
            Path(search_plan.__file__),
            Path(search_space.__file__),
            search_space.LIBRARY,
        ):
            relative = path.relative_to(evaluate.ROOT)
            destination = self.run / "source" / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
        manifest["source_sha256"] = launcher.inventory(
            self.run / "source", mutable=True
        )
        launcher.save(self.experiment.manifest_path, manifest)
        (self.run / "task/AGENTS.md").write_text(
            launcher.render_task(self.run / "source", manifest)
        )
        self.experiment = evaluate.Experiment(self.experiment.manifest_path)
        self.catalog_path = self.run / "artifacts/catalog/catalog.json"
        self.catalog = search_space.write_catalog(
            self.catalog_path.parent,
            search_space.expand({"aged_short_prefill": True, "decode_fair": True}),
        )

    def entry(self, *families):
        return next(
            row
            for row in self.catalog["entries"]
            if set(row["families"]) == set(families)
        )

    def install(self, entry):
        source = self.run / "source" / evaluate.POLICY
        shutil.rmtree(source)
        shutil.copytree(self.catalog_path.parent / entry["path"], source)
        candidate = self.experiment.freeze_candidate()
        self.assertEqual(candidate["id"], entry["evaluator_candidate_id"])
        return candidate

    def complete(self, entry, cases=None, latency=0.02):
        candidate = (
            self.experiment.freeze_candidate(baseline=True)
            if entry is None
            else self.install(entry)
        )

        def process(command, **_kwargs):
            if "--output-jsonl" in command:
                workload = Path(command[command.index("--workload") + 1])
                result = Path(command[command.index("--output-jsonl") + 1])
                rows = [json.loads(line) for line in workload.read_text().splitlines()]
                for row in rows:
                    row["sub_requests"][0]["token_latency"] = [latency] * row[
                        "sub_requests"
                    ][0]["output_tokens"]
                result.write_text("".join(json.dumps(row) + "\n" for row in rows))
                Path(command[command.index("--output-policy-stats") + 1]).write_text(
                    "{}"
                )
            else:
                Path(command[-1]).write_bytes(b"fixture plot")

        with (
            patch.object(
                self.experiment,
                "build",
                return_value=(
                    Path("/fake/simthesizer"),
                    {
                        "candidate_id": candidate["id"],
                        "source_sha256": self.experiment.manifest["source_sha256"],
                    },
                ),
            ),
            patch.object(self.experiment, "capabilities", return_value={}),
            patch.object(evaluate, "run_process", side_effect=process),
        ):
            receipts = [
                self.experiment.evaluate(candidate, case)
                for case in (cases or self.experiment.cases())
            ]
        for receipt in receipts:
            self.assertEqual(receipt["status"], "VALID_FEASIBLE", receipt)
        return receipts

    def plan(self):
        return search_plan.derive_plan(self.experiment, self.catalog_path)

    def rewrite(self, records):
        self.experiment.book.write_text(
            "".join(json.dumps(row) + "\n" for row in records)
        )

    def test_empty_plan_ignores_authored_coverage_and_screens_one_family(self):
        launcher.save(
            self.run / "artifacts/search-coverage.json",
            {"complete": True, "verification": "VERIFIED"},
        )
        plan = self.plan()
        self.assertEqual(plan["next"]["phase"], "family_screen")
        self.assertEqual(plan["verification"], "provisional")
        self.assertTrue(plan["validation_available"])
        self.assertTrue(all(not row["full_matrix"] for row in plan["entries"]))
        self.assertTrue(all(len(row["missing_train"]) == 2 for row in plan["entries"]))
        self.assertEqual(plan["remaining_budget"]["trials"], 100)
        self.assertFalse(self.experiment.book.exists())

    def test_complete_training_then_validation_then_matched_ablation(self):
        cases = self.experiment.cases()
        prefill, decode = self.entry("aged_short_prefill"), self.entry("decode_fair")
        combined = self.entry("aged_short_prefill", "decode_fair")
        self.complete(prefill, cases[:1], latency=0.03)
        self.complete(decode, cases[:1], latency=0.03)
        self.complete(combined, cases[:1])
        self.assertEqual(self.plan()["next"]["phase"], "complete_training")
        self.complete(combined, cases[:2])
        plan = self.plan()
        self.assertEqual(plan["next"]["phase"], "validation")
        self.assertEqual(plan["next"]["case_id"], cases[2]["id"])
        self.complete(combined)
        plan = self.plan()
        state = next(row for row in plan["entries"] if row["id"] == combined["id"])
        self.assertTrue(state["full_matrix"])
        self.assertEqual(plan["next"]["phase"], "ablation")
        self.assertTrue(
            all(row["matched_cases"] == [cases[0]["id"]] for row in state["ablations"])
        )
        self.assertTrue(
            all(len(row["missing_cases"]) == 3 for row in state["ablations"])
        )
        self.assertEqual(plan["rejected_receipts"], [])

    def test_invalid_and_open_attempts_never_complete_cases(self):
        entry = self.entry("aged_short_prefill")
        candidate = self.install(entry)
        with patch.object(
            self.experiment, "build", side_effect=RuntimeError("fixture failure")
        ):
            self.experiment.evaluate(candidate, self.experiment.cases()[0])
        plan = self.plan()
        state = next(row for row in plan["entries"] if row["id"] == entry["id"])
        self.assertEqual(state["attempted"], 1)
        self.assertEqual(state["failed_cases"][0]["status"], "INVALID")
        self.assertEqual(state["retryable_cases"], [self.experiment.cases()[0]["id"]])
        self.assertEqual(state["valid_completed_cases"], [])
        records = evaluate.ledger(self.experiment.book)
        self.rewrite(records[:1])
        self.assertEqual(self.plan()["open_attempts"], [1])
        self.assertIsNone(self.plan()["next"])

    def test_failed_cases_do_not_starve_unattempted_candidates(self):
        first = self.catalog["entries"][0]
        for entry in self.catalog["entries"]:
            candidate = self.install(entry)
            with patch.object(
                self.experiment, "build", side_effect=RuntimeError("fixture failure")
            ):
                cases = self.experiment.cases()[: 2 if entry == first else 1]
                for case in cases:
                    self.experiment.evaluate(candidate, case)
        plan = self.plan()
        self.assertIsNotNone(plan["next"])
        self.assertNotEqual(plan["next"]["recipe_id"], first["id"])
        self.assertEqual(plan["next"]["case_id"], self.experiment.cases()[1]["id"])
        state = next(
            row for row in plan["entries"] if row["id"] == plan["next"]["recipe_id"]
        )
        self.assertNotIn(plan["next"]["case_id"], state["attempted_case_ids"])

    def test_changed_file_wrong_case_and_stale_snapshot_are_rejected(self):
        entry = self.entry("aged_short_prefill")
        receipt = self.complete(entry, self.experiment.cases()[:1])[0]
        records = evaluate.ledger(self.experiment.book)
        result = self.run / "artifacts" / receipt["directory"] / "result.jsonl"
        original = result.read_bytes()
        result.write_bytes(original + b"\n")
        self.assertIn("evidence changed", self.plan()["rejected_receipts"][0]["reason"])
        result.write_bytes(original)
        changed = copy.deepcopy(records)
        changed[-1]["case"]["arrival_multiplier"] = 999
        self.rewrite(changed)
        self.assertIn("case differs", self.plan()["rejected_receipts"][0]["reason"])
        self.rewrite(records)
        policy = (
            self.run
            / "artifacts/candidates"
            / receipt["candidate_id"]
            / "policy/mod.rs"
        )
        policy.write_text(policy.read_text() + "// stale\n")
        plan = self.plan()
        self.assertTrue(plan["rejected_receipts"])
        self.assertTrue(
            all(not row["valid_completed_cases"] for row in plan["entries"])
        )

    def test_incomplete_result_rehashed_receipt_and_forged_score_are_rejected(self):
        entry = self.entry("aged_short_prefill")
        receipt = self.complete(entry, self.experiment.cases()[:1])[0]
        directory = self.run / "artifacts" / receipt["directory"]
        records = evaluate.ledger(self.experiment.book)
        result = directory / "result.jsonl"
        original = result.read_text()
        result.write_text(original.splitlines()[0] + "\n")
        changed = copy.deepcopy(records)
        changed[-1]["evidence_sha256"]["result.jsonl"] = launcher.digest(result)
        launcher.save(directory / "receipt.json", changed[-1])
        self.rewrite(changed)
        plan = self.plan()
        self.assertTrue(plan["rejected_receipts"])
        self.assertTrue(
            all(not row["valid_completed_cases"] for row in plan["entries"])
        )
        result.write_text(original)
        changed = copy.deepcopy(records)
        changed[-1]["objectives"]["goodput_per_gpu"] = 1e9
        metrics = evaluate.read_json(directory / "metrics.json")
        metrics["slo"]["goodput_per_gpu"] = 1e9
        launcher.save(directory / "metrics.json", metrics)
        changed[-1]["evidence_sha256"]["metrics.json"] = launcher.digest(
            directory / "metrics.json"
        )
        launcher.save(directory / "receipt.json", changed[-1])
        self.rewrite(changed)
        self.assertIn("metrics differ", self.plan()["rejected_receipts"][0]["reason"])

    def test_missing_hash_and_symlink_evidence_are_rejected(self):
        receipt = self.complete(self.entry("decode_fair"), self.experiment.cases()[:1])[
            0
        ]
        directory = self.run / "artifacts" / receipt["directory"]
        records = evaluate.ledger(self.experiment.book)
        changed = copy.deepcopy(records)
        del changed[-1]["evidence_sha256"]["metrics.png"]
        self.rewrite(changed)
        launcher.save(directory / "receipt.json", changed[-1])
        self.assertIn("all evidence", self.plan()["rejected_receipts"][0]["reason"])
        self.rewrite(records)
        launcher.save(directory / "receipt.json", receipt)
        evidence = directory / "metrics.png"
        target = self.run / "same-bytes.png"
        target.write_bytes(evidence.read_bytes())
        evidence.unlink()
        evidence.symlink_to(target)
        self.assertIn("symlinks", self.plan()["rejected_receipts"][0]["reason"])

    def test_wrong_trial_and_build_cannot_claim_matching_coverage(self):
        receipt = self.complete(self.entry("decode_fair"), self.experiment.cases()[:1])[
            0
        ]
        directory = self.run / "artifacts" / receipt["directory"]
        records = evaluate.ledger(self.experiment.book)
        changed = copy.deepcopy(records)
        changed[-1]["trial_id"] = "0" * 64
        self.rewrite(changed)
        self.assertIn("trial identity", self.plan()["rejected_receipts"][0]["reason"])
        self.rewrite(records)
        build = evaluate.read_json(directory / "build.json")
        build["source_sha256"] = {}
        launcher.save(directory / "build.json", build)
        changed = copy.deepcopy(records)
        changed[-1]["evidence_sha256"]["build.json"] = launcher.digest(
            directory / "build.json"
        )
        launcher.save(directory / "receipt.json", changed[-1])
        self.rewrite(changed)
        self.assertIn("build provenance", self.plan()["rejected_receipts"][0]["reason"])

    def test_configuration_allows_worker_relocation_but_no_policy_drift(self):
        candidate = self.install(self.entry("decode_fair"))
        config = self.experiment.configuration(candidate)
        for pool in ("prefill_instances", "decode_instances"):
            for child in config["system"][pool]:
                child["model_runner"]["command"] = [
                    "/opt/venv/bin/python",
                    "/workspace/simthesizer/scripts/rooflang_worker.py",
                ]
        path = self.run / "config.toml"
        path.write_text(evaluate.toml_document(config))
        search_plan.compatible_configuration(self.experiment, candidate, path)
        config["system"]["policy"]["seed"] += 1
        path.write_text(evaluate.toml_document(config))
        with self.assertRaisesRegex(ValueError, "configuration differs"):
            search_plan.compatible_configuration(self.experiment, candidate, path)
        config["system"]["policy"]["seed"] -= 1
        for command in (
            ["/opt/venv/bin/python", "/tmp/evil/scripts/rooflang_worker.py"],
            ["/tmp/evil/python", "/workspace/simthesizer/scripts/rooflang_worker.py"],
        ):
            config["system"]["prefill_instances"][0]["model_runner"]["command"] = (
                command
            )
            path.write_text(evaluate.toml_document(config))
            with self.assertRaisesRegex(ValueError, "worker command differs"):
                search_plan.compatible_configuration(self.experiment, candidate, path)

    def test_malformed_artifact_shapes_are_rejected_without_a_traceback(self):
        receipt = self.complete(self.entry("decode_fair"), self.experiment.cases()[:1])[
            0
        ]
        directory = self.run / "artifacts" / receipt["directory"]
        records = evaluate.ledger(self.experiment.book)
        for name in ("build.json", "metrics.json"):
            path = directory / name
            original = path.read_bytes()
            for value in (None, []):
                with self.subTest(name=name, value=value):
                    launcher.save(path, value)
                    changed = copy.deepcopy(records)
                    changed[-1]["evidence_sha256"][name] = launcher.digest(path)
                    launcher.save(directory / "receipt.json", changed[-1])
                    self.rewrite(changed)
                    plan = self.plan()
                    self.assertTrue(plan["rejected_receipts"])
                    self.assertTrue(plan["budget_unavailable"])
                    self.assertIsNone(plan["next"])
            path.write_bytes(original)

    def test_invalid_accounting_or_nonobject_ledger_withholds_budget(self):
        receipt = self.complete(self.entry("decode_fair"), self.experiment.cases()[:1])[
            0
        ]
        records = evaluate.ledger(self.experiment.book)
        for field in ("started_at", "budget_wall_seconds", "wall_seconds"):
            for value in (-1, float("nan"), None):
                with self.subTest(field=field, value=value):
                    changed = copy.deepcopy(records)
                    changed[1 if field == "wall_seconds" else 0][field] = value
                    self.rewrite(changed)
                    plan = self.plan()
                    self.assertTrue(plan["budget_unavailable"])
                    self.assertIsNone(plan["remaining_budget"])
                    self.assertIsNone(plan["next"])
        self.rewrite([None])
        plan = self.plan()
        self.assertIn("must be an object", plan["rejected_receipts"][0]["reason"])
        self.assertIsNone(plan["remaining_budget"])
        self.assertEqual(receipt["status"], "VALID_FEASIBLE")

    def test_sealed_local_reference_is_checked_without_trial_receipt(self):
        manifest = self.experiment.manifest
        manifest["baseline_mode"] = "local"
        launcher.save(self.experiment.manifest_path, manifest)
        (self.run / "task/AGENTS.md").write_text(
            launcher.render_task(self.run / "source", manifest)
        )
        self.experiment = evaluate.Experiment(self.experiment.manifest_path)
        self.experiment.preparing_baseline = True
        receipts = self.complete(None)
        accounting = self.experiment.accounting(baseline_only=True)
        baseline_store.seal(
            self.run, manifest, self.experiment.python_identity, receipts, accounting
        )
        self.experiment.preparing_baseline = False
        plan = self.plan()
        self.assertEqual(plan["rejected_receipts"], [])
        self.assertEqual(len(plan["uncatalogued_candidates"]), 1)
        self.assertTrue(plan["uncatalogued_candidates"][0]["full_matrix"])
        self.assertEqual(plan["remaining_budget"]["trials"], 96)

    def test_static_controls_report_only_identical_completed_cases(self):
        self.catalog_path = self.run / "artifacts/pressure/catalog.json"
        self.catalog = search_space.write_catalog(
            self.catalog_path.parent, search_space.expand({"pressure_chunks": True})
        )
        dynamic = next(
            row for row in self.catalog["entries"] if len(row["static_controls"]) == 2
        )
        control = next(
            row
            for row in self.catalog["entries"]
            if row["id"] == dynamic["static_controls"][0]
        )
        self.complete(dynamic, self.experiment.cases()[:2])
        self.complete(control, self.experiment.cases()[:1])
        state = next(
            row for row in self.plan()["entries"] if row["id"] == dynamic["id"]
        )
        matched = {row["id"]: row["matched_cases"] for row in state["static_controls"]}
        self.assertEqual(matched[control["id"]], [self.experiment.cases()[0]["id"]])
        self.assertEqual(matched[dynamic["static_controls"][1]], [])

    def test_frozen_matrix_without_validation_and_budget_limit_are_explicit(self):
        self.experiment.config["workloads"] = self.experiment.config["workloads"][:1]
        plan = self.plan()
        self.assertFalse(plan["validation_available"])
        self.assertTrue(all(not row["missing_validation"] for row in plan["entries"]))
        self.experiment.config["search"]["max_trials"] = 0
        plan = self.plan()
        self.assertTrue(plan["budget_exhausted"])
        self.assertIsNone(plan["next"])

    def test_routing_single_instance_is_not_applicable(self):
        catalog = self.run / "artifacts/routing/catalog.json"
        search_space.write_catalog(catalog.parent, [{"routing_load": True}])
        plan = search_plan.derive_plan(self.experiment, catalog)
        self.assertFalse(plan["families"]["routing_load"]["applicable"])
        self.assertIsNone(plan["next"])
        self.experiment.scenario["system"]["decode_instances"].append(
            copy.deepcopy(self.experiment.scenario["system"]["decode_instances"][0])
        )
        plan = search_plan.derive_plan(self.experiment, catalog)
        self.assertTrue(plan["families"]["routing_load"]["applicable"])
        self.assertEqual(plan["next"]["phase"], "family_screen")

    def test_cli_only_writes_search_report_and_checks_frozen_helpers(self):
        output = self.run / "artifacts/search-plan.json"
        argv = [
            "--manifest",
            str(self.experiment.manifest_path),
            "--catalog",
            str(self.catalog_path),
            "--output",
            str(output),
        ]
        with patch("sys.stdout", io.StringIO()):
            self.assertEqual(search_plan.main(argv), 0)
        self.assertTrue(output.is_file())
        self.assertGreaterEqual(evaluate.read_json(output)["planning_wall_seconds"], 0)
        self.assertFalse(self.experiment.book.exists())
        with self.assertRaises(SystemExit), patch("sys.stderr", io.StringIO()):
            search_plan.main(argv[:-1] + [str(self.experiment.book)])
        with (
            patch.object(search_space, "LIBRARY", self.run / "baseline-policy/mod.rs"),
            self.assertRaises(SystemExit),
            patch("sys.stderr", io.StringIO()),
        ):
            search_plan.main(argv)


if __name__ == "__main__":
    unittest.main()
