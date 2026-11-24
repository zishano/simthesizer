"""Mechanism evidence and comparison contracts; no simulator execution."""

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(
    0, str(Path(__file__).resolve().parents[2] / "agents/find_pareto_frontier")
)
import baseline_store
import candidate
import contract

CASE = "real-m1.0-s0"
OTHER_CASE = "real-m2.0-s0"
TRIAL = "b" * 32
ENGINE = "crates/simthesizer-core/src/engine.rs"
TEST = "tests/test_new_mechanism.py"


class ContractTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="qwen-contract-test-")
        self.addCleanup(temporary.cleanup)
        self.run = Path(temporary.name)
        self.manifest = {"profile": "qwen_moe", "experiment": {"seed": 0}}
        self.meta = {
            "kind": "mechanism",
            "source_sha256": "a" * 64,
            "policy_sha256": "c" * 64,
            "reference_policy_changed": False,
            "impact": candidate.impact([ENGINE]),
            "files": {
                ENGINE: "e" * 64,
                TEST: "f" * 64,
                "crates/simthesizer-core/tests/new_resource.rs": "1" * 64,
                "crates/simthesizer-cli/tests/new_command.rs": "2" * 64,
            },
        }
        self.metrics = {"service_window": {"finish_s": 4.0}}
        self.objectives = {"ttft_p99": 1.0, "tpot_p99": 0.02, "goodput_per_gpu": 10.0}
        baseline = self.record("baseline/cases/real", baseline=True)
        trial = self.record("evaluations/" + TRIAL, baseline=False)
        self.baseline = {"records": [baseline]}
        self.history = [trial]
        self.plan = {
            "schema_version": 1,
            "mode": "candidate",
            "summary": "Coordinate dependency scheduling across instances.",
            "state_ownership": "The engine owns time and the shared resource ledger.",
            "preserved_semantics": "Preserve all tokens, model work and fixed resources.",
            "expected_effect": "Reduce stalls without changing work or pricing.",
            "alternatives": ["Keep the original execution order."],
            "stop_conditions": ["Stop if any conservation test fails."],
            "evidence": [
                {"kind": "baseline", "id": CASE, "observation": "A queue is waiting."}
            ],
            "control": {"kind": "baseline", "id": CASE},
            "checks": [
                {"id": name, "argument": "Preserve the invariant.", "tests": [TEST]}
                for name in self.meta["impact"]["required_checks"]
            ],
        }

    def record(self, directory, *, baseline):
        root = self.run / "artifacts" / directory
        root.mkdir(parents=True)
        (root / "result.jsonl").write_text('{"request_id": 7, "times": [1, 2]}\n')
        (root / "metrics.json").write_text(json.dumps(self.metrics))
        value = {
            "event": "finished",
            "candidate_id": "baseline-policy" if baseline else "d" * 64,
            "status": "VALID_FEASIBLE" if baseline else "EVALUATED",
            "case": {"id": CASE},
            "case_id": CASE,
            "id": "baseline-trial" if baseline else TRIAL,
            "directory": directory,
            "objectives": dict(self.objectives),
            "evidence_sha256": {
                name: baseline_store.digest(root / name)
                for name in ("result.jsonl", "metrics.json")
            },
        }
        if not baseline:
            value.update(
                policy_kind="candidate",
                policy_sha256=self.meta["policy_sha256"],
                compatibility="MATCHED",
            )
        return value

    def seal(self, plan=None, *, meta=None, policy_kind="reference"):
        with patch.object(baseline_store, "validate", return_value=self.baseline):
            return contract.seal(
                self.run,
                self.manifest,
                self.meta if meta is None else meta,
                self.plan if plan is None else plan,
                CASE,
                policy_kind,
                self.history,
            )

    def measured(self):
        return {
            "evidence_sha256": dict(self.baseline["records"][0]["evidence_sha256"]),
            "metrics": copy.deepcopy(self.metrics),
            "objectives": dict(self.objectives),
        }

    def test_valid_plan_is_copied_and_preserves_all_host_obligations(self):
        result = contract.validate_plan(self.plan, self.meta)
        self.assertEqual(result, self.plan)
        result["checks"][0]["tests"].clear()
        self.assertEqual(self.plan["checks"][0]["tests"], [TEST])
        self.assertEqual(
            {row["id"] for row in result["checks"]},
            set(self.meta["impact"]["required_checks"]),
        )

    def test_plan_rejects_missing_extra_duplicate_or_unexecutable_checks(self):
        variants = []
        missing = copy.deepcopy(self.plan)
        missing["checks"].pop()
        variants.append(missing)
        duplicate = copy.deepcopy(self.plan)
        duplicate["checks"].append(duplicate["checks"][0])
        variants.append(duplicate)
        extra = copy.deepcopy(self.plan)
        extra["checks"].append(
            {"id": "author_selected_gate", "argument": "skip", "tests": [TEST]}
        )
        variants.append(extra)
        unavailable = copy.deepcopy(self.plan)
        unavailable["checks"][0]["tests"] = ["tests/test_missing.py"]
        variants.append(unavailable)
        no_test = copy.deepcopy(self.plan)
        no_test["checks"][0]["tests"] = []
        variants.append(no_test)
        for value in variants:
            with self.subTest(plan=value), self.assertRaises(ValueError):
                contract.validate_plan(value, self.meta)

    def test_schema_text_evidence_and_plan_size_are_bounded(self):
        for key, value in (
            ("schema_version", True),
            ("schema_version", 2),
            ("mode", "verified"),
            ("state_ownership", " "),
            ("summary", "x" * 4097),
            ("alternatives", []),
            ("stop_conditions", [""]),
            ("evidence", []),
            ("checks", []),
            ("allow_more_gpus", True),
        ):
            with self.subTest(field=key, value=value):
                plan = {**self.plan, key: value}
                with self.assertRaises(ValueError):
                    contract.validate_plan(plan, self.meta)
        plan = copy.deepcopy(self.plan)
        plan["alternatives"] = ["x" * 4096] * 16
        with self.assertRaisesRegex(ValueError, "byte limit"):
            contract.validate_plan(plan, self.meta)

    def test_test_entrypoints_support_rust_packages_and_reject_traversal(self):
        for path in (
            TEST,
            "crates/simthesizer-core/tests/new_resource.rs",
            "crates/simthesizer-cli/tests/new_command.rs",
        ):
            self.assertEqual(contract.test_path(path, self.meta["files"]), path)
        for path in (
            None,
            "/tmp/test_escape.py",
            "tests/../test_escape.py",
            "tests//test_new_mechanism.py",
            "crates/simthesizer-core/tests/helpers/mod.rs",
            ENGINE,
        ):
            with self.subTest(path=path), self.assertRaises(ValueError):
                contract.test_path(path, self.meta["files"])

    def test_baseline_and_trial_evidence_bind_case_inputs_and_hashes(self):
        for reference in (
            self.plan["evidence"][0],
            {"kind": "trial", "id": TRIAL, "observation": "A published comparison."},
        ):
            result = contract.resolve_evidence(
                self.run, self.manifest, reference, CASE, self.history, self.baseline
            )
            self.assertEqual(result["case_id"], CASE)
            self.assertEqual(
                result["manifest_sha256"], baseline_store.identity(self.manifest)
            )
            self.assertEqual(result["service_finish_s"], 4.0)
            self.assertEqual(
                result["result_sha256"],
                self.baseline["records"][0]["evidence_sha256"]["result.jsonl"],
            )

    def test_changed_baseline_or_trial_evidence_is_rejected(self):
        for kind, identifier, directory in (
            ("baseline", CASE, "baseline/cases/real"),
            ("trial", TRIAL, "evaluations/" + TRIAL),
        ):
            for name in ("metrics.json", "result.jsonl"):
                with self.subTest(kind=kind, file=name):
                    path = self.run / "artifacts" / directory / name
                    original = path.read_bytes()
                    path.write_bytes(original + b" ")
                    with self.assertRaisesRegex(ValueError, "evidence changed"):
                        contract.resolve_evidence(
                            self.run,
                            self.manifest,
                            {"kind": kind, "id": identifier, "observation": "changed"},
                            CASE,
                            self.history,
                            self.baseline,
                        )
                    path.write_bytes(original)

    def test_evidence_rejects_cross_case_invalid_duplicate_and_symlink_records(self):
        reference = {"kind": "trial", "id": TRIAL, "observation": "candidate"}
        for history in (
            [],
            [{**self.history[0], "status": "INVALID"}],
            self.history * 2,
        ):
            with (
                self.subTest(history=history),
                self.assertRaisesRegex(ValueError, "unique completed"),
            ):
                contract.resolve_evidence(
                    self.run, self.manifest, reference, CASE, history, self.baseline
                )
        for case_reference in (self.plan["evidence"][0], reference):
            with self.assertRaisesRegex(ValueError, "frozen case"):
                contract.resolve_evidence(
                    self.run,
                    self.manifest,
                    case_reference,
                    OTHER_CASE,
                    self.history,
                    self.baseline,
                )
        for reference in (
            {"kind": "trial", "id": "../escape", "observation": "escape"},
            {"kind": "file", "id": "notes/result.json", "observation": "escape"},
            {
                "kind": "baseline",
                "id": CASE,
                "observation": " ",
                "path": "notes/result.json",
            },
        ):
            with self.subTest(reference=reference), self.assertRaises(ValueError):
                contract.resolve_evidence(
                    self.run,
                    self.manifest,
                    reference,
                    CASE,
                    self.history,
                    self.baseline,
                )
        target = self.run / "artifacts/baseline/cases/real/result.jsonl"
        target.unlink()
        target.symlink_to(self.run / "artifacts/evaluations" / TRIAL / "result.jsonl")
        with self.assertRaisesRegex(ValueError, "symlink"):
            contract.resolve_evidence(
                self.run,
                self.manifest,
                self.plan["evidence"][0],
                CASE,
                self.history,
                self.baseline,
            )

    def test_pure_policy_needs_no_plan_but_mechanism_does(self):
        for kind in ("policy", "reference"):
            with patch.object(baseline_store, "validate") as validator:
                self.assertIsNone(
                    contract.seal(
                        self.run,
                        self.manifest,
                        {**self.meta, "kind": kind},
                        None,
                        CASE,
                        "candidate",
                        [],
                    )
                )
                validator.assert_not_called()
        with self.assertRaisesRegex(ValueError, "require --plan"):
            contract.seal(
                self.run, self.manifest, self.meta, None, CASE, "candidate", []
            )

    def test_seal_identity_and_matched_policy_do_not_assume_unchanged_reference(self):
        sealed = self.seal()
        body = {name: value for name, value in sealed.items() if name != "id"}
        self.assertEqual(sealed["id"], baseline_store.identity(body))
        self.assertEqual(sealed["source_sha256"], self.meta["source_sha256"])
        self.assertTrue(sealed["matched_policy"])
        self.assertFalse(
            self.seal(meta={**self.meta, "reference_policy_changed": True})[
                "matched_policy"
            ]
        )
        plan = {**self.plan, "control": {"kind": "trial", "id": TRIAL}}
        self.assertTrue(self.seal(plan, policy_kind="candidate")["matched_policy"])
        self.assertFalse(
            self.seal(
                plan,
                policy_kind="candidate",
                meta={**self.meta, "policy_sha256": "9" * 64},
            )["matched_policy"]
        )

    def research_plan(self):
        return {
            **copy.deepcopy(self.plan),
            "research": {
                "name": "Reduce the decode tail",
                "family": "new-cross-module-family",
                "causes": [
                    "Pending requests wait for decode admission.",
                    "Alternatively, decode compute may dominate after admission.",
                ],
                "discriminating_check": "Compare queue time and decode duration at fixed work.",
                "expected_metrics": ["output_tokens_per_gpu", "tokens_per_user"],
                "implementation": "Coordinate routing and admission with resource state.",
                "overhead": "Record added host time; bound policy observations.",
            },
        }

    def test_research_schema_is_optional_for_legacy_and_open_to_new_families(self):
        self.assertEqual(contract.validate_plan(self.plan, self.meta), self.plan)
        plan = self.research_plan()
        result = contract.validate_plan(plan, self.meta, require_research=True)
        self.assertEqual(result, plan)
        result["research"]["causes"].clear()
        self.assertEqual(len(plan["research"]["causes"]), 2)
        self.assertEqual(self.seal(plan)["plan"], plan)

    def test_research_requires_the_causal_chain_and_declared_costs(self):
        plan = self.research_plan()
        with self.assertRaisesRegex(ValueError, "plan.research"):
            contract.validate_plan(self.plan, self.meta, require_research=True)
        for key in plan["research"]:
            missing = copy.deepcopy(plan)
            del missing["research"][key]
            with (
                self.subTest(missing=key),
                self.assertRaisesRegex(ValueError, "research"),
            ):
                contract.validate_plan(missing, self.meta)
        for key, value in (
            ("name", " "),
            ("family", ""),
            ("causes", []),
            ("causes", ["Only one cause"]),
            ("causes", ["Same cause", " Same cause "]),
            ("causes", ["Cause", ""]),
            ("discriminating_check", ""),
            ("expected_metrics", []),
            ("expected_metrics", ["makespan"]),
            ("expected_metrics", ["tokens_per_user"] * 2),
            ("expected_metrics", [{}]),
            ("implementation", ""),
            ("overhead", ""),
            ("evidence_path", "/tmp/untrusted.json"),
        ):
            changed = copy.deepcopy(plan)
            changed["research"][key] = value
            with self.subTest(field=key, value=value), self.assertRaises(ValueError):
                contract.validate_plan(changed, self.meta)

    def test_new_workflow_requires_research_for_policy_mechanism_and_control(self):
        self.manifest["experiment"]["research"] = {"workflow": "bottleneck-v1"}
        for kind in ("policy", "reference", "mechanism"):
            meta = {**self.meta, "kind": kind}
            with (
                self.subTest(kind=kind),
                self.assertRaisesRegex(ValueError, "require --plan"),
            ):
                contract.seal(
                    self.run, self.manifest, meta, None, CASE, "reference", []
                )
            for mode in ("candidate", "control"):
                with (
                    self.subTest(kind=kind, mode=mode),
                    self.assertRaisesRegex(ValueError, "research"),
                ):
                    self.seal({**self.plan, "mode": mode}, meta=meta)
                plan = {**self.research_plan(), "mode": mode}
                self.assertEqual(self.seal(plan, meta=meta)["plan"], plan)

    def test_empty_checks_only_when_the_host_has_no_obligations(self):
        plan = {**self.research_plan(), "checks": [], "mode": "control"}
        meta = {**self.meta, "kind": "reference", "impact": candidate.impact([])}
        self.manifest["experiment"]["research"] = {"workflow": "bottleneck-v1"}
        self.assertEqual(self.seal(plan, meta=meta)["required_checks"], [])
        with self.assertRaises(ValueError):
            self.seal(plan)

    def test_research_evidence_still_requires_preexisting_host_records(self):
        self.manifest["experiment"]["research"] = {"workflow": "bottleneck-v1"}
        plan = self.research_plan()
        plan["evidence"] = [
            {"kind": "trial", "id": TRIAL, "observation": "Completed paired trial."}
        ]
        sealed = self.seal(plan)
        self.assertEqual(
            sealed["evidence"][0]["metrics_sha256"],
            self.history[0]["evidence_sha256"]["metrics.json"],
        )
        for history in ([], [{**self.history[0], "event": "started"}]):
            with (
                self.subTest(history=history),
                patch.object(baseline_store, "validate", return_value=self.baseline),
                self.assertRaisesRegex(ValueError, "unique completed"),
            ):
                contract.seal(
                    self.run, self.manifest, self.meta, plan, CASE, "reference", history
                )
        path = self.run / "artifacts/evaluations" / TRIAL / "metrics.json"
        path.write_text('{"service_window": {"finish_s": 1}}')
        with self.assertRaisesRegex(ValueError, "evidence changed"):
            self.seal(plan)

    def test_control_requires_exact_tokens_and_known_service_tail(self):
        sealed = self.seal({**self.plan, "mode": "control"})
        result = contract.compare(sealed, self.measured())
        self.assertEqual(result["compatibility"], "MATCHED")
        self.assertEqual(
            result["comparison"]["mechanism_off_equivalence"], "NOT_ESTABLISHED"
        )
        self.assertNotIn("control_reproduced", result)
        changed = self.measured()
        changed["evidence_sha256"]["result.jsonl"] = "changed"
        with self.assertRaisesRegex(ValueError, "does not reproduce"):
            contract.compare(sealed, changed)
        for finish in (4.1, None, float("nan"), True):
            changed = self.measured()
            changed["metrics"]["service_window"]["finish_s"] = finish
            with (
                self.subTest(finish=finish),
                self.assertRaisesRegex(ValueError, "service tail"),
            ):
                contract.compare(sealed, changed)

    def test_candidate_comparison_allows_new_timing_without_claiming_equivalence(self):
        plan = {**self.plan, "control": {"kind": "trial", "id": TRIAL}}
        sealed = self.seal(plan, policy_kind="candidate")
        measured = self.measured()
        measured["objectives"]["ttft_p99"] = 0.5
        measured["objectives"]["tpot_p99"] = None
        measured["evidence_sha256"]["result.jsonl"] = "new-schedule"
        compared = contract.compare(sealed, measured)
        self.assertEqual(compared["compatibility"], "NOT_CHECKED")
        self.assertTrue(compared["comparison_reference_reproduced"])
        self.assertEqual(compared["comparison"]["objective_deltas"]["ttft_p99"], -0.5)
        self.assertNotIn("tpot_p99", compared["comparison"]["objective_deltas"])
        self.assertEqual(contract.compare(None, measured), {})

    def test_delivery_requires_all_cases_replay_and_comparison_without_verification_claims(
        self,
    ):
        row = {
            "event": "finished",
            "status": "EVALUATED",
            "case_id": CASE,
            "candidate_id": self.meta["source_sha256"],
            "policy_kind": "candidate",
            "contract": {"plan": {"mode": "candidate"}},
            "comparison_reference_reproduced": True,
            "replay": "MATCHED",
        }
        incomplete = contract.delivery_candidates([row], [CASE, OTHER_CASE])[0]
        self.assertFalse(incomplete["ready_for_review"])
        self.assertEqual(incomplete["pending_cases"], [OTHER_CASE])
        ready = contract.delivery_candidates(
            [row, {**row, "case_id": OTHER_CASE}], [CASE, OTHER_CASE]
        )[0]
        self.assertTrue(ready["ready_for_review"])
        self.assertEqual(ready["semantic_validation"], "EXPLORATORY")
        self.assertIn("not established", ready["review_scope"])
        for changes in (
            {"replay": "FAILED"},
            {"comparison_reference_reproduced": False},
            {"comparison_reference_reproduced": False, "control_reproduced": True},
        ):
            pending = contract.delivery_candidates([{**row, **changes}], [CASE])[0]
            self.assertFalse(pending["ready_for_review"])
        ignored = [
            {**row, "status": "INVALID"},
            {**row, "event": "started"},
            {**row, "contract": None},
            {**row, "contract": {"plan": {"mode": "control"}}},
        ]
        self.assertEqual(contract.delivery_candidates(ignored, [CASE]), [])


if __name__ == "__main__":
    unittest.main()
