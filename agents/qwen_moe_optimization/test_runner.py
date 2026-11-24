"""Filesystem and scoring contracts; no build, Docker or simulation."""

import copy
import json
import shutil
import tempfile
import tomllib
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import runner


class JobTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="qwen-runner-test-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.run = self.root / "run"
        self.candidate = self.root / "candidate"
        self.job = self.root / "job"
        for name in ("source", "inputs", "task", "baseline-policy"):
            (self.run / name).mkdir(parents=True)
        source = self.run / "source"
        (source / "scripts/rooflang_bridge").mkdir(parents=True)
        shutil.copyfile(
            runner.ROOT / "scripts/results.py", source / "scripts/results.py"
        )
        # Frozen checks are copied as evidence; these tests never execute them.
        modules = {
            name.split(".", 1)[0]
            for name in (
                *runner.CHECKS,
                *runner.LIFECYCLE_CHECKS,
                *runner.GRAPH_CHECKS,
                *runner.OBSERVATION_CHECKS,
            )
        }
        (source / "tests").mkdir()
        for name in modules:
            shutil.copyfile(
                runner.ROOT / f"tests/{name}.py", source / f"tests/{name}.py"
            )
        shutil.copytree(
            runner.ROOT / "crates/simthesizer-core/tests",
            source / "crates/simthesizer-core/tests",
        )
        fixture = source / runner.RESOURCE_FIXTURE
        fixture.parent.mkdir(parents=True)
        shutil.copyfile(runner.ROOT / runner.RESOURCE_FIXTURE, fixture)
        (source / "scripts/rooflang_bridge/qwen3.py").write_text(
            "placement = 'original'\n"
        )
        (source / runner.launcher.POLICY_REL).mkdir(parents=True)
        (source / runner.launcher.POLICY_REL / "mod.rs").write_text(
            "// initial policy\n"
        )
        (self.run / "baseline-policy/mod.rs").write_text("// frozen reference policy\n")
        config = tomllib.loads(
            (runner.ROOT / "agents/qwen_moe_optimization/experiment.toml").read_text()
        )
        # This fixture emits TTFT=0.5s and TPOT=0.01s; user experiment SLOs vary.
        config["slo"] = {
            "ttft_limit_s": 1.0,
            "tpot_limit_s": 0.1,
            "min_joint_pass_fraction": 0.95,
        }
        config["inputs"]["scenario"] = "/workspace/inputs/scenario.toml"
        config["loads"]["arrival_rate_multipliers"] = [1.0]
        config["workloads"] = [
            {"id": "test", "path": "/workspace/inputs/workload.jsonl", "split": "train"}
        ]
        self.workload = [
            {
                "request_id": 7,
                "arrival_time": 0,
                "initial": [0],
                "sub_requests": [
                    {"kind": "llm", "input_tokens": 2, "output_tokens": 2, "next": []}
                ],
            }
        ]
        (self.run / "inputs/workload.jsonl").write_text(
            json.dumps(self.workload[0]) + "\n"
        )
        self.scenario = {
            "system": {"kind": "pd_disaggregation", "shared_execution": True}
        }
        for index, pool in enumerate(("prefill_instances", "decode_instances")):
            self.scenario["system"][pool] = [
                {
                    "model_runner": {
                        "kind": "rooflang",
                        "shared_execution": True,
                        "command": ["python", "scripts/rooflang_worker.py"],
                        "hardware": {"gpu_ids": [f"gpu-{index}"]},
                    }
                }
            ]
        (self.run / "inputs/scenario.toml").write_text(
            runner.evaluate.toml_document(self.scenario)
        )
        self.manifest = {
            "profile": "qwen_moe",
            "experiment": config,
            "baseline_mode": "local",
            "source_sha256": runner.launcher.inventory(source, mutable=True),
            "source_inventory_sha256": runner.launcher.inventory(source),
            "inputs_sha256": runner.launcher.inventory(self.run / "inputs"),
            "baseline_policy_sha256": runner.launcher.inventory(
                self.run / "baseline-policy"
            ),
        }
        runner.launcher.save(self.run / "task/manifest.json", self.manifest)
        shutil.copytree(source, self.candidate / "source")
        (self.candidate / "source/scripts/rooflang_bridge/qwen3.py").write_text(
            "placement = 'candidate'\n"
        )
        files = runner.launcher.inventory(self.candidate / "source")
        self.meta = {
            "source_sha256": runner.baseline_store.identity(files),
            "files": files,
            "mechanism_sha256": "mechanism",
            "policy_sha256": "policy",
            "kind": "mechanism",
            "changed_files": ["scripts/rooflang_bridge/qwen3.py"],
        }
        runner.launcher.save(self.candidate / "candidate.json", self.meta)
        self.case = runner.baseline_store.cases(self.manifest)[0]

        def mock_plots(_run, _manifest, destination, _deadline):
            for name in ("metrics.png", "throughput.png"):
                (destination / name).write_bytes(b"host-generated fixture")

        patcher = patch.object(runner, "draw_plots", side_effect=mock_plots)
        self.plotting = patcher.start()
        self.addCleanup(patcher.stop)

    def change(self, relative, text):
        path = self.candidate / "source" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        files = runner.launcher.inventory(self.candidate / "source")
        self.meta.update(
            files=files, source_sha256=runner.baseline_store.identity(files)
        )
        self.meta["changed_files"].append(relative)
        runner.launcher.save(self.candidate / "candidate.json", self.meta)

    def prepare(self, **kwargs):
        with patch.object(runner.launcher, "render_task", return_value="frozen task"):
            return runner.prepare_job(
                self.run, self.candidate, self.job, self.case["id"], **kwargs
            )

    def authored_contract(self, tests):
        impact = runner.candidate_contract.impact(self.meta["changed_files"])
        plan = {
            "schema_version": 1,
            "mode": "candidate",
            "summary": "Change placement",
            "state_ownership": "Placement belongs to the request",
            "preserved_semantics": "No work is removed",
            "expected_effect": "Less communication",
            "alternatives": ["Reference placement"],
            "stop_conditions": ["A resource invariant fails"],
            "evidence": [
                {
                    "kind": "baseline",
                    "id": self.case["id"],
                    "observation": "Communication pressure",
                }
            ],
            "control": {"kind": "baseline", "id": self.case["id"]},
            "checks": [
                {"id": name, "argument": "Exercise the invariant", "tests": tests}
                for name in impact["required_checks"]
            ],
        }
        contract = {
            "version": runner.research_contract.VERSION,
            "source_sha256": self.meta["source_sha256"],
            "required_checks": impact["required_checks"],
            "plan": plan,
        }
        contract["id"] = runner.baseline_store.identity(contract)
        return contract

    def evidence(self, mechanism_contract=None):
        self.prepare(mechanism_contract=mechanism_contract)
        directory = self.job / "artifacts/trials/example"
        directory.mkdir(parents=True)
        config = copy.deepcopy(self.scenario)
        settings = self.manifest["experiment"]
        config["system"]["policy"] = {
            "kind": "candidate",
            "seed": settings["search"]["seed"],
            "limits": {
                name: settings["policy"][name] for name in runner.evaluate.LIMITS
            },
            "slo": {
                name: settings["slo"][name] for name in ("ttft_limit_s", "tpot_limit_s")
            },
        }
        for pool in ("prefill_instances", "decode_instances"):
            config["system"][pool][0]["model_runner"]["command"] = [
                "/opt/venv/bin/python",
                "/workspace/simthesizer/scripts/rooflang_worker.py",
            ]
        (directory / "config.toml").write_text(runner.evaluate.toml_document(config))
        (directory / "workload.jsonl").write_text(json.dumps(self.workload[0]) + "\n")
        result = copy.deepcopy(self.workload[0])
        result["sub_requests"][0]["token_latency"] = [0.5, 0.01]
        (directory / "result.jsonl").write_text(json.dumps(result) + "\n")
        runner.launcher.save(
            directory / "policy-stats.json", {"simulation_end_s": 0.51}
        )
        runner.launcher.save(directory / "metrics.json", {"forged": True})
        build_id = "a" * 64
        cached = self.job / "cache/builds" / build_id
        cached.mkdir(parents=True)
        shutil.copytree(self.job / "source", cached / "source")
        (cached / "simthesizer").write_bytes(b"fixture binary; never executed")
        build = {
            "source_sha256": runner.read_json(self.job, "task/manifest.json")[
                "source_sha256"
            ],
            "build_id": build_id,
            "binary_sha256": runner.launcher.digest(cached / "simthesizer"),
            "candidate_id": runner.baseline_store.identity(
                {
                    "kind": "candidate",
                    "files": runner.launcher.inventory(
                        self.job / "source" / runner.launcher.POLICY_REL
                    ),
                }
            ),
        }
        runner.launcher.save(directory / "build.json", build)
        runner.launcher.save(cached / "build.json", build)
        for name in ("metrics.png", "throughput.png"):
            (directory / name).write_bytes(b"fixture image")
        runner.launcher.save(
            self.job / "artifacts/trusted-checks.json",
            {
                "passed": True,
                "checks": runner.read_json(self.job, "task/job.json")["checks"],
                "validation_plan": runner.read_json(self.job, "task/job.json")[
                    "validation_plan"
                ],
                "additional_checks": runner.read_json(self.job, "task/job.json")[
                    "additional_checks"
                ],
            },
        )
        (self.job / "artifacts/trusted-checks.log").write_text("fixture checks\n")
        runner.launcher.save(
            self.job / "artifacts/runner-result.json",
            {
                "directory": "trials/example",
                "case": self.case,
                "status": "VALID_FEASIBLE",
                "objectives": {"goodput_per_gpu": 999999},
            },
        )
        return directory

    def test_mechanism_python_and_rust_tree_are_sealed_into_actual_build_source(self):
        self.prepare()
        derived = runner.read_json(self.job, "task/manifest.json")
        self.assertNotIn("baseline_mode", derived)
        self.assertEqual(derived["experiment"], self.manifest["experiment"])
        self.assertEqual(
            derived["source_sha256"],
            runner.launcher.inventory(self.job / "source", mutable=True),
        )
        self.assertEqual(
            (self.job / "source/scripts/rooflang_bridge/qwen3.py").read_text(),
            "placement = 'candidate'\n",
        )
        self.assertEqual(
            (self.run / "source/scripts/rooflang_bridge/qwen3.py").read_text(),
            "placement = 'original'\n",
        )

    def test_reference_policy_keeps_candidate_mechanism_for_c_control(self):
        self.prepare(policy_kind="reference")
        self.assertEqual(
            (self.job / "source" / runner.launcher.POLICY_REL / "mod.rs").read_text(),
            "// frozen reference policy\n",
        )
        self.assertEqual(
            (self.job / "source/scripts/rooflang_bridge/qwen3.py").read_text(),
            "placement = 'candidate'\n",
        )

    def test_altered_snapshot_or_unknown_case_rejected_before_job_creation(self):
        (self.candidate / "source/scripts/rooflang_bridge/qwen3.py").write_text(
            "changed again\n"
        )
        with self.assertRaisesRegex(ValueError, "sealed identity"):
            self.prepare()
        self.assertFalse(self.job.exists())
        with self.assertRaisesRegex(ValueError, "unknown case"):
            runner.prepare_job(self.run, self.candidate, self.job, "unknown")

    def test_host_recomputes_metrics_and_never_promotes_mechanism_to_verified(self):
        self.evidence()
        result = runner.collect_job(self.run, self.job, self.meta, self.case["id"])
        self.assertEqual(result["status"], "EVALUATED", result)
        self.assertAlmostEqual(result["objectives"]["goodput_per_gpu"], 2 / 0.51 / 2)
        self.assertEqual(result["metrics"]["completion"]["completed_requests"], 1)
        self.assertNotIn("forged", result["metrics"])
        self.assertEqual(result["semantic_validation"], "exploratory")
        self.assertEqual(result["verification"], "unverified")
        self.assertFalse(result["checks"]["semantic_proof"])
        self.assertEqual(result["checks"]["origin"], "candidate-container")
        self.assertEqual(
            (self.job / "collected/metrics.png").read_bytes(), b"host-generated fixture"
        )
        self.plotting.assert_called_once()

    def test_changed_gpu_allocation_is_not_scored(self):
        directory = self.evidence()
        config = tomllib.loads((directory / "config.toml").read_text())
        config["system"]["prefill_instances"][0]["model_runner"]["hardware"][
            "gpu_ids"
        ].append("extra")
        (directory / "config.toml").write_text(runner.evaluate.toml_document(config))
        result = runner.collect_job(self.run, self.job, self.meta, self.case["id"])
        self.assertEqual(result["status"], "INVALID")
        self.assertIn("configuration changed", result["reason"])

    def test_incomplete_result_and_symlink_sidecar_are_rejected(self):
        directory = self.evidence()
        result = copy.deepcopy(self.workload[0])
        result["sub_requests"][0]["token_latency"] = [0.5]
        (directory / "result.jsonl").write_text(json.dumps(result))
        outcome = runner.collect_job(self.run, self.job, self.meta, self.case["id"])
        self.assertEqual(outcome["status"], "INVALID")
        self.assertIn("complete results", outcome["reason"])
        (directory / "config.toml").unlink()
        (directory / "config.toml").symlink_to(self.run / "inputs/scenario.toml")
        outcome = runner.collect_job(self.run, self.job, self.meta, self.case["id"])
        self.assertEqual(outcome["status"], "INVALID")
        self.assertIn("symlink", outcome["reason"])

    def test_evidence_traversal_is_rejected(self):
        self.evidence()
        receipt = runner.read_json(self.job, "artifacts/runner-result.json")
        receipt["directory"] = "../../outside"
        runner.launcher.save(self.job / "artifacts/runner-result.json", receipt)
        outcome = runner.collect_job(self.run, self.job, self.meta, self.case["id"])
        self.assertEqual(outcome["status"], "INVALID")
        self.assertIn("escapes", outcome["reason"])

    def test_binary_hash_is_checked_without_executing_candidate(self):
        self.evidence()
        (self.job / "cache/builds" / ("a" * 64) / "simthesizer").write_bytes(
            b"different binary"
        )
        outcome = runner.collect_job(self.run, self.job, self.meta, self.case["id"])
        self.assertEqual(outcome["status"], "INVALID")
        self.assertIn("binary content", outcome["reason"])

    def test_new_python_and_rust_tests_are_included_in_the_trial_budget(self):
        meta = copy.deepcopy(self.meta)
        meta["files"].update(
            {
                "tests/placement/test_new_owner.py": "new",
                "tests/placement/helper.py": "helper",
                "crates/simthesizer-core/tests/new_placement.rs": "new",
            }
        )
        meta["changed_files"].append("crates/simthesizer-core/src/kv.rs")
        checks = runner.additional_checks(
            meta, self.manifest["source_inventory_sha256"]
        )
        self.assertEqual(
            checks,
            [
                {"kind": "rust_unit"},
                {"kind": "rust_integration", "target": "new_placement"},
                {"kind": "python", "path": "tests/placement/test_new_owner.py"},
            ],
        )
        commands = [
            runner.check_command(check, Path("/workspace/simthesizer"))
            for check in checks
        ]
        self.assertEqual(commands[0][-1], "--lib")
        self.assertEqual(commands[1][-2:], ["--test", "new_placement"])
        self.assertEqual(
            commands[2][-2:],
            [
                "--check-python-test",
                "/workspace/simthesizer/tests/placement/test_new_owner.py",
            ],
        )

    def test_shared_cache_is_explicit_and_scoped_to_current_run(self):
        cache = self.run / "candidate-cache/one"
        descriptor = self.prepare(cache_path=cache)
        self.assertEqual(descriptor["cache_path"], str(cache))
        self.assertTrue(cache.is_dir())

    def test_cli_extensions_run_cli_unit_and_new_integration_targets(self):
        meta = copy.deepcopy(self.meta)
        meta["changed_files"].append("crates/simthesizer-cli/src/adapter.rs")
        meta["files"]["crates/simthesizer-cli/tests/adapter.rs"] = "new"
        checks = runner.additional_checks(
            meta, self.manifest["source_inventory_sha256"]
        )
        self.assertIn({"kind": "rust_binary", "package": "simthesizer-cli"}, checks)
        check = {
            "kind": "rust_integration",
            "target": "adapter",
            "package": "simthesizer-cli",
        }
        self.assertIn(check, checks)
        command = runner.check_command(check, self.job / "source")
        self.assertEqual(command[-4:], ["-p", "simthesizer-cli", "--test", "adapter"])
        command = runner.check_command(checks[1], self.job / "source")
        self.assertEqual(command[-3:], ["-p", "simthesizer-cli", "--bins"])

    def test_core_changes_inject_frozen_public_contracts_into_build_identity(self):
        self.change(
            "crates/simthesizer-core/src/resources.rs", "// changed implementation\n"
        )
        descriptor = self.prepare()
        plan = descriptor["validation_plan"]
        self.assertTrue(plan["impact"]["core_changed"])
        self.assertEqual(
            plan["rust_targets"], [*runner.NATIVE_TARGETS, runner.RESOURCE_TARGET]
        )
        fixture = self.job / "source" / runner.RESOURCE_DESTINATION
        self.assertEqual(
            fixture.read_bytes(),
            (self.run / "source" / runner.RESOURCE_FIXTURE).read_bytes(),
        )
        self.assertNotIn(runner.RESOURCE_DESTINATION, self.meta["files"])
        self.assertIn(
            runner.RESOURCE_DESTINATION, descriptor["source_inventory_sha256"]
        )
        self.assertIn(runner.LIFECYCLE_CHECKS[0], plan["python_checks"])

    def test_candidate_cannot_replace_frozen_tests_or_reserved_fixture(self):
        self.change("tests/test_qwen3_ep.py", "# skipped invariants\n")
        with self.assertRaisesRegex(ValueError, "changed frozen validation"):
            self.prepare()
        shutil.rmtree(self.job)
        self.change(runner.RESOURCE_DESTINATION, "// fake contract\n")
        with self.assertRaisesRegex(ValueError, "reserved frozen validation"):
            self.prepare()
        self.assertFalse(self.job.exists())

    def test_receipts_cannot_drop_host_selected_checks(self):
        self.evidence()
        checks = runner.read_json(self.job / "artifacts", "trusted-checks.json")
        checks["validation_plan"]["python_checks"] = []
        runner.launcher.save(self.job / "artifacts/trusted-checks.json", checks)
        outcome = runner.collect_job(self.run, self.job, self.meta, self.case["id"])
        self.assertEqual(outcome["status"], "INVALID")
        self.assertIn("checks did not finish", outcome["reason"])

    def test_missing_reference_test_rejects_job_before_container_creation(self):
        manifest = copy.deepcopy(self.manifest)
        del manifest["source_inventory_sha256"]["tests/test_qwen3_ep.py"]
        runner.launcher.save(self.run / "task/manifest.json", manifest)
        with self.assertRaisesRegex(ValueError, "missing frozen validation sources"):
            self.prepare()
        self.assertFalse(self.job.exists())

    def test_authored_existing_test_file_is_executed_as_a_whole_file(self):
        path = "tests/test_qwen3_shared.py"
        contract = self.authored_contract([path])
        descriptor = self.prepare(mechanism_contract=contract)
        self.assertEqual(descriptor["mechanism_contract"], contract)
        self.assertEqual(descriptor["validation_plan"]["declared_tests"], [path])
        self.assertEqual(
            descriptor["additional_checks"], [{"kind": "python", "path": path}]
        )
        command = runner.check_command(
            descriptor["additional_checks"][0], self.job / "source"
        )
        self.assertEqual(
            command[-2:], ["--check-python-test", str(self.job / "source" / path)]
        )

    def test_authored_native_contract_is_run_once_as_a_frozen_target(self):
        contract = self.authored_contract(["crates/simthesizer-core/tests/engine.rs"])
        descriptor = self.prepare(mechanism_contract=contract)
        self.assertIn("engine", descriptor["validation_plan"]["rust_targets"])
        self.assertEqual(descriptor["additional_checks"], [])
        self.assertTrue((self.job / "source" / runner.RESOURCE_DESTINATION).is_file())

    def test_collection_requires_the_same_host_owned_authored_contract(self):
        contract = self.authored_contract(["tests/test_qwen3_shared.py"])
        self.evidence(mechanism_contract=contract)
        result = runner.collect_job(self.run, self.job, self.meta, self.case["id"])
        self.assertEqual(result["status"], "INVALID")
        self.assertIn("job identity changed", result["reason"])
        result = runner.collect_job(
            self.run, self.job, self.meta, self.case["id"], mechanism_contract=contract
        )
        self.assertEqual(result["status"], "EVALUATED", result)

    def test_authored_contract_cannot_change_candidate_or_skip_requirements(self):
        contract = self.authored_contract(["tests/test_qwen3_shared.py"])
        contract["required_checks"] = []
        contract["id"] = runner.baseline_store.identity(
            {key: value for key, value in contract.items() if key != "id"}
        )
        with self.assertRaisesRegex(ValueError, "contract does not identify"):
            self.prepare(mechanism_contract=contract)
        self.assertFalse(self.job.exists())

    def test_unknown_implementation_uses_all_conservative_contracts(self):
        meta = copy.deepcopy(self.meta)
        meta["changed_files"] = ["scripts/rooflang_bridge/new_pipeline.py"]
        plan = runner.validation_plan(meta, self.manifest["source_inventory_sha256"])
        self.assertTrue(plan["impact"]["core_changed"])
        self.assertIn("resource_bounds", plan["impact"]["required_checks"])
        self.assertIn("observation_neutrality", plan["impact"]["required_checks"])
        self.assertIn(runner.RESOURCE_TARGET, plan["rust_targets"])

    def test_failure_diagnostics_tail_is_bounded(self):
        self.prepare()
        (self.job / "artifacts/trusted-checks.log").write_text("x" * 10000)
        experiment = SimpleNamespace(
            artifacts=self.job / "artifacts", cache=self.job / "cache"
        )
        with patch("sys.stderr") as output:
            runner.report_failure(
                experiment, {"status": "INVALID", "reason": "check failed"}
            )
        text = "".join(str(call.args[0]) for call in output.write.call_args_list)
        self.assertIn("check failed", text)
        self.assertLess(len(text), 4096)

    def test_empty_new_python_test_cannot_claim_check_coverage(self):
        path = self.run / "source/tests/test_empty.py"
        path.parent.mkdir(exist_ok=True)
        path.write_text("# No test cases.\n")
        with (
            patch.object(runner, "ROOT", self.run / "source"),
            self.assertRaisesRegex(ValueError, "no unittest cases"),
        ):
            runner.python_test(path)

    def test_build_hook_runs_new_checks_under_same_deadline_and_compiler_cache(self):
        self.prepare()
        descriptor = runner.read_json(self.job, "task/job.json")
        descriptor["additional_checks"] = [
            {"kind": "rust_unit"},
            {"kind": "python", "path": "tests/test_added.py"},
        ]
        runner.launcher.save(self.job / "task/job.json", descriptor)
        experiment = runner.CheckedExperiment.__new__(runner.CheckedExperiment)
        for name in ("source", "cache", "artifacts"):
            setattr(experiment, name, self.job / name)
        experiment.run, experiment.env = self.job, {}
        executable = self.job / "cache/builds/binary"
        with (
            patch.object(
                runner.evaluate.Experiment,
                "build",
                return_value=(executable, {"tools": {}}),
            ),
            patch.object(runner.evaluate, "run_process") as process,
        ):
            experiment.build({}, 123.0)
        self.assertEqual(process.call_count, 3)
        for call in process.call_args_list:
            self.assertEqual(call.kwargs["deadline"], 123.0)
            self.assertEqual(
                call.kwargs["env"]["CARGO_TARGET_DIR"],
                str(self.job / "cache/targets" / runner.baseline_store.identity({})),
            )
        self.assertEqual(process.call_args_list[1].args[0][-1], "--lib")
        self.assertEqual(process.call_args_list[2].args[0][-2], "--check-python-test")

    def test_native_contracts_use_sealed_targets_and_share_trial_deadline(self):
        self.change("crates/simthesizer-core/src/engine.rs", "// candidate engine\n")
        self.prepare()
        experiment = runner.CheckedExperiment.__new__(runner.CheckedExperiment)
        for name in ("source", "cache", "artifacts"):
            setattr(experiment, name, self.job / name)
        experiment.run, experiment.env = self.job, {}
        with (
            patch.object(
                runner.evaluate.Experiment,
                "build",
                return_value=(self.job / "cache/builds/binary", {"tools": {}}),
            ),
            patch.object(runner.evaluate, "run_process") as process,
        ):
            experiment.build({}, 987.0)
        self.assertEqual(process.call_count, 3)
        native = process.call_args_list[1]
        command = native.args[0]
        self.assertEqual(command[:2], ["cargo", "test"])
        self.assertEqual(native.kwargs["deadline"], 987.0)
        self.assertEqual(
            [
                command[index + 1]
                for index, part in enumerate(command)
                if part == "--test"
            ],
            [*runner.NATIVE_TARGETS, runner.RESOURCE_TARGET],
        )
        self.assertEqual(process.call_args_list[2].args[0][-1], "--lib")
        receipt = runner.read_json(self.job / "artifacts", "trusted-checks.json")
        self.assertEqual(len(receipt["timings"]), 3)


if __name__ == "__main__":
    unittest.main()
