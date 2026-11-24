"""Frozen Qwen startup boundaries; no Docker, compiler, simulation or API calls."""

import copy
import io
import json
import os
import subprocess
import sys
import tempfile
import tomllib
import unittest
from pathlib import Path
from unittest.mock import patch

import control
import runtime


class AgentEntered(Exception):
    pass


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="qwen-runtime-test-")
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)
        self.source = self.base / "simthesizer"
        for name in ("simthesizer", "inputs", "task", "baseline-policy"):
            (self.base / name).mkdir()
        directory = self.source / runtime.REL
        directory.mkdir(parents=True)
        (directory / "SEARCH_TASK.md").write_text("Frozen research instructions\n")
        (self.base / "inputs/workload.jsonl").write_text('{"fixture": true}\n')
        (self.base / "baseline-policy/mod.rs").write_text("// reference\n")
        self.config = tomllib.loads((control.HERE / "experiment.toml").read_text())
        self.manifest = {
            "profile": "qwen_moe",
            "experiment": self.config,
            "source_inventory_sha256": control.infra.inventory(self.source),
            "inputs_sha256": control.infra.inventory(self.base / "inputs"),
            "baseline_policy_sha256": control.infra.inventory(
                self.base / "baseline-policy"
            ),
        }
        (self.base / "task/manifest.json").write_text(json.dumps(self.manifest))
        (self.base / "task/AGENTS.md").write_text(
            runtime.render_task(self.source, self.manifest)
        )
        self.environment = {"python": "fixture", "packages": {"networkx": "fixture"}}

    def path(self, value):
        path = Path(value)
        return (
            self.base / path.relative_to("/workspace")
            if path.is_relative_to("/workspace")
            else path
        )

    def enter(self, **overrides):
        environment = overrides.pop("environment", self.environment)
        with (
            patch.object(runtime, "Path", side_effect=self.path),
            patch.object(
                control.baseline_store,
                "validate",
                return_value={"result_environment": environment},
            ) as baseline,
            patch.object(
                control.baseline_store,
                "result_environment",
                return_value=self.environment,
            ),
            patch.object(
                runtime.subprocess, "check_output", return_value="codex-cli 0.166.0\n"
            ) as version,
            patch.dict(
                os.environ, {"CODEX_API_KEY": "fixture-only", **overrides}, clear=True
            ),
            patch.object(runtime.os, "dup2") as stdin,
            patch.object(runtime.os, "execvpe", side_effect=AgentEntered) as execute,
            patch.object(sys, "path", list(sys.path)),
        ):
            try:
                runtime.enter_container()
            except AgentEntered:
                pass
        return baseline, version, stdin, execute

    def test_default_experiment_passes_runtime_configuration(self):
        self.assertEqual(runtime.config_issues(self.config), [])
        wrong = copy.deepcopy(self.config)
        wrong["research"]["profile"] = "legacy"
        self.assertTrue(runtime.config_issues(wrong))

    def test_task_uses_frozen_template_and_parameter_values(self):
        rendered = runtime.render_task(self.source, self.manifest)
        self.assertTrue(rendered.startswith("Frozen research instructions\n"))
        settings = json.loads(rendered.split("```json\n", 1)[1].split("\n```", 1)[0])
        self.assertEqual(settings["slo"], self.config["slo"])
        self.assertEqual(settings["search"], self.config["search"])
        self.assertNotIn("agent.env", rendered)

    def test_research_workflow_is_explicit_and_validated(self):
        self.assertIn(
            "research-status.json", runtime.render_task(self.source, self.manifest)
        )
        for value in ("unknown", False, 3):
            config = copy.deepcopy(self.config)
            config["research"]["workflow"] = value
            self.assertTrue(runtime.config_issues(config))
        for value in (0, 11, True, 1.5):
            config = copy.deepcopy(self.config)
            config["research"]["max_stalled_sessions"] = value
            self.assertTrue(runtime.config_issues(config))
        legacy = copy.deepcopy(self.manifest)
        legacy["experiment"]["research"].pop("workflow")
        self.assertEqual(runtime.config_issues(legacy["experiment"]), [])
        self.assertNotIn(
            "research-status.json", runtime.render_task(self.source, legacy)
        )

    def test_entry_uses_sealed_baseline_and_explicit_agent_paths(self):
        baseline, version, stdin, execute = self.enter()
        baseline.assert_called_once_with(self.base, self.manifest)
        version.assert_called_once_with(["codex", "--version"], text=True)
        stdin.assert_called_once()
        command, argv, env = execute.call_args.args
        self.assertEqual(command, "codex")
        self.assertEqual(argv[argv.index("--cd") + 1], "/workspace/worktree")
        self.assertEqual(
            argv[argv.index("--output-last-message") + 1],
            "/workspace/notes/agent-final.txt",
        )
        self.assertEqual(argv[argv.index("--model") + 1], self.config["agent"]["model"])
        self.assertEqual(argv[-1], "-")
        self.assertEqual(env["CODEX_API_KEY"], "fixture-only")

    def test_gateway_is_passed_as_configuration_without_shell_interpolation(self):
        _, _, _, execute = self.enter(OPENAI_BASE_URL="http://example.invalid/v1")
        _, argv, env = execute.call_args.args
        self.assertIn(
            'model_providers.experiment_gateway.base_url="http://example.invalid/v1"',
            argv,
        )
        self.assertEqual(env["OPENAI_API_KEY"], "fixture-only")

    def test_frozen_inputs_and_prompt_tampering_stop_before_codex(self):
        for relative in ("inputs/workload.jsonl", "task/AGENTS.md"):
            with self.subTest(relative=relative):
                path = self.base / relative
                before = path.read_text()
                path.write_text(before + "changed\n")
                with self.assertRaisesRegex(ValueError, "frozen (inputs|task) changed"):
                    self.enter()
                path.write_text(before)

    def test_numerical_environment_must_match_the_local_baseline(self):
        with self.assertRaisesRegex(ValueError, "environment differs"):
            self.enter(environment={"python": "incompatible"})

    def test_baseline_policy_tampering_is_rejected(self):
        (self.base / "baseline-policy/mod.rs").write_text("// changed\n")
        with self.assertRaisesRegex(ValueError, "baseline policy"):
            self.enter()

    def test_check_validates_real_configuration_without_execution(self):
        with (
            patch.object(control.infra, "build_image") as build,
            patch.object(control.infra, "prepare") as prepare,
            patch.object(control.infra, "local_baseline") as baseline,
            patch.object(control.infra, "check_env_file") as credentials,
            patch.object(control, "supervise") as supervise,
            patch.object(control.subprocess, "run") as run,
            patch.object(control.subprocess, "call") as call,
            patch.object(control.subprocess, "check_output") as output,
            patch("sys.stdout", new=io.StringIO()) as stream,
        ):
            status = control.main(["--check"])
        self.assertEqual(status, 0)
        report = json.loads(stream.getvalue())
        self.assertEqual(report["configuration_issues"], [])
        self.assertEqual(report["runtime_gaps"], [])
        self.assertEqual(report["search_scope"], "cross_module_mechanisms_and_policies")
        self.assertEqual(report["scope_contract"], runtime.SCOPE_CONTRACT)
        for operation in (
            build,
            prepare,
            baseline,
            credentials,
            supervise,
            run,
            call,
            output,
        ):
            operation.assert_not_called()

    def test_prepare_freezes_real_qwen_bundle_without_starting_baseline(self):
        runs_root = self.base / "prepared"
        with (
            patch.object(control.infra, "build_image") as build,
            patch.object(control.infra, "local_baseline") as baseline,
            patch.object(control.infra, "check_env_file") as credentials,
            patch.object(control, "supervise") as supervise,
            patch.object(control.subprocess, "run") as execute,
            patch.object(control.subprocess, "call") as call,
            patch.object(control.subprocess, "check_output") as output,
            patch("sys.stdout", new=io.StringIO()),
        ):
            status = control.main(["--prepare-only", "--runs-root", str(runs_root)])
        self.assertEqual(status, 0)
        runs = list(runs_root.iterdir())
        self.assertEqual(len(runs), 1)
        run = runs[0]
        manifest = control.infra.load_bundle(run)
        self.assertEqual(manifest["profile"], "qwen_moe")
        self.assertEqual(manifest["baseline_mode"], "local")
        self.assertEqual(
            manifest["experiment"]["research"]["scope_contract"], runtime.SCOPE_CONTRACT
        )
        assets = manifest["experiment"]["inputs"]["assets"]
        self.assertEqual(len(assets), 1)
        frozen_pdf = run / "inputs" / Path(assets[0]).relative_to("/workspace/inputs")
        self.assertEqual(
            control.baseline_store.digest(frozen_pdf),
            "a04a3f470560fbdc470a182b12c7f11d86901e5db945c80282e3f4e02a7a99ea",
        )
        self.assertEqual(
            manifest["source_inventory_sha256"],
            control.infra.inventory(run / "worktree"),
        )
        self.assertFalse((run / "task/baseline.json").exists())
        self.assertFalse(
            any(path.name == "agent.env" for path in (run / "source").rglob("*"))
        )
        # The reused local evaluator must accept the new frozen profile before
        # any compilation or baseline simulation is requested.
        import evaluate

        experiment = evaluate.Experiment(run / "task/manifest.json")
        self.assertEqual(experiment.config, manifest["experiment"])
        self.assertEqual(len(experiment.cases()), 3)
        for operation in (
            build,
            baseline,
            credentials,
            supervise,
            execute,
            call,
            output,
        ):
            operation.assert_not_called()

    def test_new_search_runs_frozen_baseline_then_build_then_frozen_search(self):
        runs_root = self.base / "new-search"
        env_file = self.base / "credentials with spaces.env"
        with (
            patch.object(control.infra, "build_image") as build,
            patch.object(control.infra, "local_baseline") as baseline,
            patch.object(control.infra, "check_env_file") as credentials,
            patch.object(control, "supervise") as supervise,
            patch.object(control.subprocess, "run") as local,
            patch.object(control.subprocess, "call", return_value=17) as search,
            patch("sys.stdout", new=io.StringIO()),
        ):
            status = control.main(
                ["--build", "--runs-root", str(runs_root), "--env-file", str(env_file)]
            )
        self.assertEqual(status, 17)
        run = next(runs_root.iterdir())
        command = [
            sys.executable,
            str(run / "source" / runtime.REL / "control.py"),
            "--resume",
            str(run),
        ]
        local.assert_called_once_with([*command, "--baseline-only"], check=True)
        search.assert_called_once_with([*command, "--env-file", str(env_file)])
        build.assert_called_once()
        self.assertEqual(build.call_args.args[1], self.config["agent"]["image"])
        baseline.assert_not_called()
        credentials.assert_called_once_with(env_file)
        supervise.assert_not_called()

    def test_failed_frozen_baseline_prevents_build_and_agent(self):
        with (
            patch.object(control.infra, "build_image") as build,
            patch.object(control.infra, "check_env_file"),
            patch.object(
                control.subprocess,
                "run",
                side_effect=subprocess.CalledProcessError(2, "fixture"),
            ),
            patch.object(control.subprocess, "call") as search,
            patch("sys.stdout", new=io.StringIO()),
            self.assertRaises(subprocess.CalledProcessError),
        ):
            control.main(["--build", "--runs-root", str(self.base / "failed-search")])
        build.assert_not_called()
        search.assert_not_called()

    def test_frozen_resume_reuses_baseline_and_checks_image_before_supervision(self):
        with patch("sys.stdout", new=io.StringIO()):
            run = control.infra.prepare(
                control.ROOT,
                self.config,
                self.base / "resume",
                control.infra.BUILD_DEFAULTS,
                "host",
                baseline_mode="local",
                profile="qwen_moe",
            )
        manifest = control.infra.load_bundle(run)
        (run / "task/baseline.json").write_text("mock anchor")
        env_file = self.base / "credentials.env"
        sealed = {"accounting": {"attempts": 3, "wall_seconds": 7.0}}
        with (
            patch.object(
                control, "__file__", str(run / "source" / runtime.REL / "control.py")
            ),
            patch.object(
                control.baseline_store, "validate", return_value=sealed
            ) as validate,
            patch.object(control.infra, "local_baseline") as baseline,
            patch.object(control.infra, "check_env_file") as credentials,
            patch.object(
                control.infra, "image_identity", return_value="sha256:frozen"
            ) as image,
            patch.object(control, "supervise", return_value=0) as supervise,
            patch.object(control.subprocess, "call") as reexec,
        ):
            status = control.main(["--resume", str(run), "--env-file", str(env_file)])
        self.assertEqual(status, 0)
        baseline.assert_not_called()
        reexec.assert_not_called()
        validate.assert_called_once_with(run, manifest)
        credentials.assert_called_once_with(env_file)
        image.assert_called_once_with(
            self.config["agent"]["image"], manifest["build_input_sha256"]
        )
        supervise.assert_called_once_with(
            run, manifest, "sha256:frozen", env_file, sealed
        )


if __name__ == "__main__":
    unittest.main()
