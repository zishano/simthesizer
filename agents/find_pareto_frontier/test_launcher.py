"""Launcher boundaries, using temporary fixtures and a fake Docker CLI only."""

import copy
import csv
import io
import json
import os
import subprocess
import tempfile
import time
import unittest
from contextlib import contextmanager, nullcontext
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import launcher


def experiment():
    # Synthetic values test validation; these are not performance recommendations.
    return {
        "schema_version": 1,
        "design_only": False,
        "inputs": {"scenario": "scenario.toml", "assets": []},
        "model_assumptions": {
            "length_model": "known_trace",
            "scheduler_cost": "zero_simulated_time",
        },
        "slo": {
            "ttft_limit_s": 1.0,
            "tpot_limit_s": 0.1,
            "min_joint_pass_fraction": 0.95,
        },
        "loads": {"arrival_rate_multipliers": [1.0, 2.0]},
        "workloads": [{"id": "test", "path": "input.jsonl", "split": "train"}],
        "policy": {
            "api_version": "test-api",
            "actions": ["route", "batch", "submit_kv", "wait"],
            **{
                field: [1, 8]
                for field in (
                    "p_token_budget_bounds",
                    "d_token_budget_bounds",
                    "p_batch_bounds",
                    "d_batch_bounds",
                    "kv_inflight_bounds",
                )
            },
            "max_wait_s": 0.1,
            "state_limit_bytes": 1024,
        },
        "agent": {
            "frontend": "codex",
            "model": "test-only",
            "cli_version": "0.166.0",
            "image": "test-image",
        },
        "search": {
            "seed": 0,
            "max_trials": 10,
            "max_wall_seconds": 10,
            "trial_wall_seconds": 2,
        },
        "verification": {"max_trials": 2, "max_wall_seconds": 10},
        "container": {
            "cpus": 1,
            "memory_mib": 128,
            "pids_limit": 32,
            "parallel_trials": 1,
        },
        "artifacts": {
            "reports": False,
            "full_graphs": False,
            "diagnostic_budget_mib": 0,
        },
    }


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="sim-agent-test-")
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.root = self.base / "project with spaces"
        self.root.mkdir()
        for relative in launcher.BUILD_FILES.values():
            self.write(relative, "fixture\n")
        self.write(
            launcher.AGENT_REL / "AGENTS.md.template",
            "Manifest {{MANIFEST_SHA256}}\nPolicy {{POLICY_DIRECTORY}}\n",
        )
        self.write("scenario.toml", '[system]\nkind = "pd_disaggregation"\n')
        self.write("input.jsonl", '{"request_id":0}\n')
        self.write("scripts/code.py", "# fixture\n")
        self.write(launcher.POLICY_REL / "policy.rs", "// original candidate\n")
        self.config = experiment()

    def write(self, relative, text):
        target = self.root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)
        return target

    def prepare(self):
        with patch("sys.stdout", new=io.StringIO()):
            return launcher.prepare(
                self.root,
                self.config,
                self.base / "runs",
                launcher.BUILD_DEFAULTS,
                "bridge",
            )

    def test_example_is_rejected_before_docker(self):
        with (
            patch.object(launcher.subprocess, "run") as execute,
            patch("sys.stdout", new=io.StringIO()) as stdout,
        ):
            code = launcher.main(
                [
                    "--check",
                    "--experiment",
                    str(launcher.HERE / "experiment.example.toml"),
                ]
            )
        self.assertEqual(code, 2)
        result = json.loads(stdout.getvalue())
        self.assertFalse(result["launch_ready"])
        self.assertTrue(result["configuration_issues"])
        self.assertEqual(result["runtime_gaps"], [])
        execute.assert_not_called()

    def test_invalid_constraints_do_not_silently_default(self):
        self.assertEqual(launcher.config_issues(self.config), [])
        for group, field, value in (
            ("slo", "ttft_limit_s", float("nan")),
            ("slo", "min_joint_pass_fraction", 1.1),
            ("search", "max_trials", True),
            ("container", "parallel_trials", 2),
            ("policy", "p_batch_bounds", [8, 1]),
        ):
            with self.subTest(field=field):
                candidate = copy.deepcopy(self.config)
                candidate[group][field] = value
                self.assertTrue(launcher.config_issues(candidate))

    def test_prepare_only_cannot_implicitly_build(self):
        with patch.object(launcher.subprocess, "run") as execute:
            with self.assertRaisesRegex(ValueError, "only valid when starting"):
                launcher.main(["--prepare-only", "--build"])
            execute.assert_not_called()

    def test_snapshot_excludes_credentials_and_unselected_data(self):
        for path in (
            "scripts/custom.env",
            "scripts/.env.secret",
            "output/huge.json",
            "examples/workloads/holdout.jsonl",
            "agents/find_pareto_frontier/runs/old/file",
            "scripts/__pycache__/x.pyc",
        ):
            self.write(path, "DO_NOT_COPY")
        run = self.prepare()
        manifest = launcher.load_bundle(run)
        self.assertFalse(
            any(
                "DO_NOT_COPY" in p.read_text()
                for p in (run / "source").rglob("*")
                if p.is_file()
            )
        )
        self.assertEqual(
            (run / "inputs/workloads/test.jsonl").read_bytes(),
            (self.root / "input.jsonl").read_bytes(),
        )
        self.assertEqual(
            manifest["experiment"]["inputs"]["scenario"], "/workspace/inputs/scenario"
        )
        self.assertNotIn("{{", (run / "task/AGENTS.md").read_text())

    def test_candidate_change_allowed_but_core_change_rejected(self):
        run = self.prepare()
        (run / "source" / launcher.POLICY_REL / "policy.rs").write_text("// changed\n")
        launcher.load_bundle(run)
        self.assertEqual(
            (run / "baseline-policy/policy.rs").read_text(), "// original candidate\n"
        )
        (run / "source/scripts/code.py").write_text("# changed core\n")
        with self.assertRaisesRegex(ValueError, "frozen source changed"):
            launcher.load_bundle(run)

    def test_input_task_and_baseline_tampering_rejected(self):
        for relative in (
            "inputs/scenario",
            "task/AGENTS.md",
            "baseline-policy/policy.rs",
        ):
            with self.subTest(relative=relative):
                run = self.prepare()
                with (run / relative).open("a") as target:
                    target.write("changed")
                with self.assertRaises(ValueError):
                    launcher.load_bundle(run)

    def test_symlink_candidate_cannot_escape_on_resume(self):
        run = self.prepare()
        candidate = run / "source" / launcher.POLICY_REL
        candidate.rename(candidate.with_name("candidate-old"))
        # Remove the old directory from this synthetic test's immutable namespace.
        candidate.with_name("candidate-old").rename(self.base / "candidate-old")
        candidate.symlink_to(self.base / "candidate-old", target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "symlink"):
            launcher.load_bundle(run)

    def test_env_rejects_path_injection_and_duplicate_empty_key(self):
        env = self.base / "agent.env"
        env.write_text("CODEX_API_KEY=test-only\nPATH=/untrusted\n")
        with self.assertRaisesRegex(ValueError, "unsupported/duplicate"):
            launcher.check_env_file(env)
        env.write_text("CODEX_API_KEY=\nCODEX_API_KEY=test-only\n")
        with self.assertRaises(ValueError):
            launcher.check_env_file(env)
        env.write_text(
            "CODEX_API_KEY=test-only\nOPENAI_BASE_URL=http://example.invalid/v1\n"
        )
        launcher.check_env_file(env)

    def test_mounts_freeze_core_without_mounting_host_docker_or_home(self):
        run = self.prepare()
        manifest = launcher.load_bundle(run)
        args = launcher.docker_arguments(
            run, manifest, "sha256:fixture", self.base / "agent.env"
        )
        mounts = [
            args[i + 1] for i, argument in enumerate(args) if argument == "--mount"
        ]
        self.assertTrue(
            any("target=/workspace/simthesizer,readonly" in m for m in mounts)
        )
        mutable = [m for m in mounts if not m.endswith(",readonly")]
        self.assertEqual(len(mutable), 3)
        self.assertTrue(any(str(launcher.POLICY_REL) in m for m in mutable))
        self.assertNotIn("docker.sock", " ".join(args))
        self.assertEqual(args[-1], "sha256:fixture")
        self.assertIn(str(run), " ".join(args))
        self.assertIn("--memory-swap", args)

    def test_build_context_is_minimal(self):
        self.write("scripts/private.env", "secret")
        contexts = []
        env = {"HTTP_PROXY": "http://fixture:1234", "NO_PROXY": "localhost,example"}

        def fake_build(argv, **kwargs):
            context = Path(argv[-1])
            contexts.append(context)
            files = {
                str(p.relative_to(context)) for p in context.rglob("*") if p.is_file()
            }
            self.assertEqual(files, set(launcher.BUILD_FILES))
            self.assertNotIn("secret", " ".join(argv))
            self.assertEqual(argv[:3], ["docker", "buildx", "build"])
            self.assertEqual(argv[argv.index("--builder") + 1], "project-builder")
            self.assertIn("--load", argv)
            self.assertIn("--network=host", argv)
            self.assertIn("--allow=network.host", argv)
            self.assertIs(kwargs["env"], env)
            for key, value in env.items():
                self.assertIn(
                    ["--build-arg", key],
                    [argv[i : i + 2] for i in range(len(argv) - 1)],
                )
                self.assertNotIn(value, " ".join(argv))

        with (
            patch.object(
                launcher,
                "project_builder",
                return_value=nullcontext(("project-builder", env)),
            ),
            patch.object(launcher, "build_command", side_effect=fake_build),
            patch.object(launcher, "image_identity") as image_identity,
        ):
            launcher.build_image(self.root, "test-image", launcher.BUILD_DEFAULTS)
        image_identity.assert_called_once_with(
            "test-image", launcher.build_identity(self.root, launcher.BUILD_DEFAULTS)
        )
        self.assertEqual(len(contexts), 1)
        self.assertFalse(contexts[0].exists())

    @contextmanager
    def mock_buildx(self, *, existing=False, driver="docker-container", failures=None):
        calls, builders = [], set()
        failures = failures or {}

        def fake_run(argv, **kwargs):
            self.assertEqual(argv[:2], ["docker", "buildx"])
            calls.append((list(argv), kwargs["env"].copy()))
            operation = "bootstrap" if "--bootstrap" in argv else argv[2]
            name = argv[argv.index("--name") + 1] if operation == "create" else argv[-1]
            if operation in failures:
                return subprocess.CompletedProcess(argv, 1, "", failures[operation])
            if operation == "inspect":
                if not existing and name not in builders:
                    return subprocess.CompletedProcess(
                        argv, 1, "", f'ERROR: no builder "{name}" found'
                    )
                return subprocess.CompletedProcess(
                    argv, 0, f"Name: {name}\nDriver: {driver}\n", ""
                )
            if operation == "create":
                builders.add(name)
            else:
                self.assertEqual(operation, "bootstrap")
            return subprocess.CompletedProcess(argv, 0, name + "\n", "")

        with (
            patch.object(launcher, "output", return_value="fixture-daemon"),
            patch.object(launcher.subprocess, "run", side_effect=fake_run),
            patch("sys.stdout", new=io.StringIO()),
        ):
            yield calls

    def test_project_builder_creates_once_and_reuses_cache_without_selecting_it(self):
        inherited_config = self.base / "other-buildx"
        inherited_config.mkdir()
        marker = inherited_config / "current"
        marker.write_text("mybuilder")
        original = {
            "BUILDX_BUILDER": "mybuilder",
            "BUILDX_CONFIG": str(inherited_config),
        }
        with patch.dict(os.environ, original, clear=True), self.mock_buildx() as calls:
            names, configs = [], []
            for _ in range(2):
                with launcher.project_builder(self.root, launcher.BUILDKIT_IMAGE) as (
                    name,
                    env,
                ):
                    names.append(name)
                    configs.append(env["BUILDX_CONFIG"])
                    self.assertNotIn("BUILDX_BUILDER", env)
                    self.assertTrue(
                        Path(env["BUILDX_CONFIG"]).is_relative_to(
                            self.root / "output/agent-build"
                        )
                    )
            self.assertEqual(dict(os.environ), original)
        self.assertEqual(names[0], names[1])
        self.assertEqual(configs[0], configs[1])
        create = [argv for argv, _ in calls if argv[2] == "create"]
        self.assertEqual(len(create), 1)
        self.assertIn("network=host", create[0])
        self.assertIn(
            "--buildkitd-flags=--allow-insecure-entitlement=network.host", create[0]
        )
        self.assertEqual(create[0][create[0].index("--driver") + 1], "docker-container")
        for argv, env in calls:
            self.assertNotIn("--use", argv)
            self.assertNotIn("mybuilder", argv)
            self.assertNotIn("BUILDX_BUILDER", env)
            self.assertIn(argv[2], ("inspect", "create"))
        self.assertEqual(marker.read_text(), "mybuilder")

    def test_builder_copies_proxies_with_csv_safe_no_proxy(self):
        proxies = {
            "HTTP_PROXY": "http://ignored.invalid:1234",
            "http_proxy": "http://chosen.invalid:1234",
            "HTTPS_PROXY": "http://secure.invalid:1234",
            "ALL_PROXY": "socks5://all.invalid:1234",
            "NO_PROXY": 'localhost,127.0.0.1,"internal"',
        }
        with (
            patch.dict(os.environ, proxies, clear=True),
            self.mock_buildx() as calls,
            launcher.project_builder(self.root, launcher.BUILDKIT_IMAGE) as (_, env),
        ):
            for upper in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY"):
                expected = proxies.get(upper.lower(), proxies[upper])
                self.assertEqual(env[upper], expected)
                self.assertEqual(env[upper.lower()], expected)
        create = next(argv for argv, _ in calls if argv[2] == "create")
        options = []
        for i, value in enumerate(create):
            if value == "--driver-opt":
                fields = next(csv.reader([create[i + 1]]))
                self.assertEqual(len(fields), 1)
                options.append(fields[0])
        for upper in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY"):
            expected = proxies.get(upper.lower(), proxies[upper])
            for key in (upper, upper.lower()):
                self.assertIn(f"env.{key}={expected}", options)

    def test_proxy_change_uses_another_builder_without_destroying_cached_one(self):
        with patch.dict(os.environ, {}, clear=True), self.mock_buildx() as calls:
            names = []
            for proxy in ("http://first.invalid:1234", "http://second.invalid:1234"):
                os.environ["HTTPS_PROXY"] = proxy
                with launcher.project_builder(self.root, launcher.BUILDKIT_IMAGE) as (
                    name,
                    _,
                ):
                    names.append(name)
        self.assertNotEqual(*names)
        self.assertEqual(sum(argv[2] == "create" for argv, _ in calls), 2)
        self.assertTrue(all(argv[2] in ("inspect", "create") for argv, _ in calls))

    def test_explicit_socks_proxy_overrides_shell_without_mutating_it(self):
        proxy = "socks5h://127.0.0.1:17890"
        original = {
            key: f"http://polluted-{key.lower()}.invalid:7890"
            for upper in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY")
            for key in (upper, upper.lower())
        }
        original.update(NO_PROXY="localhost,127.0.0.1", BUILDX_BUILDER="mybuilder")
        with patch.dict(os.environ, original, clear=True), self.mock_buildx() as calls:
            with launcher.project_builder(
                self.root, launcher.BUILDKIT_IMAGE, proxy=proxy
            ) as (_, env):
                for upper in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
                    self.assertEqual(env[upper], proxy)
                    self.assertEqual(env[upper.lower()], proxy)
                self.assertEqual(env["NO_PROXY"], original["NO_PROXY"])
                self.assertEqual(env["no_proxy"], original["NO_PROXY"])
                self.assertNotIn("BUILDX_BUILDER", env)
            self.assertEqual(dict(os.environ), original)
        create = next(argv for argv, _ in calls if argv[2] == "create")
        self.assertNotIn("polluted-", " ".join(create))
        for upper in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
            self.assertIn(f"env.{upper}={proxy}", create)
            self.assertIn(f"env.{upper.lower()}={proxy}", create)

    def test_builder_inspect_errors_never_create_or_bootstrap(self):
        for error in ("permission denied", 'no builder "another-project" found'):
            with (
                self.subTest(error=error),
                patch.dict(os.environ, {}, clear=True),
                self.mock_buildx(failures={"inspect": error}) as calls,
            ):
                with (
                    self.assertRaisesRegex(ValueError, "inspect failed"),
                    launcher.project_builder(self.root, launcher.BUILDKIT_IMAGE),
                ):
                    self.fail("inspect failure entered builder")
                self.assertEqual([argv[2] for argv, _ in calls], ["inspect"])

    def test_existing_wrong_driver_is_not_modified(self):
        with (
            patch.dict(os.environ, {}, clear=True),
            self.mock_buildx(existing=True, driver="docker") as calls,
            self.assertRaisesRegex(ValueError, "unexpected driver"),
            launcher.project_builder(self.root, launcher.BUILDKIT_IMAGE),
        ):
            self.fail("wrong driver was accepted")
        self.assertEqual([argv[2] for argv, _ in calls], ["inspect"])

    def test_builder_failure_stops_before_build_and_redacts_proxy(self):
        proxy = "http://user:private-password@127.0.0.1:17892"
        for operation in ("create", "bootstrap"):
            with (
                self.subTest(operation=operation),
                patch.dict(os.environ, {"HTTPS_PROXY": proxy}, clear=True),
                self.mock_buildx(failures={operation: f"cannot connect to {proxy}"}),
                patch.object(launcher, "build_command") as build,
                patch.object(
                    launcher, "image_identity", return_value="sha256:old-image"
                ) as old_image,
                self.assertRaises(ValueError) as raised,
            ):
                launcher.build_image(self.root, "test-image", launcher.BUILD_DEFAULTS)
            build.assert_not_called()
            old_image.assert_not_called()
            self.assertIn(operation, str(raised.exception))
            self.assertIn("<proxy>", str(raised.exception))
            self.assertNotIn("private-password", str(raised.exception))

    def test_failed_build_or_loaded_image_validation_cleans_context(self):
        for failure in ("build", "image"):
            contexts = []

            def build(argv, *, contexts=contexts, failure=failure, **kwargs):
                contexts.append(Path(argv[-1]))
                self.assertTrue(contexts[-1].is_dir())
                if failure == "build":
                    raise ValueError("fixture compilation failed")

            with (
                self.subTest(failure=failure),
                patch.object(
                    launcher,
                    "project_builder",
                    return_value=nullcontext(("project-builder", {})),
                ),
                patch.object(launcher, "build_command", side_effect=build),
                patch.object(
                    launcher,
                    "image_identity",
                    side_effect=ValueError("wrong loaded image"),
                ) as image,
                self.assertRaises(ValueError),
            ):
                launcher.build_image(self.root, "test-image", launcher.BUILD_DEFAULTS)
            self.assertEqual(len(contexts), 1)
            self.assertFalse(contexts[0].exists())
            if failure == "build":
                image.assert_not_called()
            else:
                image.assert_called_once()

    def test_primary_proxy_success_does_not_try_fallback(self):
        primary = "socks5h://127.0.0.1:17890"
        with patch.object(launcher, "build_image_once") as once:
            launcher.build_image(
                self.root,
                "test-image",
                launcher.BUILD_DEFAULTS,
                proxy=primary,
                fallback_proxy="http://10.250.0.3:7890",
            )
        once.assert_called_once_with(
            self.root,
            "test-image",
            launcher.BUILD_DEFAULTS,
            launcher.BUILDKIT_IMAGE,
            primary,
        )

    def test_network_failure_tries_fallback_once(self):
        primary, fallback = "socks5h://127.0.0.1:17890", "http://10.250.0.3:7890"
        for fallback_error in (
            None,
            launcher.BuildNetworkError("still offline"),
            ValueError("compile failed"),
        ):
            with (
                self.subTest(fallback_error=fallback_error),
                patch("sys.stdout", new=io.StringIO()),
                patch.object(
                    launcher,
                    "build_image_once",
                    side_effect=[
                        launcher.BuildNetworkError("502 Bad Gateway"),
                        fallback_error,
                    ],
                ) as once,
            ):
                if fallback_error is None:
                    launcher.build_image(
                        self.root,
                        "test-image",
                        launcher.BUILD_DEFAULTS,
                        proxy=primary,
                        fallback_proxy=fallback,
                    )
                else:
                    with self.assertRaises(type(fallback_error)) as raised:
                        launcher.build_image(
                            self.root,
                            "test-image",
                            launcher.BUILD_DEFAULTS,
                            proxy=primary,
                            fallback_proxy=fallback,
                        )
                    self.assertIs(raised.exception, fallback_error)
                self.assertEqual(
                    once.call_args_list,
                    [
                        call(
                            self.root,
                            "test-image",
                            launcher.BUILD_DEFAULTS,
                            launcher.BUILDKIT_IMAGE,
                            proxy,
                        )
                        for proxy in (primary, fallback)
                    ],
                )

    def test_regular_build_failure_does_not_try_fallback(self):
        error = ValueError("image build failed (exit 100): package not found")
        with (
            patch.object(launcher, "build_image_once", side_effect=error) as once,
            self.assertRaises(ValueError) as raised,
        ):
            launcher.build_image(
                self.root,
                "test-image",
                launcher.BUILD_DEFAULTS,
                proxy="socks5h://127.0.0.1:17890",
                fallback_proxy="http://10.250.0.3:7890",
            )
        self.assertIs(raised.exception, error)
        self.assertEqual(once.call_count, 1)

    def test_empty_or_identical_fallback_does_not_retry(self):
        primary = "socks5h://127.0.0.1:17890"
        for fallback in (None, "", primary):
            error = launcher.BuildNetworkError("502 Bad Gateway")
            with (
                self.subTest(fallback=fallback),
                patch.object(launcher, "build_image_once", side_effect=error) as once,
                self.assertRaises(launcher.BuildNetworkError) as raised,
            ):
                launcher.build_image(
                    self.root,
                    "test-image",
                    launcher.BUILD_DEFAULTS,
                    proxy=primary,
                    fallback_proxy=fallback,
                )
            self.assertIs(raised.exception, error)
            self.assertEqual(once.call_count, 1)

    def test_invalid_proxy_is_rejected_before_build(self):
        for proxy in (
            "ftp://example.invalid:1234",
            "http://example.invalid:bad",
            "socks5h://localhost:1234?invalid=1",
        ):
            with (
                self.subTest(proxy=proxy),
                patch.object(launcher, "build_image_once") as once,
                self.assertRaises(ValueError),
            ):
                launcher.build_image(
                    self.root,
                    "test-image",
                    launcher.BUILD_DEFAULTS,
                    proxy=proxy,
                    fallback_proxy="http://10.250.0.3:7890",
                )
            once.assert_not_called()

    def test_bootstrap_network_failure_is_distinct_from_builder_configuration(self):
        for operation in ("inspect", "create", "bootstrap"):
            with (
                self.subTest(operation=operation),
                patch.dict(os.environ, {}, clear=True),
                self.mock_buildx(failures={operation: "502 Bad Gateway"}),
                self.assertRaises(ValueError) as raised,
                launcher.project_builder(self.root, launcher.BUILDKIT_IMAGE),
            ):
                self.fail("failed builder was accepted")
            self.assertEqual(
                isinstance(raised.exception, launcher.BuildNetworkError),
                operation == "bootstrap",
            )

    def test_build_command_streams_and_classifies_failed_downloads(self):
        cases = (
            (
                100,
                "#8 E: Failed to fetch https://example.invalid/package 502 Bad Gateway\n",
                launcher.BuildNetworkError,
            ),
            (1, "#8 curl: (28) Connection timed out\n", launcher.BuildNetworkError),
            (
                1,
                'failed to do request: Head "https://registry-1.docker.io/v2/library/python/manifests/3.14.5-slim-bookworm": EOF\n',
                launcher.BuildNetworkError,
            ),
            (
                1,
                'Get "https://registry-1.docker.io/v2/library/python/blobs/sha256:fixture": unexpected EOF\n',
                launcher.BuildNetworkError,
            ),
            (100, "#8 E: Unable to locate package missing-package\n", ValueError),
            (1, "error[E0502]: cannot borrow value as mutable\n", ValueError),
            (
                2,
                "bash: line 1: unexpected EOF while looking for matching `\"'\n",
                ValueError,
            ),
            (1, "SyntaxError: unexpected EOF while parsing\n", ValueError),
            (
                2,
                (
                    "Downloaded https://example.invalid/source\n"
                    "sh: syntax error: unexpected EOF\n"
                ),
                ValueError,
            ),
            (0, "#8 retry recovered after 502 Bad Gateway\n", None),
        )
        for status, detail, error in cases:
            with (
                self.subTest(status=status, detail=detail),
                patch("sys.stdout", new=io.StringIO()) as stdout,
            ):
                process = MagicMock()
                process.__enter__.return_value = process
                process.wait.return_value = status

                def lines(detail=detail, stdout=stdout):
                    yield "#1 build started\n"
                    self.assertIn("#1 build started\n", stdout.getvalue())
                    yield detail

                process.stdout = lines()
                with patch.object(
                    launcher.subprocess, "Popen", return_value=process
                ) as spawn:
                    if error is None:
                        self.assertIsNone(
                            launcher.build_command(
                                ["docker", "buildx", "build"], env={}
                            )
                        )
                    else:
                        with self.assertRaises(error) as raised:
                            launcher.build_command(
                                ["docker", "buildx", "build"], env={}
                            )
                        self.assertIs(type(raised.exception), error)
                self.assertIn(detail, stdout.getvalue())
                process.wait.assert_called_once()
                self.assertEqual(spawn.call_args.kwargs["stderr"], subprocess.STDOUT)

    def test_registry_eof_log_triggers_one_fallback_build(self):
        primary, fallback = "socks5h://127.0.0.1:17890", "http://10.250.0.3:7890"
        logs = iter(
            (
                (
                    1,
                    'failed to do request: Head "https://registry-1.docker.io/v2/library/python/manifests/3.14.5-slim-bookworm": EOF\n',
                ),
                (0, "#9 exporting to docker image format\n"),
            )
        )
        contexts = []

        def builder(root, buildkit_image, proxy=None):
            self.assertEqual(root, self.root)
            self.assertEqual(buildkit_image, launcher.BUILDKIT_IMAGE)
            return nullcontext(("project-builder", {"HTTPS_PROXY": proxy}))

        def popen(argv, **kwargs):
            self.assertEqual(argv[:3], ["docker", "buildx", "build"])
            contexts.append(Path(argv[-1]))
            self.assertTrue(contexts[-1].is_dir())
            status, log = next(logs)
            process = MagicMock()
            process.__enter__.return_value = process
            process.stdout = io.StringIO(log)
            process.wait.return_value = status
            return process

        with (
            patch.object(launcher, "project_builder", side_effect=builder),
            patch.object(launcher.subprocess, "Popen", side_effect=popen) as spawn,
            patch.object(launcher, "image_identity") as image,
            patch("sys.stdout", new=io.StringIO()) as stdout,
        ):
            launcher.build_image(
                self.root,
                "test-image",
                launcher.BUILD_DEFAULTS,
                proxy=primary,
                fallback_proxy=fallback,
            )
        self.assertEqual(
            [attempt.kwargs["env"]["HTTPS_PROXY"] for attempt in spawn.call_args_list],
            [primary, fallback],
        )
        self.assertEqual(stdout.getvalue().count("fallback proxy once"), 1)
        image.assert_called_once_with(
            "test-image", launcher.build_identity(self.root, launcher.BUILD_DEFAULTS)
        )
        self.assertEqual(len(set(contexts)), 2)
        self.assertTrue(all(not context.exists() for context in contexts))

    def test_old_network_message_does_not_override_final_compile_failure(self):
        process = MagicMock()
        process.__enter__.return_value = process
        process.wait.return_value = 1
        process.stdout = io.StringIO(
            "retry recovered after 502 Bad Gateway\n"
            + "ordinary progress\n" * 1000
            + "error[E0502]: cannot borrow value as mutable\n"
        )
        with (
            patch.object(launcher.subprocess, "Popen", return_value=process),
            patch("sys.stdout", new=io.StringIO()),
            self.assertRaises(ValueError) as raised,
        ):
            launcher.build_command(["docker", "buildx", "build"], env={})
        self.assertIs(type(raised.exception), ValueError)

    def test_budget_is_not_reset_by_resume(self):
        run = self.prepare()
        manifest = launcher.load_bundle(run)
        launcher.save(
            run / "supervisor.json", {"consumed_wall_seconds": 10, "active": None}
        )
        with patch.object(launcher, "output") as execute:
            with self.assertRaisesRegex(ValueError, "exhausted"):
                launcher.supervise(run, manifest, "test-image", self.base / "agent.env")
            execute.assert_not_called()

    def fake_docker(self):
        binary = self.base / "bin/docker"
        binary.parent.mkdir()
        binary.write_text("""#!/usr/bin/env python3
import json, os, pathlib, sys, time
root = pathlib.Path(os.environ["FAKE_DOCKER_STATE"])
args = sys.argv[1:]
with (root / "calls.jsonl").open("a") as log:
    log.write(json.dumps(args) + "\\n")
if args[0] == "create":
    print("fake-container")
elif args[0] == "start":
    print("fixture log", flush=True)
    if os.environ.get("FAKE_HANG"):
        while not (root / "stopped").exists():
            time.sleep(0.02)
elif args[0] in ("stop", "kill"):
    if os.environ.get("FAKE_STOP_FAIL"):
        sys.exit(1)
    (root / "stopped").touch()
elif args[0] == "inspect":
    print(json.dumps({"ExitCode": 0, "OOMKilled": False, "Running": bool(os.environ.get("FAKE_RUNNING"))}))
""")
        binary.chmod(0o755)
        return {
            "PATH": str(binary.parent) + os.pathsep + os.environ["PATH"],
            "FAKE_DOCKER_STATE": str(self.base),
        }

    def test_supervisor_captures_exit_without_claiming_verification(self):
        run = self.prepare()
        with patch.dict(os.environ, self.fake_docker()):
            result = launcher.supervise(
                run, launcher.load_bundle(run), "test-image", self.base / "env"
            )
        state = json.loads((run / "supervisor.json").read_text())
        self.assertEqual(result, 0)
        self.assertEqual(state["last_outcome"], "agent_exited")
        self.assertIsNone(state["active"])
        self.assertIn("fixture log", (run / "agent.trace.log").read_text())
        self.assert_cleanup_calls()

    def assert_cleanup_calls(self, expected=None):
        calls = [
            json.loads(line)
            for line in (self.base / "calls.jsonl").read_text().splitlines()
        ]
        self.assertTrue(
            all(
                command[0] in {"create", "start", "inspect", "stop", "kill", "rm"}
                for command in calls
            )
        )
        cleanup = [command for command in calls if command[0] in {"stop", "kill", "rm"}]
        self.assertEqual(
            cleanup,
            expected
            if expected is not None
            else [
                ["stop", "--time", "5", "fake-container"],
                ["rm", "fake-container"],
            ],
        )

    def test_supervisor_stops_silent_container_at_budget(self):
        self.config["search"]["max_wall_seconds"] = 1
        run = self.prepare()
        with patch.dict(os.environ, {**self.fake_docker(), "FAKE_HANG": "1"}):
            result = launcher.supervise(
                run, launcher.load_bundle(run), "test-image", self.base / "env"
            )
        self.assertEqual(result, 124)
        self.assertTrue((self.base / "stopped").exists())
        self.assertEqual(
            json.loads((run / "supervisor.json").read_text())["last_outcome"],
            "budget_exhausted",
        )
        self.assert_cleanup_calls()

    def test_lost_attachment_stops_live_container(self):
        run = self.prepare()
        with (
            patch.dict(os.environ, {**self.fake_docker(), "FAKE_RUNNING": "1"}),
            self.assertRaisesRegex(ValueError, "still running"),
        ):
            launcher.supervise(
                run, launcher.load_bundle(run), "test-image", self.base / "env"
            )
        self.assertTrue((self.base / "stopped").exists())
        self.assertEqual(
            json.loads((run / "supervisor.json").read_text())["last_outcome"], "failed"
        )
        self.assert_cleanup_calls()

    def assert_signal_cleanup(self, number):
        run = self.prepare()
        previous = {
            launcher.signal.SIGINT: object(),
            launcher.signal.SIGTERM: object(),
        }

        def register(sig, handler):
            if sig == number and callable(handler):
                handler(sig, None)
            return previous[sig]

        with (
            patch.dict(os.environ, {**self.fake_docker(), "FAKE_HANG": "1"}),
            patch.object(launcher.signal, "signal", side_effect=register) as signals,
        ):
            result = launcher.supervise(
                run, launcher.load_bundle(run), "test-image", self.base / "env"
            )
        self.assertEqual(result, 130)
        state = json.loads((run / "supervisor.json").read_text())
        self.assertEqual(state["last_outcome"], "interrupted")
        self.assertIsNone(state["active"])
        self.assertEqual(
            signals.call_args_list[-2:],
            [call(sig, handler) for sig, handler in previous.items()],
        )
        self.assert_cleanup_calls()

    def test_sigint_stops_and_removes_only_owned_container(self):
        self.assert_signal_cleanup(launcher.signal.SIGINT)

    def test_sigterm_stops_and_removes_only_owned_container(self):
        self.assert_signal_cleanup(launcher.signal.SIGTERM)

    def test_stop_failure_keeps_container_identity_for_recovery(self):
        run = self.prepare()
        with (
            patch.dict(os.environ, {**self.fake_docker(), "FAKE_STOP_FAIL": "1"}),
            self.assertRaises(launcher.subprocess.CalledProcessError),
        ):
            launcher.supervise(
                run, launcher.load_bundle(run), "test-image", self.base / "env"
            )
        state = json.loads((run / "supervisor.json").read_text())
        self.assertEqual(state["last_outcome"], "cleanup_failed")
        self.assertEqual(state["active"]["container"], "fake-container")
        self.assertEqual(state["consumed_wall_seconds"], 0)
        self.assert_cleanup_calls(
            [
                ["stop", "--time", "5", "fake-container"],
                ["kill", "fake-container"],
            ]
        )

    def test_recovery_charges_unsettled_time(self):
        run = self.prepare()
        launcher.save(
            run / "supervisor.json",
            {
                "consumed_wall_seconds": 1,
                "image_id": "test-image",
                "active": {"container": "old-container", "started_at": time.time() - 2},
            },
        )
        with patch.dict(os.environ, self.fake_docker()):
            launcher.supervise(
                run, launcher.load_bundle(run), "test-image", self.base / "env"
            )
        state = json.loads((run / "supervisor.json").read_text())
        self.assertGreaterEqual(state["consumed_wall_seconds"], 3)
        calls = [
            json.loads(line)
            for line in (self.base / "calls.jsonl").read_text().splitlines()
        ]
        self.assertEqual(calls[0], ["stop", "--time", "5", "old-container"])
        self.assert_cleanup_calls(
            [
                ["stop", "--time", "5", "old-container"],
                ["rm", "old-container"],
                ["stop", "--time", "5", "fake-container"],
                ["rm", "fake-container"],
            ]
        )


if __name__ == "__main__":
    unittest.main()
