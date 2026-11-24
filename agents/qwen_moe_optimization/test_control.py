"""Controller permissions, untrusted requests and accounting without Docker."""

import hashlib
import json
import subprocess
import tempfile
import threading
import types
import unittest
from pathlib import Path
from unittest import mock

import control

REQUEST = "a" * 32
CONTAINER = "b" * 64


class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.run = Path(self.temporary.name) / "run"
        (self.run / "artifacts").mkdir(parents=True)
        self.manifest = {
            "network": "host",
            "source_inventory_sha256": {},
            "experiment": {
                "container": {"cpus": 4, "memory_mib": 4096, "pids_limit": 128},
                "search": {
                    "seed": 0,
                    "max_trials": 4,
                    "max_wall_seconds": 60,
                    "trial_wall_seconds": 10,
                },
                "verification": {"max_trials": 3, "max_wall_seconds": 15},
                "loads": {"arrival_rate_multipliers": [1.0]},
                "workloads": [
                    {
                        "id": "train",
                        "path": "/workspace/inputs/workload",
                        "split": "train",
                    }
                ],
            },
        }
        self.case = control.baseline_store.cases(self.manifest)[0]["id"]
        self.request = {
            "id": REQUEST,
            "case": self.case,
            "operation": "evaluate",
            "policy_kind": "candidate",
            "hypothesis": "Persistent group placement lowers queue pressure.",
            "bottleneck": "Attention group queue imbalance.",
        }
        self.baseline = {"accounting": {"attempts": 1, "wall_seconds": 5}}

    def test_trial_has_no_network_keys_worktree_or_docker_socket(self):
        job = self.run / "jobs/example"
        args = control.docker_arguments(
            self.run, self.manifest, "sha256:image", job=job
        )
        self.assertEqual(args[args.index("--network") + 1], "none")
        self.assertIn("--read-only", args)
        self.assertEqual(args[args.index("--cap-drop") + 1], "ALL")
        self.assertNotIn("--env-file", args)
        self.assertNotIn("--privileged", args)
        mounts = [
            args[index + 1] for index, value in enumerate(args) if value == "--mount"
        ]
        self.assertEqual(len(mounts), 6)
        for destination in ("simthesizer", "task", "inputs", "baseline-policy"):
            self.assertTrue(
                next(
                    row for row in mounts if f"target=/workspace/{destination}," in row
                ).endswith(",readonly")
            )
        self.assertFalse(
            next(
                row for row in mounts if "target=/workspace/artifacts" in row
            ).endswith(",readonly")
        )
        self.assertTrue(all(str(job) in row for row in mounts))
        self.assertFalse(
            any(
                "docker.sock" in row or "worktree" in row or "agent.env" in row
                for row in mounts
            )
        )

    def test_research_has_only_explicit_writable_research_mounts(self):
        args = control.docker_arguments(
            self.run, self.manifest, "image", env_file=self.run / "agent.env"
        )
        mounts = [
            args[index + 1] for index, value in enumerate(args) if value == "--mount"
        ]
        writable = {
            row.split("target=", 1)[1]
            for row in mounts
            if not row.endswith(",readonly")
        }
        self.assertEqual(
            writable,
            {
                "/workspace/cache",
                "/workspace/worktree",
                "/workspace/queue",
                "/workspace/notes",
            },
        )
        self.assertTrue(
            next(
                row for row in mounts if "target=/workspace/artifacts" in row
            ).endswith(",readonly")
        )
        with self.assertRaisesRegex(ValueError, "env file"):
            control.docker_arguments(self.run, self.manifest, "image")

    def test_request_types_and_unknown_commands_are_rejected_cleanly(self):
        self.assertEqual(
            control.validate_request(self.request, self.manifest, REQUEST), self.request
        )
        for field in (
            "id",
            "case",
            "operation",
            "policy_kind",
            "hypothesis",
            "bottleneck",
            "evidence",
            "ablation_of",
        ):
            for value in ([], {}, 3, None):
                with (
                    self.subTest(field=field, value=value),
                    self.assertRaises(ValueError),
                ):
                    control.validate_request(
                        {**self.request, field: value}, self.manifest, REQUEST
                    )
        for update in (
            {"command": "bash"},
            {"case": "../case"},
            {"candidate_id": "c" * 64},
            {"operation": "replay"},
            {"ablation_of": "../../other"},
            {"plan": []},
            {"plan": {"oversize": "x" * 32768}},
        ):
            with self.subTest(update=update), self.assertRaises(ValueError):
                control.validate_request(
                    {**self.request, **update}, self.manifest, REQUEST
                )
        replay = {**self.request, "operation": "replay", "candidate_id": "c" * 64}
        self.assertEqual(
            control.validate_request(replay, self.manifest, REQUEST), replay
        )
        with self.assertRaisesRegex(ValueError, "original sealed plan"):
            control.validate_request({**replay, "plan": {}}, self.manifest, REQUEST)

    def test_invalid_trials_and_baseline_consume_attempt_budget(self):
        for index in range(3):
            control.append(
                self.run,
                {
                    "event": "started",
                    "id": str(index),
                    "operation": "evaluate",
                    "budget_wall_seconds": 10,
                },
            )
            control.append(
                self.run,
                {
                    "event": "finished",
                    "id": str(index),
                    "status": "INVALID",
                    "wall_seconds": 1,
                },
            )
        with self.assertRaisesRegex(ValueError, "budget exhausted"):
            control.request_budget(self.run, self.manifest, "evaluate", self.baseline)

    def test_interrupted_verification_is_charged_reserved_budget(self):
        control.append(
            self.run,
            {
                "event": "started",
                "id": "unfinished",
                "operation": "replay",
                "budget_wall_seconds": 10,
            },
        )
        self.assertEqual(
            control.request_budget(self.run, self.manifest, "replay", self.baseline), 5
        )
        control.append(
            self.run,
            {
                "event": "started",
                "id": "second",
                "operation": "replay",
                "budget_wall_seconds": 5,
            },
        )
        control.append(
            self.run,
            {
                "event": "finished",
                "id": "second",
                "status": "INVALID",
                "wall_seconds": 5,
            },
        )
        with self.assertRaisesRegex(ValueError, "budget exhausted"):
            control.request_budget(self.run, self.manifest, "replay", self.baseline)

    def test_failed_snapshot_returns_charged_finished_record_without_execution(self):
        fake_runner = types.SimpleNamespace(
            prepare_job=mock.Mock(),
            collect_job=mock.Mock(),
            regular=mock.Mock(side_effect=FileNotFoundError),
        )
        containers = mock.Mock()
        with (
            mock.patch.dict("sys.modules", {"runner": fake_runner}),
            mock.patch.object(
                control.candidate,
                "snapshot",
                side_effect=ValueError("protected source changed"),
            ),
        ):
            result = control.evaluate_request(
                self.run,
                self.manifest,
                "image",
                self.request,
                containers,
                threading.Event(),
                1e12,
            )
        self.assertEqual(result["event"], "finished")
        self.assertEqual(result["status"], "INVALID")
        self.assertGreaterEqual(result["wall_seconds"], 0)
        self.assertIn("protected source", result["reason"])
        fake_runner.prepare_job.assert_not_called()
        containers.run_trial.assert_not_called()

    def mechanism_stubs(self, *, result_bytes=b"complete candidate result\n"):
        metadata = {
            "contract_version": control.candidate.CONTRACT_VERSION,
            "changed_files": ["crates/simthesizer-core/src/engine.rs"],
            "impact": control.candidate.impact(
                ["crates/simthesizer-core/src/engine.rs"]
            ),
            "source_sha256": "c" * 64,
            "mechanism_sha256": "e" * 64,
            "policy_sha256": "f" * 64,
            "kind": "mechanism",
        }

        def snapshot(_work, _source, destination, **_kwargs):
            destination.mkdir(parents=True)
            (destination / "candidate.json").write_text(json.dumps(metadata))
            return metadata.copy()

        def prepare(_run, candidate_dir, job, _case, **_kwargs):
            self.assertEqual(candidate_dir.name, metadata["source_sha256"])
            (job / "cache").mkdir(parents=True)

        def collect(_run, job, candidate_meta, case, **kwargs):
            self.assertEqual(candidate_meta, metadata)
            self.assertEqual(case, self.case)
            self.assertIn("deadline", kwargs)
            destination = job / "collected"
            destination.mkdir()
            (destination / "result.jsonl").write_bytes(result_bytes)
            (destination / "metrics.json").write_text('{"scored_by":"host"}\n')
            return {
                "status": "EVALUATED",
                "metrics": {"scored_by": "host"},
                "evidence_sha256": {
                    "result.jsonl": hashlib.sha256(result_bytes).hexdigest()
                },
                "evidence_directory": str(destination),
            }

        fake = types.SimpleNamespace(
            prepare_job=mock.Mock(side_effect=prepare),
            collect_job=mock.Mock(side_effect=collect),
            regular=mock.Mock(side_effect=FileNotFoundError),
        )
        return metadata, snapshot, fake

    def test_mechanism_without_plan_stops_before_container(self):
        _metadata, snapshot, fake = self.mechanism_stubs()
        containers = mock.Mock()
        with (
            mock.patch.dict("sys.modules", {"runner": fake}),
            mock.patch.object(control.candidate, "snapshot", side_effect=snapshot),
        ):
            result = control.evaluate_request(
                self.run,
                self.manifest,
                "image",
                self.request,
                containers,
                threading.Event(),
                1e12,
            )
        self.assertEqual(result["status"], "INVALID")
        self.assertIn("require --plan", result["reason"])
        containers.run_trial.assert_not_called()
        fake.prepare_job.assert_not_called()

    def test_mechanism_handoff_cache_and_published_host_results(self):
        metadata, snapshot, fake = self.mechanism_stubs()
        containers = mock.Mock()
        for index, policy_kind in enumerate(("candidate", "reference")):
            request = {
                **self.request,
                "id": str(index + 1) * 32,
                "policy_kind": policy_kind,
            }
            with (
                mock.patch.dict("sys.modules", {"runner": fake}),
                mock.patch.object(
                    control.candidate, "snapshot", side_effect=snapshot
                ) as freeze,
                mock.patch.object(
                    control.contract, "seal", return_value={"id": "sealed"}
                ) as seal,
                mock.patch.object(
                    control.contract, "compare", return_value={}
                ) as compare,
            ):
                result = control.evaluate_request(
                    self.run,
                    self.manifest,
                    "image",
                    request,
                    containers,
                    threading.Event(),
                    1e12,
                )
            self.assertEqual(result["status"], "EVALUATED")
            self.assertEqual(result["candidate_id"], metadata["source_sha256"])
            self.assertEqual(result["verification"], "EXPLORATORY")
            self.assertEqual(result["metrics"], {"scored_by": "host"})
            self.assertEqual(
                result["evidence_directory"],
                f"/workspace/artifacts/evaluations/{request['id']}",
            )
            self.assertEqual(
                (
                    self.run / "artifacts/evaluations" / request["id"] / "metrics.json"
                ).read_text(),
                '{"scored_by":"host"}\n',
            )
            freeze.assert_called_once()
            self.assertEqual(
                freeze.call_args.kwargs["expected_source_sha256"],
                self.manifest["source_inventory_sha256"],
            )
            prepared = fake.prepare_job.call_args
            self.assertEqual(prepared.kwargs["mechanism_contract"], {"id": "sealed"})
            self.assertEqual(
                fake.collect_job.call_args.kwargs["mechanism_contract"],
                {"id": "sealed"},
            )
            seal.assert_called_once()
            compare.assert_called_once()
            self.assertEqual(prepared.kwargs["policy_kind"], policy_kind)
            self.assertFalse(prepared.kwargs["clean"])
            cache = (
                self.run
                / "candidate-cache"
                / (metadata["source_sha256"] + "-" + policy_kind)
            )
            self.assertEqual(prepared.kwargs["cache_path"], cache)
            self.assertTrue(cache.is_dir())
            args = containers.run_trial.call_args.args[0]
            self.assertEqual("--reference" in args, policy_kind == "reference")
            self.assertNotIn("--clean", args)
            self.assertTrue(
                any(
                    str(cache) in value and "target=/workspace/cache" in value
                    for value in args
                )
            )

    def test_clean_replay_uses_sealed_candidate_empty_cache_and_rejects_mismatch(self):
        for index, matches in enumerate((True, False)):
            metadata, _snapshot, fake = self.mechanism_stubs(
                result_bytes=b"new complete result\n"
            )
            sealed = self.run / "artifacts/mechanisms" / metadata["source_sha256"]
            sealed.mkdir(parents=True, exist_ok=True)
            (sealed / "candidate.json").write_text(json.dumps(metadata))
            expected = hashlib.sha256(
                b"new complete result\n" if matches else b"different earlier result\n"
            ).hexdigest()
            previous = {
                "event": "finished",
                "id": "old" + str(index),
                "status": "EVALUATED",
                "candidate_id": metadata["source_sha256"],
                "case_id": self.case,
                "policy_kind": "candidate",
                "evidence_sha256": {"result.jsonl": expected},
                "metrics": {"scored_by": "host"},
                "contract": {"id": "sealed", "plan": {"schema_version": 1}},
            }
            control.append(self.run, previous)
            request = {
                **self.request,
                "id": str(index + 3) * 32,
                "operation": "replay",
                "candidate_id": metadata["source_sha256"],
            }
            containers = mock.Mock()
            with (
                mock.patch.dict("sys.modules", {"runner": fake}),
                mock.patch.object(control.candidate, "snapshot") as snapshot,
                mock.patch.object(
                    control.contract, "seal", return_value=previous["contract"]
                ) as seal,
                mock.patch.object(control.contract, "compare", return_value={}),
            ):
                result = control.evaluate_request(
                    self.run,
                    self.manifest,
                    "image",
                    request,
                    containers,
                    threading.Event(),
                    1e12,
                )
            snapshot.assert_not_called()
            self.assertEqual(seal.call_args.args[3], previous["contract"]["plan"])
            self.assertTrue(fake.prepare_job.call_args.kwargs["clean"])
            self.assertIsNone(fake.prepare_job.call_args.kwargs["cache_path"])
            args = containers.run_trial.call_args.args[0]
            self.assertIn("--clean", args)
            self.assertFalse(any("candidate-cache" in value for value in args))
            self.assertTrue(
                any(
                    str(self.run / "jobs" / request["id"] / "cache") in value
                    and "target=/workspace/cache" in value
                    for value in args
                )
            )
            self.assertEqual(result["status"], "EVALUATED" if matches else "INVALID")
            self.assertEqual(result["replay"], "MATCHED" if matches else "FAILED")
            if matches:
                self.assertEqual(result["replayed_request"], previous["id"])
            else:
                self.assertIn("clean replay result differs", result["reason"])

    def test_recovery_removes_only_labeled_recorded_container(self):
        manager = control.Containers(self.run, {"active": [CONTAINER]})
        label = hashlib.sha256(str(self.run.resolve()).encode()).hexdigest()
        inspection = subprocess.CompletedProcess(
            [], 0, json.dumps([{"Config": {"Labels": {control.LABEL: label}}}]), ""
        )
        removed = subprocess.CompletedProcess([], 0, "", "")
        discovered = subprocess.CompletedProcess([], 0, CONTAINER + "\n", "")
        with (
            mock.patch.object(
                control.subprocess, "run", side_effect=[discovered, inspection, removed]
            ) as command,
            mock.patch.object(control.infra, "stop_container") as stop,
        ):
            manager.recover()
        stop.assert_called_once_with(CONTAINER)
        self.assertEqual(
            command.call_args_list[1].args[0], ["docker", "inspect", CONTAINER]
        )
        self.assertEqual(command.call_args_list[2].args[0], ["docker", "rm", CONTAINER])
        self.assertEqual(
            command.call_args_list[0].args[0][-1], f"label={control.LABEL}={label}"
        )
        self.assertEqual(manager.state["active"], [])

    def test_recovery_never_stops_foreign_container_or_unavailable_daemon(self):
        for inspection in (
            subprocess.CompletedProcess(
                [],
                0,
                json.dumps([{"Config": {"Labels": {control.LABEL: "someone-else"}}}]),
                "",
            ),
            subprocess.CompletedProcess([], 1, "", "Cannot connect to Docker"),
        ):
            manager = control.Containers(self.run, {"active": [CONTAINER]})
            with (
                self.subTest(inspection=inspection),
                mock.patch.object(
                    control.subprocess,
                    "run",
                    side_effect=[
                        subprocess.CompletedProcess([], 0, "", ""),
                        inspection,
                    ],
                ),
                mock.patch.object(manager, "remove") as remove,
                self.assertRaises(ValueError),
            ):
                manager.recover()
            remove.assert_not_called()
            self.assertEqual(manager.state["active"], [CONTAINER])

    def test_recovery_drops_only_known_missing_id(self):
        manager = control.Containers(self.run, {"active": [CONTAINER]})
        missing = subprocess.CompletedProcess([], 1, "", "Error: No such object")
        with (
            mock.patch.object(
                control.subprocess,
                "run",
                side_effect=[subprocess.CompletedProcess([], 0, "", ""), missing],
            ),
            mock.patch.object(manager, "remove") as remove,
        ):
            manager.recover()
        remove.assert_not_called()
        self.assertEqual(manager.state["active"], [])

    def test_unrecorded_orphan_is_found_and_failed_cleanup_remains_recorded(self):
        manager = control.Containers(self.run, {"active": []})
        label = hashlib.sha256(str(self.run.resolve()).encode()).hexdigest()
        discovered = subprocess.CompletedProcess([], 0, CONTAINER + "\n", "")
        inspection = subprocess.CompletedProcess(
            [], 0, json.dumps([{"Config": {"Labels": {control.LABEL: label}}}]), ""
        )
        with (
            mock.patch.object(
                control.subprocess, "run", side_effect=[discovered, inspection]
            ),
            mock.patch.object(
                control.infra,
                "stop_container",
                side_effect=ValueError("daemon unavailable"),
            ),
            self.assertRaisesRegex(ValueError, "daemon unavailable"),
        ):
            manager.recover()
        self.assertEqual(manager.state["active"], [CONTAINER])
        self.assertEqual(
            json.loads((self.run / "supervisor.json").read_text())["active"],
            [CONTAINER],
        )

    def test_creation_failure_cleans_only_new_orphan_and_preserves_research(self):
        research = "d" * 64
        manager = control.Containers(self.run, {"active": [research]})
        label = hashlib.sha256(str(self.run.resolve()).encode()).hexdigest()
        discovered = subprocess.CompletedProcess(
            [], 0, research + "\n" + CONTAINER + "\n", ""
        )
        inspection = subprocess.CompletedProcess(
            [], 0, json.dumps([{"Config": {"Labels": {control.LABEL: label}}}]), ""
        )
        removed = subprocess.CompletedProcess([], 0, "", "")
        with (
            mock.patch.object(
                control.subprocess,
                "check_output",
                side_effect=subprocess.TimeoutExpired("docker create", 1),
            ),
            mock.patch.object(
                control.subprocess, "run", side_effect=[discovered, inspection, removed]
            ) as command,
            mock.patch.object(control.infra, "stop_container") as stop,
            self.assertRaises(subprocess.TimeoutExpired),
        ):
            manager.create(["docker", "create"])
        self.assertEqual(manager.state["active"], [research])
        stop.assert_called_once_with(CONTAINER)
        self.assertFalse(
            any(research in call.args[0] for call in command.call_args_list)
        )

    def test_create_respects_remaining_deadline_and_rejects_expired_budget(self):
        manager = control.Containers(self.run, {"active": []})
        with (
            mock.patch.object(control.time, "monotonic", return_value=10),
            mock.patch.object(
                control.subprocess, "check_output", return_value=CONTAINER
            ) as command,
        ):
            manager.create(["docker", "create"], deadline=12)
        self.assertEqual(command.call_args.kwargs["timeout"], 2)
        with (
            mock.patch.object(control.time, "monotonic", return_value=10),
            mock.patch.object(control.subprocess, "check_output") as command,
            self.assertRaises(TimeoutError),
        ):
            manager.create(["docker", "create"], deadline=10)
        command.assert_not_called()

    def test_timeout_always_cleans_its_trial_container(self):
        manager = control.Containers(self.run, {"active": []})
        process = mock.Mock()
        process.poll.return_value = None
        cancel = threading.Event()
        cancel.set()
        with (
            mock.patch.object(manager, "create", return_value=CONTAINER),
            mock.patch.object(manager, "remove") as remove,
            mock.patch.object(control.subprocess, "Popen", return_value=process),
            self.assertRaises(TimeoutError),
        ):
            manager.run_trial(
                ["docker", "create"], self.run / "trial.log", 1e12, cancel
            )
        remove.assert_called_once_with(CONTAINER)
        process.terminate.assert_called_once()
        process.wait.assert_called_once_with(timeout=5)

    def test_initialize_does_not_replace_existing_candidate_work(self):
        (self.run / "source").mkdir()
        (self.run / "source/module.rs").write_text("reference")
        control.initialize(self.run)
        (self.run / "worktree/module.rs").write_text("candidate")
        control.initialize(self.run)
        self.assertEqual((self.run / "worktree/module.rs").read_text(), "candidate")
        self.assertEqual((self.run / "source/module.rs").read_text(), "reference")


if __name__ == "__main__":
    unittest.main()
