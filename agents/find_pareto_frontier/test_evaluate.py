import copy
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import tomllib
import unittest
from pathlib import Path
from unittest.mock import patch

import archive
import evaluate
import launcher
from results import (
    Run,
    load_run,
    load_workload,
    measurements,
    parse_request,
    slo_metrics,
    validate_complete,
)


def request(identity=0, tokens=(1, 0.5), arrival=0):
    return {
        "request_id": identity,
        "arrival_time": arrival,
        "initial": [0],
        "sub_requests": [
            {
                "kind": "llm",
                "input_tokens": 2,
                "output_tokens": len(tokens),
                "next": [],
                "token_latency": list(tokens),
            }
        ],
    }


def measured(*rows, gpus=4):
    return Run(
        "test",
        Path("result.jsonl"),
        {row["request_id"]: parse_request(copy.deepcopy(row)) for row in rows},
        gpus,
    )


class ScoringTests(unittest.TestCase):
    def test_joint_slo_uses_complete_request_tokens_and_whole_service_time(self):
        run = measured(request(tokens=(0.5, 0.5, 0.5, 0.5)), request(1, (0.5, 1.5)))
        result = slo_metrics(
            run,
            {"ttft_limit_s": 1, "tpot_limit_s": 0.5, "min_joint_pass_fraction": 0.5},
        )
        self.assertEqual(result["eligible_requests"], 2)
        self.assertEqual(result["passed_requests"], 1)
        self.assertEqual(result["goodput"], 2)
        self.assertEqual(result["goodput_per_gpu"], 0.5)
        self.assertTrue(result["feasible"])

    def test_zero_output_excluded_but_still_required_and_one_token_tpot_absent(self):
        rows = [request(tokens=()), request(1, (2,), arrival=1)]
        run = measured(*rows)
        result = slo_metrics(
            run,
            {"ttft_limit_s": 2, "tpot_limit_s": 0.01, "min_joint_pass_fraction": 1},
            service_finish_s=3,
        )
        self.assertEqual(result["excluded_zero_output_requests"], 1)
        self.assertEqual(result["passed_requests"], 1)
        self.assertFalse(result["tpot_applicable"])
        self.assertEqual(validate_complete(run, rows)["completed_requests"], 2)
        with self.assertRaises(ValueError):
            validate_complete(measured(rows[1]), rows)

    def test_zero_output_tail_uses_native_window_for_all_throughput_metrics(self):
        run = measured(request(tokens=(1,)), request(1, tokens=(), arrival=1))
        slo = {"ttft_limit_s": 2, "tpot_limit_s": 1, "min_joint_pass_fraction": 1}
        with self.assertRaisesRegex(ValueError, "native service completion"):
            slo_metrics(run, slo)
        run.service_finish_s = 5
        self.assertEqual(slo_metrics(run, slo)["goodput"], 0.2)
        samples = measurements(run)
        self.assertEqual(samples["makespan"][()], 5)
        self.assertEqual(samples["output_throughput"][()], 0.2)
        self.assertEqual(samples["tokens_per_gpu"][()], 0.05)
        with self.assertRaisesRegex(ValueError, "precedes"):
            measurements(run, service_finish_s=0.5)

    def test_unscorable_run_and_changed_task_rejected(self):
        run = measured(request(tokens=(0,)), gpus=None)
        self.assertFalse(
            slo_metrics(
                run,
                {"ttft_limit_s": 1, "tpot_limit_s": 1, "min_joint_pass_fraction": 1},
            )["available"]
        )
        rows = [request()]
        other = measured(*rows)
        rows[0]["sub_requests"][0]["input_tokens"] += 1
        with self.assertRaisesRegex(ValueError, "task content"):
            validate_complete(other, rows)

    def test_partial_result_and_duplicate_input_fail(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "result.jsonl"
            row = request()
            row["sub_requests"][0]["output_tokens"] += 1
            path.write_text(json.dumps(row))
            with self.assertRaisesRegex(ValueError, "complete results"):
                load_run(str(path))
            path.write_text(json.dumps(request()) + "\n" + json.dumps(request()))
            with self.assertRaisesRegex(ValueError, "duplicate"):
                load_workload(path)

    def test_native_summary_is_optional_but_malformed_summary_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "result.jsonl"
            path.write_text(json.dumps(request()))
            self.assertIsNone(load_run(str(path)).service_finish_s)
            stats = path.with_name("policy-stats.json")
            stats.write_text("null")
            self.assertIsNone(load_run(str(path)).service_finish_s)
            stats.write_text('{"simulation_end_s": 5}')
            self.assertEqual(measurements(load_run(str(path)))["makespan"][()], 5)
            stats.write_text("[]")
            with self.assertRaisesRegex(ValueError, "object or null"):
                load_run(str(path))


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="sim-policy-eval-")
        self.addCleanup(self.temp.cleanup)
        self.run = Path(self.temp.name)
        for name in (
            "source",
            "inputs",
            "baseline-policy",
            "artifacts",
            "cache",
            "task",
        ):
            (self.run / name).mkdir()
        candidate = self.run / "source" / evaluate.POLICY
        candidate.mkdir(parents=True)
        (candidate / "mod.rs").write_text("pub fn choose() {}\n")
        (self.run / "baseline-policy/mod.rs").write_text("pub fn choose() {}\n")
        template = self.run / "source" / launcher.AGENT_REL
        template.mkdir(parents=True)
        (template / "AGENTS.md.template").write_text(
            "Frozen {{MANIFEST_SHA256}} {{POLICY_API_VERSION}}\n"
        )
        for relative, actual in evaluate.runtime_tool_paths().items():
            destination = self.run / "source" / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(actual, destination)
        config = tomllib.loads(
            (evaluate.ROOT / launcher.AGENT_REL / "experiment.toml").read_text()
        )
        config["policy"]["api_version"] = evaluate.API_VERSION
        config["inputs"]["scenario"] = "/workspace/inputs/scenario"
        config["workloads"] = [
            {"id": "fixture", "path": "/workspace/inputs/workload", "split": "train"}
        ]
        config["loads"]["arrival_rate_multipliers"] = [1.0, 2.0]
        config["search"].update(
            max_trials=3, max_wall_seconds=60, trial_wall_seconds=10
        )
        (self.run / "inputs/scenario").write_text(
            (
                evaluate.ROOT / launcher.AGENT_REL / "scenarios/shared-workload.toml"
            ).read_text()
        )
        self.rows = [request(), request(1, arrival=4)]
        for row in self.rows:
            row["sub_requests"][0].pop("token_latency")
        (self.run / "inputs/workload").write_text(
            "".join(json.dumps(row) + "\n" for row in self.rows)
        )
        manifest = {
            "schema_version": 1,
            "experiment": config,
            "source_sha256": launcher.inventory(self.run / "source", mutable=True),
            "inputs_sha256": launcher.inventory(self.run / "inputs"),
            "baseline_policy_sha256": launcher.inventory(self.run / "baseline-policy"),
            "build_versions": {"RUST_VERSION": "fixture"},
            "input_path_mapping": {},
        }
        launcher.save(self.run / "task/manifest.json", manifest)
        (self.run / "task/AGENTS.md").write_text(
            launcher.render_task(self.run / "source", manifest)
        )
        self.experiment = evaluate.Experiment(self.run / "task/manifest.json")

    def test_scaling_preserves_contents_and_input_bytes(self):
        before = launcher.digest(self.run / "inputs/workload")
        rows = self.experiment.derived(self.experiment.cases()[1])
        self.assertEqual(rows[1]["arrival_time"], 2)
        self.assertEqual(rows[1]["sub_requests"], self.rows[1]["sub_requests"])
        self.assertEqual(launcher.digest(self.run / "inputs/workload"), before)

    def test_close_arrival_multipliers_keep_distinct_case_identities(self):
        self.experiment.config["loads"]["arrival_rate_multipliers"] = [
            1.0,
            1.0000001,
            1.0000002,
        ]
        cases = self.experiment.cases()
        self.assertEqual(len({row["id"] for row in cases}), 3)

    def test_only_policy_and_worker_paths_change_configuration(self):
        candidate = self.experiment.freeze_candidate()
        config = self.experiment.configuration(candidate)
        self.assertEqual(config["system"]["policy"]["kind"], "candidate")
        self.assertEqual(
            config["system"]["kv_topology"],
            self.experiment.scenario["system"]["kv_topology"],
        )
        for pool in ("prefill_instances", "decode_instances"):
            self.assertEqual(
                config["system"][pool][0]["model_runner"]["hardware"],
                self.experiment.scenario["system"][pool][0]["model_runner"]["hardware"],
            )
            self.assertNotIn(
                "--reference-every-round",
                config["system"][pool][0]["model_runner"]["command"],
            )

    def test_failed_build_counts_towards_budget(self):
        self.experiment.config["search"]["max_trials"] = 1
        candidate = self.experiment.freeze_candidate()
        with patch.object(
            self.experiment, "build", side_effect=RuntimeError("compile failed")
        ):
            result = self.experiment.evaluate(candidate, self.experiment.cases()[0])
        self.assertEqual(result["status"], "INVALID")
        self.assertEqual(len(evaluate.ledger(self.experiment.book)), 2)
        with self.assertRaisesRegex(ValueError, "budget exhausted"):
            self.experiment.evaluate(candidate, self.experiment.cases()[1])

    def test_completed_identity_reuses_result_and_rejects_tampering(self):
        candidate = self.experiment.freeze_candidate()

        def process(command, **_kwargs):
            if "--output-jsonl" in command:
                workload = Path(command[command.index("--workload") + 1])
                result = Path(command[command.index("--output-jsonl") + 1])
                rows = [json.loads(line) for line in workload.read_text().splitlines()]
                for row in rows:
                    row["sub_requests"][0]["token_latency"] = [0.02] * row[
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
                return_value=(Path("/fake/simthesizer"), {"id": "test"}),
            ),
            patch.object(self.experiment, "capabilities", return_value={}),
            patch.object(evaluate, "run_process", side_effect=process) as run,
        ):
            first = self.experiment.evaluate(candidate, self.experiment.cases()[0])
            self.assertEqual(first["status"], "VALID_FEASIBLE", first)
            second = self.experiment.evaluate(candidate, self.experiment.cases()[0])
            self.assertTrue(second["reused"])
            self.assertEqual(run.call_count, 3)
            (
                self.experiment.artifacts / first["directory"] / "metrics.json"
            ).write_text("{}")
            with self.assertRaisesRegex(ValueError, "evidence changed"):
                self.experiment.evaluate(candidate, self.experiment.cases()[0])
            with patch.object(time, "monotonic", side_effect=[0, 0, 11, 11]):
                late = self.experiment.evaluate(
                    candidate, self.experiment.cases()[0], force=True
                )
            self.assertEqual(late["status"], "INCOMPLETE")
            self.assertIn("during validation", late["reason"])

    def test_frozen_input_and_symlink_candidate_rejected(self):
        (self.run / "inputs/workload").write_text("changed")
        with self.assertRaisesRegex(ValueError, "inputs_sha256"):
            self.experiment.verify_inputs()
        (self.run / "source" / evaluate.POLICY / "escape.rs").symlink_to("/etc/passwd")
        with self.assertRaisesRegex(ValueError, "symlink"):
            self.experiment.freeze_candidate()

    def test_running_tool_drift_cannot_reuse_frozen_identity(self):
        changed = self.run / "changed-results.py"
        changed.write_text("# different metric implementation\n")
        paths = evaluate.runtime_tool_paths()
        paths["scripts/results.py"] = changed
        with (
            patch.object(evaluate, "runtime_tool_paths", return_value=paths),
            self.assertRaisesRegex(ValueError, "executing tool differs"),
        ):
            evaluate.Experiment(self.run / "task/manifest.json")

    def test_native_capability_must_be_provided_not_guessed(self):
        with (
            patch.object(
                subprocess,
                "check_output",
                return_value=json.dumps({"policy_api_version": evaluate.API_VERSION}),
            ),
            self.assertRaisesRegex(ValueError, "does not implement"),
        ):
            self.experiment.capabilities(Path("/fake/simthesizer"))

    def test_build_cache_checks_binary_and_candidate_source_identity(self):
        candidate = self.experiment.freeze_candidate()

        def compile_fixture(_command, *, env, **_kwargs):
            target = Path(env["CARGO_TARGET_DIR"]) / "release/simthesizer"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"fixture native binary")

        with (
            patch.object(
                subprocess,
                "check_output",
                side_effect=lambda command, **_: command[0] + " fixture",
            ),
            patch.object(
                evaluate, "run_process", side_effect=compile_fixture
            ) as compile_native,
        ):
            binary, first = self.experiment.build(candidate, time.monotonic() + 10)
            _, second = self.experiment.build(candidate, time.monotonic() + 10)
            self.assertEqual(first, second)
            self.assertEqual(compile_native.call_count, 1)
            binary.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "cached native binary changed"):
                self.experiment.build(candidate, time.monotonic() + 10)

    def test_crashed_trial_is_charged_and_cannot_reset_trial_limit(self):
        candidate = self.experiment.freeze_candidate()
        self.experiment.config["search"]["max_trials"] = 1
        evaluate.append(
            self.experiment.book,
            {"event": "started", "attempt": 1, "started_at": time.time() - 1},
        )
        with self.assertRaisesRegex(ValueError, "budget exhausted"):
            self.experiment.evaluate(candidate, self.experiment.cases()[0])

    def test_timeout_kills_worker_process_group(self):
        pidfile = self.run / "pid"
        code = "import subprocess,sys,time; p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(10)']); open(sys.argv[1],'w').write(str(p.pid)); time.sleep(10)"
        with self.assertRaises(subprocess.TimeoutExpired):
            evaluate.run_process(
                [sys.executable, "-c", code, str(pidfile)],
                cwd=self.run,
                env=os.environ.copy(),
                log=self.run / "process.log",
                deadline=time.monotonic() + 0.2,
            )
        pid = int(pidfile.read_text())
        stat = Path(f"/proc/{pid}/stat")
        if stat.exists():
            self.assertIn(stat.read_text().split()[2], {"Z", "X"})


class FrontierTests(unittest.TestCase):
    def point(self, candidate, ttft, tpot, goodput, group="case1"):
        return {
            "event": "finished",
            "status": "VALID_FEASIBLE",
            "candidate_id": candidate,
            "group_id": group,
            "tpot_applicable": tpot is not None,
            "objectives": {
                "ttft_p99": ttft,
                "tpot_p99": tpot,
                "goodput_per_gpu": goodput,
            },
        }

    def test_tradeoffs_groups_and_equivalent_implementations(self):
        rows = [
            self.point("a", 1, 2, 3),
            self.point("b", 2, 1, 4),
            self.point("c", 3, 3, 2),
            self.point("d", 1, 2, 3),
            self.point("e", 4, None, 1, "case2"),
        ]
        result = archive.pareto(rows)
        self.assertEqual([p["candidate_id"] for p in result["case1"]], ["a", "b"])
        self.assertEqual(result["case1"][0]["equivalent_candidates"], ["d"])
        self.assertEqual(len(result["case2"]), 1)
        self.assertTrue(
            all(
                p["verification"] == "provisional"
                for points in result.values()
                for p in points
            )
        )


if __name__ == "__main__":
    unittest.main()
