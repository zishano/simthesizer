"""Host lifecycle regressions using controlled Docker doubles, without API calls."""

import json
import signal
import tempfile
import unittest
from concurrent.futures import Future
from pathlib import Path
from unittest.mock import Mock, patch

import control


class ImmediateExecutor:
    def __init__(self, **kwargs):
        pass

    def submit(self, function, *args):
        future = Future()
        future.set_result(function(*args))
        return future

    def shutdown(self, **kwargs):
        pass


class SupervisionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.run = Path(temporary.name)
        for name in ("artifacts/responses", "queue"):
            (self.run / name).mkdir(parents=True)
        self.state = {"consumed_wall_seconds": 0, "active": []}
        control.save(self.run / "supervisor.json", self.state)
        self.manifest = {
            "experiment": {
                "research": {"workflow": "bottleneck-v1", "max_stalled_sessions": 3},
                "search": {
                    "max_wall_seconds": 100,
                    "max_trials": 5,
                    "trial_wall_seconds": 10,
                },
                "verification": {"max_trials": 2, "max_wall_seconds": 20},
            }
        }
        self.baseline = {
            "accounting": {"attempts": 1, "wall_seconds": 0},
            "records": [{}],
        }
        self.containers = Mock()

        def construct(run, state):
            self.state = state

            def update(**values):
                state.update(values)
                control.save(run / "supervisor.json", state)

            self.containers.update.side_effect = update
            self.containers.create.side_effect = [f"container-{i}" for i in range(10)]
            return self.containers

        self.handlers = {}

        def handler(number, callback):
            previous = self.handlers.get(number)
            self.handlers[number] = callback
            return previous

        self.process = Mock()
        self.process.poll.return_value = 0
        self.clock = 0
        patches = [
            patch.object(control, "Containers", side_effect=construct),
            patch.object(control, "docker_arguments", return_value=[]),
            patch.object(control, "baseline_diagnostics"),
            patch.object(control, "update_frontier"),
            patch.object(control.subprocess, "Popen", return_value=self.process),
            patch.object(
                control.subprocess,
                "check_output",
                return_value='{"Running":false,"ExitCode":0}',
            ),
            patch.object(control.signal, "signal", side_effect=handler),
            patch.object(control.time, "monotonic", side_effect=lambda: self.clock),
            patch.object(control.threading.Event, "wait", return_value=False),
            patch.object(
                control.concurrent.futures, "ThreadPoolExecutor", ImmediateExecutor
            ),
        ]
        self.mocks = [self.enterContext(item) for item in patches]

    def supervise(self):
        return control.supervise(
            self.run, self.manifest, "image", self.run / "env", self.baseline
        )

    def test_clean_empty_exits_continue_then_report_stalled_not_finished(self):
        self.assertEqual(self.supervise(), 3)
        self.assertEqual(self.containers.create.call_count, 3)
        self.assertEqual(self.containers.remove.call_count, 3)
        self.assertEqual(self.state["status"], "STALLED")
        self.assertEqual(self.state["stalled_sessions"], 3)
        self.assertIsNone(self.state["session_started"])

    def test_agent_exit_drains_queued_job_then_continues_with_preserved_progress(self):
        request = {"id": "a" * 32, "operation": "evaluate"}
        control.save(self.run / "queue" / (request["id"] + ".json"), request)

        def evaluate(run, manifest, image, submitted, containers, cancel, deadline):
            self.assertFalse(cancel.is_set())
            self.assertEqual(deadline, 10)
            return {
                **submitted,
                "event": "finished",
                "status": "EVALUATED",
                "wall_seconds": 2,
            }

        with (
            patch.object(
                control, "validate_request", side_effect=lambda value, *_: value
            ),
            patch.object(control, "evaluate_request", side_effect=evaluate),
        ):
            self.assertEqual(self.supervise(), 3)
        self.assertEqual(self.containers.create.call_count, 4)
        history = control.events(self.run)
        self.assertEqual([row["event"] for row in history], ["started", "finished"])
        response = json.loads(
            (self.run / "artifacts/responses" / (request["id"] + ".json")).read_text()
        )
        self.assertEqual(response["status"], "EVALUATED")
        status = json.loads((self.run / "artifacts/research-status.json").read_text())
        self.assertEqual(status["budget"]["evaluate_remaining"], 3)

    def test_rejected_requests_do_not_reset_idle_guard(self):
        request_id = "a" * 32
        control.save(self.run / "queue" / (request_id + ".json"), {})
        self.assertEqual(self.supervise(), 3)
        self.assertEqual(self.containers.create.call_count, 3)
        self.assertEqual(control.events(self.run)[0]["status"], "REJECTED")

    def test_invalid_plans_are_charged_but_do_not_reset_idle_guard(self):
        count = 0

        def launch(*args, **kwargs):
            nonlocal count
            count += 1
            request = {"id": f"{count:032x}", "operation": "evaluate"}
            control.save(self.run / "queue" / (request["id"] + ".json"), request)
            return self.process

        def evaluate(run, manifest, image, request, *args):
            return {
                **request,
                "event": "finished",
                "status": "INVALID",
                "wall_seconds": 1,
                "reason": "plan rejected before isolated execution",
            }

        self.mocks[4].side_effect = launch
        with (
            patch.object(
                control, "validate_request", side_effect=lambda value, *_: value
            ),
            patch.object(control, "evaluate_request", side_effect=evaluate),
        ):
            self.assertEqual(self.supervise(), 3)
        self.assertEqual(self.containers.create.call_count, 3)
        snapshot = json.loads((self.run / "artifacts/research-status.json").read_text())
        self.assertEqual(snapshot["started_attempts"], 3)
        self.assertEqual(snapshot["executed_attempts"], 0)
        self.assertEqual(snapshot["budget"]["evaluate_remaining"], 1)

    def test_pending_future_survives_agent_exit_until_result_is_available(self):
        request = {"id": "a" * 32, "operation": "evaluate"}
        control.save(self.run / "queue" / (request["id"] + ".json"), request)
        future = Mock()
        future.done.side_effect = [False, True]

        def result():
            self.assertEqual(future.done.call_count, 2)
            self.assertEqual(self.containers.create.call_count, 1)
            return {
                **request,
                "event": "finished",
                "status": "EVALUATED",
                "wall_seconds": 2,
            }

        future.result.side_effect = result
        executor = Mock()
        executor.submit.return_value = future
        with (
            patch.object(
                control, "validate_request", side_effect=lambda value, *_: value
            ),
            patch.object(
                control.concurrent.futures, "ThreadPoolExecutor", return_value=executor
            ),
        ):
            self.assertEqual(self.supervise(), 3)
        future.result.assert_called_once()
        self.assertEqual(self.containers.create.call_count, 4)
        executor.shutdown.assert_called_once_with(wait=True, cancel_futures=False)

    def test_evaluation_allowance_preserves_final_replay_time(self):
        self.manifest["experiment"]["search"]["max_wall_seconds"] = 15
        request = {"id": "a" * 32, "operation": "evaluate"}
        control.save(self.run / "queue" / (request["id"] + ".json"), request)

        def evaluate(run, manifest, image, submitted, containers, cancel, deadline):
            self.assertEqual(deadline, 5)
            return {
                **submitted,
                "event": "finished",
                "status": "INVALID",
                "wall_seconds": 1,
            }

        with (
            patch.object(
                control, "validate_request", side_effect=lambda value, *_: value
            ),
            patch.object(control, "evaluate_request", side_effect=evaluate),
        ):
            self.assertEqual(self.supervise(), 3)
        self.assertEqual(control.events(self.run)[0]["budget_wall_seconds"], 5)

    def test_deadline_stops_and_charges_elapsed_wall(self):
        def poll():
            self.clock = 101
            return 0

        self.process.poll.side_effect = poll
        self.assertEqual(self.supervise(), 124)
        self.assertEqual(self.state["status"], "BUDGET_EXHAUSTED")
        self.assertEqual(self.state["consumed_wall_seconds"], 101)
        self.assertEqual(self.containers.create.call_count, 1)

    def test_cancellation_cleans_container_without_relaunch(self):
        def launch(*args, **kwargs):
            self.handlers[signal.SIGTERM](None, None)
            return self.process

        self.mocks[4].side_effect = launch
        self.assertEqual(self.supervise(), 130)
        self.assertEqual(self.state["status"], "CANCELLED")
        self.assertEqual(self.containers.create.call_count, 1)
        self.containers.remove.assert_called_once()

    def test_old_workflow_retains_single_session_exit(self):
        self.manifest["experiment"]["research"] = {}
        self.assertEqual(self.supervise(), 0)
        self.assertEqual(self.state["status"], "FINISHED")
        self.assertEqual(self.containers.create.call_count, 1)

    def test_resume_retains_charged_interrupted_attempt_and_stall_count(self):
        self.state.update(stalled_sessions=2, consumed_wall_seconds=20)
        control.save(self.run / "supervisor.json", self.state)
        control.append(
            self.run,
            {
                "id": "a" * 32,
                "event": "started",
                "operation": "evaluate",
                "started_at": 0,
                "budget_wall_seconds": 10,
            },
        )
        self.assertEqual(self.supervise(), 3)
        self.assertEqual(self.containers.create.call_count, 1)
        self.assertEqual(control.events(self.run)[-1]["status"], "INVALID")
        self.assertEqual(self.state["consumed_wall_seconds"], 20)
        snapshot = json.loads((self.run / "artifacts/research-status.json").read_text())
        self.assertEqual(snapshot["budget"]["evaluate_remaining"], 3)


if __name__ == "__main__":
    unittest.main()
