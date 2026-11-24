"""Host accounting and continuation decisions, without containers or model calls."""

import copy
import unittest

import research


class ResearchTests(unittest.TestCase):
    def setUp(self):
        self.manifest = {
            "experiment": {
                "research": {"workflow": "bottleneck-v1"},
                "search": {"max_trials": 20, "trial_wall_seconds": 100},
                "verification": {"max_trials": 6, "max_wall_seconds": 900},
            }
        }
        self.baseline = {
            "accounting": {"attempts": 3},
            "records": [{"case": {"id": name}} for name in ("half", "one", "two")],
        }

    def snapshot(self, history=(), wall=1000):
        return research.status(self.manifest, self.baseline, history, wall)

    def start(self, name, operation="evaluate", budget=100):
        return {
            "event": "started",
            "id": name,
            "operation": operation,
            "budget_wall_seconds": budget,
        }

    def finish(
        self, name, operation="evaluate", status="EVALUATED", wall=40, family=None
    ):
        row = {
            "event": "finished",
            "id": name,
            "operation": operation,
            "status": status,
            "wall_seconds": wall,
            "case_id": "one",
        }
        if family is not None:
            row["contract"] = {
                "plan": {"research": {"family": family, "name": "Hypothesis " + name}}
            }
        return row

    def test_workflow_is_opt_in(self):
        self.assertTrue(research.enabled(self.manifest))
        self.assertFalse(research.enabled({"experiment": {}}))
        self.manifest["experiment"]["research"]["workflow"] = "legacy"
        self.assertFalse(research.enabled(self.manifest))

    def test_baseline_failed_and_inflight_attempts_all_consume_evaluation_budget(self):
        history = [
            self.start("failed"),
            self.finish("failed", status="INVALID"),
            self.start("active"),
            self.start("replay", "replay"),
        ]
        snapshot = self.snapshot(history)
        self.assertEqual(snapshot["budget"]["evaluate_remaining"], 15)
        self.assertEqual(snapshot["budget"]["replay_remaining"], 5)
        self.assertEqual(snapshot["started_attempts"], 3)
        self.assertEqual(snapshot["verification"], "EXPLORATORY")
        self.baseline["accounting"]["attempts"] = 25
        self.assertEqual(self.snapshot(history)["budget"]["evaluate_remaining"], 0)

    def test_inflight_replay_reserves_budget_then_completion_uses_actual_wall(self):
        history = [self.start("r1", "replay", budget=120)]
        pending = self.snapshot(history)["budget"]
        self.assertEqual(pending["replay_wall_remaining_s"], 780)
        self.assertEqual(pending["replay_remaining"], 5)
        history.append(self.finish("r1", "replay", wall=40))
        complete = self.snapshot(history)["budget"]
        self.assertEqual(complete["replay_wall_remaining_s"], 860)
        self.assertEqual(complete["replay_remaining"], 5)
        self.assertEqual(complete["full_matrix_replay_estimate_s"], 120)
        history.extend(
            [self.start("r2", "replay"), self.finish("r2", "replay", wall=60)]
        )
        two = self.snapshot(history)["budget"]
        self.assertEqual(two["replay_wall_remaining_s"], 800)
        self.assertEqual(two["full_matrix_replay_estimate_s"], 180)
        self.assertEqual(two["replay_estimate_source"], "maximum_completed_cold_replay")

    def test_failed_replay_is_charged_but_is_not_a_successful_cost_sample(self):
        history = [
            self.start("failed", "replay"),
            self.finish("failed", "replay", status="INVALID", wall=55),
        ]
        budget = self.snapshot(history)["budget"]
        self.assertEqual(budget["replay_wall_remaining_s"], 845)
        self.assertEqual(budget["replay_remaining"], 5)
        self.assertEqual(budget["full_matrix_replay_estimate_s"], 300)
        self.assertEqual(budget["replay_estimate_source"], "trial_timeout_upper_bound")

    def test_missing_cold_replay_samples_do_not_use_warm_evaluation_times(self):
        history = [self.start("warm"), self.finish("warm", wall=1)]
        budget = self.snapshot(history)["budget"]
        self.assertEqual(budget["final_replay_reserve_s"], 300)
        self.assertEqual(budget["full_matrix_replay_estimate_s"], 300)
        self.assertEqual(budget["replay_estimate_source"], "trial_timeout_upper_bound")
        self.assertTrue(budget["full_matrix_replay_fits"])

    def test_global_final_replay_reserve_stops_evaluation_but_allows_replay(self):
        for wall in (300, 299, 1):
            with self.subTest(wall=wall):
                snapshot = self.snapshot(wall=wall)
                self.assertFalse(snapshot["can_evaluate"])
                self.assertTrue(snapshot["can_replay"])
                self.assertEqual(snapshot["phase"], "validate")
                self.assertEqual(snapshot["budget"]["exploration_wall_remaining_s"], 0)
                self.assertEqual(
                    snapshot["budget"]["full_matrix_replay_fits"], wall >= 300
                )
        self.assertTrue(self.snapshot(wall=301)["can_evaluate"])
        for wall in (0, -10):
            with self.subTest(wall=wall):
                snapshot = self.snapshot(wall=wall)
                self.assertFalse(snapshot["can_evaluate"])
                self.assertFalse(snapshot["can_replay"])
                self.assertEqual(snapshot["phase"], "exhausted")
                self.assertEqual(snapshot["budget"]["wall_remaining_s"], 0)

    def test_full_matrix_needs_enough_replay_trials_and_wall(self):
        self.manifest["experiment"]["verification"]["max_trials"] = 2
        self.assertFalse(self.snapshot()["budget"]["full_matrix_replay_fits"])
        self.manifest["experiment"]["verification"]["max_trials"] = 6
        self.manifest["experiment"]["verification"]["max_wall_seconds"] = 250
        budget = self.snapshot()["budget"]
        self.assertFalse(budget["full_matrix_replay_fits"])
        self.assertEqual(budget["final_replay_reserve_s"], 250)

    def test_coverage_uses_finished_sealed_plans_and_accepts_new_families(self):
        history = [
            self.finish("good", family="batch_decode"),
            self.finish("bad", status="INVALID", family="batch_decode"),
            self.finish("new", family="new_joint_direction"),
            self.finish("repeat", "replay", family="kv_communication"),
            {**self.finish("unsealed"), "research": {"family": "routing_placement"}},
            {
                **self.finish("plan-only"),
                "plan": {"research": {"family": "execution_resources"}},
            },
            {
                **self.start("pending"),
                "contract": {
                    "plan": {
                        "research": {"family": "kv_communication", "name": "Pending"}
                    }
                },
            },
        ]
        snapshot = self.snapshot(history)
        self.assertEqual(
            set(snapshot["coverage"]), {"batch_decode", "new_joint_direction"}
        )
        self.assertEqual(snapshot["coverage"]["batch_decode"]["attempted"], 2)
        self.assertEqual(snapshot["coverage"]["batch_decode"]["evaluated"], 1)
        self.assertEqual(
            snapshot["unexplored_suggested_families"],
            ["routing_placement", "kv_communication", "execution_resources"],
        )
        history.extend(
            self.finish(str(index), family="batch_decode") for index in range(10)
        )
        recent = self.snapshot(history)["coverage"]["batch_decode"]["recent"]
        self.assertEqual(
            [item["request_id"] for item in recent], list(map(str, range(2, 10)))
        )

    def test_no_progress_stalls_and_caller_progress_reset_allows_continuation(self):
        snapshot = self.snapshot()
        self.assertEqual(research.next_session(snapshot, 0, 2), "CONTINUE")
        self.assertEqual(research.next_session(snapshot, 1, 2), "CONTINUE")
        self.assertEqual(research.next_session(snapshot, 2, 2), "STALLED")
        self.assertEqual(research.next_session(snapshot, 0, 2), "CONTINUE")
        self.assertEqual(
            research.next_session(self.snapshot(wall=0), 2, 2), "BUDGET_EXHAUSTED"
        )

    def test_resume_and_next_session_do_not_reset_or_double_charge_budgets(self):
        history = [
            self.start("failed"),
            self.finish("failed", status="INVALID"),
            self.start("replay", "replay"),
            self.finish("replay", "replay", wall=45),
        ]
        before = copy.deepcopy((self.manifest, self.baseline, history))
        first = self.snapshot(history, wall=600)
        research.next_session(first, 0, 2)
        resumed = self.snapshot(history, wall=550)
        for name in (
            "evaluate_remaining",
            "replay_remaining",
            "replay_wall_remaining_s",
        ):
            self.assertEqual(first["budget"][name], resumed["budget"][name])
        self.assertEqual(resumed["budget"]["evaluate_remaining"], 16)
        self.assertEqual(resumed["budget"]["replay_wall_remaining_s"], 855)
        self.assertEqual(resumed["budget"]["wall_remaining_s"], 550)
        self.assertEqual((self.manifest, self.baseline, history), before)
        history.extend([self.start("next"), self.finish("next", status="INVALID")])
        self.assertEqual(
            self.snapshot(history, wall=500)["budget"]["evaluate_remaining"], 15
        )


if __name__ == "__main__":
    unittest.main()
