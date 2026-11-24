"""Queue client checks in temporary directories; no agent, build or simulation."""

import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "find_pareto_frontier"))
import baseline_store
import candidate
import contract
import submit

REQUEST = "a" * 32


class SubmissionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="qwen-submit-test-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        for name in ("task", "cache", "queue", "artifacts/responses"):
            (self.root / name).mkdir(parents=True)
        self.manifest = {
            "source_inventory_sha256": {"source.rs": "frozen"},
            "experiment": {"search": {"max_wall_seconds": 20}},
        }
        (self.root / "task/manifest.json").write_text(json.dumps(self.manifest))
        (self.root / f"artifacts/responses/{REQUEST}.json").write_text(
            json.dumps({"status": "EVALUATED", "id": REQUEST})
        )
        self.output = io.StringIO()
        saved_path = sys.path[:]
        self.addCleanup(lambda: sys.path.__setitem__(slice(None), saved_path))

        def mapped_path(value):
            path = Path(value)
            if path.is_absolute() and path.is_relative_to("/workspace"):
                return self.root / path.relative_to("/workspace")
            return path

        self.mapped_path = mapped_path
        for patcher in (
            patch.object(submit, "Path", side_effect=mapped_path),
            patch.object(baseline_store, "cases", return_value=[{"id": "case"}]),
            patch.object(
                submit.uuid, "uuid4", return_value=SimpleNamespace(hex=REQUEST)
            ),
            patch("sys.stdout", self.output),
            patch("sys.stderr", self.output),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    def args(self):
        return [
            "--case",
            "case",
            "--hypothesis",
            "Move work",
            "--bottleneck",
            "Link pressure",
        ]

    def request(self):
        return json.loads((self.root / f"queue/{REQUEST}.json").read_text())

    def plan(self, value):
        path = self.root / "plan.json"
        path.write_text(json.dumps(value))
        return str(path)

    def assert_not_submitted(self):
        self.assertEqual(list((self.root / "queue").iterdir()), [])

    def test_scope_snapshots_only_and_does_not_enqueue_or_wait(self):
        preview = {
            "contract_version": candidate.CONTRACT_VERSION,
            "kind": "mechanism",
            "source_sha256": "b" * 64,
            "changed_files": ["crates/simthesizer-core/src/engine.rs"],
            "impact": {"domains": ["engine"]},
        }
        original = tempfile.TemporaryDirectory

        def local_temporary(**kwargs):
            kwargs["dir"] = self.mapped_path(kwargs["dir"])
            return original(**kwargs)

        with (
            patch.object(
                submit.tempfile, "TemporaryDirectory", side_effect=local_temporary
            ),
            patch.object(candidate, "snapshot", return_value=preview) as snapshot,
            patch.object(submit.time, "sleep") as sleep,
        ):
            self.assertEqual(submit.main(["--scope"]), 0)
        self.assertEqual(snapshot.call_args.args[0], self.root / "worktree")
        self.assertEqual(snapshot.call_args.args[1], self.root / "simthesizer")
        self.assertEqual(
            snapshot.call_args.kwargs["expected_source_sha256"],
            self.manifest["source_inventory_sha256"],
        )
        self.assertEqual(json.loads(self.output.getvalue()), preview)
        sleep.assert_not_called()
        self.assert_not_submitted()
        self.assertEqual(list((self.root / "cache").iterdir()), [])

    def test_plan_is_copied_into_atomic_queue_request(self):
        plan = {"schema_version": 1, "summary": "可审查机制", "checks": []}
        with patch.object(submit.time, "sleep") as sleep:
            self.assertEqual(submit.main([*self.args(), "--plan", self.plan(plan)]), 0)
        request = self.request()
        self.assertEqual(request["plan"], plan)
        self.assertEqual(request["operation"], "evaluate")
        self.assertIsNone(request["candidate_id"])
        self.assertFalse((self.root / f"queue/{REQUEST}.tmp").exists())
        sleep.assert_not_called()

    def test_replay_cannot_supply_a_different_plan(self):
        with self.assertRaises(SystemExit) as failure:
            submit.main([*self.args(), "--replay", "b" * 64, "--plan", self.plan({})])
        self.assertEqual(failure.exception.code, 2)
        self.assertIn("reuses the sealed plan", self.output.getvalue())
        self.assert_not_submitted()

    def test_replay_submits_identity_without_a_plan(self):
        self.assertEqual(
            submit.main([*self.args(), "--replay", "b" * 64, "--reference"]), 0
        )
        request = self.request()
        self.assertEqual(request["operation"], "replay")
        self.assertEqual(request["candidate_id"], "b" * 64)
        self.assertEqual(request["policy_kind"], "reference")
        self.assertNotIn("plan", request)

    def test_oversized_or_non_object_plan_does_not_enqueue(self):
        for value in ({"text": "x" * contract.MAX_PLAN_BYTES}, ["wrong shape"]):
            with self.subTest(value_type=type(value).__name__):
                with self.assertRaises(SystemExit) as failure:
                    submit.main([*self.args(), "--plan", self.plan(value)])
                self.assertEqual(failure.exception.code, 2)
                self.assert_not_submitted()

    def test_total_request_byte_limit_includes_non_ascii_evidence(self):
        with self.assertRaises(SystemExit) as failure:
            submit.main([*self.args(), "--evidence", "证" * 22000])
        self.assertEqual(failure.exception.code, 2)
        self.assertIn("request exceeds byte limit", self.output.getvalue())
        self.assert_not_submitted()

    def test_list_has_no_submission_side_effects(self):
        self.assertEqual(submit.main(["--list"]), 0)
        self.assertEqual(json.loads(self.output.getvalue()), [{"id": "case"}])
        self.assert_not_submitted()


if __name__ == "__main__":
    unittest.main()
