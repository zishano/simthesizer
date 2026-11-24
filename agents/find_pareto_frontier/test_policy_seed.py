"""Check the real adaptive seed against freezing and source-snapshot boundaries."""

import unittest
from pathlib import Path

import evaluate
import launcher
import test_evaluate as fixtures


class PolicySeedTests(unittest.TestCase):
    def setUp(self):
        # Reuse setup without inheriting or rediscovering the evaluator's tests.
        fixtures.EvaluationTests.setUp(self)

    def test_real_seed_passes_pure_contract_and_changes_candidate_identity(self):
        example = evaluate.ROOT / "examples/policies/adaptive.rs"
        source = example.read_bytes()
        candidate = self.experiment.source / evaluate.POLICY / "mod.rs"
        candidate.write_bytes(source)

        original = self.experiment.freeze_candidate()
        self.assertEqual(original["files"], {"mod.rs": launcher.digest(example)})
        self.assertEqual(original["kind"], "candidate")
        original_snapshot = (
            self.experiment.artifacts / "candidates" / original["id"] / "policy/mod.rs"
        )
        self.assertEqual(original_snapshot.read_bytes(), source)
        self.assertEqual(self.experiment.freeze_candidate(), original)

        candidate.write_bytes(source + b"\n// Distinct search hypothesis.\n")
        changed = self.experiment.freeze_candidate()
        self.assertNotEqual(changed["id"], original["id"])
        self.assertNotEqual(changed["files"], original["files"])
        self.assertEqual(original_snapshot.read_bytes(), source)
        candidate.write_bytes(source)
        self.assertEqual(self.experiment.freeze_candidate(), original)

    def test_actual_snapshot_roots_include_example_and_private_environment(self):
        included = set(launcher.source_files(evaluate.ROOT))
        for relative in (
            "examples/policies/adaptive.rs",
            "crates/simthesizer-core/src/policies/environment.rs",
            "agents/find_pareto_frontier/generate_workloads.py",
        ):
            with self.subTest(path=relative):
                path = Path(relative)
                self.assertIn(path, included)
                self.assertTrue((evaluate.ROOT / path).is_file())


if __name__ == "__main__":
    unittest.main()
