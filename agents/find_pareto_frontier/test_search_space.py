"""Executable recipe identity, composition and source-boundary checks."""

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import evaluate
import launcher
import search_space as space
import test_evaluate as fixtures
from baseline_store import identity


class RecipeTests(unittest.TestCase):
    def test_invalid_parameters_never_create_output(self):
        invalid = [
            {"unknown": True},
            {"routing_load": 1},
            {"small_chunk_tokens": True},
            {"wait_batch_threshold": 0},
            {"large_chunk_tokens": 2**32},
            {"small_chunk_tokens": 513},
            {"wait_s": 0},
            {"age_s": float("nan")},
            {"dwell_s": -1},
            {"pressure_low_s": 2, "pressure_high_s": 1},
            {"pressure_high_s": float("inf")},
            [],
        ]
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "absent"
            for recipe in invalid:
                with self.subTest(recipe=recipe), self.assertRaises(ValueError):
                    space.write_catalog(output, [recipe])
                self.assertFalse(output.exists())

    def test_integer_and_float_parameters_have_one_identity(self):
        first = space.validate_recipe({"age_s": 1, "dwell_s": -0.0})
        second = space.validate_recipe({"age_s": 1.0, "dwell_s": 0})
        self.assertEqual(identity(first), identity(second))
        self.assertEqual(space.render(first), space.render(second))

    def test_catalog_preserves_components_and_one_change_controls(self):
        recipes = space.default_recipes()
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "catalog"
            catalog = space.write_catalog(output, recipes)
            entries = {row["id"]: row for row in catalog["entries"]}
            self.assertEqual(len(entries), len(recipes))
            self.assertEqual(
                {
                    tuple(row["families"])
                    for row in entries.values()
                    if len(row["families"]) == 1
                },
                {(component,) for component in space.COMPONENTS},
            )
            self.assertTrue(any(not row["families"] for row in entries.values()))
            self.assertTrue(any(len(row["families"]) == 7 for row in entries.values()))
            for parent in entries.values():
                for child_id in parent["ablations"]:
                    child = entries[child_id]
                    changed = [
                        key
                        for key in parent["recipe"]
                        if parent["recipe"][key] != child["recipe"][key]
                    ]
                    self.assertEqual(len(changed), 1)
                    self.assertIn(changed[0], space.COMPONENTS)
                    self.assertTrue(parent["recipe"][changed[0]])
                    self.assertFalse(child["recipe"][changed[0]])
                    self.assertIn(parent["id"], child["parents"])
                for child_id in parent["static_controls"]:
                    recipe = entries[child_id]["recipe"]
                    self.assertEqual(
                        recipe["small_chunk_tokens"], recipe["large_chunk_tokens"]
                    )
                    self.assertEqual(space.families(recipe), parent["families"])
            original = (output / "catalog.json").read_bytes()
            with self.assertRaises(FileExistsError):
                space.write_catalog(output, recipes)
            self.assertEqual((output / "catalog.json").read_bytes(), original)
            copied = space.write_catalog(Path(temporary) / "second", recipes)
            self.assertEqual(copied, catalog)
            self.assertEqual(space.validate_catalog(output / "catalog.json"), catalog)

    def test_catalog_rejects_forged_family_source_and_control_relations(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "catalog"
            catalog = space.write_catalog(
                output, space.expand({"aged_short_prefill": True, "decode_fair": True})
            )
            manifest = output / "catalog.json"
            variants = []
            forged = copy.deepcopy(catalog)
            forged["entries"][0]["families"] = ["hysteresis"]
            variants.append(forged)
            forged = copy.deepcopy(catalog)
            forged["entries"][1]["ablations"] = []
            variants.append(forged)
            forged = copy.deepcopy(catalog)
            forged["entries"][0]["path"] = "../escape"
            variants.append(forged)
            for forged in variants:
                manifest.write_text(json.dumps(forged))
                with self.assertRaises(ValueError):
                    space.validate_catalog(manifest)
            manifest.write_text(json.dumps(catalog))
            policy = output / catalog["entries"][0]["path"] / "mod.rs"
            data = policy.read_bytes()
            policy.write_bytes(data + b"\n// Changed candidate.\n")
            with self.assertRaisesRegex(ValueError, "files changed"):
                space.validate_catalog(manifest)
            policy.write_bytes(data)
            policy.unlink()
            policy.symlink_to(output / catalog["entries"][1]["path"] / "mod.rs")
            with self.assertRaisesRegex(ValueError, "symlink"):
                space.validate_catalog(manifest)

    def test_expansion_keeps_nondefault_parameters(self):
        recipe = {
            "decode_fair": True,
            "pressure_chunks": True,
            "small_chunk_tokens": 7,
            "large_chunk_tokens": 111,
        }
        recipes = space.expand(recipe)
        self.assertTrue(any(not space.families(row) for row in recipes))
        self.assertTrue(
            any(
                row["small_chunk_tokens"] == row["large_chunk_tokens"] == 7
                for row in recipes
            )
        )
        self.assertTrue(
            any(
                row["small_chunk_tokens"] == row["large_chunk_tokens"] == 111
                for row in recipes
            )
        )
        self.assertTrue(all(row["small_chunk_tokens"] in (7, 111) for row in recipes))


class FrozenRecipeTests(unittest.TestCase):
    def setUp(self):
        fixtures.EvaluationTests.setUp(self)

    def test_every_generated_policy_passes_real_freezer_with_matching_identity(self):
        destination = self.experiment.source / evaluate.POLICY
        seen = set()
        for recipe in space.default_recipes():
            source = space.render(recipe)
            files = {}
            for name, text in source.items():
                (destination / name).write_text(text)
                files[name] = hashlib.sha256(text.encode()).hexdigest()
            frozen = self.experiment.freeze_candidate()
            self.assertEqual(
                frozen["id"], identity({"kind": "candidate", "files": files})
            )
            self.assertNotIn(frozen["id"], seen)
            seen.add(frozen["id"])

    def test_frozen_source_contains_generation_and_planning_tools(self):
        included = set(launcher.source_files(evaluate.ROOT))
        for name in (
            "examples/policies/composable.rs",
            "agents/find_pareto_frontier/search_space.py",
            "agents/find_pareto_frontier/search_plan.py",
        ):
            self.assertIn(Path(name), included)


if __name__ == "__main__":
    unittest.main()
