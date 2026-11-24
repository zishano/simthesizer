"""Candidate scope, identities and filesystem boundaries; no simulator execution."""

import hashlib
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

import candidate

ROOT = Path(__file__).resolve().parents[2]
POLICY = candidate.POLICY_ROOT + "/candidate/mod.rs"
SCHEDULER = "crates/simthesizer-core/src/schedulers/chunked_prefill.rs"
PROTECTED = "crates/simthesizer-core/src/models/qwen3_235b_a22b.toml"
RESOURCES = "crates/simthesizer-core/src/resources.rs"
ENGINE = "crates/simthesizer-core/src/engine.rs"
SIMULATOR = "scripts/rooflang/runtime/simulator.py"


class CandidateTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.frozen = self.root / "frozen"
        self.work = self.root / "work"
        self.destination = self.root / "snapshot"
        source = {
            candidate.QWEN: (ROOT / candidate.QWEN).read_bytes(),
            POLICY: b"pub fn choose() -> usize { 0 }\n",
            candidate.POLICY_ROOT
            + "/mod.rs": b"pub fn reference_route() -> usize { 0 }\n",
            candidate.POLICY_ROOT + "/environment.rs": b"pub struct Observation;\n",
            SCHEDULER: b"pub fn schedule() {}\n",
            PROTECTED: b"hidden_size = 4096\n",
            RESOURCES: b"pub fn solve() {}\n",
            ENGINE: b"pub fn step() {}\n",
            "Cargo.lock": b"version = 4\n",
            "scripts/rooflang_bridge/hardware.py": b"CAPACITY = 144\n",
            "scripts/rooflang_bridge/llama.py": (
                ROOT / "scripts/rooflang_bridge/llama.py"
            ).read_bytes(),
            SIMULATOR: (ROOT / SIMULATOR).read_bytes(),
            "scripts/rooflang/language/tensor.py": (
                ROOT / "scripts/rooflang/language/tensor.py"
            ).read_bytes(),
            "scripts/rooflang/language/utils.py": (
                ROOT / "scripts/rooflang/language/utils.py"
            ).read_bytes(),
            "scripts/rooflang/language/kernels/kernel.py": (
                ROOT / "scripts/rooflang/language/kernels/kernel.py"
            ).read_bytes(),
            "tests/test_trusted.py": b"def test_invariant(): assert True\n",
            "agents/qwen_moe_optimization/candidate.py": b"TRUSTED = True\n",
            "scripts/rooflang_worker.py": b"def dispatch(): pass\n",
            "crates/simthesizer-core/src/kv.rs": b"pub struct Plan;\n",
            "crates/simthesizer-core/src/systems/pd_disaggregation.rs": b"pub fn placement() {}\n",
            "crates/simthesizer-core/src/model_runners/rooflang.rs": b"pub fn graph() {}\n",
            ".python-version": b"3.14\n",
        }
        for name, data in source.items():
            self.write(self.frozen, name, data)
        shutil.copytree(self.frozen, self.work)
        self.inventory = {
            name: hashlib.sha256(data).hexdigest() for name, data in source.items()
        }

    def write(self, root, name, data):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data if isinstance(data, bytes) else data.encode())

    def freeze(self, **kwargs):
        return candidate.snapshot(
            self.work,
            self.frozen,
            self.destination,
            expected_source_sha256=self.inventory,
            **kwargs,
        )

    def test_reference_round_trip_and_inventory_identity(self):
        result = self.freeze()
        self.assertEqual(result["kind"], "reference")
        self.assertEqual(result["contract_version"], candidate.CONTRACT_VERSION)
        self.assertEqual(result["schema_version"], 2)
        self.assertEqual(result["impact"]["domains"], [])
        self.assertEqual(result["files"], self.inventory)
        self.assertEqual(
            result["reference_source_sha256"], candidate.identity(self.inventory)
        )
        self.assertEqual(result["source_sha256"], candidate.identity(self.inventory))
        self.assertEqual(result["changed_files"], [])
        self.assertFalse(result["reference_policy_changed"])
        self.assertEqual(
            result, json.loads((self.destination / "candidate.json").read_text())
        )
        self.assertEqual((self.destination / "changes.diff").read_text(), "")
        for name in self.inventory:
            self.assertEqual(
                (self.destination / "source" / name).read_bytes(),
                (self.frozen / name).read_bytes(),
            )

    def test_policy_and_mechanism_identities_are_separate(self):
        baseline = self.freeze()
        self.destination = self.root / "policy"
        self.write(self.work, POLICY, "pub fn choose() -> usize { 1 }\n")
        policy = self.freeze()
        self.assertEqual(policy["kind"], "policy")
        self.assertEqual(policy["mechanism_sha256"], baseline["mechanism_sha256"])
        self.assertNotEqual(policy["policy_sha256"], baseline["policy_sha256"])
        self.destination = self.root / "mechanism"
        self.write(self.work, SCHEDULER, "pub fn schedule() { /* new mechanism */ }\n")
        mechanism = self.freeze()
        self.assertEqual(mechanism["kind"], "mechanism")
        self.assertNotEqual(mechanism["mechanism_sha256"], policy["mechanism_sha256"])
        self.assertEqual(mechanism["policy_sha256"], policy["policy_sha256"])
        self.assertEqual(mechanism["validation_status"], "static_scope_only")

    def test_policy_interface_is_mechanism_and_reference_mode_keeps_its_identity(self):
        baseline = self.freeze()
        self.destination = self.root / "interface"
        self.write(
            self.work,
            candidate.POLICY_ROOT + "/environment.rs",
            "pub struct Observation { pub pressure: u64 }\n",
        )
        interface = self.freeze()
        self.assertEqual(interface["kind"], "mechanism")
        self.assertNotEqual(interface["mechanism_sha256"], baseline["mechanism_sha256"])
        self.assertEqual(interface["policy_sha256"], baseline["policy_sha256"])
        self.assertFalse(interface["reference_policy_changed"])
        self.destination = self.root / "candidate-policy"
        self.write(
            self.work,
            candidate.POLICY_ROOT + "/mod.rs",
            "pub fn reference_route() -> usize { 1 }\n",
        )
        self.write(self.work, POLICY, "pub fn choose() -> usize { 2 }\n")
        customized = self.freeze()
        self.assertTrue(customized["reference_policy_changed"])
        self.assertNotEqual(
            customized["mechanism_sha256"], interface["mechanism_sha256"]
        )
        self.destination = self.root / "reference-policy"
        self.write(self.work, POLICY, (self.frozen / POLICY).read_bytes())
        restored = self.freeze()
        self.assertEqual(restored["kind"], "mechanism")
        self.assertTrue(restored["reference_policy_changed"])
        self.assertEqual(restored["mechanism_sha256"], customized["mechanism_sha256"])
        self.assertEqual(restored["policy_sha256"], baseline["policy_sha256"])
        self.assertNotEqual(restored["source_sha256"], customized["source_sha256"])

    def test_graph_mechanism_and_new_modules_are_really_editable(self):
        qwen = (self.work / candidate.QWEN).read_text()
        self.write(
            self.work,
            candidate.QWEN,
            qwen.replace(
                "source = (request % (ep // tp)) * tp",
                "source = ((request + 1) % (ep // tp)) * tp",
            ),
        )
        self.write(
            self.work,
            "scripts/rooflang_bridge/attention_placement.py",
            "def place(request, groups): return request % groups\n",
        )
        self.write(
            self.work,
            "tests/test_new_mechanism.py",
            "def test_placement(): assert 1 % 2 == 1\n",
        )
        result = self.freeze()
        self.assertEqual(result["kind"], "mechanism")
        self.assertEqual(len(result["changed_files"]), 3)
        self.assertIn("+def place", (self.destination / "changes.diff").read_text())

    def test_mutable_module_deletion_is_recorded(self):
        (self.work / SCHEDULER).unlink()
        result = self.freeze()
        self.assertEqual(result["changes"][0]["status"], "deleted")
        self.assertNotIn(SCHEDULER, result["files"])
        self.assertFalse((self.destination / "source" / SCHEDULER).exists())

    def test_persistent_attention_placement_state_and_protocol_are_editable(self):
        source = (self.work / candidate.QWEN).read_text()
        self.write(
            self.work,
            candidate.QWEN,
            source.replace(
                "self.attention_groups = groups",
                "self.attention_groups = groups\n        self.request_placement = {}",
            ),
        )
        for name in (
            "scripts/rooflang_worker.py",
            "crates/simthesizer-core/src/kv.rs",
            "crates/simthesizer-core/src/systems/pd_disaggregation.rs",
            "crates/simthesizer-core/src/model_runners/rooflang.rs",
        ):
            self.write(self.work, name, (self.work / name).read_bytes() + b"\n")
        result = self.freeze()
        self.assertEqual(result["kind"], "mechanism")
        self.assertEqual(len(result["changed_files"]), 5)

    def test_known_caches_are_excluded_but_notes_and_hidden_source_are_not(self):
        caches = {
            "scripts/rooflang_bridge/__pycache__/qwen3.cpython-314.pyc": b"bytecode",
            ".pytest_cache/v/cache/nodeids": b"[]",
            ".ruff_cache/version": b"1",
            ".git/config": b"[core]\nrepositoryformatversion=0\n",
        }
        for name, data in caches.items():
            self.write(self.work, name, data)
        result = self.freeze()
        self.assertEqual(result["kind"], "reference")
        self.assertEqual(result["files"], self.inventory)
        self.assertEqual(set(result["ignored_files"]), set(caches))
        self.destination = self.root / "next"
        self.write(
            self.work, "notes.md", "Put research notes outside the source tree.\n"
        )
        with self.assertRaisesRegex(ValueError, "protected candidate source"):
            self.freeze()
        (self.work / "notes.md").unlink()
        self.write(self.work, ".pytest_cache/hidden.py", "print('not metadata')\n")
        with self.assertRaisesRegex(ValueError, "source code inside ignored metadata"):
            self.freeze()
        (self.work / ".pytest_cache/hidden.py").unlink()
        (self.work / ".pytest_cache/escape").symlink_to(self.frozen / POLICY)
        with self.assertRaisesRegex(ValueError, "symlink"):
            self.freeze()

    def test_protected_code_tests_tooling_and_unknown_paths_rejected(self):
        for name in (
            PROTECTED,
            "Cargo.lock",
            "crates/simthesizer-core/src/models/qwen3_moe.rs",
            "crates/simthesizer-cli/src/main.rs",
            "crates/simthesizer-cli/src/write_jsonl.rs",
            "crates/simthesizer-cli/src/progress.rs",
            "scripts/rooflang/programs/presets/h200.py",
            "tests/test_trusted.py",
            "agents/qwen_moe_optimization/candidate.py",
            "scripts/results.py",
            "scripts/draw.py",
            "crates/simthesizer-core/tests/__qwen_frozen_resources.rs",
            "new_root.py",
        ):
            with self.subTest(path=name):
                existed = (self.work / name).exists()
                before = (self.work / name).read_bytes() if existed else None
                self.write(self.work, name, "CHANGED = True\n")
                with self.assertRaisesRegex(ValueError, "protected candidate source"):
                    self.freeze()
                if existed:
                    self.write(self.work, name, before)
                else:
                    (self.work / name).unlink()
                self.assertFalse(self.destination.exists())
        (self.work / PROTECTED).unlink()
        with self.assertRaisesRegex(ValueError, "protected candidate source"):
            self.freeze()

    def test_cross_module_implementation_interfaces_and_new_modules_are_open(self):
        paths = (
            ENGINE,
            RESOURCES,
            "crates/simthesizer-core/src/types.rs",
            "crates/simthesizer-core/src/lib.rs",
            "crates/simthesizer-core/src/registry.rs",
            "crates/simthesizer-core/src/module.rs",
            "crates/simthesizer-core/src/model_runners/model_runner.rs",
            "crates/simthesizer-core/src/models/mod.rs",
            "crates/simthesizer-core/src/models/model.rs",
            "crates/simthesizer-core/src/models/python.rs",
            "crates/simthesizer-core/src/new_mechanism/dependencies.rs",
            "crates/simthesizer-core/src/network_sims/flow.rs",
            "scripts/rooflang_bridge/execution.py",
            "scripts/rooflang_bridge/configuration.py",
            "scripts/rooflang_bridge/hardware.py",
            "scripts/rooflang_bridge/topology.py",
            "scripts/rooflang_bridge/observations.py",
            "scripts/rooflang_bridge/__init__.py",
            "scripts/rooflang/__init__.py",
            "scripts/rooflang/programs/__init__.py",
            "scripts/rooflang/programs/novel_pipeline.py",
            "scripts/rooflang/compiler/dependencies.py",
            "scripts/rooflang/language/new_mechanism/__init__.py",
            "scripts/rooflang/language/new_mechanism/operation.py",
            "scripts/rooflang/runtime/execution.py",
            "scripts/rooflang_model.py",
        )
        for name in paths:
            self.assertTrue(candidate.editable(name), name)
            self.write(
                self.work,
                name,
                "// mechanism\n" if name.endswith(".rs") else "# mechanism\n",
            )
        result = self.freeze()
        self.assertEqual(result["kind"], "mechanism")
        self.assertEqual(result["changed_files"], sorted(paths))
        self.assertTrue(result["impact"]["cross_module"])
        self.assertTrue(result["impact"]["core_changed"])
        self.assertEqual(set(result["impact"]["domains"]), candidate.DOMAINS)

    def test_impact_is_computed_from_paths_and_unknown_modules_are_conservative(self):
        engine = candidate.impact([ENGINE])
        self.assertEqual(engine["domains"], ["engine"])
        self.assertEqual(
            set(engine["required_checks"]),
            {
                "baseline_compatibility",
                "work_conservation",
                "resource_bounds",
                "causal_dependencies",
            },
        )
        novel = candidate.impact(["crates/simthesizer-core/src/novel.rs"])
        self.assertEqual(set(novel["domains"]), candidate.DOMAINS)
        self.assertEqual(
            set(novel["required_checks"]),
            {
                "baseline_compatibility",
                "work_conservation",
                "resource_bounds",
                "causal_dependencies",
                "kv_lifecycle",
                "interface_compatibility",
                "observation_neutrality",
                "policy_contract",
            },
        )
        for name in (
            "scripts/rooflang_bridge/hardware.py",
            "scripts/rooflang_bridge/topology.py",
        ):
            self.assertEqual(
                set(candidate.impact([name])["domains"]), candidate.DOMAINS
            )
        self.assertEqual(candidate.impact([POLICY])["domains"], ["policy"])
        self.assertEqual(candidate.impact(["tests/test_new.py"])["domains"], [])

    def test_pricing_runtime_can_extend_but_existing_cost_facts_stay_fixed(self):
        original = (self.work / SIMULATOR).read_text()
        self.write(self.work, SIMULATOR, original + "\ndef new_schedule(): return []\n")
        result = self.freeze()
        self.assertIn("engine", result["impact"]["domains"])
        self.assertIn(SIMULATOR, result["protected_fact_files"])
        self.destination = self.root / "changed-pricing"
        self.write(
            self.work,
            SIMULATOR,
            original.replace(
                "compute_time += flops / peak", "compute_time += flops / (peak * 2)"
            ),
        )
        with self.assertRaisesRegex(ValueError, "protected pricing/model AST"):
            self.freeze()

    def test_dtype_tensor_kernel_and_model_facts_are_guarded(self):
        replacements = (
            ("scripts/rooflang/language/utils.py", '"fp16": 2.0', '"fp16": 1.0'),
            (
                "scripts/rooflang/language/tensor.py",
                "return self.n_elements * dtype_bytes(self.dtype)",
                "return 0",
            ),
            (
                "scripts/rooflang/language/kernels/kernel.py",
                "return sum(t.size_bytes for t in self.weights.values())",
                "return 0",
            ),
            (
                "scripts/rooflang_bridge/llama.py",
                "4096, 14336, 32, 8, 128, 128256, 32",
                "4096, 1, 32, 8, 128, 128256, 32",
            ),
        )
        for name, before, after in replacements:
            with self.subTest(path=name):
                original = (self.work / name).read_text()
                self.assertIn(before, original)
                self.write(self.work, name, original.replace(before, after))
                with self.assertRaisesRegex(ValueError, "protected pricing/model AST"):
                    self.freeze()
                self.write(self.work, name, original)

    def test_protected_ast_deletion_and_shadowing_rejected(self):
        path = self.work / SIMULATOR
        original = path.read_text()
        path.unlink()
        with self.assertRaisesRegex(
            ValueError, "protected pricing/model source deleted"
        ):
            self.freeze()
        self.write(self.work, SIMULATOR, original + "\nSimulator = object\n")
        with self.assertRaisesRegex(ValueError, "missing or shadowed"):
            self.freeze()

    def test_fixed_qwen_constants_routing_and_pricing_rejected(self):
        original = (self.work / candidate.QWEN).read_text()
        changes = (
            ('H = MODEL["hidden_size"]', "H = 1"),
            ("digest_size=16", "digest_size=8"),
            ("if type(tp) is not int", "if False and type(tp) is not int"),
            ("read_us = self.pricing._base_times(read, src, [])[1]", "read_us = 0"),
            (
                "kernel.weight_read_fraction = read_fraction",
                "kernel.weight_read_fraction = 0",
            ),
        )
        for before, after in changes:
            with self.subTest(change=before):
                self.assertIn(before, original)
                self.write(self.work, candidate.QWEN, original.replace(before, after))
                with self.assertRaisesRegex(ValueError, "protected Qwen AST"):
                    self.freeze()
        self.write(
            self.work, candidate.QWEN, original + "\ntoken_experts = lambda *args: []\n"
        )
        with self.assertRaisesRegex(ValueError, "shadowed"):
            self.freeze()

    def test_frozen_identity_change_is_rejected(self):
        self.write(self.frozen, PROTECTED, "pub fn fake_solve() {}\n")
        with self.assertRaisesRegex(ValueError, "frozen source identity"):
            self.freeze()

    def test_symlink_and_hardlink_are_rejected_even_if_content_matches(self):
        path = self.work / POLICY
        path.unlink()
        path.symlink_to(self.frozen / POLICY)
        with self.assertRaisesRegex(ValueError, "symlink"):
            self.freeze()
        path.unlink()
        os.link(self.frozen / POLICY, path)
        with self.assertRaisesRegex(ValueError, "hardlink"):
            self.freeze()

    def test_symlink_directories_special_files_and_modes_are_rejected(self):
        alias = self.root / "alias"
        alias.symlink_to(self.work, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "symlinks"):
            candidate.snapshot(alias, self.frozen, self.destination)
        pipe = self.work / "tests/pipe.py"
        os.mkfifo(pipe)
        with self.assertRaisesRegex(ValueError, "special file"):
            self.freeze()
        pipe.unlink()
        path = self.work / POLICY
        path.chmod(0o755)
        with self.assertRaisesRegex(ValueError, "mode changed"):
            self.freeze()

    def test_bounded_sources_and_separate_exclusive_destination(self):
        for limits in (
            {"max_files": 1},
            {"max_bytes": 1},
            {"max_file_bytes": 1},
            {"max_files": 0},
        ):
            with (
                self.subTest(limits=limits),
                self.assertRaisesRegex(ValueError, "limit"),
            ):
                self.freeze(**limits)
        with self.assertRaisesRegex(ValueError, "separate"):
            candidate.snapshot(self.work, self.frozen, self.work / "output")
        self.destination.mkdir()
        marker = self.destination / "keep"
        marker.write_text("keep")
        with self.assertRaises(FileExistsError):
            self.freeze()
        self.assertEqual(marker.read_text(), "keep")

    def test_unsafe_and_non_code_editable_paths_rejected(self):
        for name in (
            "../escape.py",
            "/tmp/escape.py",
            "tests/../escape.py",
            "tests//test.py",
            "tests/a\nb.py",
        ):
            with self.subTest(path=name), self.assertRaises(ValueError):
                candidate.editable(name)
        for name in (
            "scripts/rooflang_bridge/sitecustomize.py",
            "tests/payload.bin",
            PROTECTED,
            "crates/simthesizer-core/build.rs",
            "crates/simthesizer-core/src/build.rs",
            "crates/simthesizer-cli/tests/__qwen_frozen_results.rs",
        ):
            self.assertFalse(candidate.editable(name), name)


if __name__ == "__main__":
    unittest.main()
