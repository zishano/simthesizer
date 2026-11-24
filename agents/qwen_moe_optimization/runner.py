"""Isolated mechanism trials; host scoring never imports candidate code."""

import argparse
import copy
import hashlib
import importlib.util
import json
import os
import shutil
import stat
import subprocess
import sys
import time
import tomllib
import unittest
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "agents/find_pareto_frontier"))
import baseline_store
import candidate as candidate_contract
import contract as research_contract
import evaluate
import launcher

VERSION = "qwen-mechanism-trial-v2"
MAX_EVIDENCE_BYTES = 128 * 1024 * 1024
CHECKS = (
    "test_qwen3_ep.QwenGraphTests.test_native_metadata_distinguishes_qwen_projection_and_residency",
    "test_qwen3_ep.QwenGraphTests.test_routing_is_batch_order_and_chunk_invariant_with_distinct_top8",
    "test_qwen3_ep.QwenAttentionTpTests.test_projection_expert_and_lm_work_are_conserved_across_tp",
    "test_qwen3_ep.QwenAttentionTpTests.test_tp_reduction_charges_arithmetic_memory_and_both_network_directions",
    "test_qwen3_routing_exact",
    "test_qwen3_shared.QwenSharedTests.test_ep1_h200_cannot_hide_resident_weights",
    "test_qwen3_shared.QwenSharedTests.test_tp2_handoff_gates_each_shard_and_reports_preserve_results",
)
LIFECYCLE_CHECKS = (
    "test_rooflang_shared.SharedExecutionTests.test_cross_tp_chunked_multi_request_lifecycles",
    "test_rooflang_shared.SharedExecutionTests.test_receive_capacity_blocks_next_prefill_until_decode_releases",
    "test_rooflang_shared.SharedExecutionTests.test_colocated_instances_share_compute_and_hbm",
    "test_rooflang_pd.RooflangPdTests.test_kv_handoff_units_and_link_sensitivity",
    "test_serving_policy.ServingPolicyTests.test_cross_tp_chunked_policies_complete_and_preserve_causality",
)
GRAPH_CHECKS = (
    "test_qwen3_ep.QwenGraphTests.test_active_experts_have_separate_gemms_and_weights_read_once",
    "test_qwen3_ep.QwenGraphTests.test_local_dispatch_uses_hbm_but_no_fabric_and_remote_directions_match",
    "test_qwen3_ep.QwenGraphTests.test_workspace_tracks_skew_and_capacity_includes_past_kv",
    "test_qwen3_ep.QwenGraphTests.test_graph_drains_all_work_and_rebinding_never_reuses_routes",
)
OBSERVATION_CHECKS = (
    "test_rooflang_shared.SharedExecutionTests.test_trace_does_not_change_request_results",
    "test_serving_policy.ServingPolicyTests.test_lazy_observations_and_bounded_diagnostics_preserve_execution",
)
NATIVE_TARGETS = (
    "engine",
    "execution",
    "models",
    "network",
    "observations",
    "requests",
    "routers",
    "scheduling",
    "systems",
    "qwen3_235b_a22b",
)
RESOURCE_FIXTURE = "agents/qwen_moe_optimization/validation/resource_invariants.rs"
RESOURCE_TARGET = "__qwen_frozen_resources"
RESOURCE_DESTINATION = f"crates/simthesizer-core/tests/{RESOURCE_TARGET}.rs"


def validation_plan(candidate, reference_files, mechanism_contract=None):
    """Select checks from the trusted contract, never from a research request."""
    impact = candidate_contract.impact(candidate["changed_files"])
    if "impact" in candidate and candidate["impact"] != impact:
        raise ValueError("candidate impact differs from the trusted contract")
    declared_tests = []
    if mechanism_contract is not None:
        sealed = {
            key: value for key, value in mechanism_contract.items() if key != "id"
        }
        if (
            mechanism_contract.get("id") != baseline_store.identity(sealed)
            or mechanism_contract.get("source_sha256") != candidate["source_sha256"]
            or mechanism_contract.get("required_checks") != impact["required_checks"]
        ):
            raise ValueError("mechanism contract does not identify this candidate")
        authored = research_contract.validate_plan(
            mechanism_contract["plan"], {**candidate, "impact": impact}
        )
        declared_tests = sorted(
            {path for check in authored["checks"] for path in check["tests"]}
        )
    requirements = set(impact["required_checks"])
    checks = list(CHECKS)
    if requirements & {
        "causal_dependencies",
        "resource_bounds",
        "kv_lifecycle",
        "interface_compatibility",
    }:
        checks.extend(LIFECYCLE_CHECKS)
    if "work_conservation" in requirements:
        checks.extend(GRAPH_CHECKS)
    if "observation_neutrality" in requirements:
        checks.extend(OBSERVATION_CHECKS)
    if "policy_contract" in requirements:
        checks.append("test_serving_policy")
    whole_modules = {name for name in checks if "." not in name}
    checks = list(
        dict.fromkeys(
            name
            for name in checks
            if "." not in name or name.split(".", 1)[0] not in whole_modules
        )
    )
    native = (
        impact["core_changed"]
        or any(path.endswith(".rs") for path in declared_tests)
        or any(path.endswith(".rs") for path in candidate["changed_files"])
    )
    targets = [*NATIVE_TARGETS, RESOURCE_TARGET] if native else []
    required_paths = {"tests/" + name.split(".", 1)[0] + ".py" for name in checks}
    if native:
        required_paths.update(
            f"crates/simthesizer-core/tests/{name}.rs" for name in NATIVE_TARGETS
        )
        required_paths.add(RESOURCE_FIXTURE)
    missing = required_paths - reference_files.keys()
    if missing:
        raise ValueError(
            "missing frozen validation sources: " + ", ".join(sorted(missing))
        )
    # Include helper modules too: frozen tests may import or mod them.
    protected_tests = {
        name: digest
        for name, digest in reference_files.items()
        if name.startswith("tests/")
        or (
            len(Path(name).parts) >= 4
            and Path(name).parts[0] == "crates"
            and Path(name).parts[2] == "tests"
        )
        or name == RESOURCE_FIXTURE
    }
    return {
        "version": VERSION,
        "impact": impact,
        "python_checks": checks,
        "rust_targets": targets,
        "reference_tests_sha256": protected_tests,
        "resource_fixture": RESOURCE_FIXTURE if native else None,
        "declared_tests": declared_tests,
        "mechanism_contract_id": mechanism_contract["id"]
        if mechanism_contract is not None
        else None,
    }


def stage_validation(run, source, plan):
    """Inject only host-frozen public contracts; private layouts stay editable."""
    if RESOURCE_DESTINATION in launcher.inventory(source):
        raise ValueError("candidate occupies a reserved frozen validation target")
    reference = run / "source"
    for name, expected in plan["reference_tests_sha256"].items():
        if launcher.digest(regular(reference, name)) != expected:
            raise ValueError(f"frozen validation source changed: {name}")
        if launcher.digest(regular(source, name)) != expected:
            raise ValueError(f"candidate changed frozen validation source: {name}")
    if plan["resource_fixture"]:
        destination = source / RESOURCE_DESTINATION
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(regular(reference, RESOURCE_FIXTURE), destination)


def additional_checks(candidate, reference_files, declared_tests=(), native_targets=()):
    """Run new test entrypoints and Rust unit tests for changed native code."""
    checks = []
    for name in sorted(
        (set(candidate["files"]) - set(reference_files)) | set(declared_tests)
    ):
        path = Path(name)
        crate_test = (
            len(path.parts) >= 4
            and path.parts[0] == "crates"
            and path.parts[2] == "tests"
        )
        if path.parts[0] == "tests" or crate_test:
            if path.name.startswith("test") and path.suffix == ".py":
                checks.append({"kind": "python", "path": name})
            elif crate_test and len(path.parts) == 4 and path.suffix == ".rs":
                check = {"kind": "rust_integration", "target": path.stem}
                if path.parts[1] != "simthesizer-core":
                    check["package"] = path.parts[1]
                if (
                    check.get("package", "simthesizer-core") != "simthesizer-core"
                    or check["target"] not in native_targets
                ):
                    checks.append(check)
    if any(name.endswith(".rs") for name in candidate["changed_files"]):
        checks.insert(0, {"kind": "rust_unit"})
    if any(
        name.startswith("crates/simthesizer-cli/src/") and name.endswith(".rs")
        for name in candidate["changed_files"]
    ):
        checks.insert(1, {"kind": "rust_binary", "package": "simthesizer-cli"})
    return checks


def check_command(check, source):
    if check["kind"] == "python":
        path = source / check["path"]
        return [
            sys.executable,
            "-B",
            str(source / "agents/qwen_moe_optimization/runner.py"),
            "--check-python-test",
            str(path),
        ]
    command = [
        "cargo",
        "test",
        "--release",
        "--locked",
        "--offline",
        "-p",
        check.get("package", "simthesizer-core"),
    ]
    if check["kind"] == "rust_unit":
        return command + ["--lib"]
    if check["kind"] == "rust_binary":
        return command + ["--bins"]
    return command + ["--test", check["target"]]


def regular(root, relative, limit=MAX_EVIDENCE_BYTES):
    """Require bounded, unlinked regular files with no symlink ancestors."""
    root, relative = Path(root), Path(relative)
    if relative.is_absolute() or ".." in relative.parts or root.is_symlink():
        raise ValueError("evidence path escapes its directory")
    path = root
    for part in relative.parts:
        path /= part
        if path.is_symlink():
            raise ValueError("evidence contains a symlink")
    mode = path.stat()
    if not stat.S_ISREG(mode.st_mode) or mode.st_nlink != 1:
        raise ValueError("evidence must be an unlinked regular file")
    if mode.st_size > limit:
        raise ValueError("evidence exceeds the size limit")
    return path


def read_json(root, relative):
    return json.loads(regular(root, relative, 8 * 1024 * 1024).read_text())


def case_for(manifest, case_id):
    cases = [case for case in baseline_store.cases(manifest) if case["id"] == case_id]
    if len(cases) != 1:
        raise ValueError(f"unknown case: {case_id}")
    return cases[0]


def prepare_job(
    run,
    candidate_dir,
    job,
    case_id,
    policy_kind="candidate",
    clean=False,
    cache_path=None,
    mechanism_contract=None,
):
    """Derive an evaluator bundle without changing the reference or candidate."""
    run, candidate_dir, job = map(Path, (run, candidate_dir, job))
    if policy_kind not in {"candidate", "reference"}:
        raise ValueError("policy_kind must be candidate or reference")
    manifest = read_json(run, "task/manifest.json")
    case = case_for(manifest, case_id)
    candidate = read_json(candidate_dir, "candidate.json")
    files = launcher.inventory(candidate_dir / "source")
    if (
        files != candidate["files"]
        or baseline_store.identity(files) != candidate["source_sha256"]
    ):
        raise ValueError("candidate source differs from its sealed identity")
    if launcher.inventory(run / "inputs") != manifest["inputs_sha256"]:
        raise ValueError("frozen inputs changed")
    if (
        launcher.inventory(run / "baseline-policy")
        != manifest["baseline_policy_sha256"]
    ):
        raise ValueError("frozen baseline policy changed")
    plan = validation_plan(
        candidate, manifest["source_inventory_sha256"], mechanism_contract
    )
    if RESOURCE_DESTINATION in files:
        raise ValueError("candidate occupies a reserved frozen validation target")
    job.mkdir(parents=True, exist_ok=False)
    for name, source in (
        ("source", candidate_dir / "source"),
        ("inputs", run / "inputs"),
        ("baseline-policy", run / "baseline-policy"),
    ):
        shutil.copytree(source, job / name)
    for name in ("task", "artifacts", "cache"):
        (job / name).mkdir()
    cache = Path(cache_path) if cache_path is not None else job / "cache"
    if cache_path is not None and not cache.resolve().is_relative_to(run.resolve()):
        raise ValueError("shared candidate cache must belong to this run")
    cache.mkdir(parents=True, exist_ok=True)
    if cache.is_symlink():
        raise ValueError("candidate cache must not be a symlink")
    if policy_kind == "reference":
        target = job / "source" / launcher.POLICY_REL
        shutil.rmtree(target)
        shutil.copytree(job / "baseline-policy", target)
    stage_validation(run, job / "source", plan)
    derived = copy.deepcopy(manifest)
    derived.pop("baseline_mode", None)
    derived["source_sha256"] = launcher.inventory(job / "source", mutable=True)
    # The old evaluator deliberately excludes its mutable policy directory.
    derived["source_inventory_sha256"] = launcher.inventory(job / "source")
    derived["mechanism_trial"] = {
        "version": VERSION,
        "candidate_id": candidate["source_sha256"],
        "mechanism_sha256": candidate["mechanism_sha256"],
        "policy_sha256": candidate["policy_sha256"],
        "policy_kind": policy_kind,
        "reference_manifest_sha256": baseline_store.identity(manifest),
    }
    descriptor = {
        **derived["mechanism_trial"],
        "case": case,
        "clean": bool(clean),
        "checks": plan["python_checks"],
        "validation_plan": plan,
        "mechanism_contract": mechanism_contract,
        "candidate": candidate,
        "source_inventory_sha256": derived["source_inventory_sha256"],
        "additional_checks": additional_checks(
            candidate,
            manifest["source_inventory_sha256"],
            plan["declared_tests"],
            plan["rust_targets"],
        ),
        "cache_path": str(cache.resolve()),
    }
    launcher.save(job / "task/manifest.json", derived)
    launcher.save(job / "task/job.json", descriptor)
    (job / "task/AGENTS.md").write_text(launcher.render_task(job / "source", derived))
    return descriptor


class CheckedExperiment(evaluate.Experiment):
    """Check frozen invariants in the same deadline as compilation and execution."""

    def build(self, candidate, deadline, *, clean=False):
        executable, receipt = super().build(candidate, deadline, clean=clean)
        env = dict(self.env)
        env["SIMTHESIZER_BIN"] = str(executable)
        env["CARGO_TARGET_DIR"] = str(
            executable.parent / "clean-target"
            if clean
            else self.cache / "targets" / baseline_store.identity(receipt["tools"])
        )
        env["PYTHONPATH"] = os.pathsep.join(
            str(self.source / name) for name in ("tests", "scripts")
        )
        started = time.monotonic()
        descriptor = read_json(self.run, "task/job.json")
        plan = descriptor["validation_plan"]
        timings = []

        def run_check(label, command):
            check_started = time.monotonic()
            evaluate.run_process(
                command,
                cwd=self.source,
                env=env,
                log=self.artifacts / "trusted-checks.log",
                deadline=deadline,
            )
            timings.append(
                {"check": label, "wall_seconds": time.monotonic() - check_started}
            )

        run_check(
            "frozen_python_contracts",
            [sys.executable, "-B", "-m", "unittest", *plan["python_checks"]],
        )
        if plan["rust_targets"]:
            command = [
                "cargo",
                "test",
                "--release",
                "--locked",
                "--offline",
                "-p",
                "simthesizer-core",
            ]
            for target in plan["rust_targets"]:
                command.extend(("--test", target))
            run_check("frozen_native_contracts", command)
        for check in descriptor["additional_checks"]:
            run_check("candidate_" + check["kind"], check_command(check, self.source))
        launcher.save(
            self.artifacts / "trusted-checks.json",
            {
                "checks": plan["python_checks"],
                "validation_plan": plan,
                "additional_checks": descriptor["additional_checks"],
                "passed": True,
                "wall_seconds": time.monotonic() - started,
                "origin": "candidate-container",
                "timings": timings,
            },
        )
        return executable, receipt


def inside(case_id, *, reference=False, clean=False, manifest_path=None):
    manifest_path = manifest_path or Path("/workspace/task/manifest.json")
    experiment = CheckedExperiment(manifest_path, independent_replay=clean)
    descriptor = read_json(experiment.run, "task/job.json")
    kind = "reference" if reference else "candidate"
    if descriptor["case"]["id"] != case_id or descriptor["policy_kind"] != kind:
        raise ValueError("container request differs from the sealed job")
    plan = descriptor["validation_plan"]
    recomputed = validation_plan(
        descriptor["candidate"],
        plan["reference_tests_sha256"],
        descriptor.get("mechanism_contract"),
    )
    if (
        bool(clean) != descriptor["clean"]
        or plan != recomputed
        or descriptor["checks"] != plan["python_checks"]
    ):
        raise ValueError("container checks differ from the sealed job")
    candidate = experiment.freeze_candidate(baseline=reference)
    result = experiment.evaluate(
        candidate, case_for(experiment.manifest, case_id), clean=clean
    )
    launcher.save(experiment.artifacts / "runner-result.json", result)
    if result["status"].startswith("VALID_"):
        return 0
    report_failure(experiment, result)
    return 2


def report_failure(experiment, result):
    """Make bounded compilation/check errors visible in the controller's log."""
    print(
        "Trial failed: " + str(result.get("reason", result["status"]))[:2048],
        file=sys.stderr,
    )
    paths = [(experiment.artifacts, Path("trusted-checks.log"))]
    directory = Path(result.get("directory", ""))
    if not directory.is_absolute() and ".." not in directory.parts:
        paths += [
            (experiment.artifacts, directory / name)
            for name in ("simulation.log", "plots.log")
        ]
    builds = experiment.cache / "builds"
    if builds.is_dir():
        for index, path in enumerate(builds.glob("*/build.log")):
            if index == 3:
                break
            paths.append((experiment.cache, path.relative_to(experiment.cache)))
    for root, relative in paths:
        try:
            path = regular(root, relative)
            with path.open("rb") as stream:
                stream.seek(max(0, path.stat().st_size - 3072))
                tail = stream.read(3072).decode("utf-8", errors="replace")
            print(f"--- {relative}\n{tail}", file=sys.stderr)
        except OSError, ValueError:
            continue


def python_test(path):
    path = Path(path).resolve()
    if not path.is_relative_to(ROOT) or not path.is_file():
        raise ValueError("new test must belong to the sealed source")
    suite = unittest.TestLoader().discover(str(path.parent), pattern=path.name)
    if suite.countTestCases() == 0:
        raise ValueError(f"new test collected no unittest cases: {path.name}")
    return 0 if unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful() else 2


def trusted_results(run, manifest):
    """Load only the host-frozen scorer, with a distinct module identity."""
    relative = "scripts/results.py"
    path = regular(run / "source", relative)
    if launcher.digest(path) != manifest["source_sha256"].get(relative):
        raise ValueError("frozen results scorer changed")
    name = "_qwen_host_results_" + hashlib.sha256(str(path).encode()).hexdigest()[:16]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def frozen_rows(run, manifest, case, results):
    relative = Path(case["workload_path"]).relative_to("/workspace/inputs")
    rows = results.load_workload(regular(run / "inputs", relative))
    origin = min(row["arrival_time"] for row in rows)
    for row in rows:
        row["arrival_time"] = (
            origin + (row["arrival_time"] - origin) / case["arrival_multiplier"]
        )
    return rows


def check_configuration(run, manifest, actual, policy_kind):
    relative = Path(manifest["experiment"]["inputs"]["scenario"]).relative_to(
        "/workspace/inputs"
    )
    expected = tomllib.loads(regular(run / "inputs", relative).read_text())
    actual = copy.deepcopy(actual)
    settings = manifest["experiment"]
    policy = {
        "kind": policy_kind,
        "seed": settings["search"]["seed"],
        "limits": {key: settings["policy"][key] for key in evaluate.LIMITS},
        "slo": {key: settings["slo"][key] for key in ("ttft_limit_s", "tpot_limit_s")},
    }
    if settings["artifacts"].get("policy_diagnostics") is not None:
        policy["diagnostics"] = settings["artifacts"]["policy_diagnostics"]
    if actual["system"].pop("policy", None) != policy:
        raise ValueError(
            "trial policy configuration differs from the frozen experiment"
        )
    for pool in ("prefill_instances", "decode_instances"):
        for child in actual["system"][pool]:
            command = child["model_runner"].pop("command")
            if (
                len(command) != 2
                or command[1] != "/workspace/simthesizer/scripts/rooflang_worker.py"
                or command[0]
                not in {
                    "/opt/venv/bin/python",
                    "/opt/venv/bin/python3",
                    "/opt/venv/bin/python3.14",
                }
            ):
                raise ValueError("trial did not use the sealed candidate worker")
        for child in expected["system"][pool]:
            child["model_runner"].pop("command", None)
    if actual != expected:
        raise ValueError("trial model, topology or runtime configuration changed")


def check_build(job, trial, descriptor):
    """Hash build artifacts on the host; this is identity, not execution attestation."""
    build = read_json(trial, "build.json")
    expected = read_json(job, "task/manifest.json")["source_sha256"]
    if build.get("source_sha256") != expected:
        raise ValueError("build does not identify the sealed mechanism source")
    build_id = build.get("build_id", "")
    if (
        not isinstance(build_id, str)
        or len(build_id) != 64
        or any(c not in "0123456789abcdef" for c in build_id)
    ):
        raise ValueError("invalid native build identity")
    cache = Path(descriptor["cache_path"])
    directory = cache / "builds" / build_id
    regular(cache, Path("builds") / build_id / "build.json")
    if read_json(directory, "build.json") != build:
        raise ValueError("trial and cached build receipts differ")
    executable = regular(
        cache, Path("builds") / build_id / "simthesizer", 512 * 1024 * 1024
    )
    if launcher.digest(executable) != build.get("binary_sha256"):
        raise ValueError("native binary content differs from the build receipt")
    if (
        launcher.inventory(directory / "source")
        != descriptor["source_inventory_sha256"]
    ):
        raise ValueError("cached build source differs from the sealed mechanism")
    policy = {
        "kind": descriptor["policy_kind"],
        "files": launcher.inventory(job / "source" / launcher.POLICY_REL),
    }
    if build.get("candidate_id") != baseline_store.identity(policy):
        raise ValueError("native build used a different policy identity")


def draw_plots(run, manifest, destination, deadline=None):
    """Generate published images using the host-frozen plotting code and result."""
    deadline = deadline if deadline is not None else time.monotonic() + 60
    environment = {
        key: value
        for key, value in os.environ.items()
        if key in {"PATH", "HOME", "LANG", "LC_ALL", "TMPDIR"}
    }
    environment.update(PYTHONDONTWRITEBYTECODE="1", MPLBACKEND="Agg")
    for script, output in (
        ("draw.py", "metrics.png"),
        ("draw_throughput.py", "throughput.png"),
    ):
        relative = "scripts/" + script
        path = regular(run / "source", relative)
        if launcher.digest(path) != manifest["source_sha256"].get(relative):
            raise ValueError("frozen plotting code changed")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("trial budget exhausted before host plotting")
        with (destination / "plots.log").open("ab") as log:
            subprocess.run(
                [
                    sys.executable,
                    "-B",
                    str(path),
                    str(destination / "result.jsonl"),
                    "--output",
                    str(destination / output),
                ],
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=remaining,
            )


def collect_job(
    run, job, candidate_meta, case_id, *, deadline=None, mechanism_contract=None
):
    """Recompute scores on the host; subprocess receipts are provisional evidence."""
    run, job = Path(run), Path(job)
    record = {
        "version": VERSION,
        "candidate_id": candidate_meta["source_sha256"],
        "mechanism_sha256": candidate_meta["mechanism_sha256"],
        "policy_sha256": candidate_meta["policy_sha256"],
        "status": "INVALID",
        "semantic_validation": "exploratory",
        "verification": "unverified",
    }
    try:
        manifest = read_json(run, "task/manifest.json")
        case = case_for(manifest, case_id)
        record["case"] = case
        descriptor = read_json(job, "task/job.json")
        if (
            descriptor["candidate"] != candidate_meta
            or descriptor["case"] != case
            or descriptor["reference_manifest_sha256"]
            != baseline_store.identity(manifest)
            or launcher.inventory(job / "source")
            != descriptor["source_inventory_sha256"]
            or descriptor.get("mechanism_contract") != mechanism_contract
        ):
            raise ValueError("job identity changed")
        if launcher.inventory(run / "inputs") != manifest["inputs_sha256"]:
            raise ValueError("frozen inputs changed")
        child = read_json(job / "artifacts", "runner-result.json")
        checks = read_json(job / "artifacts", "trusted-checks.json")
        expected_plan = validation_plan(
            candidate_meta, manifest["source_inventory_sha256"], mechanism_contract
        )
        expected_checks = additional_checks(
            candidate_meta,
            manifest["source_inventory_sha256"],
            expected_plan["declared_tests"],
            expected_plan["rust_targets"],
        )
        if (
            checks.get("passed") is not True
            or checks.get("checks") != expected_plan["python_checks"]
            or checks.get("validation_plan") != expected_plan
            or descriptor.get("validation_plan") != expected_plan
            or descriptor.get("checks") != expected_plan["python_checks"]
            or checks.get("additional_checks") != expected_checks
            or descriptor.get("additional_checks") != expected_checks
        ):
            raise ValueError("frozen invariant checks did not finish")
        for name, expected in expected_plan["reference_tests_sha256"].items():
            if launcher.digest(regular(job / "source", name)) != expected:
                raise ValueError(f"frozen validation source changed: {name}")
        if (
            expected_plan["resource_fixture"]
            and launcher.digest(regular(job / "source", RESOURCE_DESTINATION))
            != expected_plan["reference_tests_sha256"][RESOURCE_FIXTURE]
        ):
            raise ValueError("staged resource conformance fixture changed")
        regular(job / "artifacts", "trusted-checks.log")
        directory = Path(child["directory"])
        if directory.is_absolute() or ".." in directory.parts:
            raise ValueError("trial evidence escapes its directory")
        trial = job / "artifacts" / directory
        # Check every file before the scorer can discover its adjacent sidecars.
        for name in baseline_store.EVIDENCE:
            regular(job / "artifacts", directory / name)
        actual_config = tomllib.loads((trial / "config.toml").read_text())
        check_configuration(run, manifest, actual_config, descriptor["policy_kind"])
        results = trusted_results(run, manifest)
        rows = frozen_rows(run, manifest, case, results)
        if results.load_workload(trial / "workload.jsonl") != rows:
            raise ValueError("trial workload differs from the frozen complete case")
        measured = results.load_run(str(trial / "result.jsonl"))
        metrics = results.summarize_run(measured)
        metrics["completion"] = results.validate_complete(measured, rows)
        metrics["slo"] = results.slo_metrics(measured, manifest["experiment"]["slo"])
        metrics["policy"] = read_json(trial, "policy-stats.json")
        metrics["service_window"] = {
            "origin_s": measured.origin,
            "finish_s": measured.service_finish_s,
            "source": "native_simulation_end"
            if measured.service_finish_s is not None
            else "result_timeline",
        }
        if not metrics["slo"]["available"]:
            raise ValueError("complete case has no available goodput")
        check_build(job, trial, descriptor)
        # A failed execution/check must never become accepted merely by leaving data.
        if (
            not child.get("status", "").startswith("VALID_")
            or child.get("case") != case
        ):
            raise ValueError("container trial did not complete successfully")
        destination = job / "collected"
        destination.mkdir(exist_ok=False)
        for name in baseline_store.EVIDENCE:
            if name not in {"metrics.json", "metrics.png", "throughput.png"}:
                shutil.copyfile(trial / name, destination / name)
        launcher.save(destination / "metrics.json", metrics)
        evidence_names = baseline_store.EVIDENCE
        import research

        if research.enabled(manifest):
            from diagnostics import summarize_diagnostics

            started = time.perf_counter()
            launcher.save(
                destination / "diagnostics.json",
                summarize_diagnostics(measured, metrics),
            )
            evidence_names = (*evidence_names, "diagnostics.json")
            record["diagnostics_file"] = "diagnostics.json"
            record["diagnostics_wall_seconds"] = time.perf_counter() - started
        shutil.copyfile(
            job / "artifacts/trusted-checks.log", destination / "trusted-checks.log"
        )
        draw_plots(run, manifest, destination, deadline)
        record.update(
            status="EVALUATED",
            feasible=metrics["slo"]["feasible"],
            metrics=metrics,
            objectives={
                "ttft_p99": metrics["ttft"]["p99"],
                "tpot_p99": metrics["tpot"]["p99"]
                if metrics["slo"]["tpot_applicable"]
                else None,
                "goodput_per_gpu": metrics["slo"]["goodput_per_gpu"],
            },
            tpot_applicable=metrics["slo"]["tpot_applicable"],
            policy_kind=descriptor["policy_kind"],
            clean_rebuild=descriptor["clean"],
            evidence_directory=str(destination),
            evidence_sha256={
                name: launcher.digest(destination / name) for name in evidence_names
            },
            checks={
                "names": expected_plan["python_checks"],
                "native_targets": expected_plan["rust_targets"],
                "required": expected_plan["impact"]["required_checks"],
                "plan_sha256": baseline_store.identity(expected_plan),
                "additional": expected_checks,
                "scope": "frozen public contracts and supplemental candidate tests",
                "origin": "candidate-container",
                "semantic_proof": False,
                "wall_seconds": checks.get("wall_seconds"),
            },
            host_validation=[
                "source and binary hashes",
                "frozen configuration",
                "complete workload",
                "metric recomputation",
                "plot regeneration",
            ],
        )
        if research.enabled(manifest):
            from objectives import from_metrics

            record["objectives"] = from_metrics(metrics)
        launcher.save(destination / "host-receipt.json", record)
    except (
        ValueError,
        OSError,
        KeyError,
        TypeError,
        OverflowError,
        subprocess.SubprocessError,
    ) as error:
        record["reason"] = str(error)
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--inside", action="store_true")
    modes.add_argument("--check-python-test", type=Path)
    parser.add_argument("--case")
    parser.add_argument("--reference", action="store_true")
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args(argv)
    if args.check_python_test:
        if args.case or args.reference or args.clean:
            parser.error("test mode does not accept trial options")
        return python_test(args.check_python_test)
    if not args.case:
        parser.error("--inside requires --case")
    return inside(args.case, reference=args.reference, clean=args.clean)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, OSError, KeyError) as error:
        print(f"error: {error}", file=sys.stderr)
        sys.exit(2)
