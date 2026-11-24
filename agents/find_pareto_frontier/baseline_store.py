"""Host-sealed baseline evidence; validation needs only the Python standard library."""

import copy
import hashlib
import importlib.metadata
import json
import math
import platform
import shutil
import sys
import tempfile
from pathlib import Path

VERSION = "serving-evaluator-v2"
EVIDENCE = (
    "result.jsonl",
    "metrics.json",
    "config.toml",
    "workload.jsonl",
    "build.json",
    "policy-stats.json",
    "metrics.png",
    "throughput.png",
)


def json_bytes(value):
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        + "\n"
    ).encode()


def identity(value):
    return hashlib.sha256(json_bytes(value)).hexdigest()


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def result_environment():
    """Numerical compatibility, independent of executable paths and compiler banners."""
    return {
        "python": ".".join(str(part) for part in sys.version_info[:3]),
        "implementation": sys.implementation.name,
        "system": platform.system(),
        "machine": platform.machine(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("networkx", "matplotlib")
        },
    }


def cases(manifest):
    config = manifest["experiment"]
    result = []
    for workload in config["workloads"]:
        for multiplier in config["loads"]["arrival_rate_multipliers"]:
            case = {
                "workload_id": workload["id"],
                "workload_path": workload["path"],
                "split": workload["split"],
                "arrival_multiplier": multiplier,
                "seed": config["search"]["seed"],
            }
            case["id"] = f"{workload['id']}-m{float(multiplier)!r}-s{case['seed']}"
            result.append(case)
    if len({row["id"] for row in result}) != len(result):
        raise ValueError("duplicate case identities")
    return result


def reference(manifest):
    candidate = {"kind": "reference", "files": manifest["baseline_policy_sha256"]}
    return {**candidate, "id": identity(candidate)}


def trial_identity(manifest, environment, candidate, case, rows):
    config = manifest["experiment"]
    group = {
        "version": VERSION,
        "base": manifest["source_sha256"],
        "inputs": manifest["inputs_sha256"],
        "build_versions": manifest["build_versions"],
        "python_environment": environment,
        "case": case,
        "derived_workload_sha256": identity(rows),
        "slo": config["slo"],
        "model_assumptions": config["model_assumptions"],
        "policy": config["policy"],
    }
    diagnostics = config.get("artifacts", {}).get("policy_diagnostics")
    if diagnostics is not None:
        group["policy_diagnostics"] = diagnostics
    return identity({"group": group, "candidate": candidate}), identity(group)


def regular(root, relative):
    """Reject traversal and symlinks, including symlinked ancestor directories."""
    root = Path(root)
    relative = Path(relative)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("baseline evidence path escapes its directory")
    path = root
    for part in relative.parts:
        path = path / part
        if path.is_symlink():
            raise ValueError("baseline evidence must not contain symlinks")
    if not path.is_file():
        raise ValueError(f"baseline evidence missing: {relative}")
    return path


def validate(run, manifest):
    """Check the read-only anchor, complete case matrix, identity and all evidence."""
    run = Path(run)
    anchor = read_json(regular(run, "task/baseline.json"))
    index_path = regular(run, "artifacts/baseline/index.json")
    manifest_hash = identity(manifest)
    if anchor != {
        "schema_version": 1,
        "manifest_sha256": manifest_hash,
        "index_sha256": digest(index_path),
    }:
        raise ValueError("baseline anchor or manifest differs")
    index = read_json(index_path)
    expected_cases = cases(manifest)
    expected_reference = reference(manifest)
    if (
        index.get("schema_version") != 1
        or index.get("evaluator_version") != VERSION
        or index.get("manifest_sha256") != manifest_hash
        or index.get("cases") != expected_cases
        or index.get("reference") != expected_reference
    ):
        raise ValueError("baseline frozen identity differs")
    records = index.get("records", [])
    if len(records) != len(expected_cases):
        raise ValueError("baseline case matrix is incomplete")
    for case, row in zip(expected_cases, records):
        if (
            row.get("event") != "finished"
            or row.get("status") not in {"VALID_FEASIBLE", "VALID_INFEASIBLE"}
            or row.get("kind") != "reference"
            or row.get("candidate_id") != expected_reference["id"]
            or row.get("case") != case
            or set(row.get("evidence_sha256", {})) != set(EVIDENCE)
        ):
            raise ValueError("baseline requires complete, valid reference receipts")
        directory = Path(row["directory"])
        if directory.parts[:2] != ("baseline", "cases"):
            raise ValueError("baseline receipt is outside the sealed evidence")
        for name in EVIDENCE:
            path = regular(run, Path("artifacts") / directory / name)
            if digest(path) != row["evidence_sha256"][name]:
                raise ValueError(f"baseline evidence changed: {case['id']}/{name}")
        relative = Path(case["workload_path"]).relative_to("/workspace/inputs")
        source = regular(run, Path("inputs") / relative)
        if digest(source) != manifest["inputs_sha256"].get(str(relative)):
            raise ValueError("baseline frozen workload changed")
        derived = [
            json.loads(line) for line in source.read_text().splitlines() if line.strip()
        ]
        origin = min(item["arrival_time"] for item in derived)
        for item in derived:
            item["arrival_time"] = (
                origin + (item["arrival_time"] - origin) / case["arrival_multiplier"]
            )
        saved = regular(run, Path("artifacts") / directory / "workload.jsonl")
        rows = [
            json.loads(line) for line in saved.read_text().splitlines() if line.strip()
        ]
        trial, group = trial_identity(
            manifest, index["result_environment"], expected_reference, case, derived
        )
        if (
            rows != derived
            or row.get("trial_id") != trial
            or row.get("group_id") != group
        ):
            raise ValueError("baseline trial or workload identity differs")
    accounting = index.get("accounting", {})
    attempts, wall = accounting.get("attempts"), accounting.get("wall_seconds")
    charges = accounting.get("charges", [])
    if (
        type(attempts) is not int
        or attempts < len(records)
        or not isinstance(wall, (float, int))
        or not math.isfinite(wall)
        or wall < 0
        or wall < sum(row["wall_seconds"] for row in records)
        or len(charges) != attempts
        or len({row["started"]["attempt"] for row in charges}) != attempts
        or any(row["started"].get("kind") != "reference" for row in charges)
        or any(
            not math.isfinite(row["wall_seconds"]) or row["wall_seconds"] < 0
            for row in charges
        )
        or wall != sum(row["wall_seconds"] for row in charges)
    ):
        raise ValueError("invalid baseline budget accounting")
    return index


def seal(run, manifest, environment, records, accounting):
    """Publish once, with the task anchor as the final commit marker."""
    run = Path(run)
    anchor = run / "task/baseline.json"
    if anchor.exists():
        return validate(run, manifest)
    artifacts = run / "artifacts"
    destination = artifacts / "baseline"
    staging = Path(tempfile.mkdtemp(prefix=".baseline-", dir=artifacts))
    try:
        saved = []
        for row in records:
            row = copy.deepcopy(row)
            target = staging / "cases" / identity(row["case"])
            target.mkdir(parents=True)
            for name in EVIDENCE:
                source = regular(artifacts, Path(row["directory"]) / name)
                if digest(source) != row["evidence_sha256"][name]:
                    raise ValueError("baseline evidence changed before sealing")
                shutil.copy2(source, target / name)
            row["directory"] = str(Path("baseline/cases") / target.name)
            row.pop("reused", None)
            saved.append(row)
        index = {
            "schema_version": 1,
            "evaluator_version": VERSION,
            "manifest_sha256": identity(manifest),
            "result_environment": environment,
            "reference": reference(manifest),
            "cases": cases(manifest),
            "records": saved,
            "accounting": accounting,
        }
        (staging / "index.json").write_bytes(json_bytes(index))
        # An interrupted publication can leave an unanchored directory. Retain it
        # as evidence rather than trusting it or overwriting its contents.
        if destination.exists() or destination.is_symlink():
            if destination.is_symlink():
                raise ValueError("baseline directory must not be a symlink")
            destination.rename(staging.with_name(staging.name + "-unsealed"))
        staging.rename(destination)
        temporary = anchor.with_suffix(".tmp")
        temporary.write_bytes(
            json_bytes(
                {
                    "schema_version": 1,
                    "manifest_sha256": identity(manifest),
                    "index_sha256": digest(destination / "index.json"),
                }
            )
        )
        temporary.replace(anchor)
        try:
            return validate(run, manifest)
        except ValueError, OSError, KeyError, TypeError:
            anchor.unlink()
            raise
    finally:
        if staging.exists():
            shutil.rmtree(staging)
