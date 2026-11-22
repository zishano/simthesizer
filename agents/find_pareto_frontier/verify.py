"""External clean replay; never run inside the search container."""

import argparse
import copy
import fcntl
import json
import shutil
import sys
import time
from pathlib import Path

import evaluate
import launcher


def verify(manifest, candidate_id, selected=()):
    original = evaluate.Experiment(manifest)
    if original.run == Path("/workspace") or Path("/.dockerenv").exists():
        raise ValueError("verification must run outside the search container")
    verified_root = original.run / "verified"
    verified_root.mkdir(exist_ok=True)
    with (verified_root / ".verify.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return replay_candidate(original, candidate_id, selected)


def replay_candidate(original, candidate_id, selected):
    if len(candidate_id) != 64 or any(
        c not in "0123456789abcdef" for c in candidate_id
    ):
        raise ValueError("expected a candidate SHA256 ID")
    candidate_path = original.artifacts / "candidates" / candidate_id
    candidate = evaluate.read_json(candidate_path / "candidate.json")
    if (
        candidate["id"] != candidate_id
        or evaluate.identity({key: candidate[key] for key in ("kind", "files")})
        != candidate_id
    ):
        raise ValueError("candidate identity differs")
    if launcher.inventory(candidate_path / "policy") != candidate["files"]:
        raise ValueError("candidate snapshot changed")
    cases = original.cases()
    if set(selected) - {case["id"] for case in cases}:
        raise ValueError("unknown verification case")
    cases = [case for case in cases if not selected or case["id"] in selected]
    receipts = {}
    for row in original.records():
        if (
            row.get("event") == "finished"
            and row.get("candidate_id") == candidate_id
            and row.get("status", "").startswith("VALID_")
        ):
            receipts[row["case"]["id"]] = row
    if any(case["id"] not in receipts for case in cases):
        raise ValueError(
            "candidate lacks complete search receipts for the requested matrix"
        )
    for case in cases:
        previous = receipts[case["id"]]
        trial_id, group_id = original.trial_identity(
            candidate, case, original.derived(case)
        )
        if (
            previous["case"] != case
            or previous["trial_id"] != trial_id
            or previous["group_id"] != group_id
        ):
            raise ValueError(
                "search receipt does not match the frozen candidate/case identity"
            )
        directory = original.artifacts / previous["directory"]
        if not directory.resolve().is_relative_to(original.artifacts.resolve()):
            raise ValueError("search receipt path escapes artifacts")
        if any(
            launcher.digest(directory / name) != digest
            for name, digest in previous["evidence_sha256"].items()
        ):
            raise ValueError("search evidence changed before independent verification")
    destination = (
        original.run
        / "verified"
        / f"{time.strftime('%Y%m%d-%H%M%S')}-{candidate_id[:12]}"
    )
    destination.mkdir(parents=True, exist_ok=False)
    for name in ("source", "inputs", "baseline-policy"):
        shutil.copytree(
            getattr(
                original,
                {"source": "source", "inputs": "inputs", "baseline-policy": "baseline"}[
                    name
                ],
            ),
            destination / name,
        )
    shutil.rmtree(destination / "source" / evaluate.POLICY)
    shutil.copytree(candidate_path / "policy", destination / "source" / evaluate.POLICY)
    for name in ("task", "cache", "artifacts"):
        (destination / name).mkdir()
    frozen = copy.deepcopy(original.manifest)
    budget = frozen["experiment"]["verification"]
    frozen["experiment"]["search"].update(
        {
            "max_trials": budget["max_trials"],
            "max_wall_seconds": budget["max_wall_seconds"],
        }
    )
    launcher.save(destination / "task/manifest.json", frozen)
    (destination / "task/AGENTS.md").write_text(
        launcher.render_task(destination / "source", frozen)
    )
    replay = evaluate.Experiment(
        destination / "task/manifest.json", independent_replay=True
    )
    # This ledger is outside every search-container mount. Repeated external
    # verification calls share the frozen budget instead of resetting it.
    replay.book = original.run / "verified/experiments.jsonl"
    snapshot = replay.freeze_candidate(baseline=candidate["kind"] == "reference")
    checks = []
    for index, case in enumerate(cases):
        result = replay.evaluate(snapshot, case, force=True, clean=index == 0)
        previous = receipts[case["id"]]
        matches = (
            result["status"].startswith("VALID_")
            and result["status"] == previous["status"]
            and result.get("objectives") == previous.get("objectives")
        )
        if matches:
            expected = previous["evidence_sha256"]["result.jsonl"]
            matches = result["evidence_sha256"]["result.jsonl"] == expected
        checks.append({"case_id": case["id"], "matches": matches, "receipt": result})
    verified = bool(checks) and all(row["matches"] for row in checks)
    receipt = {
        "status": "VERIFIED" if verified else "VERIFICATION_FAILED",
        "candidate_id": candidate_id,
        "source_manifest_sha256": launcher.digest(original.manifest_path),
        "covered_cases": [row["case_id"] for row in checks],
        "holdout_evaluated": False,
        "checks": checks,
        "note": "Independent clean build/replay on the listed frozen cases; no unseen-workload or hardware accuracy claim.",
    }
    launcher.save(destination / "verification.json", receipt)
    return destination, receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--case", action="append", default=[])
    args = parser.parse_args()
    path, result = verify(args.manifest, args.candidate, args.case)
    print(
        json.dumps(
            {
                "directory": str(path),
                "status": result["status"],
                "covered_cases": result["covered_cases"],
            },
            indent=2,
        )
    )
    return 0 if result["status"] == "VERIFIED" else 2


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, OSError, KeyError) as error:
        print(f"error: {error}", file=sys.stderr)
        sys.exit(2)
