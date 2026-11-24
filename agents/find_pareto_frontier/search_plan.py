"""Derive provisional search coverage and a next-case suggestion from frozen evidence."""

import argparse
import json
import math
import re
import sys
import time
import tomllib
from pathlib import Path

import archive
import baseline_store
import evaluate
import launcher

VALID = {"VALID_FEASIBLE", "VALID_INFEASIBLE"}
IDENTITY_FIELDS = (
    "attempt",
    "candidate_id",
    "kind",
    "case",
    "trial_id",
    "group_id",
    "directory",
    "started_at",
    "budget_wall_seconds",
)


def candidate_snapshot(experiment, candidate_id):
    if not isinstance(candidate_id, str) or not re.fullmatch(
        "[0-9a-f]{64}", candidate_id
    ):
        raise ValueError("invalid candidate ID")
    relative = Path("candidates") / candidate_id
    candidate = evaluate.read_json(
        baseline_store.regular(experiment.artifacts, relative / "candidate.json")
    )
    policy = experiment.artifacts / relative / "policy"
    if policy.is_symlink() or any(path.is_symlink() for path in policy.rglob("*")):
        raise ValueError("candidate snapshot contains symlinks")
    if (
        not isinstance(candidate, dict)
        or set(candidate) != {"id", "kind", "files"}
        or candidate["kind"] not in {"candidate", "reference"}
        or candidate["id"] != candidate_id
        or evaluate.identity({key: candidate[key] for key in ("kind", "files")})
        != candidate_id
        or not candidate["files"]
        or launcher.inventory(policy) != candidate["files"]
    ):
        raise ValueError("candidate snapshot identity differs")
    if candidate["kind"] == "reference" and candidate != baseline_store.reference(
        experiment.manifest
    ):
        raise ValueError("reference differs from frozen baseline")
    return candidate


def compatible_configuration(experiment, candidate, path, *, sealed=False):
    actual = tomllib.loads(path.read_text())
    expected = experiment.configuration(candidate)
    # Host and container use different absolute Python/worker paths. All other
    # configuration, including hardware and diagnostics, must remain identical.
    for pool in ("prefill_instances", "decode_instances"):
        children = actual["system"][pool]
        if len(children) != len(expected["system"][pool]):
            raise ValueError("evaluation deployment differs")
        for child, original in zip(children, expected["system"][pool]):
            command = child["model_runner"]["command"]
            expected_command = original["model_runner"]["command"]
            container_command = [
                "/opt/venv/bin/python",
                "/workspace/simthesizer/scripts/rooflang_worker.py",
            ]
            if (
                not isinstance(command, list)
                or len(command) != 2
                or any(not isinstance(arg, str) or not arg for arg in command)
                or (not sealed and command not in (expected_command, container_command))
            ):
                raise ValueError("evaluation worker command differs")
            # A local baseline's host command is anchored by its read-only seal;
            # the host paths are unavailable inside the search container.
            child["model_runner"]["command"] = expected_command
    if actual != expected:
        raise ValueError("evaluation configuration differs")


def completed_evidence(experiment, row, candidate, expected_rows, *, sealed=False):
    directory = Path(row["directory"])
    if not sealed:
        saved = evaluate.read_json(
            baseline_store.regular(experiment.artifacts, directory / "receipt.json")
        )
        if saved != row:
            raise ValueError("ledger differs from saved receipt")
    hashes = row.get("evidence_sha256", {})
    if not isinstance(hashes, dict) or set(hashes) != set(baseline_store.EVIDENCE):
        raise ValueError("complete receipt requires all evidence files")
    paths = {}
    for name, expected in hashes.items():
        path = baseline_store.regular(experiment.artifacts, directory / name)
        if launcher.digest(path) != expected:
            raise ValueError(f"receipt evidence changed: {name}")
        paths[name] = path
    if evaluate.load_workload(paths["workload.jsonl"]) != expected_rows:
        raise ValueError("derived workload differs")
    compatible_configuration(experiment, candidate, paths["config.toml"], sealed=sealed)
    build = evaluate.read_json(paths["build.json"])
    if (
        not isinstance(build, dict)
        or build.get("candidate_id") != candidate["id"]
        or build.get("source_sha256") != experiment.manifest["source_sha256"]
    ):
        raise ValueError("build provenance differs")
    run = evaluate.load_run(str(paths["result.jsonl"]))
    complete = evaluate.validate_complete(run, expected_rows)
    metrics = evaluate.summarize_run(run)
    slo = evaluate.slo_metrics(run, experiment.config["slo"])
    if not slo["available"]:
        raise ValueError("complete result is not scorable")
    objectives = {
        "ttft_p99": metrics["ttft"]["p99"],
        "tpot_p99": metrics["tpot"]["p99"] if slo["tpot_applicable"] else None,
        "goodput_per_gpu": slo["goodput_per_gpu"],
    }
    saved_metrics = evaluate.read_json(paths["metrics.json"])
    if (
        not isinstance(saved_metrics, dict)
        or row["status"]
        != ("VALID_FEASIBLE" if slo["feasible"] else "VALID_INFEASIBLE")
        or row.get("objectives") != objectives
        or row.get("tpot_applicable") != slo["tpot_applicable"]
        or saved_metrics.get("completion") != complete
        or saved_metrics.get("slo") != slo
        or any(saved_metrics.get(key) != value for key, value in metrics.items())
        or saved_metrics.get("policy") != evaluate.read_json(paths["policy-stats.json"])
        or saved_metrics.get("service_window")
        != {
            "origin_s": run.origin,
            "finish_s": run.service_finish_s,
            "source": "native_simulation_end"
            if run.service_finish_s is not None
            else "result_timeline",
        }
    ):
        raise ValueError("reported metrics differ from complete native results")


def checked_records(experiment):
    """Malformed evidence is reported, never turned into coverage or a low score."""
    records = evaluate.ledger(experiment.book)
    if experiment.local_baseline:
        baseline = experiment.check_baseline()
        records = baseline["records"] + [
            row
            for row in records
            if not isinstance(row, dict) or row.get("kind") != "reference"
        ]
    cases = {case["id"]: case for case in experiment.cases()}
    derived = {key: experiment.derived(case) for key, case in cases.items()}
    starts, finished, valid, rejected, snapshots = {}, set(), [], [], {}
    failures = []
    attempts, attempted_cases = {}, {}
    for index, row in enumerate(records):
        try:
            if not isinstance(row, dict):
                raise TypeError("receipt must be an object")
            for field in ("started_at", "budget_wall_seconds"):
                value = row.get(field)
                if (
                    type(value) not in (int, float)
                    or not math.isfinite(value)
                    or value < 0
                    or (field == "budget_wall_seconds" and value == 0)
                ):
                    raise ValueError(f"invalid receipt accounting: {field}")
            if row.get("event") == "finished":
                value = row.get("wall_seconds")
                if (
                    type(value) not in (int, float)
                    or not math.isfinite(value)
                    or value < 0
                ):
                    raise ValueError("invalid receipt accounting: wall_seconds")
            case = cases[row["case"]["id"]]
            if row["case"] != case:
                raise ValueError("receipt case differs from frozen matrix")
            candidate_id = row["candidate_id"]
            sealed = experiment.local_baseline and row.get("kind") == "reference"
            if candidate_id not in snapshots:
                snapshots[candidate_id] = (
                    baseline_store.reference(experiment.manifest)
                    if sealed
                    else candidate_snapshot(experiment, candidate_id)
                )
            candidate = snapshots[candidate_id]
            trial, group = experiment.trial_identity(
                candidate, case, derived[case["id"]]
            )
            if (
                candidate["kind"] != row["kind"]
                or candidate["id"] != candidate_id
                or (row["trial_id"], row["group_id"]) != (trial, group)
            ):
                raise ValueError("receipt trial identity differs")
            attempt = row["attempt"]
            if type(attempt) is not int or attempt < 1:
                raise ValueError("invalid attempt ID")
            if row["event"] == "started":
                if attempt in starts:
                    raise ValueError("duplicate started attempt")
                starts[attempt] = row
                attempts.setdefault(candidate_id, set()).add(attempt)
                attempted_cases.setdefault(candidate_id, set()).add(case["id"])
                continue
            if row["event"] != "finished" or attempt in finished:
                raise ValueError("invalid or duplicate finished attempt")
            if not sealed and (
                attempt not in starts
                or any(
                    starts[attempt].get(key) != row.get(key) for key in IDENTITY_FIELDS
                )
            ):
                raise ValueError("finished receipt has no matching start")
            if (
                not sealed
                and evaluate.read_json(
                    baseline_store.regular(
                        experiment.artifacts, Path(row["directory"]) / "receipt.json"
                    )
                )
                != row
            ):
                raise ValueError("ledger differs from saved receipt")
            finished.add(attempt)
            attempts.setdefault(candidate_id, set()).add(attempt)
            attempted_cases.setdefault(candidate_id, set()).add(case["id"])
            if row["status"] in {"INVALID", "INCOMPLETE"}:
                failures.append(row)
                continue
            if row["status"] not in VALID:
                raise ValueError("unrecognized evaluation status")
            completed_evidence(
                experiment, row, candidate, derived[case["id"]], sealed=sealed
            )
            valid.append(row)
        except (KeyError, TypeError, ValueError, OSError, OverflowError) as error:
            rejected.append(
                {
                    "record": index,
                    "candidate_id": row.get("candidate_id")
                    if isinstance(row, dict)
                    else None,
                    "reason": str(error),
                }
            )
    return {
        "valid": valid,
        "attempts": attempts,
        "attempted_cases": attempted_cases,
        "rejected": rejected,
        "failures": failures,
        "open_attempts": sorted(set(starts) - finished),
    }


def load_catalog(path):
    # The generator owns recipe normalization, rendering and family derivation.
    import search_space

    return search_space.validate_catalog(path)


def derive_plan(experiment, catalog_path):
    catalog = load_catalog(catalog_path)
    evidence = checked_records(experiment)
    cases = experiment.cases()
    train = [row["id"] for row in cases if row["split"] == "train"]
    validation = [row["id"] for row in cases if row["split"] == "validation"]
    rows = evidence["valid"]
    completed = {}
    for row in rows:
        completed.setdefault(row["candidate_id"], {})[row["case"]["id"]] = row
    frontier = archive.pareto([row for row in rows if row["case"]["split"] == "train"])
    promising = set()
    for points in frontier.values():
        for point in points:
            promising.add(point["candidate_id"])
            promising.update(point["equivalent_candidates"])
    routing_available = any(
        len(experiment.scenario["system"][pool]) > 1
        for pool in ("prefill_instances", "decode_instances")
    )
    entries, coverage = [], {}
    for item in catalog["entries"]:
        candidate_id = item["evaluator_candidate_id"]
        done = completed.get(candidate_id, {})
        state = {
            "id": item["id"],
            "candidate_id": candidate_id,
            "families": item["families"],
            "source": str((catalog_path.parent / item["path"]).resolve()),
            "files": item["files"],
            "attempted": len(evidence["attempts"].get(candidate_id, ())),
            "attempted_case_ids": sorted(
                evidence["attempted_cases"].get(candidate_id, ())
            ),
            "retryable_cases": sorted(
                {
                    row["case"]["id"]
                    for row in evidence["failures"]
                    if row["candidate_id"] == candidate_id
                    and row["case"]["id"] not in done
                }
            ),
            "failed_cases": [
                {
                    "case_id": row["case"]["id"],
                    "attempt": row["attempt"],
                    "status": row["status"],
                }
                for row in evidence["failures"]
                if row["candidate_id"] == candidate_id
            ],
            "valid_completed_cases": sorted(done),
            "feasible_cases": sorted(
                key for key, row in done.items() if row["status"] == "VALID_FEASIBLE"
            ),
            "missing_train": [key for key in train if key not in done],
            "missing_validation": [key for key in validation if key not in done],
            "full_matrix": bool(cases) and len(done) == len(cases),
            "training_frontier": candidate_id in promising,
            "ablations": [],
            "static_controls": [],
        }
        for family in item["families"]:
            entry = coverage.setdefault(
                family,
                {
                    "applicable": routing_available
                    if family == "routing_load"
                    else True,
                    "generated": [],
                    "attempted": [],
                    "valid_completed": [],
                },
            )
            entry["generated"].append(item["id"])
            if state["attempted"]:
                entry["attempted"].append(item["id"])
            if done:
                entry["valid_completed"].append(item["id"])
        entries.append(state)
    by_id = {entry["id"]: entry for entry in entries}
    for item in catalog["entries"]:
        state = by_id[item["id"]]
        for relation in ("ablations", "static_controls"):
            for other in item.get(relation, []):
                ablation = by_id[other]
                matching = sorted(
                    set(state["valid_completed_cases"])
                    & set(ablation["valid_completed_cases"])
                )
                state[relation].append(
                    {
                        "id": other,
                        "matched_cases": matching,
                        "missing_cases": [
                            key
                            for key in state["valid_completed_cases"]
                            if key not in matching
                        ],
                    }
                )

    suggestion = None

    def suggest(state, missing, phase):
        nonlocal suggestion
        missing = [case for case in missing if case not in state["attempted_case_ids"]]
        if (
            suggestion is None
            and missing
            and all(coverage[family]["applicable"] for family in state["families"])
        ):
            case = missing[0]
            suggestion = {
                "recipe_id": state["id"],
                "candidate_id": state["candidate_id"],
                "source": state["source"],
                "candidate_directory": str(experiment.source / evaluate.POLICY),
                "files": state["files"],
                "case_id": case,
                "phase": phase,
                "instruction": "Replace all existing candidate Rust files with exactly these source files, check the candidate ID, then run the frozen evaluator; this is a suggestion, not verification.",
                "evaluate_argv": [
                    sys.executable,
                    str(experiment.source / launcher.AGENT_REL / "evaluate.py"),
                    "--manifest",
                    str(experiment.manifest_path),
                    "--candidate",
                    "--case",
                    case,
                ],
            }

    for state in sorted(entries, key=lambda item: (len(item["families"]), item["id"])):
        if any(
            coverage[family]["applicable"] and not coverage[family]["attempted"]
            for family in state["families"]
        ):
            suggest(state, state["missing_train"], "family_screen")
    for state in entries:
        if state["training_frontier"]:
            suggest(state, state["missing_train"], "complete_training")
            if not state["missing_train"]:
                suggest(state, state["missing_validation"], "validation")
    for state in entries:
        if state["training_frontier"]:
            for ablation in state["ablations"] + state["static_controls"]:
                suggest(by_id[ablation["id"]], ablation["missing_cases"], "ablation")
    for state in entries:
        if not state["valid_completed_cases"]:
            suggest(state, state["missing_train"], "screen")
    for state in entries:
        suggest(state, state["missing_train"], "explore_training")

    unavailable = bool(evidence["rejected"])
    remaining = None
    exhausted = None
    if not unavailable:
        accounting = experiment.accounting()
        exhausted = (
            accounting["attempts"] >= experiment.config["search"]["max_trials"]
            or accounting["wall_seconds"]
            >= experiment.config["search"]["max_wall_seconds"]
        )
        remaining = {
            "trials": max(
                0, experiment.config["search"]["max_trials"] - accounting["attempts"]
            ),
            "wall_seconds": max(
                0,
                experiment.config["search"]["max_wall_seconds"]
                - accounting["wall_seconds"],
            ),
        }
    if exhausted or unavailable or evidence["open_attempts"]:
        suggestion = None
    catalog_ids = {entry["candidate_id"] for entry in entries}
    return {
        "schema_version": 1,
        "verification": "provisional",
        "manifest_sha256": launcher.digest(experiment.manifest_path),
        "catalog_sha256": launcher.digest(catalog_path),
        "validation_available": bool(validation),
        "families": coverage,
        "entries": entries,
        "uncatalogued_candidates": [
            {
                "candidate_id": candidate_id,
                "attempted": len(evidence["attempts"].get(candidate_id, ())),
                "valid_completed_cases": sorted(done),
                "full_matrix": len(done) == len(cases),
            }
            for candidate_id, done in sorted(completed.items())
            if candidate_id not in catalog_ids
        ],
        "rejected_receipts": evidence["rejected"],
        "open_attempts": evidence["open_attempts"],
        "budget_exhausted": exhausted,
        "budget_unavailable": unavailable,
        "remaining_budget": remaining,
        "next": suggestion,
        "note": "Coverage is reconstructed from checked receipts. Catalog exhaustion is not global optimality; synthesize new policies. The evaluator/supervisor alone enforce budgets; external replay alone verifies results.",
    }


def main(argv=None):
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--catalog", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    experiment = evaluate.Experiment(args.manifest)
    import search_space

    for relative, actual in (
        (launcher.AGENT_REL / "search_plan.py", Path(__file__)),
        (launcher.AGENT_REL / "search_space.py", Path(search_space.__file__)),
        (Path("examples/policies/composable.rs"), search_space.LIBRARY),
    ):
        if experiment.manifest["source_sha256"].get(str(relative)) != launcher.digest(
            actual
        ):
            parser.error("search tools differ from frozen source; use that run's tools")
    output = args.output.absolute()
    if (
        output.parent.resolve() != experiment.artifacts.resolve()
        or not output.name.startswith("search-")
        or output.suffix != ".json"
        or output.is_symlink()
        or output.resolve() == args.catalog.resolve()
    ):
        parser.error("output must be a separate artifacts/search-*.json file")
    result = derive_plan(experiment, args.catalog)
    result["planning_wall_seconds"] = time.perf_counter() - started
    launcher.save(output, result)
    print(
        json.dumps(
            {
                "output": str(output),
                "next": result["next"],
                "rejected_receipts": len(result["rejected_receipts"]),
            }
        )
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, OSError, KeyError, TypeError) as error:
        print(f"error: {error}", file=sys.stderr)
        sys.exit(2)
