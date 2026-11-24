"""Bind mechanism hypotheses and counterfactuals to host-owned evidence."""

import copy
import json
import math
import re
from pathlib import PurePosixPath

import baseline_store
import objectives
import research

VERSION = "qwen-mechanism-contract-v1"
REQUEST_ID = re.compile(r"[a-f0-9]{32}")
MAX_PLAN_BYTES = 32768


def text(value, name):
    if not isinstance(value, str) or not value.strip() or len(value) > 4096:
        raise ValueError(
            f"plan.{name} requires nonempty text (at most 4096 characters)"
        )
    return value


def test_path(name, files):
    if not isinstance(name, str):
        raise ValueError("plan check requires a source test path")  # noqa: TRY004 - request rejection
    path = PurePosixPath(name)
    python = (
        path.is_relative_to("tests")
        and path.name.startswith("test")
        and path.suffix == ".py"
    )
    rust = (
        len(path.parts) == 4
        and path.parts[0] == "crates"
        and path.parts[2] == "tests"
        and path.suffix == ".rs"
    )
    if (
        str(path) != name
        or ".." in path.parts
        or name not in files
        or not (python or rust)
    ):
        raise ValueError(f"plan test is not a sealed test entrypoint: {name}")
    return name


def validate_research(research):
    """Require a falsifiable proposal, without certifying its causal claims."""
    fields = {
        "name",
        "family",
        "causes",
        "discriminating_check",
        "expected_metrics",
        "implementation",
        "overhead",
    }
    if not isinstance(research, dict) or set(research) != fields:
        raise ValueError("plan.research requires " + ", ".join(sorted(fields)))
    for key in fields - {"causes", "expected_metrics"}:
        text(research[key], "research." + key)
    causes = research["causes"]
    if not isinstance(causes, list) or not 2 <= len(causes) <= 16:
        raise ValueError(
            "plan.research.causes requires a primary cause and alternatives"
        )
    for cause in causes:
        text(cause, "research.causes")
    if len({cause.strip() for cause in causes}) != len(causes):
        raise ValueError("plan.research.causes requires distinct explanations")
    metrics = research["expected_metrics"]
    if (
        not isinstance(metrics, list)
        or not 1 <= len(metrics) <= len(objectives.METRIC_DIRECTIONS)
        or any(not isinstance(name, str) for name in metrics)
        or len(set(metrics)) != len(metrics)
        or set(metrics) - objectives.METRIC_DIRECTIONS.keys()
    ):
        raise ValueError(
            "plan.research.expected_metrics requires distinct objective names"
        )


def validate_plan(plan, candidate, *, require_research=False):
    """The host computes required checks; authors cannot remove obligations."""
    if (
        not isinstance(plan, dict)
        or type(plan.get("schema_version")) is not int
        or plan["schema_version"] != 1
    ):
        raise ValueError("mechanism changes require --plan with schema_version=1")
    if len(json.dumps(plan, ensure_ascii=False).encode()) > MAX_PLAN_BYTES:
        raise ValueError("mechanism plan exceeds byte limit")
    allowed = {
        "schema_version",
        "mode",
        "summary",
        "state_ownership",
        "preserved_semantics",
        "expected_effect",
        "alternatives",
        "stop_conditions",
        "evidence",
        "control",
        "checks",
        "research",
    }
    if set(plan) - allowed:
        raise ValueError("unknown mechanism plan fields")
    if plan.get("mode") not in ("candidate", "control"):
        raise ValueError("plan.mode must be candidate or control")
    if "research" in plan or require_research:
        validate_research(plan.get("research"))
    for key in ("summary", "state_ownership", "preserved_semantics", "expected_effect"):
        text(plan.get(key), key)
    for key in ("alternatives", "stop_conditions"):
        values = plan.get(key)
        if not isinstance(values, list) or not 1 <= len(values) <= 16:
            raise ValueError(f"plan.{key} requires 1..16 entries")
        for value in values:
            text(value, key)
    evidence = plan.get("evidence")
    if not isinstance(evidence, list) or not 1 <= len(evidence) <= 16:
        raise ValueError(
            "plan.evidence requires 1..16 sealed baseline/trial references"
        )
    checks = plan.get("checks")
    required = set(candidate["impact"]["required_checks"])
    if not isinstance(checks, list) or not bool(required) <= len(checks) <= 32:
        raise ValueError("plan.checks must cover host obligations (at most 32 claims)")
    supplied = set()
    for check in checks:
        if not isinstance(check, dict) or set(check) != {"id", "argument", "tests"}:
            raise ValueError("each plan check requires id, argument, tests")
        name = text(check["id"], "checks.id")
        if name in supplied or name not in required:
            raise ValueError("plan check is duplicate or not a host-derived obligation")
        supplied.add(name)
        text(check["argument"], "checks.argument")
        tests = check["tests"]
        if not isinstance(tests, list) or not 1 <= len(tests) <= 16:
            raise ValueError("each invariant must name at least one executable test")
        for path in tests:
            test_path(path, candidate["files"])
    if supplied != required:
        raise ValueError(
            "missing mechanism obligations: " + ", ".join(sorted(required - supplied))
        )
    return copy.deepcopy(plan)


def resolve_evidence(
    run, manifest, reference, case_id, history, baseline, *, observation=True
):
    """Resolve IDs only; researcher-controlled filesystem paths are never evidence."""
    fields = {"kind", "id", "observation"} if observation else {"kind", "id"}
    if not isinstance(reference, dict) or set(reference) != fields:
        raise ValueError("evidence must identify a sealed baseline/trial record")
    kind, identifier = reference["kind"], reference["id"]
    if not isinstance(identifier, str) or kind not in ("baseline", "trial"):
        raise ValueError("invalid evidence identity")
    if observation:
        text(reference["observation"], "evidence.observation")
    if kind == "baseline":
        records = [
            row
            for row in baseline["records"]
            if row["case"]["id"] == identifier == case_id
        ]
    else:
        if not REQUEST_ID.fullmatch(identifier):
            raise ValueError("trial evidence requires a controller request ID")
        records = [
            row
            for row in history
            if row.get("event") == "finished"
            and row.get("id") == identifier
            and row.get("status") == "EVALUATED"
            and row.get("case_id") == case_id
        ]
    if len(records) != 1:
        raise ValueError(
            "evidence must be a unique completed record of this frozen case"
        )
    row = records[0]
    directory = row["directory"] if kind == "baseline" else "evaluations/" + identifier
    # Baseline anchors are verified by the controller; hash result and score
    # again to bind the specific evidence consumed by this hypothesis.
    for name in ("result.jsonl", "metrics.json"):
        path = baseline_store.regular(run, "artifacts/" + directory + "/" + name)
        if baseline_store.digest(path) != row["evidence_sha256"].get(name):
            raise ValueError("referenced evidence changed")
    metrics_path = baseline_store.regular(
        run, "artifacts/" + directory + "/metrics.json"
    )
    if metrics_path.stat().st_size > 8 * 1024 * 1024:
        raise ValueError("referenced metrics exceed size limit")
    metrics = json.loads(metrics_path.read_text())
    return {
        "kind": kind,
        "id": identifier,
        "case_id": case_id,
        "result_sha256": row["evidence_sha256"]["result.jsonl"],
        "metrics_sha256": row["evidence_sha256"]["metrics.json"],
        "candidate_id": row["candidate_id"],
        "policy_kind": "reference" if kind == "baseline" else row["policy_kind"],
        "policy_sha256": row.get("policy_sha256"),
        "compatibility": row.get("compatibility"),
        "objectives": objectives.from_metrics(metrics)
        if research.enabled(manifest)
        else row["objectives"],
        "service_finish_s": metrics.get("service_window", {}).get("finish_s"),
        "manifest_sha256": baseline_store.identity(manifest),
    }


def seal(run, manifest, candidate, plan, case_id, policy_kind, history):
    require_research = research.enabled(manifest)
    if candidate.get("kind") != "mechanism" and plan is None and not require_research:
        return None
    plan = validate_plan(plan, candidate, require_research=require_research)
    baseline = baseline_store.validate(run, manifest)
    evidence = [
        resolve_evidence(run, manifest, ref, case_id, history, baseline)
        for ref in plan["evidence"]
    ]
    control = resolve_evidence(
        run,
        manifest,
        plan.get("control"),
        case_id,
        history,
        baseline,
        observation=False,
    )
    matched_policy = control["policy_kind"] == policy_kind
    if policy_kind == "candidate":
        matched_policy &= control["policy_sha256"] == candidate["policy_sha256"]
    # The reference implementation may itself have changed across mechanisms;
    # a name alone never establishes a matched decision rule.
    elif candidate.get("reference_policy_changed"):
        matched_policy = False
    result = {
        "version": VERSION,
        "source_sha256": candidate["source_sha256"],
        "manifest_sha256": baseline_store.identity(manifest),
        "case_id": case_id,
        "policy_kind": policy_kind,
        "plan": plan,
        "evidence": evidence,
        "control": control,
        "matched_policy": bool(matched_policy),
        "required_checks": candidate["impact"]["required_checks"],
    }
    result["id"] = baseline_store.identity(result)
    return result


def compare(contract, result):
    """A control reproduces its reference; this does not establish mechanism-off equivalence."""
    if contract is None:
        return {}
    control = contract["control"]
    finish = result["metrics"].get("service_window", {}).get("finish_s")
    reference_finish = control["service_finish_s"]
    known_tails = all(
        type(value) in (int, float) and math.isfinite(value) and value >= 0
        for value in (finish, reference_finish)
    )
    same = (
        result["evidence_sha256"]["result.jsonl"] == control["result_sha256"]
        and known_tails
        and finish == reference_finish
    )
    if contract["plan"]["mode"] == "control" and not same:
        raise ValueError(
            "control does not reproduce its sealed reference (tokens or known service tail)"
        )
    deltas = {
        key: result["objectives"][key] - value
        for key, value in control["objectives"].items()
        if value is not None and result["objectives"].get(key) is not None
    }
    return {
        "compatibility": "MATCHED"
        if contract["plan"]["mode"] == "control"
        else "NOT_CHECKED",
        "comparison": {
            "control": control,
            "matched_policy": contract["matched_policy"],
            "objective_deltas": deltas,
            "mechanism_off_equivalence": "NOT_ESTABLISHED",
        },
        "comparison_reference_reproduced": control["compatibility"] == "MATCHED"
        and contract["matched_policy"],
    }


def delivery_candidates(records, case_ids):
    """A review shortlist, never a semantic VERIFIED or a replacement for scores."""
    groups = {}
    for row in records:
        if row.get("event") != "finished" or row.get("status") != "EVALUATED":
            continue
        if not row.get("contract") or row["contract"]["plan"]["mode"] != "candidate":
            continue
        groups.setdefault((row["candidate_id"], row["policy_kind"]), []).append(row)
    result = []
    for (candidate_id, policy_kind), rows in sorted(groups.items()):
        covered = {
            row["case_id"]
            for row in rows
            if row.get("comparison_reference_reproduced")
            and row.get("replay") == "MATCHED"
        }
        result.append(
            {
                "candidate_id": candidate_id,
                "policy_kind": policy_kind,
                "ready_for_review": set(case_ids) <= covered,
                "pending_cases": sorted(set(case_ids) - covered),
                "review_scope": "replayed cases with reproduced comparison references; mechanism-off equivalence not established",
                "semantic_validation": "EXPLORATORY",
            }
        )
    return result
