"""Host-derived research progress and budget; no simulator hot-path work."""

WORKFLOW = "bottleneck-v1"
FAMILIES = (
    "routing_placement",
    "batch_decode",
    "kv_communication",
    "execution_resources",
)


def enabled(manifest):
    return manifest["experiment"].get("research", {}).get("workflow") == WORKFLOW


def status(manifest, baseline, history, wall_remaining):
    config = manifest["experiment"]
    starts = [row for row in history if row.get("event") == "started"]
    ends = {row["id"]: row for row in history if row.get("event") == "finished"}
    replays = [row for row in starts if row.get("operation") == "replay"]
    charged = sum(
        ends.get(row["id"], {}).get("wall_seconds", row["budget_wall_seconds"])
        for row in replays
    )
    replay_wall = max(0, config["verification"]["max_wall_seconds"] - charged)
    replay_count = max(0, config["verification"]["max_trials"] - len(replays))
    evaluations = max(
        0,
        config["search"]["max_trials"]
        - baseline["accounting"]["attempts"]
        - (len(starts) - len(replays)),
    )
    case_count = len(baseline["records"])
    # Cold replay includes rebuild/checks. Warm simulation wall is no substitute.
    measured = [
        ends[row["id"]]["wall_seconds"]
        for row in replays
        if ends.get(row["id"], {}).get("status") == "EVALUATED"
    ]
    estimate = max(measured) if measured else config["search"]["trial_wall_seconds"]
    matrix_estimate = case_count * estimate
    reserve = min(replay_wall, matrix_estimate) if replay_count else 0
    wall_remaining = max(0, wall_remaining)
    exploration_wall = max(0, wall_remaining - reserve)
    can_evaluate = evaluations > 0 and exploration_wall > 0
    can_replay = replay_count > 0 and replay_wall > 0 and wall_remaining > 0
    phase = "explore" if can_evaluate else "validate" if can_replay else "exhausted"
    coverage = {}
    for row in ends.values():
        plan = (row.get("contract") or {}).get("plan", {})
        hypothesis = plan.get("research")
        if not hypothesis or row.get("operation") == "replay":
            continue
        family = hypothesis["family"]
        group = coverage.setdefault(
            family, {"attempted": 0, "evaluated": 0, "recent": []}
        )
        group["attempted"] += 1
        group["evaluated"] += row.get("status") == "EVALUATED"
        group["recent"].append(
            {
                "request_id": row["id"],
                "name": hypothesis["name"],
                "case": row.get("case_id"),
                "status": row["status"],
            }
        )
        group["recent"] = group["recent"][-8:]
    return {
        "schema_version": 1,
        "workflow": WORKFLOW,
        "verification": "EXPLORATORY",
        "phase": phase,
        "budget": {
            "wall_remaining_s": wall_remaining,
            "evaluate_remaining": evaluations,
            "replay_remaining": replay_count,
            "replay_wall_remaining_s": replay_wall,
            "final_replay_reserve_s": reserve,
            "exploration_wall_remaining_s": exploration_wall,
            "full_matrix_replay_estimate_s": matrix_estimate,
            "replay_estimate_source": "maximum_completed_cold_replay"
            if measured
            else "trial_timeout_upper_bound",
            "full_matrix_replay_fits": replay_count >= case_count
            and replay_wall >= matrix_estimate
            and wall_remaining >= matrix_estimate,
        },
        "can_evaluate": can_evaluate,
        "can_replay": can_replay,
        "started_attempts": len(starts),
        "executed_attempts": sum(
            row.get("execution_started", False) or row.get("status") == "EVALUATED"
            for row in ends.values()
        ),
        "coverage": coverage,
        "unexplored_suggested_families": [
            name for name in FAMILIES if name not in coverage
        ],
        "evidence": {
            "baseline": "/workspace/artifacts/baseline-diagnostics.json",
            "frontier": "/workspace/artifacts/frontier.json",
            "review_candidates": "/workspace/artifacts/review-candidates.json",
        },
        "limitations": [
            "Family names and causal claims are researcher-supplied; coverage is not proof of causality or exhaustion.",
            "Replay estimates are planning bounds, not guaranteed completion times.",
            "The reserve limits evaluation allowances; reasoning time and exploratory replays can still consume final-validation capacity.",
            "SLO attainment and agent exit do not complete research; sealed snapshots survive further edits.",
            "Missing resource evidence requires a bounded observation or distinguishing experiment.",
        ],
    }


def next_session(snapshot, stalled_sessions, limit):
    if snapshot["phase"] == "exhausted":
        return "BUDGET_EXHAUSTED"
    if stalled_sessions >= limit:
        return "STALLED"
    return "CONTINUE"
