"""Qwen research objectives from complete host scores; no simulated timing changes."""

import math

METRIC_DIRECTIONS = {
    "ttft_p99": 1,
    "tpot_p99": 1,
    "goodput_per_gpu": -1,
    "output_tokens_per_gpu": -1,
    "tokens_per_user": -1,
}


def _number(value, name):
    if value is not None and (
        type(value) not in (int, float) or not math.isfinite(value) or value < 0
    ):
        raise ValueError(f"{name} must be a finite nonnegative number or null")
    return value


def from_metrics(metrics):
    """Keep throughput and mean per-user rate, without duplicating makespan."""
    slo = metrics.get("slo", {})
    user = metrics.get("tokens_per_user", {})
    applicable = slo.get("tpot_applicable") is not False
    point = {
        "ttft_p99": metrics.get("ttft", {}).get("p99"),
        "tpot_p99": metrics.get("tpot", {}).get("p99") if applicable else None,
        "goodput_per_gpu": slo.get("goodput_per_gpu"),
        "output_tokens_per_gpu": metrics.get("tokens_per_gpu", {}).get("mean"),
        "tokens_per_user": user.get("mean")
        if applicable and user.get("samples") != 0
        else None,
    }
    return {name: _number(value, name) for name, value in point.items()}


def _coordinates(row):
    applicable = row["tpot_applicable"]
    if type(applicable) is not bool:
        raise ValueError("TPOT applicability must be boolean")
    point = row["objectives"]
    coordinates = [
        _number(point.get(name), name)
        for name in METRIC_DIRECTIONS
        if applicable or name not in ("tpot_p99", "tokens_per_user")
    ]
    if None in coordinates:
        return None
    directions = [
        direction
        for name, direction in METRIC_DIRECTIONS.items()
        if applicable or name not in ("tpot_p99", "tokens_per_user")
    ]
    return tuple(value * direction for value, direction in zip(coordinates, directions))


def pareto(rows):
    """Retain incomplete points as incomparable; never infer a missing objective."""
    groups = {}
    for row in rows:
        if row.get("event") == "finished" and row.get("status") == "VALID_FEASIBLE":
            groups.setdefault(row["group_id"], {})[row["candidate_id"]] = row
    result = {}
    for group, candidates in groups.items():
        points = {name: _coordinates(row) for name, row in candidates.items()}
        if len({row["tpot_applicable"] for row in candidates.values()}) != 1:
            raise ValueError("inconsistent TPOT applicability within a case group")
        frontier = []
        for name in sorted(candidates):
            point = points[name]
            if point is not None and any(
                other is not None
                and all(x <= y for x, y in zip(other, point))
                and any(x < y for x, y in zip(other, point))
                for other in points.values()
            ):
                continue
            equivalent = next(
                (
                    old
                    for old in frontier
                    if point is not None and points[old["candidate_id"]] == point
                ),
                None,
            )
            if equivalent is not None:
                equivalent["equivalent_candidates"].append(name)
            else:
                frontier.append(
                    {
                        **candidates[name],
                        "equivalent_candidates": [],
                        "verification": "EXPLORATORY",
                    }
                )
        result[group] = frontier
    return result
