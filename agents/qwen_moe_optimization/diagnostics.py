"""Bounded host analysis of complete results; never part of simulation execution."""

import heapq
import math

MAX_ITEMS = 16
POLICY_FIELDS = (
    "route_decisions",
    "batch_decisions",
    "kv_decisions",
    "waits",
    "decision_wall_seconds",
    "simulation_end_s",
)
SAMPLE_FIELDS = ("decision", "kind", "time_s", "registered_read_count")
OBSERVATIONS = {
    "pressure": (
        "resource",
        "active_remaining_service_s",
        "waiting_service_s",
        "setup_remaining_service_s",
        "active_demands",
        "waiting_demands",
        "setup_demands",
    ),
    "memory": ("device", "capacity_bytes", "allocated_bytes"),
    "registered_reads": (
        "request_id",
        "layer",
        "target_rank",
        "required_tokens",
        "ready_tokens",
    ),
}


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


def _fields(row, names):
    return {
        name: value
        for name in names
        if (value := row.get(name)) is not None
        and (_number(value) or isinstance(value, str))
    }


def _policy(policy, limit):
    policy = policy if isinstance(policy, dict) else {}
    samples = policy.get("samples") or []
    selected = []
    for sample in samples[:limit]:
        item = _fields(sample, SAMPLE_FIELDS)
        for name, fields in OBSERVATIONS.items():
            rows = sample.get(name) or []
            item[name] = [_fields(row, fields) for row in rows[:limit]]
            item[name + "_count"] = len(rows)
        selected.append(item)
    return {
        "summary": _fields(policy, POLICY_FIELDS),
        "sample_count": len(samples),
        "samples": selected,
        "samples_truncated": len(samples) > limit,
        "scope": "First bounded decision samples, not whole-service coverage",
    }


def summarize_diagnostics(run, metrics, limit=8):
    """Return JSON data from a trusted results.Run and its recomputed metrics.

    Request decomposition follows the single-LLM search contract. Missing native
    completion or observations remain explicit; timing never supplies utilization.
    """
    if type(limit) is not int or not 1 <= limit <= MAX_ITEMS:
        raise ValueError(f"diagnostic limit must be an integer in 1..{MAX_ITEMS}")
    rows = list(run.requests.values())
    arrivals = [row["arrival_time"] for row in rows]
    origin = min(arrivals, default=None)
    last_arrival = max(arrivals, default=None)
    limits = (metrics.get("slo") or {}).get("limits") or {}
    ttft_limit, tpot_limit = (
        limits.get(name) for name in ("ttft_limit_s", "tpot_limit_s")
    )
    slo_available = all(
        _number(value) and value > 0 for value in (ttft_limit, tpot_limit)
    )
    requests, failures = [], []
    unsupported = zero_output = 0
    for row in rows:
        nodes = row["sub_requests"]
        if len(nodes) != 1 or nodes[0]["kind"] != "llm":
            unsupported += 1
            continue
        node = nodes[0]
        latencies = node["token_latency"]
        if not latencies:
            zero_output += 1
            continue
        start = row["arrival_time"] + (node.get("interval") or 0)
        elapsed = math.fsum(latencies)
        post_first = math.fsum(latencies[1:])
        tpot = post_first / (len(latencies) - 1) if len(latencies) > 1 else None
        request = {
            "request_id": row["request_id"],
            "input_tokens": node["input_tokens"],
            "output_tokens": node["output_tokens"],
            "arrival_s": row["arrival_time"],
            "start_s": start,
            "ttft_s": latencies[0],
            "first_s": start + latencies[0],
            "last_s": start + elapsed,
            "e2e_s": start + elapsed - row["arrival_time"],
            "post_first_span_s": post_first,
            "tpot_s": tpot,
        }
        requests.append(request)
        if slo_available:
            reasons = []
            if latencies[0] > ttft_limit:
                reasons.append("ttft")
            if tpot is not None and tpot > tpot_limit:
                reasons.append("tpot")
            if reasons:
                failures.append({**request, "failed_metrics": reasons})

    last_output = max((row["last_s"] for row in requests), default=None)
    native_finish = run.service_finish_s
    finish = native_finish if native_finish is not None else last_output
    if finish is not None and last_output is not None:
        finish = max(finish, last_output)
    # An unsupported DAG may finish later; a partial decomposition is not a window.
    if unsupported:
        last_output = None
        finish = native_finish
    if zero_output and native_finish is None:
        finish = None
    policy = _policy(metrics.get("policy"), limit)
    missing = ["resource_causal_attribution", "gpu_utilization"]
    if native_finish is None:
        missing.append("native_service_finish")
    if not slo_available:
        missing.append("slo_limits")
    if not policy["summary"]:
        missing.append("policy_statistics")
    if not policy["samples"]:
        missing.append("policy_samples")
    if unsupported:
        missing.append("non_single_llm_request_decomposition")
    if zero_output:
        missing.append("zero_output_request_completion")

    return {
        "schema_version": 1,
        "source": "host_complete_result_analysis",
        "limit": limit,
        "request_count": len(rows),
        "timed_request_count": len(requests),
        "zero_output_requests": zero_output,
        "unsupported_requests": unsupported,
        "window": {
            "origin_s": origin,
            "last_arrival_s": last_arrival,
            "arrival_span_s": last_arrival - origin if origin is not None else None,
            "last_output_s": last_output,
            "native_finish_s": native_finish,
            "finish_s": finish,
            "duration_s": finish - origin
            if finish is not None and origin is not None
            else None,
            "drain_s": finish - last_arrival
            if finish is not None and last_arrival is not None
            else None,
            "native_after_last_output_s": native_finish - last_output
            if native_finish is not None and last_output is not None
            else None,
        },
        "last_completing_requests": heapq.nlargest(
            limit, requests, key=lambda row: (row["last_s"], -row["request_id"])
        ),
        "longest_requests": heapq.nlargest(
            limit, requests, key=lambda row: (row["e2e_s"], -row["request_id"])
        ),
        "slo_failure_count": len(failures)
        if slo_available and not unsupported
        else None,
        "slo_failures": heapq.nlargest(
            limit, failures, key=lambda row: (row["e2e_s"], -row["request_id"])
        ),
        "policy": policy,
        "missing_evidence": missing,
        "limitations": [
            "Post-first-token span includes execution and waiting, not pure GPU compute",
            "Resource pressure is remaining full-capacity service demand, not utilization",
            "Registered KV reads need not be current critical-path stalls",
            "Request timings and bounded samples do not establish a resource bottleneck",
        ],
    }
