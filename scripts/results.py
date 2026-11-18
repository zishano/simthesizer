"""Shared result loading, DAG timing, and serving metrics. All times are seconds."""

import argparse
import copy
import json
import math
import statistics
import tomllib
from collections import deque
from dataclasses import dataclass
from itertools import accumulate
from pathlib import Path

LATENCIES = {
    "ttft": "TTFT",
    "tpot": "TPOT",
    "itl": "ITL",
    "e2e": "LLM E2E",
    "normalized_e2e": "LLM E2E / output token",
    "request_e2e": "Request DAG E2E",
}
SCALARS = {
    "makespan": "s",
    "output_throughput": "token/s",
    "request_throughput": "request/s",
}
TOKEN_RATES = {
    "tokens_per_user": "token/s/user",
    "tokens_per_gpu": "token/s/GPU",
}


def metric_unit(metric):
    if metric == "normalized_e2e":
        return "s/token"
    return TOKEN_RATES.get(metric, SCALARS.get(metric, "s"))


def number(value, field):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{field} must be finite and non-negative")
    return value


def integer(value, field):
    if type(value) is not int or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def indices(value, size, field):
    if not isinstance(value, list):
        raise TypeError(f"{field} must be a list")
    for index in value:
        if integer(index, field) >= size:
            raise ValueError(f"{field} index {index} is out of range")
    if len(set(value)) != len(value):
        raise ValueError(f"{field} contains duplicate indices")
    return sorted(value)


def parse_request(row):
    if not isinstance(row, dict):
        raise TypeError("expected a request object")
    integer(row.get("request_id"), "request_id")
    number(row.get("arrival_time"), "arrival_time")
    nodes = row.get("sub_requests")
    if not isinstance(nodes, list) or not nodes:
        raise ValueError("sub_requests must be a non-empty list")
    row["initial"] = indices(row.get("initial"), len(nodes), "initial")
    for index, node in enumerate(nodes):
        if not isinstance(node, dict) or node.get("kind") not in ("llm", "tool_call"):
            raise ValueError(f"sub-request {index}: expected llm or tool_call")
        for field in ("input_tokens", "output_tokens"):
            integer(node.get(field), field)
        node["next"] = indices(node.get("next"), len(nodes), "next")
        node["interval"] = number(
            0 if node.get("interval") is None else node["interval"], "interval"
        )
        if node["kind"] == "tool_call":
            node["duration"] = number(node.get("duration"), "duration")
            continue
        latencies = node.get("token_latency")
        if not isinstance(latencies, list) or len(latencies) != node["output_tokens"]:
            raise ValueError(
                f"sub-request {index}: token_latency must contain exactly "
                f"{node['output_tokens']} entries; only complete results are supported"
            )
        for value in latencies:
            number(value, "token_latency")
        for field, count in (
            ("input_token_ids", "input_tokens"),
            ("output_token_ids", "output_tokens"),
        ):
            tokens = node.get(field)
            if tokens is not None:
                if not isinstance(tokens, list) or len(tokens) != node[count]:
                    raise ValueError(f"{field} length differs from {count}")
                for token in tokens:
                    integer(token, field)
    timeline(row)
    return row


def timeline(row):
    """A node starts after all parents finish, plus its own interval."""
    nodes = row["sub_requests"]
    degrees = [0] * len(nodes)
    starts = [row["arrival_time"]] * len(nodes)
    ends = [0.0] * len(nodes)
    for node in nodes:
        for child in node["next"]:
            degrees[child] += 1
    roots = [i for i, degree in enumerate(degrees) if degree == 0]
    if row["initial"] != roots:
        raise ValueError("initial does not match DAG roots")
    queue = deque(roots)
    visited = 0
    while queue:
        index = queue.popleft()
        node = nodes[index]
        starts[index] += node["interval"]
        duration = (
            math.fsum(node["token_latency"])
            if node["kind"] == "llm"
            else node["duration"]
        )
        ends[index] = number(starts[index] + duration, "completion time")
        visited += 1
        for child in node["next"]:
            starts[child] = max(starts[child], ends[index])
            degrees[child] -= 1
            if degrees[child] == 0:
                queue.append(child)
    if visited != len(nodes):
        raise ValueError("request DAG contains a cycle")
    return starts, ends


@dataclass
class Run:
    label: str
    path: Path
    requests: dict
    gpu_count: int | None = None
    service_finish_s: float | None = None

    @property
    def origin(self):
        return min(row["arrival_time"] for row in self.requests.values())

    def token_times(self):
        for row in self.requests.values():
            starts, _ = timeline(row)
            for index, node in enumerate(row["sub_requests"]):
                if node["kind"] == "llm":
                    for elapsed in accumulate(node["token_latency"]):
                        yield starts[index] + elapsed


def physical_gpu_count(config):
    """Count allocated devices, excluding unused topology GPUs; None if unknown."""
    pending = [config.get("system", {})]
    allocations = []
    sharing = False
    while pending:
        system = pending.pop()
        sharing |= bool(system.get("allow_gpu_sharing", False))
        children = [
            child
            for pool in ("instances", "prefill_instances", "decode_instances")
            for child in system.get(pool, [])
        ]
        if children:
            pending.extend(children)
        else:
            allocations.append(system.get("model_runner", {}).get("hardware", {}))
    ids = [hardware.get("gpu_ids") for hardware in allocations]
    if all(group is not None for group in ids):
        devices = set()
        for hardware, group in zip(allocations, ids):
            if (
                not isinstance(group, list)
                or not group
                or any(not isinstance(device, str) or not device for device in group)
                or len(set(group)) != len(group)
                or hardware.get("gpu_count", len(group)) != len(group)
            ):
                return None
            devices.update(group)
        return len(devices) or None
    if any(group is not None for group in ids) or sharing:
        return None
    counts = [hardware.get("gpu_count") for hardware in allocations]
    if counts and all(type(count) is int and count > 0 for count in counts):
        return sum(counts)
    return None


def load_run(spec):
    label, separator, filename = spec.partition("=")
    path = Path(filename if separator else spec)
    label = label if separator else path.stem
    if not label or (separator and not filename):
        raise ValueError("expected [LABEL=]PATH")
    requests = {}
    with path.open() as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                row = parse_request(json.loads(line))
                request_id = row["request_id"]
                if request_id in requests:
                    raise ValueError(f"duplicate request_id {request_id}")
                requests[request_id] = row
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
    if not requests:
        raise ValueError(f"{path}: empty result")
    config = path.with_name("config.toml")
    gpu_count = None
    if config.is_file():
        with config.open("rb") as source:
            gpu_count = physical_gpu_count(tomllib.load(source))
    stats = path.with_name("policy-stats.json")
    service_finish = None
    if stats.is_file():
        summary = json.loads(stats.read_text())
        if summary is not None and not isinstance(summary, dict):
            raise ValueError("policy-stats.json must contain an object or null")
        service_finish = (summary or {}).get("simulation_end_s")
        if service_finish is not None:
            number(service_finish, "simulation_end_s")
    return Run(label, path, dict(sorted(requests.items())), gpu_count, service_finish)


def match_runs(reference, candidate):
    """Require the same workload; optional traces are checked when both exist."""
    if reference.requests.keys() != candidate.requests.keys():
        raise ValueError(
            f"{candidate.label}: request IDs differ from {reference.label}"
        )
    for request_id, left in reference.requests.items():
        right = candidate.requests[request_id]
        prefix = f"{candidate.label}: request {request_id}"
        for field in ("arrival_time", "initial"):
            if left[field] != right[field]:
                raise ValueError(f"{prefix}: {field} differs")
        if len(left["sub_requests"]) != len(right["sub_requests"]):
            raise ValueError(f"{prefix}: sub-request count differs")
        for index, (a, b) in enumerate(
            zip(left["sub_requests"], right["sub_requests"])
        ):
            for field in (
                "kind",
                "input_tokens",
                "output_tokens",
                "next",
                "interval",
                "duration",
            ):
                if a.get(field) != b.get(field):
                    raise ValueError(f"{prefix}, sub-request {index}: {field} differs")
            for field in ("model", "input_token_ids", "output_token_ids"):
                if (
                    a.get(field) is not None
                    and b.get(field) is not None
                    and a[field] != b[field]
                ):
                    raise ValueError(f"{prefix}, sub-request {index}: {field} differs")


def load_runs(specs):
    runs = [load_run(spec) for spec in specs]
    if len({run.label for run in runs}) != len(runs):
        raise ValueError("duplicate run labels; use LABEL=PATH to distinguish them")
    for run in runs[1:]:
        match_runs(runs[0], run)
    return runs


def measurements(run, service_finish_s=None):
    samples = {metric: {} for metric in (*LATENCIES, *TOKEN_RATES)}
    finish = run.origin
    output_tokens = 0
    for request_id, row in run.requests.items():
        _, ends = timeline(row)
        finish = max(finish, max(ends))
        samples["request_e2e"][(request_id,)] = max(ends) - row["arrival_time"]
        for index, node in enumerate(row["sub_requests"]):
            if node["kind"] != "llm" or not node["token_latency"]:
                continue
            latencies = node["token_latency"]
            key = (request_id, index)
            output_tokens += len(latencies)
            samples["ttft"][key] = latencies[0]
            samples["e2e"][key] = math.fsum(latencies)
            samples["normalized_e2e"][key] = math.fsum(latencies) / len(latencies)
            if len(latencies) > 1:
                tpot = statistics.fmean(latencies[1:])
                samples["tpot"][key] = tpot
                if tpot > 0 and math.isfinite(rate := 1 / tpot):
                    samples["tokens_per_user"][key] = rate
            for token, latency in enumerate(latencies[1:], 1):
                samples["itl"][(*key, token)] = latency
    service_finish_s = (
        run.service_finish_s if service_finish_s is None else service_finish_s
    )
    if service_finish_s is not None:
        number(service_finish_s, "service_finish_s")
        if service_finish_s + 1e-9 * max(1, finish) < finish:
            raise ValueError("native service finish precedes observed token completion")
        finish = max(finish, service_finish_s)
    duration = finish - run.origin
    samples["makespan"] = {(): duration}
    samples["output_throughput"] = {(): output_tokens / duration} if duration else {}
    if duration and run.gpu_count:
        samples["tokens_per_gpu"] = {(): output_tokens / duration / run.gpu_count}
    samples["request_throughput"] = (
        {(): len(run.requests) / duration} if duration else {}
    )
    return samples


def percentile(sorted_values, percent):
    position = (len(sorted_values) - 1) * percent / 100
    low = math.floor(position)
    high = math.ceil(position)
    return sorted_values[low] + (sorted_values[high] - sorted_values[low]) * (
        position - low
    )


def summarize(samples, percentiles):
    values = sorted(samples.values())
    return {
        "mean": statistics.fmean(values) if values else None,
        **{f"p{p:g}": percentile(values, p) if values else None for p in percentiles},
    }


def summarize_run(run, service_finish_s=None):
    samples = measurements(run, service_finish_s)
    metrics = {
        name: {"unit": metric_unit(name), **summarize(values, [50, 95, 99])}
        for name, values in samples.items()
    }
    metrics["tokens_per_user"]["samples"] = len(samples["tokens_per_user"])
    metrics["tokens_per_gpu"]["gpu_count"] = run.gpu_count
    return metrics


def load_workload(path):
    """Validate an input trace without inventing any execution timing."""
    rows = []
    seen = set()
    with Path(path).open() as source:
        for index, line in enumerate(source, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            checked = copy.deepcopy(row)
            for node in checked.get("sub_requests", []):
                if node.get("kind") == "llm":
                    count = integer(node.get("output_tokens"), "output_tokens")
                    node["token_latency"] = [0.0] * count
            try:
                parse_request(checked)
                if row["request_id"] in seen:
                    raise ValueError("duplicate request_id")
            except (ValueError, TypeError, KeyError) as error:
                raise ValueError(f"{path}:{index}: {error}") from error
            seen.add(row["request_id"])
            rows.append(row)
    if not rows:
        raise ValueError(f"{path}: empty workload")
    return rows


def validate_complete(run, workload):
    """Match every result against its immutable input, including zero-output work."""
    expected = {row["request_id"]: row for row in workload}
    if len(expected) != len(workload) or expected.keys() != run.requests.keys():
        raise ValueError("result request IDs do not exactly match the workload")
    outputs = 0
    for identity, row in expected.items():
        result = run.requests[identity]
        if (
            row["arrival_time"] != result["arrival_time"]
            or sorted(row["initial"]) != result["initial"]
        ):
            raise ValueError(f"request {identity}: arrival/initial differs")
        if len(row["sub_requests"]) != len(result["sub_requests"]):
            raise ValueError(f"request {identity}: sub-request count differs")
        for index, (source, actual) in enumerate(
            zip(row["sub_requests"], result["sub_requests"])
        ):
            fields = ("kind", "input_tokens", "output_tokens", "model", "duration")
            if any(source.get(key) != actual.get(key) for key in fields):
                raise ValueError(f"request {identity}/{index}: task content differs")
            if (
                sorted(source["next"]) != actual["next"]
                or (source.get("interval") or 0) != actual["interval"]
            ):
                raise ValueError(
                    f"request {identity}/{index}: dependency/interval differs"
                )
            if source["kind"] == "llm":
                outputs += source["output_tokens"]
                # Native results omit token IDs. When present, they must agree;
                # the evaluator separately hashes the entire input, including IDs.
                for key in ("input_token_ids", "output_token_ids"):
                    if key in actual and actual[key] != source.get(key):
                        raise ValueError(f"request {identity}/{index}: {key} differs")
    return {
        "expected_requests": len(workload),
        "completed_requests": len(run.requests),
        "output_tokens": outputs,
    }


def slo_metrics(run, slo, service_finish_s=None):
    """Joint per-request SLO goodput for the single-LLM policy-search contract."""
    limits = [slo["ttft_limit_s"], slo["tpot_limit_s"]]
    fraction = slo["min_joint_pass_fraction"]
    if any(
        type(x) not in (int, float) or not math.isfinite(x) or x <= 0 for x in limits
    ):
        raise ValueError("SLO latency limits must be finite and positive")
    if (
        type(fraction) not in (int, float)
        or not math.isfinite(fraction)
        or not 0 < fraction <= 1
    ):
        raise ValueError("SLO pass fraction must be in (0, 1]")
    eligible = passed = passed_tokens = excluded = single_token = 0
    for row in run.requests.values():
        if len(row["sub_requests"]) != 1 or row["sub_requests"][0]["kind"] != "llm":
            raise ValueError("policy-search SLO scoring requires one LLM per request")
        tokens = row["sub_requests"][0]["token_latency"]
        if not tokens:
            excluded += 1
            continue
        eligible += 1
        single_token += len(tokens) == 1
        if tokens[0] <= limits[0] and (
            len(tokens) == 1 or statistics.fmean(tokens[1:]) <= limits[1]
        ):
            passed += 1
            passed_tokens += len(tokens)
    service_finish_s = (
        run.service_finish_s if service_finish_s is None else service_finish_s
    )
    if excluded and service_finish_s is None:
        raise ValueError(
            "zero-output requests require the native service completion time"
        )
    duration = measurements(run, service_finish_s)["makespan"][()]
    available = eligible > 0 and duration > 0 and bool(run.gpu_count)
    achieved = passed / eligible if eligible else None
    goodput = passed_tokens / duration if available else None
    if goodput is not None and not math.isfinite(goodput):
        available, goodput = False, None
    return {
        "limits": dict(slo),
        "eligible_requests": eligible,
        "passed_requests": passed,
        "excluded_zero_output_requests": excluded,
        "single_output_requests": single_token,
        "joint_pass_fraction": achieved,
        "passed_output_tokens": passed_tokens,
        "duration_s": duration,
        "gpu_count": run.gpu_count,
        "available": available,
        "goodput": goodput,
        "goodput_per_gpu": goodput / run.gpu_count if available else None,
        "feasible": bool(available and achieved >= fraction),
        "tpot_applicable": eligible > single_token,
    }


def relative_error(reference, candidate):
    if reference is None or candidate is None or reference == 0:
        return None
    return 100 * (candidate - reference) / reference


def paired_errors(reference, candidate):
    if reference.keys() != candidate.keys():
        raise ValueError("metric sample IDs differ")
    differences = [candidate[key] - value for key, value in reference.items()]
    absolute = [abs(value) for value in differences]
    percentages = [
        abs(relative_error(value, candidate[key]))
        for key, value in reference.items()
        if value > 0
    ]
    total = math.fsum(reference.values())
    return {
        "count": len(differences),
        "mae": statistics.fmean(absolute) if absolute else None,
        "bias": statistics.fmean(differences) if differences else None,
        "mape_pct": statistics.fmean(percentages) if percentages else None,
        "mape_count": len(percentages),
        "wape_pct": 100 * math.fsum(absolute) / total if total else None,
    }


def throughput(run, window, end):
    """Fixed-width [left, right) bins on the shared arrival-relative clock."""
    edges = [i * window for i in range(math.floor(end / window) + 2)]
    counts = [0] * (len(edges) - 1)
    origin = run.origin
    for time in run.token_times():
        counts[min(math.floor((time - origin) / window), len(counts) - 1)] += 1
    return edges, [count / window for count in counts]


def positive_float(value):
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return parsed


def percent(value):
    parsed = float(value)
    if not math.isfinite(parsed) or not 0 <= parsed <= 100:
        raise argparse.ArgumentTypeError("must be between 0 and 100")
    return parsed


def protect_output(path, runs):
    inputs = {
        source.resolve()
        for run in runs
        for source in (run.path, run.path.with_name("config.toml"))
    }
    if path.resolve() in inputs:
        raise ValueError("output must not overwrite an input result or config")


def table(headers, rows):
    def display(value):
        if value is None:
            return "n/a"
        return f"{value:.6g}" if isinstance(value, float) else str(value)

    lines = [headers, *[[display(value) for value in row] for row in rows]]
    widths = [max(len(row[i]) for row in lines) for i in range(len(headers))]
    return "\n".join(
        "  ".join(value.ljust(width) for value, width in zip(row, widths)).rstrip()
        for row in lines
    )
