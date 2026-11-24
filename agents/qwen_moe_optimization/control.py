"""Host-owned Qwen research queue, isolated candidate execution, and accounting."""

import argparse
import concurrent.futures
import fcntl
import hashlib
import json
import os
import re
import shutil
import signal
import stat
import subprocess
import sys
import threading
import time
import tomllib
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "agents/find_pareto_frontier"))
import archive
import baseline_store
import candidate
import contract
import launcher as infra
import objectives
import research
import runtime

REL = Path("agents/qwen_moe_optimization")
LABEL = "simthesizer.qwen.run"
REQUEST_ID = re.compile(r"[a-f0-9]{32}")
SOURCE_ID = re.compile(r"[a-f0-9]{64}")


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(infra.json_bytes(value))
    temporary.replace(path)


def read_json(path, limit=65536):
    """Read a queue file without following links or unbounded allocation."""
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_size > limit:
            raise ValueError("request must be a bounded regular file")
        data = stream.read(limit + 1)
    if len(data) > limit:
        raise ValueError("request exceeds size limit")
    try:
        return json.loads(data)
    except RecursionError as error:
        raise ValueError("request JSON nesting exceeds decoder limits") from error


def events(run):
    book = run / "artifacts/research.jsonl"
    return (
        [json.loads(line) for line in book.read_text().splitlines()]
        if book.exists()
        else []
    )


def append(run, row):
    with (run / "artifacts/research.jsonl").open("a") as stream:
        stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def validate_request(request, manifest, request_id):
    if not isinstance(request, dict) or request.get("id") != request_id:
        raise ValueError("request identity mismatch")
    for key in ("id", "case", "operation", "policy_kind"):
        if not isinstance(request.get(key), str):
            raise ValueError(f"{key} must be a string")  # noqa: TRY004 - protocol rejection
    allowed = {
        "id",
        "case",
        "operation",
        "candidate_id",
        "policy_kind",
        "hypothesis",
        "bottleneck",
        "evidence",
        "ablation_of",
        "plan",
    }
    if set(request) - allowed:
        raise ValueError("unsupported request fields")
    if request.get("operation") not in {"evaluate", "replay"}:
        raise ValueError("operation must be evaluate or replay")
    if request.get("case") not in {row["id"] for row in baseline_store.cases(manifest)}:
        raise ValueError("case is not in the frozen matrix")
    if request.get("policy_kind") not in {"reference", "candidate"}:
        raise ValueError("unsupported policy kind")
    for key in ("hypothesis", "bottleneck", "evidence", "ablation_of"):
        value = request.get(key, "")
        if not isinstance(value, str) or len(value) > 8192:
            raise ValueError(f"invalid {key}")
        if key in {"hypothesis", "bottleneck"} and not value.strip():
            raise ValueError(f"{key} is required")
    if request.get("ablation_of") and not SOURCE_ID.fullmatch(request["ablation_of"]):
        raise ValueError("invalid ablation candidate identity")
    plan = request.get("plan")
    if plan is not None and (
        not isinstance(plan, dict)
        or len(json.dumps(plan, ensure_ascii=False).encode()) > contract.MAX_PLAN_BYTES
    ):
        raise ValueError("plan must be a bounded JSON object")
    if request["operation"] == "replay":
        if plan is not None:
            raise ValueError("replay uses the original sealed plan")
        if not isinstance(request.get("candidate_id"), str) or not SOURCE_ID.fullmatch(
            request["candidate_id"]
        ):
            raise ValueError("replay requires a sealed candidate identity")
    elif request.get("candidate_id") is not None:
        raise ValueError("evaluate snapshots the current worktree")
    return request


def initialize(run):
    for name in (
        "queue",
        "notes",
        "research-cache",
        "candidate-cache",
        "artifacts/responses",
        "artifacts/mechanisms",
        "artifacts/evaluations",
        "jobs",
    ):
        (run / name).mkdir(parents=True, exist_ok=True)
    if not (run / "worktree").exists():
        temporary = run / "worktree.preparing"
        if temporary.exists():
            shutil.rmtree(temporary)
        shutil.copytree(run / "source", temporary)
        temporary.rename(run / "worktree")


def docker_arguments(
    run, manifest, image, *, job=None, env_file=None, trial_cache=None
):
    limits = manifest["experiment"]["container"]
    label = hashlib.sha256(str(run.resolve()).encode()).hexdigest()
    args = [
        "docker",
        "create",
        "--init",
        "--read-only",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "--user",
        f"{os.getuid()}:{os.getgid()}",
        "--cpus",
        str(limits["cpus"]),
        "--memory",
        f"{limits['memory_mib']}m",
        "--memory-swap",
        f"{limits['memory_mib']}m",
        "--pids-limit",
        str(limits["pids_limit"]),
        "--label",
        f"{LABEL}={label}",
        "--network",
        "none" if job else manifest["network"],
        "--env",
        "PYTHONDONTWRITEBYTECODE=1",
    ]
    if job is None:
        if env_file is None:
            raise ValueError("research container requires an env file")
        args += ["--env-file", str(env_file)]
        if manifest["network"] == "bridge":
            args += ["--add-host", "host.docker.internal:host-gateway"]
    for path, mode in (("/tmp", "1777"), ("/home/agent", "0700")):
        args += [
            "--tmpfs",
            f"{path}:rw,exec,nosuid,nodev,size=512m,uid={os.getuid()},gid={os.getgid()},mode={mode}",
        ]
    base = job or run
    mounts = [
        (base / "source", "/workspace/simthesizer", True),
        (base / "task", "/workspace/task", True),
        (base / "inputs", "/workspace/inputs", True),
        (base / "baseline-policy", "/workspace/baseline-policy", True),
        (base / "artifacts", "/workspace/artifacts", job is None),
        (
            (trial_cache or base / "cache") if job else run / "research-cache",
            "/workspace/cache",
            False,
        ),
    ]
    if job is None:
        mounts += [
            (run / name, f"/workspace/{name}", False)
            for name in ("worktree", "queue", "notes")
        ]
    for source, target, readonly in mounts:
        args += [
            "--mount",
            f"type=bind,source={source},target={target}"
            + (",readonly" if readonly else ""),
        ]
    return [
        *args,
        "--entrypoint",
        "/bin/bash",
        image,
        f"/workspace/simthesizer/{REL}/entrypoint.sh",
        "trial" if job else "research",
    ]


class Containers:
    """Own exactly the IDs created by this run; never prune global Docker state."""

    def __init__(self, run, state):
        self.run, self.state = run, state
        self.lock = threading.Lock()

    def update(self, **values):
        with self.lock:
            self.state.update(values)
            save(self.run / "supervisor.json", self.state)

    def record(self, identifier, add):
        with self.lock:
            active = self.state.setdefault("active", [])
            if add:
                active.append(identifier)
            elif identifier in active:
                active.remove(identifier)
            save(self.run / "supervisor.json", self.state)

    def create(self, args, *, deadline=None):
        timeout = 60 if deadline is None else min(60, deadline - time.monotonic())
        if timeout <= 0:
            raise TimeoutError("container creation budget exhausted")
        try:
            identifier = subprocess.check_output(
                args, text=True, timeout=timeout
            ).strip()
            if not SOURCE_ID.fullmatch(identifier):
                raise ValueError("Docker returned an invalid container ID")
        except (ValueError, OSError, subprocess.SubprocessError) as error:
            # Docker may create a container before losing its response. Preserve
            # the recorded research container while removing unrecorded orphans.
            try:
                self.recover(orphans_only=True)
            except (ValueError, OSError, subprocess.SubprocessError) as cleanup:
                error.add_note(f"owned-container recovery requires retry: {cleanup}")
            raise
        self.record(identifier, True)
        return identifier

    def remove(self, identifier):
        # A failed stop leaves the owned ID recorded for recovery.
        infra.stop_container(identifier)
        subprocess.run(
            ["docker", "rm", identifier],
            check=True,
            timeout=15,
            stdout=subprocess.DEVNULL,
        )
        self.record(identifier, False)

    def recover(self, *, orphans_only=False):
        expected = hashlib.sha256(str(self.run.resolve()).encode()).hexdigest()
        recorded = list(self.state.get("active") or [])
        found = subprocess.run(
            [
                "docker",
                "ps",
                "--all",
                "--quiet",
                "--no-trunc",
                "--filter",
                f"label={LABEL}={expected}",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
        if found.returncode:
            raise ValueError(
                "cannot discover owned containers; retry when Docker is available"
            )
        discovered = found.stdout.split()
        if any(
            not SOURCE_ID.fullmatch(identifier)
            for identifier in [*recorded, *discovered]
        ):
            raise ValueError("invalid recorded or discovered container ID")
        selected = (
            [identifier for identifier in discovered if identifier not in recorded]
            if orphans_only
            else list(dict.fromkeys([*recorded, *discovered]))
        )
        for identifier in selected:
            inspected = subprocess.run(
                ["docker", "inspect", identifier],
                check=False,
                capture_output=True,
                text=True,
                timeout=15,
            )
            if inspected.returncode:
                # Distinguish an absent ID from an unavailable daemon.
                if "No such" in inspected.stderr:
                    self.record(identifier, False)
                    continue
                raise ValueError(
                    "cannot inspect a recorded container; retry when Docker is available"
                )
            labels = json.loads(inspected.stdout)[0]["Config"].get("Labels") or {}
            if labels.get(LABEL) != expected:
                raise ValueError("recorded container ownership mismatch")
            if identifier not in (self.state.get("active") or []):
                self.record(identifier, True)
            self.remove(identifier)

    def run_trial(self, args, log, deadline, cancel):
        identifier = self.create(args, deadline=deadline)
        process = None
        try:
            with log.open("wb") as stream:
                process = subprocess.Popen(
                    ["docker", "start", "--attach", identifier],
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                )
                while process.poll() is None:
                    if cancel.wait(0.2) or time.monotonic() >= deadline:
                        raise TimeoutError(
                            "candidate execution interrupted or timed out"
                        )
                if process.returncode:
                    raise ValueError(
                        f"candidate container failed ({process.returncode}); see {log.name}"
                    )
                details = json.loads(
                    subprocess.check_output(
                        [
                            "docker",
                            "inspect",
                            "--format",
                            "{{json .State}}",
                            identifier,
                        ],
                        text=True,
                        timeout=15,
                    )
                )
                if (
                    details.get("Running")
                    or details.get("ExitCode") != 0
                    or details.get("OOMKilled")
                ):
                    raise ValueError("candidate did not exit successfully")
        finally:
            try:
                self.remove(identifier)
            finally:
                if process is not None and process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=5)


def request_budget(run, manifest, operation, baseline):
    starts = [row for row in events(run) if row["event"] == "started"]
    config = manifest["experiment"]
    if operation == "replay":
        selected = [row for row in starts if row.get("operation") == "replay"]
        limit = config["verification"]["max_trials"]
        ends = {row["id"]: row for row in events(run) if row["event"] == "finished"}
        charged = sum(
            ends.get(row["id"], {}).get("wall_seconds", row["budget_wall_seconds"])
            for row in selected
        )
        remaining = config["verification"]["max_wall_seconds"] - charged
    else:
        selected = [row for row in starts if row.get("operation") != "replay"]
        limit = config["search"]["max_trials"] - baseline["accounting"]["attempts"]
        remaining = config["search"]["trial_wall_seconds"]
    if len(selected) >= limit or remaining <= 0:
        raise ValueError(f"{operation} budget exhausted")
    return min(config["search"]["trial_wall_seconds"], remaining)


def evaluate_request(run, manifest, image, request, containers, cancel, deadline):
    from runner import collect_job, prepare_job

    request_id = request["id"]
    started = time.monotonic()
    result = {
        "id": request_id,
        "operation": request["operation"],
        "case_id": request["case"],
        "policy_kind": request["policy_kind"],
        "status": "INVALID",
        "verification": "EXPLORATORY",
    }
    try:
        sealed = run / "artifacts/mechanisms"
        if request["operation"] == "replay":
            candidate_dir = sealed / request["candidate_id"]
            meta = read_json(candidate_dir / "candidate.json", 8 * 1024 * 1024)
            previous = [
                row
                for row in events(run)
                if row.get("event") == "finished"
                and row.get("status") == "EVALUATED"
                and row.get("candidate_id") == request["candidate_id"]
                and row.get("case_id") == request["case"]
                and row.get("policy_kind") == request["policy_kind"]
            ]
            if not previous:
                raise ValueError(
                    "replay requires an evaluated candidate with the same case and policy"
                )
        else:
            staging = sealed / ("pending-" + request_id)
            meta = candidate.snapshot(
                run / "worktree",
                run / "source",
                staging,
                expected_source_sha256=manifest["source_inventory_sha256"],
            )
            candidate_dir = sealed / meta["source_sha256"]
            if candidate_dir.exists():
                existing = read_json(candidate_dir / "candidate.json", 8 * 1024 * 1024)
                if existing != meta:
                    raise ValueError(
                        "sealed candidate metadata differs; start a new experiment"
                    )
                shutil.rmtree(staging)
            else:
                staging.rename(candidate_dir)
        if meta.get("contract_version") != candidate.CONTRACT_VERSION or meta.get(
            "impact"
        ) != candidate.impact(meta["changed_files"]):
            raise ValueError("candidate scope contract differs from frozen controller")
        result["candidate_id"] = meta["source_sha256"]
        result["policy_sha256"] = meta["policy_sha256"]
        plan = request.get("plan")
        if request["operation"] == "replay":
            original_contract = previous[-1].get("contract")
            plan = original_contract["plan"] if original_contract else None
        mechanism_contract = contract.seal(
            run,
            manifest,
            meta,
            plan,
            request["case"],
            request["policy_kind"],
            events(run),
        )
        if request["operation"] == "replay" and mechanism_contract != original_contract:
            raise ValueError("replay contract differs from the evaluated candidate")
        result["contract"] = mechanism_contract
        job = run / "jobs" / request_id
        # Sequential cases of an identical sealed candidate share compilation;
        # clean replay always starts with the job's empty cache.
        trial_cache = None
        if request["operation"] != "replay":
            trial_cache = (
                run
                / "candidate-cache"
                / (meta["source_sha256"] + "-" + request["policy_kind"])
            )
            trial_cache.mkdir(parents=True, exist_ok=True)
        prepare_job(
            run,
            candidate_dir,
            job,
            request["case"],
            policy_kind=request["policy_kind"],
            clean=request["operation"] == "replay",
            cache_path=trial_cache,
            mechanism_contract=mechanism_contract,
        )
        if time.monotonic() >= deadline or cancel.is_set():
            raise TimeoutError("budget exhausted during snapshot preparation")
        args = docker_arguments(
            run, manifest, image, job=job, trial_cache=trial_cache
        ) + ["--case", request["case"]]
        if request["policy_kind"] == "reference":
            args.append("--reference")
        if request["operation"] == "replay":
            args.append("--clean")
        result["execution_started"] = True
        containers.run_trial(args, job / "container.log", deadline, cancel)
        result.update(
            collect_job(
                run,
                job,
                meta,
                request["case"],
                deadline=deadline,
                mechanism_contract=mechanism_contract,
            )
        )
        if result.get("status") == "EVALUATED":
            result.update(contract.compare(mechanism_contract, result))
        result["verification"] = "EXPLORATORY"
        destination = run / "artifacts/evaluations" / request_id
        if result.get("status") == "EVALUATED":
            shutil.copytree(job / "collected", destination)
            result["evidence_directory"] = (
                f"/workspace/artifacts/evaluations/{request_id}"
            )
        else:
            destination.mkdir(parents=True, exist_ok=True)
        if result.get("status") == "EVALUATED" and request["operation"] == "replay":
            expected = previous[-1]["evidence_sha256"]["result.jsonl"]
            tail = result["metrics"].get("service_window", {}).get("finish_s")
            prior_tail = (
                previous[-1]["metrics"].get("service_window", {}).get("finish_s")
            )
            if (
                result["evidence_sha256"]["result.jsonl"] != expected
                or tail != prior_tail
            ):
                raise ValueError(
                    "clean replay result differs from the evaluated result"
                )
            result["replay"] = "MATCHED"
            result["replayed_request"] = previous[-1]["id"]
        if time.monotonic() >= deadline or cancel.is_set():
            raise TimeoutError("budget exhausted during collection")
    except Exception as error:  # noqa: BLE001 - a failed candidate must yield a receipt
        result.update(
            status="INVALID",
            reason=str(error),
            replay="FAILED" if request["operation"] == "replay" else None,
        )
    finally:
        # Publish bounded logs even when compilation/checks failed. Never follow
        # a candidate-created link while copying out of its writable mounts.
        job = run / "jobs" / request_id
        destination = run / "artifacts/evaluations" / request_id
        destination.mkdir(parents=True, exist_ok=True)
        from runner import regular

        for relative, name in (
            ("container.log", "container.log"),
            ("artifacts/trusted-checks.log", "trusted-checks.log"),
        ):
            try:
                path = regular(job, relative)
                shutil.copyfile(path, destination / name)
            except ValueError, OSError:
                pass
        # Old evaluator captures build/simulation details in its own logs;
        # runner exports a bounded failure summary to the container log.
        result["logs_directory"] = f"/workspace/artifacts/evaluations/{request_id}"
        result.update(
            event="finished",
            wall_seconds=time.monotonic() - started,
            hypothesis=request["hypothesis"],
            bottleneck=request["bottleneck"],
            evidence=request.get("evidence", ""),
            ablation_of=request.get("ablation_of", ""),
        )
    return result


def update_frontier(run, manifest, baseline):
    rows = []
    for row in baseline["records"]:
        item = {**row, "candidate_id": "baseline", "group_id": row["case"]["id"]}
        if research.enabled(manifest):
            metrics = json.loads(
                (run / "artifacts" / row["directory"] / "metrics.json").read_text()
            )
            item["objectives"] = objectives.from_metrics(metrics)
        rows.append(item)
    for row in events(run):
        if row.get("event") == "finished" and row.get("status") == "EVALUATED":
            rows.append(
                {
                    **row,
                    "status": "VALID_FEASIBLE"
                    if row["feasible"]
                    else "VALID_INFEASIBLE",
                    "candidate_id": row["candidate_id"] + ":" + row["policy_kind"],
                    "group_id": row["case_id"],
                    **(
                        {"objectives": objectives.from_metrics(row["metrics"])}
                        if research.enabled(manifest)
                        else {}
                    ),
                }
            )
    groups = (objectives if research.enabled(manifest) else archive).pareto(rows)
    save(
        run / "artifacts/frontier.json",
        {"verification": "EXPLORATORY", "groups": groups},
    )
    save(
        run / "artifacts/review-candidates.json",
        contract.delivery_candidates(
            events(run), [row["id"] for row in baseline_store.cases(manifest)]
        ),
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    for group, points in groups.items():
        for axis, metric in zip(axes, ("ttft_p99", "tpot_p99")):
            visible = [
                row for row in points if row["objectives"].get(metric) is not None
            ]
            axis.scatter(
                [row["objectives"][metric] for row in visible],
                [row["objectives"]["goodput_per_gpu"] for row in visible],
                label=group,
            )
    for axis, metric in zip(axes, ("TTFT p99 (s)", "TPOT p99 (s)")):
        axis.set(xlabel=metric, ylabel="SLO goodput (output token/s/GPU)")
        axis.grid(alpha=0.25)
        if groups:
            axis.legend(fontsize=7)
    figure.suptitle(
        "Exploratory frontier; mechanism semantics require dedicated validation"
    )
    figure.tight_layout()
    figure.savefig(run / "artifacts/frontier.png", dpi=140)
    plt.close(figure)


def baseline_diagnostics(run, manifest, baseline):
    from diagnostics import summarize_diagnostics
    from runner import trusted_results

    results = trusted_results(run, manifest)
    records = []
    for row in baseline["records"]:
        directory = run / "artifacts" / row["directory"]
        metrics = json.loads((directory / "metrics.json").read_text())
        measured = results.load_run(str(directory / "result.jsonl"))
        records.append(
            {
                "case_id": row["case"]["id"],
                "evidence_sha256": row["evidence_sha256"],
                "objectives": objectives.from_metrics(metrics),
                "diagnostics": summarize_diagnostics(measured, metrics),
            }
        )
    save(run / "artifacts/baseline-diagnostics.json", {"records": records})


def research_status(run, manifest, baseline, wall_remaining):
    snapshot = research.status(manifest, baseline, events(run), wall_remaining)
    snapshot["generated_at_unix_s"] = time.time()
    save(run / "artifacts/research-status.json", snapshot)
    return snapshot


def queued_requests(run, seen):
    queued = []
    with os.scandir(run / "queue") as entries:
        for index, entry in enumerate(entries):
            if index >= 4096:
                raise ValueError("queue entry limit exceeded")
            if (
                entry.name.endswith(".json")
                and REQUEST_ID.fullmatch(entry.name[:-5])
                and entry.name[:-5] not in seen
            ):
                queued.append(entry.name)
    return sorted(queued)


def supervise(run, manifest, image, env_file, baseline):
    state = read_json(run / "supervisor.json")
    if state.get("image_id", image) != image:
        raise ValueError("resume image ID differs from the original run")
    state["active"] = state.get("active") or []
    containers = Containers(run, state)
    containers.recover()
    if state.get("session_started"):
        state["consumed_wall_seconds"] += min(
            state["session_budget"], max(0, time.time() - state["session_started"])
        )
    remaining = (
        manifest["experiment"]["search"]["max_wall_seconds"]
        - state["consumed_wall_seconds"]
        - baseline["accounting"]["wall_seconds"]
    )
    if remaining <= 0:
        raise ValueError("search wall budget exhausted; resume does not reset it")
    # Interrupted attempts stay charged; never reuse a partial result.
    done = {row["id"] for row in events(run) if row["event"] == "finished"}
    for row in events(run):
        if row["event"] == "started" and row["id"] not in done:
            failed = {
                **row,
                "event": "finished",
                "status": "INVALID",
                "reason": "previous controller interrupted",
                "wall_seconds": min(
                    row["budget_wall_seconds"], max(0, time.time() - row["started_at"])
                ),
            }
            append(run, failed)
            save(run / "artifacts/responses" / (row["id"] + ".json"), failed)
    continuous = research.enabled(manifest)
    if continuous:
        baseline_diagnostics(run, manifest, baseline)
        update_frontier(run, manifest, baseline)
    cancel = threading.Event()
    handlers = {
        number: signal.signal(number, lambda *_: cancel.set())
        for number in (signal.SIGINT, signal.SIGTERM)
    }
    start = time.monotonic()
    deadline = start + remaining
    containers.update(
        image_id=image,
        session_started=time.time(),
        session_budget=remaining,
        status="ACTIVE",
    )
    process = None
    identifier = None
    future = None
    exit_code = 1
    outcome = "STOPPED"
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    try:
        session_executions = 0
        if continuous:
            snapshot = research_status(
                run, manifest, baseline, deadline - time.monotonic()
            )
            session_executions = snapshot["executed_attempts"]
        identifier = containers.create(
            docker_arguments(run, manifest, image, env_file=env_file), deadline=deadline
        )
        with (run / "agent.trace.log").open("ab") as log:
            process = subprocess.Popen(
                ["docker", "start", "--attach", identifier],
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            print(f"Agent running: {run}\nTrace: {run / 'agent.trace.log'}", flush=True)
            seen = {row["id"] for row in events(run)}
            next_status_update = time.monotonic() + 5
            while True:
                if cancel.is_set() or time.monotonic() >= deadline:
                    exit_code = 130 if cancel.is_set() else 124
                    outcome = "CANCELLED" if cancel.is_set() else "BUDGET_EXHAUSTED"
                    break
                if continuous and time.monotonic() >= next_status_update:
                    research_status(
                        run, manifest, baseline, deadline - time.monotonic()
                    )
                    next_status_update = time.monotonic() + 5
                exited = process.poll() is not None
                if exited and not continuous:
                    details = json.loads(
                        subprocess.check_output(
                            [
                                "docker",
                                "inspect",
                                "--format",
                                "{{json .State}}",
                                identifier,
                            ],
                            text=True,
                            timeout=15,
                        )
                    )
                    if details.get("Running"):
                        raise ValueError(
                            "Docker attach ended while research container is still running"
                        )
                    exit_code = details["ExitCode"]
                    outcome = "FINISHED" if exit_code == 0 else "STOPPED"
                    break
                if future is not None and future.done():
                    result = future.result()
                    append(run, result)
                    save(run / "artifacts/responses" / (result["id"] + ".json"), result)
                    update_frontier(run, manifest, baseline)
                    print(
                        f"{result['id']}: {result['status']} ({result['wall_seconds']:.1f}s)",
                        flush=True,
                    )
                    future = None
                    if continuous:
                        research_status(
                            run, manifest, baseline, deadline - time.monotonic()
                        )
                if future is None:
                    queued = queued_requests(run, seen)
                    for name in queued[:1]:
                        request_id = name[:-5]
                        seen.add(request_id)
                        try:
                            request = validate_request(
                                read_json(run / "queue" / name), manifest, request_id
                            )
                            allowance = request_budget(
                                run, manifest, request["operation"], baseline
                            )
                            allowance = min(allowance, deadline - time.monotonic())
                            if continuous:
                                snapshot = research_status(
                                    run, manifest, baseline, deadline - time.monotonic()
                                )
                                if request["operation"] == "evaluate":
                                    allowance = min(
                                        allowance,
                                        snapshot["budget"][
                                            "exploration_wall_remaining_s"
                                        ],
                                    )
                                    if allowance <= 0:
                                        raise ValueError(
                                            "final replay time reserved; use replay on sealed candidates"
                                        )
                            append(
                                run,
                                {
                                    **request,
                                    "event": "started",
                                    "started_at": time.time(),
                                    "budget_wall_seconds": allowance,
                                },
                            )
                            future = executor.submit(
                                evaluate_request,
                                run,
                                manifest,
                                image,
                                request,
                                containers,
                                cancel,
                                time.monotonic() + allowance,
                            )
                        except (ValueError, OSError) as error:
                            rejected = {
                                "event": "finished",
                                "id": request_id,
                                "status": "REJECTED",
                                "reason": str(error),
                                "wall_seconds": 0,
                            }
                            append(run, rejected)
                            save(run / "artifacts/responses" / name, rejected)
                    if continuous and exited and not queued and future is None:
                        details = json.loads(
                            subprocess.check_output(
                                [
                                    "docker",
                                    "inspect",
                                    "--format",
                                    "{{json .State}}",
                                    identifier,
                                ],
                                text=True,
                                timeout=15,
                            )
                        )
                        if details.get("Running"):
                            raise ValueError(
                                "Docker attach ended while research container is still running"
                            )
                        snapshot = research_status(
                            run, manifest, baseline, deadline - time.monotonic()
                        )
                        progress = snapshot["executed_attempts"] > session_executions
                        stalled = (
                            0 if progress else state.get("stalled_sessions", 0) + 1
                        )
                        containers.update(
                            stalled_sessions=stalled,
                            research_sessions=state.get("research_sessions", 0) + 1,
                            last_agent_exit_code=details["ExitCode"],
                        )
                        decision = research.next_session(
                            snapshot,
                            stalled,
                            manifest["experiment"]["research"].get(
                                "max_stalled_sessions", 3
                            ),
                        )
                        if decision != "CONTINUE":
                            outcome = decision
                            exit_code = 124 if decision == "BUDGET_EXHAUSTED" else 3
                            break
                        containers.remove(identifier)
                        identifier = None
                        session_executions = snapshot["executed_attempts"]
                        print(
                            f"Agent exited; continuing {snapshot['phase']} research (idle exits: {stalled}).",
                            flush=True,
                        )
                        identifier = containers.create(
                            docker_arguments(run, manifest, image, env_file=env_file),
                            deadline=deadline,
                        )
                        process = subprocess.Popen(
                            ["docker", "start", "--attach", identifier],
                            stdout=log,
                            stderr=subprocess.STDOUT,
                        )
                cancel.wait(0.2)
    finally:
        cancel.set()
        try:
            if identifier:
                containers.remove(identifier)
        finally:
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
            executor.shutdown(wait=True, cancel_futures=False)
            if future is not None:
                result = future.result()
                append(run, result)
                save(run / "artifacts/responses" / (result["id"] + ".json"), result)
            for number, previous in handlers.items():
                signal.signal(number, previous)
            containers.update(
                consumed_wall_seconds=state["consumed_wall_seconds"]
                + time.monotonic()
                - start,
                session_started=None,
                status=outcome,
                last_exit_code=exit_code,
            )
    update_frontier(run, manifest, baseline)
    if continuous:
        research_status(run, manifest, baseline, deadline - time.monotonic())
    print(f"Run: {run}\nStatus: {outcome}; results remain EXPLORATORY.")
    return exit_code


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    for name in ("check", "prepare-only", "build-only"):
        modes.add_argument("--" + name, action="store_true")
    modes.add_argument("--resume", type=Path)
    parser.add_argument("--baseline-only", action="store_true")
    parser.add_argument("--experiment", type=Path, default=None)
    parser.add_argument("--runs-root", type=Path)
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--network", choices=("host", "bridge"), default="host")
    parser.add_argument("--buildkit-image", default=infra.BUILDKIT_IMAGE)
    parser.add_argument("--build-proxy")
    parser.add_argument("--build-fallback-proxy")
    for key, default in infra.BUILD_DEFAULTS.items():
        parser.add_argument("--" + key.lower().replace("_", "-"), default=default)
    args = parser.parse_args(argv)
    if args.baseline_only and (
        args.check or args.prepare_only or args.build_only or args.build
    ):
        parser.error("--baseline-only cannot combine with check/prepare/build")
    if args.build and (
        args.check or args.prepare_only or args.build_only or args.resume
    ):
        parser.error("--build is only for a new search")
    versions = {key: getattr(args, key.lower()) for key in infra.BUILD_DEFAULTS}
    if args.build_only:
        infra.build_image(
            ROOT,
            infra.IMAGE_DEFAULT,
            versions,
            args.buildkit_image,
            args.build_proxy,
            args.build_fallback_proxy,
        )
        return 0
    if not (args.check or args.prepare_only or args.baseline_only):
        args.env_file = (
            args.env_file or ROOT / "agents/find_pareto_frontier/agent.env"
        ).resolve()
        infra.check_env_file(args.env_file)
    if args.resume:
        if args.experiment or args.runs_root:
            parser.error("resume uses frozen experiment and run directory")
        run = infra.safe_path(args.resume)
        manifest = infra.load_bundle(run)
        if manifest.get("profile") != "qwen_moe":
            raise ValueError("resume requires a Qwen mechanism experiment")
        # Resume the actual frozen controller, not a later checkout's code.
        frozen = run / "source" / REL / "control.py"
        if Path(__file__).resolve() != frozen.resolve():
            command = [sys.executable, str(frozen), "--resume", str(run)]
            if args.baseline_only:
                command.append("--baseline-only")
            if args.env_file:
                command += ["--env-file", str(args.env_file.resolve())]
            return subprocess.call(command)
    else:
        config_path = args.experiment or HERE / "experiment.toml"
        config = tomllib.loads(config_path.read_text())
        issues = runtime.config_issues(config)
        if not issues:
            for value in [
                config["inputs"]["scenario"],
                *config["inputs"].get("assets", []),
                *(row["path"] for row in config["workloads"]),
            ]:
                try:
                    infra.input_file(ROOT, value)
                except ValueError as error:
                    issues.append(str(error))
        if config.get("agent", {}).get("cli_version") != versions["CODEX_VERSION"]:
            issues.append("agent.cli_version differs from build version")
        required = (
            "runtime.py",
            "control.py",
            "runner.py",
            "candidate.py",
            "contract.py",
            "research.py",
            "diagnostics.py",
            "objectives.py",
            "submit.py",
            "entrypoint.sh",
            "SEARCH_TASK.md",
            "validation/resource_invariants.rs",
            "references/compute-data-movement.txt",
        )
        gaps = [name for name in required if not (HERE / name).is_file()]
        if args.check:
            print(
                json.dumps(
                    {
                        "configuration_issues": issues,
                        "runtime_gaps": gaps,
                        "launch_ready": not issues and not gaps,
                        "search_scope": "cross_module_mechanisms_and_policies",
                        "scope_contract": candidate.CONTRACT_VERSION,
                        "validation": "exploratory",
                        "integration_executed": False,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 2 if issues or gaps else 0
        if issues or gaps:
            raise ValueError("; ".join(issues + gaps))
        runs_root = infra.resolve_runs_root(
            ROOT, config, args.runs_root, profile="qwen_moe"
        )
        run = infra.prepare(
            ROOT,
            config,
            runs_root,
            versions,
            args.network,
            baseline_mode="local",
            profile="qwen_moe",
        )
        manifest = infra.load_bundle(run)
    if args.prepare_only:
        initialize(run)
        return 0
    if not args.resume:
        frozen_command = [
            sys.executable,
            str(run / "source" / REL / "control.py"),
            "--resume",
            str(run),
        ]
        if args.build:
            # Baseline and all search execution use the frozen controller.
            subprocess.run([*frozen_command, "--baseline-only"], check=True)
            if infra.build_identity(ROOT, versions) != manifest["build_input_sha256"]:
                raise ValueError(
                    "image inputs changed after freezing; start a new experiment"
                )
            infra.build_image(
                ROOT,
                manifest["experiment"]["agent"]["image"],
                versions,
                args.buildkit_image,
                args.build_proxy,
                args.build_fallback_proxy,
            )
        if args.baseline_only:
            frozen_command.append("--baseline-only")
        if args.env_file:
            frozen_command += ["--env-file", str(args.env_file.resolve())]
        return subprocess.call(frozen_command)
    with (run / "controller.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError(
                "this experiment already has an active controller"
            ) from None
        initialize(run)
        if not (run / "task/baseline.json").exists():
            infra.local_baseline(run, manifest)
        baseline = baseline_store.validate(run, manifest)
        if args.baseline_only:
            print(f"Start Qwen search: ./run_qwen_moe_agent.sh --resume {run}")
            return 0
        image = infra.image_identity(
            manifest["experiment"]["agent"]["image"], manifest["build_input_sha256"]
        )
        return supervise(run, manifest, image, args.env_file, baseline)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2) from None
