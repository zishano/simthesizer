"""Frozen-case policy evaluation. Artifacts remain provisional until external replay."""

import argparse
import copy
import fcntl
import hashlib
import importlib.metadata
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import tomllib
from pathlib import Path

sys.dont_write_bytecode = True
import launcher

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from results import (
    load_run,
    load_workload,
    physical_gpu_count,
    slo_metrics,
    summarize_run,
    validate_complete,
)
from rooflang_bridge.configuration import toml_document

VERSION = "serving-evaluator-v1"
API_VERSION = "serving-policy-v1"
POLICY = launcher.POLICY_REL
LIMITS = (
    "p_token_budget_bounds",
    "d_token_budget_bounds",
    "p_batch_bounds",
    "d_batch_bounds",
    "kv_inflight_bounds",
    "max_wait_s",
    "state_limit_bytes",
)


def identity(value):
    return hashlib.sha256(launcher.json_bytes(value)).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def runtime_tool_paths():
    """Identify the Python code actually loaded, not merely a snapshot on disk."""
    paths = {
        str(launcher.AGENT_REL / "evaluate.py"): Path(__file__),
        str(launcher.AGENT_REL / "launcher.py"): Path(launcher.__file__),
        "scripts/results.py": Path(sys.modules["results"].__file__),
        "scripts/rooflang_bridge/configuration.py": Path(
            sys.modules["rooflang_bridge.configuration"].__file__
        ),
    }
    for name in ("verify", "archive", "__main__"):
        module = sys.modules.get(name)
        path = Path(getattr(module, "__file__", ""))
        if path.name in ("verify.py", "archive.py"):
            paths[str(launcher.AGENT_REL / path.name)] = path
    return paths


def append(path, value):
    with path.open("a") as stream:
        stream.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def ledger(path):
    if not path.exists():
        return []
    return [json.loads(row) for row in path.read_text().splitlines() if row.strip()]


def clean_environment(cache):
    allowed = {
        "PATH",
        "HOME",
        "RUSTUP_HOME",
        "RUSTUP_TOOLCHAIN",
        "CARGO_HOME",
        "LANG",
        "LC_ALL",
        "TMPDIR",
        "SSL_CERT_FILE",
        "SSL_CERT_DIR",
    }
    env = {key: value for key, value in os.environ.items() if key in allowed}
    env.update(
        {
            "CARGO_TARGET_DIR": str(cache / "target"),
            "CARGO_NET_OFFLINE": "true",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONUNBUFFERED": "1",
            "MPLBACKEND": "Agg",
            "MPLCONFIGDIR": str(cache / "matplotlib"),
            "UV_OFFLINE": "1",
            "UV_NO_SYNC": "1",
        }
    )
    return env


def run_process(command, *, cwd, env, log, deadline):
    """One process group per command; workers die with the timed-out simulator."""
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("trial wall-clock budget exhausted")
    handlers = {}

    def interrupted(signum, _frame):
        raise InterruptedError(f"interrupted by signal {signum}")

    for signum in (signal.SIGTERM, signal.SIGINT):
        handlers[signum] = signal.signal(signum, interrupted)
    with Path(log).open("ab") as stream:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            status = process.wait(timeout=remaining)
            if status:
                raise RuntimeError(f"command exited {status}; see {log}")
        except BaseException:
            try:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=2)
            except ProcessLookupError, subprocess.TimeoutExpired:
                pass
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
            raise
        finally:
            # A worker must not outlive an otherwise successful parent either.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            for signum, handler in handlers.items():
                signal.signal(signum, handler)


class Experiment:
    def __init__(self, manifest):
        self.manifest_path = Path(manifest).resolve()
        self.run = self.manifest_path.parent.parent
        self.manifest = read_json(self.manifest_path)
        self.config = self.manifest["experiment"]
        self.source = self.run / (
            "simthesizer" if self.run == Path("/workspace") else "source"
        )
        self.inputs = self.run / "inputs"
        self.artifacts = self.run / "artifacts"
        self.cache = self.run / "cache"
        self.baseline = self.run / "baseline-policy"
        self.env = clean_environment(self.cache)
        self.python_identity = {
            "python": sys.version,
            "packages": {
                name: importlib.metadata.version(name)
                for name in ("networkx", "matplotlib")
            },
        }
        self.book = self.artifacts / "experiments.jsonl"
        self.verify_inputs()
        for relative, actual in runtime_tool_paths().items():
            if launcher.digest(actual) != self.manifest["source_sha256"].get(relative):
                raise ValueError(
                    f"executing tool differs from frozen source: {relative}; use {self.source / launcher.AGENT_REL / 'evaluate.py'}"
                )
        self.scenario = tomllib.loads(
            self.input(self.config["inputs"]["scenario"]).read_text()
        )
        self.validate_scenario()

    def input(self, value):
        path = Path(value)
        try:
            relative = path.relative_to("/workspace/inputs")
        except ValueError:
            raise ValueError(f"input is not a frozen input path: {value}") from None
        result = self.inputs / relative
        if (
            result.is_symlink()
            or not result.is_file()
            or not result.resolve().is_relative_to(self.inputs.resolve())
        ):
            raise ValueError(f"invalid frozen input: {value}")
        return result

    def verify_inputs(self):
        issues = launcher.config_issues(self.config)
        if issues:
            raise ValueError("invalid frozen experiment: " + "; ".join(issues))
        if self.config["policy"]["api_version"] != API_VERSION:
            raise ValueError(f"expected policy.api_version={API_VERSION}")
        for path, expected, mutable in (
            (self.source, "source_sha256", True),
            (self.inputs, "inputs_sha256", False),
            (self.baseline, "baseline_policy_sha256", False),
        ):
            if launcher.inventory(path, mutable=mutable) != self.manifest[expected]:
                raise ValueError(f"frozen {expected} changed")
        if (self.run / "task/AGENTS.md").read_text() != launcher.render_task(
            self.source, self.manifest
        ):
            raise ValueError("frozen agent task changed")

    def validate_scenario(self):
        system = self.scenario.get("system", {})
        if (
            system.get("kind") != "pd_disaggregation"
            or system.get("shared_execution") is not True
        ):
            raise ValueError("policy search requires shared P/D execution")
        if "policy" in system:
            raise ValueError("scenario policy is owned by the evaluator")
        if physical_gpu_count(self.scenario) is None:
            raise ValueError("physical GPU allocation is unknown")
        for pool in ("prefill_instances", "decode_instances"):
            children = system.get(pool)
            if not children:
                raise ValueError(f"missing {pool}")
            for child in children:
                runner = child.get("model_runner", {})
                if (
                    runner.get("kind") != "rooflang"
                    or runner.get("shared_execution") is not True
                ):
                    raise ValueError("all instances must use shared Rooflang")
                command = runner.get("command", [])
                # Only the built-in worker is part of this evaluator contract.
                workers = [
                    str(arg)
                    for arg in command
                    if str(arg).endswith("rooflang_worker.py")
                ]
                if len(workers) != 1 or any(
                    str(arg).startswith("--") and arg != "--locked" for arg in command
                ):
                    raise ValueError(
                        "scenario must use the built-in worker without audit/reference flags"
                    )
        for workload in self.config["workloads"]:
            rows = load_workload(self.input(workload["path"]))
            if any(
                len(r["sub_requests"]) != 1 or r["sub_requests"][0]["kind"] != "llm"
                for r in rows
            ):
                raise ValueError("policy search currently supports one LLM per request")

    def cases(self):
        result = []
        for workload in self.config["workloads"]:
            for multiplier in self.config["loads"]["arrival_rate_multipliers"]:
                case = {
                    "workload_id": workload["id"],
                    "workload_path": workload["path"],
                    "split": workload["split"],
                    "arrival_multiplier": multiplier,
                    "seed": self.config["search"]["seed"],
                }
                case["id"] = f"{workload['id']}-m{float(multiplier)!r}-s{case['seed']}"
                result.append(case)
        if len({case["id"] for case in result}) != len(result):
            raise ValueError("duplicate case identities")
        return result

    def freeze_candidate(self, baseline=False):
        source = self.baseline if baseline else self.source / POLICY
        files = launcher.inventory(source)
        if not files or any(not name.endswith(".rs") for name in files):
            raise ValueError("candidate must contain only Rust source files")
        for name in files:
            text = (source / name).read_text()
            if re.search(
                r"\bunsafe\b|\bextern\b|\b(?:std|core)::(?:fs|net|process|env|time|thread|ffi)\b|\binclude(?:_str|_bytes)?\s*!|\blibc\b",
                text,
            ):
                raise ValueError(
                    f"candidate violates the pure-policy source contract: {name}"
                )
        candidate = {"kind": "reference" if baseline else "candidate", "files": files}
        candidate["id"] = identity(candidate)
        destination = self.artifacts / "candidates" / candidate["id"]
        if not destination.exists():
            destination.mkdir(parents=True)
            shutil.copytree(source, destination / "policy")
            launcher.save(destination / "candidate.json", candidate)
        elif (
            read_json(destination / "candidate.json") != candidate
            or launcher.inventory(destination / "policy") != files
        ):
            raise ValueError("candidate snapshot changed")
        return candidate

    def build(self, candidate, deadline, *, clean=False):
        tools = {}
        for tool in ("rustc", "cargo"):
            tools[tool] = subprocess.check_output(
                [tool, "--version"], env=self.env, text=True, timeout=10
            ).strip()
        if tools["rustc"].split()[1] != self.manifest["build_versions"]["RUST_VERSION"]:
            raise ValueError(
                "installed Rust toolchain differs from the frozen build version"
            )
        tools.update(self.python_identity)
        build_key = identity(
            {
                "base": self.manifest["source_sha256"],
                "candidate": candidate,
                "tools": tools,
                "version": VERSION,
            }
        )
        target = self.cache / "builds" / build_key
        receipt = target / "build.json"
        executable = target / "simthesizer"
        if not clean and receipt.exists():
            record = read_json(receipt)
            if record["binary_sha256"] != launcher.digest(executable):
                raise ValueError("cached native binary changed")
            return executable, record
        target.mkdir(parents=True, exist_ok=True)
        source = target / "source"
        if source.exists():
            shutil.rmtree(source)
        shutil.copytree(self.source, source)
        if (source / POLICY).exists():
            shutil.rmtree(source / POLICY)
        shutil.copytree(
            self.artifacts / "candidates" / candidate["id"] / "policy", source / POLICY
        )
        if launcher.inventory(source, mutable=True) != self.manifest["source_sha256"]:
            raise ValueError("build source differs from frozen source")
        env = dict(self.env)
        if clean:
            env["CARGO_TARGET_DIR"] = str(target / "clean-target")
        run_process(
            [
                "cargo",
                "build",
                "--release",
                "--locked",
                "--offline",
                "--bin",
                "simthesizer",
            ],
            cwd=source,
            env=env,
            log=target / "build.log",
            deadline=deadline,
        )
        shutil.copy2(Path(env["CARGO_TARGET_DIR"]) / "release/simthesizer", executable)
        record = {
            "build_id": build_key,
            "binary_sha256": launcher.digest(executable),
            "tools": tools,
            "source_sha256": self.manifest["source_sha256"],
            "candidate_id": candidate["id"],
        }
        launcher.save(receipt, record)
        return executable, record

    def capabilities(self, executable):
        data = json.loads(
            subprocess.check_output(
                [str(executable), "--policy-capabilities"],
                cwd=self.source,
                env=self.env,
                text=True,
                timeout=10,
            )
        )
        if data.get("policy_api_version") != API_VERSION or not set(
            self.config["policy"]["actions"]
        ).issubset(data.get("actions", [])):
            raise ValueError("native binary does not implement the frozen policy API")
        for key in ("validates_actions", "checks_full_completion"):
            if data.get(key) is not True:
                raise ValueError(f"native capability missing: {key}")
        for key in ("length_model", "scheduler_cost"):
            if data.get(key) != self.config["model_assumptions"][key]:
                raise ValueError(f"native model assumption differs: {key}")
        return data

    def derived(self, case):
        rows = load_workload(self.input(case["workload_path"]))
        origin = min(row["arrival_time"] for row in rows)
        for row in rows:
            row["arrival_time"] = (
                origin + (row["arrival_time"] - origin) / case["arrival_multiplier"]
            )
        return rows

    def configuration(self, candidate):
        config = copy.deepcopy(self.scenario)
        policy = self.config["policy"]
        config["system"]["policy"] = {
            "kind": candidate["kind"],
            "seed": self.config["search"]["seed"],
            "limits": {key: policy[key] for key in LIMITS},
            "slo": {
                key: self.config["slo"][key] for key in ("ttft_limit_s", "tpot_limit_s")
            },
        }
        for pool in ("prefill_instances", "decode_instances"):
            for child in config["system"][pool]:
                child["model_runner"]["command"] = [
                    sys.executable,
                    str(self.source / "scripts/rooflang_worker.py"),
                ]
        return config

    def trial_identity(self, candidate, case, rows):
        group = {
            "version": VERSION,
            "base": self.manifest["source_sha256"],
            "inputs": self.manifest["inputs_sha256"],
            "build_versions": self.manifest["build_versions"],
            "python_environment": self.python_identity,
            "case": case,
            "derived_workload_sha256": identity(rows),
            "slo": self.config["slo"],
            "model_assumptions": self.config["model_assumptions"],
            "policy": self.config["policy"],
        }
        return identity({"group": group, "candidate": candidate}), identity(group)

    def reserve(self, trial_id, group_id, candidate, case, directory):
        events = ledger(self.book)
        starts = {row["attempt"]: row for row in events if row["event"] == "started"}
        ends = {row["attempt"]: row for row in events if row["event"] == "finished"}
        elapsed = sum(
            ends[key]["wall_seconds"]
            if key in ends
            else max(0, time.time() - value["started_at"])
            for key, value in starts.items()
        )
        budget = self.config["search"]
        if len(starts) >= budget["max_trials"] or elapsed >= budget["max_wall_seconds"]:
            raise ValueError("search evaluation budget exhausted")
        attempt = len(starts) + 1
        record = {
            "event": "started",
            "attempt": attempt,
            "trial_id": trial_id,
            "group_id": group_id,
            "candidate_id": candidate["id"],
            "kind": candidate["kind"],
            "case": case,
            "started_at": time.time(),
            "directory": str(directory.relative_to(self.artifacts)),
        }
        append(self.book, record)
        return record, time.monotonic() + min(
            budget["trial_wall_seconds"], budget["max_wall_seconds"] - elapsed
        )

    def evaluate(self, candidate, case, *, force=False, clean=False):
        rows = self.derived(case)
        trial_id, group_id = self.trial_identity(candidate, case, rows)
        events = ledger(self.book)
        if not force:
            for old in reversed(events):
                if (
                    old.get("event") == "finished"
                    and old["trial_id"] == trial_id
                    and old["status"].startswith("VALID_")
                ):
                    path = self.artifacts / old["directory"]
                    if any(
                        launcher.digest(path / key) != value
                        for key, value in old["evidence_sha256"].items()
                    ):
                        raise ValueError("completed trial evidence changed")
                    return {**old, "reused": True}
        directory = (
            self.artifacts
            / "trials"
            / candidate["id"]
            / f"{case['id']}-{len(events) + 1}"
        )
        record, deadline = self.reserve(trial_id, group_id, candidate, case, directory)
        started = time.monotonic()
        result = {
            **record,
            "event": "finished",
            "status": "INVALID",
            "verification": "provisional",
        }
        try:
            directory.mkdir(parents=True)
            executable, build = self.build(candidate, deadline, clean=clean)
            self.capabilities(executable)
            launcher.save(directory / "build.json", build)
            config = self.configuration(candidate)
            (directory / "config.toml").write_text(toml_document(config))
            workload = directory / "workload.jsonl"
            workload.write_text(
                "".join(
                    json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n"
                    for row in rows
                )
            )
            command = [
                str(executable),
                "--config",
                str(directory / "config.toml"),
                "--workload",
                str(workload),
                "--output-jsonl",
                str(directory / "result.jsonl"),
                "--output-policy-stats",
                str(directory / "policy-stats.json"),
            ]
            run_process(
                command,
                cwd=self.source,
                env=self.env,
                log=directory / "simulation.log",
                deadline=deadline,
            )
            run = load_run(str(directory / "result.jsonl"))
            complete = validate_complete(run, rows)
            metrics = summarize_run(run)
            metrics["completion"] = complete
            metrics["slo"] = slo_metrics(run, self.config["slo"])
            metrics["policy"] = read_json(directory / "policy-stats.json")
            metrics["service_window"] = {
                "origin_s": run.origin,
                "finish_s": run.service_finish_s,
                "source": "native_simulation_end"
                if run.service_finish_s is not None
                else "result_timeline",
            }
            launcher.save(directory / "metrics.json", metrics)
            if not metrics["slo"]["available"]:
                raise ValueError("goodput unavailable for this complete run")
            for script, image in (
                ("draw.py", "metrics.png"),
                ("draw_throughput.py", "throughput.png"),
            ):
                run_process(
                    [
                        sys.executable,
                        str(self.source / "scripts" / script),
                        str(directory / "result.jsonl"),
                        "--output",
                        str(directory / image),
                    ],
                    cwd=self.source,
                    env=self.env,
                    log=directory / "plots.log",
                    deadline=deadline,
                )
            self.verify_inputs()
            if (
                launcher.inventory(
                    self.artifacts / "candidates" / candidate["id"] / "policy"
                )
                != candidate["files"]
            ):
                raise ValueError("candidate snapshot changed while running")
            result.update(
                {
                    "status": "VALID_FEASIBLE"
                    if metrics["slo"]["feasible"]
                    else "VALID_INFEASIBLE",
                    "objectives": {
                        "ttft_p99": metrics["ttft"]["p99"],
                        "tpot_p99": metrics["tpot"]["p99"]
                        if metrics["slo"]["tpot_applicable"]
                        else None,
                        "goodput_per_gpu": metrics["slo"]["goodput_per_gpu"],
                    },
                    "tpot_applicable": metrics["slo"]["tpot_applicable"],
                    "evidence_sha256": {
                        name: launcher.digest(directory / name)
                        for name in (
                            "result.jsonl",
                            "metrics.json",
                            "config.toml",
                            "workload.jsonl",
                            "build.json",
                            "policy-stats.json",
                            "metrics.png",
                            "throughput.png",
                        )
                    },
                }
            )
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    "trial wall-clock budget exhausted during validation"
                )
        except (
            ValueError,
            OSError,
            RuntimeError,
            KeyError,
            TypeError,
            subprocess.SubprocessError,
            KeyboardInterrupt,
        ) as error:
            result["status"] = (
                "INCOMPLETE"
                if isinstance(
                    error,
                    (
                        TimeoutError,
                        InterruptedError,
                        subprocess.TimeoutExpired,
                        KeyboardInterrupt,
                    ),
                )
                else "INVALID"
            )
            result["reason"] = str(error) or type(error).__name__
        finally:
            result["wall_seconds"] = time.monotonic() - started
            append(self.book, result)
            launcher.save(directory / "receipt.json", result)
        return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--preflight", action="store_true")
    modes.add_argument("--list-cases", action="store_true")
    modes.add_argument("--baseline", action="store_true")
    modes.add_argument("--candidate", action="store_true")
    parser.add_argument("--case", action="append", default=[])
    args = parser.parse_args(argv)
    experiment = Experiment(args.manifest)
    experiment.artifacts.mkdir(exist_ok=True)
    with (experiment.artifacts / ".evaluate.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.list_cases:
            print(json.dumps(experiment.cases(), indent=2))
            return 0
        candidate = experiment.freeze_candidate(
            baseline=args.baseline or args.preflight
        )
        if args.preflight:
            executable, build = experiment.build(
                candidate,
                time.monotonic() + experiment.config["search"]["trial_wall_seconds"],
            )
            capabilities = experiment.capabilities(executable)
            print(
                json.dumps(
                    {
                        "policy_api_version": API_VERSION,
                        "ready": True,
                        "enforces_trial_budget": True,
                        "enforces_trial_timeout": True,
                        "native": capabilities,
                        "build_id": build["build_id"],
                    },
                    sort_keys=True,
                )
            )
            return 0
        cases = experiment.cases()
        if set(args.case) - {row["id"] for row in cases}:
            parser.error("unknown case; use --list-cases")
        selected = [row for row in cases if not args.case or row["id"] in args.case]
        results = [experiment.evaluate(candidate, case) for case in selected]
        from archive import update_archive

        update_archive(experiment.artifacts)
        print(json.dumps(results, indent=2))
        return 0 if all(row["status"].startswith("VALID_") for row in results) else 2


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, OSError, subprocess.SubprocessError, KeyError) as error:
        print(f"error: {error}", file=sys.stderr)
        sys.exit(2)
