"""Compare native executables sequentially with identical simulation inputs."""

import argparse
import hashlib
import json
import math
import os
import signal
import statistics
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from results import load_run, load_workload, validate_complete


class BenchmarkInterrupted(Exception):
    pass


def interrupt(signum, _frame):
    raise BenchmarkInterrupted(f"interrupted by {signal.Signals(signum).name}")


def process_group_exists(pid):
    try:
        os.killpg(pid, 0)
        return True
    except ProcessLookupError:
        return False


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def process_tree_rss(pid):
    """Sample Linux resident pages, summed across the current child tree.

    Shared pages may be counted more than once. Short-lived allocations between
    samples are not captured; this is not the OS maximum resident set size.
    """
    page_size = os.sysconf("SC_PAGE_SIZE")
    total, pending, seen = 0, [pid], set()
    while pending:
        current = pending.pop()
        if current in seen:
            continue
        seen.add(current)
        try:
            total += (
                int(Path(f"/proc/{current}/statm").read_text().split()[1]) * page_size
            )
            for task in Path(f"/proc/{current}/task").iterdir():
                try:
                    pending.extend(map(int, (task / "children").read_text().split()))
                except FileNotFoundError, ProcessLookupError:
                    continue
        except FileNotFoundError, ProcessLookupError:
            # Processes may finish during a sample.
            continue
    return total


def stop_group(process):
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()
    except ProcessLookupError:
        process.wait()
    finally:
        # A parent can exit on SIGTERM while a worker ignores it.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def measure(binary, config, workload, directory, cwd, timeout, interval):
    directory.mkdir(parents=True, exist_ok=False)
    command = [
        str(binary),
        "--config",
        str(config),
        "--workload",
        str(workload),
        "--output-jsonl",
        str(directory / "result.jsonl"),
        "--output-policy-stats",
        str(directory / "policy-stats.json"),
    ]
    record = {
        "command": command,
        "cwd": str(cwd),
        "timeout_s": timeout,
        "rss_sample_interval_s": interval,
        "status": "started",
    }
    save(directory / "measurement.json", record)
    started = time.monotonic()
    peak, count, finished = 0, 0, None
    with (
        (directory / "run.log").open("w") as log,
        (directory / "rss.jsonl").open("w") as samples,
    ):
        process = None
        try:
            process = subprocess.Popen(
                command,
                cwd=cwd,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            while process.poll() is None:
                elapsed = time.monotonic() - started
                rss = process_tree_rss(process.pid)
                peak, count = max(peak, rss), count + 1
                samples.write(
                    json.dumps({"elapsed_s": elapsed, "rss_bytes": rss}) + "\n"
                )
                if elapsed >= timeout:
                    raise TimeoutError(f"native run exceeded {timeout}s")
                try:
                    process.wait(timeout=min(interval, timeout - elapsed))
                except subprocess.TimeoutExpired:
                    pass
            finished = time.monotonic()
            if process.returncode:
                raise RuntimeError(
                    f"native process exit {process.returncode}; see {directory / 'run.log'}"
                )
            if process_group_exists(process.pid):
                raise RuntimeError("native parent exited with surviving descendants")
            result = directory / "result.jsonl"
            if not result.is_file():
                raise RuntimeError("native process did not write result.jsonl")
            record.update(
                status="complete",
                result_sha256=digest(result),
                result_bytes=result.stat().st_size,
                completion=validate_complete(
                    load_run(str(result)), load_workload(workload)
                ),
            )
        except BaseException as error:
            if process is not None:
                stop_group(process)
            record.update(
                status="incomplete"
                if isinstance(error, (KeyboardInterrupt, BenchmarkInterrupted))
                else "failed",
                error=str(error),
            )
            raise
        finally:
            record.update(
                exit_code=process.poll() if process is not None else None,
                wall_s=(finished if finished is not None else time.monotonic())
                - started,
                sampled_process_tree_peak_rss_bytes=peak,
                rss_samples=count,
                loadavg_at_finish=list(os.getloadavg()),
            )
            save(directory / "measurement.json", record)
    return record


def summarize(records, names, max_wall_regression, max_rss_regression):
    summary, failures = {}, []
    for name in names:
        selected = [r for r in records if r["case"] == name and not r["warmup"]]
        pairs = {}
        for record in selected:
            pairs.setdefault(record["pair"], {})[record["variant"]] = record
        wall_ratios, rss_ratios = [], []
        for pair in pairs.values():
            wall_ratios.append(pair["candidate"]["wall_s"] / pair["baseline"]["wall_s"])
            old_rss = pair["baseline"]["sampled_process_tree_peak_rss_bytes"]
            new_rss = pair["candidate"]["sampled_process_tree_peak_rss_bytes"]
            if old_rss > 0 and new_rss > 0:
                rss_ratios.append(new_rss / old_rss)
        item = {
            "pairs": len(pairs),
            "paired_wall_ratios": wall_ratios,
            "median_paired_wall_ratio": statistics.median(wall_ratios),
            "paired_sampled_rss_ratios": rss_ratios,
            "median_paired_sampled_rss_ratio": statistics.median(rss_ratios)
            if rss_ratios
            else None,
        }
        for variant in ("baseline", "candidate"):
            rows = [r for r in selected if r["variant"] == variant]
            item[variant] = {
                "median_wall_s": statistics.median(r["wall_s"] for r in rows),
                "median_sampled_process_tree_peak_rss_bytes": statistics.median(
                    r["sampled_process_tree_peak_rss_bytes"] for r in rows
                ),
            }
        if item["median_paired_wall_ratio"] > 1 + max_wall_regression:
            failures.append(
                f"{name}: median paired runtime exceeds {max_wall_regression:.1%} regression"
            )
        if max_rss_regression is not None and (
            len(rss_ratios) != len(pairs)
            or item["median_paired_sampled_rss_ratio"] > 1 + max_rss_regression
        ):
            failures.append(
                f"{name}: sampled process-tree RSS missing or exceeds {max_rss_regression:.1%} regression"
            )
        summary[name] = item
    return summary, failures


def benchmark(args):
    if not Path("/proc/self/statm").exists():
        raise ValueError("process-tree RSS sampling requires Linux /proc")
    for name in ("pairs", "warmups"):
        if getattr(args, name) < (1 if name == "pairs" else 0):
            raise ValueError(f"invalid {name}")
    for name in ("timeout", "sample_interval"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            raise ValueError(f"invalid {name}")
    for name in ("max_wall_regression", "max_rss_regression"):
        value = getattr(args, name)
        if value is not None and (not math.isfinite(value) or value < 0):
            raise ValueError(f"invalid {name}")
    names = [case[0] for case in args.case]
    if len(set(names)) != len(names) or any(
        not n
        or any(
            c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
            for c in n
        )
        for n in names
    ):
        raise ValueError("case names must be unique path-safe identifiers")
    binaries = {
        key: Path(getattr(args, key)).resolve() for key in ("baseline", "candidate")
    }
    cases = [
        (name, Path(config).resolve(), Path(workload).resolve())
        for name, config, workload in args.case
    ]
    inputs = set(binaries.values()) | {
        p for _, config, workload in cases for p in (config, workload)
    }
    inputs.update(Path(p).resolve() for p in args.identity)
    identities = {str(path): digest(path) for path in sorted(inputs)}
    directory, cwd = Path(args.output).resolve(), Path(args.cwd).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "running",
        "identities": identities,
        "arguments": vars(args),
        "records": [],
        "measurement": "wall time includes subprocess startup/output; RSS is sampled sum over process tree, not OS peak",
    }
    save(directory / "report.json", report)
    try:
        for warmup, rounds in ((True, args.warmups), (False, args.pairs)):
            for pair in range(rounds):
                order = (
                    ("baseline", "candidate")
                    if pair % 2 == 0
                    else ("candidate", "baseline")
                )
                for name, config, workload in cases:
                    paired = {}
                    for variant in order:
                        for path in (binaries[variant], config, workload):
                            if digest(path) != identities[str(path)]:
                                raise RuntimeError(f"benchmark input changed: {path}")
                        path = (
                            directory
                            / f"{'warmup' if warmup else 'pair'}-{pair:02d}-{name}-{variant}"
                        )
                        record = measure(
                            binaries[variant],
                            config,
                            workload,
                            path,
                            cwd,
                            args.timeout,
                            args.sample_interval,
                        )
                        record.update(
                            case=name,
                            pair=pair,
                            warmup=warmup,
                            variant=variant,
                            directory=str(path),
                        )
                        report["records"].append(record)
                        paired[variant] = record
                        save(directory / "report.json", report)
                    if (
                        paired["baseline"]["result_sha256"]
                        != paired["candidate"]["result_sha256"]
                    ):
                        raise RuntimeError(
                            f"byte-exact result mismatch: {name}, {'warmup' if warmup else 'pair'} {pair}"
                        )
        for path in inputs:
            if digest(path) != identities[str(path)]:
                raise RuntimeError(f"benchmark input changed: {path}")
        report["summary"], report["failures"] = summarize(
            report["records"], names, args.max_wall_regression, args.max_rss_regression
        )
        report["status"] = "failed" if report["failures"] else "passed"
    except (KeyboardInterrupt, BenchmarkInterrupted) as error:
        report.update(status="incomplete", failures=[str(error) or "interrupted"])
    except (
        OSError,
        ValueError,
        RuntimeError,
        TimeoutError,
        KeyError,
        TypeError,
    ) as error:
        report.update(status="failed", failures=[str(error)])
    finally:
        save(directory / "report.json", report)
    return report


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--baseline", required=True)
    result.add_argument("--candidate", required=True)
    result.add_argument(
        "--case",
        nargs=3,
        action="append",
        metavar=("NAME", "CONFIG", "WORKLOAD"),
        required=True,
    )
    result.add_argument(
        "--output", required=True, help="new directory for all samples and logs"
    )
    result.add_argument("--cwd", default=str(Path.cwd()))
    result.add_argument(
        "--identity",
        action="append",
        default=[],
        help="additional unchanged dependency to hash",
    )
    result.add_argument("--pairs", type=int, default=3)
    result.add_argument("--warmups", type=int, default=1)
    result.add_argument("--timeout", type=float, default=1800)
    result.add_argument("--sample-interval", type=float, default=0.02)
    result.add_argument("--max-wall-regression", type=float, default=0.05)
    result.add_argument(
        "--max-rss-regression",
        type=float,
        help="optional sampled RSS regression fraction",
    )
    return result


def main():
    previous = signal.signal(signal.SIGTERM, interrupt)
    try:
        report = benchmark(parser().parse_args())
    finally:
        signal.signal(signal.SIGTERM, previous)
    print(
        json.dumps(
            {
                key: report[key]
                for key in ("status", "summary", "failures")
                if key in report
            },
            indent=2,
        )
    )
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
