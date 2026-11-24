"""Docker launcher and frozen run bundles; does not implement policy search."""

import argparse
import csv
import fcntl
import hashlib
import importlib.util
import io
import json
import math
import os
import re
import selectors
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import tomllib
import uuid
from collections import deque
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import urlsplit

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
AGENT_REL = Path("agents/find_pareto_frontier")
POLICY_REL = Path("crates/simthesizer-core/src/policies/candidate")
EVALUATOR_REL = AGENT_REL / "evaluate.py"
PROFILES = {
    "legacy": AGENT_REL,
    "qwen_moe": Path("agents/qwen_moe_optimization"),
}
IMAGE_DEFAULT = "simthesizer-pareto-agent:codex-0.166.0"
BUILDKIT_IMAGE = "moby/buildkit:buildx-stable-1"
BUILD_DEFAULTS = {
    "PYTHON_IMAGE": "python:3.14.5-slim-bookworm",
    "UV_IMAGE": "ghcr.io/astral-sh/uv:0.11.19",
    "NODE_IMAGE": "node:26.7.0-bookworm-slim",
    "RUST_VERSION": "1.96.0",
    "CODEX_VERSION": "0.166.0",
}
BUILD_FILES = {
    "Dockerfile": AGENT_REL / "Dockerfile",
    "entrypoint.sh": AGENT_REL / "entrypoint.sh",
    "launcher.py": AGENT_REL / "launcher.py",
    "baseline_store.py": AGENT_REL / "baseline_store.py",
    **{
        p: Path(p)
        for p in (
            "Cargo.toml",
            "Cargo.lock",
            "pyproject.toml",
            "uv.lock",
            "crates/simthesizer-core/Cargo.toml",
            "crates/simthesizer-cli/Cargo.toml",
        )
    },
}
BUILD_LABEL = "io.simthesizer.agent.build-input-sha256"
ENVIRONMENT_LABEL = "io.simthesizer.agent.environment-input-sha256"
CONTROLLER_FILES = ("entrypoint.sh", "launcher.py", "baseline_store.py")
PROXIES = {
    name
    for base in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY")
    for name in (base, base.lower())
}
ENV_KEYS = PROXIES | {"CODEX_API_KEY", "OPENAI_API_KEY", "OPENAI_BASE_URL"}
EXCLUDED = {
    ".git",
    ".agents",
    ".codex",
    ".claude",
    ".aws",
    ".venv",
    ".devenv",
    "__pycache__",
    ".ruff_cache",
    ".pytest_cache",
    "node_modules",
    "target",
    "output",
    "runs",
    "AGENTS.md",
    "AGENTS.override.md",
    "CLAUDE.md",
    ".mcp.json",
    "experiment.toml",
    "agent.env",
}


def fail(message):
    raise ValueError(message)


def digest(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def json_bytes(value):
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode()


def save(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(json_bytes(value))
    temporary.replace(path)


def command(argv, **kwargs):
    return subprocess.run(argv, check=True, **kwargs)


def output(argv):
    return command(argv, stdout=subprocess.PIPE, text=True).stdout.strip()


def safe_path(path):
    path = Path(path).expanduser().resolve()
    if any(c in str(path) for c in (",", "\n", "\r")):
        fail("Docker mount paths cannot contain commas or newlines")
    return path


def input_file(root, value):
    path = root / value
    if path.is_symlink() or not path.is_file():
        fail(f"input must be a regular file: {path}")
    return path.resolve()


def profile_name(config=None, *, profile=None):
    """Select a known task; never interpret a profile as a filesystem path."""
    research = (config or {}).get("research", {})
    if not isinstance(research, dict):
        fail("research must be a table")
    configured = research.get("profile")
    for value in (configured, profile):
        if value is not None and (not isinstance(value, str) or value not in PROFILES):
            fail(f"unsupported research profile: {value!r}")
    if configured is not None and profile is not None and configured != profile:
        fail("research.profile differs from the selected profile")
    return profile or configured or "legacy"


def manifest_profile(manifest):
    # Legacy manifests lack this field; their rendering and hashes stay unchanged.
    return profile_name(
        manifest.get("experiment", {}), profile=manifest.get("profile", "legacy")
    )


def profile_directory(profile="legacy"):
    return PROFILES[profile_name(profile=profile)]


def resolve_runs_root(root, config, runs_root=None, *, profile=None):
    root = safe_path(root)
    selected = profile_name(config, profile=profile)
    default = profile_directory(selected) / "runs"
    result = safe_path(
        runs_root or root / config.get("artifacts", {}).get("runs_root", str(default))
    )
    if result == root or root.is_relative_to(result):
        fail("runs-root must not be the project or an ancestor")
    if result.is_relative_to(root) and result != root / default:
        fail(f"inside the project, use {default}")
    return result


def profile_runtime(root, profile):
    """Load the selected frozen task adapter without changing import search paths."""
    relative = profile_directory(profile) / "runtime.py"
    path = input_file(root, relative)
    spec = importlib.util.spec_from_file_location("simthesizer_task_runtime", path)
    if spec is None or spec.loader is None:
        fail(f"cannot load task runtime: {relative}")
    module = importlib.util.module_from_spec(spec)
    # Do not create __pycache__ inside the immutable source snapshot.
    exec(compile(path.read_bytes(), str(path), "exec"), module.__dict__)  # noqa: S102 -- fixed, frozen task adapter
    return module


def config_issues(config):
    issues = []
    for group in (
        "inputs",
        "model_assumptions",
        "slo",
        "loads",
        "policy",
        "agent",
        "search",
        "verification",
        "container",
        "artifacts",
    ):
        if not isinstance(config.get(group), dict):
            issues.append(f"{group}: table required")
    if issues:
        return issues
    if not isinstance(config.get("workloads"), list) or any(
        not isinstance(row, dict) for row in config["workloads"]
    ):
        return ["workloads: array of tables required"]

    def walk(value, prefix=""):
        if isinstance(value, dict):
            for key, child in value.items():
                walk(child, f"{prefix}.{key}".strip("."))
        elif isinstance(value, list):
            for index, child in enumerate(value):
                walk(child, f"{prefix}[{index}]")
        elif isinstance(value, str) and "REQUIRED" in value:
            issues.append(f"{prefix}: not configured")

    walk(config)
    if (
        type(config.get("schema_version")) is not int
        or config["schema_version"] != 1
        or config.get("design_only") is not False
    ):
        issues.append("schema_version must be 1 and design_only must be false")
    for group, fields in {
        "slo": ("ttft_limit_s", "tpot_limit_s", "min_joint_pass_fraction"),
        "search": ("max_trials", "max_wall_seconds", "trial_wall_seconds"),
        "verification": ("max_trials", "max_wall_seconds"),
        "container": ("cpus", "memory_mib", "pids_limit", "parallel_trials"),
        "policy": ("max_wait_s", "state_limit_bytes"),
    }.items():
        for field in fields:
            value = config.get(group, {}).get(field)
            if (
                type(value) not in (int, float)
                or not math.isfinite(value)
                or value <= 0
            ):
                issues.append(f"{group}.{field}: positive finite number required")
            elif (
                field
                not in {
                    "ttft_limit_s",
                    "tpot_limit_s",
                    "min_joint_pass_fraction",
                    "cpus",
                    "max_wait_s",
                }
                and type(value) is not int
            ):
                issues.append(f"{group}.{field}: integer required")
    fraction = config.get("slo", {}).get("min_joint_pass_fraction")
    if type(fraction) in (int, float) and fraction > 1:
        issues.append("slo.min_joint_pass_fraction must be <= 1")
    if config.get("container", {}).get("parallel_trials") != 1:
        issues.append(
            "this launcher requires parallel_trials=1; parallel scheduling is not implemented"
        )
    if type(config.get("search", {}).get("seed")) is not int:
        issues.append("search.seed: integer required")
    for field in ("reports", "full_graphs"):
        if type(config.get("artifacts", {}).get(field)) is not bool:
            issues.append(f"artifacts.{field}: boolean required")
    diagnostic_budget = config.get("artifacts", {}).get("diagnostic_budget_mib")
    if type(diagnostic_budget) is not int or diagnostic_budget < 0:
        issues.append("artifacts.diagnostic_budget_mib: nonnegative integer required")
    policy_diagnostics = config["artifacts"].get("policy_diagnostics")
    if policy_diagnostics is not None and (
        not isinstance(policy_diagnostics, dict)
        or set(policy_diagnostics) != {"max_samples", "every_decisions"}
        or type(policy_diagnostics.get("max_samples")) is not int
        or not 1 <= policy_diagnostics["max_samples"] <= 128
        or type(policy_diagnostics.get("every_decisions")) is not int
        or policy_diagnostics["every_decisions"] < 1
    ):
        issues.append(
            "artifacts.policy_diagnostics: max_samples must be 1..128 and "
            "every_decisions a positive integer"
        )
    for field in (
        "p_token_budget_bounds",
        "d_token_budget_bounds",
        "p_batch_bounds",
        "d_batch_bounds",
        "kv_inflight_bounds",
    ):
        pair = config.get("policy", {}).get(field)
        if (
            not isinstance(pair, list)
            or len(pair) != 2
            or any(type(x) is not int or x <= 0 for x in pair)
            or pair[0] > pair[1]
        ):
            issues.append(f"policy.{field}: ordered positive integer pair required")
    rates = config.get("loads", {}).get("arrival_rate_multipliers")
    if (
        not isinstance(rates, list)
        or not rates
        or any(
            type(x) not in (int, float) or not math.isfinite(x) or x <= 0 for x in rates
        )
    ):
        issues.append(
            "loads.arrival_rate_multipliers: positive finite numbers required"
        )
    elif 1 not in rates or len(set(rates)) != len(rates):
        issues.append("load grid must contain 1.0 and no duplicates")
    if config.get("model_assumptions", {}).get("length_model") != "known_trace":
        issues.append("only the existing known_trace length model can be prepared")
    if (
        config.get("model_assumptions", {}).get("scheduler_cost")
        != "zero_simulated_time"
    ):
        issues.append("online scheduler cost modelling is not implemented")
    agent = config.get("agent", {})
    if agent.get("frontend") != "codex":
        issues.append("this image provides the codex frontend")
    for field in ("model", "cli_version", "image"):
        if not isinstance(agent.get(field), str) or not agent[field].strip():
            issues.append(f"agent.{field}: required")
    workloads = config.get("workloads", [])
    ids = []
    for row in workloads:
        identifier = row.get("id", "")
        if (
            not isinstance(identifier, str)
            or not re.fullmatch(r"[A-Za-z0-9_-]+", identifier)
            or row.get("split") not in ("train", "validation")
        ):
            issues.append(
                "workloads require safe unique IDs and train/validation splits; holdout stays outside the agent"
            )
        if not isinstance(row.get("path"), str) or not row["path"].strip():
            issues.append("workload path required")
        ids.append(identifier)
    if (
        not ids
        or any(not isinstance(identifier, str) for identifier in ids)
        or len(set(ids)) != len(ids)
    ):
        issues.append("at least one workload and unique IDs required")
    if (
        not isinstance(config["inputs"].get("scenario"), str)
        or not config["inputs"]["scenario"].strip()
    ):
        issues.append("inputs.scenario required")
    if (
        not isinstance(config["policy"].get("api_version"), str)
        or not config["policy"]["api_version"].strip()
    ):
        issues.append("policy.api_version required")
    actions = config["policy"].get("actions")
    if (
        not isinstance(actions, list)
        or any(not isinstance(x, str) for x in actions)
        or set(actions) != {"route", "batch", "submit_kv", "wait"}
    ):
        issues.append("policy.actions must declare route, batch, submit_kv, wait")
    assets = config["inputs"].get("assets", [])
    if not isinstance(assets, list) or any(not isinstance(p, str) for p in assets):
        issues.append("inputs.assets: array of file paths required")
    return list(dict.fromkeys(issues))


def runtime_gaps(root, *, profile="legacy"):
    selected = profile_name(profile=profile)
    gaps = []
    if not (root / POLICY_REL).is_dir() or not list((root / POLICY_REL).glob("*.rs")):
        gaps.append(f"policy implementation is missing: {POLICY_REL}")
    if not (root / EVALUATOR_REL).is_file():
        gaps.append(f"trusted evaluator is missing: {EVALUATOR_REL}")
    if selected != "legacy":
        for name in ("runtime.py", "control.py"):
            relative = profile_directory(selected) / name
            if not (root / relative).is_file():
                gaps.append(f"task runtime is missing: {relative}")
    return gaps


def build_identity(root, versions):
    return hashlib.sha256(
        json_bytes(
            {
                "versions": versions,
                "files": {
                    name: digest(root / path) for name, path in BUILD_FILES.items()
                },
            }
        )
    ).hexdigest()


def environment_identity(root, versions):
    """Dependency installation is independent of controllers and Cargo profiles."""
    files = {}
    for name, path in BUILD_FILES.items():
        if name in CONTROLLER_FILES:
            continue
        if name == "Cargo.toml":
            manifest = tomllib.loads((root / path).read_text())
            manifest.pop("profile", None)
            files[name] = hashlib.sha256(json_bytes(manifest)).hexdigest()
        else:
            files[name] = digest(root / path)
    return hashlib.sha256(
        json_bytes({"schema_version": 1, "versions": versions, "files": files})
    ).hexdigest()


def local_image(image):
    result = subprocess.run(
        ["docker", "image", "inspect", image],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    if result.returncode:
        if "no such image" in result.stderr.lower():
            return None
        fail("cannot inspect local Docker image: " + result.stderr.strip())
    return json.loads(result.stdout)[0]


def reuse_build_image(root, image, versions):
    metadata = local_image(image)
    if metadata is None:
        return False
    labels = metadata.get("Config", {}).get("Labels") or {}
    expected = build_identity(root, versions)
    if labels.get(BUILD_LABEL) == expected:
        print("Build: local image already matches; reusing it.", flush=True)
        return True
    environment = environment_identity(root, versions)
    compatible = labels.get(ENVIRONMENT_LABEL) == environment
    if ENVIRONMENT_LABEL not in labels and labels.get(BUILD_LABEL):
        # Older images have only the complete fingerprint. Recompute it from
        # a frozen source bundle before trusting its environment inputs.
        for directory in PROFILES.values():
            for source in sorted(
                (root / directory / "runs").glob("*/source"), reverse=True
            ):
                try:
                    if build_identity(source, versions) != labels[BUILD_LABEL]:
                        continue
                    compatible = environment_identity(source, versions) == environment
                except FileNotFoundError, tomllib.TOMLDecodeError:
                    continue
                break
            if compatible:
                break
    if not compatible:
        return False
    print(
        "Build: verified existing dependencies; refreshing image offline.", flush=True
    )
    refresh_image(root, image, metadata["Id"], versions, expected, environment)
    return True


def refresh_image(root, image, base_id, versions, expected, environment):
    # The daemon's docker driver can use local images. A docker-container
    # builder would try to resolve this base from a registry instead.
    builder = output(["docker", "context", "show"])
    description = output(["docker", "buildx", "inspect", builder])
    if not re.search(r"^Driver:\s+docker\s*$", description, re.MULTILINE):
        fail("offline refresh requires the current Docker context's docker driver")
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", base_id):
        fail("invalid local base image ID")
    base_tag = f"simthesizer-agent-refresh:{os.getuid()}-{uuid.uuid4().hex}"
    with tempfile.TemporaryDirectory(prefix="sim-agent-refresh-") as name:
        temporary = Path(name)
        context = temporary / "context"
        context.mkdir()
        for target in (*CONTROLLER_FILES, "Cargo.toml"):
            shutil.copyfile(root / BUILD_FILES[target], context / target)
        (context / "Dockerfile").write_text(
            f"FROM {base_tag}\n"
            "COPY entrypoint.sh launcher.py baseline_store.py /opt/sim-agent/\n"
            "COPY Cargo.toml /opt/cargo-project/Cargo.toml\n"
            "RUN chmod 0755 /opt/sim-agent/entrypoint.sh "
            "&& chmod 0644 /opt/sim-agent/*.py /opt/cargo-project/Cargo.toml "
            "&& CARGO_HOME=/opt/cargo-seed /opt/sim-agent/entrypoint.sh --check-env "
            "&& CARGO_HOME=/opt/cargo-seed cargo fetch --offline --locked "
            "--manifest-path /opt/cargo-project/Cargo.toml "
            "&& python -m py_compile /opt/sim-agent/launcher.py /opt/sim-agent/baseline_store.py "
            f'&& test "$(codex --version)" = {shlex.quote("codex-cli " + versions["CODEX_VERSION"])}\n'
        )
        iidfile = temporary / "image-id"
        command(["docker", "tag", base_id, base_tag])
        try:
            build_command(
                [
                    "docker",
                    "buildx",
                    "build",
                    "--builder",
                    builder,
                    "--pull=false",
                    "--network=none",
                    "--load",
                    "--progress=plain",
                    "--iidfile",
                    str(iidfile),
                    "--label",
                    f"{BUILD_LABEL}={expected}",
                    "--label",
                    f"{ENVIRONMENT_LABEL}={environment}",
                    str(context),
                ],
                env=os.environ.copy(),
            )
            new_id = iidfile.read_text().strip()
            if not re.fullmatch(r"sha256:[0-9a-f]{64}", new_id):
                fail("offline refresh returned an invalid image ID")
            image_identity(new_id, expected)
        finally:
            command(
                ["docker", "image", "rm", "--no-prune", base_tag],
                stdout=subprocess.DEVNULL,
            )
        # Remove the temporary tag while the original image is still tagged.
        # Retagging only after validation also leaves that original image intact.
        command(["docker", "tag", new_id, image])


def driver_option(value):
    stream = io.StringIO()
    csv.writer(stream, lineterminator="").writerow([value])
    return stream.getvalue()


class BuildNetworkError(ValueError):
    pass


def network_failure(detail):
    # EOF can also mean a syntax error; require an HTTP request context.
    return bool(
        re.search(
            r"(?:502\s+bad gateway|503\s+service unavailable|504\s+gateway time.?out|"
            r"temporary failure resolving|could not resolve|could not connect|"
            r"connection (?:timed out|refused|reset)|proxyconnect tcp|dial tcp|"
            r"i/o timeout|tls handshake timeout|network is unreachable|"
            r"econnreset|econnrefused|etimedout|enotfound|eai_again|"
            r"operation timed out|"
            r'\b(?:head|get|post|put|patch|delete|connect|options)\s+"https?://[^"\r\n]+":\s*(?:unexpected\s+)?eof\b|'
            r"failed to do request:[^\r\n]*\beof\b|"
            r"proxy.*unsupported.*scheme|"
            r"unsupported.*proxy.*(?:scheme|protocol))",
            detail,
            re.IGNORECASE,
        )
    )


def validate_build_proxy(proxy):
    if not proxy:
        return
    parsed = urlsplit(proxy)
    if (
        parsed.scheme not in {"http", "https", "socks5", "socks5h"}
        or not parsed.hostname
        or parsed.query
        or parsed.fragment
    ):
        fail("build proxy must be an http(s) or socks5(h) URL")
    try:
        _ = parsed.port
    except ValueError:
        fail("invalid build proxy port")


@contextmanager
def project_builder(root, buildkit_image, proxy=None):
    # Buildx metadata and proxy settings belong to this project, never ~/.docker.
    state = root / "output/agent-build" / str(os.getuid())
    state.mkdir(parents=True, exist_ok=True, mode=0o700)
    proxies = {}
    for upper in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY"):
        value = (
            proxy
            if proxy and upper != "NO_PROXY"
            else os.environ.get(upper.lower(), os.environ.get(upper))
        )
        if value is not None:
            proxies[upper] = proxies[upper.lower()] = value
    daemon = output(["docker", "info", "--format", "{{.ID}}"])
    identity = hashlib.sha256(
        json_bytes(
            {
                "root": str(root.resolve()),
                "uid": os.getuid(),
                "daemon": daemon,
                "image": buildkit_image,
                "proxies": proxies,
            }
        )
    ).hexdigest()[:20]
    name = f"simthesizer-{identity}"
    config = state / identity
    config.mkdir(mode=0o700, exist_ok=True)
    env = {**os.environ, **proxies, "BUILDX_CONFIG": str(config)}
    # Never inherit another user's selected builder.
    env.pop("BUILDX_BUILDER", None)

    def invoke(argv, operation, *, allow_missing=False, timeout=None):
        result = subprocess.run(
            argv, env=env, capture_output=True, text=True, check=False, timeout=timeout
        )
        if result.returncode:
            if (
                allow_missing
                and "no builder" in result.stderr
                and name in result.stderr
            ):
                return None
            detail = result.stderr.strip()
            for value in sorted(set(proxies.values()), key=len, reverse=True):
                if value:
                    detail = detail.replace(value, "<proxy>")
            message = f"project builder {operation} failed: {detail}"
            if operation.startswith("bootstrap") and network_failure(detail):
                raise BuildNetworkError(message)
            fail(message)
        return result.stdout

    with (state / "build.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        info = invoke(
            ["docker", "buildx", "inspect", name], "inspect", allow_missing=True
        )
        if info is None:
            args = [
                "docker",
                "buildx",
                "create",
                "--name",
                name,
                "--driver",
                "docker-container",
                "--driver-opt",
                "network=host",
                "--driver-opt",
                driver_option(f"image={buildkit_image}"),
                "--buildkitd-flags=--allow-insecure-entitlement=network.host",
            ]
            for key, value in sorted(proxies.items()):
                args += ["--driver-opt", driver_option(f"env.{key}={value}")]
            invoke(args, "create")
        elif not re.search(r"^Driver:\s+docker-container\s*$", info, re.MULTILINE):
            fail(
                f"project builder {name} has an unexpected driver; refusing to change it"
            )
        print(f"Build: {name}; host network; metadata: {config}", flush=True)
        failed = False
        try:
            invoke(
                ["docker", "buildx", "inspect", "--bootstrap", name],
                f"bootstrap (requires local {buildkit_image} or registry access to fetch it)",
            )
            yield name, env
        except BaseException:
            failed = True
            raise
        finally:
            # The project lock covers bootstrap, build, and shutdown. Keep the
            # builder metadata/cache; stop only this UID/project/daemon identity.
            try:
                invoke(["docker", "buildx", "stop", name], "stop", timeout=30)
            except (OSError, ValueError, subprocess.SubprocessError) as error:
                print(
                    f"warning: {error}; builder metadata retained at {config}",
                    file=sys.stderr,
                )
                if not failed:
                    raise


def build_command(argv, env):
    # Keep progress visible without retaining an unbounded build log in memory.
    tail = deque(maxlen=160)
    with subprocess.Popen(
        argv, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    ) as process:
        for line in process.stdout:
            print(line, end="", flush=True)
            tail.append(line)
        status = process.wait()
    if status:
        if network_failure("".join(tail)):
            raise BuildNetworkError(
                f"image download/build network failure (exit {status})"
            )
        fail(f"image build failed (exit {status}); see build output above")


def build_image_once(root, image, versions, buildkit_image, proxy):
    with (
        project_builder(root, buildkit_image, proxy=proxy) as (builder, env),
        tempfile.TemporaryDirectory(prefix="sim-agent-build-") as name,
    ):
        context = Path(name)
        for target, relative in BUILD_FILES.items():
            destination = context / target
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(root / relative, destination)
        args = [
            "docker",
            "buildx",
            "build",
            "--builder",
            builder,
            "--load",
            "--network=host",
            "--allow=network.host",
            "--progress=plain",
            "--tag",
            image,
            "--file",
            str(context / "Dockerfile"),
            "--label",
            f"{ENVIRONMENT_LABEL}={environment_identity(root, versions)}",
        ]
        for key, value in {
            **versions,
            "BUILD_INPUT_SHA256": build_identity(root, versions),
        }.items():
            args += ["--build-arg", f"{key}={value}"]
        for key in sorted(PROXIES & env.keys()):
            args += ["--build-arg", key]
        build_command([*args, str(context)], env=env)
        image_identity(image, build_identity(root, versions))


def build_image(
    root,
    image,
    versions,
    buildkit_image=BUILDKIT_IMAGE,
    proxy=None,
    fallback_proxy=None,
):
    validate_build_proxy(proxy)
    validate_build_proxy(fallback_proxy)
    if reuse_build_image(root, image, versions):
        return
    try:
        build_image_once(root, image, versions, buildkit_image, proxy)
    except BuildNetworkError:
        if not fallback_proxy or fallback_proxy == proxy:
            raise
        print(
            "Build network failed; trying the configured fallback proxy once.",
            flush=True,
        )
        build_image_once(root, image, versions, buildkit_image, fallback_proxy)


def image_identity(image, expected):
    metadata = json.loads(output(["docker", "image", "inspect", image]))[0]
    labels = metadata.get("Config", {}).get("Labels") or {}
    if labels.get(BUILD_LABEL) != expected:
        fail("image build inputs differ; rebuild with run-agent.sh --build-only")
    return metadata["Id"]


def excluded(relative):
    return any(
        p in EXCLUDED or p.startswith(".env") or p.endswith((".env", ".pem", ".key"))
        for p in relative.parts
    )


def source_files(root, *, profile="legacy"):
    # Positive roots prevent copying arbitrary local experiments and credentials.
    for name in (
        "Cargo.toml",
        "Cargo.lock",
        "pyproject.toml",
        "uv.lock",
        ".python-version",
        "ARCHITECTURE.md",
    ):
        if (root / name).is_file():
            yield Path(name)
    names = (
        "crates",
        "scripts",
        "tests",
        "tools",
        "architectures",
        "examples/scenarios",
        "examples/policies",
        str(AGENT_REL),
    )
    selected = profile_directory(profile)
    if selected != AGENT_REL:
        names += (str(selected),)
    for name in names:
        for base, directories, files in os.walk(root / name, followlinks=False):
            relative = Path(base).relative_to(root)
            directories[:] = sorted(
                d
                for d in directories
                if not excluded(relative / d) and not (root / relative / d).is_symlink()
            )
            for filename in sorted(files):
                path = relative / filename
                if not excluded(path):
                    if (root / path).is_symlink():
                        fail(f"source symlink requires explicit resolution: {path}")
                    yield path


def inventory(root, *, mutable=False):
    if root.is_symlink():
        fail(f"unexpected symlink: {root}")
    result = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if mutable and relative.is_relative_to(POLICY_REL):
            continue
        if path.is_symlink():
            fail(f"unexpected symlink: {path}")
        if path.is_file():
            result[str(relative)] = digest(path)
    return result


def render_task(root, manifest):
    profile = manifest_profile(manifest)
    if profile != "legacy":
        return profile_runtime(root, profile).render_task(root, manifest)
    values = {
        "MANIFEST_PATH": "/workspace/task/manifest.json",
        "MANIFEST_SHA256": hashlib.sha256(json_bytes(manifest)).hexdigest(),
        "CONTRACT_PATH": f"/workspace/simthesizer/{AGENT_REL}/DESIGN.md",
        "POLICY_API_VERSION": manifest["experiment"]["policy"]["api_version"],
        "SOURCE_ROOT": "/workspace/simthesizer",
        "POLICY_DIRECTORY": f"/workspace/simthesizer/{POLICY_REL}",
        "ARTIFACTS_DIRECTORY": "/workspace/artifacts",
        "EVALUATOR_COMMAND": f"python /workspace/simthesizer/{EVALUATOR_REL} --manifest /workspace/task/manifest.json",
    }
    task = (root / AGENT_REL / "AGENTS.md.template").read_text()
    for key, value in values.items():
        task = task.replace("{{" + key + "}}", value)
    if re.search(r"\{\{[A-Z_]+\}\}", task):
        fail("unresolved task placeholders")
    if manifest.get("baseline_mode") == "local":
        task += """

## Frozen local baseline

The complete reference baseline was prepared before this container started.
Its evidence at /workspace/artifacts/baseline and anchor at
/workspace/task/baseline.json are read-only. The trusted evaluator validates
their identity and runtime compatibility before reuse. Never run a replacement
baseline, copy receipts into the search ledger, or change baseline evidence.
If reuse validation fails, report the error and stop; do not simulate a fallback
reference. Local baseline attempts and time already count toward search budgets.
"""
    return task


def prepare(
    root, config, runs_root, versions, network, *, baseline_mode=None, profile=None
):
    profile = profile_name(config, profile=profile)
    if baseline_mode not in (None, "local"):
        fail("unsupported baseline mode")
    runs_root.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    for suffix in range(10000):
        run = runs_root / (stamp if suffix == 0 else f"{stamp}-{suffix:03d}")
        try:
            run.mkdir(mode=0o700)
            break
        except FileExistsError:
            continue
    else:
        fail("cannot allocate a unique run directory")
    for name in ("source", "task", "inputs", "artifacts", "cache", "baseline-policy"):
        (run / name).mkdir()
    for relative in source_files(root, profile=profile):
        target = run / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / relative, target)
    if (run / "source" / POLICY_REL).is_dir():
        shutil.copytree(
            run / "source" / POLICY_REL, run / "baseline-policy", dirs_exist_ok=True
        )
    experiment = json.loads(json.dumps(config))
    inputs = [("scenario", config["inputs"]["scenario"])]
    inputs += [
        (f"workloads/{row['id']}.jsonl", row["path"]) for row in config["workloads"]
    ]
    inputs += [
        (f"assets/{i}/{Path(path).name}", path)
        for i, path in enumerate(config["inputs"].get("assets", []))
    ]
    mapping = {}
    for target, original in inputs:
        source = input_file(root, original)
        destination = run / "inputs" / target
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        mapping[str(source)] = f"/workspace/inputs/{target}"
    experiment["inputs"]["scenario"] = "/workspace/inputs/scenario"
    experiment["inputs"]["assets"] = [
        mapping[str(input_file(root, p))] for p in config["inputs"].get("assets", [])
    ]
    for row in experiment["workloads"]:
        row["path"] = f"/workspace/inputs/workloads/{row['id']}.jsonl"
    manifest = {
        "schema_version": 1,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "experiment": experiment,
        "network": network,
        "build_versions": versions,
        "build_input_sha256": build_identity(root, versions),
        "source_sha256": inventory(run / "source", mutable=True),
        "baseline_policy_sha256": inventory(run / "baseline-policy"),
        "inputs_sha256": inventory(run / "inputs"),
        "input_path_mapping": mapping,
        "runtime_gaps": runtime_gaps(root, profile=profile),
    }
    if baseline_mode is not None:
        manifest["baseline_mode"] = baseline_mode
    if profile != "legacy":
        manifest["profile"] = profile
        manifest["source_inventory_sha256"] = inventory(run / "source")
    save(run / "task/manifest.json", manifest)
    (run / "task/AGENTS.md").write_text(render_task(run / "source", manifest))
    save(run / "supervisor.json", {"consumed_wall_seconds": 0, "active": None})
    print(f"Prepared: {run}")
    return run


def load_bundle(run):
    manifest = json.loads((run / "task/manifest.json").read_text())
    profile = manifest_profile(manifest)
    if manifest.get("baseline_mode") not in (None, "local"):
        fail("unsupported frozen baseline mode")
    for folder, field in (("source", "source_sha256"), ("inputs", "inputs_sha256")):
        if inventory(run / folder, mutable=folder == "source") != manifest[field]:
            fail(f"frozen {folder} changed; start a new experiment")
    if profile != "legacy" and inventory(run / "source") != manifest.get(
        "source_inventory_sha256"
    ):
        fail("frozen source inventory changed; start a new experiment")
    if inventory(run / "baseline-policy") != manifest["baseline_policy_sha256"]:
        fail("baseline policy changed")
    if (run / "task/AGENTS.md").read_text() != render_task(run / "source", manifest):
        fail("rendered task changed")
    issues = (
        config_issues(manifest["experiment"])
        if profile == "legacy"
        else profile_runtime(run / "source", profile).config_issues(
            manifest["experiment"]
        )
    )
    if issues:
        fail("frozen experiment is invalid")
    candidate = run / "source" / POLICY_REL
    if candidate.exists():
        inventory(candidate)  # Reject symlinks even in the writable subtree.
    return manifest


def baseline_command(argv, *, env, cwd):
    process = None

    def interrupted(number, frame):
        raise InterruptedError(
            f"local baseline interrupted by {signal.Signals(number).name}"
        )

    handlers = {
        number: signal.signal(number, interrupted)
        for number in (signal.SIGINT, signal.SIGTERM)
    }
    try:
        process = subprocess.Popen(argv, env=env, cwd=cwd, start_new_session=True)
        status = process.wait()
        if status:
            raise subprocess.CalledProcessError(status, argv)
    finally:
        for number in handlers:
            signal.signal(number, signal.SIG_IGN)
        try:
            if process is not None and process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.wait(timeout=5)
        finally:
            for number, previous in handlers.items():
                signal.signal(number, previous)


def local_baseline(run, manifest):
    if manifest.get("baseline_mode") != "local":
        fail(
            "this frozen run has no local baseline mode; create a new --baseline-only run"
        )
    state = json.loads((run / "supervisor.json").read_text())
    if state.get("active"):
        fail(
            "a container is still recorded for this run; local baseline cannot run concurrently"
        )
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in {"CODEX_API_KEY", "OPENAI_API_KEY"}
    }
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    profile = manifest_profile(manifest)
    controller = (
        Path(__file__).resolve()
        if profile == "legacy"
        else run / "source" / profile_directory(profile) / "control.py"
    )
    resume = shlex.join(
        [
            sys.executable,
            str(controller),
            "--baseline-only",
            "--resume",
            str(run),
        ]
    )
    try:
        command(
            [
                sys.executable,
                "-c",
                (
                    "import sys, importlib.metadata; "
                    'assert sys.version_info[:2] == (3, 14), "local baseline requires project Python 3.14"; '
                    "import matplotlib, networkx; "
                    '[importlib.metadata.version(name) for name in ("matplotlib", "networkx")]'
                ),
            ],
            env=env,
            cwd=run / "source",
            timeout=30,
        )
        baseline_command(
            [
                sys.executable,
                str(run / "source" / EVALUATOR_REL),
                "--manifest",
                str(run / "task/manifest.json"),
                "--prepare-baseline",
            ],
            env=env,
            cwd=run / "source",
        )
        import baseline_store

        baseline_store.validate(run, manifest)
    except OSError, ValueError, subprocess.SubprocessError, KeyboardInterrupt:
        print(
            f"Local baseline remains incomplete. Run: {run}\nResume: {resume}",
            file=sys.stderr,
        )
        raise
    script = "run_rooflang_agent.sh" if profile == "legacy" else "run_qwen_moe_agent.sh"
    print(
        f"Local baseline ready: {run}\nStart agent with: ./{script} --resume {shlex.quote(str(run))}"
    )
    return 0


def check_env_file(path):
    found, nonempty = set(), set()
    for line in path.read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        key, separator, value = line.partition("=")
        if not separator or key not in ENV_KEYS or key in found:
            fail(
                "env file contains an unsupported/duplicate key; use agent.env.example"
            )
        found.add(key)
        if value:
            nonempty.add(key)
    if not nonempty & {"CODEX_API_KEY", "OPENAI_API_KEY"}:
        fail("env file needs CODEX_API_KEY or OPENAI_API_KEY")


def docker_arguments(run, manifest, image_id, env_file):
    limits = manifest["experiment"]["container"]
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
        "--network",
        manifest["network"],
        "--env-file",
        str(env_file),
    ]
    if manifest["network"] == "bridge":
        args += ["--add-host", "host.docker.internal:host-gateway"]
    for path, mode in (("/tmp", "1777"), ("/home/agent", "0700")):
        args += [
            "--tmpfs",
            f"{path}:rw,exec,nosuid,nodev,size=512m,uid={os.getuid()},gid={os.getgid()},mode={mode}",
        ]
    for source, target, readonly in (
        (run / "source", "/workspace/simthesizer", True),
        (run / "source" / POLICY_REL, f"/workspace/simthesizer/{POLICY_REL}", False),
        (run / "task", "/workspace/task", True),
        (run / "inputs", "/workspace/inputs", True),
        (run / "baseline-policy", "/workspace/baseline-policy", True),
        (run / "artifacts", "/workspace/artifacts", False),
        (run / "cache", "/workspace/cache", False),
    ):
        args += [
            "--mount",
            f"type=bind,source={source},target={target}"
            + (",readonly" if readonly else ""),
        ]
    if manifest.get("baseline_mode") == "local":
        args += [
            "--mount",
            f"type=bind,source={run / 'artifacts/baseline'},target=/workspace/artifacts/baseline,readonly",
        ]
    return [*args, image_id]


def stop_container(identifier):
    try:
        command(
            ["docker", "stop", "--time", "5", identifier],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=10,
        )
    except subprocess.SubprocessError, OSError:
        command(
            ["docker", "kill", identifier],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=10,
        )


def supervise(run, manifest, image_id, env_file, *, baseline=None):
    state_path = run / "supervisor.json"
    state = json.loads(state_path.read_text())
    if state.get("image_id", image_id) != image_id:
        fail("resume image ID differs from the original run")
    if state.get("active"):
        active = state["active"]
        stop_container(active["container"])
        subprocess.run(
            ["docker", "rm", active["container"]],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=10,
        )
        state["consumed_wall_seconds"] += max(0, time.time() - active["started_at"])
        state["active"] = None
        save(state_path, state)
    baseline_wall = 0
    if manifest.get("baseline_mode") == "local":
        if baseline is None:
            import baseline_store

            baseline = baseline_store.validate(run, manifest)
        baseline_wall = baseline["accounting"]["wall_seconds"]
    remaining = (
        manifest["experiment"]["search"]["max_wall_seconds"]
        - state["consumed_wall_seconds"]
        - baseline_wall
    )
    if remaining <= 0:
        fail("search wall-clock budget is exhausted; resume does not reset it")
    identifier = output(docker_arguments(run, manifest, image_id, env_file))
    started = time.monotonic()
    state.update(
        image_id=image_id, active={"container": identifier, "started_at": time.time()}
    )
    save(state_path, state)
    interrupted = []
    old_handlers = {
        sig: signal.signal(sig, lambda number, frame: interrupted.append(number))
        for sig in (signal.SIGINT, signal.SIGTERM)
    }
    process = None
    stopped = False
    outcome = "failed"
    exit_code = 1
    try:
        process = subprocess.Popen(
            ["docker", "start", "--attach", identifier],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        with (
            (run / "agent.trace.log").open("ab") as log,
            selectors.DefaultSelector() as selector,
        ):
            selector.register(process.stdout, selectors.EVENT_READ)
            while process.poll() is None or selector.get_map():
                if interrupted or time.monotonic() - started >= remaining:
                    outcome = "interrupted" if interrupted else "budget_exhausted"
                    stop_container(identifier)
                    stopped = True
                    break
                for key, _ in selector.select(timeout=0.2):
                    block = os.read(key.fileobj.fileno(), 65536)
                    if not block:
                        selector.unregister(key.fileobj)
                    else:
                        log.write(block)
                        log.flush()
                        sys.stdout.buffer.write(block)
                        sys.stdout.buffer.flush()
            # Drain final output while waiting, so a full pipe cannot block exit.
            tail, _ = process.communicate(timeout=15)
            if tail:
                log.write(tail)
                sys.stdout.buffer.write(tail)
                sys.stdout.buffer.flush()
        details = json.loads(
            output(["docker", "inspect", "--format", "{{json .State}}", identifier])
        )
        if details.get("Running"):
            fail("Docker attach exited while the container was still running")
        exit_code = details["ExitCode"]
        if details.get("OOMKilled"):
            outcome = "oom"
        elif outcome == "failed":
            outcome = "agent_exited" if exit_code == 0 else "failed"
        state["last_container_state"] = details
    finally:
        try:
            if not stopped:
                stop_container(identifier)
                stopped = True
        finally:
            try:
                if process is not None:
                    if process.poll() is None:
                        process.terminate()
                        try:
                            process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait(timeout=5)
                    process.stdout.close()
            finally:
                for sig, previous in old_handlers.items():
                    signal.signal(sig, previous)
                state.update(
                    last_outcome=outcome if stopped else "cleanup_failed",
                    last_exit_code=exit_code,
                )
                if stopped:
                    state["active"] = None
                    state["consumed_wall_seconds"] += time.monotonic() - started
                # If Docker cannot stop the container, preserve its ID and start
                # time for recovery instead of forgetting a potentially live run.
                save(state_path, state)
                if stopped:
                    subprocess.run(
                        ["docker", "rm", identifier],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        check=False,
                        timeout=10,
                    )
    print(f"\nRun: {run}\nStatus: {outcome}; agent exit is not verification.")
    return {"budget_exhausted": 124, "interrupted": 130, "oom": 137}.get(
        outcome, exit_code
    )


def enter_container():
    manifest_path = Path("/workspace/task/manifest.json")
    manifest = json.loads(manifest_path.read_text())
    if manifest_profile(manifest) != "legacy":
        fail("Qwen mechanism search requires its dedicated controller and entrypoint")
    root = Path("/workspace/simthesizer")
    if (
        inventory(root, mutable=True) != manifest["source_sha256"]
        or inventory(Path("/workspace/inputs")) != manifest["inputs_sha256"]
    ):
        fail("container source/input identity differs")
    if Path("/workspace/task/AGENTS.md").read_text() != render_task(root, manifest):
        fail("container task identity differs")
    if (
        inventory(Path("/workspace/baseline-policy"))
        != manifest["baseline_policy_sha256"]
    ):
        fail("container baseline policy identity differs")
    gaps = runtime_gaps(root)
    if gaps:
        fail("; ".join(gaps))
    clean_env = {
        k: v
        for k, v in os.environ.items()
        if k not in {"CODEX_API_KEY", "OPENAI_API_KEY"}
    }
    preflight = command(
        [
            sys.executable,
            str(root / EVALUATOR_REL),
            "--preflight",
            "--manifest",
            str(manifest_path),
        ],
        env=clean_env,
        timeout=manifest["experiment"]["search"]["trial_wall_seconds"],
        stdout=subprocess.PIPE,
        text=True,
    )
    capabilities = json.loads(preflight.stdout)
    if capabilities.get("policy_api_version") != manifest["experiment"]["policy"][
        "api_version"
    ] or not all(
        capabilities.get(key) is True
        for key in ("ready", "enforces_trial_budget", "enforces_trial_timeout")
    ):
        fail("evaluator preflight did not confirm the policy API and trial guards")
    agent = manifest["experiment"]["agent"]
    version = output(["codex", "--version"])
    if version.split()[-1] != agent["cli_version"]:
        fail("Codex CLI version differs from the frozen experiment")
    args = [
        "codex",
        "exec",
        "--dangerously-bypass-approvals-and-sandbox",
        "--skip-git-repo-check",
        "--ignore-user-config",
        "--ignore-rules",
        "--ephemeral",
        "--json",
        "--color",
        "never",
        "--cd",
        str(root),
        "--model",
        agent["model"],
        "--output-last-message",
        "/workspace/artifacts/agent-final.txt",
    ]
    env = os.environ.copy()
    key = env.get("CODEX_API_KEY") or env.get("OPENAI_API_KEY")
    if not key:
        fail("missing API key")
    env["CODEX_API_KEY"] = key
    if env.get("OPENAI_BASE_URL"):
        env["OPENAI_API_KEY"] = key
        args += [
            "-c",
            'model_provider="experiment_gateway"',
            "-c",
            'model_providers.experiment_gateway.name="Experiment Gateway"',
            "-c",
            "model_providers.experiment_gateway.base_url="
            + json.dumps(env["OPENAI_BASE_URL"]),
            "-c",
            'model_providers.experiment_gateway.env_key="OPENAI_API_KEY"',
            "-c",
            'model_providers.experiment_gateway.wire_api="responses"',
            "-c",
            "model_providers.experiment_gateway.requires_openai_auth=false",
        ]
    with Path("/workspace/task/AGENTS.md").open() as task:
        os.dup2(task.fileno(), 0)
        os.execvpe(args[0], [*args, "-"], env)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--build-only",
        action="store_true",
        help="build the environment image; no agent/API calls",
    )
    mode.add_argument(
        "--check",
        action="store_true",
        help="validate experiment and list missing runtime support; no Docker calls",
    )
    mode.add_argument(
        "--prepare-only",
        action="store_true",
        help="freeze a valid experiment; no Docker/API calls",
    )
    mode.add_argument(
        "--resume",
        type=Path,
        metavar="RUN_DIR",
        help="resume the frozen run without resetting its budget",
    )
    parser.add_argument(
        "--baseline-only",
        action="store_true",
        help="prepare a local baseline without Docker/API; may resume a local baseline run",
    )
    parser.add_argument("--experiment", type=Path)
    parser.add_argument("--profile", choices=tuple(PROFILES))
    parser.add_argument("--project", type=Path, default=ROOT)
    parser.add_argument("--runs-root", type=Path)
    parser.add_argument("--env-file", type=Path)
    parser.add_argument(
        "--image", help=f"build-only image tag (default: {IMAGE_DEFAULT})"
    )
    parser.add_argument(
        "--build", action="store_true", help="build before starting a new run"
    )
    parser.add_argument("--network", choices=["bridge", "host"], default="host")
    parser.add_argument("--buildkit-image", default=BUILDKIT_IMAGE)
    parser.add_argument(
        "--build-proxy", help="proxy for this build only; otherwise inherit shell"
    )
    parser.add_argument(
        "--build-fallback-proxy", help="retry once after a build network failure"
    )
    for key, default in BUILD_DEFAULTS.items():
        parser.add_argument("--" + key.lower().replace("_", "-"), default=default)
    parser.add_argument(
        "--inside-container", action="store_true", help=argparse.SUPPRESS
    )
    args = parser.parse_args(argv)
    if args.baseline_only and (
        args.build_only
        or args.build
        or args.check
        or args.prepare_only
        or args.inside_container
    ):
        fail(
            "--baseline-only only combines with --resume, not build/check/prepare modes"
        )
    if args.build and (
        args.build_only or args.check or args.prepare_only or args.resume
    ):
        fail("--build is only valid when starting a new search")
    if args.inside_container:
        enter_container()
        return 0
    root = safe_path(args.project)
    versions = {key: getattr(args, key.lower()) for key in BUILD_DEFAULTS}
    if args.build_only:
        build_image(
            root,
            args.image or IMAGE_DEFAULT,
            versions,
            args.buildkit_image,
            args.build_proxy,
            args.build_fallback_proxy,
        )
        return 0
    if args.resume:
        if args.experiment or args.build or args.image:
            fail(
                "resume uses frozen inputs/image; do not pass --experiment, --build or --image"
            )
        run = safe_path(args.resume)
    else:
        if not args.experiment:
            parser.error("--experiment is required (or use --build-only / --resume)")
        config = tomllib.loads(args.experiment.read_text())
        profile = profile_name(config, profile=args.profile)
        if profile != "legacy" and not (args.check or args.prepare_only):
            fail("Qwen mechanism search must use run_qwen_moe_agent.sh")
        gaps = runtime_gaps(root, profile=profile)
        issues = (
            config_issues(config)
            if profile == "legacy"
            else profile_runtime(root, profile).config_issues(config)
        )
        if not issues:
            for path in [
                config["inputs"]["scenario"],
                *config["inputs"].get("assets", []),
                *(r["path"] for r in config["workloads"]),
            ]:
                try:
                    input_file(root, path)
                except ValueError as error:
                    issues.append(str(error))
            if config["agent"]["cli_version"] != versions["CODEX_VERSION"]:
                issues.append("agent.cli_version differs from --codex-version")
            if args.image and args.image != config["agent"]["image"]:
                issues.append("--image differs from frozen agent.image")
        if args.check:
            print(
                json.dumps(
                    {
                        "configuration_issues": issues,
                        "runtime_gaps": gaps,
                        "launch_ready": not issues and not gaps,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 2 if issues or gaps else 0
        if issues:
            fail("invalid experiment:\n  " + "\n  ".join(issues))
        if gaps and not args.prepare_only:
            fail("search cannot start:\n  " + "\n  ".join(gaps))
        runs_root = resolve_runs_root(root, config, args.runs_root, profile=profile)
        if args.build:
            build_image(
                root,
                config["agent"]["image"],
                versions,
                args.buildkit_image,
                args.build_proxy,
                args.build_fallback_proxy,
            )
        run = prepare(
            root,
            config,
            runs_root,
            versions,
            args.network,
            baseline_mode="local" if args.baseline_only else None,
            profile=profile,
        )
        if args.prepare_only:
            print(
                "Search remains blocked: " + "; ".join(gaps)
                if gaps
                else "Prepared; container preflight still required."
            )
            return 0
    with (run / ".launcher.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            fail("another launcher already owns this run")
        manifest = load_bundle(run)
        if manifest_profile(manifest) != "legacy":
            fail("Qwen mechanism search must use run_qwen_moe_agent.sh")
        if args.profile is not None and args.profile != manifest_profile(manifest):
            fail("--profile differs from the frozen profile")
        gaps = runtime_gaps(run / "source")
        if gaps:
            fail("search cannot start: " + "; ".join(gaps))
        if args.baseline_only:
            return local_baseline(run, manifest)
        baseline = None
        if manifest.get("baseline_mode") == "local":
            import baseline_store

            baseline = baseline_store.validate(run, manifest)
        if args.env_file is None:
            fail("--env-file is required to start an agent")
        env_file = safe_path(args.env_file)
        check_env_file(env_file)
        image_id = image_identity(
            manifest["experiment"]["agent"]["image"], manifest["build_input_sha256"]
        )
        return supervise(run, manifest, image_id, env_file, baseline=baseline)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, OSError, KeyError, subprocess.SubprocessError) as error:
        print(f"error: {error}", file=sys.stderr)
        sys.exit(2)
