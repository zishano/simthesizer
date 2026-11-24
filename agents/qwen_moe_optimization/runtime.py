"""Frozen instructions and research-container entrypoint for mechanism search."""

import json
import os
import subprocess
import sys
from pathlib import Path

REL = Path("agents/qwen_moe_optimization")
SCOPE_CONTRACT = "qwen-cross-module-v1"


def config_issues(config):
    # Shared launcher loads this module without changing sys.path.
    import launcher

    issues = launcher.config_issues(config)
    if config.get("research", {}).get("profile") != "qwen_moe":
        issues.append("research.profile must be qwen_moe")
    if config.get("research", {}).get("scope_contract") != SCOPE_CONTRACT:
        issues.append(f"research.scope_contract must be {SCOPE_CONTRACT}")
    research = config.get("research", {})
    if research.get("workflow") not in (None, "bottleneck-v1"):
        issues.append("research.workflow must be bottleneck-v1 or omitted")
    limit = research.get("max_stalled_sessions", 3)
    if type(limit) is not int or not 1 <= limit <= 10:
        issues.append("research.max_stalled_sessions must be an integer in 1..10")
    if config.get("container", {}).get("parallel_trials") != 1:
        issues.append("Qwen mechanism evaluation currently requires parallel_trials=1")
    return issues


def render_task(root, manifest):
    template = (Path(root) / REL / "SEARCH_TASK.md").read_text()
    config = manifest["experiment"]
    settings = {
        key: config[key]
        for key in (
            "slo",
            "loads",
            "workloads",
            "policy",
            "search",
            "verification",
            "research",
            "inputs",
        )
    }
    return (
        template
        + (
            "\n本 run 启用 bottleneck-v1。每次启动先读取 "
            "`/workspace/artifacts/research-status.json`、基线诊断和已有 notes；"
            "这是持续研究的新会话，不是从头重置预算。\n"
            if config["research"].get("workflow") == "bottleneck-v1"
            else ""
        )
        + "\n## 冻结参数\n\n```json\n"
        + json.dumps(settings, ensure_ascii=False, sort_keys=True, indent=2)
        + "\n```\n"
    )


def enter_container():
    root = Path("/workspace/simthesizer")
    sys.path.insert(0, str(root / "agents/find_pareto_frontier"))
    import baseline_store
    import launcher

    manifest = json.loads(Path("/workspace/task/manifest.json").read_text())
    if manifest.get("profile") != "qwen_moe" or config_issues(manifest["experiment"]):
        raise ValueError("invalid Qwen experiment")
    if launcher.inventory(root) != manifest["source_inventory_sha256"]:
        raise ValueError("frozen source changed")
    if launcher.inventory(Path("/workspace/inputs")) != manifest["inputs_sha256"]:
        raise ValueError("frozen inputs changed")
    prompt = Path("/workspace/task/AGENTS.md")
    if prompt.read_text() != render_task(root, manifest):
        raise ValueError("frozen task changed")
    if (
        launcher.inventory(Path("/workspace/baseline-policy"))
        != manifest["baseline_policy_sha256"]
    ):
        raise ValueError("frozen baseline policy changed")
    baseline = baseline_store.validate(Path("/workspace"), manifest)
    if baseline["result_environment"] != baseline_store.result_environment():
        raise ValueError("container result environment differs from the local baseline")
    agent = manifest["experiment"]["agent"]
    version = subprocess.check_output(["codex", "--version"], text=True).strip()
    if version.split()[-1] != agent["cli_version"]:
        raise ValueError("Codex version differs from frozen experiment")
    env = os.environ.copy()
    key = env.get("CODEX_API_KEY") or env.get("OPENAI_API_KEY")
    if not key:
        raise ValueError("missing API key")
    env["CODEX_API_KEY"] = key
    env["PYTHONDONTWRITEBYTECODE"] = "1"
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
        "/workspace/worktree",
        "--model",
        agent["model"],
        "--output-last-message",
        "/workspace/notes/agent-final.txt",
    ]
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
    with prompt.open() as stream:
        os.dup2(stream.fileno(), 0)
        os.execvpe(args[0], [*args, "-"], env)


if __name__ == "__main__":
    if sys.argv[1:] != ["--inside"]:
        raise SystemExit("use run_qwen_moe_agent.sh")
    enter_container()
