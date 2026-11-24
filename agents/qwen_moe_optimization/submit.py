"""Submit a bounded evaluation request to the host controller; never run candidates."""

import argparse
import json
import tempfile
import time
import uuid
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true")
    parser.add_argument(
        "--status", action="store_true", help="read host research budget and coverage"
    )
    parser.add_argument(
        "--scope",
        action="store_true",
        help="preview changed files and required checks without executing",
    )
    parser.add_argument("--plan", type=Path, help="structured mechanism plan JSON")
    parser.add_argument("--case")
    parser.add_argument("--reference", action="store_true")
    parser.add_argument("--replay")
    parser.add_argument("--hypothesis")
    parser.add_argument("--bottleneck")
    parser.add_argument("--evidence", default="")
    parser.add_argument("--ablation-of", default="")
    args = parser.parse_args(argv)
    if args.status:
        print(Path("/workspace/artifacts/research-status.json").read_text())
        return 0
    manifest = json.loads(Path("/workspace/task/manifest.json").read_text())
    import sys

    sys.path.insert(0, "/workspace/simthesizer/agents/find_pareto_frontier")
    import baseline_store
    import candidate
    import contract

    if args.scope:
        with tempfile.TemporaryDirectory(
            dir="/workspace/cache", prefix="scope-"
        ) as temporary:
            preview = candidate.snapshot(
                Path("/workspace/worktree"),
                Path("/workspace/simthesizer"),
                Path(temporary) / "snapshot",
                expected_source_sha256=manifest["source_inventory_sha256"],
            )
        print(
            json.dumps(
                {
                    key: preview[key]
                    for key in (
                        "contract_version",
                        "kind",
                        "source_sha256",
                        "changed_files",
                        "impact",
                    )
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    cases = baseline_store.cases(manifest)
    if args.list:
        print(json.dumps(cases, ensure_ascii=False, indent=2))
        return 0
    if args.case not in {row["id"] for row in cases}:
        parser.error("choose a frozen case from --list")
    if not args.hypothesis or not args.bottleneck:
        parser.error("--hypothesis and --bottleneck are required")
    request_id = uuid.uuid4().hex
    request = {
        "id": request_id,
        "case": args.case,
        "operation": "replay" if args.replay else "evaluate",
        "candidate_id": args.replay,
        "policy_kind": "reference" if args.reference else "candidate",
        "hypothesis": args.hypothesis,
        "bottleneck": args.bottleneck,
        "evidence": args.evidence,
        "ablation_of": args.ablation_of,
    }
    if args.plan:
        if args.replay:
            parser.error("replay reuses the sealed plan; omit --plan")
        with args.plan.open("rb") as stream:
            raw = stream.read(contract.MAX_PLAN_BYTES + 1)
        if len(raw) > contract.MAX_PLAN_BYTES:
            parser.error("plan exceeds byte limit")
        plan = json.loads(raw)
        if not isinstance(plan, dict):
            parser.error("plan must be a JSON object")
        request["plan"] = plan
    serialized = json.dumps(request, ensure_ascii=False)
    if len(serialized.encode()) > 65536:
        parser.error("request exceeds byte limit")
    queue = Path("/workspace/queue")
    temporary = queue / (request_id + ".tmp")
    temporary.write_text(serialized)
    temporary.rename(queue / (request_id + ".json"))
    print(f"Submitted {request_id}; waiting for controller", flush=True)
    response = Path("/workspace/artifacts/responses") / (request_id + ".json")
    timeout = manifest["experiment"]["search"]["max_wall_seconds"] + 60
    deadline = time.monotonic() + timeout
    while not response.exists():
        if time.monotonic() >= deadline:
            raise TimeoutError(
                "controller response not received; inspect host run logs"
            )
        time.sleep(0.5)
    result = json.loads(response.read_text())
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("status") == "EVALUATED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
