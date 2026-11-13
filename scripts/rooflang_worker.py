"""JSONL worker for the exclusive-resource Rooflang iteration runner."""

import argparse
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", type=Path)
    args = parser.parse_args()
    from rooflang.runtime.trace_export import export_trace
    from rooflang_bridge.llama import describe, simulate, validate_model

    if args.audit_dir:
        args.audit_dir.mkdir(parents=True, exist_ok=False)
    config = None
    iteration = 0
    for line in sys.stdin:
        try:
            req = json.loads(line)
            if req["op"] == "init":
                if req.get("version") != 1:
                    raise ValueError("unsupported protocol")
                validate_model(req["model"], req["tp"])
                expected = describe({"tp": req["tp"], "pp": 1, "ep": 1})
                if req["memory"] != {
                    k: expected[k]
                    for k in (
                        "weight_bytes_per_device",
                        "kv_bytes_per_token_per_device",
                    )
                }:
                    raise ValueError("model memory metadata does not match Llama graph")
                config = req
                reply = {"ready": True, "version": 1}
                if args.audit_dir:
                    (args.audit_dir / "init.json").write_text(json.dumps(req, indent=2))
            elif req["op"] == "iteration" and config:
                result, reply = simulate(req["batch"], config["tp"], config["hardware"])
                if args.audit_dir:
                    (args.audit_dir / f"{iteration:05d}.json").write_text(
                        json.dumps({"batch": req["batch"], **reply}, indent=2)
                    )
                    trace_path = args.audit_dir / f"{iteration:05d}.trace.json"
                    export_trace(result, str(trace_path))
                    trace = json.loads(trace_path.read_text())
                    # Rooflang exports class names; retain concrete model operation names.
                    events = trace["traceEvents"] if isinstance(trace, dict) else trace
                    timed = [e for e in events if e.get("ph") == "X"]
                    for event, entry in zip(timed, result.trace):
                        event["name"] = getattr(
                            entry.kernel, "name", type(entry.kernel).__name__
                        )
                    trace_path.write_text(json.dumps(trace))
                iteration += 1
            else:
                raise ValueError("initialize worker before iterations")
        except Exception as error:  # noqa: BLE001 -- translate worker failures into protocol errors
            reply = {"error": f"{type(error).__name__}: {error}"}
        print(json.dumps(reply, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
