"""JSONL worker for the exclusive-resource Rooflang iteration runner."""

import argparse
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", type=Path)
    parser.add_argument(
        "--rebuild-graph",
        action="store_true",
        help="diagnostic reference: rebuild the graph every iteration",
    )
    parser.add_argument(
        "--audit-trace-rounds",
        type=int,
        nargs="+",
        help="zero-based rounds to export detailed traces for; summaries still cover every round",
    )
    args = parser.parse_args()
    if args.audit_trace_rounds is not None and (
        args.audit_dir is None or any(i < 0 for i in args.audit_trace_rounds)
    ):
        parser.error(
            "--audit-trace-rounds requires --audit-dir and nonnegative indices"
        )
    trace_rounds = (
        None if args.audit_trace_rounds is None else set(args.audit_trace_rounds)
    )
    from rooflang.runtime.trace_export import export_trace
    from rooflang_bridge.hardware import build_hardware
    from rooflang_bridge.llama import LlamaTemplate, describe, simulate, validate_model

    if args.audit_dir:
        args.audit_dir.mkdir(parents=True, exist_ok=False)
    config = None
    template = None
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
                if args.rebuild_graph:
                    _, _, _, hardware_info = build_hardware(req["hardware"], req["tp"])
                    template = None
                else:
                    template = LlamaTemplate(req["tp"], req["hardware"])
                    hardware_info = template.summary["hardware"]
                req["resolved_hardware"] = hardware_info
                config = req
                reply = {"ready": True, "version": 1, "hardware": hardware_info}
                if args.audit_dir:
                    (args.audit_dir / "init.json").write_text(json.dumps(req, indent=2))
            elif req["op"] == "iteration" and config:
                if template is None:
                    result, reply = simulate(
                        req["batch"], config["tp"], config["hardware"]
                    )
                else:
                    result, reply = template.simulate(req["batch"])
                if args.audit_dir:
                    (args.audit_dir / f"{iteration:05d}.json").write_text(
                        json.dumps({"batch": req["batch"], **reply}, indent=2)
                    )
                    if trace_rounds is None or iteration in trace_rounds:
                        trace_path = args.audit_dir / f"{iteration:05d}.trace.json"
                        export_trace(result, str(trace_path))
                        trace = json.loads(trace_path.read_text())
                        # Rooflang exports class names; retain concrete model operation names.
                        events = (
                            trace["traceEvents"] if isinstance(trace, dict) else trace
                        )
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
