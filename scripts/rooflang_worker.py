"""JSONL worker for Rooflang iteration pricing and native operator lowering."""

import argparse
import gc
import json
import sys
from pathlib import Path

from rooflang_bridge.wall_profile import encode_reply, request_profile, scope


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", type=Path)
    parser.add_argument(
        "--reference-every-round",
        action="store_true",
        help="diagnostic: rerun the isolated shared-bridge reference every round",
    )
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

    if args.audit_dir:
        args.audit_dir.mkdir(parents=True, exist_ok=False)
    config = None
    template = None
    qwen = None
    iteration = 0
    for line in sys.stdin:
        profile = request_profile(line)
        restore_gc = False
        try:
            with scope(profile, "request_decode"):
                req = json.loads(line)
            if req.get("profile_wall") is not True:
                profile = None
            if req["op"] == "init":
                with scope(profile, "worker_init"):
                    from rooflang.runtime.trace_export import export_trace
                    from rooflang_bridge.hardware import build_hardware
                    from rooflang_bridge.llama import (
                        LlamaTemplate,
                        describe,
                        simulate,
                        validate_model,
                    )

                    if req.get("version") != 1:
                        raise ValueError("unsupported protocol")
                    if req.get("model_kind") == "qwen3_235b_a22b":
                        from rooflang_bridge.qwen3 import QwenTemplate
                        from rooflang_bridge.qwen3 import (
                            validate_model as validate_qwen,
                        )

                        if args.reference_every_round:
                            raise ValueError(
                                "Qwen shared EP has no isolated reference; --reference-every-round is unsupported"
                            )
                        expert_tp = req.get("expert_tp", 1)
                        validate_qwen(
                            req["model"], req["memory"], req["tp"], req["ep"], expert_tp
                        )
                        if req["tp"] > 1 and "attention_groups" not in req:
                            raise ValueError(
                                "Qwen TP requires explicit attention_groups"
                            )
                        qwen = QwenTemplate(
                            req["tp"],
                            req["ep"],
                            req["hardware"],
                            req["routing_seed"],
                            expert_tp=expert_tp,
                            attention_groups=req.get("attention_groups"),
                        )
                        template = None
                        hardware_info = qwen.hardware
                    else:
                        if req.get("model_kind") is not None:
                            raise ValueError("unsupported Rooflang graph model_kind")
                        qwen = None
                        validate_model(req["model"], req["tp"])
                        expected = describe({"tp": req["tp"], "pp": 1, "ep": 1})
                        if req["memory"] != {
                            k: expected[k]
                            for k in (
                                "weight_bytes_per_device",
                                "kv_bytes_per_token_per_device",
                            )
                        }:
                            raise ValueError(
                                "model memory metadata does not match Llama graph"
                            )
                        if args.rebuild_graph:
                            _, _, _, hardware_info = build_hardware(
                                req["hardware"], req["tp"]
                            )
                            template = None
                        else:
                            template = LlamaTemplate(req["tp"], req["hardware"])
                            hardware_info = template.summary["hardware"]
                    req["resolved_hardware"] = hardware_info
                    config = req
                    reply = {"ready": True, "version": 1, "hardware": hardware_info}
                if args.audit_dir:
                    with scope(profile, "audit_write"):
                        (args.audit_dir / "init.json").write_text(
                            json.dumps(req, indent=2)
                        )
            elif req["op"] == "graph" and config:
                if qwen is not None and gc.isenabled():
                    # Qwen replies contain only acyclic values and integer edges.
                    # Refcounts reclaim them; repeated cyclic-GC scans only walk
                    # the growing reply. Resume after releasing that reply.
                    gc.disable()
                    restore_gc = True
                with scope(profile, "graph_build_other"):
                    from rooflang_bridge.execution import export_execution

                    selected = args.audit_dir and (
                        trace_rounds is None or iteration in trace_rounds
                    )
                    if qwen is not None:
                        reply = qwen.export(
                            req["batch"],
                            req["requests"],
                            details=bool(args.audit_dir) or req.get("details", True),
                            routing=req.get("routing"),
                            wall_profile=profile,
                            compact_work=req.get("compact_work") is True,
                            operation_rows=req.get("operation_rows") is True,
                        )
                    else:
                        if template is None:
                            template = LlamaTemplate(config["tp"], config["hardware"])
                        reply = export_execution(
                            template,
                            req["batch"],
                            reference=args.reference_every_round or bool(selected),
                            details=bool(args.audit_dir) or req.get("details", True),
                        )
                if args.audit_dir:
                    with scope(profile, "audit_write"):
                        identities = (
                            {"requests": req["requests"]} if qwen is not None else {}
                        )
                        (args.audit_dir / f"{iteration:05d}.json").write_text(
                            json.dumps(
                                {
                                    "batch": req["batch"],
                                    **identities,
                                    "isolated_reference": reply["reference"],
                                }
                            )
                        )
                        if trace_rounds is None or iteration in trace_rounds:
                            (args.audit_dir / f"{iteration:05d}.plan.json").write_text(
                                json.dumps(
                                    {"batch": req["batch"], **identities, **reply}
                                )
                            )
                iteration += 1
            elif req["op"] == "iteration" and config:
                with scope(profile, "graph_build_other"):
                    if qwen is not None:
                        raise ValueError(
                            "Qwen graph requires shared execution; isolated iteration pricing is unsupported"
                        )
                    if template is None:
                        result, reply = simulate(
                            req["batch"], config["tp"], config["hardware"]
                        )
                    else:
                        result, reply = template.simulate(req["batch"])
                if args.audit_dir:
                    with scope(profile, "audit_write"):
                        (args.audit_dir / f"{iteration:05d}.json").write_text(
                            json.dumps({"batch": req["batch"], **reply}, indent=2)
                        )
                        if trace_rounds is None or iteration in trace_rounds:
                            trace_path = args.audit_dir / f"{iteration:05d}.trace.json"
                            export_trace(result, str(trace_path))
                            trace = json.loads(trace_path.read_text())
                            # Rooflang exports class names; retain concrete model operation names.
                            events = (
                                trace["traceEvents"]
                                if isinstance(trace, dict)
                                else trace
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
        try:
            try:
                payload = encode_reply(reply, profile)
            except Exception as error:  # noqa: BLE001 -- encoding failures are protocol errors too
                payload = encode_reply(
                    {"error": f"{type(error).__name__}: {error}"}, profile
                )
            print(payload, flush=True)
        finally:
            reply = payload = None
            if restore_gc:
                gc.enable()


if __name__ == "__main__":
    main()
