"""Generate a frozen-input-ready, visible Qwen pressure suite; never overwrite."""

import argparse
import copy
import hashlib
import json
import random
import sys
import tomllib
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from results import load_workload
from rooflang_bridge.configuration import toml_document

ANCHOR = Path("examples/workloads/realistic.jsonl")
SCENARIO = "agents/qwen_moe_optimization/scenarios/shared-workload.toml"
PROFILES = {
    "short-input-long-output": ((16, 64), (256, 768)),
    "long-input-short-output": ((512, 1024), (8, 32)),
    "bursty": ((32, 256), (32, 256)),
}


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def synthetic_rows(name, count, seed):
    """Lengths are independent of opaque IDs; requests always run to completion."""
    if name not in PROFILES:
        raise ValueError("unknown pressure profile")
    if type(count) is not int or not 1 <= count <= 128:
        raise ValueError("requests_per_workload must be an integer in [1, 128]")
    if type(seed) is not int or not 0 <= seed < 2**64:
        raise ValueError("seed must be a u64 integer")
    rng = random.Random(f"qwen-pressure-v1:{seed}:{name}")
    ids = rng.sample(range(2**32), count)
    input_range, output_range = PROFILES[name]
    return [
        {
            "request_id": identity,
            "arrival_time": (index // 6) * 0.25
            if name == "bursty"
            else round(index * 0.05, 8),
            "initial": [0],
            "sub_requests": [
                {
                    "kind": "llm",
                    "input_tokens": rng.randint(*input_range),
                    "output_tokens": rng.randint(*output_range),
                    "next": [],
                }
            ],
        }
        for index, identity in enumerate(ids)
    ]


def generate(output, spec=HERE / "experiment.pressure.toml"):
    settings_path = Path(spec).resolve()
    settings_data = settings_path.read_bytes()
    settings = tomllib.loads(settings_data.decode())
    if set(settings) != {"base_experiment", "seed", "requests_per_workload"}:
        raise ValueError(
            "suite settings require base_experiment, seed, requests_per_workload"
        )
    if not isinstance(settings["base_experiment"], str):
        raise TypeError("base_experiment must be a path")
    base_path = ROOT / settings["base_experiment"]
    base_data = base_path.read_bytes()
    config = tomllib.loads(base_data.decode())
    if config["inputs"]["scenario"] != SCENARIO:
        raise ValueError(
            "pressure suite requires the unchanged shared-workload scenario"
        )
    if config.get("research", {}).get("profile") != "qwen_moe":
        raise ValueError("pressure suite requires a Qwen experiment")
    anchor_path = ROOT / ANCHOR
    anchor_data = anchor_path.read_bytes()
    anchor = load_workload(anchor_path)
    if len(anchor) != 100:
        raise ValueError("canonical anchor must contain all 100 requests")
    records = {"anchor-realistic": anchor}
    for name in PROFILES:
        records[name] = synthetic_rows(
            name, settings["requests_per_workload"], settings["seed"]
        )
    payloads = {"anchor-realistic.jsonl": anchor_data}
    payloads.update(
        {
            name + ".jsonl": b"".join(
                (json.dumps(row, sort_keys=True) + "\n").encode() for row in rows
            )
            for name, rows in records.items()
            if name != "anchor-realistic"
        }
    )
    manifest = {
        "schema_version": 1,
        "generator": "qwen-pressure-v1",
        "generator_sha256": sha256(Path(__file__).read_bytes()),
        "settings_sha256": sha256(settings_data),
        "base_experiment_sha256": sha256(base_data),
        "seed": settings["seed"],
        "visibility": "visible",
        "hidden_holdout": False,
        "assumptions": [
            "Synthetic length-only inputs are diagnostic, not real text or traffic.",
            "Requested output lengths are complete; no completion truncation.",
            "Existing known_trace and deterministic model routing are unchanged.",
        ],
        "workloads": [
            {
                "id": name,
                "path": name + ".jsonl",
                "split": "train" if name == "anchor-realistic" else "validation",
                "origin": "canonical_anchor"
                if name == "anchor-realistic"
                else "synthetic",
                "sha256": sha256(payloads[name + ".jsonl"]),
                "requests": len(rows),
                "input_tokens": sum(r["sub_requests"][0]["input_tokens"] for r in rows),
                "output_tokens": sum(
                    r["sub_requests"][0]["output_tokens"] for r in rows
                ),
            }
            for name, rows in records.items()
        ],
    }
    output = Path(output).absolute()
    config = copy.deepcopy(config)
    config["workloads"] = [
        {"id": row["id"], "path": str(output / row["path"]), "split": row["split"]}
        for row in manifest["workloads"]
    ]
    config["inputs"]["assets"] = [
        *config["inputs"].get("assets", []),
        str(output / "suite.json"),
    ]
    payloads["suite.json"] = json_bytes(manifest)
    payloads["experiment.toml"] = (
        "# Generated visible pressure suite; synthetic validation is not hidden holdout.\n"
        + toml_document(config)
    ).encode()
    # Refuse existing directories, including empty ones and symlinks.
    output.mkdir(parents=True, exist_ok=False)
    for name, data in payloads.items():
        with (output / name).open("xb") as stream:
            stream.write(data)
    return output / "experiment.toml"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, required=True, help="new suite directory"
    )
    parser.add_argument("--spec", type=Path, default=HERE / "experiment.pressure.toml")
    args = parser.parse_args()
    try:
        path = generate(args.output, args.spec)
    except (OSError, ValueError, TypeError, KeyError) as error:
        parser.exit(2, f"{error}\n")
    print(path)


if __name__ == "__main__":
    main()
