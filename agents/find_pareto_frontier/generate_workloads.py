"""Generate synthetic, seeded single-LLM workloads offline; never changes an experiment."""

import argparse
import hashlib
import json
import math
import random
from pathlib import Path

VERSION = "serving-synthetic-v1"
PROFILES = ("sustained", "burst", "mixed", "context")


def encoded(value):
    return (json.dumps(value, sort_keys=True, allow_nan=False) + "\n").encode()


def validate(parameters):
    for name in ("num_requests", "burst_size", "max_context_tokens"):
        if type(parameters[name]) is not int or parameters[name] <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if parameters["max_context_tokens"] > 2**32 - 1:
        raise ValueError("max_context_tokens must fit native u32 token counts")
    for name in ("train_seed", "validation_seed"):
        if type(parameters[name]) is not int:
            raise ValueError(f"{name} must be an integer")
    if parameters["train_seed"] == parameters["validation_seed"]:
        raise ValueError("train and validation seeds must differ")
    for name in ("short_input_tokens", "long_input_tokens", "output_tokens"):
        bounds = parameters[name]
        if (
            not isinstance(bounds, (list, tuple))
            or len(bounds) != 2
            or any(type(x) is not int or x <= 0 for x in bounds)
            or bounds[0] > bounds[1]
        ):
            raise ValueError(f"{name} must be ordered positive integer bounds")
    if parameters["short_input_tokens"][1] >= parameters["long_input_tokens"][0]:
        raise ValueError("short and long input ranges must be disjoint and ordered")
    if (
        parameters["long_input_tokens"][1] + parameters["output_tokens"][1]
        > parameters["max_context_tokens"]
    ):
        raise ValueError("input plus output length exceeds max_context_tokens")
    for name in ("arrival_start_s", "arrival_interval_s", "arrival_jitter"):
        value = parameters[name]
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    if parameters["arrival_interval_s"] == 0:
        raise ValueError("arrival_interval_s must be positive")
    if parameters["arrival_jitter"] >= 1:
        raise ValueError("arrival_jitter must be less than 1")
    profiles = parameters["profiles"]
    if (
        not isinstance(profiles, (list, tuple))
        or not profiles
        or any(profile not in PROFILES for profile in profiles)
        or len(set(profiles)) != len(profiles)
    ):
        raise ValueError(f"profiles must be unique members of {PROFILES}")


def generate(parameters, profile, seed):
    """The profile name isolates random streams from selection/order of other profiles."""
    derived_seed = int.from_bytes(
        hashlib.sha256(f"{VERSION}:{profile}:{seed}".encode()).digest()
    )
    rng = random.Random(derived_seed)
    now = parameters["arrival_start_s"]
    burst_size = parameters["burst_size"] if profile == "burst" else 1
    for identity in range(parameters["num_requests"]):
        if identity and identity % burst_size == 0:
            gap = parameters["arrival_interval_s"] * burst_size
            jitter = parameters["arrival_jitter"]
            following = now + gap * rng.uniform(1 - jitter, 1 + jitter)
            if not math.isfinite(following) or following <= now:
                raise ValueError(
                    "arrival interval overflows or loses floating-point precision"
                )
            now = following
        long = profile == "context" or (profile == "mixed" and identity % 2 == 1)
        bounds = parameters["long_input_tokens" if long else "short_input_tokens"]
        yield {
            "request_id": identity,
            "arrival_time": now,
            "initial": [0],
            "sub_requests": [
                {
                    "kind": "llm",
                    "input_tokens": rng.randint(*bounds),
                    "output_tokens": rng.randint(*parameters["output_tokens"]),
                    "next": [],
                }
            ],
        }


def write_suite(output_dir, parameters):
    validate(parameters)
    output_dir = Path(output_dir)
    # A fresh directory prevents regenerating inputs of an existing frozen experiment.
    output_dir.mkdir(parents=True, exist_ok=False)
    manifest = {
        "schema_version": 1,
        "generator": VERSION,
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "parameters": parameters,
        "assumptions": {
            "synthetic": True,
            "holdout": False,
            "arrival_interval_s": "mean interarrival; burst groups use interval * burst_size",
            "arrival_jitter": "uniform fractional gap jitter; none inside a burst",
            "lengths": "inclusive uniform integer ranges; mixed alternates short/long prompts",
            "token_content": "omitted; not a prefix-cache workload",
            "capacity_claim": "finite diagnostic workload; no sustained-capacity guarantee",
        },
        "workloads": [],
    }
    for profile in parameters["profiles"]:
        for split in ("train", "validation"):
            seed = parameters[f"{split}_seed"]
            name = f"{profile}-{split}"
            path = output_dir / f"{name}.jsonl"
            checksum = hashlib.sha256()
            first = last = None
            with path.open("wb") as target:
                for row in generate(parameters, profile, seed):
                    data = encoded(row)
                    target.write(data)
                    checksum.update(data)
                    if first is None:
                        first = row["arrival_time"]
                    last = row["arrival_time"]
            manifest["workloads"].append(
                {
                    "id": name,
                    "path": path.name,
                    "split": split,
                    "seed": seed,
                    "profile": profile,
                    "sha256": checksum.hexdigest(),
                    "num_requests": parameters["num_requests"],
                    "first_arrival_s": first,
                    "last_arrival_s": last,
                }
            )
    # Only a complete suite has a manifest. Interrupted output can be inspected;
    # rerunning must use a new directory.
    (output_dir / "manifest.json").write_bytes(encoded(manifest))
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path, help="new directory")
    parser.add_argument(
        "--num-requests", required=True, type=int, help="per profile/split"
    )
    parser.add_argument("--arrival-interval-s", required=True, type=float)
    parser.add_argument("--arrival-start-s", type=float, default=0.0)
    parser.add_argument(
        "--arrival-jitter", type=float, default=0.25, help="fraction in [0, 1)"
    )
    parser.add_argument("--burst-size", type=int, default=8)
    parser.add_argument(
        "--short-input-tokens",
        type=int,
        nargs=2,
        default=[16, 128],
        metavar=("MIN", "MAX"),
    )
    parser.add_argument(
        "--long-input-tokens",
        type=int,
        nargs=2,
        default=[2048, 4096],
        metavar=("MIN", "MAX"),
    )
    parser.add_argument(
        "--output-tokens", type=int, nargs=2, default=[16, 128], metavar=("MIN", "MAX")
    )
    parser.add_argument(
        "--max-context-tokens",
        type=int,
        default=131072,
        help="input + output bound; match the selected model",
    )
    parser.add_argument("--train-seed", type=int, default=0)
    parser.add_argument("--validation-seed", type=int, default=1)
    parser.add_argument(
        "--profiles", nargs="+", choices=PROFILES, default=list(PROFILES)
    )
    args = vars(parser.parse_args(argv))
    output_dir = args.pop("output_dir")
    try:
        write_suite(output_dir, args)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print(output_dir / "manifest.json")


if __name__ == "__main__":
    main()
