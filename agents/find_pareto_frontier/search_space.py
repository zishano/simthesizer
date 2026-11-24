"""Generate opt-in scheduling policies, combinations and component ablations."""

import argparse
import hashlib
import json
import math
from pathlib import Path

from baseline_store import identity, json_bytes

ROOT = Path(__file__).resolve().parents[2]
LIBRARY = ROOT / "examples/policies/composable.rs"
COMPONENTS = (
    "routing_load",
    "aged_short_prefill",
    "decode_fair",
    "microbatch_wait",
    "pressure_chunks",
    "kv_demand_aging",
    "hysteresis",
)
DEFAULT_RECIPE = {
    **dict.fromkeys(COMPONENTS, False),
    "wait_s": 0.0001,
    "age_s": 0.01,
    "pressure_low_s": 0.0001,
    "pressure_high_s": 0.001,
    "dwell_s": 0.001,
    "small_chunk_tokens": 32,
    "large_chunk_tokens": 512,
    "wait_batch_threshold": 4,
}


def validate_recipe(recipe):
    """Normalize only known keys; bool is not an integer parameter."""
    if not isinstance(recipe, dict) or set(recipe) - DEFAULT_RECIPE.keys():
        raise ValueError("recipe must be an object containing only known parameters")
    result = DEFAULT_RECIPE | recipe
    for key in COMPONENTS:
        if type(result[key]) is not bool:
            raise ValueError(f"{key} must be boolean")
    for key in ("wait_s", "age_s", "pressure_low_s", "pressure_high_s", "dwell_s"):
        value = result[key]
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{key} must be finite and nonnegative")
        # Canonical bytes should not depend on JSON integer vs floating spelling.
        result[key] = float(value) if value else 0.0
    if result["wait_s"] <= 0 or result["age_s"] <= 0:
        raise ValueError("wait_s and age_s must be positive")
    if result["pressure_low_s"] >= result["pressure_high_s"]:
        raise ValueError("pressure_low_s must be less than pressure_high_s")
    for key in ("small_chunk_tokens", "large_chunk_tokens", "wait_batch_threshold"):
        if type(result[key]) is not int or not 1 <= result[key] <= 2**32 - 1:
            raise ValueError(f"{key} must be a positive u32 integer")
    if result["small_chunk_tokens"] > result["large_chunk_tokens"]:
        raise ValueError("small_chunk_tokens must not exceed large_chunk_tokens")
    return result


def families(recipe):
    normalized = validate_recipe(recipe)
    return [name for name in COMPONENTS if normalized[name]]


def render(recipe):
    recipe = validate_recipe(recipe)
    fields = "\n".join(
        f"    {key}: {str(value).lower() if type(value) is bool else repr(value)},"
        for key, value in recipe.items()
    )
    wrapper = f"""// Generated scheduling hypothesis; evaluate before promotion.
mod scheduler;
use super::{{
    BatchAction, BatchObservation, KvAction, KvObservation, PolicyContext,
    RouteAction, RouteObservation,
}};
const OPTIONS: scheduler::Options = scheduler::Options {{
{fields}
}};

pub fn route(o: &RouteObservation<'_>, c: &PolicyContext<'_>, m: &mut [u8]) -> RouteAction {{
    scheduler::route_with(o, c, m, &OPTIONS)
}}
pub fn batch(o: &BatchObservation<'_>, c: &PolicyContext<'_>, m: &mut [u8]) -> BatchAction {{
    scheduler::batch_with(o, c, m, &OPTIONS)
}}
pub fn kv(o: &KvObservation<'_>, c: &PolicyContext<'_>, m: &mut [u8]) -> KvAction {{
    scheduler::kv_with(o, c, m, &OPTIONS)
}}
"""
    return {"mod.rs": wrapper, "scheduler.rs": LIBRARY.read_text()}


def expand(recipe):
    """Full recipe, baseline, individual components and leave-one-out controls."""
    recipe = validate_recipe(recipe)
    active = families(recipe)
    baseline = recipe | dict.fromkeys(COMPONENTS, False)
    rows = [baseline, recipe]
    for name in active:
        rows.extend((baseline | {name: True}, recipe | {name: False}))
    if recipe["pressure_chunks"]:
        # Same ordering, wait and KV rules; only adaptive chunk choice is removed.
        for size in (recipe["small_chunk_tokens"], recipe["large_chunk_tokens"]):
            rows.append(
                recipe | {"small_chunk_tokens": size, "large_chunk_tokens": size}
            )
    return list({identity(row): row for row in rows}.values())


def default_recipes():
    combinations = (
        ("aged_short_prefill", "decode_fair"),
        ("aged_short_prefill", "microbatch_wait"),
        ("kv_demand_aging", "hysteresis"),
        ("pressure_chunks", "decode_fair", "kv_demand_aging", "hysteresis"),
        COMPONENTS,
    )
    rows = []
    for group in combinations:
        rows.extend(expand({name: True for name in group}))
    return list({identity(row): row for row in rows}.values())


def write_catalog(output_dir, recipes):
    # Validate and render everything before creating output. A manifest only
    # appears after all candidate files exist; interrupted output is inspectable.
    normalized = {identity(row): row for row in map(validate_recipe, recipes)}
    if not normalized:
        raise ValueError("catalog needs at least one recipe")
    sources = {key: render(row) for key, row in normalized.items()}
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    entries = []
    for key, recipe in normalized.items():
        relative = Path("policies") / key
        destination = output_dir / relative
        destination.mkdir(parents=True)
        source = sources[key]
        files = {}
        for name, text in source.items():
            data = text.encode()
            (destination / name).write_bytes(data)
            files[name] = hashlib.sha256(data).hexdigest()
        ablations = []
        for component in families(recipe):
            other = identity(recipe | {component: False})
            if other in normalized:
                ablations.append(other)
        static_controls = []
        if (
            recipe["pressure_chunks"]
            and recipe["small_chunk_tokens"] != recipe["large_chunk_tokens"]
        ):
            for size in (recipe["small_chunk_tokens"], recipe["large_chunk_tokens"]):
                other = identity(
                    recipe | {"small_chunk_tokens": size, "large_chunk_tokens": size}
                )
                if other in normalized:
                    static_controls.append(other)
        entries.append(
            {
                "id": key,
                "evaluator_candidate_id": identity(
                    {"kind": "candidate", "files": files}
                ),
                "recipe": recipe,
                "families": families(recipe),
                "files": files,
                "path": str(relative),
                "parents": [],
                "ablations": ablations,
                "static_controls": static_controls,
            }
        )
    by_id = {row["id"]: row for row in entries}
    for row in entries:
        for child in row["ablations"] + row["static_controls"]:
            by_id[child]["parents"].append(row["id"])
    catalog = {"schema_version": 1, "entries": entries}
    (output_dir / "catalog.json").write_bytes(json_bytes(catalog))
    return catalog


def validate_catalog(path):
    """Bind declared strategy families to exact generated source and controls."""
    path = Path(path)
    if path.is_symlink():
        raise ValueError("catalog must not be a symlink")
    catalog = json.loads(path.read_text())
    if not isinstance(catalog, dict) or catalog.get("schema_version") != 1:
        raise ValueError("unsupported scheduling catalog")
    entries = catalog.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("catalog needs entries")
    by_id = {}
    for row in entries:
        if not isinstance(row, dict):
            raise TypeError("invalid catalog entry")
        recipe = validate_recipe(row.get("recipe"))
        key = identity(recipe)
        if row.get("id") != key or key in by_id:
            raise ValueError("catalog recipe identity is invalid or duplicated")
        source = render(recipe)
        files = {
            name: hashlib.sha256(text.encode()).hexdigest()
            for name, text in source.items()
        }
        expected_id = identity({"kind": "candidate", "files": files})
        if (
            row.get("files") != files
            or row.get("evaluator_candidate_id") != expected_id
        ):
            raise ValueError("catalog generated source identity differs")
        if row.get("families") != families(recipe):
            raise ValueError("catalog families differ from executable recipe")
        raw = row.get("path")
        if not isinstance(raw, str):
            raise TypeError("catalog source path must be relative")
        relative = Path(raw)
        if relative.is_absolute() or ".." in relative.parts or not relative.parts:
            raise ValueError("catalog source path must stay within its directory")
        destination = path.parent
        for part in relative.parts:
            destination = destination / part
            if destination.is_symlink():
                raise ValueError("catalog source path must not contain symlinks")
        if not destination.is_dir():
            raise ValueError("catalog source directory is missing")
        actual = {}
        for item in destination.rglob("*"):
            if item.is_symlink():
                raise ValueError("catalog source must not contain symlinks")
            if item.is_file():
                actual[str(item.relative_to(destination))] = hashlib.sha256(
                    item.read_bytes()
                ).hexdigest()
        if actual != files:
            raise ValueError("catalog source files changed")
        by_id[key] = row
    expected_parents = {key: set() for key in by_id}
    for key, row in by_id.items():
        recipe = validate_recipe(row["recipe"])
        ablations = {
            identity(recipe | {name: False}) for name in families(recipe)
        } & by_id.keys()
        controls = set()
        if (
            recipe["pressure_chunks"]
            and recipe["small_chunk_tokens"] != recipe["large_chunk_tokens"]
        ):
            controls = {
                identity(
                    recipe | {"small_chunk_tokens": size, "large_chunk_tokens": size}
                )
                for size in (recipe["small_chunk_tokens"], recipe["large_chunk_tokens"])
            } & by_id.keys()
        for field, expected in (
            ("ablations", ablations),
            ("static_controls", controls),
        ):
            declared = row.get(field)
            if (
                not isinstance(declared, list)
                or any(not isinstance(item, str) for item in declared)
                or len(declared) != len(set(declared))
                or set(declared) != expected
            ):
                raise ValueError(f"catalog {field} differ from recipe changes")
            for other in expected:
                expected_parents[other].add(key)
    for key, row in by_id.items():
        declared = row.get("parents")
        if (
            not isinstance(declared, list)
            or any(not isinstance(item, str) for item in declared)
            or len(declared) != len(set(declared))
            or set(declared) != expected_parents[key]
        ):
            raise ValueError("catalog parents differ from component/control relations")
    return catalog


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path, help="new directory")
    parser.add_argument(
        "--recipe",
        type=Path,
        help="JSON object; otherwise generate the starter catalog",
    )
    parser.add_argument(
        "--expand", action="store_true", help="add components and controls for --recipe"
    )
    args = parser.parse_args(argv)
    try:
        if args.expand and args.recipe is None:
            raise ValueError(
                "--expand requires --recipe; default catalog already includes controls"
            )
        if args.recipe:
            recipe = validate_recipe(json.loads(args.recipe.read_text()))
            recipes = expand(recipe) if args.expand else [recipe]
        else:
            recipes = default_recipes()
        catalog = write_catalog(args.output_dir, recipes)
    except (OSError, ValueError, TypeError, OverflowError) as error:
        parser.error(str(error))
    print(
        json.dumps(
            {
                "catalog": str(args.output_dir / "catalog.json"),
                "candidates": len(catalog["entries"]),
            }
        )
    )


if __name__ == "__main__":
    main()
