"""Freeze bounded mechanism candidates; static guards are not a semantic proof."""

import ast
import difflib
import hashlib
import json
import os
import shutil
import stat
from pathlib import Path, PurePosixPath

PROFILE = "qwen_moe"
CONTRACT_VERSION = "qwen-cross-module-v1"
POLICY_ROOT = "crates/simthesizer-core/src/policies"
POLICY_CANDIDATE_ROOT = POLICY_ROOT + "/candidate"
QWEN = "scripts/rooflang_bridge/qwen3.py"
IMPLEMENTATION_FILES = frozenset(
    {
        "scripts/rooflang_worker.py",
        "scripts/rooflang_model.py",
    }
)
PYTHON_IMPLEMENTATION_ROOTS = (
    "scripts/rooflang_bridge",
    "scripts/rooflang",
)
# Fixed model definitions, not a list of permitted optimization mechanisms.
IMMUTABLE_FACT_FILES = (
    frozenset(
        "crates/simthesizer-core/src/models/" + name
        for name in (
            "qwen3_235b_a22b.rs",
            "qwen3_moe.rs",
            "llama3_1.rs",
            "nemotron_h.rs",
        )
    )
    | frozenset(
        "crates/simthesizer-cli/src/" + name
        for name in ("main.rs", "write_jsonl.rs", "write_trace.rs", "progress.rs")
    )
    | frozenset(
        "scripts/rooflang/programs/presets/" + name
        for name in ("h200.py", "b300.py", "rtx6000d.py")
    )
)
IGNORED_DIRECTORIES = frozenset({"__pycache__", ".pytest_cache", ".ruff_cache", ".git"})
PROTECTED_GLOBALS = frozenset(
    {
        "CONFIG_PATH",
        "MODEL",
        "H",
        "Q",
        "K",
        "D",
        "E",
        "TOP",
        "M",
        "V",
        "L",
        "validate_parallel",
        "describe",
        "validate_model",
        "token_experts",
    }
)
PRICING_METHODS = frozenset({"_compute_cost", "_copy_cost"})
AST_FACTS = {
    "scripts/rooflang/runtime/simulator.py": (
        "_fabric_key",
        "_direction_bw",
        "Simulator._collective_fabric_info",
        "Simulator._base_times",
        "Simulator.compute_time_us",
        "Simulator._infer_dtype",
    ),
    "scripts/rooflang/language/tensor.py": (
        "Tensor.n_elements",
        "Tensor.size_bytes",
    ),
    "scripts/rooflang/language/utils.py": ("dtype_bytes", "gemm_scale_bytes"),
    "scripts/rooflang_bridge/llama.py": (
        "H",
        "FF",
        "QH",
        "KVH",
        "D",
        "V",
        "L",
        "Work.flops",
        "layer_plan",
        "describe",
        "validate_model",
        "batch_shape",
    ),
}
COST_KERNEL_ROOT = "scripts/rooflang/language/kernels"
COST_PROPERTIES = frozenset(
    {
        "flops",
        "flops_by_dtype",
        "transferred_bytes",
        "input_read_fraction",
        "input_tensor_bytes",
        "input_bytes",
        "weight_bytes",
        "loaded_weight_bytes",
        "output_bytes",
        "attention_flops",
        "indexer_flops",
        "kv_transform_flops",
        "indexer_score_flops",
        "indexer_reduce_flops",
        "_default_selected_pairs",
        "_default_indexer_pairs",
    }
)
DOMAINS = frozenset(
    {
        "engine",
        "resources",
        "kv",
        "graph",
        "policy",
        "interfaces",
        "observability",
    }
)
DOMAIN_CHECKS = {
    "engine": {"work_conservation", "causal_dependencies", "resource_bounds"},
    "resources": {"resource_bounds", "work_conservation"},
    "kv": {"kv_lifecycle", "resource_bounds", "causal_dependencies"},
    "graph": {"work_conservation", "causal_dependencies"},
    "policy": {"policy_contract"},
    "interfaces": {"interface_compatibility"},
    "observability": {"observation_neutrality"},
}


def json_bytes(value):
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode()


def identity(value):
    return hashlib.sha256(json_bytes(value)).hexdigest()


def _relative(value):
    text = str(value)
    path = PurePosixPath(text)
    if (
        not text
        or path.is_absolute()
        or str(path) != text
        or any(part in (".", "..") or part.startswith(".") for part in path.parts)
        or any(character in text for character in ("\\", "\x00", "\n", "\r"))
    ):
        raise ValueError(f"invalid candidate path: {text!r}")
    return path


def _under(path, root):
    return path == root or path.startswith(root + "/")


def editable(relative):
    """Implementation roots are open; facts, tooling and existing tests are not."""
    path = _relative(relative)
    name = str(path)
    if name in IMMUTABLE_FACT_FILES or path.name in {
        "sitecustomize.py",
        "usercustomize.py",
        "build.rs",
    }:
        return False
    if _test_path(path) and path.name.startswith("__qwen_frozen_"):
        return False
    if name in IMPLEMENTATION_FILES:
        return True
    if _rust_tree(path, "src"):
        return path.suffix == ".rs"
    if any(_under(name, root) for root in PYTHON_IMPLEMENTATION_ROOTS):
        return path.suffix == ".py"
    return _test_path(path) and path.suffix in {".py", ".rs"}


def _rust_tree(path, directory):
    return (
        len(path.parts) >= 4
        and path.parts[0] == "crates"
        and path.parts[2] == directory
    )


def _test_path(path):
    return _under(str(path), "tests") or _rust_tree(path, "tests")


def _allowed_change(name, reference):
    return editable(name) and not (
        _test_path(PurePosixPath(name)) and name in reference
    )


def impact(changed):
    """Conservative review routing from paths, never an optimization allowlist."""
    files = {name: sorted(_domains(name)) for name in sorted(changed)}
    domains = set().union(*map(set, files.values())) if files else set()
    checks = {"baseline_compatibility"} if files else set()
    for domain in domains:
        checks.update(DOMAIN_CHECKS[domain])
    return {
        "domains": sorted(domains),
        "files": files,
        "core_changed": bool(domains & {"engine", "resources", "interfaces"}),
        "cross_module": len(files) > 1,
        "required_checks": sorted(checks),
    }


def _domains(name):
    path = PurePosixPath(name)
    stem = path.stem
    if _test_path(path):
        return set()
    if _under(name, POLICY_CANDIDATE_ROOT):
        return {"policy"}
    if (
        _under(name, POLICY_ROOT)
        or "schedulers" in path.parts
        or "routers" in path.parts
    ):
        domains = {"policy"}
        if path.name in {
            "mod.rs",
            "policy.rs",
            "environment.rs",
            "scheduler.rs",
            "router.rs",
        }:
            domains.add("interfaces")
        if "prefix_cache" in stem:
            domains.add("kv")
        return domains
    if stem == "engine":
        return {"engine"}
    if (
        stem == "resources"
        or "network_sims" in path.parts
        or "compute_sims" in path.parts
    ):
        return {"resources"}
    if stem == "kv" or "kv" in path.parts:
        return {"kv"}
    if "systems" in path.parts:
        return {"engine", "kv", "interfaces"}
    if "model_runners" in path.parts or "models" in path.parts:
        return {"graph", "interfaces", "resources"}
    if stem in {"qwen3", "llama"}:
        return {"graph"}
    if stem in {
        "rooflang_worker",
        "rooflang_model",
        "bindings",
        "configuration",
        "execution",
    }:
        return {"graph", "interfaces", "resources"}
    if stem == "simulator":
        return {"engine", "resources", "graph"}
    if stem in {"hardware", "topology", "component"}:
        return set(DOMAINS)
    if stem in {"graph", "placement", "tensor"} or "kernels" in path.parts:
        return {"graph", "resources"}
    if stem in {
        "wall_profile",
        "wall_report",
        "trace_export",
        "module_flow",
        "observations",
        "pd_report",
        "shared_report",
        "write_trace",
        "write_jsonl",
        "progress",
    }:
        return {"observability"}
    # A registry, public type or unfamiliar module may change any consuming path.
    return set(DOMAINS)


def _open_directory(path):
    """Open each component without following symlinks, including root ancestors."""
    path = Path(os.path.abspath(path))
    fd = os.open(path.anchor, os.O_RDONLY | os.O_DIRECTORY)
    try:
        for part in path.parts[1:]:
            previous = fd
            fd = os.open(
                part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=previous
            )
            os.close(previous)
        return fd
    except OSError as error:
        os.close(fd)
        raise ValueError(
            f"candidate directory must not contain symlinks: {path}"
        ) from error


def _read_tree(root, *, max_files, max_bytes, max_file_bytes):
    content, modes = {}, {}
    ignored = []
    total = 0
    entries = 0

    def walk(fd, prefix, skip=False):
        nonlocal total, entries
        names = []
        with os.scandir(fd) as listing:
            for entry in listing:
                names.append(entry.name)
                if len(names) + entries > max_files * 2:
                    raise ValueError("candidate source exceeds directory/file limits")
        for name in sorted(names):
            entries += 1
            if entries > max_files * 2:
                raise ValueError("candidate source exceeds directory/file limits")
            relative = f"{prefix}/{name}" if prefix else name
            # Frozen source can contain .python-version; only changed paths use
            # the stricter editable-name rules.
            info = os.stat(name, dir_fd=fd, follow_symlinks=False)
            if stat.S_ISLNK(info.st_mode):
                raise ValueError(f"candidate symlink: {relative}")
            if stat.S_ISDIR(info.st_mode):
                child = os.open(
                    name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd
                )
                try:
                    walk(child, relative, skip or name in IGNORED_DIRECTORIES)
                finally:
                    os.close(child)
                continue
            if not stat.S_ISREG(info.st_mode):
                raise ValueError(f"candidate special file: {relative}")
            if info.st_nlink != 1:
                raise ValueError(f"candidate hardlink: {relative}")
            if info.st_mode & (stat.S_ISUID | stat.S_ISGID | stat.S_ISVTX):
                raise ValueError(f"candidate special permission bits: {relative}")
            if skip:
                if name.endswith((".py", ".rs")):
                    raise ValueError(f"source code inside ignored metadata: {relative}")
                ignored.append(relative)
                continue
            if (
                len(content) >= max_files
                or info.st_size > max_file_bytes
                or total + info.st_size > max_bytes
            ):
                raise ValueError("candidate source exceeds file/byte limits")
            file_fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=fd)
            with os.fdopen(file_fd, "rb") as stream:
                before = os.fstat(stream.fileno())
                if (
                    not stat.S_ISREG(before.st_mode)
                    or before.st_nlink != 1
                    or (before.st_dev, before.st_ino) != (info.st_dev, info.st_ino)
                ):
                    raise ValueError(f"candidate file changed during read: {relative}")
                data = stream.read(max_file_bytes + 1)
                after = os.fstat(stream.fileno())
            if (
                len(data) != before.st_size
                or len(data) > max_file_bytes
                or before.st_mtime_ns != after.st_mtime_ns
                or before.st_ctime_ns != after.st_ctime_ns
                or after.st_nlink != 1
            ):
                raise ValueError(f"candidate file changed during read: {relative}")
            total += len(data)
            if total > max_bytes:
                raise ValueError("candidate source exceeds byte limit")
            content[relative], modes[relative] = (
                data,
                stat.S_IMODE(info.st_mode) & 0o777,
            )

    fd = _open_directory(root)
    try:
        walk(fd, "")
    finally:
        os.close(fd)
    return content, modes, ignored


def _fingerprint(node):
    return ast.dump(node, annotate_fields=True, include_attributes=False)


def _bindings(statement):
    """Module-scope bindings, including conditional redefinitions."""
    if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        yield statement.name, statement
        return
    if isinstance(statement, (ast.Import, ast.ImportFrom)):
        for alias in statement.names:
            yield alias.asname or alias.name.split(".")[0], statement
        return
    if isinstance(statement, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
        targets = (
            statement.targets
            if isinstance(statement, ast.Assign)
            else [statement.target]
        )
        for target in targets:
            for node in ast.walk(target):
                if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                    yield node.id, statement
    for child in ast.iter_child_nodes(statement):
        if isinstance(child, ast.stmt):
            yield from _bindings(child)


def _qwen_contract(data, *, import_names=None):
    try:
        tree = ast.parse(data.decode("utf-8"), filename=QWEN)
    except (SyntaxError, UnicodeError) as error:
        raise ValueError("candidate Qwen source must be valid UTF-8 Python") from error
    bindings = {}
    for statement in tree.body:
        for name, node in _bindings(statement):
            bindings.setdefault(name, []).append(node)
    if import_names is None:
        import_names = {
            name
            for name, nodes in bindings.items()
            if any(isinstance(node, (ast.Import, ast.ImportFrom)) for node in nodes)
        }
    protected = {}
    for name in sorted(PROTECTED_GLOBALS | import_names):
        nodes = bindings.get(name, [])
        if len(nodes) != 1:
            raise ValueError(f"protected Qwen binding missing or shadowed: {name}")
        protected[name] = _fingerprint(nodes[0])
    classes = bindings.get("QwenTemplate", [])
    if len(classes) != 1 or not isinstance(classes[0], ast.ClassDef):
        raise ValueError("protected QwenTemplate binding missing or shadowed")
    template = classes[0]
    protected["QwenTemplate.header"] = (
        [_fingerprint(node) for node in template.bases],
        [_fingerprint(node) for node in template.keywords],
        [_fingerprint(node) for node in template.decorator_list],
    )
    methods = {}
    for statement in template.body:
        for name, node in _bindings(statement):
            methods.setdefault(name, []).append(node)
    for name in sorted(PRICING_METHODS):
        nodes = methods.get(name, [])
        if len(nodes) != 1 or not isinstance(nodes[0], ast.FunctionDef):
            raise ValueError(
                f"protected Qwen pricing method missing or shadowed: {name}"
            )
        protected[f"QwenTemplate.{name}"] = _fingerprint(nodes[0])
    return protected, import_names


def _parse_python(data, name):
    try:
        return ast.parse(data.decode("utf-8"), filename=name)
    except (SyntaxError, UnicodeError) as error:
        raise ValueError(
            f"candidate source must be valid UTF-8 Python: {name}"
        ) from error


def _bound_nodes(body):
    bindings = {}
    for statement in body:
        for name, node in _bindings(statement):
            bindings.setdefault(name, []).append(node)
    return bindings


def _fact_contract(tree, selectors, import_names):
    """Pin existing facts while allowing new declarations and runtime methods."""
    bindings = _bound_nodes(tree.body)
    result = {}
    for selector in sorted(set(selectors) | set(import_names)):
        namespace = bindings
        prefix = []
        for index, part in enumerate(selector.split(".")):
            nodes = namespace.get(part, [])
            if len(nodes) != 1:
                raise ValueError(
                    f"protected pricing/model binding missing or shadowed: {selector}"
                )
            node = nodes[0]
            prefix.append(part)
            if index == selector.count("."):
                result[selector] = _fingerprint(node)
            elif isinstance(node, ast.ClassDef):
                result[".".join(prefix) + ".header"] = (
                    [_fingerprint(value) for value in node.bases],
                    [_fingerprint(value) for value in node.keywords],
                    [_fingerprint(value) for value in node.decorator_list],
                )
                namespace = _bound_nodes(node.body)
            else:
                raise ValueError(f"protected pricing/model class changed: {selector}")
    return result


def _protected_facts(reference, current):
    protected, imports = _qwen_contract(reference[QWEN])
    updated, _ = _qwen_contract(current[QWEN], import_names=imports)
    if updated != protected:
        fields = sorted(
            name for name in protected if protected[name] != updated.get(name)
        )
        raise ValueError("protected Qwen AST changed: " + ", ".join(fields))
    contracts = {QWEN: protected}
    for name in sorted(reference):
        selectors = set(AST_FACTS.get(name, ()))
        kernel = _under(name, COST_KERNEL_ROOT) and name.endswith(".py")
        if not selectors and not kernel:
            continue
        tree = _parse_python(reference[name], name)
        if kernel:
            for statement in tree.body:
                if isinstance(statement, ast.ClassDef):
                    for method in statement.body:
                        if isinstance(
                            method, (ast.FunctionDef, ast.AsyncFunctionDef)
                        ) and (
                            method.name in COST_PROPERTIES
                            or method.name.endswith("_flops")
                            or any(
                                isinstance(decorator, ast.Name)
                                and decorator.id == "property"
                                for decorator in method.decorator_list
                            )
                        ):
                            selectors.add(f"{statement.name}.{method.name}")
        if not selectors:
            continue
        if name not in current:
            raise ValueError(f"protected pricing/model source deleted: {name}")
        imports = {
            name
            for name, nodes in _bound_nodes(tree.body).items()
            if any(isinstance(node, (ast.Import, ast.ImportFrom)) for node in nodes)
        }
        before = _fact_contract(tree, selectors, imports)
        after = _fact_contract(_parse_python(current[name], name), selectors, imports)
        if before != after:
            fields = sorted(key for key in before if before[key] != after.get(key))
            raise ValueError(
                f"protected pricing/model AST changed in {name}: " + ", ".join(fields)
            )
        contracts[name] = before
    return contracts


def _write_file(root_fd, relative, data, mode=0o644):
    parts = PurePosixPath(relative).parts
    fd = os.dup(root_fd)
    try:
        for part in parts[:-1]:
            try:
                os.mkdir(part, 0o755, dir_fd=fd)
            except FileExistsError:
                pass
            next_fd = os.open(
                part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd
            )
            os.close(fd)
            fd = next_fd
        output = os.open(
            parts[-1],
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            mode,
            dir_fd=fd,
        )
        with os.fdopen(output, "wb") as stream:
            stream.write(data)
    finally:
        os.close(fd)


def snapshot(
    worktree,
    frozen_source,
    destination,
    *,
    expected_source_sha256=None,
    max_files=10000,
    max_bytes=128 * 1024 * 1024,
    max_file_bytes=8 * 1024 * 1024,
):
    """Write source/, changes.diff and candidate.json into a new destination.

    expected_source_sha256 accepts the full frozen inventory or its identity().
    Existing tests, pricing/model facts and tooling remain immutable. AST checks
    fix declarations, but cannot prove that arbitrary implementation changes use
    those declarations or preserve work, resource bounds and causal visibility.
    """
    limits = {
        "max_files": max_files,
        "max_bytes": max_bytes,
        "max_file_bytes": max_file_bytes,
    }
    if any(type(value) is not int or value <= 0 for value in limits.values()):
        raise ValueError("candidate limits must be positive integers")
    roots = [Path(os.path.abspath(path)) for path in (worktree, frozen_source)]
    destination = Path(os.path.abspath(destination))
    if any(
        destination == root
        or destination.is_relative_to(root)
        or root.is_relative_to(destination)
        for root in roots
    ):
        raise ValueError("candidate destination must be separate from input trees")
    reference, reference_modes, _ = _read_tree(frozen_source, **limits)
    current, current_modes, ignored = _read_tree(worktree, **limits)
    reference_files = {
        name: hashlib.sha256(data).hexdigest() for name, data in reference.items()
    }
    files = {name: hashlib.sha256(data).hexdigest() for name, data in current.items()}
    reference_id = identity(reference_files)
    if expected_source_sha256 is not None and expected_source_sha256 not in (
        reference_id,
        reference_files,
    ):
        raise ValueError("frozen source identity differs from manifest")
    changed = sorted(
        name
        for name in reference_files.keys() | files.keys()
        if reference_files.get(name) != files.get(name)
    )
    for name in changed:
        if not _allowed_change(name, reference_files):
            raise ValueError(f"protected candidate source changed: {name}")
    for name in reference.keys() & current.keys():
        if reference_modes[name] != current_modes[name]:
            raise ValueError(f"candidate file mode changed: {name}")
    if QWEN not in reference or QWEN not in current:
        raise ValueError("candidate requires the frozen Qwen graph module")
    protected = _protected_facts(reference, current)
    changes, diff = [], []
    for name in changed:
        try:
            before = reference.get(name, b"").decode("utf-8").splitlines(keepends=True)
            after = current.get(name, b"").decode("utf-8").splitlines(keepends=True)
        except UnicodeError as error:
            raise ValueError(
                f"changed candidate source must be UTF-8: {name}"
            ) from error
        diff.extend(
            difflib.unified_diff(
                before,
                after,
                fromfile=f"a/{name}" if name in reference else "/dev/null",
                tofile=f"b/{name}" if name in current else "/dev/null",
            )
        )
        changes.append(
            {
                "path": name,
                "status": "added"
                if name not in reference
                else "deleted"
                if name not in current
                else "modified",
                "before_sha256": reference_files.get(name),
                "after_sha256": files.get(name),
            }
        )
    policy_files = {
        name: value
        for name, value in files.items()
        if _under(name, POLICY_CANDIDATE_ROOT)
    }
    mechanism_files = {
        name: value
        for name, value in files.items()
        if not _under(name, POLICY_CANDIDATE_ROOT)
    }
    result = {
        "schema_version": 2,
        "contract_version": CONTRACT_VERSION,
        "profile": PROFILE,
        "kind": "reference"
        if not changed
        else "policy"
        if all(_under(name, POLICY_CANDIDATE_ROOT) for name in changed)
        else "mechanism",
        "reference_source_sha256": reference_id,
        "source_sha256": identity(files),
        "mechanism_sha256": identity(mechanism_files),
        "policy_sha256": identity(policy_files),
        # The built-in reference implementation shares this mechanism interface.
        # This flags a file change; it does not infer unchanged Rust semantics.
        "reference_policy_changed": POLICY_ROOT + "/mod.rs" in changed,
        "changed_files": changed,
        "changes": changes,
        "files": files,
        "protected_qwen_ast_sha256": identity(protected[QWEN]),
        "protected_facts_sha256": identity(protected),
        "protected_fact_files": sorted(protected),
        "impact": impact(changed),
        "validation_status": "static_scope_only",
        "ignored_files": ignored,
    }
    parent_fd = _open_directory(destination.parent)
    output_fd = None
    created = False
    try:
        os.mkdir(destination.name, 0o700, dir_fd=parent_fd)
        created = True
        output_fd = os.open(
            destination.name,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
            dir_fd=parent_fd,
        )
        for name, data in current.items():
            _write_file(output_fd, "source/" + name, data, current_modes[name])
        _write_file(output_fd, "changes.diff", "".join(diff).encode())
        _write_file(output_fd, "candidate.json", json_bytes(result))
    except BaseException:
        if created:
            shutil.rmtree(destination.name, dir_fd=parent_fd)
        raise
    finally:
        if output_fd is not None:
            os.close(output_fd)
        os.close(parent_fd)
    return result
