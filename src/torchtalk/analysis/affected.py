"""Map changed C++ funcs → impacted Python test runs.

Output mirrors PyTorch's `tools/testing/target_determination` TestRun shape
(file + included_classes). Class-level granularity dodges runtime-parametrized
test names from `instantiate_device_type_tests`.
"""

from __future__ import annotations

import re
from typing import Any

from .cpp_call_graph import CppCallGraphExtractor
from .patterns import is_vendor_path

_OVERLOAD_TAG_LITERALS = {"int", "default", "out", "self"}


def normalize_api(python_name: str) -> str:
    """Reduce a binding's python_name to its op identifier.

    Drops leading namespace, then drops a trailing overload tag (last
    segment containing any uppercase letter, or matching a small literal
    set). Sub-namespaces like `masked.sum` are preserved.

    `aten.size.int` -> `size`; `aten.fill_.Scalar` -> `fill_`;
    `aten.masked.sum` -> `masked.sum`; `aten.zero_` -> `zero_`.
    """
    parts = python_name.split(".")
    if len(parts) == 1:
        return parts[0]
    rest = parts[1:]
    if len(rest) >= 2:
        last = rest[-1]
        if last and (any(c.isupper() for c in last) or last in _OVERLOAD_TAG_LITERALS):
            rest = rest[:-1]
    return ".".join(rest)


def _class_matches_api(class_name: str, api: str) -> bool:
    """Match `Test<Api>` or `Test<Api><Word>` in PascalCase class names.

    PyTorch test classes are PascalCase (`TestCopy`), so word-boundary
    matching fails — use case transitions as boundaries. Strips trailing `_`
    on api so in-place ops share their non-mutating op's test class.
    """
    if not class_name.startswith("Test"):
        return False
    rest = class_name[4:]
    api_norm = api.rstrip("_").replace("_", "").replace(".", "").lower()
    if not rest or not api_norm:
        return False
    rest_lower = rest.lower()
    if rest_lower == api_norm:
        return True
    if rest_lower.startswith(api_norm):
        boundary = len(api_norm)
        if boundary < len(rest) and rest[boundary].isupper():
            return True
    return False


def _walk_callers(
    extractor: CppCallGraphExtractor, funcs: list[str], depth: int
) -> set[str]:
    visited: set[str] = set(funcs)
    current: set[str] = set(funcs)
    for level in range(depth):
        next_level: set[str] = set()
        for func in current:
            for item in extractor.get_callers(func, fuzzy=(level == 0)):
                caller = item["caller"]
                if caller not in visited:
                    visited.add(caller)
                    next_level.add(caller)
        if not next_level:
            break
        current = next_level
    return visited


_IMPL_SUFFIXES = (
    "_kernel_impl",
    "_cuda_kernel",
    "_kernel",
    "_forward",
    "_symint",
)


def _strip_impl_suffix(name: str) -> list[str]:
    """Strip a trailing impl-marker suffix (`_kernel`, `_forward`, `_symint`, ...).

    Returns a single-element list with the bare stem, or empty if no suffix
    matched. The caller must confirm the stem is in `native_functions` before
    treating it as an op name (keeps false positives out).
    """
    for suffix in _IMPL_SUFFIXES:
        if name.endswith(suffix):
            return [name[: -len(suffix)]]
    return []


_PLATFORM_TAGS = ("CUDA", "CPU", "MPS")
_PASCAL_BOUNDARY_RE = re.compile(r"(?<=[a-z0-9])([A-Z])")
_PASCAL_ACRONYM_RE = re.compile(r"([A-Z]+)(?=[A-Z][a-z])")


def _pascal_kernel_impl_candidates(name: str) -> list[str]:
    """PascalCase `<Op>[Platform]KernelImpl` → snake_case op candidates.

    `GeluCUDAKernelImpl` → `['gelu', 'native_gelu']`;
    `LayerNormBackwardKernelImpl` → `['layer_norm_backward',
    'native_layer_norm_backward']`. Acronym runs split correctly:
    `RMSNormBackwardKernelImpl` → `['rms_norm_backward', ...]`.

    The `native_` prefix variant covers ops whose schema name carries the
    `native_` prefix (`native_layer_norm_backward`) while the kernel symbol
    drops it. Caller must verify candidates exist in `native_functions`.
    """
    if not name.endswith("KernelImpl"):
        return []
    base = name[: -len("KernelImpl")]
    for tag in _PLATFORM_TAGS:
        if base.endswith(tag):
            base = base[: -len(tag)]
            break
    if not base or not base[0].isupper():
        return []
    snake = _PASCAL_ACRONYM_RE.sub(r"\1_", base)
    snake = _PASCAL_BOUNDARY_RE.sub(r"_\1", snake)
    snake = snake.lower()
    return [snake, f"native_{snake}"]


def _seed_file_op_cohort(
    funcs: list[str],
    cpp_extractor: CppCallGraphExtractor,
    ops_by_file: dict[str, set[str]],
    cohort_cap: int,
    symbol_to_file: dict[str, str] | None = None,
    dir_cap: int = 30,
) -> set[str]:
    """Bridge inner vendor-backend helpers to their parent op family.

    Three-step resolution for an input symbol:
      1. `cpp_extractor.function_locations` lookup (call-graph derived).
      2. If libclang missed the symbol (e.g. ConvShared.cpp's body is gated
         by `#if AT_CUDNN_ENABLED()` and the build has cuDNN off), fall back
         to `symbol_to_file` — the regex-derived index of vendor-dir helpers.
      3. If the resolved file's `ops_by_file` is empty (e.g. MHA.cpp holds
         only helpers, no registered op), aggregate ops from sibling files
         in the same vendor directory, capped at `dir_cap`.
    """
    locs = cpp_extractor.function_locations
    extra: set[str] = set()
    for func in funcs:
        bare = func.rsplit("::", 1)[-1]
        loc = (
            locs.get(func)
            or locs.get(f"at::native::{bare}")
            or locs.get(bare)
        )
        file = loc[0] if loc else (symbol_to_file or {}).get(bare)
        if not file:
            continue
        siblings = ops_by_file.get(file, set())
        cap = cohort_cap
        if not siblings and is_vendor_path(file):
            dir_prefix = file.rsplit("/", 1)[0] + "/"
            agg: set[str] = set()
            for fp, ops in ops_by_file.items():
                if fp.startswith(dir_prefix):
                    agg |= ops
            siblings = agg
            cap = dir_cap
        if not siblings or len(siblings) > cap:
            continue
        extra |= siblings
    return extra


def _file_cohort_apis(
    matched: list[dict],
    bindings_by_file: dict[str, list[dict]],
    cohort_cap: int,
) -> set[str]:
    """Return sibling-binding APIs from each matched binding's source file.

    A dev editing a kernel file typically touches multiple kernels in that
    file; their tests should run together. Mirrors PyTorch's own file-level
    `Profiling`/`Filepath` heuristics. Files exceeding `cohort_cap` (registry
    files like `NamedRegistrations.cpp`) are skipped to avoid flooding.
    """
    cohort_files = {b.get("file_path") for b in matched if b.get("file_path")}
    extra: set[str] = set()
    for fp in cohort_files:
        siblings = bindings_by_file.get(fp, [])
        if len(siblings) > cohort_cap:
            continue
        for sib in siblings:
            if py_name := sib.get("python_name"):
                extra.add(normalize_api(py_name))
    return extra


def _bindings_for(
    cpp_funcs: set[str],
    by_cpp_name: dict[str, list[dict]],
    native_functions: dict[str, dict] | None = None,
    native_implementations: dict[str, list[dict]] | None = None,
    kernel_impl_to_op: dict[str, str] | None = None,
    dispatch_to_op: dict[str, str] | None = None,
    bindings_by_file: dict[str, list[dict]] | None = None,
    cohort_cap: int = 15,
) -> tuple[list[dict], set[str]]:
    matched: list[dict] = []
    apis: set[str] = set()
    for fn in cpp_funcs:
        # Walked names are qualified (`at::native::add`); binding keys are bare
        # (`add`). Fall back to last `::` segment.
        bare = fn.rsplit("::", 1)[-1]
        candidates = by_cpp_name.get(fn) or by_cpp_name.get(bare, [])
        for binding in candidates:
            matched.append(binding)
            if py_name := binding.get("python_name"):
                apis.add(normalize_api(py_name))
        if candidates:
            continue
        # Fallback A: bare symbol is itself an ATen op name (implicit-dispatch
        # rule + structured/composite kernels). Also try a kernel-suffix-
        # stripped candidate so `binary_cross_entropy_kernel` resolves to
        # `binary_cross_entropy`.
        base = bare.rstrip("_")
        base = base[:-4] if base.endswith("_out") else base
        keys = [
            bare,
            base,
            *_strip_impl_suffix(bare),
            *_pascal_kernel_impl_candidates(bare),
        ]
        for key in keys:
            if (native_implementations and key in native_implementations) or (
                native_functions and key in native_functions
            ):
                apis.add(key)
                break
        else:
            # Fallback B: bare symbol is a CPU/CUDA kernel impl registered via
            # REGISTER_DISPATCH(stub, &kernel) — resolve via stub-to-op map.
            # Fallback C: vendor-backend symbol referenced from a parent op's
            # `dispatch:` table (e.g. `cudnn_convolution_forward` →
            # `cudnn_convolution`).
            if (kernel_impl_to_op and (op := kernel_impl_to_op.get(bare))) or (
                dispatch_to_op and (op := dispatch_to_op.get(bare))
            ):
                apis.add(op)
    if bindings_by_file:
        apis |= _file_cohort_apis(matched, bindings_by_file, cohort_cap)
    return matched, apis


def _tests_for_apis(
    apis: set[str],
    test_classes: dict[str, list[dict]],
    test_files: dict[str, dict],
) -> dict[str, set[str]]:
    by_file: dict[str, set[str]] = {}
    for api in apis:
        for cls_name, locations in test_classes.items():
            if not _class_matches_api(cls_name, api):
                continue
            for loc in locations:
                # Skip non-test helpers (e.g. NeuralNetwork under test/) and
                # files outside the indexed test tree.
                if not loc.get("is_test_class"):
                    continue
                file_path = loc["file"]
                if file_path not in test_files:
                    continue
                by_file.setdefault(file_path, set()).add(cls_name)
    return by_file


def api_attr_variants(api: str) -> set[str]:
    """API-name forms a test source might reference as an attribute access."""
    base = api.rstrip("_")
    leaf = base.rsplit(".", 1)[-1] if "." in base else base
    variants = {api, base, leaf, leaf + "_"}
    return {v for v in variants if v}


# Receivers known to NOT be torch types — drop their hits (`dict.copy()`,
# `list.copy()`, etc.). Unknown receivers (None) pass through as conservative.
_NON_TORCH_RECEIVERS = {"dict", "list", "set", "tuple", "str", "number", "bool"}


def _api_to_source_paths(api: str) -> list[str]:
    """Best-effort mapping of API qualname to candidate Python source paths."""
    paths: list[str] = []
    parts = api.split(".") if "." in api else None
    # ATen schemas often use underscore-namespacing (`linalg_cross`); profiling
    # keys are dot-namespaced (`torch/linalg.py`). Treat the first underscore
    # as a namespace separator when no dot is present.
    if parts is None and "_" in api:
        first_us = api.find("_")
        if first_us > 0:
            parts = [api[:first_us], api[first_us + 1 :]]
    if not parts or len(parts) < 2 or not parts[0] or not parts[-1]:
        return []
    prefix = "torch/" + "/".join(parts[:-1])
    paths.append(prefix + ".py")
    paths.append(prefix + "/__init__.py")
    return paths


def _tests_via_profiling(
    apis: set[str],
    python_profiling: dict[str, dict[str, float]],
    test_files: dict[str, dict],
) -> dict[str, set[str]]:
    """Look up tests via PyTorch's coverage-based file→test mapping."""
    by_file: dict[str, set[str]] = {}
    for api in apis:
        for src_file in _api_to_source_paths(api):
            for test_name in python_profiling.get(src_file, ()):
                test_path = f"test/{test_name}.py"
                if test_path in test_files:
                    by_file.setdefault(test_path, set())
    return by_file


def _tests_mentioning_apis(
    apis: set[str],
    test_attr_index: dict[str, list[dict]],
    test_files: dict[str, dict],
    per_api_cap: int = 50,
) -> dict[str, set[str]]:
    """Map test file → classes whose `test_*` methods reference any of `apis`.

    Drops an API's mention-only contribution entirely when it would exceed
    `per_api_cap`. Generic names like `add` mention-match thousands of unrelated
    tests; over-cap APIs are too low-specificity to be a reliable signal. The
    class-name match path (`_tests_for_apis`) is unaffected.
    """
    by_file: dict[str, set[str]] = {}
    for api in apis:
        api_hits: list[tuple[str, str]] = []
        for variant in api_attr_variants(api):
            for hit in test_attr_index.get(variant, []):
                if hit["file"] not in test_files:
                    continue
                if hit.get("receiver_type") in _NON_TORCH_RECEIVERS:
                    continue
                if cls := hit.get("class"):
                    api_hits.append((hit["file"], cls))
        if len(api_hits) > per_api_cap:
            continue
        for path, cls in api_hits:
            by_file.setdefault(path, set()).add(cls)
    return by_file


def affected_tests(
    funcs: list[str],
    cpp_extractor: CppCallGraphExtractor,
    by_cpp_name: dict[str, list[dict]],
    test_classes: dict[str, list[dict]],
    test_files: dict[str, dict],
    opinfo_registry: dict[str, dict] | None = None,
    opinfo_alias_map: dict[str, list[dict]] | None = None,
    opinfo_test_files: set[str] | None = None,
    test_attr_index: dict[str, list[dict]] | None = None,
    python_profiling: dict[str, dict[str, float]] | None = None,
    decomp_alias_map: dict[str, list[str]] | None = None,
    backward_to_forward: dict[str, list[str]] | None = None,
    native_functions: dict[str, dict] | None = None,
    native_implementations: dict[str, list[dict]] | None = None,
    kernel_impl_to_op: dict[str, str] | None = None,
    dispatch_to_op: dict[str, str] | None = None,
    bindings_by_file: dict[str, list[dict]] | None = None,
    ops_by_file: dict[str, set[str]] | None = None,
    symbol_to_file: dict[str, str] | None = None,
    depth: int = 3,
    mention_cap: int = 50,
    cohort_cap: int = 15,
    dir_cap: int = 30,
) -> dict[str, Any]:
    """Walk callers, derive Python APIs, return PyTorch-TestRun-shaped runs."""
    walked = _walk_callers(cpp_extractor, funcs, depth)
    bindings, apis = _bindings_for(
        walked,
        by_cpp_name,
        native_functions,
        native_implementations,
        kernel_impl_to_op,
        dispatch_to_op,
        bindings_by_file,
        cohort_cap,
    )

    # Vendor-helper bridge: when the input symbol's source file declares native
    # ops (e.g. cudnn helpers in `ConvShared.cpp` alongside `cudnn_convolution`),
    # union those ops in. Catches cases where the call graph misses upward edges.
    if ops_by_file:
        apis |= _seed_file_op_cohort(
            funcs,
            cpp_extractor,
            ops_by_file,
            cohort_cap,
            symbol_to_file=symbol_to_file,
            dir_cap=dir_cap,
        )

    # Backward kernels are tested via gradcheck on the forward op's TestCase,
    # so any `*_backward` API expands to its forward op name(s) before the
    # downstream test-class / OpInfo lookups run.
    if backward_to_forward:
        expanded: set[str] = set()
        for api in apis:
            expanded.update(backward_to_forward.get(api, ()))
        apis |= expanded

    # Bridge internal aten names to user-facing python ops via the decomp/refs
    # registry (e.g. convolution_overrideable → conv2d) so downstream lookups
    # find the test classes / OpInfo entries that actually exist.
    if decomp_alias_map:
        expanded = set()
        for api in apis:
            expanded.update(decomp_alias_map.get(api, ()))
        apis |= expanded

    by_file = _tests_for_apis(apis, test_classes, test_files)

    # Symbol-mention catch generic-class tests (TestTorch::test_sizes) that
    # class-name matching can't reach. Merge into class-name results.
    if test_attr_index:
        for path, classes in _tests_mentioning_apis(
            apis, test_attr_index, test_files, per_api_cap=mention_cap
        ).items():
            by_file.setdefault(path, set()).update(classes)

    # An API matches OpInfo directly OR via an `aliases=`/`aten_name=` link.
    opinfo_keys: set[str] = set(opinfo_registry or {}) | set(opinfo_alias_map or {})
    if opinfo_test_files and apis & opinfo_keys:
        for path in opinfo_test_files:
            by_file.setdefault(path, set())

    # PyTorch CI's coverage-based map adds whole-file runs for tests that
    # touched the API's Python source file at runtime.
    if python_profiling:
        for path in _tests_via_profiling(apis, python_profiling, test_files):
            by_file.setdefault(path, set())

    # Catch-all: APIs resolved but no test file matched anywhere. Internal
    # ops (`convolution_overrideable`, `_safe_softmax`) often have no dedicated
    # test class and aren't in OpInfo, but they're exercised transitively from
    # the catch-all `@ops(op_db)` test classes in test_ops.py / test_meta.py.
    # Cost is low (parametrized tests skip non-matching ops).
    if apis and not by_file and opinfo_test_files:
        for path in opinfo_test_files:
            by_file.setdefault(path, set())

    return {
        "input_functions": list(funcs),
        "callers_walked": len(walked),
        "bindings_matched": [
            {
                "python_name": b.get("python_name"),
                "cpp_name": b.get("cpp_name"),
                "dispatch_key": b.get("dispatch_key"),
            }
            for b in bindings
        ],
        "python_apis": sorted(apis),
        "test_runs": [
            {"file": f, "included_classes": sorted(classes)}
            for f, classes in sorted(by_file.items())
        ],
    }


def symbols_in_file(path: str, cpp_extractor: CppCallGraphExtractor) -> dict[str, Any]:
    """Return C++ functions defined in the given file (suffix-matched)."""
    matches = [
        {"function": func, "file": loc[0], "line": loc[1]}
        for func, loc in cpp_extractor.function_locations.items()
        if loc and loc[0].endswith(path)
    ]
    matches.sort(key=lambda m: (m["file"], m["line"] or 0))
    return {"path": path, "functions": matches}
