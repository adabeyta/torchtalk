"""Map changed C++ funcs → impacted Python test runs.

Output mirrors PyTorch's `tools/testing/target_determination` TestRun shape
(file + included_classes). Class-level granularity dodges runtime-parametrized
test names from `instantiate_device_type_tests`.
"""

from __future__ import annotations

from typing import Any

from .cpp_call_graph import CppCallGraphExtractor

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


def _bindings_for(
    cpp_funcs: set[str],
    by_cpp_name: dict[str, list[dict]],
    native_functions: dict[str, dict] | None = None,
    native_implementations: dict[str, list[dict]] | None = None,
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
        # native_functions.yaml resolution: when no binding has cpp_name == bare,
        # the bare symbol may itself be the ATen op name (the implicit-dispatch
        # rule + structured/composite kernels). Use native_functions /
        # native_implementations as the source of truth.
        if not candidates:
            base = bare.rstrip("_")
            base = base[:-4] if base.endswith("_out") else base
            for key in (bare, base):
                if (native_implementations and key in native_implementations) or (
                    native_functions and key in native_functions
                ):
                    apis.add(key)
                    break
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
) -> dict[str, set[str]]:
    """Map test file → classes whose `test_*` methods reference any of `apis`."""
    by_file: dict[str, set[str]] = {}
    for api in apis:
        for variant in api_attr_variants(api):
            for hit in test_attr_index.get(variant, []):
                if hit["file"] not in test_files:
                    continue
                if hit.get("receiver_type") in _NON_TORCH_RECEIVERS:
                    continue
                if cls := hit.get("class"):
                    by_file.setdefault(hit["file"], set()).add(cls)
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
    native_functions: dict[str, dict] | None = None,
    native_implementations: dict[str, list[dict]] | None = None,
    depth: int = 3,
) -> dict[str, Any]:
    """Walk callers, derive Python APIs, return PyTorch-TestRun-shaped runs."""
    walked = _walk_callers(cpp_extractor, funcs, depth)
    bindings, apis = _bindings_for(
        walked, by_cpp_name, native_functions, native_implementations
    )

    # Bridge internal aten names to user-facing python ops via the decomp/refs
    # registry (e.g. convolution_overrideable → conv2d) so downstream lookups
    # find the test classes / OpInfo entries that actually exist.
    if decomp_alias_map:
        expanded: set[str] = set()
        for api in apis:
            expanded.update(decomp_alias_map.get(api, ()))
        apis |= expanded

    by_file = _tests_for_apis(apis, test_classes, test_files)

    # Symbol-mention catch generic-class tests (TestTorch::test_sizes) that
    # class-name matching can't reach. Merge into class-name results.
    if test_attr_index:
        for path, classes in _tests_mentioning_apis(
            apis, test_attr_index, test_files
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
