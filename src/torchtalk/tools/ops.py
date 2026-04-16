"""Operator trace and search tool implementations."""

from __future__ import annotations

from typing import Literal

from ..analysis.helpers import safe_sort_key, truncate
from ..formatting import create_formatter, relative_path
from ..indexer import _ensure_loaded, _fuzzy_find, _state


def _rel_path(path: str) -> str:
    return relative_path(path, _state.pytorch_source)


def _get_native_func(name: str) -> dict | None:
    """Look up a native function by name with fallback matching."""
    if name in _state.native_functions:
        return _state.native_functions[name]
    name_lower = name.lower()
    for k, v in _state.native_functions.items():
        if name_lower == k.lower() or name_lower == v.get("base_name", "").lower():
            return v
    matches = [
        (k, v) for k, v in _state.native_functions.items() if name_lower in k.lower()
    ]
    if matches:
        matches.sort(key=lambda x: len(x[0]))
        return matches[0][1]
    return None


def _similar_functions(name: str, limit: int = 10) -> list[str]:
    """Find similar function names for suggestions."""
    from ..analysis.helpers import levenshtein_distance

    name_lower = name.lower()
    results = []
    for key in _state.native_functions:
        if name_lower in key.lower():
            results.append((0, key))
    for key in _state.native_functions:
        if len(key) < 50 and abs(len(key) - len(name)) <= 5:
            dist = levenshtein_distance(name_lower, key.lower())
            if 0 < dist <= max(3, len(name) // 2):
                results.append((dist, key))
    results.sort(key=lambda x: (x[0], len(x[1])))
    seen = set()
    unique = []
    for _, key in results:
        if key not in seen:
            seen.add(key)
            unique.append(key)
        if len(unique) >= limit:
            break
    return unique


async def trace(
    function_name: str, focus: Literal["full", "yaml", "dispatch"] = "full"
) -> str:
    """Trace a PyTorch op from Python to C++ implementation with file:line locations."""
    _ensure_loaded()

    md = create_formatter()
    md.h2(f"Trace: `{function_name}`")

    native = _get_native_func(function_name)
    base_name = native.get("base_name", function_name) if native else function_name

    if focus in ("full", "yaml") and native:
        md.h3("Definition (native_functions.yaml)")
        md.code("Signature", native.get("signature", "N/A"))

        if native.get("variants"):
            md.bold("Variants", native["variants"])
        if native.get("structured"):
            md.bold("Structured", "Yes")
        if native.get("structured_delegate"):
            md.bold("Delegate", native["structured_delegate"])
        if native.get("tags"):
            md.bold("Tags", ", ".join(str(t) for t in native["tags"]))

        dispatch = native.get("dispatch", {})
        if dispatch:
            md.blank().text("**Dispatch Config:**")
            # Safe sort handling None keys
            for key, impl in sorted(
                dispatch.items(), key=lambda x: safe_sort_key(x[0])
            ):
                md.item(f"{key or 'default'}: `{impl}`")
        md.blank()

        # Derivative formula
        deriv = _state.derivatives.get(function_name) or _state.derivatives.get(
            base_name
        )
        if deriv and deriv.get("gradients"):
            md.text("**Derivative:**")
            for input_name, formula in deriv["gradients"].items():
                md.item(f"`{input_name}`: `{truncate(formula, 60)}`")
            md.blank()

    if focus in ("full", "dispatch"):
        # From binding detector (TORCH_LIBRARY_IMPL registrations)
        bindings = _state.by_python_name.get(function_name, [])
        if not bindings:
            bindings = _state.by_cpp_name.get(function_name, [])
        if not bindings:
            found = _fuzzy_find(function_name, _state.by_python_name)
            if found:
                bindings = found

        if bindings:
            md.h3("Dispatch Registrations (TORCH_LIBRARY_IMPL)")
            rows = []
            seen = set()
            for b in bindings:
                key = (b.get("dispatch_key"), b.get("cpp_name"), b.get("file_path"))
                if key in seen:
                    continue
                seen.add(key)
                dispatch = b.get("dispatch_key") or "default"
                cpp = b.get("cpp_name") or "N/A"
                path = _rel_path(b.get("file_path") or "")
                line = f":{b['line_number']}" if b.get("line_number") else ""
                rows.append([dispatch, f"`{cpp}`", f"`{path}{line}`"])

            if rows:
                md.table(["Backend", "C++ Function", "File"], rows[:20])
            md.blank()

        # From native_functions.yaml dispatch config
        if native and native.get("dispatch") and not bindings:
            md.h3("Dispatch Config (from YAML)")
            # Safe sort handling None keys
            for key, impl in sorted(
                native["dispatch"].items(), key=lambda x: safe_sort_key(x[0])
            ):
                md.item(f"**{key or 'default'}**: `{impl}`")
            md.blank()

    impls = []  # Initialize for "not found" check later
    if focus == "full":
        if base_name in _state.native_implementations:
            impls.extend(_state.native_implementations[base_name])
        else:
            found = _fuzzy_find(base_name, _state.native_implementations)
            if found:
                impls.extend(found)

        # Also check dispatch targets
        if native:
            for impl_name in native.get("dispatch", {}).values():
                if impl_name in _state.native_implementations:
                    impls.extend(_state.native_implementations[impl_name])

        if impls:
            md.h3("C++ Implementations")
            seen = set()
            for impl in impls[:10]:
                key = (
                    impl["function_name"],
                    impl.get("file_path"),
                    impl.get("line_number"),
                )
                if key in seen:
                    continue
                seen.add(key)

                path = _rel_path(impl.get("file_path", ""))
                line = impl.get("line_number", "")
                md.item(f"`{impl['function_name']}` → `{path}:{line}`")
            md.blank()

    bindings_found = _state.by_python_name.get(function_name) or _state.by_cpp_name.get(
        function_name
    )
    found_anything = native or bindings_found or impls

    if not found_anything:
        similar = _similar_functions(function_name)
        if similar:
            md.h3("Function Not Found")
            md.text(f"No exact match for `{function_name}`. Similar functions:")
            for s in similar[:5]:
                md.item(f"`{s}`")
        else:
            md.text(f"Function `{function_name}` not found in PyTorch bindings.")

    return md.build()


async def _do_cuda_kernels(function_name: str = "") -> str:
    _ensure_loaded()

    md = create_formatter()
    title = f"CUDA Kernels: '{function_name}'" if function_name else "CUDA Kernels"
    md.h2(title)

    kernels = _state.cuda_kernels
    if function_name:
        name_lower = function_name.lower()
        kernels = [k for k in kernels if name_lower in (k.get("name") or "").lower()]

    if not kernels:
        if function_name:
            return f"No CUDA kernels found matching '{function_name}'."
        return "No CUDA kernels found."

    md.text(f"Found {len(kernels)} kernel(s)\n")

    for kernel in kernels[:15]:
        name = kernel.get("name", "unnamed")
        path = _rel_path(kernel.get("file_path", ""))
        line = kernel.get("line_number", "")
        md.item(f"`{name}` → `{path}:{line}`")

        callers = kernel.get("callers", [])
        if callers:
            md.item(f"Called by: {', '.join(f'`{c}`' for c in callers[:3])}", 1)

    if len(kernels) > 15:
        md.text(f"\n*Showing 15 of {len(kernels)} kernels*")

    return md.build()


async def _do_search_bindings(query: str, backend: str = "", limit: int = 10) -> str:
    _ensure_loaded()

    query_lower = query.lower()
    backend_lower = backend.lower() if backend else ""

    matches = []
    for b in _state.bindings:
        # Name match
        name_match = (
            query_lower in (b.get("python_name") or "").lower()
            or query_lower in (b.get("cpp_name") or "").lower()
        )
        if not name_match:
            continue

        # Backend filter
        if backend_lower:
            dispatch_key = (b.get("dispatch_key") or "").lower()
            if backend_lower not in dispatch_key:
                continue

        matches.append(b)

    if not matches:
        filter_msg = f" with backend '{backend}'" if backend else ""
        return f"No bindings found matching '{query}'{filter_msg}."

    seen = set()
    unique = []
    for m in matches:
        key = (m.get("python_name"), m.get("cpp_name"), m.get("dispatch_key"))
        if key not in seen:
            seen.add(key)
            unique.append(m)

    md = create_formatter()
    filter_msg = f" (backend: {backend})" if backend else ""
    md.h2(f"Search: '{query}'{filter_msg}")
    md.text(f"Found {len(unique)} binding(s)\n")

    for b in unique[:limit]:
        dispatch = f" [{b['dispatch_key']}]" if b.get("dispatch_key") else ""
        py_name = b.get("python_name", "N/A")
        cpp_name = b.get("cpp_name", "N/A")
        path = _rel_path(b.get("file_path", ""))
        line = f":{b['line_number']}" if b.get("line_number") else ""
        md.item(f"**{py_name}**{dispatch} → `{cpp_name}` (`{path}{line}`)")

    if len(unique) > limit:
        md.text(f"\n*Showing {limit} of {len(unique)} results*")

    return md.build()
