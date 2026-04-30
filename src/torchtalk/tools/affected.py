"""Affected-tests tool implementation."""

from __future__ import annotations

from ..analysis.affected import affected_tests
from ..formatting import create_formatter
from ..indexer import _cpp_status, _ensure_loaded, _state


async def _do_affected(funcs: str, depth: int = 3) -> str:
    _ensure_loaded()
    if status := _cpp_status():
        return status

    func_list = [f.strip() for f in funcs.split(",") if f.strip()]
    if not func_list:
        return "No functions provided."

    result = affected_tests(
        funcs=func_list,
        cpp_extractor=_state.cpp_extractor,
        by_cpp_name=_state.by_cpp_name,
        test_classes=_state.test_classes,
        test_files=_state.test_files,
        opinfo_registry=_state.opinfo_registry,
        opinfo_alias_map=_state.opinfo_alias_map,
        opinfo_test_files=_state.opinfo_test_files,
        test_attr_index=_state.test_attr_index,
        python_profiling=_state.python_profiling or None,
        decomp_alias_map=_state.decomp_alias_map or None,
        native_functions=_state.native_functions or None,
        native_implementations=_state.native_implementations or None,
        depth=depth,
    )

    md = create_formatter()
    md.h2(f"Affected tests for: `{', '.join(func_list)}`")
    md.item(f"Callers walked: {result['callers_walked']}")
    md.item(f"Bindings matched: {len(result['bindings_matched'])}")
    apis = ", ".join(result["python_apis"]) or "(none)"
    md.item(f"Python APIs: {apis}")
    md.blank()

    runs = result["test_runs"]
    if not runs:
        md.text("*No matching test runs found.*")
        return md.build()

    md.h3(f"Test runs ({len(runs)} files)")
    for tr in runs[:30]:
        classes = tr["included_classes"]
        if classes:
            md.item(f"`{tr['file']}` — {', '.join(classes[:5])}")
            if len(classes) > 5:
                md.item(f"...and {len(classes) - 5} more", 1)
        else:
            md.item(f"`{tr['file']}` *(whole file)*")

    if len(runs) > 30:
        md.text(f"\n*Showing 30 of {len(runs)} files.*")

    return md.build()
