"""Call graph tool implementations."""

from __future__ import annotations

from ..analysis.helpers import dedupe_by_key
from ..formatting import create_formatter, relative_path
from ..indexer import _cpp_status, _ensure_loaded, _state


def _rel_path(path: str) -> str:
    return relative_path(path, _state.pytorch_source)


def _format_call_item(md, item: dict, name_key: str, file_key: str, line_key: str):
    name = item[name_key]
    if file_path := item.get(file_key):
        line = f":{item[line_key]}" if item.get(line_key) else ""
        md.item(f"`{name}` \u2192 `{_rel_path(file_path)}{line}`")
    else:
        md.item(f"`{name}`")


async def _do_calls(function_name: str) -> str:
    _ensure_loaded()
    if status := _cpp_status():
        return status

    callees = _state.cpp_extractor.get_callees(function_name, fuzzy=True)
    if not callees:
        return f"No outbound calls found for '{function_name}'."

    results = dedupe_by_key(callees, "callee")

    md = create_formatter()
    md.h2(f"Calls: `{function_name}`")
    md.text("*Functions this calls (outbound dependencies):*\n")

    for item in results[:15]:
        _format_call_item(md, item, "callee", "callee_file", "callee_line")

    if len(results) > 15:
        md.text(f"\n*Showing 15 of {len(results)} calls.*")

    return md.build()


async def _do_called_by(function_name: str) -> str:
    _ensure_loaded()
    if status := _cpp_status():
        return status

    callers = _state.cpp_extractor.get_callers(function_name, fuzzy=True)
    if not callers:
        return f"No inbound callers found for '{function_name}'."

    results = dedupe_by_key(callers, "caller")

    md = create_formatter()
    md.h2(f"Called by: `{function_name}`")
    md.text("*Functions that call this (inbound dependents):*\n")

    for item in results[:15]:
        _format_call_item(md, item, "caller", "caller_file", "caller_line")

    if len(results) > 15:
        md.text(f"\n*Showing 15 of {len(results)} callers.*")

    return md.build()


async def _do_impact(function_name: str, depth: int = 2, focus: str = "callers") -> str:
    _ensure_loaded()
    if status := _cpp_status():
        return status

    depth = min(max(depth, 1), 5)

    visited = set()
    current_level = {function_name}
    callers_by_depth: dict[int, list[dict]] = {}

    for level in range(1, depth + 1):
        next_level = set()
        level_callers = []

        for func in current_level:
            for item in _state.cpp_extractor.get_callers(func, fuzzy=(level == 1)):
                caller = item["caller"]
                if caller not in visited and caller != function_name:
                    visited.add(caller)
                    next_level.add(caller)
                    level_callers.append(item)

        if level_callers:
            callers_by_depth[level] = level_callers
        current_level = next_level
        if not current_level:
            break

    if not callers_by_depth:
        return f"No callers found for '{function_name}'."

    md = create_formatter()
    md.h2(f"Impact Analysis: `{function_name}`")
    md.text(f"*Tracing callers up to {depth} levels deep*\n")

    total = 0
    for level, callers in callers_by_depth.items():
        unique = dedupe_by_key(callers, "caller")
        total += len(unique)
        md.h3(f"Depth {level} ({len(unique)} callers)")

        for item in unique[:15]:
            _format_call_item(md, item, "caller", "caller_file", "caller_line")

        if len(unique) > 15:
            md.item(f"*... and {len(unique) - 15} more*")
        md.blank()

    if focus == "full":
        python_entries = [
            {
                "python": b.get("python_name", c),
                "cpp": c,
                "dispatch": b.get("dispatch_key", ""),
            }
            for c in visited
            if c in _state.by_cpp_name
            for b in _state.by_cpp_name[c][:1]
        ]

        if python_entries:
            md.h3(f"Python Entry Points ({len(python_entries)} found)")
            for entry in python_entries[:10]:
                dispatch = f" [{entry['dispatch']}]" if entry["dispatch"] else ""
                md.item(f"`{entry['python']}`{dispatch} → `{entry['cpp']}`")
            if len(python_entries) > 10:
                md.item(f"*... and {len(python_entries) - 10} more*")

    md.text(f"Total impact: {total} functions across {len(callers_by_depth)} levels")

    return md.build()
