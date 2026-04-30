"""TorchTalk MCP Server - Tool definitions and entry point."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

from mcp.server.fastmcp import FastMCP

from .formatting import create_formatter
from .indexer import (
    _auto_detect_pytorch,
    _init_from_source,
    _load_from_json,
    _state,
)
from .tools.affected import _do_affected
from .tools.graph import _do_called_by, _do_calls, _do_impact
from .tools.modules import _do_list_modules, _do_trace_module
from .tools.ops import (
    _do_cuda_kernels,
    _do_search_bindings,
)
from .tools.ops import (
    trace as _do_trace,
)
from .tools.tests import _do_find_similar_tests, _do_list_test_utils, _do_test_file_info

log = logging.getLogger(__name__)
mcp = FastMCP("torchtalk")


@mcp.tool()
async def get_status() -> str:
    """Get TorchTalk status and available tools."""
    md = create_formatter()
    md.h2("TorchTalk Status")

    if _state.pytorch_source:
        md.code("PyTorch Source", _state.pytorch_source)
    else:
        md.bold("PyTorch Source", "Not configured")
    md.blank()

    if _state.bindings:
        md.bold("Bindings", f"{len(_state.bindings):,} loaded")
        md.item(f"CUDA kernels: {len(_state.cuda_kernels):,}", 1)
        md.item(f"Dispatch keys: {len(_state.by_dispatch_key)}", 1)
        if _state.by_dispatch_key:
            top_keys = sorted(_state.by_dispatch_key.items(), key=lambda x: -len(x[1]))[
                :5
            ]
            for key, bindings in top_keys:
                md.item(f"{key}: {len(bindings)}", 2)
    else:
        md.bold("Bindings", "Not loaded")
    md.blank()

    if _state.native_functions:
        md.bold(
            "Native Functions",
            f"{len(_state.native_functions):,} operators",
        )
    if _state.derivatives:
        md.bold("Derivatives", f"{len(_state.derivatives):,} formulas")
    md.blank()

    md.h3("C++ Call Graph")
    if _state.cpp_building:
        md.bold("Status", "Building (~1-2 min)")
    elif _state.cpp_extractor:
        stats = _state.cpp_extractor.get_call_graph_data()["stats"]
        md.bold("Status", "Ready")
        md.item(f"Functions: {stats['total_functions']:,}", 1)
        md.item(f"Call edges: {stats['total_call_edges']:,}", 1)
    else:
        md.bold("Status", "Not available")
        if _state.pytorch_source:
            src = Path(_state.pytorch_source)
            if not (src / "compile_commands.json").exists():
                md.item("Missing compile_commands.json", 1)
                md.item(
                    "Fix: Build PyTorch with `python setup.py develop`",
                    1,
                )
    md.blank()

    md.h3("Python Modules")
    if _state.py_modules:
        md.bold("Status", "Ready")
        md.item(f"Modules: {len(_state.py_modules):,}", 1)
        md.item(
            f"Classes: {sum(len(v) for v in _state.py_classes.values()):,}",
            1,
        )
        md.item(f"nn.Module subclasses: {len(_state.nn_modules):,}", 1)
    else:
        md.bold("Status", "Not loaded")
    md.blank()

    md.h3("Test Infrastructure")
    if _state.test_files:
        md.bold("Status", "Ready")
        md.item(f"Test files: {len(_state.test_files):,}", 1)
        md.item(f"Test classes: {len(_state.test_classes):,}", 1)
        md.item(f"Test functions: {len(_state.test_functions):,}", 1)
        md.item(
            f"OpInfo definitions: {len(_state.opinfo_registry):,}",
            1,
        )
    else:
        md.bold("Status", "Not loaded")
    md.blank()

    ready = "Ready" if _state.bindings else "Not ready"
    cpp_ready = (
        "Ready"
        if _state.cpp_extractor
        else ("Building..." if _state.cpp_building else "Not ready")
    )
    py_ready = "Ready" if _state.py_modules else "Not ready"
    test_ready = "Ready" if _state.test_files else "Not ready"

    md.h3("Available Tools")
    md.table(
        ["Tool", "Status", "Description"],
        [
            [
                "`trace`",
                ready,
                "Trace a PyTorch op: Python → C++ → file:line",
            ],
            [
                "`search`",
                ready,
                "Search bindings (mode=bindings) or CUDA kernels (mode=kernels)",
            ],
            [
                "`graph`",
                cpp_ready,
                "C++ call graph: callers, calls, or impact analysis",
            ],
            [
                "`modules`",
                py_ready,
                "Python modules: trace a class or list by category",
            ],
            [
                "`tests`",
                test_ready,
                "Find tests, list utils, or get test file details",
            ],
            [
                "`affected`",
                cpp_ready,
                "Map changed C++ functions to impacted Python tests",
            ],
        ],
    )

    return md.build()


@mcp.tool()
async def trace(
    function_name: str,
    focus: Literal["full", "yaml", "dispatch"] = "full",
) -> str:
    """Trace a PyTorch op from Python to C++ implementation with file:line locations."""
    return await _do_trace(function_name, focus)


@mcp.tool()
async def search(
    query: str,
    mode: Literal["bindings", "kernels"] = "bindings",
    backend: str = "",
    limit: int = 10,
) -> str:
    """Search PyTorch bindings or CUDA kernels by name.

    mode='bindings' for dispatch registrations,
    mode='kernels' for GPU kernel launches.
    """
    if mode == "kernels":
        return await _do_cuda_kernels(query)
    return await _do_search_bindings(query, backend=backend, limit=limit)


@mcp.tool()
async def graph(
    function_name: str,
    mode: Literal["calls", "callers", "impact"] = "callers",
    depth: int = 2,
) -> str:
    """Query the C++ call graph.

    mode='callers' for inbound, 'calls' for outbound,
    'impact' for transitive callers.
    """
    if mode == "calls":
        return await _do_calls(function_name)
    elif mode == "impact":
        return await _do_impact(function_name, depth=depth)
    return await _do_called_by(function_name)


@mcp.tool()
async def modules(
    name: str,
    mode: Literal["trace", "list"] = "trace",
) -> str:
    """Query Python modules.

    mode='trace' for class details,
    mode='list' for browsing by category ('nn', 'optim', 'all').
    """
    if mode == "list":
        return await _do_list_modules(category=name)
    return await _do_trace_module(module_name=name)


@mcp.tool()
async def tests(
    query: str,
    mode: Literal["find", "utils", "file_info"] = "find",
    limit: int = 10,
) -> str:
    """Query PyTorch test infrastructure.

    mode='find' to search tests, 'utils' for utilities,
    'file_info' for details on a test file.
    """
    if mode == "utils":
        return await _do_list_test_utils()
    elif mode == "file_info":
        return await _do_test_file_info(query)
    return await _do_find_similar_tests(query, limit=limit)


@mcp.tool()
async def affected(funcs: str, depth: int = 3) -> str:
    """Map changed C++ functions (comma-separated) to impacted Python test files."""
    return await _do_affected(funcs, depth)


def run_server(
    pytorch_source: str | None = None,
    index_path: str | None = None,
    transport: str = "stdio",
):
    """Start MCP server. Heavy init runs in background so the MCP client's
    initialize handshake completes immediately; tools return a 'not loaded'
    error via `_ensure_loaded` until the data is ready."""
    import threading

    source = pytorch_source or _auto_detect_pytorch()

    def _bg_init():
        try:
            if source:
                _init_from_source(source)
            elif index_path:
                _load_from_json(index_path)
            else:
                log.warning(
                    "No PyTorch source specified. "
                    "Tools will return errors until data is loaded."
                )
        except Exception:
            log.exception("Background init failed")

    threading.Thread(target=_bg_init, daemon=True).start()

    log.info("Starting TorchTalk MCP server...")
    mcp.run(transport=transport)
