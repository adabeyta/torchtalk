# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## Build and Development Commands

```bash
# Install
pip install -e .

# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run MCP server directly
python -m torchtalk mcp-serve --pytorch-source /path/to/pytorch
```

## Project Structure

```
src/torchtalk/
├── server.py              # MCP server - main entry point for Claude Code
├── cli.py                 # CLI (`torchtalk mcp-serve`)
└── analysis/
    ├── binding_detector.py    # pybind11/TORCH_LIBRARY detection (tree-sitter)
    ├── cpp_call_graph.py      # C++ call graph extraction (libclang)
    └── repo_analyzer.py       # Python import/call graphs (AST)
```

## Architecture

TorchTalk is an MCP server providing cross-language binding analysis for PyTorch.

### Core Components

**MCP Server** (`src/torchtalk/server.py`):
- FastMCP-based server with tools for binding chain lookup
- Auto-builds and caches index from PyTorch source
- Background thread for C++ call graph building (non-blocking startup)

**Analysis Modules** (`src/torchtalk/analysis/`):
- `BindingDetector`: Parses C++ for pybind11 patterns using tree-sitter
- `CppCallGraphExtractor`: Parallel libclang-based call graph (60K+ functions)
- `RepoAnalyzer`: Python AST analysis for import/call graphs

### Data Sources

1. **native_functions.yaml**: Authoritative ATen operator definitions
2. **derivatives.yaml**: Backward pass formulas
3. **compile_commands.json**: For libclang C++ parsing
4. **Tree-sitter**: TORCH_LIBRARY_IMPL and pybind11 detection

### Caching

All data cached to `~/.cache/torchtalk/`:
- `bindings_*.json`: Binding data (~10MB)
- `call_graph/pytorch_callgraph_*.json`: C++ call graph (~50MB)

## MCP Tools

| Tool | Description |
|------|-------------|
| `get_binding_chain(func)` | Full Python → C++ → file mapping |
| `get_native_function(func)` | Definition from native_functions.yaml |
| `get_dispatch_implementations(func)` | Backend implementations (CPU, CUDA) |
| `get_cpp_callees(func)` | What does this function call? |
| `get_cpp_callers(func)` | What calls this function? |
| `get_cuda_kernels(func)` | CUDA kernel information |
| `search_bindings(query)` | Search all bindings |

## Key Files

- `.mcp.json` - MCP server configuration for Claude Code
- `.claude/skills/torchtalk-analyzer/SKILL.md` - Skill definition
- `.claude/commands/trace.md` - `/trace` slash command
