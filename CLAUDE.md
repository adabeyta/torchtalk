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
PYTORCH_SOURCE=/path/to/pytorch pytest tests/test_binding_detector_pytorch.py

# Run MCP server directly
python -m torchtalk mcp-serve --pytorch-source /path/to/pytorch
```

## Project Structure

```
src/torchtalk/
├── server.py              # MCP server - main entry point for Claude Code
├── cli.py                 # CLI (`torchtalk mcp-serve`)
├── formatting.py          # Markdown output formatting
└── analysis/
    ├── binding_detector.py    # pybind11/TORCH_LIBRARY detection (tree-sitter)
    ├── cpp_call_graph.py      # C++ call graph extraction (libclang)
    ├── python_analyzer.py     # Python module/class analysis (AST)
    ├── config.py              # Search directories, exclusion patterns
    └── helpers.py             # Utility functions
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
- `PythonAnalyzer`: Python AST analysis for modules, classes, nn.Module subclasses

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

**IMPORTANT**: Use the `mcp__torchtalk__*` tools directly (e.g., `mcp__torchtalk__trace`). Do NOT try to import or run Python code from `torchtalk.server` manually.

### Operators
| Tool | Description |
|------|-------------|
| `trace(func, focus?)` | Trace any PyTorch op: Python → YAML → C++ → file:line. Focus: "full", "yaml", "dispatch" |
| `search(query, mode?, backend?)` | mode="bindings" (default): find dispatch registrations. mode="kernels": find CUDA kernel launches |

### Call Graph (requires PyTorch build)
| Tool | Description |
|------|-------------|
| `graph(func, mode?, depth?)` | mode="callers" (default): inbound. mode="calls": outbound. mode="impact": transitive callers |

### Python Modules
| Tool | Description |
|------|-------------|
| `modules(name, mode?)` | mode="trace" (default): class details. mode="list": browse by category ("nn", "optim", "all") |

### Test Infrastructure
| Tool | Description |
|------|-------------|
| `tests(query, mode?)` | mode="find" (default): search tests. mode="utils": list utilities. mode="file_info": details on a test file |

## Key Files

- `.mcp.json` - MCP server configuration for Claude Code
- `.claude/skills/torchtalk-analyzer/SKILL.md` - Skill definition
- `.claude/commands/trace.md` - `/trace` slash command
