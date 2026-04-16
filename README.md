# TorchTalk

An MCP server that gives Claude Code deep understanding of PyTorch's cross-language architecture (Python → C++ → CUDA).

## What It Does

TorchTalk provides **structural knowledge** that Claude can't get from just reading code:

- **Binding chains**: Trace `torch.matmul` → `at::native::matmul` → `LinearAlgebra.cpp:1996`
- **Impact analysis**: "If I modify GEMM, what breaks?" → Shows all 15 callers with file:line
- **Dispatch mapping**: Which backend (CPU/CUDA/MPS) handles each operation.
- **Call graphs**: C++ functions, call edges
- **Test discovery**: Find existing tests for any operator, browse test utilities

## Quick Start

```bash
# Install
pip install -e .

# Add to Claude Code (one command)
claude mcp add torchtalk -s user -- torchtalk mcp-serve --pytorch-source /path/to/pytorch

# Add to Cursor: copies .claude/ into the project's .cursor/ and adds the torchtalk MCP
torchtalk cursor-add -C /path/to/your/project -p /path/to/pytorch
```

## Requirements

- **PyTorch source code**: `git clone https://github.com/pytorch/pytorch`
- **compile_commands.json** (optional): For full C++ call graph, build PyTorch once:
  ```bash
  cd /path/to/pytorch && python setup.py develop
  ```

## Available Tools

| Tool | Description |
|------|-------------|
| `trace(func, focus?)` | Trace any PyTorch op: Python → YAML → C++ → file:line |
| `search(query, mode?, backend?)` | mode="bindings": dispatch registrations. mode="kernels": CUDA kernel launches |
| `graph(func, mode?, depth?)` | mode="callers": inbound. mode="calls": outbound. mode="impact": transitive callers |
| `modules(name, mode?)` | mode="trace": class details. mode="list": browse by category ("nn", "optim", "all") |
| `tests(query, mode?)` | mode="find": search tests. mode="utils": list utilities. mode="file_info": test file details |

## Project Structure

```
torchtalk/
├── src/torchtalk/
│   ├── server.py              # MCP server (6 consolidated tools)
│   ├── indexer.py             # Data loading, caching, initialization
│   ├── cli.py                 # CLI (torchtalk mcp-serve)
│   ├── formatting.py          # Response formatting (CompactText/Markdown)
│   ├── tools/
│   │   ├── ops.py             # trace, search, cuda_kernels
│   │   ├── graph.py           # calls, called_by, impact
│   │   ├── modules.py         # trace_module, list_modules
│   │   └── tests.py           # find_similar_tests, list_test_utils, test_file_info
│   └── analysis/
│       ├── binding_detector.py    # pybind11/TORCH_LIBRARY detection (tree-sitter)
│       ├── cpp_call_graph.py      # C++ call graph extraction (libclang)
│       ├── python_analyzer.py     # Python module/class analysis (AST)
│       ├── patterns.py            # Search directories, exclusion patterns
│       └── helpers.py             # Utility functions
├── .claude/
│   ├── commands/trace.md      # /trace slash command
│   └── skills/.../SKILL.md    # Skill definition
├── .mcp.json                  # MCP server config
├── CLAUDE.md                  # Project context
└── pyproject.toml             # Package config
```

## How It Works

1. **On first run**: Parses `native_functions.yaml`, detects pybind11 bindings, builds C++ call graph
2. **Caches everything**: Subsequent startups load from `~/.cache/torchtalk/`
3. **Background building**: C++ call graph builds in background, tools work immediately
4. **Test indexing**: Scans `test/` and `torch/testing/` for test classes, functions, and OpInfo definitions

## Indexed Data

| Data Source | What's Extracted |
|-------------|------------------|
| `native_functions.yaml` | ATen operator definitions with dispatch configs |
| `derivatives.yaml` | Backward pass formulas for autograd |
| C++ source | TORCH_LIBRARY bindings, pybind11, CUDA kernels |
| Python source | torch.nn modules, optimizers, method signatures |
| Test files | Test classes, test functions, OpInfo registry |
