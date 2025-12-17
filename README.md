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
| `trace(func, focus?)` | Python → YAML → C++ → file:line. Focus: "full", "yaml", "dispatch" |
| `search(query, backend?)` | Find bindings by name with optional backend filter (CPU/CUDA/Meta) |
| `cuda_kernels(func?)` | GPU kernel `<<<>>>` launches with file:line |
| `impact(func, depth?)` | Transitive callers + Python entry points (security/refactoring) |
| `calls(func)` | Outbound: functions `func` invokes internally |
| `called_by(func)` | Inbound: functions that invoke `func` |
| `trace_module(name)` | Trace Python module: torch.nn.Linear, torch.optim.Adam, etc. |
| `list_modules(category)` | List nn.Module classes, optimizers, or search by name |
| `find_similar_tests(query)` | Find tests for an operator/function/concept |
| `list_test_utils(category?)` | List test utilities: fixtures, assertions, decorators |
| `test_file_info(path)` | Details about a specific test file |

## Project Structure

```
torchtalk/
├── src/torchtalk/
│   ├── server.py              # MCP server implementation
│   ├── cli.py                 # CLI entry point
│   ├── formatting.py          # Markdown output utilities
│   └── analysis/
│       ├── config.py          # Shared configuration (search dirs, patterns)
│       ├── binding_detector.py    # pybind11/TORCH_LIBRARY detection
│       ├── cpp_call_graph.py      # libclang-based call graph
│       ├── python_analyzer.py     # Python AST analysis (torch.nn, etc.)
│       ├── repo_analyzer.py       # Import/call graph analysis
│       └── helpers.py             # Shared utilities
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
