# TorchTalk

An MCP server that gives Claude Code deep understanding of PyTorch's cross-language architecture (Python → C++ → CUDA).

## What It Does

TorchTalk provides **structural knowledge** that Claude can't get from just reading code:

- **Binding chains**: Trace `torch.matmul` → `at::native::matmul` → `LinearAlgebra.cpp:1996`
- **Impact analysis**: "If I modify GEMM, what breaks?" → Shows all 15 callers with file:line
- **Dispatch mapping**: Which backend (CPU/CUDA/MPS) handles each operation
- **Call graphs**: 60,000+ C++ functions, 139,000+ call edges

## Quick Start

```bash
# Install
pip install -e .

# Add to Claude Code (one command)
claude mcp add torchtalk -s user -- torchtalk mcp-serve --pytorch-source /path/to/pytorch
```

That's it. Claude Code now has access to PyTorch's cross-language bindings.

## Requirements

- **Python 3.10+**
- **PyTorch source code**: `git clone https://github.com/pytorch/pytorch`
- **compile_commands.json** (optional): For full C++ call graph, build PyTorch once:
  ```bash
  cd /path/to/pytorch && python setup.py develop
  ```

## Available Tools

| Tool | Description |
|------|-------------|
| `get_binding_chain(func)` | Full Python → C++ → file mapping |
| `get_native_function(func)` | Definition from native_functions.yaml |
| `get_dispatch_implementations(func)` | Backend implementations (CPU, CUDA, etc.) |
| `get_cpp_callees(func)` | What does this function call? |
| `get_cpp_callers(func)` | What calls this function? (impact analysis) |
| `get_cuda_kernels(func)` | CUDA kernel information |
| `search_bindings(query)` | Search all bindings by name |

## Example Usage

```
You: "What would break if I changed the GEMM implementation?"

Claude uses get_cpp_callers("gemm") →
  gemm is called by:
  - at::native::cpublas::brgemm at CPUBlas.cpp:1347

You: "How does torch.matmul work internally?"

Claude uses get_binding_chain("matmul") + get_cpp_callees("matmul") →
  matmul → _matmul_impl → mm, mv, dot, squeeze, unsqueeze...
```

## Performance

| Scenario | Time |
|----------|------|
| First startup | ~0.3s (C++ call graph builds in background) |
| Background build | ~90s (60K functions, 139K edges) |
| Cached startup | ~0.5s |

## Project Structure

```
torchtalk/
├── src/torchtalk/
│   ├── server.py          # MCP server implementation
│   ├── cli.py             # CLI entry point
│   └── analysis/          # Code analysis modules
│       ├── binding_detector.py   # pybind11/TORCH_LIBRARY detection
│       ├── cpp_call_graph.py     # libclang-based call graph
│       └── repo_analyzer.py      # Python AST analysis
├── .claude/
│   ├── commands/trace.md         # /trace slash command
│   └── skills/.../SKILL.md       # Skill definition
├── .mcp.json              # MCP server config
├── CLAUDE.md              # Project context
└── pyproject.toml         # Package config
```

## How It Works

1. **On first run**: Parses `native_functions.yaml`, detects pybind11 bindings, builds C++ call graph
2. **Caches everything**: Subsequent startups load from `~/.cache/torchtalk/` (~0.5s)
3. **Background building**: C++ call graph builds in background, tools work immediately

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run MCP server directly
python -m torchtalk mcp-serve --pytorch-source /path/to/pytorch
```
