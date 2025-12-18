---
name: torchtalk-analyzer
description: Analyze PyTorch internals across Python, C++, and CUDA layers. Use when asked about how PyTorch operators work internally, where functions are implemented (CPU/CUDA backends), what would break if code is modified (impact analysis), how torch.nn modules connect to native code, or finding tests for PyTorch operators. Covers ATen ops, nn.Module classes, dispatch mechanisms, and test infrastructure.
allowed-tools: mcp__torchtalk__get_status, mcp__torchtalk__trace, mcp__torchtalk__search, mcp__torchtalk__impact, mcp__torchtalk__calls, mcp__torchtalk__called_by, mcp__torchtalk__cuda_kernels, mcp__torchtalk__trace_module, mcp__torchtalk__list_modules, mcp__torchtalk__find_similar_tests, mcp__torchtalk__list_test_utils, mcp__torchtalk__test_file_info, Read, Grep, Glob
---

# TorchTalk PyTorch Analyzer

## When to Use

- "How does torch.X work internally?"
- "Where is X implemented?" (Python/C++/CUDA)
- "What would break if I change X?"
- "How does nn.Linear connect to native code?"
- "Find tests for the softmax operator"

## Quick Start

```
get_status()  # Check what's loaded and available
```

## Tools by Category

### ATen Operators (torch.add, torch.matmul, etc.)

| Tool | Use For |
|------|---------|
| `trace(name, focus?)` | Python → YAML → C++ → file:line |
| `search(query, backend?)` | Find ops by name, filter by CPU/CUDA |
| `cuda_kernels(name?)` | GPU kernel launches |

### Call Graph Analysis (requires PyTorch build)

| Tool | Use For |
|------|---------|
| `impact(name, depth?)` | What breaks if I change this? (transitive callers) |
| `calls(name)` | What does this function call? |
| `called_by(name)` | What calls this function? |

### Python Modules (torch.nn, torch.optim)

| Tool | Use For |
|------|---------|
| `trace_module(name)` | Trace nn.Linear, optim.Adam, etc. |
| `list_modules(category)` | List available modules ("nn", "optim", "all") |

### Test Infrastructure

| Tool | Use For |
|------|---------|
| `find_similar_tests(query)` | Find tests for an operator/concept |
| `list_test_utils(category)` | Available test utilities and patterns |
| `test_file_info(path)` | Details about a specific test file |

## Common Workflows

### "How does torch.matmul work?"
```
trace("matmul")           # Get binding chain
calls("matmul")           # See internal dependencies
```

### "What breaks if I modify GEMM?"
```
impact("gemm", depth=4)   # Transitive callers + Python entry points
```

### "Where is conv2d for CUDA?"
```
search("conv2d", backend="CUDA")
# or
trace("conv2d", focus="dispatch")
```

### "How does nn.Linear work?"
```
trace_module("Linear")    # Python class details
trace("linear")           # Underlying ATen op
```

### "Find tests for softmax"
```
find_similar_tests("softmax")
```

## Requirements

- **Always available:** trace, search, cuda_kernels, trace_module, list_modules, test tools
- **Requires PyTorch build:** impact, calls, called_by (need `compile_commands.json`)

Run `get_status()` to check availability.
