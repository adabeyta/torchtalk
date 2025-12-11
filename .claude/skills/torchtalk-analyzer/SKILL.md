---
name: torchtalk-analyzer
description: Get cross-language binding information for PyTorch codebases (Python → C++ → CUDA). Use when you need to understand dispatch paths, find backend implementations (CPU/CUDA), trace CUDA kernels, or understand how Python APIs connect to native code. Provides structural architectural data, not code search.
allowed-tools: mcp__torchtalk__get_binding_chain, mcp__torchtalk__get_dispatch_implementations, mcp__torchtalk__get_cuda_kernels, mcp__torchtalk__get_implementation_files, mcp__torchtalk__get_call_graph, mcp__torchtalk__list_binding_types, mcp__torchtalk__search_bindings, Read, Grep, Glob
---

# TorchTalk Cross-Language Binding Analyzer

## What This Provides

Structural knowledge about PyTorch-style codebases:
- **Binding chains**: Python API → C++ implementation → file location
- **Dispatch keys**: CPU, CUDA, Meta backend implementations
- **CUDA kernels**: `__global__` functions and what calls them
- **Call graphs**: Function relationships

## Available Tools

### Primary Tools

#### `get_binding_chain(function_name)`
Get the full Python → C++ → CUDA mapping, grouped by backend.

```
get_binding_chain("matmul")
→ CPU: at::native::matmul_cpu → LinearAlgebra.cpp
→ CUDA: at::native::matmul_cuda → Blas.cu
```

#### `get_dispatch_implementations(function_name)`
Table showing all backend implementations.

```
get_dispatch_implementations("add")
→ | CPU | at::native::add | Add.cpp |
→ | CUDA | at::native::add_cuda | Add.cu |
```

#### `get_cuda_kernels(function_name)`
Find CUDA `__global__` kernels and what C++ functions launch them.

#### `get_implementation_files(function_name)`
Get just file paths, organized by type (bindings, C++ impl, CUDA).

### Secondary Tools

#### `get_call_graph(function_name)`
See what calls a function and what it calls.

#### `list_binding_types()`
Summary of all binding types (pybind, torch_library, cuda_kernel, etc.)

#### `search_bindings(query)`
Search all bindings by name.

## Detected Binding Patterns

| Pattern | Description |
|---------|-------------|
| `pybind_function` | `m.def("name", &func)` |
| `pybind_class` | `py::class_<Cpp>(m, "Py")` |
| `torch_library` | `TORCH_LIBRARY(ns, m)` |
| `torch_library_impl` | `TORCH_LIBRARY_IMPL(ns, CUDA, m)` |
| `cuda_kernel` | `__global__ void kernel(...)` |
| `at_dispatch` | `AT_DISPATCH_FLOATING_TYPES(...)` |

## Workflow Example

**User asks:** "How does torch.matmul work on GPU?"

1. Get the binding chain:
   ```
   get_binding_chain("matmul")
   ```
   → Shows CPU and CUDA implementations with file paths

2. Get CUDA kernels:
   ```
   get_cuda_kernels("matmul")
   ```
   → Shows `__global__` functions involved

3. Read the implementation:
   ```
   Read the CUDA file from step 1
   ```

4. Answer with specific file references and code

## Why This Helps

Claude knows PyTorch conceptually. TorchTalk provides:
- **Exact file paths** for this codebase version
- **Dispatch key mappings** (which backend handles what)
- **CUDA kernel locations** (not exposed in Python docs)
- **pybind11/TORCH_LIBRARY binding sites**

This architectural map enables precise answers with real code references.
