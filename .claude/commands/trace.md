---
argument-hint: [function-name]
description: Trace a PyTorch function's cross-language binding chain
allowed-tools: mcp__torchtalk__get_binding_chain, mcp__torchtalk__get_implementation_files, Read
---

Trace the PyTorch function `$ARGUMENTS`:

1. Use `get_binding_chain` to find the Python → C++ → CUDA mapping
2. Use `get_implementation_files` to get all relevant file paths
3. Summarize the dispatch path clearly

Show the complete binding chain from Python API to native implementation.
