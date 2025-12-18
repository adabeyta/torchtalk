---
argument-hint: [function-name]
description: Trace a PyTorch function's cross-language binding chain
allowed-tools: mcp__torchtalk__trace, mcp__torchtalk__calls, Read
---

Trace the PyTorch function `$ARGUMENTS`:

1. Use `trace("$ARGUMENTS")` to get the Python → YAML → C++ binding chain
2. If available, use `calls("$ARGUMENTS")` to show internal dependencies
3. Summarize the dispatch path and implementation locations

Show file:line references for each layer.
