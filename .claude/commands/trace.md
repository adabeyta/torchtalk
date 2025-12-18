---
argument-hint: [function-name]
description: Trace a PyTorch function's cross-language binding chain
allowed-tools: mcp__torchtalk__trace, mcp__torchtalk__calls, Read
---

Trace the PyTorch function `$ARGUMENTS`:

1. Use the `mcp__torchtalk__trace` tool with function_name="$ARGUMENTS" to get the binding chain
2. Use the `mcp__torchtalk__calls` tool with function_name="$ARGUMENTS" to show internal dependencies
3. Summarize the dispatch path and implementation locations

IMPORTANT: Use the MCP tools directly. Do NOT try to import/run Python code from torchtalk.server.

Show file:line references for each layer.
