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

## CLI Commands

| Command | Description |
|---------|-------------|
| `init --pytorch-source <path>` | Save PyTorch source path to config |
| `status` | Show config and cache status |
| `mcp-serve` | Start the MCP server |
| `index build [--no-wait]` | Build or refresh the index and exit (headless) |
| `index update --since <snapshot>` | Incrementally refresh bindings for files changed since `<snapshot>`'s commit |
| `snapshot save <name>` | Capture current cache as a named snapshot |
| `snapshot load <name\|--nearest> [--force]` | Restore a snapshot into the cache |
| `snapshot list` | List saved snapshots |
| `snapshot delete <name>` | Delete a snapshot |
| `snapshot diff <a> <b> [--json]` | Structural diff between two snapshots |
| `snapshot export <name> [-o file]` | Package a snapshot into a `.tar.gz` |
| `snapshot import <archive> [--name new]` | Extract a snapshot tarball |

Snapshot names may use up to three `/`-separated components (e.g. `main/abc1234/v1`), so you can namespace snapshots by branch, commit, or release.

## Snapshot Matching

Each snapshot records:

- **`source_fingerprint`** — hash of the indexed PyTorch source path (per-checkout).
- **`git_commit`** — short HEAD at save time.
- **`content_fingerprint`** — BLAKE2b over `HEAD^{tree}` + uncommitted diff; a Merkle-style content hash that's identical across checkouts of the same code.

`snapshot load` accepts a snapshot whose content or path fingerprint matches the current source. `snapshot load --nearest` resolves in tiered order: exact content match → exact commit match → most recent ancestor commit (via `git merge-base --is-ancestor`).

## CI Integration

Snapshots make TorchTalk usable in CI without rebuilding the index per job. Build the index once on a nightly runner, ship the `.tar.gz` as a build artifact, and pull it into PR jobs.

**Nightly job: build and publish**

```yaml
- run: torchtalk init --pytorch-source $GITHUB_WORKSPACE/pytorch
- run: torchtalk index build
- run: torchtalk snapshot save nightly/${{ github.sha }}
- run: torchtalk snapshot export nightly/${{ github.sha }} -o torchtalk-index.tar.gz
- uses: actions/upload-artifact@v4
  with: { name: torchtalk-index, path: torchtalk-index.tar.gz }
```

**PR job: load and use**

```yaml
- uses: actions/download-artifact@v4
  with: { name: torchtalk-index }
- run: torchtalk snapshot import torchtalk-index.tar.gz
- run: torchtalk snapshot load --nearest
- run: torchtalk mcp-serve &
```

**Fast PR refresh with `index update`**

When only a few files changed vs. the baseline, skip the full rebuild:

```yaml
- run: torchtalk snapshot load baseline --force
- run: torchtalk index update --since baseline
```

Incremental update re-parses only the C++/CUDA files that `git diff <baseline-commit>..HEAD` reports as changed, and evicts their contributions from the C++ call graph before re-attributing. Header changes (`.h`/`.hpp`/`.hxx`/`.hh`/`.inc`) are resolved via per-TU include sets captured during the baseline build (`TranslationUnit.get_includes()`): every TU whose include closure contains a changed header is added to the re-parse set. Over-invalidation is possible (textual inclusion is a superset of semantic dependency) but never under-invalidation.

A changed header that isn't in any TU's baseline include set — typically from a generated header added after baseline, a truly unused header, or a TU that failed to parse at baseline — is surfaced as a warning with up to 5 sample paths. The incremental update still proceeds for the covered set; run `torchtalk index build` if the warning matters for the task.

**Change-gated workflow**

Use `snapshot diff --json` upstream to decide what (if anything) to re-run:

```bash
torchtalk snapshot diff nightly/latest current --json \
  | jq '.files_modified | length'
```

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
