# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build and Development Commands

```bash
# Install dependencies
pip install -r requirements.txt
pip install -e .

# Run all tests
pytest

# Run specific test files
pytest tests/test_conversation_engine.py -v

# Run only integration tests (requires TEST_INDEX_PATH and running vLLM)
pytest -m integration

# Run only unit tests (skip slow integration tests)
pytest -m "not slow"
```

## CLI Commands

```bash
# Build index for a repository
torchtalk index /path/to/repo --output ./index

# Start chat interface (auto-launches vLLM if not running)
torchtalk chat --index ./index

# Start vLLM server only
torchtalk serve-vllm --model meta-llama/llama-4-maverick --port 8000
```

## Architecture

TorchTalk is a PyTorch codebase chatbot with cross-language tracing (Python/C++/CUDA).

### Core Components

**Indexing Pipeline** (`torchtalk/indexing/graph_enhanced_indexer.py`):
- Uses `BindingDetector` to detect pybind11 bindings between Python and C++
- Uses `RepoAnalyzer` to build import and call graphs via AST analysis
- Creates LlamaIndex nodes with graph metadata injected
- Persists to vector store using HuggingFace embeddings (BAAI/bge-small-en-v1.5)

**Analysis Modules** (`torchtalk/analysis/`):
- `BindingDetector`: Parses C++ files for pybind11 patterns using tree-sitter
- `RepoAnalyzer`: Builds NetworkX graphs for imports and function calls from Python AST

**Conversation Engine** (`torchtalk/engine/conversation_engine.py`):
- Wraps LlamaIndex's `CondensePlusContextChatEngine` for follow-up handling
- Includes workaround for llama-index-llms-vllm API mode bug (patches `Vllm.complete`)
- Uses `PostprocessedRetriever` for rerank → filter → reorder pipeline

**Retrieval Pipeline** (`torchtalk/engine/postprocessed_retriever.py`):
- Vector search → Cross-encoder rerank → Similarity filter → Long context reorder
- Falls back to a placeholder node if no relevant context found

**UI** (`app.py`): Gradio chat interface with conversation memory

**CLI** (`torchtalk/cli.py`): Entry point for `torchtalk` command with subcommands for index/chat/serve-vllm/mcp-serve

**MCP Server** (`torchtalk/mcp_server.py`): Model Context Protocol server for Claude Code integration
- Provides STRUCTURAL data: binding chains, call graphs, import graphs
- Does NOT do semantic search (Claude has Grep/Glob for that)
- Tells Claude WHERE code connects across Python/C++/CUDA boundaries

### Data Flow

1. `torchtalk index` → `GraphEnhancedIndexer.build_index()` → persisted vector index
2. `torchtalk chat` → starts vLLM (if needed) → loads index → `ConversationEngine` → Gradio UI
3. User query → `PostprocessedRetriever` → graph-enhanced nodes → LLM response

### Key Dependencies

- LlamaIndex for indexing/retrieval framework
- vLLM for LLM inference (OpenAI-compatible API)
- tree-sitter for C++ parsing (binding detection)
- ChromaDB as vector store
- Gradio for web UI
- MCP SDK for Claude Code integration

## Claude Code MCP Integration

TorchTalk provides an MCP server that gives Claude **structural knowledge** about cross-language codebases. Instead of semantic search (which Claude can do with Grep/Glob), it provides the architectural map: which Python functions bind to which C++ implementations, call graphs, and import relationships.

### Setup

```bash
# 1. Build the index first
torchtalk index /path/to/pytorch --output ./index

# 2. Add MCP server to Claude Code
claude mcp add torchtalk -- torchtalk mcp-serve --index ./index

# Or use the project's .mcp.json (auto-detected by Claude Code)
```

### Available MCP Tools

| Tool | Description |
|------|-------------|
| `get_binding_chain(function)` | Get Python → C++ → file mapping for a function |
| `get_implementation_files(function)` | Get file paths by language (Python/C++/CUDA) |
| `get_call_graph(function)` | See what calls what |
| `get_import_graph(file)` | See module dependencies |
| `list_all_bindings()` | Summary of available binding data |

### How It Helps Claude

Claude already knows PyTorch conceptually. TorchTalk provides:
- **Exact file paths** for this specific codebase version
- **Binding mappings** (which C++ function implements which Python API)
- **pybind11 locations** for cross-language dispatch

Example workflow:
```
User: "How does torch.matmul work?"

Claude uses: get_binding_chain("matmul")
→ Returns: torch.matmul → at::matmul → aten/src/ATen/native/LinearAlgebra.cpp

Claude then uses: Read to examine that file
→ Gives precise answer with actual code
```

### Files

- `.mcp.json` - MCP server configuration for Claude Code
- `.claude/skills/torchtalk-analyzer/SKILL.md` - Skill with usage guidance
- `.claude/commands/trace.md` - `/trace` slash command
