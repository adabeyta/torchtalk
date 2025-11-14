# TorchTalk 🔥

A PyTorch codebase chatbot with cross-language tracing (Python ↔ C++ ↔ CUDA) powered by graph-enhanced RAG and 1M context windows.

## Features

- **Cross-language binding detection**: Automatically discovers pybind11 bindings between Python, C++, and CUDA
- **Graph-enhanced retrieval**: Indexes call graphs, import graphs, and cross-language relationships
- **Conversation memory**: Automatic follow-up question handling with context
- **1M context window support**: Uses vLLM with Llama 4 Maverick for long-context understanding
- **Minimal dependencies**: Built on open-source tools (LlamaIndex, vLLM, Gradio)

## Quick Start

### 1. Install

```bash
# Install dependencies
pip install -r requirements.txt

# Install torchtalk CLI
pip install -e .
```

### 2. Build Index (One-time setup)

```bash
# Index the PyTorch repository (or any Python/C++/CUDA codebase)
torchtalk index /path/to/pytorch --output ./index
```

This will:
- Detect cross-language bindings
- Build call and import graphs
- Create vector embeddings with graph metadata
- Persist to `./index/`

### 3. Start Chatting (One command!)

```bash
# Automatically starts vLLM + Gradio UI
torchtalk chat --index ./index
```

That's it! Visit http://localhost:7860 to start chatting.

The `chat` command will:
- Check if vLLM is already running at http://localhost:8000
- If not, automatically launch it with sensible defaults
- Stream vLLM logs to console for debugging
- Launch Gradio UI at http://localhost:7860
- Gracefully shutdown both services on Ctrl+C

### Advanced Usage

```bash
# Use a different model
torchtalk chat --index ./index --model meta-llama/llama-4-maverick

# Use a different vLLM port
torchtalk chat --index ./index --vllm-server http://localhost:9000

# Create a public share link
torchtalk chat --index ./index --share

# Start vLLM server only (without UI)
torchtalk serve-vllm --model meta-llama/llama-4-maverick --port 8000
```

## Example Questions

- "How does torch.matmul connect to the C++ implementation?"
- "Explain the CUDA kernel for matrix multiplication"
- "What are the main components of the autograd engine?"
- "Show me how Python tensor operations bind to C++ implementations"
- "How does PyTorch handle gradient computation?"

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Phase 1: Indexing                         │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐    │
│  │ Binding      │   │ Call Graph   │   │ Import Graph │    │
│  │ Detector     │   │ Builder      │   │ Builder      │    │
│  └──────┬───────┘   └──────┬───────┘   └──────┬───────┘    │
│         │                  │                   │             │
│         └──────────────────┴───────────────────┘             │
│                            │                                 │
│                  ┌─────────▼──────────┐                      │
│                  │ Graph-Enhanced     │                      │
│                  │ Indexer            │                      │
│                  │ (LlamaIndex)       │                      │
│                  └─────────┬──────────┘                      │
│                            │                                 │
│                  ┌─────────▼──────────┐                      │
│                  │ Vector Store       │                      │
│                  │ (ChromaDB)         │                      │
│                  └────────────────────┘                      │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                 Phase 2: vLLM Server                         │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ vLLM (Llama 4 Maverick, 1M context)                 │   │
│  │ - KV cache paging                                    │   │
│  │ - Prefix caching                                     │   │
│  │ - Chunked prefill                                    │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│           Phase 3: Conversation Engine                       │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ CondensePlusContextChatEngine (LlamaIndex)          │   │
│  │ - Automatic query condensation for follow-ups       │   │
│  │ - ChatMemoryBuffer for conversation state           │   │
│  │ - Retrieval from graph-enhanced index               │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                   Phase 4: Gradio UI                         │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ Minimal chat interface with:                        │   │
│  │ - Message history                                    │   │
│  │ - Example questions                                  │   │
│  │ - Clear/reset button                                 │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## Testing

Run unit tests (fast, no external dependencies):

```bash
# All unit tests
pytest -v -m "not integration and not slow"

# Phase-specific tests
pytest tests/test_graph_indexing.py -v -m "not slow"
pytest tests/test_vllm_server.py -v -m "not integration"
pytest tests/test_conversation_engine.py -v -m "not integration"
pytest tests/test_app.py -v -m "not integration"
```

Run integration tests (requires index + vLLM server):

```bash
# Set environment variables
export TEST_INDEX_PATH=./index
export VLLM_SERVER_URL=http://localhost:8000

# Run integration tests
pytest -v -m "integration"

# Run slow integration tests
pytest -v -m "slow and integration"
```

## Configuration

### Environment Variables

vLLM server (`scripts/start_vllm_server.py`):
- `MODEL_NAME`: Model to serve (default: meta-llama/llama-4-maverick)
- `MAX_MODEL_LEN`: Max context length (default: 1000000)
- `PORT`: Server port (default: 8000)
- `HOST`: Server host (default: 0.0.0.0)
- `GPU_MEMORY_UTIL`: GPU memory utilization 0-1 (default: 0.9)
- `CUDA_VISIBLE_DEVICES`: GPU devices (default: 0)
- `VLLM_ATTENTION_BACKEND`: FlashInfer/Triton (optional, for benchmarking)

Gradio UI (`app.py`):
- No environment variables required; use CLI args instead

Testing:
- `TEST_INDEX_PATH`: Path to test index for integration tests
- `VLLM_SERVER_URL`: vLLM server URL (default: http://localhost:8000)
- `TEST_VLLM_MODEL`: Model name for tests (default: meta-llama/llama-4-maverick)

## Project Structure

```
torchtalk/
├── app.py                      # Gradio UI entry point
├── scripts/
│   └── start_vllm_server.py    # vLLM launcher with preflight checks
├── torchtalk/
│   ├── analysis/               # Code analysis (bindings, graphs)
│   │   ├── binding_detector.py
│   │   └── repo_analyzer.py
│   ├── core/
│   │   └── config.py           # Configuration
│   ├── engine/
│   │   └── conversation_engine.py  # LlamaIndex chat engine
│   └── indexing/
│       └── graph_enhanced_indexer.py  # Graph metadata injection
└── tests/
    ├── test_graph_indexing.py      # Phase 1 tests
    ├── test_vllm_server.py         # Phase 2 tests
    ├── test_conversation_engine.py # Phase 3 tests
    ├── test_app.py                 # Phase 4 tests
    └── test_integration.py         # Phase 5 end-to-end tests
```

## Development

### Adding New Features

1. **New analysis modules**: Add to `torchtalk/analysis/`
2. **Custom retrievers**: Extend `ConversationEngine` in `torchtalk/engine/`
3. **UI improvements**: Modify `app.py` (keep it minimal!)
4. **Tests**: Add to `tests/` with appropriate markers (`@pytest.mark.integration`, `@pytest.mark.slow`)

### Key Design Principles

- **Minimize custom code**: Use LlamaIndex, vLLM, Gradio for heavy lifting
- **Test-driven**: Every phase has unit tests
- **Env-first config**: Environment variables > CLI args > defaults
- **Fail fast**: Preflight checks catch issues before exec
- **Observable**: Logging at every phase

## Troubleshooting

### "Index not found"
Build the index first with `GraphEnhancedIndexer.build_index()`.

### "vLLM server not responding"
1. Check server is running: `curl http://localhost:8000/health`
2. Verify model loaded: `curl http://localhost:8000/v1/models`
3. Check logs for OOM errors (reduce `--gpu-memory-utilization`)

### "Port already in use"
The launcher checks this automatically. If using vLLM directly, change the port or kill the existing process.

### Import errors
Make sure all dependencies are installed: `pip install -r requirements.txt`

## License

MIT

## Credits

Built with:
- [LlamaIndex](https://www.llamaindex.ai/) - RAG framework
- [vLLM](https://github.com/vllm-project/vllm) - Fast LLM inference
- [Gradio](https://www.gradio.app/) - ML web interfaces
- [tree-sitter](https://tree-sitter.github.io/tree-sitter/) - Multi-language parsing
