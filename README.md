# TorchTalk

A PyTorch codebase chatbot with cross-language tracing (Python ↔ C++ ↔ CUDA) powered by graph-enhanced retrieval and 1M context windows.

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

# Configure GPU usage
torchtalk chat --index ./index --tp 2 --gpu-util 0.85 --cuda-devices "0,1"

# Adjust context length
torchtalk chat --index ./index --max-len 500000

# Create a public share link
torchtalk chat --index ./index --share

# Start vLLM server only (without UI)
torchtalk serve-vllm --model meta-llama/llama-4-maverick --port 8000

# See all available options
torchtalk chat --help
torchtalk serve-vllm --help
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                       Indexing                               │
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
│                    vLLM Server                               │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ vLLM (Llama 4 Maverick, 1M context)                 │   │
│  │ - KV cache paging                                    │   │
│  │ - Prefix caching                                     │   │
│  │ - Chunked prefill                                    │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│              Conversation Engine                             │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ CondensePlusContextChatEngine (LlamaIndex)          │   │
│  │ - Automatic query condensation for follow-ups       │   │
│  │ - ChatMemoryBuffer for conversation state           │   │
│  │ - Retrieval from graph-enhanced index               │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                      Gradio UI                               │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ Minimal chat interface with:                        │   │
│  │ - Message history                                    │   │
│  │ - Example questions                                  │   │
│  │ - Clear/reset button                                 │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

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
│   ├── engine/
│   │   └── conversation_engine.py  # LlamaIndex chat engine
│   └── indexing/
│       └── graph_enhanced_indexer.py  # Graph metadata injection
└── tests/                      # Unit and integration tests
```

## Credits

Built with:
- [LlamaIndex](https://www.llamaindex.ai/) - Retrieval framework
- [vLLM](https://github.com/vllm-project/vllm) - Fast LLM inference
- [Gradio](https://www.gradio.app/) - ML web interfaces
- [tree-sitter](https://tree-sitter.github.io/tree-sitter/) - Multi-language parsing
