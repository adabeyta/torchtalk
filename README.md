# TorchTalk

An intelligent code assistant for PyTorch codebases with semantic search and multi-language support (Python/C++/CUDA). Uses LlamaIndex for code understanding with hybrid retrieval combining vector search, graph augmentation, and re-ranking.

## Features

- **Multi-language support**: Python, C++, and CUDA with cross-language binding detection
- **Hybrid retrieval**: Vector search + graph augmentation + re-ranking for precise results
- **Graph-based analysis**: Import, call, and inheritance graphs using NetworkX
- **Adaptive context building**: Dynamic query classification and context assembly
- **Web UI**: Gradio interface with chat history and streaming responses
- **vLLM integration**: Efficient LLM inference with tensor parallelism support

## Installation

```bash
pip install -e .
```

## Quick Start

```bash
# Index a repository (required first step)
torchtalk index /path/to/pytorch

# Start chat interface with web UI
torchtalk chat /path/to/pytorch

# Use a specific model
torchtalk chat /path/to/pytorch --model meta-llama/Llama-3.1-70B-Instruct

# Use tensor parallelism for large models
torchtalk chat /path/to/pytorch --model meta-llama/Llama-3.1-70B-Instruct --tp 8
```

## Basic Usage

### Indexing
```bash
# Force rebuild existing index
torchtalk index /path/to/pytorch --rebuild

# Check index status
torchtalk status /path/to/pytorch
```

### Chat Commands
The `chat` command automatically starts:
- vLLM server (port 8080)
- FastAPI backend (port 8001)
- Gradio UI (port 7860)

A public link will be posted at successful startup.

### Query Types
TorchTalk automatically handles different query types:
- **Location**: "Where is the optimizer implementation?"
- **Call Stack**: "How does backward() call autograd?"
- **Architecture**: "What is the structure of the nn.Module?"
- **Implementation**: "How does DataLoader work?"
- **Explanation**: "What does torch.compile do?"

## Architecture

- **Indexing** (`torchtalk/indexing/`): Embeddings, vector store, index building
- **Analysis** (`torchtalk/analysis/`): Graph construction, chunking, parsing
- **Retrieval** (`torchtalk/retrieval/`): Hybrid search, re-ranking, adaptive context
- **Web** (`torchtalk/web/`): FastAPI + Gradio interfaces

## Configuration

Configuration via `torchtalk_config.json` or environment variables:
- `REPO_PATH`, `MODEL_NAME`, `VLLM_ENDPOINT`
- `FASTAPI_PORT`, `GRADIO_PORT`, `VLLM_PORT`
- `MAX_MODEL_LEN`, `TENSOR_PARALLEL_SIZE`

Indexes are stored in `~/.torchtalk/indexes/<repo-hash>/`
