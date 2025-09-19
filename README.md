# TorchTalk v1.0

An intelligent PyTorch repository analysis and chat system that combines graph-based context retrieval with adaptive response generation.

## Features

- **Graph-based Context Retrieval**: Analyzes import graphs and call graphs to understand code relationships
- **Semantic Search**: Intelligent pattern matching and keyword-based content discovery
- **Adaptive Response System**: Adjusts response style based on question relevance to PyTorch
- **Multi-modal Interfaces**: CLI, REST API, and Gradio web UI
- **Scalable Architecture**: Supports multiple context profiles (4K, 128K, 1M tokens)
- **Automatic Analysis**: Repository analysis runs automatically when needed

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/torchtalk.git
cd torchtalk

# Install dependencies
pip install -r requirements.txt
```

### 2. Start TorchTalk

```bash
# Basic usage
python torchtalk.py start --repo /path/to/pytorch

# With custom model and context size
python torchtalk.py start --repo /path/to/pytorch \
  --context production_128k \
  --model meta-llama/Llama-3.2-8B-Instruct
```

The system will:
1. Automatically analyze the repository if needed
2. Build knowledge graphs and semantic indices
3. Start all services:
   - vLLM server on port 8000
   - FastAPI backend on port 8001
   - Gradio UI on port 7860

### 3. Access the System

- **Web UI**: http://localhost:7860
- **API**: http://localhost:8001/docs
- **Health Check**: http://localhost:8001/health

## Architecture

### Core Components

- **`torchtalk.py`**: Main CLI that orchestrates analysis and services
- **`repo_analyzer.py`**: Analyzes repositories and builds knowledge graphs
- **`graph_context_retriever.py`**: Graph-based context retrieval using NetworkX
- **`semantic_search.py`**: Semantic search and pattern matching
- **`adaptive_response_manager.py`**: Intelligently adjusts response style based on relevance
- **`app.py`**: FastAPI REST service
- **`ui.py`**: Gradio web interface

### Context Profiles

| Profile | Context Window | Use Case |
|---------|---------------|----------|
| `dev` | 4K tokens | Development and testing |
| `production_128k` | 128K tokens | Standard production use |
| `production_1m` | 1M tokens | Large-scale analysis |

## Configuration

Configuration can be managed through:
- Command line arguments
- `torchtalk_config.json` (auto-created)
- Environment variables

## API Usage

The system provides a REST API at http://localhost:8001 with endpoints for:
- `/chat` - Process chat requests
- `/config` - View current configuration
- `/health` - Health check

## Response Styles

The system adapts its response based on question relevance:

- **PyTorch Expert** (score ≥ 8.0): Detailed technical responses with code examples
- **PyTorch Aware** (score ≥ 4.0): General ML/DL responses with PyTorch context
- **Programming Helper** (score ≥ 2.0): General programming assistance
- **Casual Assistant** (score < 2.0): Conversational responses