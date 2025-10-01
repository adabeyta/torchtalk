from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
from pathlib import Path
from torchtalk.retrieval.hybrid_retriever import HybridRetriever
from torchtalk.core.config import get_config

# Load unified configuration
config = get_config()

# Initialize v2.1 hybrid retriever with adaptive context (required)
hybrid_retriever = None
if hasattr(config, 'index_dir') and config.index_dir:
    index_path = Path(config.index_dir)
    if index_path.exists():
        try:
            # Get max_model_len from config
            max_model_len = config.max_model_len if hasattr(config, 'max_model_len') and config.max_model_len else 131072

            hybrid_retriever = HybridRetriever(
                index_dir=str(index_path),
                collection_name="torchtalk_code",
                use_reranking=True,
                use_graph_augmentation=True,
                max_model_len=max_model_len
            )
            print(f" HybridRetriever initialized with index: {index_path}")
            print(f" Dynamic context assembly enabled (max_model_len: {max_model_len:,})")
        except Exception as e:
            print(f" Failed to initialize HybridRetriever: {e}")
            print("  Please run 'torchtalk index <repo>' to build the index")
            hybrid_retriever = None
else:
    print(" No index configured")
    print("  Run 'torchtalk index <repo>' to enable TorchTalk v2.1")

if hybrid_retriever is None:
    print("\n WARNING: TorchTalk API requires a v2.1 index to function")
    print("  The /chat endpoint will return errors until an index is built")

app = FastAPI(title="TorchTalk POC")


class ChatRequest(BaseModel):
    message: str
    chat_history: list = []


class ChatResponse(BaseModel):
    response: str
    context_info: dict = None
    response_style: str = None
    relevance_score: float = 0.0


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "max_model_len": config.max_model_len if hasattr(config, 'max_model_len') else None,
        "repo": config.repo_name,
        "model": config.model_name,
        "version": "v2.1-adaptive"
    }


@app.post("/chat")
def chat(request: ChatRequest):
    try:
        # Require v2.1 HybridRetriever
        if not hybrid_retriever:
            raise HTTPException(
                status_code=503,
                detail="No index available. Please run 'torchtalk index <repo>' to build the semantic search index."
            )

        # V2.1: Adaptive retrieval with dynamic context assembly
        adaptive_result = hybrid_retriever.retrieve_adaptive(
            query=request.message,
            user_preferences=None  # Can add user overrides here
        )

        context_result = adaptive_result['context']
        query_analysis = adaptive_result['query_analysis']

        # Use the pre-formatted context from adaptive builder
        context = context_result.context_text

        # Modern temperature settings for natural responses
        max_tokens = 4000
        temperature = 0.8
        response_style = query_analysis['type']

        # Enhanced system prompt based on query type
        if response_style == 'location':
            system_prompt = """You are a PyTorch codebase expert. The user is asking WHERE code is located.

Focus on providing:
1. Exact file paths with line numbers (use the Location field from context)
2. Brief description of what's at that location
3. Any alternative locations if multiple definitions exist

Be concise and prioritize file:line references over code explanations."""

        elif response_style == 'call_stack':
            system_prompt = """You are a PyTorch codebase expert. The user is asking about CALL STACK or execution flow.

Focus on providing:
1. Step-by-step execution flow with file:line references at each step
2. Function call chains from high-level API down to implementation
3. Cross-language transitions (Python → C++ → CUDA) when relevant
4. Use the "Related Code" depth information to show call hierarchy

Format as a numbered list showing the execution path through the codebase."""

        else:
            # Default prompt for other query types
            system_prompt = """You are a PyTorch codebase expert. Answer questions by directly analyzing the provided source code.

Include specific file:line references (format: `file.py:123-145`), quote relevant code snippets, and trace implementation details across files when needed. Be direct and technical."""

        context_info = {
            'retrieval_method': 'adaptive_v2.1',
            'query_type': query_analysis['type'],
            'query_complexity': query_analysis['complexity'],
            'chunks_included': context_result.chunks_included,
            'tokens_used': context_result.tokens_used,
            'quality_metrics': context_result.quality_metrics,
            'truncated': context_result.truncated
        }

        relevance_score = context_result.quality_metrics.get('avg_score', 0.0)

        # Build messages - context in user message (modern best practice)
        messages = [{"role": "system", "content": system_prompt}]

        # Add chat history
        for msg in request.chat_history[-10:]:
            messages.append(msg)

        # Add user message with context embedded (better than multiple system messages)
        if context:
            user_message = f"""{request.message}

<context>
{context}
</context>"""
        else:
            user_message = request.message

        messages.append({"role": "user", "content": user_message})

        # Call vLLM for response generation
        vllm_request = {
            "model": config.model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": 0.95,
            "stream": False,
        }

        vllm_response = requests.post(config.vllm_endpoint, json=vllm_request)

        if vllm_response.status_code == 200:
            vllm_data = vllm_response.json()
            response_text = vllm_data["choices"][0]["message"]["content"]

            # Return response with metadata
            return ChatResponse(
                response=response_text,
                context_info=context_info,
                response_style=response_style,
                relevance_score=relevance_score,
            )
        else:
            raise HTTPException(
                status_code=vllm_response.status_code,
                detail=f"vLLM error: {vllm_response.text}",
            )

    except HTTPException:
        raise
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=503,
            detail=f"System not ready: {str(e)}. Please run 'torchtalk index <repo>' first.",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@app.get("/config")
def get_configuration():
    max_model_len = config.max_model_len if hasattr(config, 'max_model_len') and config.max_model_len else None

    config_data = {
        "max_model_len": max_model_len,
        "repo": config.repo_name,
        "model": config.model_name,
        "vllm_endpoint": config.vllm_endpoint,
        "retrieval_version": "v2.1-adaptive" if hybrid_retriever else "none",
    }

    # Add v2.1 index stats if available
    if hybrid_retriever:
        try:
            stats = hybrid_retriever.get_stats()
            config_data["index_stats"] = {
                "index_dir": stats.get('index_dir'),
                "total_documents": stats.get('total_documents', 0),
                "files_indexed": stats.get('total_files', 0),
                "embedding_model": stats.get('embedding_model', 'unknown'),
                "use_reranking": stats['hybrid_config']['use_reranking'],
                "use_graph_augmentation": stats['hybrid_config']['use_graph_augmentation'],
                "adaptive_context": stats['hybrid_config'].get('adaptive_context', False),
                "max_model_len": stats['hybrid_config'].get('max_model_len')
            }
        except Exception:
            pass

    return config_data


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=config.fastapi_port)