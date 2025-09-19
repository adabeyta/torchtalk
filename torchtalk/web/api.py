from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
from torchtalk.core.context_manager import ContextManager
from torchtalk.context.assembler import DynamicContextAssembler
from torchtalk.core.adaptive_response import AdaptiveResponseManager
from torchtalk.core.config import get_config

# Load unified configuration
config = get_config()
context_manager = ContextManager(config.context_profile)

# Initialize context assembler and adaptive response manager
context_assembler = DynamicContextAssembler()
adaptive_manager = AdaptiveResponseManager(context_assembler)

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
        "context_profile": config.context_profile,
        "repo": config.repo_name,
        "model": config.model_name,
    }


@app.post("/chat")
def chat(request: ChatRequest):
    try:
        # Analyze question and get appropriate context
        analysis = adaptive_manager.analyze_question(
            request.message, max_context_chars=context_manager.budget.compendium_chars
        )

        # Prepare the full prompt
        system_prompt = analysis["system_prompt"]
        context = analysis["context"]

        # Build messages for vLLM
        messages = [{"role": "system", "content": system_prompt}]

        # Add context if available
        if context:
            messages.append(
                {
                    "role": "system",
                    "content": f"Context:\n{context}",
                }
            )

        # Add chat history
        for msg in request.chat_history[-10:]:  # Limit history
            messages.append(msg)

        # Add current user message
        messages.append({"role": "user", "content": request.message})

        # Call vLLM for response generation
        vllm_request = {
            "model": config.model_name,
            "messages": messages,
            "max_tokens": analysis.get("max_tokens", 300),
            "temperature": analysis.get("temperature", 0.3),
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
                context_info=analysis.get("context_info", {}),
                response_style=analysis.get("response_style", "unknown"),
                relevance_score=analysis.get("relevance_metrics", {}).get(
                    "max_relevance", 0.0
                ),
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
            detail=f"System not ready: {str(e)}. Please run enhanced_analyzer.py first.",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@app.get("/config")
def get_configuration():
    return {
        "context_profile": config.context_profile,
        "repo": config.repo_name,
        "model": config.model_name,
        "vllm_endpoint": config.vllm_endpoint,
        "context_budget": {
            "total": context_manager.budget.total_context,
            "compendium": context_manager.budget.compendium_chars,
            "history": context_manager.budget.chat_history,
            "response": context_manager.budget.response_generation,
        },
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=config.fastapi_port)