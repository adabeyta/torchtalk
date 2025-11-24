import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
import openai

from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
    Settings,
)
from llama_index.core.chat_engine import CondensePlusContextChatEngine
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.base.llms.types import CompletionResponse
from llama_index.llms.vllm import Vllm
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

log = logging.getLogger(__name__)


class ConversationEngine:
    """
    Conversation engine with graph-enhanced retrieval and automatic follow-up handling.

    Uses LlamaIndex's CondensePlusContextChatEngine for:
    - Automatic conversation memory (ChatMemoryBuffer)
    - Query condensation for follow-ups
    - Context retrieval from graph-enhanced index
    """

    def __init__(
        self,
        index_path: str,
        vllm_server: str = "http://localhost:8000",
        model_name: str = "meta-llama/llama-4-maverick",
        context_window: int = 1000000,
        memory_token_limit: int = 3000,
        similarity_top_k: int = 40,
        system_prompt: Optional[str] = None,
        served_model_name: Optional[str] = None,
        rerank_top_n: Optional[int] = None,
        similarity_cutoff: float = 0.5,
        rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ):
        """
        Initialize conversation engine.

        Args:
            index_path: Path to persisted LlamaIndex storage
            vllm_server: vLLM server URL
            model_name: Model name for vLLM
            context_window: Model's max context length (default: 1000000)
            memory_token_limit: Max tokens to keep in conversation memory
            similarity_top_k: Initial retrieval count (default: 40)
            system_prompt: Optional system prompt for the chat
            served_model_name: Name under which the model is served (defaults to "torchtalk-maverick")
            rerank_top_n: Final count after reranking (auto-adjusted based on context_window if None)
            similarity_cutoff: Minimum similarity score threshold (default: 0.5)
            rerank_model: Cross-encoder model for reranking (default: ms-marco-MiniLM-L-6-v2)
        """
        self.index_path = Path(index_path)
        self.vllm_server = vllm_server
        self.model_name = model_name
        self.served_model_name = served_model_name or "torchtalk-maverick"
        self.context_window = context_window
        self.memory_token_limit = memory_token_limit
        self.similarity_top_k = similarity_top_k
        self.similarity_cutoff = similarity_cutoff
        self.rerank_model = rerank_model

        if rerank_top_n is None:
            if context_window <= 8192:
                self.rerank_top_n = 2
                log.info(f"Small context window ({context_window}), using rerank_top_n=2")
            elif context_window <= 32768:
                self.rerank_top_n = 5
                log.info(f"Medium context window ({context_window}), using rerank_top_n=5")
            elif context_window <= 131072:
                self.rerank_top_n = 7
                log.info(f"Large context window ({context_window}), using rerank_top_n=7")
            else:
                self.rerank_top_n = 10
                log.info(f"Very large context window ({context_window}), using rerank_top_n=10")
        else:
            self.rerank_top_n = rerank_top_n

        # Default system prompt
        self.system_prompt = system_prompt or (
            "You are a PyTorch codebase expert assistant. "
            "You have access to the full PyTorch source code including Python, C++, and CUDA implementations. "
            "When answering questions:\n"
            "1. ALWAYS include specific file paths from the retrieved context\n"
            "2. Include relevant code snippets with line numbers when available\n"
            "3. Trace cross-language calls: Python → C++ → CUDA when applicable\n"
            "4. Reference the actual source files, not just general descriptions\n"
            "Base your answers on the retrieved code context provided to you."
        )

        # Initialize LLM with vLLM endpoint
        log.info(f"Connecting to vLLM server at {vllm_server}...")

        # Set context window for LlamaIndex
        Settings.context_window = context_window

        self.llm = Vllm(
            model=model_name,
            temperature=0.1,
            max_new_tokens=2048,
            api_url=vllm_server,
            is_chat_model=True,
        )

        # Patch the Vllm class to support API mode (bug in llama-index-llms-vllm 0.6.1)
        self._patch_vllm_api_mode()

        # Load index
        log.info(f"Loading index from {index_path}...")
        self.index = self._load_index()

        # Initialize chat engine
        log.info("Initializing chat engine with conversation memory...")
        self.chat_engine = self._create_chat_engine()

        log.info("Conversation engine ready")

    def _patch_vllm_api_mode(self):
        """
        Patch the Vllm instance to support API mode.

        WORKAROUND: This fixes a bug in llama-index-llms-vllm 0.6.1 where api_url mode
        sets _client to None but then tries to call _client.generate(), causing
        AttributeError when using vLLM via HTTP API.

        This monkey-patch replaces Vllm.complete() at the class level to use the
        OpenAI-compatible API that vLLM provides. This is necessary until the
        llama-index-llms-vllm library properly supports API mode.

        Related issue: The Vllm class was designed for in-process vLLM usage and
        doesn't properly handle the api_url parameter for remote vLLM servers.
        """
        # Create OpenAI client for the vLLM API endpoint
        api_base = self.vllm_server.rstrip('/') + '/v1'
        openai_client = openai.OpenAI(
            api_key="dummy",  # vLLM doesn't require authentication
            base_url=api_base,
        )

        # Store served model name for the patched method
        served_model_name = self.served_model_name

        # Create patched complete method
        def patched_complete(llm_self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponse:
            """Patched complete method that uses OpenAI API for vLLM server."""
            # Use OpenAI API to call vLLM server with the served model name
            response = openai_client.chat.completions.create(
                model=served_model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=kwargs.get("temperature", llm_self.temperature),
                max_tokens=kwargs.get("max_tokens", llm_self.max_new_tokens),
                top_p=kwargs.get("top_p", llm_self.top_p),
                frequency_penalty=kwargs.get("frequency_penalty", llm_self.frequency_penalty),
                presence_penalty=kwargs.get("presence_penalty", llm_self.presence_penalty),
                stop=kwargs.get("stop", llm_self.stop),
            )
            return CompletionResponse(text=response.choices[0].message.content)

        # Monkey patch at the class level to work with Pydantic
        Vllm.complete = patched_complete
        log.info("Patched Vllm for API mode support")

    def _load_index(self) -> VectorStoreIndex:
        """Load persisted index with progress indicator"""
        if not self.index_path.exists():
            raise FileNotFoundError(f"Index not found at {self.index_path}")

        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
        ) as progress:
            # Step 1: Initialize embedding model
            task1 = progress.add_task("Loading embedding model", total=100)

            embed_model = HuggingFaceEmbedding(
                model_name="BAAI/bge-small-en-v1.5"
            )
            Settings.embed_model = embed_model
            progress.update(task1, completed=100)

            # Step 2: Load storage context
            task2 = progress.add_task("Loading storage context", total=100)

            storage_context = StorageContext.from_defaults(
                persist_dir=str(self.index_path)
            )
            progress.update(task2, completed=100)

            # Step 3: Load index from storage
            task3 = progress.add_task("Loading vector index", total=100)

            index = load_index_from_storage(storage_context)
            progress.update(task3, completed=100)

        return index

    def _create_chat_engine(self) -> CondensePlusContextChatEngine:
        """Create chat engine with memory"""
        from torchtalk.engine.postprocessed_retriever import PostprocessedRetriever

        # Create memory buffer
        memory = ChatMemoryBuffer.from_defaults(
            token_limit=self.memory_token_limit,
        )

        retriever = PostprocessedRetriever(
            index=self.index,
            similarity_top_k=self.similarity_top_k,
            rerank_top_n=self.rerank_top_n,
            similarity_cutoff=self.similarity_cutoff,
            rerank_model=self.rerank_model,
        )

        chat_engine = CondensePlusContextChatEngine.from_defaults(
            retriever=retriever,
            llm=self.llm,
            memory=memory,
            system_prompt=self.system_prompt,
            verbose=True,
        )

        return chat_engine

    def chat(self, message: str) -> str:
        """
        Send a message and get a response.

        The engine automatically:
        - Maintains conversation history
        - Condenses follow-up questions using context
        - Retrieves relevant code snippets
        - Generates informed responses

        Args:
            message: User message

        Returns:
            Assistant response
        """
        response = self.chat_engine.chat(message)
        log.info(f"LlamaIndex response object type: {type(response)}")
        log.info(f"LlamaIndex response has 'response' attr: {hasattr(response, 'response')}")

        # Extract the actual response text from the response object for LlamaIndex
        if hasattr(response, 'response'):
            response_text = response.response
            log.info(f"Extracted response.response: '{response_text}' (length: {len(str(response_text))})")
            return response_text
        else:
            response_text = str(response)
            log.info(f"Converted to str: '{response_text}' (length: {len(response_text)})")
            return response_text

    def reset(self):
        """Reset conversation memory"""
        log.info("Resetting conversation memory...")
        self.chat_engine.reset()

    def get_chat_history(self) -> List[Dict[str, Any]]:
        """Get current chat history"""
        return self.chat_engine.chat_history

    @property
    def memory_stats(self) -> Dict[str, Any]:
        """Get memory statistics"""
        memory = self.chat_engine._memory
        return {
            "token_limit": memory.token_limit,
            "current_tokens": len(memory.get_all()),  # Rough estimate
        }
