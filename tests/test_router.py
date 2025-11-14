"""
Tests for router engine (Phase 2)
"""

import pytest
from pathlib import Path


def test_tool_descriptions_import():
    """Tool descriptions can be imported"""
    from torchtalk.engine.tool_descriptions import (
        LOCATION_TOOL_DESC,
        CALLSTACK_TOOL_DESC,
        GENERAL_TOOL_DESC,
    )
    assert LOCATION_TOOL_DESC
    assert CALLSTACK_TOOL_DESC
    assert GENERAL_TOOL_DESC


def test_router_engine_import():
    """Router engine can be imported"""
    from torchtalk.engine.router_engine import create_router_engine
    assert create_router_engine is not None


@pytest.mark.integration
def test_router_engine_creation(test_index_path, vllm_server_url):
    """Router engine can be created with real index and LLM"""
    from torchtalk.engine.router_engine import create_router_engine
    from torchtalk.engine import ConversationEngine
    from llama_index.core import load_index_from_storage, StorageContext, Settings
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    from llama_index.llms.vllm import Vllm

    if not test_index_path.exists():
        pytest.skip(f"Index not found at {test_index_path}")

    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    storage_context = StorageContext.from_defaults(persist_dir=str(test_index_path))
    index = load_index_from_storage(storage_context)

    llm = Vllm(
        model="meta-llama/llama-4-maverick",
        api_url=vllm_server_url,
        is_chat_model=True,
    )

    router = create_router_engine(index=index, llm=llm)
    assert router is not None


@pytest.mark.integration
def test_conversation_engine_with_router(test_index_path, vllm_server_url):
    """ConversationEngine works with router enabled"""
    from torchtalk.engine import ConversationEngine

    if not test_index_path.exists():
        pytest.skip(f"Index not found at {test_index_path}")

    engine = ConversationEngine(
        index_path=str(test_index_path),
        vllm_server=vllm_server_url,
        use_router=True,
    )

    assert engine.chat_engine is not None


# Fixtures
@pytest.fixture
def test_index_path():
    """Test index path from env or default"""
    import os
    test_index = os.getenv("TEST_INDEX_PATH")
    return Path(test_index) if test_index else Path("./index")


@pytest.fixture
def vllm_server_url():
    """vLLM server URL from env or default"""
    import os
    return os.getenv("VLLM_SERVER_URL", "http://localhost:8000")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
