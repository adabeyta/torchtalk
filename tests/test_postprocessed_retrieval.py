"""
Tests for PostprocessedRetriever (Phase 1)
"""

import pytest
from pathlib import Path
from unittest.mock import Mock


def test_import():
    """PostprocessedRetriever can be imported"""
    from torchtalk.engine import PostprocessedRetriever
    assert PostprocessedRetriever is not None


def test_init_with_mock():
    """PostprocessedRetriever initializes with mock index"""
    from torchtalk.engine import PostprocessedRetriever

    mock_index = Mock()
    retriever = PostprocessedRetriever(
        index=mock_index,
        similarity_top_k=40,
        rerank_top_n=10,
        similarity_cutoff=0.5,
    )

    assert retriever.retriever is not None
    assert retriever.reranker is not None
    assert retriever.similarity_filter is not None
    assert retriever.reorderer is not None


@pytest.mark.integration
def test_retrieval_with_real_index(test_index_path):
    """PostprocessedRetriever works with real index"""
    from torchtalk.engine import PostprocessedRetriever
    from llama_index.core import load_index_from_storage, StorageContext, Settings
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding

    if not test_index_path.exists():
        pytest.skip(f"Index not found at {test_index_path}")

    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    storage_context = StorageContext.from_defaults(persist_dir=str(test_index_path))
    index = load_index_from_storage(storage_context)

    retriever = PostprocessedRetriever(
        index=index,
        similarity_top_k=40,
        rerank_top_n=10,
        similarity_cutoff=0.5,
    )

    nodes = retriever.retrieve("where is conv2d defined")

    assert len(nodes) <= 10, "Should respect rerank_top_n"
    assert len(nodes) > 0, "Should return results"
    assert all(hasattr(n, 'score') and n.score is not None for n in nodes)


@pytest.mark.integration
def test_postprocessing_returns_valid_scores(test_index_path):
    """Postprocessed nodes have valid scores"""
    from torchtalk.engine import PostprocessedRetriever
    from llama_index.core import load_index_from_storage, StorageContext, Settings
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding

    if not test_index_path.exists():
        pytest.skip(f"Index not found at {test_index_path}")

    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    storage_context = StorageContext.from_defaults(persist_dir=str(test_index_path))
    index = load_index_from_storage(storage_context)

    retriever = PostprocessedRetriever(index=index, similarity_top_k=40, rerank_top_n=10)
    nodes = retriever.retrieve("pytorch autograd")

    assert all(n.score is not None for n in nodes)
    assert len(nodes) <= 10


@pytest.mark.integration
def test_conversation_engine_integration(test_index_path, vllm_server_url):
    """ConversationEngine uses PostprocessedRetriever"""
    from torchtalk.engine import ConversationEngine

    if not test_index_path.exists():
        pytest.skip(f"Index not found at {test_index_path}")

    engine = ConversationEngine(
        index_path=str(test_index_path),
        vllm_server=vllm_server_url
    )

    assert engine.chat_engine is not None

    response = engine.chat("where is conv2d")
    assert isinstance(response, str)
    assert len(response) > 0


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
