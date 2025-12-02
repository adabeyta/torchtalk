import pytest
from pathlib import Path
from unittest.mock import patch, Mock


def test_conversation_engine_init_missing_index():
    """Test that ConversationEngine raises error for missing index"""
    from torchtalk.engine import ConversationEngine

    with pytest.raises(FileNotFoundError):
        ConversationEngine(
            index_path="/nonexistent/path", vllm_server="http://localhost:8000"
        )


@pytest.mark.integration
def test_conversation_engine_init(mock_index_path, vllm_server_url):
    """Test ConversationEngine initialization (requires index + vLLM server)"""
    from torchtalk.engine import ConversationEngine

    # Skip if index doesn't exist
    if not mock_index_path.exists():
        pytest.skip(f"Index not found at {mock_index_path}")

    ConversationEngine(index_path=str(mock_index_path), vllm_server=vllm_server_url)
    # If we get here without exception, initialization succeeded


@pytest.mark.integration
def test_conversation_basic_chat(mock_index_path, vllm_server_url):
    """Test basic chat functionality (requires index + vLLM server)"""
    from torchtalk.engine import ConversationEngine

    if not mock_index_path.exists():
        pytest.skip(f"Index not found at {mock_index_path}")

    engine = ConversationEngine(
        index_path=str(mock_index_path), vllm_server=vllm_server_url
    )

    response = engine.chat("What is PyTorch?")
    assert isinstance(response, str)
    assert len(response) > 0


@pytest.mark.integration
def test_conversation_follow_up(mock_index_path, vllm_server_url):
    """Test follow-up question handling (requires index + vLLM server)"""
    from torchtalk.engine import ConversationEngine

    if not mock_index_path.exists():
        pytest.skip(f"Index not found at {mock_index_path}")

    engine = ConversationEngine(
        index_path=str(mock_index_path), vllm_server=vllm_server_url
    )

    # First question
    response1 = engine.chat("What is torch.nn.Module?")
    assert isinstance(response1, str)

    # Follow-up (should understand context)
    response2 = engine.chat("Can you explain more about that?")
    assert isinstance(response2, str)
    assert len(response2) > 0

    # Check that history is maintained
    history = engine.get_chat_history()
    assert len(history) >= 2


@pytest.mark.integration
def test_conversation_memory_reset(mock_index_path, vllm_server_url):
    """Test conversation memory reset (requires index + vLLM server)"""
    from torchtalk.engine import ConversationEngine

    if not mock_index_path.exists():
        pytest.skip(f"Index not found at {mock_index_path}")

    engine = ConversationEngine(
        index_path=str(mock_index_path), vllm_server=vllm_server_url
    )

    # Chat and check history
    engine.chat("Hello")
    history_before = engine.get_chat_history()
    assert len(history_before) > 0

    # Reset and check
    engine.reset()
    history_after = engine.get_chat_history()
    assert len(history_after) == 0


def test_memory_stats():
    """Test memory statistics structure"""
    from torchtalk.engine import ConversationEngine
    from llama_index.core.storage.docstore import SimpleDocumentStore
    from llama_index.core.schema import TextNode

    # Create a mock index with a proper docstore
    mock_docstore = SimpleDocumentStore()
    mock_docstore.add_documents([TextNode(text="test document", id_="test_id")])

    mock_index = Mock()
    mock_index.docstore = mock_docstore

    # Mock the required components
    with (
        patch("torchtalk.engine.conversation_engine.Vllm"),
        patch(
            "torchtalk.engine.conversation_engine.load_index_from_storage",
            return_value=mock_index,
        ),
        patch("torchtalk.engine.conversation_engine.StorageContext"),
        patch("torchtalk.engine.conversation_engine.HuggingFaceEmbedding"),
        patch("torchtalk.engine.conversation_engine.Settings"),
        patch("torchtalk.engine.conversation_engine.Path.exists", return_value=True),
    ):
        engine = ConversationEngine(
            index_path="/mock/path", vllm_server="http://localhost:8000"
        )

        stats = engine.memory_stats
        assert "token_limit" in stats
        assert isinstance(stats["token_limit"], int)


@pytest.mark.integration
def test_cross_language_tracing(mock_index_path, vllm_server_url):
    """Test cross-language tracing capability (requires index + vLLM server)"""
    from torchtalk.engine import ConversationEngine

    if not mock_index_path.exists():
        pytest.skip(f"Index not found at {mock_index_path}")

    engine = ConversationEngine(
        index_path=str(mock_index_path), vllm_server=vllm_server_url
    )

    # Ask a question that requires cross-language knowledge
    response = engine.chat("How does torch.matmul connect to the C++ implementation?")
    assert isinstance(response, str)
    assert len(response) > 0


# Fixtures
@pytest.fixture
def mock_index_path(tmp_path):
    """Path to test index (use tmp_path for unit tests, real path for integration)"""
    import os

    # Check if TEST_INDEX_PATH env is set for integration tests
    test_index = os.getenv("TEST_INDEX_PATH")
    if test_index:
        return Path(test_index)
    return tmp_path / "test_index"


@pytest.fixture
def vllm_server_url():
    """vLLM server URL for integration tests"""
    import os

    return os.getenv("VLLM_SERVER_URL", "http://localhost:8000")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
