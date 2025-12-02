"""
Integration tests for end-to-end TorchTalk functionality.

These tests require:
- Built index at TEST_INDEX_PATH
- Running vLLM server at VLLM_SERVER_URL
"""

import pytest
import os
from pathlib import Path


@pytest.mark.integration
@pytest.mark.slow
def test_cross_language_matmul_question(conversation_engine):
    """Test cross-language tracing question about torch.matmul"""
    response = conversation_engine.chat(
        "How does torch.matmul connect to the C++ implementation?"
    )

    assert isinstance(response, str)
    assert len(response) > 100  # Substantive response
    # Should mention cross-language concepts
    keywords = ["cpp", "c++", "python", "implementation", "bind"]
    assert any(keyword.lower() in response.lower() for keyword in keywords)


@pytest.mark.integration
@pytest.mark.slow
def test_follow_up_question(conversation_engine):
    """Test that follow-up questions maintain context"""
    # First question
    response1 = conversation_engine.chat("What is torch.nn.Module?")
    assert len(response1) > 50

    # Follow-up without explicit context
    response2 = conversation_engine.chat("Can you show me an example of how to use it?")
    assert len(response2) > 50

    # Verify history was maintained
    history = conversation_engine.get_chat_history()
    assert len(history) >= 4  # 2 user messages + 2 assistant responses


@pytest.mark.integration
@pytest.mark.slow
def test_cuda_kernel_question(conversation_engine):
    """Test question about CUDA kernels"""
    response = conversation_engine.chat(
        "Explain how CUDA kernels are used in PyTorch for GPU operations"
    )

    assert isinstance(response, str)
    assert len(response) > 100
    keywords = ["cuda", "kernel", "gpu", "device"]
    assert any(keyword.lower() in response.lower() for keyword in keywords)


@pytest.mark.integration
@pytest.mark.slow
def test_autograd_question(conversation_engine):
    """Test question about autograd engine"""
    response = conversation_engine.chat(
        "How does PyTorch's autograd engine compute gradients?"
    )

    assert isinstance(response, str)
    assert len(response) > 100
    keywords = ["autograd", "gradient", "backward", "computation"]
    assert any(keyword.lower() in response.lower() for keyword in keywords)


@pytest.mark.integration
@pytest.mark.slow
def test_binding_detection_question(conversation_engine):
    """Test that binding information is used in responses"""
    response = conversation_engine.chat(
        "Show me how Python tensor operations bind to C++ implementations"
    )

    assert isinstance(response, str)
    assert len(response) > 100
    # Should reference bindings or cross-language connections
    keywords = ["binding", "pybind", "c++", "python", "aten"]
    assert any(keyword.lower() in response.lower() for keyword in keywords)


@pytest.mark.integration
def test_memory_reset(conversation_engine):
    """Test conversation memory reset"""
    # Chat and verify history exists
    conversation_engine.chat("Hello")
    history_before = conversation_engine.get_chat_history()
    assert len(history_before) > 0

    # Reset
    conversation_engine.reset()
    history_after = conversation_engine.get_chat_history()
    assert len(history_after) == 0


@pytest.mark.integration
def test_memory_stats(conversation_engine):
    """Test memory statistics"""
    stats = conversation_engine.memory_stats
    assert "token_limit" in stats
    assert stats["token_limit"] > 0


# Fixtures
@pytest.fixture(scope="module")
def conversation_engine():
    """Create conversation engine for integration tests"""
    from torchtalk.engine import ConversationEngine

    index_path = os.getenv("TEST_INDEX_PATH")
    vllm_url = os.getenv("VLLM_SERVER_URL", "http://localhost:8000")

    if not index_path or not Path(index_path).exists():
        pytest.skip("TEST_INDEX_PATH not set or index not found")

    engine = ConversationEngine(
        index_path=index_path,
        vllm_server=vllm_url,
    )

    yield engine

    # Cleanup
    engine.reset()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "integration"])
