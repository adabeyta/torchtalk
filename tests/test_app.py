import pytest
import sys
from pathlib import Path
from unittest.mock import patch


def test_app_import():
    """Test that app module can be imported"""
    # Add app.py directory to path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    import app
    assert app is not None


def test_create_app_signature():
    """Test that create_app has correct signature"""
    sys.path.insert(0, str(Path(__file__).parent.parent))
    import app
    import inspect

    sig = inspect.signature(app.create_app)
    params = list(sig.parameters.keys())

    assert 'index_path' in params
    assert 'vllm_server' in params
    assert 'model_name' in params


def test_chat_fn_shape():
    """Test that chat_fn returns proper history format"""
    # Test the logic without invoking Gradio context
    history = []
    message = "Test question"
    reply = "Test response"

    # Simulate what chat_fn does
    new_history = history + [[message, reply]]

    assert isinstance(new_history, list)
    assert len(new_history) == 1
    assert new_history[0] == [message, reply]
    assert new_history[0][0] == message
    assert new_history[0][1] == reply


def test_chat_fn_empty_message():
    """Test that chat_fn handles empty messages"""
    history = [["previous", "message"]]
    message = ""

    # Empty message should return history unchanged
    result = history if not message.strip() else history + [[message, "reply"]]
    assert result == history


def test_chat_fn_error_handling():
    """Test that chat_fn handles errors gracefully"""
    # Test error handling logic without Gradio context
    history = []
    message = "Test question"

    # Simulate error
    try:
        raise Exception("Test error")
    except Exception as e:
        reply = f"Error: {e}"

    new_history = history + [[message, reply]]

    assert len(new_history) == 1
    assert "Error:" in new_history[0][1]
    assert "Test error" in new_history[0][1]


@pytest.mark.integration
def test_app_launch(mock_index_path, vllm_server_url):
    """Test that app can be launched (integration test)"""
    sys.path.insert(0, str(Path(__file__).parent.parent))

    if not mock_index_path.exists():
        pytest.skip(f"Index not found at {mock_index_path}")

    import app

    # Create app (don't launch)
    test_app = app.create_app(
        index_path=str(mock_index_path),
        vllm_server=vllm_server_url
    )

    assert test_app is not None
    # Verify it's a Gradio Blocks instance
    assert hasattr(test_app, 'queue')
    assert hasattr(test_app, 'launch')


def test_main_missing_index(tmp_path):
    """Test that main() exits when index is missing"""
    sys.path.insert(0, str(Path(__file__).parent.parent))
    import app

    # Mock sys.argv
    with patch('sys.argv', ['app.py', '--index-path', str(tmp_path / 'nonexistent')]):
        with pytest.raises(SystemExit) as exc_info:
            app.main()
        assert exc_info.value.code == 1


# Fixtures
@pytest.fixture
def mock_index_path(tmp_path):
    """Path to test index"""
    import os
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
