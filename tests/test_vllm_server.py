import pytest
import socket
import os
from pathlib import Path
import sys

# Add scripts to path for importing
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

# Test configuration
MODEL = os.getenv("TEST_VLLM_MODEL", "meta-llama/llama-4-maverick")


def test_can_bind_function():
    """Test the can_bind helper function"""
    from start_vllm_server import can_bind

    # Test with ephemeral port (free port)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        free_port = s.getsockname()[1]
    assert can_bind("127.0.0.1", free_port) is True

    # Test with in-use port
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    used_port = s.getsockname()[1]
    s.listen(1)
    try:
        assert can_bind("127.0.0.1", used_port) is False
    finally:
        s.close()


def test_vllm_import():
    """Test that vLLM can be imported"""
    try:
        import vllm
        assert vllm.__version__ is not None
    except ImportError:
        pytest.skip("vLLM not installed")


@pytest.mark.skipif(not Path("/proc/driver/nvidia/version").exists(),
                    reason="No NVIDIA GPU detected")
def test_nvidia_smi():
    """Test that nvidia-smi is available"""
    import subprocess
    result = subprocess.run(["nvidia-smi"], capture_output=True)
    assert result.returncode == 0


@pytest.mark.integration
@pytest.mark.skipif(not Path("/proc/driver/nvidia/version").exists(),
                    reason="No NVIDIA GPU detected")
def test_vllm_server_health(vllm_server_url):
    """Test vLLM server health endpoint (requires running server)"""
    requests = pytest.importorskip("requests")

    try:
        response = requests.get(f"{vllm_server_url}/health", timeout=5)
        assert response.status_code == 200
    except requests.exceptions.ConnectionError:
        pytest.skip(f"vLLM server not running at {vllm_server_url}")


@pytest.mark.integration
@pytest.mark.skipif(not Path("/proc/driver/nvidia/version").exists(),
                    reason="No NVIDIA GPU detected")
def test_vllm_models_endpoint(vllm_server_url):
    """Test vLLM /v1/models endpoint (requires running server)"""
    requests = pytest.importorskip("requests")

    try:
        response = requests.get(f"{vllm_server_url}/v1/models", timeout=5)
        assert response.status_code == 200
        data = response.json()
        assert "data" in data
        assert len(data["data"]) > 0
    except requests.exceptions.ConnectionError:
        pytest.skip(f"vLLM server not running at {vllm_server_url}")


@pytest.mark.integration
@pytest.mark.skipif(not Path("/proc/driver/nvidia/version").exists(),
                    reason="No NVIDIA GPU detected")
def test_vllm_basic_inference(vllm_server_url):
    """Test basic inference with short prompt (requires running server)"""
    requests = pytest.importorskip("requests")

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "user", "content": "What is 2+2?"}
        ],
        "max_tokens": 50,
        "temperature": 0.0
    }

    try:
        response = requests.post(
            f"{vllm_server_url}/v1/chat/completions",
            json=payload,
            timeout=30
        )
        assert response.status_code == 200
        data = response.json()
        assert "choices" in data
        assert len(data["choices"]) > 0
        assert "message" in data["choices"][0]
        assert "content" in data["choices"][0]["message"]
    except requests.exceptions.ConnectionError:
        pytest.skip(f"vLLM server not running at {vllm_server_url}")


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.skipif(not Path("/proc/driver/nvidia/version").exists(),
                    reason="No NVIDIA GPU detected")
def test_vllm_long_context(vllm_server_url):
    """Test long context handling (~10k tokens) (requires running server)"""
    requests = pytest.importorskip("requests")

    # Generate a ~10k token prompt (rough estimate: 1 token ≈ 4 chars)
    long_text = "The following is a very long document. " * 1000

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "user", "content": f"{long_text}\n\nSummarize the above in one word."}
        ],
        "max_tokens": 10,
        "temperature": 0.0
    }

    try:
        response = requests.post(
            f"{vllm_server_url}/v1/chat/completions",
            json=payload,
            timeout=60
        )
        assert response.status_code == 200
        data = response.json()
        assert "choices" in data
        assert len(data["choices"]) > 0
    except requests.exceptions.ConnectionError:
        pytest.skip(f"vLLM server not running at {vllm_server_url}")


# Fixtures
@pytest.fixture
def vllm_server_url():
    """vLLM server URL (default: http://localhost:8000)"""
    return os.getenv("VLLM_SERVER_URL", "http://localhost:8000")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
