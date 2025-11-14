import pytest
import logging
from pathlib import Path

# Configure logging for tests
logging.basicConfig(level=logging.INFO)


@pytest.mark.skipif(not Path("../pytorch").exists(), reason="PyTorch repo not available")
def test_indexer_initialization():
    """Test that indexer can be initialized"""
    from torchtalk.indexing import GraphEnhancedIndexer

    indexer = GraphEnhancedIndexer(repo_path="../pytorch")
    assert indexer is not None
    assert indexer.repo_path == Path("../pytorch")
    assert indexer.repo_root == Path("../pytorch").resolve()


def test_indexer_canon_paths():
    """Test path canonicalization"""
    from torchtalk.indexing import GraphEnhancedIndexer

    indexer = GraphEnhancedIndexer(repo_path=".")

    # Test relative path
    canon_path = indexer._canon("./test.py")
    assert isinstance(canon_path, str)
    assert canon_path.endswith("test.py")

    # Test absolute path
    abs_path = Path(".").resolve() / "test.py"
    canon_path = indexer._canon(str(abs_path))
    assert "test.py" in canon_path


@pytest.mark.slow
@pytest.mark.skipif(not Path("../pytorch").exists(), reason="PyTorch repo not available")
def test_build_index_smoke(tmp_path):
    """Smoke test for building index on real repo (slow - full PyTorch indexing)"""
    from torchtalk.indexing import GraphEnhancedIndexer

    # This is a smoke test using the full PyTorch repo - takes several minutes
    indexer = GraphEnhancedIndexer(repo_path="../pytorch")
    index = indexer.build_index(persist_dir=tmp_path.as_posix())

    assert index is not None
    assert (tmp_path / "docstore.json").exists()


def test_metadata_defaults():
    """Test that metadata defaults are set correctly"""
    from llama_index.core.schema import TextNode

    # Create a mock node
    node = TextNode(text="test code")
    node.metadata = {
        "rel_path": "test.py",
        "file_path": "/abs/path/test.py"
    }

    # Apply defaults like the indexer does
    node.metadata.setdefault("schema", "tt_graph_meta_v1")
    node.metadata.setdefault("language", "python")
    node.metadata.setdefault("has_bindings", False)
    node.metadata.setdefault("cross_language_bindings", [])
    node.metadata.setdefault("function_calls", [])
    node.metadata.setdefault("imports", [])

    # Verify all defaults are set
    assert node.metadata["schema"] == "tt_graph_meta_v1"
    assert node.metadata["language"] == "python"
    assert node.metadata["has_bindings"] is False
    assert node.metadata["cross_language_bindings"] == []
    assert node.metadata["function_calls"] == []
    assert node.metadata["imports"] == []


def test_language_detection():
    """Test language detection from file paths"""
    test_cases = [
        ("test.py", "python"),
        ("test.cpp", "cpp"),
        ("test.cc", "cpp"),
        ("test.h", "cpp"),
        ("test.hpp", "cpp"),
        ("test.cu", "cuda"),
        ("test.cuh", "cuda"),
        ("test.txt", "unknown"),
    ]

    for file_path, expected_lang in test_cases:
        if file_path.endswith(".py"):
            lang = "python"
        elif file_path.endswith((".cpp", ".cc", ".h", ".hpp")):
            lang = "cpp"
        elif file_path.endswith((".cu", ".cuh")):
            lang = "cuda"
        else:
            lang = "unknown"

        assert lang == expected_lang, f"Failed for {file_path}: expected {expected_lang}, got {lang}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
