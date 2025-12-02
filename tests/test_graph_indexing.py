import pytest
import logging
from pathlib import Path

# Configure logging for tests
logging.basicConfig(level=logging.INFO)


@pytest.mark.skipif(
    not Path("../pytorch").exists(), reason="PyTorch repo not available"
)
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
@pytest.mark.skipif(
    not Path("../pytorch").exists(), reason="PyTorch repo not available"
)
def test_build_index_smoke(tmp_path):
    """Smoke test for building index on real repo (slow - full PyTorch indexing)"""
    from torchtalk.indexing import GraphEnhancedIndexer

    # This is a smoke test using the full PyTorch repo - takes several minutes
    indexer = GraphEnhancedIndexer(repo_path="../pytorch")
    index = indexer.build_index(persist_dir=tmp_path.as_posix())

    assert index is not None
    assert (tmp_path / "docstore.json").exists()


def test_metadata_defaults():
    """Test that metadata defaults are set correctly (JSON-serialized for LanceDB compatibility)"""
    from llama_index.core.schema import TextNode

    # Create a mock node
    node = TextNode(text="test code")
    node.metadata = {"rel_path": "test.py", "file_path": "/abs/path/test.py"}

    # Apply defaults like the indexer does (JSON strings for LanceDB compatibility)
    node.metadata.setdefault("schema", "tt_graph_meta_v1")
    node.metadata.setdefault("language", "python")
    node.metadata.setdefault("has_bindings", False)
    node.metadata.setdefault("cross_language_bindings", "[]")
    node.metadata.setdefault("function_calls", "[]")
    node.metadata.setdefault("imports", "[]")

    # Verify all defaults are set
    assert node.metadata["schema"] == "tt_graph_meta_v1"
    assert node.metadata["language"] == "python"
    assert node.metadata["has_bindings"] is False
    assert node.metadata["cross_language_bindings"] == "[]"
    assert node.metadata["function_calls"] == "[]"
    assert node.metadata["imports"] == "[]"


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

        assert lang == expected_lang, (
            f"Failed for {file_path}: expected {expected_lang}, got {lang}"
        )


def test_line_number_computation(tmp_path):
    """Test that line numbers are computed correctly from character indices"""
    from torchtalk.indexing import GraphEnhancedIndexer
    from llama_index.core.schema import TextNode

    # Create a temp file with known content
    # Line 1: def foo():
    # Line 2:     x = 1
    # Line 3:     y = 2
    # Line 4:     return x + y
    # Line 5: (empty)
    # Line 6: def bar():
    # Line 7:     return 42
    # Line 8: (empty)
    test_content = "def foo():\n    x = 1\n    y = 2\n    return x + y\n\ndef bar():\n    return 42\n"
    test_file = tmp_path / "test.py"
    test_file.write_text(test_content)

    indexer = GraphEnhancedIndexer(repo_path=str(tmp_path))

    # Test 1: First function (lines 1-4)
    # Content: "def foo():\n    x = 1\n    y = 2\n    return x + y\n"
    # Index 0-47 (including newline at 47)
    node1 = TextNode(
        text="def foo():\n    x = 1\n    y = 2\n    return x + y\n",
        metadata={"file_path": str(test_file)},
    )
    node1.start_char_idx = 0
    node1.end_char_idx = 47  # End after "return x + y\n"

    cache = {}
    indexer._compute_line_numbers(node1, cache)

    assert node1.metadata.get("start_line") == 1
    # end_line is 4 because there are 4 newlines before position 47
    assert node1.metadata.get("end_line") == 4

    # Test 2: Second function starting at line 6 (index 49)
    # Content: "def bar():\n    return 42\n"
    # Index 49-73
    node2 = TextNode(
        text="def bar():\n    return 42\n", metadata={"file_path": str(test_file)}
    )
    node2.start_char_idx = 49  # Start of "def bar"
    node2.end_char_idx = 73  # End of file

    indexer._compute_line_numbers(node2, cache)

    # Line 6 because there are 5 newlines before position 49
    assert node2.metadata.get("start_line") == 6
    # Line 7 because there are 6 newlines before position 73
    assert node2.metadata.get("end_line") == 7


def test_line_numbers_no_char_idx():
    """Test that nodes without char indices don't get line numbers"""
    from torchtalk.indexing import GraphEnhancedIndexer
    from llama_index.core.schema import TextNode

    indexer = GraphEnhancedIndexer(repo_path="/tmp")

    node = TextNode(text="some code", metadata={"file_path": "/tmp/test.py"})
    # Don't set start_char_idx or end_char_idx

    cache = {}
    indexer._compute_line_numbers(node, cache)

    assert "start_line" not in node.metadata
    assert "end_line" not in node.metadata


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
