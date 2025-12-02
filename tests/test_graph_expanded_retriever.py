"""
Tests for GraphExpandedRetriever (cross-language PyTorch code search)
"""

import json
import pytest
from pathlib import Path


def test_context_formatter_basic():
    """ContextFormatterPostprocessor adds file path header"""
    from torchtalk.engine.graph_expanded_retriever import ContextFormatterPostprocessor
    from llama_index.core.schema import TextNode, NodeWithScore

    formatter = ContextFormatterPostprocessor()

    # Create a test node with metadata
    node = TextNode(
        text="def layer_norm(x):\n    return x",
        metadata={
            "rel_path": "torch/nn/functional.py",
            "language": "python",
        },
    )
    nodes = [NodeWithScore(node=node, score=0.9)]

    result = formatter.postprocess_nodes(nodes)

    # Check header was added
    formatted_text = result[0].node.get_content()
    assert "[FILE: torch/nn/functional.py]" in formatted_text
    assert "[LAYER: PYTHON]" in formatted_text
    assert "def layer_norm(x):" in formatted_text


def test_context_formatter_with_line_numbers():
    """ContextFormatterPostprocessor includes line numbers when available"""
    from torchtalk.engine.graph_expanded_retriever import ContextFormatterPostprocessor
    from llama_index.core.schema import TextNode, NodeWithScore

    formatter = ContextFormatterPostprocessor()

    node = TextNode(
        text="void layer_norm_cuda() {}",
        metadata={
            "rel_path": "aten/src/ATen/native/cuda/layer_norm_kernel.cu",
            "language": "cuda",
            "start_line": 45,
            "end_line": 120,
        },
    )
    nodes = [NodeWithScore(node=node, score=0.8)]

    result = formatter.postprocess_nodes(nodes)

    formatted_text = result[0].node.get_content()
    assert (
        "[FILE: aten/src/ATen/native/cuda/layer_norm_kernel.cu:45-120]"
        in formatted_text
    )
    assert "[LAYER: CUDA]" in formatted_text


def test_context_formatter_with_bindings():
    """ContextFormatterPostprocessor surfaces cross-language bindings"""
    from torchtalk.engine.graph_expanded_retriever import ContextFormatterPostprocessor
    from llama_index.core.schema import TextNode, NodeWithScore

    formatter = ContextFormatterPostprocessor()

    node = TextNode(
        text="void layer_norm_cuda_impl() {}",
        metadata={
            "rel_path": "aten/src/ATen/native/cuda/layer_norm_kernel.cu",
            "language": "cuda",
            "cross_language_bindings": [
                {
                    "python_name": "layer_norm",
                    "cpp_name": "layer_norm_cuda",
                    "type": "native_dispatch_cuda",
                },
                {
                    "python_name": "native_layer_norm",
                    "cpp_name": "native_layer_norm_cuda",
                    "type": "native_dispatch_cuda",
                },
            ],
        },
    )
    nodes = [NodeWithScore(node=node, score=0.85)]

    result = formatter.postprocess_nodes(nodes)

    formatted_text = result[0].node.get_content()
    assert "[BINDINGS:" in formatted_text
    assert "layer_norm → layer_norm_cuda" in formatted_text
    assert "native_dispatch_cuda" in formatted_text


def test_context_formatter_with_json_serialized_bindings():
    """ContextFormatterPostprocessor handles JSON-serialized bindings (LanceDB compatibility)"""
    from torchtalk.engine.graph_expanded_retriever import ContextFormatterPostprocessor
    from llama_index.core.schema import TextNode, NodeWithScore

    formatter = ContextFormatterPostprocessor()

    # Simulate metadata stored in LanceDB (JSON-serialized)
    bindings = [
        {
            "python_name": "softmax",
            "cpp_name": "softmax_cuda",
            "type": "native_dispatch_cuda",
        }
    ]

    node = TextNode(
        text="void softmax_cuda_kernel() {}",
        metadata={
            "rel_path": "aten/src/ATen/native/cuda/softmax.cu",
            "language": "cuda",
            "cross_language_bindings": json.dumps(bindings),  # JSON string, not list
        },
    )
    nodes = [NodeWithScore(node=node, score=0.85)]

    result = formatter.postprocess_nodes(nodes)

    formatted_text = result[0].node.get_content()
    assert "[BINDINGS:" in formatted_text
    assert "softmax → softmax_cuda" in formatted_text


def test_get_bindings_metadata_helper():
    """_get_bindings_metadata handles both JSON strings and native lists"""
    from torchtalk.engine.graph_expanded_retriever import _get_bindings_metadata

    bindings_list = [{"python_name": "foo", "cpp_name": "bar", "type": "pybind11"}]

    # Test with native list
    result1 = _get_bindings_metadata({"cross_language_bindings": bindings_list})
    assert result1 == bindings_list

    # Test with JSON string
    result2 = _get_bindings_metadata(
        {"cross_language_bindings": json.dumps(bindings_list)}
    )
    assert result2 == bindings_list

    # Test with empty string
    result3 = _get_bindings_metadata({"cross_language_bindings": ""})
    assert result3 == []

    # Test with empty list JSON
    result4 = _get_bindings_metadata({"cross_language_bindings": "[]"})
    assert result4 == []

    # Test with missing key
    result5 = _get_bindings_metadata({})
    assert result5 == []

    # Test with invalid JSON
    result6 = _get_bindings_metadata({"cross_language_bindings": "not valid json"})
    assert result6 == []


def test_context_formatter_no_double_format():
    """ContextFormatterPostprocessor doesn't double-format nodes"""
    from torchtalk.engine.graph_expanded_retriever import ContextFormatterPostprocessor
    from llama_index.core.schema import TextNode, NodeWithScore

    formatter = ContextFormatterPostprocessor()

    node = TextNode(
        text="def foo(): pass", metadata={"rel_path": "test.py", "language": "python"}
    )
    nodes = [NodeWithScore(node=node, score=0.9)]

    # Format once
    result1 = formatter.postprocess_nodes(nodes)
    text_after_first = result1[0].node.get_content()

    # Format again - should not change
    result2 = formatter.postprocess_nodes(result1)
    text_after_second = result2[0].node.get_content()

    assert text_after_first == text_after_second
    assert text_after_first.count("[FILE:") == 1  # Only one header


def test_binding_index_finds_related_nodes():
    """BindingIndex finds cross-language implementations via graph expansion"""
    from torchtalk.engine.graph_expanded_retriever import BindingIndex
    from llama_index.core.schema import TextNode
    from llama_index.core.storage.docstore import SimpleDocumentStore

    # Create a mock docstore with related nodes
    docstore = SimpleDocumentStore()

    # Python API node
    python_node = TextNode(
        id_="python_node",
        text="def layer_norm(x): ...",
        metadata={
            "rel_path": "torch/nn/functional.py",
            "language": "python",
            "cross_language_bindings": [
                {
                    "python_name": "layer_norm",
                    "cpp_name": "layer_norm_cuda",
                    "type": "native_dispatch_cuda",
                }
            ],
        },
    )

    # CUDA kernel node
    cuda_node = TextNode(
        id_="cuda_node",
        text="void layer_norm_cuda_kernel() {}",
        metadata={
            "rel_path": "aten/src/ATen/native/cuda/layer_norm_kernel.cu",
            "language": "cuda",
            "cross_language_bindings": [
                {
                    "python_name": "layer_norm",
                    "cpp_name": "layer_norm_cuda",
                    "type": "native_dispatch_cuda",
                }
            ],
        },
    )

    # Unrelated node
    unrelated_node = TextNode(
        id_="unrelated_node",
        text="void some_other_function() {}",
        metadata={
            "rel_path": "some/other/file.cpp",
            "language": "cpp",
            "cross_language_bindings": [],
        },
    )

    docstore.add_documents([python_node, cuda_node, unrelated_node])

    # Test BindingIndex graph expansion
    binding_index = BindingIndex(docstore)

    # Get nodes connected to python_node
    connected = binding_index.get_connected_node_ids("python_node")

    # Should find the CUDA node as connected
    assert "cuda_node" in connected
    assert "python_node" not in connected  # Shouldn't include itself
    assert "unrelated_node" not in connected


def test_binding_index_lookup_by_name():
    """BindingIndex can lookup nodes by operation name"""
    from torchtalk.engine.graph_expanded_retriever import BindingIndex
    from llama_index.core.schema import TextNode
    from llama_index.core.storage.docstore import SimpleDocumentStore

    docstore = SimpleDocumentStore()

    dropout_node = TextNode(
        id_="dropout_node",
        text="def dropout(x): ...",
        metadata={
            "rel_path": "torch/nn/functional.py",
            "cross_language_bindings": [
                {
                    "python_name": "dropout",
                    "cpp_name": "native_dropout_cuda",
                    "type": "native_dispatch_cuda",
                }
            ],
        },
    )

    docstore.add_documents([dropout_node])

    binding_index = BindingIndex(docstore)

    # Lookup by base name
    results = binding_index.lookup_by_name("dropout")
    assert "dropout_node" in results

    # Lookup by full cpp name
    results = binding_index.lookup_by_name("native_dropout_cuda")
    assert "dropout_node" in results


def test_binding_index_no_docstore():
    """BindingIndex handles missing docstore gracefully"""
    from torchtalk.engine.graph_expanded_retriever import BindingIndex

    binding_index = BindingIndex(docstore=None)

    # Should return empty set without error
    connected = binding_index.get_connected_node_ids("some_id")
    assert connected == set()

    lookup = binding_index.lookup_by_name("dropout")
    assert lookup == set()


def test_get_layer_type():
    """_get_layer_type correctly identifies PyTorch code layers"""
    from torchtalk.engine.graph_expanded_retriever import _get_layer_type

    assert _get_layer_type("torch/nn/functional.py") == "python"
    assert _get_layer_type("aten/src/ATen/native/cuda/Dropout.cu") == "cuda"
    assert _get_layer_type("aten/src/ATen/native/layer_norm.cpp") == "cpp"
    assert _get_layer_type("aten/src/ATen/native/native_functions.yaml") == "yaml"
    assert _get_layer_type("some/random/file.txt") == "other"
    assert _get_layer_type("") == "other"


def test_extract_base_op_name():
    """_extract_base_op_name strips prefixes and suffixes"""
    from torchtalk.engine.graph_expanded_retriever import _extract_base_op_name

    assert _extract_base_op_name("native_dropout_cuda") == "dropout"
    assert _extract_base_op_name("native_dropout") == "dropout"
    assert _extract_base_op_name("fused_layer_norm_cuda") == "layer_norm"
    assert _extract_base_op_name("layer_norm_backward") == "layer_norm"
    assert _extract_base_op_name("cudnn_batch_norm") == "batch_norm"

    # Edge cases
    assert _extract_base_op_name("dropout") is None  # No change
    assert _extract_base_op_name("a") is None  # Too short
    assert _extract_base_op_name("") is None


def test_file_importance_postprocessor():
    """FileImportancePostprocessor uses PageRank-based scores"""
    from torchtalk.engine.graph_expanded_retriever import FileImportancePostprocessor
    from llama_index.core.schema import TextNode, NodeWithScore

    processor = FileImportancePostprocessor()

    # Create nodes with file_importance metadata (simulating indexed data)
    high_importance_node = TextNode(
        id_="high_importance",
        text="def important_function(): ...",
        metadata={
            "rel_path": "torch/core_module.py",
            "file_importance": 0.9,  # High PageRank score
            "cross_language_bindings": "[]",
        },
    )

    low_importance_node = TextNode(
        id_="low_importance",
        text="def helper(): ...",
        metadata={
            "rel_path": "torch/utils/helper.py",
            "file_importance": 0.1,  # Low PageRank score
            "cross_language_bindings": "[]",
        },
    )

    test_node = TextNode(
        id_="test_node",
        text="def test_something(): ...",
        metadata={
            "rel_path": "tests/test_module.py",
            "file_importance": 0.5,  # Medium but penalized for being test
            "cross_language_bindings": "[]",
        },
    )

    nodes = [
        NodeWithScore(node=low_importance_node, score=0.5),
        NodeWithScore(node=test_node, score=0.6),
        NodeWithScore(node=high_importance_node, score=0.5),
    ]

    result = processor.postprocess_nodes(nodes)

    # High importance file should be boosted to the top
    assert result[0].node.id_ == "high_importance"
    # Test file should be penalized and at the bottom
    assert result[-1].node.id_ == "test_node"


def test_query_expansion_general():
    """_expand_pytorch_query uses general pattern detection"""
    from torchtalk.engine.graph_expanded_retriever import _expand_pytorch_query

    # Standard operation names should get native_ prefix added
    expanded = _expand_pytorch_query("How does layer_norm work?")
    assert "native_layer_norm" in expanded
    assert "dispatch" in expanded

    # Works for any operation-like word
    expanded = _expand_pytorch_query("What is my_custom_op?")
    assert "native_my_custom_op" in expanded

    # Implementation queries should add yaml reference
    expanded = _expand_pytorch_query("Where is dropout implemented in CUDA?")
    assert "native_functions.yaml" in expanded

    # Common English words should NOT be expanded
    expanded = _expand_pytorch_query("What is the function?")
    assert "native_the" not in expanded
    assert "native_function" not in expanded  # In common words list


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
