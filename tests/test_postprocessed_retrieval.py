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
    from llama_index.core.storage.docstore import SimpleDocumentStore
    from llama_index.core.schema import TextNode

    # Create a more complete mock with required attributes
    mock_index = Mock()
    mock_docstore = SimpleDocumentStore()
    # Add at least one document to avoid BM25 empty corpus error
    mock_docstore.add_documents([TextNode(text="test document", id_="test_id")])
    mock_index.docstore = mock_docstore

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
    assert retriever.context_formatter is not None
    assert retriever.binding_expander is not None


def test_context_formatter_basic():
    """ContextFormatterPostprocessor adds file path header"""
    from torchtalk.engine.postprocessed_retriever import ContextFormatterPostprocessor
    from llama_index.core.schema import TextNode, NodeWithScore

    formatter = ContextFormatterPostprocessor()

    # Create a test node with metadata
    node = TextNode(
        text="def layer_norm(x):\n    return x",
        metadata={
            "rel_path": "torch/nn/functional.py",
            "language": "python",
        }
    )
    nodes = [NodeWithScore(node=node, score=0.9)]

    result = formatter.postprocess_nodes(nodes)

    # Check header was added
    formatted_text = result[0].node.get_content()
    assert "[FILE: torch/nn/functional.py]" in formatted_text
    assert "[LANGUAGE: PYTHON]" in formatted_text
    assert "def layer_norm(x):" in formatted_text


def test_context_formatter_with_line_numbers():
    """ContextFormatterPostprocessor includes line numbers when available"""
    from torchtalk.engine.postprocessed_retriever import ContextFormatterPostprocessor
    from llama_index.core.schema import TextNode, NodeWithScore

    formatter = ContextFormatterPostprocessor()

    node = TextNode(
        text="void layer_norm_cuda() {}",
        metadata={
            "rel_path": "aten/src/ATen/native/cuda/layer_norm_kernel.cu",
            "language": "cuda",
            "start_line": 45,
            "end_line": 120,
        }
    )
    nodes = [NodeWithScore(node=node, score=0.8)]

    result = formatter.postprocess_nodes(nodes)

    formatted_text = result[0].node.get_content()
    assert "[FILE: aten/src/ATen/native/cuda/layer_norm_kernel.cu:45-120]" in formatted_text
    assert "[LANGUAGE: CUDA]" in formatted_text


def test_context_formatter_with_bindings():
    """ContextFormatterPostprocessor surfaces cross-language bindings"""
    from torchtalk.engine.postprocessed_retriever import ContextFormatterPostprocessor
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
                    "type": "native_dispatch_cuda"
                },
                {
                    "python_name": "native_layer_norm",
                    "cpp_name": "native_layer_norm_cuda",
                    "type": "native_dispatch_cuda"
                }
            ]
        }
    )
    nodes = [NodeWithScore(node=node, score=0.85)]

    result = formatter.postprocess_nodes(nodes)

    formatted_text = result[0].node.get_content()
    assert "[CROSS-LANGUAGE BINDINGS (computed, not file content):" in formatted_text
    assert "layer_norm → layer_norm_cuda" in formatted_text
    assert "native_dispatch_cuda" in formatted_text


def test_context_formatter_with_json_serialized_bindings():
    """ContextFormatterPostprocessor handles JSON-serialized bindings (LanceDB compatibility)"""
    import json
    from torchtalk.engine.postprocessed_retriever import ContextFormatterPostprocessor
    from llama_index.core.schema import TextNode, NodeWithScore

    formatter = ContextFormatterPostprocessor()

    # Simulate metadata stored in LanceDB (JSON-serialized)
    bindings = [
        {"python_name": "softmax", "cpp_name": "softmax_cuda", "type": "native_dispatch_cuda"}
    ]

    node = TextNode(
        text="void softmax_cuda_kernel() {}",
        metadata={
            "rel_path": "aten/src/ATen/native/cuda/softmax.cu",
            "language": "cuda",
            "cross_language_bindings": json.dumps(bindings)  # JSON string, not list
        }
    )
    nodes = [NodeWithScore(node=node, score=0.85)]

    result = formatter.postprocess_nodes(nodes)

    formatted_text = result[0].node.get_content()
    assert "[CROSS-LANGUAGE BINDINGS (computed, not file content):" in formatted_text
    assert "softmax → softmax_cuda" in formatted_text


def test_get_bindings_metadata_helper():
    """_get_bindings_metadata handles both JSON strings and native lists"""
    import json
    from torchtalk.engine.postprocessed_retriever import _get_bindings_metadata

    bindings_list = [{"python_name": "foo", "cpp_name": "bar", "type": "pybind11"}]

    # Test with native list
    result1 = _get_bindings_metadata({"cross_language_bindings": bindings_list})
    assert result1 == bindings_list

    # Test with JSON string
    result2 = _get_bindings_metadata({"cross_language_bindings": json.dumps(bindings_list)})
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
    from torchtalk.engine.postprocessed_retriever import ContextFormatterPostprocessor
    from llama_index.core.schema import TextNode, NodeWithScore

    formatter = ContextFormatterPostprocessor()

    node = TextNode(
        text="def foo(): pass",
        metadata={"rel_path": "test.py", "language": "python"}
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


def test_binding_expansion_finds_related_nodes():
    """BindingExpansionPostprocessor finds cross-language implementations"""
    from torchtalk.engine.postprocessed_retriever import BindingExpansionPostprocessor
    from llama_index.core.schema import TextNode, NodeWithScore
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
                {"python_name": "layer_norm", "cpp_name": "layer_norm_cuda", "type": "native_dispatch_cuda"}
            ]
        }
    )

    # CUDA kernel node
    cuda_node = TextNode(
        id_="cuda_node",
        text="void layer_norm_cuda_kernel() {}",
        metadata={
            "rel_path": "aten/src/ATen/native/cuda/layer_norm_kernel.cu",
            "language": "cuda",
            "cross_language_bindings": [
                {"python_name": "layer_norm", "cpp_name": "layer_norm_cuda", "type": "native_dispatch_cuda"}
            ]
        }
    )

    # Unrelated node
    unrelated_node = TextNode(
        id_="unrelated_node",
        text="void some_other_function() {}",
        metadata={
            "rel_path": "some/other/file.cpp",
            "language": "cpp",
            "cross_language_bindings": []
        }
    )

    docstore.add_documents([python_node, cuda_node, unrelated_node])

    expander = BindingExpansionPostprocessor(docstore=docstore, max_expansion=5)

    # Start with just the Python node
    initial_nodes = [NodeWithScore(node=python_node, score=0.9)]

    result = expander.postprocess_nodes(initial_nodes)

    # Should have found the CUDA node as related
    assert len(result) == 2
    result_ids = {n.node.node_id for n in result}
    assert "python_node" in result_ids
    assert "cuda_node" in result_ids
    assert "unrelated_node" not in result_ids


def test_binding_expansion_no_docstore():
    """BindingExpansionPostprocessor handles missing docstore gracefully"""
    from torchtalk.engine.postprocessed_retriever import BindingExpansionPostprocessor
    from llama_index.core.schema import TextNode, NodeWithScore

    expander = BindingExpansionPostprocessor(docstore=None)

    node = TextNode(
        text="some code",
        metadata={"cross_language_bindings": [{"python_name": "foo", "cpp_name": "bar"}]}
    )
    nodes = [NodeWithScore(node=node, score=0.9)]

    result = expander.postprocess_nodes(nodes)

    # Should return original nodes unchanged
    assert len(result) == 1


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
