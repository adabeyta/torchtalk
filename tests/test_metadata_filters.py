"""
Tests for metadata filtering (Phase 3)
"""

import pytest
from unittest.mock import Mock


def test_metadata_filters_import():
    """Metadata filters can be imported"""
    from torchtalk.engine.metadata_filters import filter_by_metadata
    assert filter_by_metadata is not None


def test_language_filter_cpp():
    """C++ filter works"""
    from torchtalk.engine.metadata_filters import filter_by_metadata

    # Create mock nodes with metadata
    cpp_node = Mock()
    cpp_node.node.metadata = {"language": "cpp"}

    python_node = Mock()
    python_node.node.metadata = {"language": "python"}

    nodes = [cpp_node, python_node]

    # Query with C++ hint
    filtered = filter_by_metadata(nodes, "where is conv2d in c++")

    assert len(filtered) == 1
    assert filtered[0] == cpp_node


def test_language_filter_cuda():
    """CUDA filter works"""
    from torchtalk.engine.metadata_filters import filter_by_metadata

    cuda_node = Mock()
    cuda_node.node.metadata = {"language": "cuda"}

    cpp_node = Mock()
    cpp_node.node.metadata = {"language": "cpp"}

    nodes = [cuda_node, cpp_node]

    filtered = filter_by_metadata(nodes, "find CUDA kernel")

    assert len(filtered) == 1
    assert filtered[0] == cuda_node


def test_binding_filter():
    """Binding filter works"""
    from torchtalk.engine.metadata_filters import filter_by_metadata

    binding_node = Mock()
    binding_node.node.metadata = {"cross_language_bindings": ["some_binding"]}

    normal_node = Mock()
    normal_node.node.metadata = {}

    nodes = [binding_node, normal_node]

    filtered = filter_by_metadata(nodes, "show me pybind code")

    assert len(filtered) == 1
    assert filtered[0] == binding_node


def test_no_filter():
    """No filter applied when no hints"""
    from torchtalk.engine.metadata_filters import filter_by_metadata

    node1 = Mock()
    node1.node.metadata = {"language": "python"}

    node2 = Mock()
    node2.node.metadata = {"language": "cpp"}

    nodes = [node1, node2]

    filtered = filter_by_metadata(nodes, "generic query")

    assert len(filtered) == 2


def test_empty_result():
    """Returns empty list when all filtered out"""
    from torchtalk.engine.metadata_filters import filter_by_metadata

    python_node = Mock()
    python_node.node.metadata = {"language": "python"}

    nodes = [python_node]

    # Ask for C++ but only have Python
    filtered = filter_by_metadata(nodes, "c++ implementation")

    assert len(filtered) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
