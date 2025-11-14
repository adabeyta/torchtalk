"""Tests for relationship builder (Phase 0)"""

import pytest
from llama_index.core import load_index_from_storage, StorageContext, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from torchtalk.indexing import add_relationships_to_index, RelationshipBuilder


@pytest.fixture
def index_path():
    """Get test index path from environment"""
    import os
    path = os.getenv("TEST_INDEX_PATH", "./index")
    return path


@pytest.mark.slow
def test_add_relationships_to_index(index_path):
    """Test adding relationships to an existing index"""
    stats = add_relationships_to_index(index_path)

    assert stats['total_nodes'] > 0, "Index should have nodes"
    assert stats['nodes_with_refs'] > 0, "Some nodes should have relationships"
    assert stats['coverage'] > 0.1, f"Low coverage: {stats['coverage']:.1%}"
    assert stats['avg_refs_per_node'] > 0, "Nodes with refs should have avg > 0"

    print(f"\n✓ Relationship stats:")
    print(f"  Total nodes: {stats['total_nodes']}")
    print(f"  Nodes with refs: {stats['nodes_with_refs']}")
    print(f"  Coverage: {stats['coverage']:.1%}")
    print(f"  Avg refs/node: {stats['avg_refs_per_node']:.1f}")


@pytest.mark.slow
def test_relationships_exist(index_path):
    """Verify nodes have relationships after building"""
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    Settings.embed_model = embed_model

    storage = StorageContext.from_defaults(persist_dir=index_path)
    index = load_index_from_storage(storage)

    nodes_with_refs = sum(
        1 for n in index.docstore.docs.values()
        if 'references' in n.relationships and n.relationships['references']
    )

    total_nodes = len(index.docstore.docs)
    coverage = nodes_with_refs / total_nodes

    print(f"\nRelationship coverage: {nodes_with_refs}/{total_nodes} ({coverage:.1%})")

    assert coverage >= 0.1, f"Low relationship coverage: {coverage:.1%}"


@pytest.mark.unit
def test_relationship_builder_mapping():
    """Unit test for entity mapping logic"""
    from llama_index.core import VectorStoreIndex, Settings
    from llama_index.core.schema import TextNode
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding

    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

    nodes = [
        TextNode(
            text="def foo(): pass",
            metadata={
                "rel_path": "test.py",
                "function_calls": [{"function": "foo", "calls": ["bar"]}],
                "imports": ["test2.py"]
            }
        ),
        TextNode(
            text="def bar(): pass",
            metadata={
                "rel_path": "test2.py",
                "function_calls": [{"function": "bar", "calls": []}],
                "imports": []
            }
        )
    ]

    index = VectorStoreIndex(nodes)
    builder = RelationshipBuilder(index)
    builder._build_entity_mapping()

    assert "test.py" in builder.entity_to_nodes
    assert "test2.py" in builder.entity_to_nodes
    assert "foo" in builder.entity_to_nodes
    assert "bar" in builder.entity_to_nodes


@pytest.mark.unit
def test_relationship_builder_stats():
    """Unit test for relationship statistics"""
    from llama_index.core import VectorStoreIndex, Settings
    from llama_index.core.schema import TextNode
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding

    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

    nodes = [
        TextNode(
            text="def foo(): bar()",
            metadata={
                "rel_path": "test.py",
                "function_calls": [{"function": "foo", "calls": ["bar"]}],
                "imports": []
            }
        ),
        TextNode(
            text="def bar(): pass",
            metadata={
                "rel_path": "test.py",
                "function_calls": [{"function": "bar", "calls": []}],
                "imports": []
            }
        )
    ]

    index = VectorStoreIndex(nodes)
    builder = RelationshipBuilder(index)
    stats = builder.build_relationships()

    assert stats['total_nodes'] == 2
    assert stats['nodes_with_refs'] >= 1
    assert stats['coverage'] > 0
