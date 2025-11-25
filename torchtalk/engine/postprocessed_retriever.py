"""
Hybrid retriever with BM25 + vector search and postprocessing pipeline.

Uses LlamaIndex QueryFusionRetriever for hybrid retrieval (dense + sparse),
followed by reranking, filtering, and context optimization.

Supports both VectorStoreIndex and PropertyGraphIndex:
- VectorStoreIndex: BM25 + vector fusion with metadata-only graph info
- PropertyGraphIndex: BM25 + vector fusion (text embeddings stored in vector store)
"""

import logging
import re
from typing import ClassVar, List, Optional, Tuple, Union
from llama_index.core import VectorStoreIndex, PropertyGraphIndex, QueryBundle, Settings
from llama_index.core.retrievers import VectorIndexRetriever, BaseRetriever, QueryFusionRetriever
from llama_index.core.postprocessor import (
    SentenceTransformerRerank,
    SimilarityPostprocessor,
    LongContextReorder,
)
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore
from llama_index.retrievers.bm25 import BM25Retriever

log = logging.getLogger(__name__)


class PathBoostPostprocessor(BaseNodePostprocessor):
    """
    Boost scores for nodes from high-value implementation paths.

    PyTorch's real implementations live in aten/src/ATen/native/, not in
    the API wrappers (torch/csrc/api/). This postprocessor boosts scores
    for ATen native files to ensure they rank higher than wrappers.
    """

    # Paths to boost (real implementations)
    BOOST_PATTERNS: ClassVar[List[Tuple[str, float]]] = [
        (r"aten/src/ATen/native/cuda/", 0.15),   # CUDA kernels
        (r"aten/src/ATen/native/cpu/", 0.12),    # CPU kernels
        (r"aten/src/ATen/native/.+\.(cpp|h)$", 0.10),  # ATen native implementations (any depth)
        (r"native_functions\.yaml", 0.08),       # Dispatch definitions
    ]

    # Paths to slightly penalize (wrappers, not implementations)
    PENALIZE_PATTERNS: ClassVar[List[Tuple[str, float]]] = [
        (r"torch/csrc/api/", -0.05),  # C++ API wrappers
        (r"torch/_refs/", -0.02),     # Reference implementations (not primary)
    ]

    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        """Apply path-based score adjustments."""
        for node in nodes:
            # Robust path extraction - handle different metadata key names
            md = node.node.metadata or {}
            rel_path = md.get("rel_path") or md.get("file_path") or md.get("path") or ""
            adjustment = 0.0

            # Check boost patterns
            for pattern, boost in self.BOOST_PATTERNS:
                if re.search(pattern, rel_path, re.IGNORECASE):
                    adjustment = max(adjustment, boost)
                    break

            # Check penalize patterns (only if no boost applied)
            if adjustment == 0:
                for pattern, penalty in self.PENALIZE_PATTERNS:
                    if re.search(pattern, rel_path, re.IGNORECASE):
                        adjustment = penalty
                        break

            if adjustment != 0:
                node.score = (node.score or 0) + adjustment

        # Re-sort by adjusted score
        nodes.sort(key=lambda n: n.score or 0, reverse=True)
        return nodes


class _SimpleVectorRetriever(BaseRetriever):
    """
    Simple vector retriever that queries a SimpleVectorStore directly and fetches
    nodes from a docstore. Used for PropertyGraphIndex where we store text node
    embeddings in the vector store but SimpleVectorStore doesn't store text itself.
    """

    def __init__(
        self,
        vector_store,
        docstore,
        embed_model,
        similarity_top_k: int = 10,
    ):
        self._vector_store = vector_store
        self._docstore = docstore
        self._embed_model = embed_model
        self._similarity_top_k = similarity_top_k
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Query vector store and fetch nodes from docstore."""
        from llama_index.core.vector_stores import VectorStoreQuery

        # Get query embedding
        query_embedding = self._embed_model.get_query_embedding(query_bundle.query_str)

        # Query vector store
        query = VectorStoreQuery(
            query_embedding=query_embedding,
            similarity_top_k=self._similarity_top_k,
        )
        result = self._vector_store.query(query)

        # Fetch nodes from docstore and build results
        nodes_with_scores = []
        if result.ids and result.similarities:
            for node_id, similarity in zip(result.ids, result.similarities):
                node = self._docstore.get_node(node_id)
                if node:
                    nodes_with_scores.append(NodeWithScore(node=node, score=float(similarity)))

        return nodes_with_scores


class PostprocessedRetriever(BaseRetriever):
    """
    Hybrid retriever with postprocessing: BM25+vector fusion → path boost → rerank → filter → reorder.

    Pipeline (same for both VectorStoreIndex and PropertyGraphIndex):
    1. Hybrid search: BM25 (keyword) + vector (semantic) with reciprocal rank fusion
    2. Path boost: Lift ATen native files, penalize API wrappers
    3. Cross-encoder rerank (high precision)
    4. Similarity filter (quality threshold) - skipped for PropertyGraphIndex
    5. Long context reorder (LLM optimization)

    Note: PropertyGraphIndex stores text node embeddings in the vector store (not graph entity
    embeddings), so we use the same hybrid retrieval approach. Cross-language binding metadata
    is stored in node.metadata for the LLM to use when tracing code paths.
    """

    def __init__(
        self,
        index: Union[VectorStoreIndex, PropertyGraphIndex],
        similarity_top_k: int = 60,
        rerank_top_n: int = 10,
        similarity_cutoff: float = 0.5,
        rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        graph_path_depth: int = 2,
    ):
        self.is_property_graph = isinstance(index, PropertyGraphIndex)

        if self.is_property_graph:
            # PropertyGraphIndex: The vector store contains TEXT NODE embeddings (not graph entity
            # embeddings), so we use hybrid retrieval. The graph metadata (cross-language bindings)
            # is stored in node.metadata for the LLM to use when tracing code paths.
            #
            # We use a custom vector retriever that queries the SimpleVectorStore directly
            # and fetches nodes from the docstore.
            vector_retriever = _SimpleVectorRetriever(
                vector_store=index.storage_context.vector_stores.get("default"),
                docstore=index.storage_context.docstore,
                embed_model=Settings.embed_model,
                similarity_top_k=similarity_top_k,
            )

            bm25_retriever = BM25Retriever.from_defaults(
                docstore=index.storage_context.docstore,
                similarity_top_k=similarity_top_k,
            )

            from llama_index.core.llms import MockLLM
            self.retriever = QueryFusionRetriever(
                [vector_retriever, bm25_retriever],
                llm=MockLLM(),
                similarity_top_k=similarity_top_k,
                num_queries=1,
                mode="reciprocal_rerank",
                use_async=False,
            )
            log.info(f"PostprocessedRetriever: PropertyGraph (hybrid vector+bm25) "
                     f"retrieve={similarity_top_k}")
        else:
            # VectorStoreIndex: use hybrid BM25 + vector fusion
            vector_retriever = VectorIndexRetriever(index=index, similarity_top_k=similarity_top_k)

            bm25_retriever = BM25Retriever.from_defaults(
                docstore=index.docstore,
                similarity_top_k=similarity_top_k,
            )

            # Use a mock LLM to prevent QueryFusionRetriever from trying to resolve Settings.llm
            from llama_index.core.llms import MockLLM
            self.retriever = QueryFusionRetriever(
                [vector_retriever, bm25_retriever],
                llm=MockLLM(),
                similarity_top_k=similarity_top_k,
                num_queries=1,
                mode="reciprocal_rerank",
                use_async=False,
            )
            log.info(f"PostprocessedRetriever: hybrid(vector+bm25) retrieve={similarity_top_k}")

        self.path_booster = PathBoostPostprocessor()
        self.reranker = SentenceTransformerRerank(model=rerank_model, top_n=rerank_top_n)
        self.similarity_filter = SimilarityPostprocessor(similarity_cutoff=similarity_cutoff)
        self.reorderer = LongContextReorder()

        log.info(f"  → rerank={rerank_top_n} → path_boost → filter(cutoff={similarity_cutoff})")

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve and postprocess nodes through the pipeline."""
        # Hybrid retrieval (BM25 + vector with reciprocal rank fusion)
        nodes = self.retriever.retrieve(query_bundle)
        log.info(f"[Retrieval] Initial hybrid retrieval returned {len(nodes)} nodes")

        # Extract query string for postprocessors
        query_str = query_bundle.query_str

        # Rerank first with cross-encoder
        nodes = self.reranker.postprocess_nodes(nodes, query_str=query_str)
        log.info(f"[Retrieval] After reranking: {len(nodes)} nodes")

        # Path boost AFTER reranking: lift ATen native/CUDA files in final results
        # This ensures implementation files rank higher than API wrappers
        nodes = self.path_booster.postprocess_nodes(nodes, query_bundle=query_bundle)

        # NOTE: Skip similarity filter for PropertyGraphIndex because:
        # 1. PGRetriever returns scores in [0,1] range (cosine similarity)
        # 2. SentenceTransformerRerank returns cross-encoder logits (can be negative)
        # The reranker already selects top_n so further filtering is unnecessary
        if not self.is_property_graph:
            pre_filter_count = len(nodes)
            nodes = self.similarity_filter.postprocess_nodes(nodes, query_str=query_str)
            log.info(f"[Retrieval] After similarity filter: {len(nodes)} nodes (filtered {pre_filter_count - len(nodes)})")

        if not nodes:
            log.info("No relevant code context found - returning fallback node")
            from llama_index.core.schema import TextNode
            fallback_node = TextNode(
                text="No relevant PyTorch code was found for this query. "
                     "Respond naturally as a helpful PyTorch assistant. "
                     "If the user is greeting you or asking off-topic questions, "
                     "respond politely and offer to help with PyTorch codebase questions.",
                metadata={"file_path": "system", "is_fallback": True}
            )
            return [NodeWithScore(node=fallback_node, score=1.0)]

        nodes = self.reorderer.postprocess_nodes(nodes, query_str=query_str)

        # Log final retrieved files for debugging
        log.info(f"[Retrieval] Final {len(nodes)} nodes from files:")
        for i, node in enumerate(nodes[:10]):  # Log top 10
            file_path = node.node.metadata.get("file_path", "unknown")
            score = node.score or 0
            log.info(f"  {i+1}. [{score:.3f}] {file_path}")
        if len(nodes) > 10:
            log.info(f"  ... and {len(nodes) - 10} more")

        return nodes
