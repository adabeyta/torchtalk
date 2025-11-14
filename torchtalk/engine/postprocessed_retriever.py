"""
Retriever with postprocessing pipeline for improved retrieval quality.

Uses LlamaIndex postprocessors: reranking, filtering, and context optimization.
"""

import logging
from typing import List
from llama_index.core import VectorStoreIndex, QueryBundle
from llama_index.core.retrievers import VectorIndexRetriever, BaseRetriever
from llama_index.core.postprocessor import (
    SentenceTransformerRerank,
    SimilarityPostprocessor,
    LongContextReorder,
)
from llama_index.core.schema import NodeWithScore

log = logging.getLogger(__name__)


class PostprocessedRetriever(BaseRetriever):
    """
    Retriever with postprocessing: rerank → filter → reorder.

    Pipeline:
    1. Vector search (high recall)
    2. Cross-encoder rerank (high precision)
    3. Similarity filter (quality threshold)
    4. Long context reorder (LLM optimization)
    """

    def __init__(
        self,
        index: VectorStoreIndex,
        similarity_top_k: int = 40,
        rerank_top_n: int = 10,
        similarity_cutoff: float = 0.5,
        rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        use_metadata_filter: bool = False,
    ):
        self.retriever = VectorIndexRetriever(index=index, similarity_top_k=similarity_top_k)
        self.reranker = SentenceTransformerRerank(model=rerank_model, top_n=rerank_top_n)
        self.similarity_filter = SimilarityPostprocessor(similarity_cutoff=similarity_cutoff)
        self.reorderer = LongContextReorder()
        self.use_metadata_filter = use_metadata_filter

        log.info(f"PostprocessedRetriever: retrieve={similarity_top_k} → rerank={rerank_top_n} (metadata_filter={use_metadata_filter})")

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve and postprocess nodes through the pipeline."""
        # Use the underlying VectorIndexRetriever which handles QueryBundle properly
        nodes = self.retriever.retrieve(query_bundle)

        # Extract query string for postprocessors
        query_str = query_bundle.query_str

        nodes = self.reranker.postprocess_nodes(nodes, query_str=query_str)
        nodes = self.similarity_filter.postprocess_nodes(nodes, query_str=query_str)

        if not nodes:
            log.debug("No nodes above similarity cutoff; returning empty result")
            return []

        # Optional metadata filtering (Phase 3)
        if self.use_metadata_filter:
            from torchtalk.engine.metadata_filters import filter_by_metadata
            nodes = filter_by_metadata(nodes, query_str)

        nodes = self.reorderer.postprocess_nodes(nodes, query_str=query_str)
        return nodes
