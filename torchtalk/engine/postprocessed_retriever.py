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
    ):
        self.retriever = VectorIndexRetriever(index=index, similarity_top_k=similarity_top_k)
        self.reranker = SentenceTransformerRerank(model=rerank_model, top_n=rerank_top_n)
        self.similarity_filter = SimilarityPostprocessor(similarity_cutoff=similarity_cutoff)
        self.reorderer = LongContextReorder()

        log.info(f"PostprocessedRetriever: retrieve={similarity_top_k} → rerank={rerank_top_n}")

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve and postprocess nodes through the pipeline."""
        # Use the underlying VectorIndexRetriever which handles QueryBundle properly
        nodes = self.retriever.retrieve(query_bundle)

        # Extract query string for postprocessors
        query_str = query_bundle.query_str

        nodes = self.reranker.postprocess_nodes(nodes, query_str=query_str)
        nodes = self.similarity_filter.postprocess_nodes(nodes, query_str=query_str)

        if not nodes:
            log.info("No relevant code context found - returning fallback node")
            # Create a fallback node so LLM can respond naturally
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
        return nodes
