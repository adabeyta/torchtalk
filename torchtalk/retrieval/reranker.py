#!/usr/bin/env python3
"""
Re-ranking retrieval results using CrossEncoder for improved relevance.
"""

from typing import List, Dict, Any, Optional
import numpy as np


class Reranker:
    """
    Re-rank retrieval results using a CrossEncoder model.

    CrossEncoders compute relevance scores by encoding query + document pairs
    together, which is more accurate than comparing independent embeddings but
    slower (so we apply it after initial retrieval).

    Model: sentence-transformers/cross-encoder/ms-marco-MiniLM-L-6-v2
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: Optional[str] = None
    ):
        """
        Initialize the reranker.

        Args:
            model_name: HuggingFace CrossEncoder model name
            device: Device to run on ('cpu' or 'cuda', auto-detect if None)
        """
        self.model_name = model_name

        # Auto-detect device
        if device is None:
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model = None
        self._lazy_load()

    def _lazy_load(self):
        """Lazy load the model"""
        if self.model is not None:
            return

        print(f"Loading reranker model: {self.model_name}")
        print("(This will download ~80MB on first run)")

        try:
            from sentence_transformers import CrossEncoder

            self.model = CrossEncoder(self.model_name, device=self.device)

            print(f" Reranker loaded: {self.model_name}")
            if self.device == "cuda":
                print(" Using GPU acceleration")
            else:
                print(" Using CPU")

        except ImportError as e:
            raise ImportError(
                "sentence-transformers is required for re-ranking. "
                "Install with: pip install sentence-transformers"
            ) from e

    def rerank(
        self,
        query: str,
        results: List[Dict[str, Any]],
        top_k: Optional[int] = None,
        batch_size: int = 32
    ) -> List[Dict[str, Any]]:
        """
        Re-rank retrieval results by relevance to query.

        Args:
            query: The original query text
            results: List of retrieval results (from VectorRetriever or augmented)
            top_k: Number of top results to return (None = return all, reranked)
            batch_size: Batch size for processing

        Returns:
            Re-ranked results with updated 'rerank_score' field
        """
        if not results:
            return results

        self._lazy_load()

        # Prepare query-document pairs
        pairs = []
        for result in results:
            doc_text = result['content']
            pairs.append([query, doc_text])

        # Score all pairs
        scores = self.model.predict(pairs, batch_size=batch_size, show_progress_bar=False)

        # Add rerank scores to results
        reranked_results = []
        for result, score in zip(results, scores):
            reranked_result = {
                **result,
                'rerank_score': float(score),
                'original_score': result.get('score', 0.0)  # Preserve original vector similarity score
            }
            reranked_results.append(reranked_result)

        # Sort by rerank score (higher is better)
        reranked_results.sort(key=lambda x: x['rerank_score'], reverse=True)

        # Return top_k if specified
        if top_k is not None:
            reranked_results = reranked_results[:top_k]

        return reranked_results

    def compute_relevance_score(
        self,
        query: str,
        document: str
    ) -> float:
        """
        Compute relevance score for a single query-document pair.

        Args:
            query: Query text
            document: Document text

        Returns:
            Relevance score (higher = more relevant)
        """
        self._lazy_load()
        score = self.model.predict([[query, document]])[0]
        return float(score)

    def batch_score(
        self,
        query: str,
        documents: List[str],
        batch_size: int = 32
    ) -> List[float]:
        """
        Score multiple documents against a query.

        Args:
            query: Query text
            documents: List of document texts
            batch_size: Batch size for processing

        Returns:
            List of relevance scores
        """
        self._lazy_load()

        pairs = [[query, doc] for doc in documents]
        scores = self.model.predict(pairs, batch_size=batch_size, show_progress_bar=False)

        return [float(score) for score in scores]


class HybridScorer:
    """
    Combine vector similarity scores with reranking scores.

    Different strategies for combining scores:
    - weighted: Weighted average of vector and rerank scores
    - rerank_only: Use only rerank scores
    - cascade: Use rerank for top candidates, vector for rest
    """

    def __init__(
        self,
        strategy: str = "weighted",
        vector_weight: float = 0.3,
        rerank_weight: float = 0.7
    ):
        """
        Initialize hybrid scorer.

        Args:
            strategy: Scoring strategy ('weighted', 'rerank_only', 'cascade')
            vector_weight: Weight for vector similarity scores (for 'weighted' strategy)
            rerank_weight: Weight for reranking scores (for 'weighted' strategy)
        """
        self.strategy = strategy
        self.vector_weight = vector_weight
        self.rerank_weight = rerank_weight

        # Normalize weights
        if strategy == "weighted":
            total = vector_weight + rerank_weight
            self.vector_weight = vector_weight / total
            self.rerank_weight = rerank_weight / total

    def score(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Apply hybrid scoring to results.

        Args:
            results: Results with both 'score' and 'rerank_score' fields

        Returns:
            Results with 'final_score' field and sorted by it
        """
        if self.strategy == "rerank_only":
            # Use rerank score only
            for result in results:
                result['final_score'] = result.get('rerank_score', 0.0)

        elif self.strategy == "weighted":
            # Weighted combination
            # Normalize scores to [0, 1] first
            vector_scores = [r.get('score', 0.0) for r in results]
            rerank_scores = [r.get('rerank_score', 0.0) for r in results]

            # Min-max normalization
            v_min, v_max = min(vector_scores), max(vector_scores)
            r_min, r_max = min(rerank_scores), max(rerank_scores)

            for i, result in enumerate(results):
                v_norm = (vector_scores[i] - v_min) / (v_max - v_min + 1e-10)
                r_norm = (rerank_scores[i] - r_min) / (r_max - r_min + 1e-10)

                result['final_score'] = (
                    self.vector_weight * v_norm +
                    self.rerank_weight * r_norm
                )

        else:  # cascade
            # Use rerank for top results, vector for rest
            # Simple implementation: just use rerank score
            for result in results:
                result['final_score'] = result.get('rerank_score', result.get('score', 0.0))

        # Sort by final score
        results.sort(key=lambda x: x['final_score'], reverse=True)

        return results


