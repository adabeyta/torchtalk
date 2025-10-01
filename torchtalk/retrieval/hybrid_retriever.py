#!/usr/bin/env python3
"""
Hybrid retrieval pipeline combining vector search, graph augmentation, and re-ranking.

This is the main interface for code retrieval in TorchTalk v2.0.
"""

from typing import List, Dict, Any, Optional
from pathlib import Path

from torchtalk.retrieval.vector_retriever import VectorRetriever
from torchtalk.retrieval.graph_augmenter import GraphAugmenter
from torchtalk.retrieval.reranker import Reranker, HybridScorer
from torchtalk.retrieval.query_analyzer import QueryAnalyzer, QueryType
from torchtalk.retrieval.adaptive_context_builder import AdaptiveContextBuilder, ContextBuildResult
from torchtalk.retrieval.graph_expansion_retriever import GraphExpansionRetriever


class HybridRetriever:
    """
    Full retrieval pipeline with vector search, graph augmentation, and re-ranking.

    Pipeline stages:
    1. Vector search: Semantic similarity search using embeddings
    2. Graph augmentation: Expand results with related code (imports, calls, inheritance)
    3. Re-ranking: Improve precision using CrossEncoder
    4. Scoring: Combine signals for final ranking

    This provides state-of-the-art code retrieval with both semantic understanding
    and structural awareness.
    """

    def __init__(
        self,
        index_dir: str,
        collection_name: str = "torchtalk_code",
        use_reranking: bool = True,
        use_graph_augmentation: bool = True,
        device: Optional[str] = None,
        max_model_len: Optional[int] = None
    ):
        """
        Initialize the hybrid retriever.

        Args:
            index_dir: Directory containing the index
            collection_name: Name of the vector store collection
            use_reranking: Whether to apply re-ranking (slower but more accurate)
            use_graph_augmentation: Whether to augment with graph relationships
            device: Device for models ('cpu' or 'cuda', auto-detect if None)
            max_model_len: Maximum model context length (for adaptive context building)
        """
        self.index_dir = Path(index_dir)
        self.use_reranking = use_reranking
        self.use_graph_augmentation = use_graph_augmentation

        # Initialize components
        print("Initializing HybridRetriever...")

        self.vector_retriever = VectorRetriever(
            index_dir=str(index_dir),
            collection_name=collection_name,
            device=device
        )

        if use_graph_augmentation:
            self.graph_augmenter = GraphAugmenter(index_dir=str(index_dir))
        else:
            self.graph_augmenter = None

        if use_reranking:
            self.reranker = Reranker(device=device)
            self.hybrid_scorer = HybridScorer(strategy="weighted", vector_weight=0.3, rerank_weight=0.7)
        else:
            self.reranker = None
            self.hybrid_scorer = None

        # V2.1: Dynamic context assembly
        self.query_analyzer = QueryAnalyzer()
        self.max_model_len = max_model_len or 131072  # Default 128k
        self.context_builder = AdaptiveContextBuilder(max_model_len=self.max_model_len)

        # V2.2: Graph-expansion retrieval for cross-layer tracing
        if use_graph_augmentation and self.graph_augmenter:
            self.graph_expansion_retriever = GraphExpansionRetriever(
                vector_retriever=self.vector_retriever,
                graph_augmenter=self.graph_augmenter
            )
        else:
            self.graph_expansion_retriever = None

        print(" HybridRetriever ready")

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        initial_k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
        augment_top_n: int = 3,
        max_augmentations: int = 5,
        augment_depth: int = 1
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant code for a query using the full pipeline.

        Args:
            query: Natural language or code query
            top_k: Final number of results to return
            initial_k: Number of results from vector search (None = top_k * 2 for reranking)
            filters: Metadata filters for vector search
            augment_top_n: Number of top results to augment with graph relationships
            max_augmentations: Maximum related code items to add per result
            augment_depth: Graph traversal depth for augmentation (1-5)

        Returns:
            List of results with code, metadata, scores, and optionally related code
        """
        # Stage 1: Vector search
        if initial_k is None:
            # Retrieve more initially if we're reranking (common practice)
            initial_k = top_k * 2 if self.use_reranking else top_k

        results = self.vector_retriever.retrieve(
            query=query,
            top_k=initial_k,
            filters=filters
        )

        if not results:
            return []

        # Stage 2: Graph augmentation (for top results only)
        if self.use_graph_augmentation and self.graph_augmenter:
            # Only augment the top few results to avoid slowdown
            top_results = results[:augment_top_n]
            other_results = results[augment_top_n:]

            augmented_top = self.graph_augmenter.augment(
                results=top_results,
                max_expansions=max_augmentations,
                depth=augment_depth
            )

            # Add empty related_code to non-augmented results for consistency
            for result in other_results:
                result['related_code'] = []

            results = augmented_top + other_results

        # Stage 3: Re-ranking
        if self.use_reranking and self.reranker:
            results = self.reranker.rerank(
                query=query,
                results=results,
                top_k=None  # Rerank all, filter later
            )

            # Stage 4: Hybrid scoring
            results = self.hybrid_scorer.score(results)

        # Return top_k final results
        return results[:top_k]

    def retrieve_with_context(
        self,
        query: str,
        top_k: int = 5,
        context_depth: int = 1
    ) -> Dict[str, Any]:
        """
        Retrieve code with additional context for LLM prompting.

        Args:
            query: Query text
            top_k: Number of results
            context_depth: Depth of graph traversal for context

        Returns:
            Dictionary with 'results', 'context_summary', and 'total_tokens' (estimate)
        """
        # Get results with augmentation
        results = self.retrieve(
            query=query,
            top_k=top_k,
            augment_top_n=top_k,  # Augment all results
            max_augmentations=5
        )

        # Build context summary
        context_summary = self._build_context_summary(results)

        # Estimate tokens (rough: 1 token ≈ 4 characters)
        total_chars = sum(len(r['content']) for r in results)
        for r in results:
            for rel in r.get('related_code', []):
                total_chars += len(rel.get('description', ''))

        estimated_tokens = total_chars // 4

        return {
            'results': results,
            'context_summary': context_summary,
            'estimated_tokens': estimated_tokens,
            'num_results': len(results)
        }

    def _build_context_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Build a summary of the retrieved context"""
        files_involved = set()
        types_seen = {}
        has_related = 0

        for result in results:
            file_path = result['metadata'].get('file', 'unknown')
            code_type = result['metadata'].get('type', 'unknown')

            files_involved.add(file_path)
            types_seen[code_type] = types_seen.get(code_type, 0) + 1

            if result.get('related_code'):
                has_related += 1

        return {
            'files_involved': list(files_involved),
            'num_files': len(files_involved),
            'types_distribution': types_seen,
            'results_with_relationships': has_related
        }

    def retrieve_for_symbol(
        self,
        symbol_name: str,
        top_k: int = 5,
        include_callers: bool = True,
        include_callees: bool = True
    ) -> Dict[str, Any]:
        """
        Retrieve code related to a specific symbol (function/class name).

        Args:
            symbol_name: Name of the symbol
            top_k: Number of results
            include_callers: Include functions that call this symbol
            include_callees: Include functions called by this symbol

        Returns:
            Dictionary with symbol definition and related code
        """
        # Search for the symbol
        results = self.retrieve(
            query=f"definition of {symbol_name}",
            top_k=top_k,
            augment_top_n=top_k
        )

        # Filter to most relevant result (likely the actual definition)
        if results:
            definition = results[0]

            # Get call graph relationships
            call_info = {}
            if self.graph_augmenter:
                if include_callees:
                    callees = self.graph_augmenter.get_call_chain(
                        symbol_name, direction='callees', max_depth=1
                    )
                    call_info['callees'] = callees

                if include_callers:
                    callers = self.graph_augmenter.get_call_chain(
                        symbol_name, direction='callers', max_depth=1
                    )
                    call_info['callers'] = callers

            return {
                'definition': definition,
                'related_results': results[1:],
                'call_graph': call_info
            }

        return {'definition': None, 'related_results': [], 'call_graph': {}}

    def retrieve_adaptive(
        self,
        query: str,
        user_preferences: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        **V2.1 Dynamic Context Assembly**

        Adaptive retrieval that analyzes the query and dynamically determines:
        - How many chunks to retrieve
        - Quality threshold for inclusion
        - Whether to use graph expansion
        - Context token budget

        This is the recommended method for production use.

        Args:
            query: Natural language or code query
            user_preferences: Optional overrides (e.g., {'force_complex': True, 'max_chunks': 30})

        Returns:
            Dict with:
                - 'context': ContextBuildResult (formatted context, tokens, metrics)
                - 'query_analysis': Dict (query type, complexity, strategy)
                - 'raw_results': List[Dict] (all retrieved chunks for debugging)
        """
        # Step 1: Analyze query
        analysis = self.query_analyzer.analyze(query)

        # Apply user preferences if provided
        if user_preferences:
            if 'initial_k' in user_preferences:
                analysis['initial_k'] = user_preferences['initial_k']
            if 'max_k' in user_preferences:
                analysis['max_k'] = user_preferences['max_k']
            if 'quality_threshold' in user_preferences:
                analysis['quality_threshold'] = user_preferences['quality_threshold']

        # Step 2: Decide retrieval strategy based on query type
        # V2.2: Use graph-expansion for architectural/explanation/call_stack queries
        use_graph_expansion = (
            analysis['type'] in [QueryType.ARCHITECTURE, QueryType.EXPLANATION, QueryType.CALL_STACK] and
            self.graph_expansion_retriever is not None
        )

        if use_graph_expansion:
            # Adjust expansion depth based on query type
            if analysis['type'] == QueryType.CALL_STACK:
                expansion_depth = 5  # Deep tracing for call stacks
                semantic_weight = 0.3
                graph_weight = 0.7  # Prioritize graph relationships for call chains
            else:
                expansion_depth = 2
                semantic_weight = 0.4
                graph_weight = 0.6

            raw_results = self.graph_expansion_retriever.retrieve_with_expansion(
                query=query,
                seed_k=analysis['initial_k'],
                max_total=analysis['max_k'],
                expansion_depth=expansion_depth,
                semantic_weight=semantic_weight,
                graph_weight=graph_weight,
                prioritize_cross_language=True
            )
        else:
            # Standard semantic retrieval with augmentation
            # Use analysis['max_graph_depth'] for augmentation depth
            augment_depth = analysis.get('max_graph_depth', 1)

            raw_results = self.retrieve(
                query=query,
                top_k=analysis['max_k'],
                initial_k=analysis['initial_k'] * 2,  # Over-fetch for reranking
                augment_top_n=min(analysis['initial_k'], 10) if analysis['use_graph_expansion'] else 0,
                max_augmentations=10 if analysis['use_graph_expansion'] else 0,
                augment_depth=augment_depth  # Deep traversal for call stacks
            )

        if not raw_results:
            return {
                'context': ContextBuildResult(
                    context_text="No relevant code found for this query.",
                    chunks_included=0,
                    tokens_used=0,
                    quality_metrics={'error': 'no_results'},
                    truncated=False
                ),
                'query_analysis': analysis,
                'raw_results': []
            }

        # Step 3: Build context adaptively with quality threshold
        context_result = self.context_builder.build_context_iterative(
            results=raw_results,
            initial_quality_threshold=analysis['quality_threshold'],
            min_chunks=3,  # Always try to get at least 3 chunks
            max_chunks=analysis['max_k'],
            include_related=analysis['use_graph_expansion']
        )

        # Step 4: Return comprehensive result
        return {
            'context': context_result,
            'query_analysis': {
                'query': analysis['query'],
                'type': analysis['type'].value,
                'complexity': analysis['complexity'].value,
                'initial_k': analysis['initial_k'],
                'max_k': analysis['max_k'],
                'quality_threshold': analysis['quality_threshold'],
                'use_graph_expansion': analysis['use_graph_expansion']
            },
            'raw_results': raw_results  # For debugging/inspection
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the retriever and index"""
        stats = self.vector_retriever.get_stats()
        stats['hybrid_config'] = {
            'use_reranking': self.use_reranking,
            'use_graph_augmentation': self.use_graph_augmentation,
            'max_model_len': self.max_model_len,
            'adaptive_context': True
        }
        return stats
