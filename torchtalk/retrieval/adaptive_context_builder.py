#!/usr/bin/env python3
"""
Adaptive context builder that dynamically assembles context based on query analysis.

This module builds context with:
- Quality-based filtering (threshold scoring)
- Token budget awareness
- Incremental expansion if needed
- Deduplication and relevance ranking
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class ContextBuildResult:
    """Result of context building"""
    context_text: str              # Formatted context for LLM
    chunks_included: int           # Number of chunks included
    tokens_used: int              # Estimated tokens
    quality_metrics: Dict[str, Any]  # Quality info
    truncated: bool               # Whether context was truncated


class AdaptiveContextBuilder:
    """
    Builds context dynamically based on query analysis and token budgets.
    """

    def __init__(self, max_model_len: int):
        """
        Args:
            max_model_len: Maximum context length of the model
        """
        self.max_model_len = max_model_len

        # Reserve tokens for system prompt, user message, and response
        self.system_prompt_tokens = 200  # Rough estimate
        self.user_message_tokens = 500   # Rough estimate
        self.response_tokens = 2000      # Reserve for response

        # Available for code context
        self.max_context_tokens = max_model_len - (
            self.system_prompt_tokens +
            self.user_message_tokens +
            self.response_tokens
        )

    def build_context(
        self,
        results: List[Dict[str, Any]],
        quality_threshold: float,
        token_budget: Optional[int] = None,
        include_related: bool = True
    ) -> ContextBuildResult:
        """
        Build context from retrieval results with quality filtering.

        Args:
            results: Retrieved chunks from HybridRetriever
            quality_threshold: Minimum score to include (0.0-1.0)
            token_budget: Optional override for context token budget
            include_related: Whether to include graph-augmented related code

        Returns:
            ContextBuildResult with formatted context and metrics
        """
        if token_budget is None:
            token_budget = self.max_context_tokens

        context_parts = []
        tokens_used = 0
        chunks_included = 0
        scores = []

        # Filter by quality threshold first
        filtered_results = [
            r for r in results
            if self._get_score(r) >= quality_threshold
        ]

        if not filtered_results:
            # If threshold too strict, take top results anyway
            filtered_results = results[:5]

        # Add chunks until budget exhausted
        for result in filtered_results:
            chunk_text, chunk_tokens = self._format_chunk(result, include_related)

            # Check if adding this chunk would exceed budget
            if tokens_used + chunk_tokens > token_budget:
                # Try without related code if that would fit
                if include_related and result.get('related_code'):
                    chunk_text_minimal, chunk_tokens_minimal = self._format_chunk(
                        result, include_related=False
                    )
                    if tokens_used + chunk_tokens_minimal <= token_budget:
                        context_parts.append(chunk_text_minimal)
                        tokens_used += chunk_tokens_minimal
                        chunks_included += 1
                        scores.append(self._get_score(result))
                break  # Can't fit more

            context_parts.append(chunk_text)
            tokens_used += chunk_tokens
            chunks_included += 1
            scores.append(self._get_score(result))

        # Format final context
        context_text = self._format_final_context(context_parts)

        # Calculate quality metrics
        quality_metrics = {
            'avg_score': sum(scores) / len(scores) if scores else 0.0,
            'min_score': min(scores) if scores else 0.0,
            'max_score': max(scores) if scores else 0.0,
            'total_available': len(results),
            'after_filtering': len(filtered_results),
            'threshold_used': quality_threshold
        }

        truncated = chunks_included < len(filtered_results)

        return ContextBuildResult(
            context_text=context_text,
            chunks_included=chunks_included,
            tokens_used=tokens_used,
            quality_metrics=quality_metrics,
            truncated=truncated
        )

    def build_context_iterative(
        self,
        results: List[Dict[str, Any]],
        initial_quality_threshold: float,
        min_chunks: int = 3,
        max_chunks: int = 50,
        include_related: bool = True
    ) -> ContextBuildResult:
        """
        Build context iteratively, relaxing quality threshold if not enough chunks.

        This ensures we always get meaningful context even for obscure queries.

        Args:
            results: Retrieved chunks
            initial_quality_threshold: Starting quality threshold
            min_chunks: Minimum chunks to include (will relax threshold to achieve)
            max_chunks: Maximum chunks regardless of quality
            include_related: Whether to include related code

        Returns:
            ContextBuildResult with adaptive quality threshold
        """
        threshold = initial_quality_threshold

        # Try with initial threshold
        context_result = self.build_context(
            results=results,
            quality_threshold=threshold,
            include_related=include_related
        )

        # If not enough chunks, relax threshold
        iterations = 0
        while context_result.chunks_included < min_chunks and threshold > 0.1 and iterations < 5:
            threshold -= 0.1
            context_result = self.build_context(
                results=results,
                quality_threshold=threshold,
                include_related=include_related
            )
            iterations += 1

        # Cap at max_chunks
        if context_result.chunks_included > max_chunks:
            context_result = self.build_context(
                results=results[:max_chunks],
                quality_threshold=threshold,
                include_related=include_related
            )

        return context_result

    def _format_chunk(
        self,
        result: Dict[str, Any],
        include_related: bool
    ) -> tuple[str, int]:
        """Format a single chunk with metadata. Returns (text, token_estimate)"""
        meta = result['metadata']
        score = self._get_score(result)

        parts = []
        parts.append(f"## {meta.get('name', 'N/A')}")

        # Format file location using industry-standard file:line-line format
        # PRIORITIZE this for easy reference
        file_path = meta.get('file', 'unknown')
        line_start = meta.get('line_start')
        line_end = meta.get('line_end')

        if line_start and line_end:
            parts.append(f"**Location:** `{file_path}:{line_start}-{line_end}`")
        elif line_start:
            parts.append(f"**Location:** `{file_path}:{line_start}`")
        else:
            parts.append(f"**Location:** `{file_path}`")

        parts.append(f"**Type:** {meta.get('type', 'unknown')} | **Language:** {meta.get('language', 'python')} | **Relevance:** {score:.3f}")
        parts.append("")
        parts.append("```" + meta.get('language', 'python'))
        parts.append(result['content'])
        parts.append("```")

        # Add related code if available - with depth information for call stacks
        if include_related and result.get('related_code'):
            parts.append("")
            parts.append("### Related Code:")
            for rel in result['related_code'][:5]:  # Show more for deep tracing
                depth_info = f" (depth {rel['depth']})" if 'depth' in rel else ""
                parts.append(f"  - {rel['description']}{depth_info}")

        parts.append("")  # Blank line separator

        text = "\n".join(parts)
        tokens = self._estimate_tokens(text)

        return text, tokens

    def _format_final_context(self, context_parts: List[str]) -> str:
        """Format the final context with header"""
        if not context_parts:
            return "No relevant code found."

        header = [
            "# Relevant Code Context",
            f"Found {len(context_parts)} relevant code sections:",
            ""
        ]

        return "\n".join(header + context_parts)

    def _get_score(self, result: Dict[str, Any]) -> float:
        """Extract score from result (handles various score keys)"""
        return result.get(
            'final_score',
            result.get('rerank_score', result.get('score', 0.0))
        )

    def _estimate_tokens(self, text: str) -> int:
        """Estimate tokens from text (rough: 4 chars per token)"""
        return len(text) // 4


