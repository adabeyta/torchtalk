#!/usr/bin/env python3
"""
Query analyzer for determining retrieval strategy and complexity.

This module analyzes incoming queries to determine:
- Query complexity (simple, moderate, complex)
- Query type (definition, explanation, debugging, architecture)
- Recommended retrieval budget
"""

from typing import Dict, List, Tuple
from enum import Enum
import re


class QueryComplexity(Enum):
    SIMPLE = "simple"           # Single concept lookup (10-20 chunks)
    MODERATE = "moderate"       # Related concepts (20-50 chunks)
    COMPLEX = "complex"         # Multi-faceted analysis (50-100 chunks)


class QueryType(Enum):
    DEFINITION = "definition"         # "What is X?"
    LOCATION = "location"            # "Where is X located?" - needs file paths/line numbers
    USAGE = "usage"                  # "How do I use X?"
    EXPLANATION = "explanation"      # "How does X work?"
    CALL_STACK = "call_stack"        # "Show me the call stack" - needs deep graph traversal
    COMPARISON = "comparison"        # "X vs Y"
    DEBUGGING = "debugging"          # "Why doesn't X work?"
    ARCHITECTURE = "architecture"    # "How is X organized?"


class QueryAnalyzer:
    """
    Analyzes queries to determine optimal retrieval strategy.
    """

    # Keywords for query type classification
    TYPE_PATTERNS = {
        QueryType.LOCATION: [
            r"\bwhere is\b", r"\bwhere are\b", r"\blocated\b", r"\bfind\b.*\bfile\b",
            r"\bshow me where\b", r"\bwhich file\b", r"\bin what file\b",
            r"\bfile location\b", r"\bsource location\b"
        ],
        QueryType.CALL_STACK: [
            r"\bcall stack\b", r"\bcall chain\b", r"\bcall graph\b", r"\bcall flow\b",
            r"\bexecution flow\b", r"\btrace\b.*\bcall\b", r"\bstack trace\b",
            r"\bcode execution\b", r"\bgeneral call stack\b", r"\bsystem overview\b"
        ],
        QueryType.DEFINITION: [
            r"\bwhat is\b", r"\bdefine\b", r"\bdefinition of\b",
            r"\bwhat does\b.*\bdo\b", r"\bmeans?\b"
        ],
        QueryType.USAGE: [
            r"\bhow (do|can) (i|we|you)\b", r"\buse\b", r"\bexample of\b",
            r"\bcode for\b", r"\bimplement\b"
        ],
        QueryType.EXPLANATION: [
            r"\bhow does\b", r"\bwhy does\b", r"\bexplain\b",
            r"\bworks?\b", r"\bprocess\b", r"\bmechanism\b"
        ],
        QueryType.COMPARISON: [
            r"\bvs\b", r"\bversus\b", r"\bcompare\b", r"\bdifference between\b",
            r"\bbetter\b.*\bor\b"
        ],
        QueryType.DEBUGGING: [
            r"\berror\b", r"\bfails?\b", r"\bwhy (doesn't|does not|won't)\b",
            r"\bnot working\b", r"\bissue with\b", r"\bbug\b"
        ],
        QueryType.ARCHITECTURE: [
            r"\barchitecture\b", r"\bstructure\b", r"\borganized\b",
            r"\bdesign\b", r"\bhow .* (laid out|structured|organized)\b"
        ]
    }

    # Complexity indicators
    COMPLEXITY_INDICATORS = {
        'multiple_concepts': r"\b(and|or|with|between)\b",  # Multiple entities
        'distributed': r"\b(distributed|parallel|multi)\b",  # Distributed concepts
        'interaction': r"\b(interact|integrate|work together|communicate)\b",
        'deep_dive': r"\b(internal|implementation|under the hood|deep)\b",
        'broad': r"\b(all|every|entire|overall|whole)\b"
    }

    def __init__(self):
        # Compile patterns for efficiency
        self.type_patterns_compiled = {
            qtype: [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
            for qtype, patterns in self.TYPE_PATTERNS.items()
        }
        self.complexity_patterns_compiled = {
            indicator: re.compile(pattern, re.IGNORECASE)
            for indicator, pattern in self.COMPLEXITY_INDICATORS.items()
        }

    def analyze(self, query: str) -> Dict:
        """
        Analyze a query and return retrieval strategy.

        Returns:
            Dict with 'type', 'complexity', 'initial_k', 'max_k', 'quality_threshold'
        """
        query_type = self._classify_type(query)
        complexity = self._assess_complexity(query)

        # Determine retrieval budget based on complexity
        initial_k, max_k, quality_threshold = self._get_retrieval_budget(
            query_type, complexity
        )

        # Determine graph expansion settings
        use_graph_expansion = complexity != QueryComplexity.SIMPLE

        # Deep graph traversal for call stack queries
        if query_type == QueryType.CALL_STACK:
            max_graph_depth = 5  # Trace through multiple layers
            use_graph_expansion = True
        elif query_type == QueryType.ARCHITECTURE:
            max_graph_depth = 3
        elif complexity == QueryComplexity.COMPLEX:
            max_graph_depth = 2
        else:
            max_graph_depth = 1

        return {
            'query': query,
            'type': query_type,
            'complexity': complexity,
            'initial_k': initial_k,          # Initial retrieval count
            'max_k': max_k,                  # Maximum chunks to consider
            'quality_threshold': quality_threshold,  # Min score to include
            'use_graph_expansion': use_graph_expansion,
            'max_graph_depth': max_graph_depth
        }

    def _classify_type(self, query: str) -> QueryType:
        """Classify the type of query"""
        scores = {}

        for qtype, patterns in self.type_patterns_compiled.items():
            matches = sum(1 for pattern in patterns if pattern.search(query))
            if matches > 0:
                scores[qtype] = matches

        if scores:
            # Return type with most pattern matches
            return max(scores, key=scores.get)

        # Default: EXPLANATION for longer queries, DEFINITION for short
        return QueryType.EXPLANATION if len(query.split()) > 5 else QueryType.DEFINITION

    def _assess_complexity(self, query: str) -> QueryComplexity:
        """Assess query complexity based on indicators"""

        # Word count as baseline
        word_count = len(query.split())

        # Check complexity indicators
        complexity_score = 0
        for indicator, pattern in self.complexity_patterns_compiled.items():
            if pattern.search(query):
                complexity_score += 1

        # Count entities/concepts (rough heuristic: capitalized words, technical terms)
        entities = len(re.findall(r'\b[A-Z][a-z]+(?:\.[A-Z][a-z]+)*\b', query))

        # Scoring logic
        if word_count <= 5 and complexity_score == 0 and entities <= 1:
            return QueryComplexity.SIMPLE
        elif word_count > 15 or complexity_score >= 2 or entities >= 3:
            return QueryComplexity.COMPLEX
        else:
            return QueryComplexity.MODERATE

    def _get_retrieval_budget(
        self,
        query_type: QueryType,
        complexity: QueryComplexity
    ) -> Tuple[int, int, float]:
        """
        Determine retrieval budget: (initial_k, max_k, quality_threshold)

        Returns:
            - initial_k: Number of chunks to retrieve initially
            - max_k: Maximum chunks to consider after expansion
            - quality_threshold: Minimum relevance score to include (0.0-1.0)
        """

        # Base budgets by complexity
        budgets = {
            QueryComplexity.SIMPLE: (10, 20, 0.5),
            QueryComplexity.MODERATE: (20, 50, 0.4),
            QueryComplexity.COMPLEX: (40, 100, 0.3)
        }

        initial_k, max_k, threshold = budgets[complexity]

        # Adjust based on query type
        if query_type == QueryType.LOCATION:
            # Location queries need precise, high-quality matches with file paths
            initial_k = min(initial_k, 5)
            max_k = min(max_k, 15)
            threshold = max(threshold, 0.7)  # Very high precision

        elif query_type == QueryType.CALL_STACK:
            # Call stack queries need extensive context with deep graph traversal
            initial_k = int(initial_k * 2)
            max_k = int(max_k * 2)
            threshold = max(threshold - 0.15, 0.2)  # Lower threshold for comprehensive tracing

        elif query_type == QueryType.DEFINITION:
            # Definitions need fewer but higher quality results
            initial_k = min(initial_k, 10)
            max_k = min(max_k, 20)
            threshold = max(threshold, 0.6)

        elif query_type == QueryType.ARCHITECTURE:
            # Architecture questions benefit from more context
            initial_k = int(initial_k * 1.5)
            max_k = int(max_k * 1.5)
            threshold = max(threshold - 0.1, 0.2)

        elif query_type == QueryType.DEBUGGING:
            # Debugging needs diverse context
            initial_k = int(initial_k * 1.2)
            max_k = int(max_k * 1.3)
            threshold = max(threshold - 0.05, 0.25)

        return initial_k, max_k, threshold


# Convenience function
def analyze_query(query: str) -> Dict:
    """Quick function to analyze a query"""
    analyzer = QueryAnalyzer()
    return analyzer.analyze(query)


