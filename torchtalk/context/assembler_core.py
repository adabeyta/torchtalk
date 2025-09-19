#!/usr/bin/env python3
import os
from typing import Dict, Any
from torchtalk.context.graph_retriever import GraphContextRetriever
from torchtalk.context.semantic_search import SemanticContentFinder


class ContextAssemblerCore:
    def __init__(self, enhanced_analysis_path: str):
        if not os.path.exists(enhanced_analysis_path):
            raise FileNotFoundError(f"Enhanced analysis not found: {enhanced_analysis_path}")

        self.enhanced_analysis_path = enhanced_analysis_path

        # Initialize retrieval components
        self.graph_retriever = GraphContextRetriever(enhanced_analysis_path)
        self.semantic_finder = SemanticContentFinder(enhanced_analysis_path)

    def assemble_context(self, question: str, max_chars: int = 3200000) -> str:
        return self._assemble_enhanced_context(question, max_chars)

    def _assemble_enhanced_context(self, question: str, max_chars: int) -> str:
        context_parts = []

        # Get graph-based context (import/call relationships)
        graph_context = self.graph_retriever.get_context_for_query(question, max_tokens=max_chars//4)
        if graph_context:
            context_parts.append("# Graph-Based Context\n" + graph_context)

        # Get semantic matches (pattern and similarity based)
        semantic_matches = self.semantic_finder.find_semantic_matches(question, max_results=10)

        if semantic_matches:
            context_parts.append("\n# Semantic Matches\n")
            for match in semantic_matches[:5]:
                context_parts.append(self._format_semantic_match(match))

        # Combine and trim to size
        full_context = "\n".join(context_parts)

        if len(full_context) > max_chars:
            # Intelligent truncation, keep most relevant parts (could use some work here)
            return self._intelligent_truncate(full_context, max_chars, question)

        return full_context

    def _format_semantic_match(self, match) -> str:
        formatted = f"\n## {match.content_type.title()}: Relevance {match.score:.1f}\n"
        formatted += f"**Match Reasons**: {', '.join(match.match_reasons)}\n"

        content = match.content_data
        if content.get('signature'):
            formatted += f"\n### Signature\n```python\n{content['signature']}\n```\n"

        if content.get('docstring'):
            formatted += f"\n### Documentation\n{content['docstring'][:500]}\n"

        if content.get('source_code') or content.get('source_preview'):
            code = content.get('source_preview') or content.get('source_code', '')[:800]
            if code:
                formatted += f"\n### Code\n```python\n{code}\n```\n"

        return formatted

    def _intelligent_truncate(self, context: str, max_chars: int, question: str) -> str:
        # Split into sections
        sections = context.split('\n#')

        # Prioritize sections
        prioritized = []
        high_priority_keywords = ['semantic', 'exact match', 'high score']

        for section in sections:
            priority = 1.0
            section_lower = section.lower()

            # Check for high priority indicators
            for keyword in high_priority_keywords:
                if keyword in section_lower:
                    priority *= 1.5

            # Check for query terms
            for term in question.lower().split():
                if len(term) > 3 and term in section_lower:
                    priority *= 1.2

            prioritized.append((section, priority))

        # Sort by priority
        prioritized.sort(key=lambda x: x[1], reverse=True)

        # Rebuild context with priority
        result = []
        chars_used = 0

        for section, priority in prioritized:
            section_with_header = '#' + section if not section.startswith('#') else section
            section_chars = len(section_with_header)

            if chars_used + section_chars <= max_chars * 0.95:  # Leave 5% buffer, can adjust.
                result.append(section_with_header)
                chars_used += section_chars
            elif chars_used < max_chars * 0.9:  # Try to include partial
                remaining = max_chars - chars_used - 100
                if remaining > 500:  # Only include if meaningful
                    truncated = section_with_header[:remaining] + "\n\n[TRUNCATED]"
                    result.append(truncated)
                    break

        return '\n'.join(result)

    def get_context_info(self, question: str) -> Dict[str, Any]:
        info = {
            'mode': 'enhanced',
            'available_retrievers': ['graph_based', 'semantic_search'],
            'question': question,
            'relevant_items': []  # Expected by adaptive response manager
        }

        # Get retrieval stats
        context_items = self.graph_retriever.find_relevant_context(question, max_items=5)
        info['graph_matches'] = len(context_items)
        info['top_graph_matches'] = [
            {'type': item.type, 'score': item.relevance_score, 'reason': item.reason}
            for item in context_items[:3]
        ]

        # Add graph items to relevant_items in expected format: (item_type, score)
        for item in context_items:
            info['relevant_items'].append((item.type, item.relevance_score))

        semantic_matches = self.semantic_finder.find_semantic_matches(question, max_results=5)
        info['semantic_matches'] = len(semantic_matches)
        info['top_semantic_matches'] = [
            {'type': match.content_type, 'score': match.score, 'reasons': match.match_reasons}
            for match in semantic_matches[:3]
        ]

        # Add semantic items to relevant_items
        for match in semantic_matches:
            info['relevant_items'].append((match.content_type, match.score))

        return info