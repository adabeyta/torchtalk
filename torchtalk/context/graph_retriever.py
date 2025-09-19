#!/usr/bin/env python3
import json
import re
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple, Any
from collections import defaultdict, Counter
import networkx as nx
from dataclasses import dataclass
import math


@dataclass
class ContextItem:
    type: str  # 'module', 'function', 'class', 'code_block'
    identifier: str
    relevance_score: float
    content: Dict[str, Any]
    reason: str  # Why this was included
    source_path: str


class GraphContextRetriever:
    def __init__(self, enhanced_analysis_path: str):
        self.analysis_path = enhanced_analysis_path
        self.analysis_data = self._load_analysis()

        # Rebuild graphs from saved data
        self.import_graph = self._rebuild_import_graph()
        self.call_graph = self._rebuild_call_graph()

        # Build semantic indices
        self.keyword_index = self._build_keyword_index()
        self.symbol_index = self._build_symbol_index()
        self.code_embeddings = {}  # Could add vector embeddings here

    def _load_analysis(self) -> Dict[str, Any]:
        with open(self.analysis_path, 'r') as f:
            return json.load(f)

    def _rebuild_import_graph(self) -> nx.DiGraph:
        graph = nx.DiGraph()

        modules = self.analysis_data.get('modules_with_code', {})
        for module_path, module_info in modules.items():
            graph.add_node(module_info['path'])

            # Add import edges
            for imported in module_info.get('imports', []):
                # Try to find this import in our modules
                for other_path, other_info in modules.items():
                    if imported in other_info['path'] or imported == other_info['name']:
                        graph.add_edge(module_info['path'], other_info['path'])
                        break

        return graph

    def _rebuild_call_graph(self) -> nx.DiGraph:
        graph = nx.DiGraph()

        modules = self.analysis_data.get('modules_with_code', {})
        for module_path, module_info in modules.items():
            module_name = module_info['name']

            # Add function nodes and their calls
            for func_name, func_data in module_info.get('functions', {}).items():
                node_id = f"{module_name}.{func_name}"
                graph.add_node(node_id, type='function', module=module_info['path'])

                # Add call edges
                for called in func_data.get('calls', []):
                    called_id = f"{module_name}.{called}"
                    graph.add_edge(node_id, called_id)

            # Add class nodes and their method calls
            for class_name, class_data in module_info.get('classes', {}).items():
                node_id = f"{module_name}.{class_name}"
                graph.add_node(node_id, type='class', module=module_info['path'])

                for called in class_data.get('calls', []):
                    called_id = f"{module_name}.{called}"
                    graph.add_edge(node_id, called_id)

        return graph

    def _build_keyword_index(self) -> Dict[str, List[str]]:
        index = defaultdict(list)

        modules = self.analysis_data.get('modules_with_code', {})
        for module_path, module_info in modules.items():
            for keyword in module_info.get('keywords', []):
                index[keyword.lower()].append(module_info['path'])

        return dict(index)

    def _build_symbol_index(self) -> Dict[str, List[Tuple[str, str]]]:
        index = defaultdict(list)

        modules = self.analysis_data.get('modules_with_code', {})
        for module_path, module_info in modules.items():
            module_name = module_info['name']

            # Index classes
            for class_name in module_info.get('classes', {}).keys():
                index[class_name.lower()].append((module_info['path'], 'class'))

            # Index functions
            for func_name in module_info.get('functions', {}).keys():
                index[func_name.lower()].append((module_info['path'], 'function'))

        return dict(index)

    def find_relevant_context(self, query: str, max_items: int = 20) -> List[ContextItem]:
        context_items = []
        seen_identifiers = set()

        # Direct symbol matches
        symbol_matches = self._find_symbol_matches(query)
        for item in symbol_matches:
            if item.identifier not in seen_identifiers:
                context_items.append(item)
                seen_identifiers.add(item.identifier)

        # Keyword-based module discovery
        keyword_matches = self._find_keyword_matches(query)
        for item in keyword_matches:
            if item.identifier not in seen_identifiers:
                context_items.append(item)
                seen_identifiers.add(item.identifier)

        # Graph traversal from initial matches
        if context_items:
            graph_expansions = self._expand_via_graphs(context_items[:5])
            for item in graph_expansions:
                if item.identifier not in seen_identifiers:
                    context_items.append(item)
                    seen_identifiers.add(item.identifier)

        # Semantic similarity search
        semantic_matches = self._find_semantic_matches(query, exclude=seen_identifiers)
        for item in semantic_matches:
            if item.identifier not in seen_identifiers:
                context_items.append(item)
                seen_identifiers.add(item.identifier)

        # Sort by relevance and return top items
        context_items.sort(key=lambda x: x.relevance_score, reverse=True)
        return context_items[:max_items]

    def _find_symbol_matches(self, query: str) -> List[ContextItem]:
        matches = []
        query_lower = query.lower()
        query_words = re.findall(r'\b\w+\b', query_lower)

        for word in query_words:
            if word in self.symbol_index:
                for module_path, symbol_type in self.symbol_index[word]:
                    # Get the actual symbol data
                    module_data = self._get_module_data(module_path)
                    if not module_data:
                        continue

                    if symbol_type == 'class':
                        for class_name, class_data in module_data.get('classes', {}).items():
                            if class_name.lower() == word:
                                matches.append(ContextItem(
                                    type='class',
                                    identifier=f"{module_path}::{class_name}",
                                    relevance_score=10.0,  # High score for exact match
                                    content=class_data,
                                    reason=f"Exact match for class '{class_name}'",
                                    source_path=module_path
                                ))

                    elif symbol_type == 'function':
                        for func_name, func_data in module_data.get('functions', {}).items():
                            if func_name.lower() == word:
                                matches.append(ContextItem(
                                    type='function',
                                    identifier=f"{module_path}::{func_name}",
                                    relevance_score=10.0,
                                    content=func_data,
                                    reason=f"Exact match for function '{func_name}'",
                                    source_path=module_path
                                ))

        return matches

    def _find_keyword_matches(self, query: str) -> List[ContextItem]:
        matches = []
        query_lower = query.lower()
        query_words = set(re.findall(r'\b\w+\b', query_lower))

        # Score modules by keyword overlap
        module_scores = defaultdict(float)
        for word in query_words:
            if word in self.keyword_index:
                for module_path in self.keyword_index[word]:
                    module_scores[module_path] += 1.0

        # Create context items for top modules
        for module_path, score in sorted(module_scores.items(), key=lambda x: x[1], reverse=True)[:5]:
            module_data = self._get_module_data(module_path)
            if module_data:
                matches.append(ContextItem(
                    type='module',
                    identifier=module_path,
                    relevance_score=score * 2,  # Boost keyword matches
                    content=module_data,
                    reason=f"Keyword match (score: {score:.1f})",
                    source_path=module_path
                ))

        return matches

    def _expand_via_graphs(self, initial_items: List[ContextItem]) -> List[ContextItem]:
        expansions = []

        for item in initial_items:
            if item.type == 'module':
                # Find closely related modules via import graph
                related = self._find_related_via_imports(item.source_path)
                for module_path, relationship, score in related[:3]:
                    module_data = self._get_module_data(module_path)
                    if module_data:
                        expansions.append(ContextItem(
                            type='module',
                            identifier=module_path,
                            relevance_score=item.relevance_score * score,
                            content=module_data,
                            reason=f"{relationship} of {item.source_path}",
                            source_path=module_path
                        ))

            elif item.type in ['function', 'class']:
                # Find related functions via call graph
                entity_name = item.identifier.split('::')[-1]
                module_name = Path(item.source_path).stem
                node_id = f"{module_name}.{entity_name}"

                related = self._find_related_via_calls(node_id)
                for related_id, relationship, score in related[:3]:
                    # Parse the related ID to get module and entity
                    parts = related_id.split('.')
                    if len(parts) >= 2:
                        related_entity = parts[-1]
                        related_module = '.'.join(parts[:-1])

                        # Find the module containing this entity
                        for module_path, module_data in self.analysis_data.get('modules_with_code', {}).items():
                            if module_data['name'] == related_module:
                                # Get the specific entity data
                                entity_data = None
                                entity_type = None

                                if related_entity in module_data.get('functions', {}):
                                    entity_data = module_data['functions'][related_entity]
                                    entity_type = 'function'
                                elif related_entity in module_data.get('classes', {}):
                                    entity_data = module_data['classes'][related_entity]
                                    entity_type = 'class'

                                if entity_data:
                                    expansions.append(ContextItem(
                                        type=entity_type,
                                        identifier=f"{module_data['path']}::{related_entity}",
                                        relevance_score=item.relevance_score * score * 0.7,
                                        content=entity_data,
                                        reason=f"{relationship} {entity_name}",
                                        source_path=module_data['path']
                                    ))
                                break

        return expansions

    def _find_related_via_imports(self, module_path: str) -> List[Tuple[str, str, float]]:
        related = []

        if module_path in self.import_graph:
            # Direct imports
            for imported in self.import_graph.successors(module_path):
                related.append((imported, "Imported by", 0.8))

            # Direct importers
            for importer in self.import_graph.predecessors(module_path):
                related.append((importer, "Imports", 0.7))

            # Modules imported by the same parent
            for importer in self.import_graph.predecessors(module_path):
                for sibling in self.import_graph.successors(importer):
                    if sibling != module_path:
                        related.append((sibling, "Co-imported with", 0.5))

        return related

    def _find_related_via_calls(self, node_id: str) -> List[Tuple[str, str, float]]:
        related = []

        if node_id in self.call_graph:
            # Functions this one calls
            for called in self.call_graph.successors(node_id):
                related.append((called, "Called by", 0.9))

            # Functions that call this one
            for caller in self.call_graph.predecessors(node_id):
                related.append((caller, "Calls", 0.8))

        return related

    def _find_semantic_matches(self, query: str, exclude: Set[str]) -> List[ContextItem]:
        matches = []
        query_lower = query.lower()

        # Look for patterns in the query
        patterns = self._extract_query_patterns(query_lower)

        modules = self.analysis_data.get('modules_with_code', {})
        for module_path, module_data in modules.items():
            if module_path in exclude:
                continue

            # Score based on docstring similarity
            score = 0.0

            # Check functions
            for func_name, func_data in module_data.get('functions', {}).items():
                func_id = f"{module_path}::{func_name}"
                if func_id in exclude:
                    continue

                if func_data.get('docstring'):
                    doc_lower = func_data['docstring'].lower()
                    pattern_score = self._score_patterns(doc_lower, patterns)
                    if pattern_score > 0.3:
                        matches.append(ContextItem(
                            type='function',
                            identifier=func_id,
                            relevance_score=pattern_score * 5,
                            content=func_data,
                            reason=f"Semantic similarity in docstring",
                            source_path=module_path
                        ))

            # Check classes
            for class_name, class_data in module_data.get('classes', {}).items():
                class_id = f"{module_path}::{class_name}"
                if class_id in exclude:
                    continue

                if class_data.get('docstring'):
                    doc_lower = class_data['docstring'].lower()
                    pattern_score = self._score_patterns(doc_lower, patterns)
                    if pattern_score > 0.3:
                        matches.append(ContextItem(
                            type='class',
                            identifier=class_id,
                            relevance_score=pattern_score * 5,
                            content=class_data,
                            reason=f"Semantic similarity in docstring",
                            source_path=module_path
                        ))

        return matches

    def _extract_query_patterns(self, query: str) -> List[str]:
        patterns = []

        # Extract technical terms
        tech_terms = re.findall(r'\b(?:implement|create|build|process|handle|compute|calculate|transform|convert|parse|serialize|optimize|train|infer|forward|backward|gradient|loss|tensor|layer|network|model|data|batch)\w*\b', query)
        patterns.extend(tech_terms)

        # Extract noun phrases (simplified)
        noun_phrases = re.findall(r'\b(?:the\s+)?(\w+(?:\s+\w+)?)\s+(?:of|in|for|with|from|to)\b', query)
        patterns.extend(noun_phrases)

        return patterns

    def _score_patterns(self, text: str, patterns: List[str]) -> float:
        if not patterns:
            return 0.0

        matches = 0
        for pattern in patterns:
            if pattern in text:
                matches += 1

        return matches / len(patterns)

    def _get_module_data(self, module_path: str) -> Optional[Dict[str, Any]]:
        modules = self.analysis_data.get('modules_with_code', {})
        for path, data in modules.items():
            if data.get('path') == module_path:
                return data
        return None

    def get_context_for_query(self, query: str, max_tokens: int = 100000) -> str:
        context_items = self.find_relevant_context(query)

        formatted_context = []
        formatted_context.append("# Relevant Code Context\n")
        formatted_context.append(f"Query: {query}\n")
        formatted_context.append("=" * 80 + "\n")

        tokens_used = 0

        for item in context_items:
            # Format based on item type
            if item.type == 'module':
                block = self._format_module_context(item)
            elif item.type == 'function':
                block = self._format_function_context(item)
            elif item.type == 'class':
                block = self._format_class_context(item)
            else:
                continue

            # Simple token estimation (4 chars = 1 token)
            block_tokens = len(block) // 4
            if tokens_used + block_tokens > max_tokens:
                break

            formatted_context.append(block)
            tokens_used += block_tokens

        return '\n'.join(formatted_context)

    def _format_module_context(self, item: ContextItem) -> str:
        content = item.content

        formatted = f"\n## Module: {content.get('name', 'Unknown')}\n"
        formatted += f"**Path**: `{item.source_path}`\n"
        formatted += f"**Relevance**: {item.reason}\n"
        formatted += f"**Score**: {item.relevance_score:.1f}\n\n"

        # Add key classes
        if content.get('classes'):
            formatted += "### Key Classes:\n"
            for class_name in list(content['classes'].keys())[:3]:
                formatted += f"- `{class_name}`\n"
            formatted += "\n"

        # Add key functions
        if content.get('functions'):
            formatted += "### Key Functions:\n"
            for func_name in list(content['functions'].keys())[:5]:
                formatted += f"- `{func_name}`\n"
            formatted += "\n"

        return formatted

    def _format_function_context(self, item: ContextItem) -> str:
        content = item.content
        func_name = item.identifier.split('::')[-1]

        formatted = f"\n## Function: `{func_name}`\n"
        formatted += f"**Location**: `{item.source_path}`\n"
        formatted += f"**Relevance**: {item.reason}\n"
        formatted += f"**Score**: {item.relevance_score:.1f}\n\n"

        # Add signature
        if content.get('signature'):
            formatted += f"### Signature\n```python\n{content['signature']}\n```\n\n"

        # Add docstring
        if content.get('docstring'):
            formatted += f"### Documentation\n{content['docstring']}\n\n"

        # Add source code preview
        if content.get('source_preview') or content.get('source_code'):
            code = content.get('source_preview') or content.get('source_code', '')[:500]
            if code:
                formatted += f"### Implementation\n```python\n{code}\n```\n\n"

        # Add what it calls
        if content.get('calls'):
            formatted += f"### Calls\n"
            for called in content['calls'][:5]:
                formatted += f"- `{called}`\n"
            formatted += "\n"

        return formatted

    def _format_class_context(self, item: ContextItem) -> str:
        content = item.content
        class_name = item.identifier.split('::')[-1]

        formatted = f"\n## Class: `{class_name}`\n"
        formatted += f"**Location**: `{item.source_path}`\n"
        formatted += f"**Relevance**: {item.reason}\n"
        formatted += f"**Score**: {item.relevance_score:.1f}\n\n"

        # Add signature
        if content.get('signature'):
            formatted += f"### Signature\n```python\n{content['signature']}\n```\n\n"

        # Add docstring
        if content.get('docstring'):
            formatted += f"### Documentation\n{content['docstring']}\n\n"

        # Add source code preview
        if content.get('source_preview') or content.get('source_code'):
            code = content.get('source_preview') or content.get('source_code', '')[:800]
            if code:
                formatted += f"### Implementation\n```python\n{code}\n```\n\n"

        # Add inheritance
        if content.get('references'):
            formatted += f"### Inherits From\n"
            for base in content['references'][:3]:
                formatted += f"- `{base}`\n"
            formatted += "\n"

        return formatted