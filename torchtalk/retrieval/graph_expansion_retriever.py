#!/usr/bin/env python3
"""
Graph-expansion retrieval: Seed-and-expand approach for cross-layer code tracing.

Strategy:
1. Semantic search finds seed results (usually Python API)
2. Expand via graph relationships to find related implementations
3. Fetch actual code chunks for expanded nodes
4. Score by: semantic_similarity + graph_distance
5. Deduplicate and rank

This ensures we trace from Python → C++ → CUDA following actual code paths.
"""

from typing import List, Dict, Any, Set, Optional, Tuple
from dataclasses import dataclass
import networkx as nx


@dataclass
class ExpansionCandidate:
    """A code entity found via graph expansion"""
    name: str
    file_path: str
    language: str
    relationship: str  # 'seed', 'binding', 'call', 'import', 'inheritance'
    distance: int  # Graph distance from seed
    source_seed: str  # Which seed led to this
    cross_language: bool = False


class GraphExpansionRetriever:
    """
    Retrieves code by expanding from semantic search seeds through graph relationships.

    This solves the fundamental problem: semantic embeddings don't understand
    code structure. Python/C++/CUDA use different vocabulary even when implementing
    the same feature. Graph traversal finds the actual related code.
    """

    def __init__(self, vector_retriever, graph_augmenter):
        """
        Args:
            vector_retriever: VectorRetriever instance for semantic search
            graph_augmenter: GraphAugmenter instance for graph traversal
        """
        self.vector_retriever = vector_retriever
        self.graph_augmenter = graph_augmenter

    def retrieve_with_expansion(
        self,
        query: str,
        seed_k: int = 10,
        max_total: int = 30,
        expansion_depth: int = 2,
        semantic_weight: float = 0.4,
        graph_weight: float = 0.6,
        prioritize_cross_language: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Retrieve code using seed-and-expand strategy.

        Args:
            query: User query
            seed_k: Number of seed results from semantic search
            max_total: Maximum total results to return
            expansion_depth: How far to traverse graphs (1=direct, 2=neighbors of neighbors)
            semantic_weight: Weight for semantic similarity score
            graph_weight: Weight for graph distance score
            prioritize_cross_language: Give bonus to cross-language connections

        Returns:
            List of results with code, metadata, and combined scores
        """
        # Step 1: Get seed results via semantic search
        seed_results = self.vector_retriever.retrieve(
            query=query,
            top_k=seed_k,
            filters=None
        )

        if not seed_results:
            return []

        # Step 2: Expand via graphs to find related code
        candidates = self._expand_from_seeds(
            seeds=seed_results,
            depth=expansion_depth,
            prioritize_cross_language=prioritize_cross_language
        )

        # Step 2.5: Heuristic expansion if graph didn't find enough
        # Look for related code in corresponding C++/CUDA directories
        if len(candidates) < 5:
            heuristic_candidates = self._heuristic_expansion(seed_results)
            candidates.extend(heuristic_candidates)

        # Step 3: Fetch actual code chunks for candidates
        expanded_results = self._fetch_candidate_chunks(candidates)

        # Step 4: Combine seeds + expanded results
        all_results = seed_results + expanded_results

        # Step 5: Score and rank
        scored_results = self._score_combined_results(
            results=all_results,
            query=query,
            seed_results=seed_results,
            semantic_weight=semantic_weight,
            graph_weight=graph_weight
        )

        # Step 6: Deduplicate and return top results
        deduped = self._deduplicate(scored_results)

        return deduped[:max_total]

    def _expand_from_seeds(
        self,
        seeds: List[Dict[str, Any]],
        depth: int,
        prioritize_cross_language: bool
    ) -> List[ExpansionCandidate]:
        """
        Expand from seed results through graph relationships.

        Returns candidates (name, file, language) that need to be fetched.
        """
        candidates = []
        seen = set()

        for seed in seeds:
            meta = seed['metadata']
            seed_name = meta.get('name', '')
            seed_file = meta.get('file', '')
            seed_lang = meta.get('language', 'python')
            seed_type = meta.get('type', '')
            full_name = meta.get('full_name', seed_name)

            # Expand via bindings (Python ↔ C++)
            if self.graph_augmenter.bindings:
                binding_candidates = self._expand_via_bindings(
                    name=seed_name,
                    language=seed_lang,
                    seed_name=seed_name,
                    distance=1
                )
                for cand in binding_candidates:
                    key = (cand.name, cand.file_path)
                    if key not in seen:
                        candidates.append(cand)
                        seen.add(key)

            # Expand via call graph (if function/method)
            if seed_type in ['function', 'method'] and full_name:
                call_candidates = self._expand_via_calls(
                    function_name=full_name,
                    seed_name=seed_name,
                    depth=depth
                )
                for cand in call_candidates:
                    key = (cand.name, cand.file_path)
                    if key not in seen:
                        candidates.append(cand)
                        seen.add(key)

            # Expand via inheritance (if class)
            if seed_type in ['class', 'class_definition'] and seed_name:
                inheritance_candidates = self._expand_via_inheritance(
                    class_name=seed_name,
                    seed_name=seed_name,
                    depth=depth
                )
                for cand in inheritance_candidates:
                    key = (cand.name, cand.file_path)
                    if key not in seen:
                        candidates.append(cand)
                        seen.add(key)

            # Expand via imports (to find related modules)
            if seed_file:
                import_candidates = self._expand_via_imports(
                    file_path=seed_file,
                    seed_name=seed_name,
                    depth=min(depth, 1)  # Don't go too deep on imports
                )
                for cand in import_candidates:
                    key = (cand.name, cand.file_path)
                    if key not in seen:
                        candidates.append(cand)
                        seen.add(key)

        # Prioritize cross-language if requested
        if prioritize_cross_language:
            candidates.sort(key=lambda c: (not c.cross_language, c.distance))

        return candidates

    def _expand_via_bindings(
        self,
        name: str,
        language: str,
        seed_name: str,
        distance: int
    ) -> List[ExpansionCandidate]:
        """Expand via pybind11 bindings"""
        candidates = []

        python_to_cpp = self.graph_augmenter.bindings.get('python_to_cpp', {})
        cpp_to_python = self.graph_augmenter.bindings.get('cpp_to_python', {})

        # Python → C++
        if language == 'python' and name in python_to_cpp:
            for cpp_name in python_to_cpp[name]:
                # Find the binding entry to get file path
                for binding in self.graph_augmenter.bindings.get('bindings', []):
                    if binding['python_name'] == name and binding['cpp_name'] == cpp_name:
                        candidates.append(ExpansionCandidate(
                            name=cpp_name,
                            file_path=binding['file'],
                            language='cpp',
                            relationship='binding',
                            distance=distance,
                            source_seed=seed_name,
                            cross_language=True
                        ))

        # C++ → Python
        elif language in ['cpp', 'cuda'] and name in cpp_to_python:
            for py_name in cpp_to_python[name]:
                for binding in self.graph_augmenter.bindings.get('bindings', []):
                    if binding['cpp_name'] == name and binding['python_name'] == py_name:
                        candidates.append(ExpansionCandidate(
                            name=py_name,
                            file_path=binding.get('python_file', ''),
                            language='python',
                            relationship='binding',
                            distance=distance,
                            source_seed=seed_name,
                            cross_language=True
                        ))

        return candidates

    def _expand_via_calls(
        self,
        function_name: str,
        seed_name: str,
        depth: int
    ) -> List[ExpansionCandidate]:
        """Expand via call graph"""
        candidates = []

        if function_name not in self.graph_augmenter.call_graph:
            return candidates

        # Get functions this calls (callees)
        for d in range(1, depth + 1):
            for neighbor in nx.descendants_at_distance(
                self.graph_augmenter.call_graph, function_name, d
            ):
                # Try to determine file from graph node attributes
                node_data = self.graph_augmenter.call_graph.nodes.get(neighbor, {})
                file_path = node_data.get('file', '')
                language = self._infer_language_from_file(file_path)

                candidates.append(ExpansionCandidate(
                    name=neighbor,
                    file_path=file_path,
                    language=language,
                    relationship='calls',
                    distance=d,
                    source_seed=seed_name,
                    cross_language=False
                ))

        # Get functions that call this (callers)
        reverse_graph = self.graph_augmenter.call_graph.reverse()
        for d in range(1, depth + 1):
            for neighbor in nx.descendants_at_distance(reverse_graph, function_name, d):
                node_data = self.graph_augmenter.call_graph.nodes.get(neighbor, {})
                file_path = node_data.get('file', '')
                language = self._infer_language_from_file(file_path)

                candidates.append(ExpansionCandidate(
                    name=neighbor,
                    file_path=file_path,
                    language=language,
                    relationship='called_by',
                    distance=d,
                    source_seed=seed_name,
                    cross_language=False
                ))

        return candidates

    def _expand_via_inheritance(
        self,
        class_name: str,
        seed_name: str,
        depth: int
    ) -> List[ExpansionCandidate]:
        """Expand via inheritance graph"""
        candidates = []

        if class_name not in self.graph_augmenter.inheritance_graph:
            return candidates

        # Parent classes
        for d in range(1, depth + 1):
            for parent in nx.descendants_at_distance(
                self.graph_augmenter.inheritance_graph, class_name, d
            ):
                node_data = self.graph_augmenter.inheritance_graph.nodes.get(parent, {})
                file_path = node_data.get('file', '')
                language = self._infer_language_from_file(file_path)

                candidates.append(ExpansionCandidate(
                    name=parent,
                    file_path=file_path,
                    language=language,
                    relationship='inherits_from',
                    distance=d,
                    source_seed=seed_name,
                    cross_language=False
                ))

        return candidates

    def _expand_via_imports(
        self,
        file_path: str,
        seed_name: str,
        depth: int
    ) -> List[ExpansionCandidate]:
        """Expand via import graph"""
        candidates = []

        if file_path not in self.graph_augmenter.import_graph:
            return candidates

        # Modules this imports
        for d in range(1, depth + 1):
            for imported in nx.descendants_at_distance(
                self.graph_augmenter.import_graph, file_path, d
            ):
                language = self._infer_language_from_file(imported)
                candidates.append(ExpansionCandidate(
                    name=imported.split('/')[-1],  # Use filename as name
                    file_path=imported,
                    language=language,
                    relationship='imports',
                    distance=d,
                    source_seed=seed_name,
                    cross_language=False
                ))

        return candidates

    def _fetch_candidate_chunks(
        self,
        candidates: List[ExpansionCandidate]
    ) -> List[Dict[str, Any]]:
        """
        Fetch actual code chunks for expansion candidates.

        Strategy: Query vector store by name (most reliable identifier).
        """
        results = []

        for cand in candidates:
            if not cand.name:
                continue

            try:
                # Try exact name match first
                chunk_results = self.vector_retriever.vector_store.collection.get(
                    where={'name': cand.name},
                    limit=5  # Get a few matches in case of duplicates
                )

                # If no exact match, try partial match using query
                if not chunk_results or not chunk_results['documents']:
                    # Use semantic search as fallback
                    query_emb = self.vector_retriever.embedder.get_query_embedding(cand.name)
                    chunk_results = self.vector_retriever.vector_store.query(
                        query_embedding=query_emb,
                        n_results=3,
                        where={'language': cand.language} if cand.language != 'unknown' else None
                    )

                if chunk_results and chunk_results['documents']:
                    # Take the best match
                    for idx in range(min(1, len(chunk_results['documents']))):
                        result = {
                            'content': chunk_results['documents'][idx],
                            'metadata': chunk_results['metadatas'][idx] if chunk_results.get('metadatas') else {},
                            'score': 1.0 / (cand.distance + 1),  # Inverse distance as initial score
                            'expansion_info': {
                                'relationship': cand.relationship,
                                'distance': cand.distance,
                                'source_seed': cand.source_seed,
                                'cross_language': cand.cross_language
                            }
                        }
                        results.append(result)

            except Exception as e:
                # Skip candidates we can't fetch
                continue

        return results

    def _score_combined_results(
        self,
        results: List[Dict[str, Any]],
        query: str,
        seed_results: List[Dict[str, Any]],
        semantic_weight: float,
        graph_weight: float
    ) -> List[Dict[str, Any]]:
        """
        Score results combining semantic similarity + graph distance.

        Seed results have high semantic scores.
        Expanded results have high graph scores (low distance).
        """
        seed_ids = {id(r) for r in seed_results}

        for result in results:
            # Semantic score (0-1, higher is better)
            semantic_score = result.get('score', 0.5)

            # Graph score (0-1, higher is better)
            if id(result) in seed_ids:
                graph_score = 1.0  # Seeds have perfect graph score
            else:
                expansion_info = result.get('expansion_info', {})
                distance = expansion_info.get('distance', 10)
                cross_language = expansion_info.get('cross_language', False)

                # Inverse distance: distance 1 → 1.0, distance 2 → 0.5, etc.
                graph_score = 1.0 / distance

                # Bonus for cross-language connections (these are most valuable!)
                if cross_language:
                    graph_score *= 1.5

            # Combined score
            combined_score = (semantic_weight * semantic_score) + (graph_weight * graph_score)
            result['combined_score'] = combined_score
            result['semantic_score'] = semantic_score
            result['graph_score'] = graph_score

        # Sort by combined score
        results.sort(key=lambda r: r.get('combined_score', 0), reverse=True)

        return results

    def _deduplicate(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate chunks based on file + line range"""
        seen = set()
        deduped = []

        for result in results:
            meta = result['metadata']
            key = (
                meta.get('file', ''),
                meta.get('line_start', 0),
                meta.get('line_end', 0)
            )

            if key not in seen:
                deduped.append(result)
                seen.add(key)

        return deduped

    def _heuristic_expansion(self, seed_results: List[Dict[str, Any]]) -> List[ExpansionCandidate]:
        """
        Heuristic-based expansion when graph traversal doesn't find enough.

        Strategy: Extract keywords from seeds and search for related files in
        corresponding C++/CUDA directories.

        For example:
        - Python seed from torch/autograd/ → Search torch/csrc/autograd/
        - Contains "version_counter" → Search for "version" in C++ files
        """
        candidates = []

        # Extract keywords and paths from seeds
        keywords = set()
        python_paths = set()

        for seed in seed_results[:10]:  # Only use top seeds
            meta = seed['metadata']
            file_path = meta.get('file', '')
            lang = meta.get('language', '')
            name = meta.get('name', '')
            content = seed.get('content', '')

            # Collect Python file paths
            if lang == 'python' and file_path:
                python_paths.add(file_path)

            # Extract keywords from names (split on underscores, camelCase)
            if name:
                # Split snake_case
                parts = name.split('_')
                keywords.update(p for p in parts if len(p) > 3)

                # Split CamelCase
                import re
                camel_parts = re.findall(r'[A-Z][a-z]+', name)
                keywords.update(p.lower() for p in camel_parts if len(p) > 3)

        # Remove common words
        keywords = keywords - {'test', 'self', 'init', 'get', 'set', 'this', 'that', 'from', 'with'}

        # Map Python paths to corresponding C++/CUDA paths
        cpp_search_paths = set()
        for py_path in python_paths:
            # torch/autograd/ → torch/csrc/autograd/
            if 'torch/autograd' in py_path or 'torch/_dynamo' in py_path:
                cpp_search_paths.add('torch/csrc/autograd')
                cpp_search_paths.add('aten/src/ATen/autograd')
            # torch/nn/ → torch/csrc/nn/
            elif 'torch/nn' in py_path:
                cpp_search_paths.add('torch/csrc/api')
                cpp_search_paths.add('aten/src/ATen/nn')

        # If no specific paths, use generic autograd
        if not cpp_search_paths:
            cpp_search_paths.add('torch/csrc/autograd')
            cpp_search_paths.add('aten/src/ATen')

        # Search for chunks in C++/CUDA files matching keywords + paths
        for keyword in list(keywords)[:5]:  # Limit to avoid too many queries
            try:
                # Search for this keyword in C++ files
                query_emb = self.vector_retriever.embedder.get_query_embedding(keyword)
                results = self.vector_retriever.vector_store.query(
                    query_embedding=query_emb,
                    n_results=10,
                    where={'language': 'cpp'}
                )

                for idx, doc in enumerate(results.get('documents', [])):
                    if idx >= 3:  # Limit per keyword
                        break

                    meta = results['metadatas'][idx] if results.get('metadatas') else {}
                    file_path = meta.get('file', '')
                    name = meta.get('name', '')

                    # Check if file is in relevant path
                    relevant = any(cpp_path in file_path for cpp_path in cpp_search_paths)
                    if not relevant and cpp_search_paths:
                        continue

                    candidates.append(ExpansionCandidate(
                        name=name,
                        file_path=file_path,
                        language='cpp',
                        relationship='heuristic_keyword',
                        distance=2,
                        source_seed=f"keyword:{keyword}",
                        cross_language=True
                    ))

            except Exception as e:
                continue

        # Also search for CUDA kernels
        for keyword in list(keywords)[:3]:
            try:
                query_emb = self.vector_retriever.embedder.get_query_embedding(f"{keyword} kernel")
                results = self.vector_retriever.vector_store.query(
                    query_embedding=query_emb,
                    n_results=3,
                    where={'language': 'cuda'}
                )

                for idx, doc in enumerate(results.get('documents', [])):
                    meta = results['metadatas'][idx] if results.get('metadatas') else {}
                    candidates.append(ExpansionCandidate(
                        name=meta.get('name', ''),
                        file_path=meta.get('file', ''),
                        language='cuda',
                        relationship='heuristic_kernel',
                        distance=3,
                        source_seed=f"keyword:{keyword}",
                        cross_language=True
                    ))

            except Exception as e:
                continue

        return candidates

    def _infer_language_from_file(self, file_path: str) -> str:
        """Infer language from file extension"""
        if file_path.endswith('.py'):
            return 'python'
        elif file_path.endswith(('.cpp', '.cc', '.cxx', '.h', '.hpp')):
            return 'cpp'
        elif file_path.endswith('.cu'):
            return 'cuda'
        return 'unknown'


