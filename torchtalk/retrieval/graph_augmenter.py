#!/usr/bin/env python3
"""
Graph-based retrieval augmentation using code relationships.

Expands vector search results by traversing:
- Import graph (what modules are imported/import this)
- Call graph (what functions are called/call this)
- Inheritance graph (parent/child classes)
- Cross-language bindings (Python ↔ C++/CUDA via pybind11) [Phase 6]
"""

from typing import List, Dict, Any, Set, Optional
from pathlib import Path
import pickle
import json
import networkx as nx


class GraphAugmenter:
    """
    Augment retrieval results using code relationship graphs.

    Strategy:
    1. Start with vector search results (seed set)
    2. Expand by traversing graphs to find related code:
       - Functions called by seed functions
       - Functions that call seed functions
       - Classes that inherit from seed classes
       - Modules imported by seed modules
    3. Return expanded context with relationship information
    """

    def __init__(self, index_dir: str):
        """
        Initialize the graph augmenter.

        Args:
            index_dir: Directory containing the index with saved graphs
        """
        self.index_dir = Path(index_dir)

        # Load graphs
        self.import_graph = self._load_graph('import_graph.gpickle')
        self.call_graph = self._load_graph('call_graph.gpickle')
        self.inheritance_graph = self._load_graph('inheritance_graph.gpickle')

        # Load cross-language bindings (Phase 6)
        self.bindings = self._load_bindings('bindings.json')

        print(f" GraphAugmenter initialized")
        print(f"  Import graph: {self.import_graph.number_of_nodes()} nodes, "
              f"{self.import_graph.number_of_edges()} edges")
        print(f"  Call graph: {self.call_graph.number_of_nodes()} nodes, "
              f"{self.call_graph.number_of_edges()} edges")
        print(f"  Inheritance graph: {self.inheritance_graph.number_of_nodes()} nodes, "
              f"{self.inheritance_graph.number_of_edges()} edges")
        if self.bindings:
            print(f"  Cross-language bindings: {len(self.bindings.get('bindings', []))} total")

    def _load_graph(self, filename: str) -> nx.DiGraph:
        """Load a NetworkX graph from pickle"""
        graph_path = self.index_dir / filename

        if not graph_path.exists():
            print(f" Graph file not found: {filename}, creating empty graph")
            return nx.DiGraph()

        with open(graph_path, 'rb') as f:
            return pickle.load(f)

    def _load_bindings(self, filename: str) -> Optional[Dict]:
        """Load cross-language bindings from JSON (Phase 6)"""
        bindings_path = self.index_dir / filename

        if not bindings_path.exists():
            return None

        with open(bindings_path, 'r') as f:
            return json.load(f)

    def augment(
        self,
        results: List[Dict[str, Any]],
        max_expansions: int = 5,
        traverse_imports: bool = True,
        traverse_calls: bool = True,
        traverse_inheritance: bool = True,
        traverse_bindings: bool = True,  # Phase 6: cross-language
        depth: int = 1
    ) -> List[Dict[str, Any]]:
        """
        Augment retrieval results with related code from graphs.

        Args:
            results: List of retrieval results (from VectorRetriever)
            max_expansions: Maximum number of related items to add per result
            traverse_imports: Include imported/importing modules
            traverse_calls: Include called/calling functions
            traverse_inheritance: Include parent/child classes
            traverse_bindings: Include cross-language bindings (Python↔C++/CUDA) [Phase 6]
            depth: Graph traversal depth (1 = immediate neighbors, 2 = neighbors of neighbors)

        Returns:
            Augmented results with additional 'related_code' field containing
            related entities and their relationships
        """
        augmented_results = []

        for result in results:
            # Extract identifiers from metadata
            file_path = result['metadata'].get('file', '')
            code_type = result['metadata'].get('type', '')
            name = result['metadata'].get('name', '')
            full_name = result['metadata'].get('full_name', name)
            language = result['metadata'].get('language', 'python')

            # Find related code
            related = []

            if traverse_imports and file_path:
                related.extend(self._find_import_related(file_path, depth, max_expansions))

            if traverse_calls and name:
                if code_type in ['function', 'method']:
                    related.extend(self._find_call_related(full_name, depth, max_expansions))

            if traverse_inheritance and name:
                if code_type in ['class', 'class_definition']:
                    related.extend(self._find_inheritance_related(name, depth))

            # Phase 6: Cross-language traversal via pybind11 bindings
            if traverse_bindings and self.bindings and name:
                related.extend(self._find_binding_related(name, language))

            # Add related code to result
            augmented_result = {
                **result,
                'related_code': related[:max_expansions]  # Limit total expansions
            }
            augmented_results.append(augmented_result)

        return augmented_results

    def _find_import_related(
        self,
        file_path: str,
        depth: int,
        max_results: int
    ) -> List[Dict[str, Any]]:
        """Find modules related through imports"""
        related = []

        if file_path not in self.import_graph:
            return related

        # Modules this file imports (outgoing edges)
        # Collect from all depths up to max depth
        for d in range(1, depth + 1):
            for neighbor in nx.descendants_at_distance(self.import_graph, file_path, d):
                if len(related) >= max_results:
                    break
                related.append({
                    'type': 'import',
                    'relationship': 'imports',
                    'target': neighbor,
                    'depth': d,
                    'description': f"{file_path} imports {neighbor} (depth {d})"
                })

        # Modules that import this file (incoming edges)
        if len(related) < max_results:
            reverse_graph = self.import_graph.reverse()
            for d in range(1, depth + 1):
                for neighbor in nx.descendants_at_distance(reverse_graph, file_path, d):
                    if len(related) >= max_results:
                        break
                    related.append({
                        'type': 'import',
                        'relationship': 'imported_by',
                        'target': neighbor,
                        'depth': d,
                        'description': f"{file_path} is imported by {neighbor} (depth {d})"
                    })

        return related

    def _find_call_related(
        self,
        function_name: str,
        depth: int,
        max_results: int
    ) -> List[Dict[str, Any]]:
        """Find functions related through calls"""
        related = []

        if function_name not in self.call_graph:
            return related

        # Functions this function calls (outgoing edges - callees)
        # Collect from all depths up to max depth for deep call tracing
        for d in range(1, depth + 1):
            for neighbor in nx.descendants_at_distance(self.call_graph, function_name, d):
                if len(related) >= max_results:
                    break
                related.append({
                    'type': 'call',
                    'relationship': 'calls',
                    'target': neighbor,
                    'depth': d,
                    'description': f"{function_name} calls {neighbor} (depth {d})"
                })

        # Functions that call this function (incoming edges - callers)
        if len(related) < max_results:
            reverse_graph = self.call_graph.reverse()
            for d in range(1, depth + 1):
                for neighbor in nx.descendants_at_distance(reverse_graph, function_name, d):
                    if len(related) >= max_results:
                        break
                    related.append({
                        'type': 'call',
                        'relationship': 'called_by',
                        'target': neighbor,
                        'depth': d,
                        'description': f"{function_name} is called by {neighbor} (depth {d})"
                    })

        return related

    def _find_inheritance_related(
        self,
        class_name: str,
        depth: int
    ) -> List[Dict[str, Any]]:
        """Find classes related through inheritance"""
        related = []

        if class_name not in self.inheritance_graph:
            return related

        # Parent classes (outgoing edges - what this class inherits from)
        for parent in nx.descendants_at_distance(self.inheritance_graph, class_name, depth):
            related.append({
                'type': 'inheritance',
                'relationship': 'inherits_from',
                'target': parent,
                'description': f"{class_name} inherits from {parent}"
            })

        # Child classes (incoming edges - what inherits from this class)
        reverse_graph = self.inheritance_graph.reverse()
        for child in nx.descendants_at_distance(reverse_graph, class_name, depth):
            related.append({
                'type': 'inheritance',
                'relationship': 'inherited_by',
                'target': child,
                'description': f"{class_name} is inherited by {child}"
            })

        return related

    def _find_binding_related(
        self,
        name: str,
        language: str
    ) -> List[Dict[str, Any]]:
        """
        Find cross-language bindings (Phase 6: Python ↔ C++/CUDA).

        Args:
            name: Function/class name to find bindings for
            language: Source language ('python', 'cpp', 'cuda')

        Returns:
            List of related entities across language boundaries
        """
        related = []

        if not self.bindings:
            return related

        python_to_cpp = self.bindings.get('python_to_cpp', {})
        cpp_to_python = self.bindings.get('cpp_to_python', {})

        # Python → C++/CUDA
        if language == 'python' and name in python_to_cpp:
            for cpp_name in python_to_cpp[name]:
                related.append({
                    'type': 'binding',
                    'relationship': 'binds_to_cpp',
                    'target': cpp_name,
                    'description': f"Python '{name}' binds to C++ '{cpp_name}' via pybind11",
                    'cross_language': True,
                    'source_lang': 'python',
                    'target_lang': 'cpp'
                })

        # C++/CUDA → Python
        elif language in ['cpp', 'cuda'] and name in cpp_to_python:
            for py_name in cpp_to_python[name]:
                related.append({
                    'type': 'binding',
                    'relationship': 'bound_from_python',
                    'target': py_name,
                    'description': f"C++ '{name}' is bound to Python '{py_name}' via pybind11",
                    'cross_language': True,
                    'source_lang': 'cpp',
                    'target_lang': 'python'
                })

        return related

    def find_path_between(
        self,
        source: str,
        target: str,
        graph_type: str = 'call'
    ) -> Optional[List[str]]:
        """
        Find a path between two entities in a graph.

        Args:
            source: Source entity name
            target: Target entity name
            graph_type: Which graph to search ('import', 'call', 'inheritance')

        Returns:
            List of entity names forming the path, or None if no path exists
        """
        graph = {
            'import': self.import_graph,
            'call': self.call_graph,
            'inheritance': self.inheritance_graph
        }.get(graph_type)

        if not graph or source not in graph or target not in graph:
            return None

        try:
            path = nx.shortest_path(graph, source, target)
            return path
        except nx.NetworkXNoPath:
            return None

    def get_related_files(
        self,
        file_path: str,
        max_depth: int = 2
    ) -> Set[str]:
        """
        Get all files related to a given file through imports.

        Args:
            file_path: Path to the file
            max_depth: Maximum import depth to traverse

        Returns:
            Set of related file paths
        """
        if file_path not in self.import_graph:
            return set()

        related_files = set()

        # BFS traversal up to max_depth
        for depth in range(1, max_depth + 1):
            # Imported files
            for node in nx.descendants_at_distance(self.import_graph, file_path, depth):
                related_files.add(node)

            # Importing files
            reverse_graph = self.import_graph.reverse()
            for node in nx.descendants_at_distance(reverse_graph, file_path, depth):
                related_files.add(node)

        return related_files

    def get_call_chain(
        self,
        function_name: str,
        direction: str = 'callees',
        max_depth: int = 2
    ) -> Dict[str, List[str]]:
        """
        Get the call chain for a function.

        Args:
            function_name: Name of the function
            direction: 'callees' (functions this calls) or 'callers' (functions that call this)
            max_depth: Maximum depth to traverse

        Returns:
            Dictionary mapping depth level to list of functions at that level
        """
        if function_name not in self.call_graph:
            return {}

        graph = self.call_graph if direction == 'callees' else self.call_graph.reverse()
        call_chain = {}

        for depth in range(1, max_depth + 1):
            functions = list(nx.descendants_at_distance(graph, function_name, depth))
            if functions:
                call_chain[f"depth_{depth}"] = functions

        return call_chain


