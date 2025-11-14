#!/usr/bin/env python3
"""
Lightweight repository analyzer for building import and call graphs.

This module extracts minimal information needed for graph-enhanced indexing:
- Import relationships between Python files
- Function call relationships within the codebase

The graphs (NetworkX DiGraph) are used only during indexing to populate
LlamaIndex node metadata. They are NOT persisted in the index itself.
"""

import ast
from pathlib import Path
from typing import List, Optional
import networkx as nx


class RepoAnalyzer:
    """
    Builds import and call graphs for a Python repository.

    This is a lightweight analyzer focused on relationships needed for
    graph-enhanced RAG. For the POC, we only extract:
    - import_graph: module → imported modules
    - call_graph: function → called functions

    More sophisticated analysis (AST extraction, centrality, etc.) can be
    added as needed, but is not required for the default indexing flow.
    """

    def __init__(self, repo_path: str):
        self.repo_path = Path(repo_path)
        self.import_graph = nx.DiGraph()
        self.call_graph = nx.DiGraph()

    def analyze_repository(self):
        """
        Analyze repository and build graphs.

        Returns:
            dict: Empty dict for backward compatibility. The graphs themselves
                  (self.import_graph, self.call_graph) are what get used.
        """
        print("Repository analysis starting...")

        # Find Python files
        python_files = self._find_python_files()
        print(f"Found {len(python_files)} Python files")

        # Build graphs
        for file_path in python_files:
            self._analyze_file(file_path)

        print(f"Import graph: {self.import_graph.number_of_nodes()} nodes, "
              f"{self.import_graph.number_of_edges()} edges")
        print(f"Call graph: {self.call_graph.number_of_nodes()} nodes, "
              f"{self.call_graph.number_of_edges()} edges")

        # Return empty dict for backward compatibility
        # (graph_enhanced_indexer.py stores this but never uses it)
        return {}

    def _find_python_files(self) -> List[Path]:
        """Find Python files, excluding common test/build directories."""
        all_files = list(self.repo_path.rglob("*.py"))

        filtered_files = []
        for file_path in all_files:
            path_parts = file_path.parts
            skip = False

            for part in path_parts:
                part_lower = part.lower()
                if any(pattern in part_lower for pattern in
                       ['test', 'example', '__pycache__', 'build', 'dist', '.git', 'benchmark']):
                    skip = True
                    break

            if not skip:
                filtered_files.append(file_path)

        # Limit for performance during large repo analysis
        return filtered_files[:500]

    def _analyze_file(self, file_path: Path):
        """Analyze a single Python file for imports and calls."""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            tree = ast.parse(content)
            relative_path = str(file_path.relative_to(self.repo_path))

            # Add file to import graph
            self.import_graph.add_node(relative_path)

            # Extract imports
            imports = self._extract_imports(tree)
            for imported in imports:
                resolved = self._resolve_import(imported, relative_path)
                if resolved:
                    self.import_graph.add_edge(relative_path, resolved)

            # Extract function calls (simple extraction)
            self._extract_calls(tree, relative_path)

        except Exception as e:
            # Silently skip unparseable files
            pass

    def _extract_imports(self, tree: ast.AST) -> List[str]:
        """Extract import statements from AST."""
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend([alias.name for alias in node.names])
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.append(node.module)
        return imports

    def _resolve_import(self, import_name: str, from_module: str) -> Optional[str]:
        """
        Try to resolve an import to a file path in the repo.

        This is a best-effort resolution for relative imports and local modules.
        External imports (torch, numpy, etc.) won't resolve and that's fine.
        """
        # Convert import to potential file paths
        potential_paths = [
            import_name.replace('.', '/') + '.py',
            import_name.replace('.', '/') + '/__init__.py',
        ]

        for path in potential_paths:
            full_path = self.repo_path / path
            if full_path.exists():
                return path

        # Check relative imports
        if from_module:
            parent_dir = Path(from_module).parent
            for path in potential_paths:
                full_path = self.repo_path / parent_dir / path
                if full_path.exists():
                    return str((parent_dir / path).as_posix())

        return None

    def _extract_calls(self, tree: ast.AST, module_path: str):
        """
        Extract function definitions and their calls.

        Builds call graph with nodes like "module.function" -> "called_function".
        This is a simplified extraction focused on top-level functions.
        """
        module_name = Path(module_path).stem

        # Find all function definitions
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                func_name = f"{module_name}.{node.name}"
                self.call_graph.add_node(func_name, module=module_path)

                # Extract calls within this function
                calls = self._extract_function_calls(node)
                for called in calls:
                    called_name = f"{module_name}.{called}"
                    self.call_graph.add_edge(func_name, called_name)

    def _extract_function_calls(self, node: ast.AST) -> List[str]:
        """Extract function call names from an AST node."""
        calls = []
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                if isinstance(child.func, ast.Name):
                    calls.append(child.func.id)
                elif isinstance(child.func, ast.Attribute):
                    calls.append(child.func.attr)
        return calls
