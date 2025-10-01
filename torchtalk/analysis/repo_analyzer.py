#!/usr/bin/env python3
"""
Builds import graphs, call graphs, and extracts actual code content.
"""

import ast
import json
import os
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple, Any
from collections import defaultdict, Counter
import re
from dataclasses import dataclass, asdict, field
import networkx as nx


@dataclass
class CodeEntity:
    name: str
    type: str  # 'class', 'function', 'method'
    signature: str
    docstring: Optional[str]
    source_code: str
    line_start: int
    line_end: int
    calls: List[str] = field(default_factory=list)  # Functions this entity calls
    references: List[str] = field(default_factory=list)  # Other entities referenced


@dataclass
class EnhancedModuleInfo:
    name: str
    path: str
    relative_path: str
    directory: str
    docstring: Optional[str]

    # Metadata
    size_lines: int
    depth: int

    # Code entities with actual source
    classes: Dict[str, CodeEntity]
    functions: Dict[str, CodeEntity]

    # Relationships
    imports: List[str]  # What this module imports
    imported_by: List[str] = field(default_factory=list)  # Modules that import this
    internal_calls: Dict[str, List[str]] = field(default_factory=dict)  # Call graph within module
    external_calls: Dict[str, List[str]] = field(default_factory=dict)  # Calls to other modules

    # Semantic information
    keywords: Set[str] = field(default_factory=set)
    symbols_defined: List[str] = field(default_factory=list)
    symbols_used: List[str] = field(default_factory=list)


class RepoAnalyzer:
    def __init__(self, repo_path: str):
        self.repo_path = Path(repo_path)
        self.modules: Dict[str, EnhancedModuleInfo] = {}
        self.import_graph = nx.DiGraph()
        self.call_graph = nx.DiGraph()
        self.inheritance_graph = nx.DiGraph()

    def analyze_repository(self) -> Dict[str, Any]:
        print("Enhanced analysis starting...")

        # Find and analyze all Python files
        python_files = self._find_python_files()
        print(f"Found {len(python_files)} Python files")

        # First pass: analyze each file individually
        for file_path in python_files:
            module_info = self._analyze_file_with_code(file_path)
            if module_info:
                self.modules[str(file_path)] = module_info

        # Second pass: build relationship graphs
        self._build_import_graph()
        self._build_call_graph()
        self._build_inheritance_graph()

        # Analyze graph metrics
        graph_metrics = self._analyze_graph_metrics()

        # Build final results
        return self._build_analysis_results(graph_metrics)

    def _find_python_files(self) -> List[Path]:
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

        return filtered_files[:500]  # Limit for performance during development

    def _analyze_file_with_code(self, file_path: Path) -> Optional[EnhancedModuleInfo]:
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            tree = ast.parse(content)
            lines = content.splitlines()

            relative_path = file_path.relative_to(self.repo_path)
            directory = str(relative_path.parent) if relative_path.parent != Path('.') else 'root'

            # Extract code entities with source
            classes = self._extract_classes_with_code(tree, lines, content)
            functions = self._extract_functions_with_code(tree, lines, content)

            # Extract imports and build module name mapping
            imports = self._extract_imports(tree)

            # Extract symbols and keywords
            symbols_defined = list(classes.keys()) + list(functions.keys())
            symbols_used = self._extract_used_symbols(tree)
            keywords = self._extract_semantic_keywords(tree, classes, functions)

            return EnhancedModuleInfo(
                name=file_path.stem,
                path=str(file_path),
                relative_path=str(relative_path),
                directory=directory,
                docstring=ast.get_docstring(tree),
                classes=classes,
                functions=functions,
                imports=imports,
                size_lines=len(lines),
                depth=len(relative_path.parts) - 1,
                keywords=keywords,
                symbols_defined=symbols_defined,
                symbols_used=symbols_used
            )

        except Exception as e:
            print(f"Error analyzing {file_path}: {e}")
            return None

    def _extract_classes_with_code(self, tree: ast.AST, lines: List[str], content: str) -> Dict[str, CodeEntity]:
        classes = {}

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                try:
                    source = ast.get_source_segment(content, node)
                    if not source:
                        # Fallback to line extraction
                        source = '\n'.join(lines[node.lineno - 1:node.end_lineno]) if hasattr(node, 'end_lineno') else ""

                    # Extract methods and their calls
                    methods = []
                    calls = []
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef):
                            methods.append(item.name)
                            calls.extend(self._extract_function_calls(item))

                    # Build signature
                    bases = [self._get_name(base) for base in node.bases]
                    signature = f"class {node.name}" + (f"({', '.join(bases)})" if bases else "")

                    classes[node.name] = CodeEntity(
                        name=node.name,
                        type='class',
                        signature=signature,
                        docstring=ast.get_docstring(node),
                        source_code=source[:2000] if source else "",  # Limit size
                        line_start=node.lineno,
                        line_end=getattr(node, 'end_lineno', node.lineno),
                        calls=calls,
                        references=bases
                    )
                except Exception as e:
                    print(f"Error extracting class {node.name}: {e}")

        return classes

    def _extract_functions_with_code(self, tree: ast.AST, lines: List[str], content: str) -> Dict[str, CodeEntity]:
        functions = {}

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and not self._is_method(node):
                try:
                    source = ast.get_source_segment(content, node)
                    if not source:
                        # Fallback to line extraction
                        source = '\n'.join(lines[node.lineno - 1:node.end_lineno]) if hasattr(node, 'end_lineno') else ""

                    # Extract function calls
                    calls = self._extract_function_calls(node)

                    # Build signature
                    args = []
                    for arg in node.args.args:
                        args.append(arg.arg)
                    signature = f"def {node.name}({', '.join(args)})"

                    functions[node.name] = CodeEntity(
                        name=node.name,
                        type='function',
                        signature=signature,
                        docstring=ast.get_docstring(node),
                        source_code=source[:1500] if source else "",  # Limit size
                        line_start=node.lineno,
                        line_end=getattr(node, 'end_lineno', node.lineno),
                        calls=calls,
                        references=[]
                    )
                except Exception as e:
                    print(f"Error extracting function {node.name}: {e}")

        return functions

    def _is_method(self, node: ast.FunctionDef) -> bool:
        for parent in ast.walk(ast.Module()):
            if isinstance(parent, ast.ClassDef):
                for item in parent.body:
                    if item is node:
                        return True
        return False

    def _extract_function_calls(self, node: ast.AST) -> List[str]:
        calls = []
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                if isinstance(child.func, ast.Name):
                    calls.append(child.func.id)
                elif isinstance(child.func, ast.Attribute):
                    calls.append(child.func.attr)
        return calls

    def _extract_imports(self, tree: ast.AST) -> List[str]:
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend([alias.name for alias in node.names])
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.append(node.module)
        return imports

    def _extract_used_symbols(self, tree: ast.AST) -> List[str]:
        symbols = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                symbols.append(node.id)
            elif isinstance(node, ast.Attribute):
                symbols.append(node.attr)
        return list(set(symbols))

    def _extract_semantic_keywords(self, tree: ast.AST, classes: Dict, functions: Dict) -> Set[str]:
        keywords = set()

        # From class and function names
        for name in list(classes.keys()) + list(functions.keys()):
            keywords.update(self._split_camel_case(name))

        # From docstrings
        for entity in list(classes.values()) + list(functions.values()):
            if entity.docstring:
                words = re.findall(r'\b[a-zA-Z]{3,}\b', entity.docstring.lower())
                keywords.update([w for w in words if self._is_technical_word(w)])

        return keywords

    def _split_camel_case(self, name: str) -> List[str]:
        words = re.sub('([a-z0-9])([A-Z])', r'\1 \2', name).split()
        return [w.lower() for w in words if len(w) > 2]

    def _is_technical_word(self, word: str) -> bool:
        common_words = {
            'the', 'and', 'for', 'are', 'with', 'this', 'that', 'from', 'can', 'will',
            'not', 'you', 'all', 'has', 'had', 'but', 'what', 'use', 'get', 'set', 'new'
        }
        return word not in common_words and len(word) >= 4

    def _get_name(self, node: ast.AST) -> str:
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self._get_name(node.value)}.{node.attr}"
        return "unknown"

    def _build_import_graph(self):
        print("Building import graph...")

        # Add all modules as nodes
        for module_path, module_info in self.modules.items():
            self.import_graph.add_node(module_info.relative_path,
                                     name=module_info.name,
                                     directory=module_info.directory)

        # Add import edges
        for module_path, module_info in self.modules.items():
            for imported in module_info.imports:
                # Try to resolve the import to a module in our repo
                resolved = self._resolve_import(imported, module_info.relative_path)
                if resolved:
                    self.import_graph.add_edge(module_info.relative_path, resolved)
                    # Update imported_by relationship
                    if resolved in self.modules:
                        self.modules[resolved].imported_by.append(module_info.relative_path)

    def _resolve_import(self, import_name: str, from_module: str) -> Optional[str]:
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

    def _build_call_graph(self):
        print("Building call graph...")

        for module_path, module_info in self.modules.items():
            module_prefix = module_info.name

            # Add nodes for all functions and classes
            for func_name, func_entity in module_info.functions.items():
                node_name = f"{module_prefix}.{func_name}"
                self.call_graph.add_node(node_name, type='function', module=module_info.relative_path)

                # Add edges for function calls
                for called in func_entity.calls:
                    called_name = f"{module_prefix}.{called}"
                    self.call_graph.add_edge(node_name, called_name)

            for class_name, class_entity in module_info.classes.items():
                node_name = f"{module_prefix}.{class_name}"
                self.call_graph.add_node(node_name, type='class', module=module_info.relative_path)

                # Add edges for method calls
                for called in class_entity.calls:
                    called_name = f"{module_prefix}.{called}"
                    self.call_graph.add_edge(node_name, called_name)

    def _build_inheritance_graph(self):
        print("Building inheritance graph...")

        for module_path, module_info in self.modules.items():
            for class_name, class_entity in module_info.classes.items():
                full_name = f"{module_info.name}.{class_name}"
                self.inheritance_graph.add_node(full_name, module=module_info.relative_path)

                # Add inheritance edges
                for base in class_entity.references:
                    if '.' in base:
                        base_full = base
                    else:
                        base_full = f"{module_info.name}.{base}"
                    self.inheritance_graph.add_edge(full_name, base_full)

    def _analyze_graph_metrics(self) -> Dict[str, Any]:
        metrics = {}

        # Import graph metrics
        if self.import_graph.number_of_nodes() > 0:
            metrics['import_graph'] = {
                'nodes': self.import_graph.number_of_nodes(),
                'edges': self.import_graph.number_of_edges(),
                'most_imported': self._get_most_connected(self.import_graph, 'in'),
                'most_importing': self._get_most_connected(self.import_graph, 'out'),
                'clusters': self._detect_clusters(self.import_graph)
            }

        # Call graph metrics
        if self.call_graph.number_of_nodes() > 0:
            metrics['call_graph'] = {
                'nodes': self.call_graph.number_of_nodes(),
                'edges': self.call_graph.number_of_edges(),
                'most_called': self._get_most_connected(self.call_graph, 'in'),
                'most_calling': self._get_most_connected(self.call_graph, 'out')
            }

        # Inheritance metrics
        if self.inheritance_graph.number_of_nodes() > 0:
            metrics['inheritance_graph'] = {
                'nodes': self.inheritance_graph.number_of_nodes(),
                'edges': self.inheritance_graph.number_of_edges(),
                'base_classes': self._get_root_nodes(self.inheritance_graph),
                'deepest_inheritance': self._get_deepest_path(self.inheritance_graph)
            }

        return metrics

    def _get_most_connected(self, graph: nx.DiGraph, direction: str, top_n: int = 10) -> List[Tuple[str, int]]:
        if direction == 'in':
            degrees = graph.in_degree()
        else:
            degrees = graph.out_degree()

        sorted_nodes = sorted(degrees, key=lambda x: x[1], reverse=True)
        return sorted_nodes[:top_n]

    def _detect_clusters(self, graph: nx.DiGraph) -> List[Set[str]]:
        undirected = graph.to_undirected()
        clusters = list(nx.connected_components(undirected))
        # Return only significant clusters
        return [cluster for cluster in clusters if len(cluster) > 2][:10]

    def _get_root_nodes(self, graph: nx.DiGraph) -> List[str]:
        roots = [node for node in graph.nodes() if graph.in_degree(node) == 0]
        return roots[:10]

    def _get_deepest_path(self, graph: nx.DiGraph) -> int:
        if graph.number_of_nodes() == 0:
            return 0

        max_depth = 0
        for root in self._get_root_nodes(graph):
            try:
                paths = nx.single_source_shortest_path_length(graph, root)
                depth = max(paths.values()) if paths else 0
                max_depth = max(max_depth, depth)
            except:
                continue

        return max_depth

    def _build_analysis_results(self, graph_metrics: Dict[str, Any]) -> Dict[str, Any]:
        # Group modules by category
        categories = defaultdict(list)
        for module_path, module_info in self.modules.items():
            categories[module_info.directory].append(module_info)

        # Build category details with code samples
        category_details = {}
        for category, modules in categories.items():
            if len(modules) >= 2:  # Only significant categories
                top_modules = sorted(modules, key=lambda m: m.size_lines, reverse=True)[:5]

                category_details[category] = {
                    'module_count': len(modules),
                    'total_lines': sum(m.size_lines for m in modules),
                    'top_modules': [
                        {
                            'name': m.name,
                            'path': m.relative_path,
                            'classes': {name: {
                                'signature': entity.signature,
                                'docstring': entity.docstring,
                                'source_preview': entity.source_code[:300] if entity.source_code else None
                            } for name, entity in list(m.classes.items())[:3]},
                            'functions': {name: {
                                'signature': entity.signature,
                                'docstring': entity.docstring,
                                'source_preview': entity.source_code[:200] if entity.source_code else None
                            } for name, entity in list(m.functions.items())[:5]},
                            'imports': m.imports[:10],
                            'keywords': list(m.keywords)[:15]
                        }
                        for m in top_modules
                    ]
                }

        # Find key modules based on graph centrality
        key_modules = []
        if self.import_graph.number_of_nodes() > 0:
            centrality = nx.betweenness_centrality(self.import_graph)
            top_central = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:10]
            for module_path, score in top_central:
                for full_path, module_info in self.modules.items():
                    if module_info.relative_path == module_path:
                        key_modules.append({
                            'path': module_path,
                            'name': module_info.name,
                            'centrality_score': score,
                            'imports': len(module_info.imports),
                            'imported_by': len(module_info.imported_by)
                        })
                        break

        return {
            'repo_path': str(self.repo_path),
            'total_modules': len(self.modules),
            'category_details': category_details,
            'graph_metrics': graph_metrics,
            'key_modules': key_modules,
            'modules_with_code': {
                path: {
                    'name': info.name,
                    'path': info.relative_path,
                    'classes': {k: asdict(v) for k, v in info.classes.items()},
                    'functions': {k: asdict(v) for k, v in info.functions.items()},
                    'imports': info.imports,
                    'imported_by': info.imported_by,
                    'keywords': list(info.keywords)
                }
                for path, info in list(self.modules.items())[:100]  # Limit for file size
            }
        }

    def save_analysis(self, output_path: str = None) -> str:
        if not output_path:
            repo_name = self.repo_path.name
            output_path = f"artifacts/{repo_name}_enhanced_analysis.json"

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        results = self.analyze_repository()

        # Convert sets to lists for JSON serialization
        def convert_sets(obj):
            if isinstance(obj, set):
                return list(obj)
            elif isinstance(obj, dict):
                return {k: convert_sets(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_sets(v) for v in obj]
            return obj

        results = convert_sets(results)

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"Enhanced analysis saved to: {output_path}")
        print(f"Import graph: {self.import_graph.number_of_nodes()} nodes, {self.import_graph.number_of_edges()} edges")
        print(f"Call graph: {self.call_graph.number_of_nodes()} nodes, {self.call_graph.number_of_edges()} edges")

        return output_path


