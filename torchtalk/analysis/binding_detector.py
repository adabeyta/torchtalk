#!/usr/bin/env python3
"""
Binding detector for pybind11 C++/Python connections.

Parses pybind11 binding code to build a mapping between:
- Python function names → C++ function implementations
- Python class names → C++ class implementations
- Python methods → C++ method implementations

This enables cross-language code tracing for PyTorch-style codebases.
"""

from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import re
import tree_sitter


@dataclass
class Binding:
    """Represents a Python↔C++ binding"""
    python_name: str          # Name visible in Python
    cpp_name: str             # C++ function/class name
    binding_type: str         # 'function', 'class', 'method', 'module'
    file_path: str            # Path to binding file
    line_number: int          # Line where binding is defined
    module_name: Optional[str] = None  # PYBIND11_MODULE name
    parent_class: Optional[str] = None # For methods: parent class
    signature: Optional[str] = None    # Full binding signature


@dataclass
class BindingGraph:
    """Cross-language binding relationships"""
    bindings: List[Binding] = field(default_factory=list)
    python_to_cpp: Dict[str, List[str]] = field(default_factory=dict)  # Python name → C++ names
    cpp_to_python: Dict[str, List[str]] = field(default_factory=dict)  # C++ name → Python names
    modules: Dict[str, List[Binding]] = field(default_factory=dict)    # Module → bindings

    def add_binding(self, binding: Binding):
        """Add a binding and update indices"""
        self.bindings.append(binding)

        # Index by Python name
        if binding.python_name not in self.python_to_cpp:
            self.python_to_cpp[binding.python_name] = []
        self.python_to_cpp[binding.python_name].append(binding.cpp_name)

        # Index by C++ name
        if binding.cpp_name not in self.cpp_to_python:
            self.cpp_to_python[binding.cpp_name] = []
        self.cpp_to_python[binding.cpp_name].append(binding.python_name)

        # Index by module
        if binding.module_name:
            if binding.module_name not in self.modules:
                self.modules[binding.module_name] = []
            self.modules[binding.module_name].append(binding)

    def get_cpp_for_python(self, python_name: str) -> List[str]:
        """Get C++ implementations for a Python name"""
        return self.python_to_cpp.get(python_name, [])

    def get_python_for_cpp(self, cpp_name: str) -> List[str]:
        """Get Python names for a C++ implementation"""
        return self.cpp_to_python.get(cpp_name, [])


class BindingDetector:
    """
    Detect pybind11 bindings in C++ files.

    Supported patterns:
    - PYBIND11_MODULE(module_name, m) { ... }
    - m.def("func_name", &cpp_function)
    - py::class_<CppClass>(m, "PyClass")
    - .def("method", &CppClass::method)
    - .def_static("static_method", &CppClass::static_method)
    - .def_readwrite("attr", &CppClass::attr)
    """

    def __init__(self):
        """Initialize C++ parser for binding detection"""
        from tree_sitter_language_pack import get_parser
        self.parser = get_parser('cpp')
        print(" BindingDetector initialized")

    def detect_bindings(self, file_path: str, content: str) -> BindingGraph:
        """
        Parse a C++ file and extract all pybind11 bindings.

        Args:
            file_path: Path to the C++ file
            content: File content as string

        Returns:
            BindingGraph with all detected bindings
        """
        tree = self.parser.parse(bytes(content, 'utf8'))
        root_node = tree.root_node

        graph = BindingGraph()

        # Find PYBIND11_MODULE declarations
        modules = self._find_pybind11_modules(root_node, content)

        for module_name, module_node in modules:
            # Extract bindings within this module
            self._extract_module_bindings(
                module_node, content, file_path, module_name, graph
            )

        return graph

    def _find_pybind11_modules(self, node, content: str) -> List[Tuple[str, Any]]:
        """Find all PYBIND11_MODULE declarations"""
        modules = []

        # Look for function_definition nodes that might be PYBIND11_MODULE
        if node.type == 'function_definition':
            text = self._get_node_text(node, content)

            # Check if it's a PYBIND11_MODULE
            match = re.search(r'PYBIND11_MODULE\s*\(\s*(\w+)\s*,', text)
            if match:
                module_name = match.group(1)
                modules.append((module_name, node))

        # Recurse into children
        for child in node.children:
            modules.extend(self._find_pybind11_modules(child, content))

        return modules

    def _extract_module_bindings(
        self,
        module_node,
        content: str,
        file_path: str,
        module_name: str,
        graph: BindingGraph
    ):
        """Extract all bindings from a PYBIND11_MODULE body"""

        # Get the function body
        body = None
        for child in module_node.children:
            if child.type == 'compound_statement':
                body = child
                break

        if not body:
            return

        # Parse the body text for binding patterns
        body_text = self._get_node_text(body, content)
        body_start_line = body.start_point[0] + 1

        # Extract function bindings: m.def("name", &function)
        self._extract_function_bindings(
            body_text, body_start_line, file_path, module_name, graph
        )

        # Extract class bindings: py::class_<CppClass>(m, "PyClass")
        self._extract_class_bindings(
            body_text, body_start_line, file_path, module_name, graph
        )

    def _extract_function_bindings(
        self,
        text: str,
        start_line: int,
        file_path: str,
        module_name: str,
        graph: BindingGraph
    ):
        """Extract m.def(...) function bindings"""

        # Pattern: m.def("python_name", &cpp_function)
        # or: m.def("python_name", &Namespace::function)
        pattern = r'm\.def\s*\(\s*"([^"]+)"\s*,\s*&([^\s,)]+)'

        for match in re.finditer(pattern, text):
            python_name = match.group(1)
            cpp_name = match.group(2)

            # Calculate line number
            line_offset = text[:match.start()].count('\n')
            line_number = start_line + line_offset

            binding = Binding(
                python_name=python_name,
                cpp_name=cpp_name,
                binding_type='function',
                file_path=file_path,
                line_number=line_number,
                module_name=module_name,
                signature=match.group(0)
            )

            graph.add_binding(binding)

    def _extract_class_bindings(
        self,
        text: str,
        start_line: int,
        file_path: str,
        module_name: str,
        graph: BindingGraph
    ):
        """Extract py::class_<CppClass>(...) class bindings"""

        # Pattern: py::class_<CppClass>(m, "PyClass")
        # or: py::class_<CppClass, BaseClass>(m, "PyClass")
        class_pattern = r'py::class_<([^>]+)>\s*\(\s*m\s*,\s*"([^"]+)"'

        for match in re.finditer(class_pattern, text):
            cpp_class = match.group(1).split(',')[0].strip()  # First template arg
            python_name = match.group(2)

            # Calculate line number
            line_offset = text[:match.start()].count('\n')
            line_number = start_line + line_offset

            binding = Binding(
                python_name=python_name,
                cpp_name=cpp_class,
                binding_type='class',
                file_path=file_path,
                line_number=line_number,
                module_name=module_name,
                signature=match.group(0)
            )

            graph.add_binding(binding)

            # Find method bindings for this class
            # Look for .def("method", &CppClass::method) after this class definition
            self._extract_method_bindings(
                text, match.end(), start_line, file_path, module_name,
                cpp_class, python_name, graph
            )

    def _extract_method_bindings(
        self,
        text: str,
        class_def_end: int,
        start_line: int,
        file_path: str,
        module_name: str,
        cpp_class: str,
        python_class: str,
        graph: BindingGraph
    ):
        """Extract .def(...) method bindings for a class"""

        # Find the extent of this class binding (until next py::class_ or semicolon)
        next_class = text.find('py::class_', class_def_end)
        semicolon = text.find(';', class_def_end)

        if semicolon == -1:
            class_extent = len(text)
        elif next_class == -1:
            class_extent = semicolon
        else:
            class_extent = min(semicolon, next_class)

        class_body = text[class_def_end:class_extent]

        # Pattern: .def("method", &CppClass::method)
        method_pattern = r'\.def(?:_static|_readwrite|_readonly)?\s*\(\s*"([^"]+)"\s*,\s*&([^\s,)]+)'

        for match in re.finditer(method_pattern, class_body):
            python_method = match.group(1)
            cpp_method = match.group(2)

            # Calculate line number
            line_offset = text[:class_def_end + match.start()].count('\n')
            line_number = start_line + line_offset

            binding = Binding(
                python_name=f"{python_class}.{python_method}",
                cpp_name=cpp_method,
                binding_type='method',
                file_path=file_path,
                line_number=line_number,
                module_name=module_name,
                parent_class=cpp_class,
                signature=match.group(0)
            )

            graph.add_binding(binding)

    def _get_node_text(self, node, content: str) -> str:
        """Get text content of a node"""
        return content[node.start_byte:node.end_byte]

    def detect_bindings_in_directory(self, directory: str) -> BindingGraph:
        """
        Recursively scan a directory for pybind11 bindings.

        Args:
            directory: Root directory to scan

        Returns:
            Combined BindingGraph for all files
        """
        dir_path = Path(directory)
        combined_graph = BindingGraph()

        # Find all .cpp files
        binding_files = list(dir_path.rglob("*.cpp"))

        print(f"\n🔍 Scanning {len(binding_files)} C++ files for bindings...")

        for cpp_file in binding_files:
            try:
                content = cpp_file.read_text(encoding='utf-8')

                # Quick check for pybind11 before parsing
                if 'PYBIND11_MODULE' not in content and 'pybind11' not in content:
                    continue

                file_graph = self.detect_bindings(str(cpp_file), content)

                if file_graph.bindings:
                    print(f"   {cpp_file.name}: {len(file_graph.bindings)} bindings")

                    # Merge into combined graph
                    for binding in file_graph.bindings:
                        combined_graph.add_binding(binding)

            except Exception as e:
                print(f"   Error parsing {cpp_file.name}: {e}")

        return combined_graph


