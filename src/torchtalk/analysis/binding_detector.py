#!/usr/bin/env python3
"""Cross-language binding detector for Python/C++/CUDA connections."""

import logging
from typing import List, Any, Tuple, Dict, Optional
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum
import re

from .config import should_exclude, has_binding_patterns

log = logging.getLogger(__name__)


class BindingType(Enum):
    """Cross-language binding types."""

    PYBIND_FUNCTION = "pybind_function"  # m.def("name", &func)
    PYBIND_CLASS = "pybind_class"  # py::class_<Cpp>(m, "Py")
    PYBIND_METHOD = "pybind_method"  # .def("method", &Class::method)
    TORCH_LIBRARY = "torch_library"  # TORCH_LIBRARY(namespace, m)
    TORCH_LIBRARY_IMPL = "torch_library_impl"  # TORCH_LIBRARY_IMPL(ns, dispatch_key, m)
    TORCH_OP = "torch_op"  # m.def("op", ...)
    CUDA_KERNEL = "cuda_kernel"  # __global__ void kernel(...)
    CUDA_WRAPPER = "cuda_wrapper"  # C++ function that calls CUDA kernel
    AT_DISPATCH = "at_dispatch"  # AT_DISPATCH_FLOATING_TYPES(...)


@dataclass
class Binding:
    """Cross-language binding with metadata."""

    python_name: str
    cpp_name: str
    binding_type: str
    file_path: str
    line_number: int
    dispatch_key: Optional[str] = None
    namespace: Optional[str] = None
    cuda_kernel: Optional[str] = None
    implementation_file: Optional[str] = None
    docstring: Optional[str] = None
    signature: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            "python_name": self.python_name,
            "cpp_name": self.cpp_name,
            "type": self.binding_type,
            "file_path": self.file_path,
            "line": self.line_number,
            "dispatch_key": self.dispatch_key,
            "namespace": self.namespace,
            "cuda_kernel": self.cuda_kernel,
            "implementation_file": self.implementation_file,
            "signature": self.signature,
        }


@dataclass
class CUDAKernel:
    """CUDA kernel (__global__ function)."""

    name: str
    file_path: str
    line_number: int
    template_params: Optional[str] = None
    parameters: Optional[str] = None
    called_by: List[str] = field(default_factory=list)


@dataclass
class BindingGraph:
    """Container for detected bindings and CUDA kernels."""

    bindings: List[Binding] = field(default_factory=list)
    cuda_kernels: List[CUDAKernel] = field(default_factory=list)

    def add_binding(self, binding: Binding):
        self.bindings.append(binding)

    def add_cuda_kernel(self, kernel: CUDAKernel):
        self.cuda_kernels.append(kernel)


class BindingDetector:
    """Detects pybind11, TORCH_LIBRARY, and CUDA binding patterns."""

    def __init__(self):
        from tree_sitter_language_pack import get_parser

        self.cpp_parser = get_parser("cpp")
        self.cuda_parser = get_parser("cuda")  # If available, falls back to cpp
        log.info("BindingDetector initialized with C++/CUDA support")

    def detect_bindings(self, file_path: str, content: str) -> BindingGraph:
        """Parse a C++/CUDA file and extract bindings."""
        graph = BindingGraph()

        is_cuda = file_path.endswith((".cu", ".cuh"))
        parser = self.cuda_parser if is_cuda else self.cpp_parser

        try:
            tree = parser.parse(bytes(content, "utf8"))
            root_node = tree.root_node
        except Exception as e:
            log.warning(f"Parse error for {file_path}: {e}")
            return graph

        # Detect different binding patterns
        self._detect_pybind11_bindings(root_node, content, file_path, graph)
        self._detect_torch_library_bindings(content, file_path, graph)

        if is_cuda:
            self._detect_cuda_kernels(content, file_path, graph)

        self._detect_at_dispatch(content, file_path, graph)

        return graph

    def _detect_pybind11_bindings(
        self, node, content: str, file_path: str, graph: BindingGraph
    ):
        # Find PYBIND11_MODULE declarations
        modules = self._find_pybind11_modules(node, content)

        for module_name, module_node in modules:
            self._extract_module_bindings(
                module_node, content, file_path, module_name, graph
            )

        # Also scan entire file for py::class_ patterns (PyTorch style)
        # These may not be inside PYBIND11_MODULE but are still pybind11 bindings
        self._extract_standalone_pybind_patterns(content, file_path, graph)

    def _find_pybind11_modules(self, node, content: str) -> List[Tuple[str, Any]]:
        modules = []

        if node.type == "function_definition":
            text = self._get_node_text(node, content)
            match = re.search(r"PYBIND11_MODULE\s*\(\s*(\w+)\s*,", text)
            if match:
                module_name = match.group(1)
                modules.append((module_name, node))

        for child in node.children:
            modules.extend(self._find_pybind11_modules(child, content))

        return modules

    def _extract_module_bindings(
        self,
        module_node,
        content: str,
        file_path: str,
        module_name: str,
        graph: BindingGraph,
    ):
        body = None
        for child in module_node.children:
            if child.type == "compound_statement":
                body = child
                break

        if not body:
            return

        body_text = self._get_node_text(body, content)
        body_start_line = body.start_point[0] + 1

        # Extract function bindings
        self._extract_function_bindings(
            body_text, body_start_line, file_path, module_name, graph
        )

        # Extract class bindings
        self._extract_class_bindings(
            body_text, body_start_line, file_path, module_name, graph
        )

    def _extract_standalone_pybind_patterns(
        self, content: str, file_path: str, graph: BindingGraph
    ):
        # py::class_<CppClass>(m, "PyClass") or py::class_<CppClass>(parent, "PyClass")
        class_pattern = r'py::class_<([^>]+)>\s*\(\s*\w+\s*,\s*"([^"]+)"'

        for match in re.finditer(class_pattern, content):
            cpp_class = match.group(1).split(",")[0].strip()
            python_name = match.group(2)
            line_number = content[: match.start()].count("\n") + 1

            binding = Binding(
                python_name=python_name,
                cpp_name=cpp_class,
                binding_type=BindingType.PYBIND_CLASS.value,
                file_path=file_path,
                line_number=line_number,
            )
            graph.add_binding(binding)

            # Find chained .def() methods for this class
            # Look for .def("name", ...) patterns after the class definition
            class_end = match.end()
            # Find the semicolon or next py::class_ to bound the search
            next_class = content.find("py::class_", class_end)
            if next_class == -1:
                next_class = len(content)

            class_body = content[class_end:next_class]

            # .def("method", &Class::method) or .def("method", [...])
            method_pattern = r'\.def(?:_static|_readwrite|_readonly)?\s*\(\s*"([^"]+)"'
            for method_match in re.finditer(method_pattern, class_body):
                method_name = method_match.group(1)
                method_line = (
                    content[: class_end + method_match.start()].count("\n") + 1
                )

                binding = Binding(
                    python_name=f"{python_name}.{method_name}",
                    cpp_name=f"{cpp_class}::{method_name}",
                    binding_type=BindingType.PYBIND_METHOD.value,
                    file_path=file_path,
                    line_number=method_line,
                )
                graph.add_binding(binding)

    def _extract_function_bindings(
        self,
        text: str,
        start_line: int,
        file_path: str,
        module_name: str,
        graph: BindingGraph,
    ):
        pattern = r'm\.def\s*\(\s*"([^"]+)"\s*,\s*&([^\s,)]+)'

        for match in re.finditer(pattern, text):
            python_name = match.group(1)
            cpp_name = match.group(2)
            line_offset = text[: match.start()].count("\n")
            line_number = start_line + line_offset

            binding = Binding(
                python_name=python_name,
                cpp_name=cpp_name,
                binding_type=BindingType.PYBIND_FUNCTION.value,
                file_path=file_path,
                line_number=line_number,
                namespace=module_name,
            )
            graph.add_binding(binding)

    def _extract_class_bindings(
        self,
        text: str,
        start_line: int,
        file_path: str,
        module_name: str,
        graph: BindingGraph,
    ):
        class_pattern = r'py::class_<([^>]+)>\s*\(\s*m\s*,\s*"([^"]+)"'

        for match in re.finditer(class_pattern, text):
            cpp_class = match.group(1).split(",")[0].strip()
            python_name = match.group(2)
            line_offset = text[: match.start()].count("\n")
            line_number = start_line + line_offset

            binding = Binding(
                python_name=python_name,
                cpp_name=cpp_class,
                binding_type=BindingType.PYBIND_CLASS.value,
                file_path=file_path,
                line_number=line_number,
                namespace=module_name,
            )
            graph.add_binding(binding)

            # Extract method bindings
            self._extract_method_bindings(
                text,
                match.end(),
                start_line,
                file_path,
                module_name,
                cpp_class,
                python_name,
                graph,
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
        graph: BindingGraph,
    ):
        next_class = text.find("py::class_", class_def_end)
        semicolon = text.find(";", class_def_end)

        if semicolon == -1:
            class_extent = len(text)
        elif next_class == -1:
            class_extent = semicolon
        else:
            class_extent = min(semicolon, next_class)

        class_body = text[class_def_end:class_extent]
        method_pattern = (
            r'\.def(?:_static|_readwrite|_readonly)?\s*\(\s*"([^"]+)"\s*,\s*&([^\s,)]+)'
        )

        for match in re.finditer(method_pattern, class_body):
            python_method = match.group(1)
            cpp_method = match.group(2)
            line_offset = text[: class_def_end + match.start()].count("\n")
            line_number = start_line + line_offset

            binding = Binding(
                python_name=f"{python_class}.{python_method}",
                cpp_name=cpp_method,
                binding_type=BindingType.PYBIND_METHOD.value,
                file_path=file_path,
                line_number=line_number,
                namespace=module_name,
            )
            graph.add_binding(binding)

    def _detect_torch_library_bindings(
        self, content: str, file_path: str, graph: BindingGraph
    ):
        # TORCH_LIBRARY(namespace, m) { ... }
        # TORCH_LIBRARY_FRAGMENT(namespace, m) { ... }
        lib_pattern = r"TORCH_LIBRARY(?:_FRAGMENT)?\s*\(\s*(\w+)\s*,\s*(\w+)\s*\)"
        for match in re.finditer(lib_pattern, content):
            namespace = match.group(1)
            line_number = content[: match.start()].count("\n") + 1

            # Find ops defined in this library block
            block_start = content.find("{", match.end())
            if block_start != -1:
                block_end = self._find_matching_brace(content, block_start)
                block_content = content[block_start:block_end]

                self._extract_torch_ops(
                    block_content, line_number, file_path, namespace, None, graph
                )

        # TORCH_LIBRARY_IMPL(namespace, dispatch_key, m) { ... }
        # namespace can be _ for catch-all
        impl_pattern = (
            r"TORCH_LIBRARY_IMPL\s*\(\s*([_\w]+)\s*,\s*(\w+)\s*,\s*(\w+)\s*\)"
        )
        for match in re.finditer(impl_pattern, content):
            namespace = match.group(1)
            dispatch_key = match.group(2)
            line_number = content[: match.start()].count("\n") + 1

            block_start = content.find("{", match.end())
            if block_start != -1:
                block_end = self._find_matching_brace(content, block_start)
                block_content = content[block_start:block_end]

                self._extract_torch_ops(
                    block_content,
                    line_number,
                    file_path,
                    namespace,
                    dispatch_key,
                    graph,
                )

    def _extract_torch_ops(
        self,
        block_content: str,
        base_line: int,
        file_path: str,
        namespace: str,
        dispatch_key: Optional[str],
        graph: BindingGraph,
    ):
        # m.def("op_name(Tensor self) -> Tensor", ...)
        # Also handles: m.def(TORCH_SELECTIVE_SCHEMA("aten::op_name(...)"))
        def_patterns = [
            r'm\.def\s*\(\s*"([^"]+)"',  # Direct string
            r'm\.def\s*\(\s*TORCH_SELECTIVE_SCHEMA\s*\(\s*"([^"]+)"',  # Via macro
        ]

        for def_pattern in def_patterns:
            for match in re.finditer(def_pattern, block_content):
                op_signature = match.group(1)
                # Extract op name - may be prefixed like "aten::op_name" or "quantized::op_name"
                # Pattern: optional_namespace::op_name(...)
                op_name_match = re.match(r"(?:\w+::)?(\w+)", op_signature)
                if op_name_match:
                    op_name = op_name_match.group(1)
                    line_offset = block_content[: match.start()].count("\n")

                    binding = Binding(
                        python_name=f"{namespace}.{op_name}",
                        cpp_name=op_name,
                        binding_type=BindingType.TORCH_OP.value,
                        file_path=file_path,
                        line_number=base_line + line_offset,
                        dispatch_key=dispatch_key,
                        namespace=namespace,
                        signature=op_signature,
                    )
                    graph.add_binding(binding)

        # m.impl("op_name", function_ptr)
        impl_pattern = r'm\.impl\s*\(\s*"([^"]+)"\s*,\s*([^\s,)]+)'
        for match in re.finditer(impl_pattern, block_content):
            op_name = match.group(1)
            cpp_func = match.group(2)
            line_offset = block_content[: match.start()].count("\n")

            binding = Binding(
                python_name=f"{namespace}.{op_name}",
                cpp_name=cpp_func,
                binding_type=BindingType.TORCH_LIBRARY_IMPL.value,
                file_path=file_path,
                line_number=base_line + line_offset,
                dispatch_key=dispatch_key,
                namespace=namespace,
            )
            graph.add_binding(binding)

    def _detect_cuda_kernels(self, content: str, file_path: str, graph: BindingGraph):
        # __global__ void kernel_name<T>(args...) or __global__ void kernel_name(args...)
        kernel_pattern = r"__global__\s+void\s+(\w+)(?:<([^>]+)>)?\s*\(([^)]*)\)"

        for match in re.finditer(kernel_pattern, content):
            kernel_name = match.group(1)
            template_params = match.group(2)
            parameters = match.group(3)
            line_number = content[: match.start()].count("\n") + 1

            kernel = CUDAKernel(
                name=kernel_name,
                file_path=file_path,
                line_number=line_number,
                template_params=template_params,
                parameters=parameters,
            )

            # Find what calls this kernel (look for kernel<<<...>>>)
            launch_pattern = rf"{kernel_name}\s*<<<"
            for launch_match in re.finditer(launch_pattern, content):
                # Try to find enclosing function
                enclosing_func = self._find_enclosing_function(
                    content, launch_match.start()
                )
                if enclosing_func and enclosing_func not in kernel.called_by:
                    kernel.called_by.append(enclosing_func)

            graph.add_cuda_kernel(kernel)

            # Also add as a binding for unified search
            binding = Binding(
                python_name=kernel_name,  # Kernels typically have internal names
                cpp_name=kernel_name,
                binding_type=BindingType.CUDA_KERNEL.value,
                file_path=file_path,
                line_number=line_number,
                dispatch_key="CUDA",
                signature=parameters,
            )
            graph.add_binding(binding)

    def _detect_at_dispatch(self, content: str, file_path: str, graph: BindingGraph):
        # AT_DISPATCH_FLOATING_TYPES, AT_DISPATCH_ALL_TYPES, etc.
        dispatch_pattern = r'(AT_DISPATCH_\w+)\s*\(\s*([^,]+)\s*,\s*"([^"]+)"'

        for match in re.finditer(dispatch_pattern, content):
            macro_name = match.group(1)
            tensor_type = match.group(2)
            op_name = match.group(3)
            line_number = content[: match.start()].count("\n") + 1

            # Find the enclosing function to understand what op this is for
            enclosing_func = self._find_enclosing_function(content, match.start())

            binding = Binding(
                python_name=op_name,
                cpp_name=enclosing_func or op_name,
                binding_type=BindingType.AT_DISPATCH.value,
                file_path=file_path,
                line_number=line_number,
                signature=f"{macro_name}({tensor_type})",
            )
            graph.add_binding(binding)

    def _find_matching_brace(self, content: str, start: int) -> int:
        depth = 1
        i = start + 1
        while i < len(content) and depth > 0:
            if content[i] == "{":
                depth += 1
            elif content[i] == "}":
                depth -= 1
            i += 1
        return i

    def _find_enclosing_function(self, content: str, position: int) -> Optional[str]:
        # Look backwards for function definition
        search_region = content[:position]

        # Pattern for C++ function definition
        func_pattern = (
            r"(?:static\s+)?(?:inline\s+)?(?:\w+::)*(\w+)\s*\([^)]*\)\s*(?:const\s*)?\{"
        )

        matches = list(re.finditer(func_pattern, search_region))
        if matches:
            # Return the last (most recent) match
            return matches[-1].group(1)
        return None

    def _get_node_text(self, node, content: str) -> str:
        return content[node.start_byte : node.end_byte]

    def detect_bindings_in_directory(self, directory: str) -> BindingGraph:
        """Scan a directory for cross-language bindings.

        Uses shared configuration from config.py for exclusion and binding patterns.

        Args:
            directory: Directory to scan
        """
        dir_path = Path(directory)
        combined_graph = BindingGraph()

        # Find all C++/CUDA files
        extensions = ["*.cpp", "*.cc", "*.cxx", "*.cu", "*.cuh"]
        files = []
        for ext in extensions:
            files.extend(dir_path.rglob(ext))

        log.info(f"Scanning {len(files)} C++/CUDA files for bindings...")

        skipped_excluded = 0
        skipped_no_patterns = 0
        for cpp_file in files:
            file_str = str(cpp_file)

            # Apply exclusion patterns (from shared config)
            if should_exclude(file_str):
                skipped_excluded += 1
                continue

            try:
                content = cpp_file.read_text(encoding="utf-8", errors="replace")

                # Fuzzy grep: check for binding-related patterns (from shared config)
                if not has_binding_patterns(content):
                    skipped_no_patterns += 1
                    continue

                file_graph = self.detect_bindings(str(cpp_file), content)

                if file_graph.bindings or file_graph.cuda_kernels:
                    log.debug(
                        f"   {cpp_file.name}: {len(file_graph.bindings)} bindings, "
                        f"{len(file_graph.cuda_kernels)} kernels"
                    )

                    # Merge into combined graph
                    for binding in file_graph.bindings:
                        combined_graph.add_binding(binding)
                    for kernel in file_graph.cuda_kernels:
                        combined_graph.add_cuda_kernel(kernel)

            except Exception as e:
                log.warning(f"Error parsing {cpp_file.name}: {e}")

        log.info(
            f"Total: {len(combined_graph.bindings)} bindings, "
            f"{len(combined_graph.cuda_kernels)} CUDA kernels "
            f"(skipped {skipped_excluded} excluded, {skipped_no_patterns} without patterns)"
        )
        return combined_graph
