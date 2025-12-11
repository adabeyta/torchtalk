#!/usr/bin/env python3
"""
Enhanced binding detector for cross-language Python/C++/CUDA connections.

Detects multiple binding patterns:
1. pybind11: PYBIND11_MODULE, m.def(), py::class_
2. PyTorch ATen: TORCH_LIBRARY, TORCH_LIBRARY_IMPL, native_functions.yaml dispatch
3. CUDA kernels: __global__ functions and their C++ wrappers

This enables rich cross-language code tracing for PyTorch-style codebases.
"""

import logging
from typing import List, Any, Tuple, Dict, Optional, Set
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum
import re

log = logging.getLogger(__name__)


class BindingType(Enum):
    """Types of cross-language bindings"""
    PYBIND_FUNCTION = "pybind_function"      # m.def("name", &func)
    PYBIND_CLASS = "pybind_class"            # py::class_<Cpp>(m, "Py")
    PYBIND_METHOD = "pybind_method"          # .def("method", &Class::method)
    TORCH_LIBRARY = "torch_library"          # TORCH_LIBRARY(namespace, m)
    TORCH_LIBRARY_IMPL = "torch_library_impl"  # TORCH_LIBRARY_IMPL(ns, dispatch_key, m)
    TORCH_OP = "torch_op"                    # m.def("op", ...)
    CUDA_KERNEL = "cuda_kernel"              # __global__ void kernel(...)
    CUDA_WRAPPER = "cuda_wrapper"            # C++ function that calls CUDA kernel
    AT_DISPATCH = "at_dispatch"              # AT_DISPATCH_FLOATING_TYPES(...)


class DispatchKey(Enum):
    """PyTorch dispatch keys for backend routing"""
    CPU = "CPU"
    CUDA = "CUDA"
    HIP = "HIP"
    MPS = "MPS"
    XLA = "XLA"
    Meta = "Meta"
    CompositeImplicitAutograd = "CompositeImplicitAutograd"
    CompositeExplicitAutograd = "CompositeExplicitAutograd"
    Autograd = "Autograd"
    AutogradCPU = "AutogradCPU"
    AutogradCUDA = "AutogradCUDA"


@dataclass
class Binding:
    """Represents a cross-language binding with rich metadata"""
    python_name: str              # Name visible in Python (e.g., "torch.matmul")
    cpp_name: str                 # C++ function/class name (e.g., "at::matmul")
    binding_type: str             # Type of binding (from BindingType)
    file_path: str                # Path to binding file
    line_number: int              # Line where binding is defined

    # Enhanced metadata
    dispatch_key: Optional[str] = None    # CPU, CUDA, etc.
    namespace: Optional[str] = None       # ATen namespace (aten, torch, etc.)
    cuda_kernel: Optional[str] = None     # Associated CUDA kernel name
    implementation_file: Optional[str] = None  # File with actual implementation
    docstring: Optional[str] = None       # Any docstring/description found
    signature: Optional[str] = None       # Function signature if available

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
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
    """Represents a CUDA kernel function"""
    name: str                     # Kernel function name
    file_path: str                # .cu file path
    line_number: int              # Line where kernel is defined
    template_params: Optional[str] = None  # Template parameters
    parameters: Optional[str] = None       # Function parameters
    called_by: List[str] = field(default_factory=list)  # C++ functions that invoke this


@dataclass
class BindingGraph:
    """Cross-language binding relationships with rich query support"""
    bindings: List[Binding] = field(default_factory=list)
    cuda_kernels: List[CUDAKernel] = field(default_factory=list)

    # Indexes for fast lookup
    _by_python_name: Dict[str, List[Binding]] = field(default_factory=dict)
    _by_cpp_name: Dict[str, List[Binding]] = field(default_factory=dict)
    _by_file: Dict[str, List[Binding]] = field(default_factory=dict)
    _by_dispatch_key: Dict[str, List[Binding]] = field(default_factory=dict)
    _kernels_by_name: Dict[str, CUDAKernel] = field(default_factory=dict)

    def add_binding(self, binding: Binding):
        """Add a binding and update indexes"""
        self.bindings.append(binding)

        # Index by Python name
        self._by_python_name.setdefault(binding.python_name, []).append(binding)

        # Index by C++ name
        self._by_cpp_name.setdefault(binding.cpp_name, []).append(binding)

        # Index by file
        self._by_file.setdefault(binding.file_path, []).append(binding)

        # Index by dispatch key
        if binding.dispatch_key:
            self._by_dispatch_key.setdefault(binding.dispatch_key, []).append(binding)

    def add_cuda_kernel(self, kernel: CUDAKernel):
        """Add a CUDA kernel"""
        self.cuda_kernels.append(kernel)
        self._kernels_by_name[kernel.name] = kernel

    def find_by_python_name(self, name: str, partial: bool = True) -> List[Binding]:
        """Find bindings by Python name"""
        if partial:
            return [b for b in self.bindings if name.lower() in b.python_name.lower()]
        return self._by_python_name.get(name, [])

    def find_by_cpp_name(self, name: str, partial: bool = True) -> List[Binding]:
        """Find bindings by C++ name"""
        if partial:
            return [b for b in self.bindings if name.lower() in b.cpp_name.lower()]
        return self._by_cpp_name.get(name, [])

    def find_dispatch_chain(self, function_name: str) -> Dict[str, List[Binding]]:
        """Find all dispatch implementations for a function (CPU, CUDA, etc.)"""
        result = {}
        for binding in self.find_by_python_name(function_name):
            key = binding.dispatch_key or "default"
            result.setdefault(key, []).append(binding)
        return result

    def find_cuda_kernel_for_function(self, function_name: str) -> List[CUDAKernel]:
        """Find CUDA kernels associated with a function"""
        kernels = []
        for kernel in self.cuda_kernels:
            if function_name.lower() in kernel.name.lower():
                kernels.append(kernel)
            # Also check callers
            for caller in kernel.called_by:
                if function_name.lower() in caller.lower():
                    kernels.append(kernel)
                    break
        return kernels


class BindingDetector:
    """
    Enhanced detector for cross-language bindings.

    Supported patterns:
    - pybind11: PYBIND11_MODULE, m.def(), py::class_
    - PyTorch: TORCH_LIBRARY, TORCH_LIBRARY_IMPL
    - CUDA: __global__ kernels, kernel launch syntax
    """

    def __init__(self):
        """Initialize parsers for binding detection"""
        from tree_sitter_language_pack import get_parser
        self.cpp_parser = get_parser('cpp')
        self.cuda_parser = get_parser('cuda')  # If available, falls back to cpp
        log.info("BindingDetector initialized with C++/CUDA support")

    def detect_bindings(self, file_path: str, content: str) -> BindingGraph:
        """
        Parse a C++/CUDA file and extract all cross-language bindings.

        Args:
            file_path: Path to the file
            content: File content as string

        Returns:
            BindingGraph with all detected bindings
        """
        graph = BindingGraph()

        is_cuda = file_path.endswith(('.cu', '.cuh'))
        parser = self.cuda_parser if is_cuda else self.cpp_parser

        try:
            tree = parser.parse(bytes(content, 'utf8'))
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

    def _detect_pybind11_bindings(self, node, content: str, file_path: str, graph: BindingGraph):
        """Detect pybind11 binding patterns"""

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
        """Find all PYBIND11_MODULE declarations"""
        modules = []

        if node.type == 'function_definition':
            text = self._get_node_text(node, content)
            match = re.search(r'PYBIND11_MODULE\s*\(\s*(\w+)\s*,', text)
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
        graph: BindingGraph
    ):
        """Extract all bindings from a PYBIND11_MODULE body"""
        body = None
        for child in module_node.children:
            if child.type == 'compound_statement':
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

    def _extract_standalone_pybind_patterns(self, content: str, file_path: str, graph: BindingGraph):
        """Extract pybind11 patterns not inside PYBIND11_MODULE (PyTorch style)"""

        # py::class_<CppClass>(m, "PyClass") or py::class_<CppClass>(parent, "PyClass")
        class_pattern = r'py::class_<([^>]+)>\s*\(\s*\w+\s*,\s*"([^"]+)"'

        for match in re.finditer(class_pattern, content):
            cpp_class = match.group(1).split(',')[0].strip()
            python_name = match.group(2)
            line_number = content[:match.start()].count('\n') + 1

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
            next_class = content.find('py::class_', class_end)
            if next_class == -1:
                next_class = len(content)

            class_body = content[class_end:next_class]

            # .def("method", &Class::method) or .def("method", [...])
            method_pattern = r'\.def(?:_static|_readwrite|_readonly)?\s*\(\s*"([^"]+)"'
            for method_match in re.finditer(method_pattern, class_body):
                method_name = method_match.group(1)
                method_line = content[:class_end + method_match.start()].count('\n') + 1

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
        graph: BindingGraph
    ):
        """Extract m.def(...) function bindings"""
        pattern = r'm\.def\s*\(\s*"([^"]+)"\s*,\s*&([^\s,)]+)'

        for match in re.finditer(pattern, text):
            python_name = match.group(1)
            cpp_name = match.group(2)
            line_offset = text[:match.start()].count('\n')
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
        graph: BindingGraph
    ):
        """Extract py::class_<CppClass>(...) class bindings"""
        class_pattern = r'py::class_<([^>]+)>\s*\(\s*m\s*,\s*"([^"]+)"'

        for match in re.finditer(class_pattern, text):
            cpp_class = match.group(1).split(',')[0].strip()
            python_name = match.group(2)
            line_offset = text[:match.start()].count('\n')
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
        next_class = text.find('py::class_', class_def_end)
        semicolon = text.find(';', class_def_end)

        if semicolon == -1:
            class_extent = len(text)
        elif next_class == -1:
            class_extent = semicolon
        else:
            class_extent = min(semicolon, next_class)

        class_body = text[class_def_end:class_extent]
        method_pattern = r'\.def(?:_static|_readwrite|_readonly)?\s*\(\s*"([^"]+)"\s*,\s*&([^\s,)]+)'

        for match in re.finditer(method_pattern, class_body):
            python_method = match.group(1)
            cpp_method = match.group(2)
            line_offset = text[:class_def_end + match.start()].count('\n')
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

    def _detect_torch_library_bindings(self, content: str, file_path: str, graph: BindingGraph):
        """Detect TORCH_LIBRARY, TORCH_LIBRARY_IMPL, and TORCH_LIBRARY_FRAGMENT patterns"""

        # TORCH_LIBRARY(namespace, m) { ... }
        # TORCH_LIBRARY_FRAGMENT(namespace, m) { ... }
        lib_pattern = r'TORCH_LIBRARY(?:_FRAGMENT)?\s*\(\s*(\w+)\s*,\s*(\w+)\s*\)'
        for match in re.finditer(lib_pattern, content):
            namespace = match.group(1)
            line_number = content[:match.start()].count('\n') + 1

            # Find ops defined in this library block
            block_start = content.find('{', match.end())
            if block_start != -1:
                block_end = self._find_matching_brace(content, block_start)
                block_content = content[block_start:block_end]

                self._extract_torch_ops(block_content, line_number, file_path, namespace, None, graph)

        # TORCH_LIBRARY_IMPL(namespace, dispatch_key, m) { ... }
        # namespace can be _ for catch-all
        impl_pattern = r'TORCH_LIBRARY_IMPL\s*\(\s*([_\w]+)\s*,\s*(\w+)\s*,\s*(\w+)\s*\)'
        for match in re.finditer(impl_pattern, content):
            namespace = match.group(1)
            dispatch_key = match.group(2)
            line_number = content[:match.start()].count('\n') + 1

            block_start = content.find('{', match.end())
            if block_start != -1:
                block_end = self._find_matching_brace(content, block_start)
                block_content = content[block_start:block_end]

                self._extract_torch_ops(block_content, line_number, file_path, namespace, dispatch_key, graph)

    def _extract_torch_ops(
        self,
        block_content: str,
        base_line: int,
        file_path: str,
        namespace: str,
        dispatch_key: Optional[str],
        graph: BindingGraph
    ):
        """Extract operator definitions from TORCH_LIBRARY blocks"""

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
                op_name_match = re.match(r'(?:\w+::)?(\w+)', op_signature)
                if op_name_match:
                    op_name = op_name_match.group(1)
                    line_offset = block_content[:match.start()].count('\n')

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
            line_offset = block_content[:match.start()].count('\n')

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
        """Detect CUDA kernel definitions"""

        # __global__ void kernel_name<T>(args...) or __global__ void kernel_name(args...)
        kernel_pattern = r'__global__\s+void\s+(\w+)(?:<([^>]+)>)?\s*\(([^)]*)\)'

        for match in re.finditer(kernel_pattern, content):
            kernel_name = match.group(1)
            template_params = match.group(2)
            parameters = match.group(3)
            line_number = content[:match.start()].count('\n') + 1

            kernel = CUDAKernel(
                name=kernel_name,
                file_path=file_path,
                line_number=line_number,
                template_params=template_params,
                parameters=parameters,
            )

            # Find what calls this kernel (look for kernel<<<...>>>)
            launch_pattern = rf'{kernel_name}\s*<<<'
            for launch_match in re.finditer(launch_pattern, content):
                # Try to find enclosing function
                enclosing_func = self._find_enclosing_function(content, launch_match.start())
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
        """Detect AT_DISPATCH macros for type dispatch"""

        # AT_DISPATCH_FLOATING_TYPES, AT_DISPATCH_ALL_TYPES, etc.
        dispatch_pattern = r'(AT_DISPATCH_\w+)\s*\(\s*([^,]+)\s*,\s*"([^"]+)"'

        for match in re.finditer(dispatch_pattern, content):
            macro_name = match.group(1)
            tensor_type = match.group(2)
            op_name = match.group(3)
            line_number = content[:match.start()].count('\n') + 1

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
        """Find the matching closing brace"""
        depth = 1
        i = start + 1
        while i < len(content) and depth > 0:
            if content[i] == '{':
                depth += 1
            elif content[i] == '}':
                depth -= 1
            i += 1
        return i

    def _find_enclosing_function(self, content: str, position: int) -> Optional[str]:
        """Find the function name that encloses a given position"""
        # Look backwards for function definition
        search_region = content[:position]

        # Pattern for C++ function definition
        func_pattern = r'(?:static\s+)?(?:inline\s+)?(?:\w+::)*(\w+)\s*\([^)]*\)\s*(?:const\s*)?\{'

        matches = list(re.finditer(func_pattern, search_region))
        if matches:
            # Return the last (most recent) match
            return matches[-1].group(1)
        return None

    def _get_node_text(self, node, content: str) -> str:
        """Get text content of a node"""
        return content[node.start_byte:node.end_byte]

    def detect_bindings_in_directory(self, directory: str) -> BindingGraph:
        """
        Recursively scan a directory for cross-language bindings.

        Args:
            directory: Root directory to scan

        Returns:
            Combined BindingGraph for all files
        """
        dir_path = Path(directory)
        combined_graph = BindingGraph()

        # Find all C++/CUDA files
        extensions = ['*.cpp', '*.cc', '*.cxx', '*.cu', '*.cuh']
        files = []
        for ext in extensions:
            files.extend(dir_path.rglob(ext))

        log.info(f"Scanning {len(files)} C++/CUDA files for bindings...")

        for cpp_file in files:
            try:
                content = cpp_file.read_text(encoding='utf-8', errors='replace')

                # Quick check for relevant patterns
                has_pybind = 'PYBIND11_MODULE' in content or 'pybind11' in content
                has_torch_lib = 'TORCH_LIBRARY' in content
                has_cuda = '__global__' in content
                has_dispatch = 'AT_DISPATCH' in content

                if not (has_pybind or has_torch_lib or has_cuda or has_dispatch):
                    continue

                file_graph = self.detect_bindings(str(cpp_file), content)

                if file_graph.bindings:
                    log.info(f"   {cpp_file.name}: {len(file_graph.bindings)} bindings, {len(file_graph.cuda_kernels)} kernels")

                    # Merge into combined graph
                    for binding in file_graph.bindings:
                        combined_graph.add_binding(binding)
                    for kernel in file_graph.cuda_kernels:
                        combined_graph.add_cuda_kernel(kernel)

            except Exception as e:
                log.warning(f"   Error parsing {cpp_file.name}: {e}")

        log.info(f"Total: {len(combined_graph.bindings)} bindings, {len(combined_graph.cuda_kernels)} CUDA kernels")
        return combined_graph
