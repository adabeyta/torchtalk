#!/usr/bin/env python3
"""
Code chunker for splitting source files into meaningful chunks for embedding.
Supports Python, C++, and CUDA code.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path
import ast

# Import C++/CUDA parser (Phase 6)
from torchtalk.analysis.cpp_cuda_parser import CppCudaParser, CppEntity


@dataclass
class CodeChunk:
    """Represents a chunk of code with metadata"""
    content: str
    metadata: Dict[str, Any]
    start_line: int
    end_line: int
    chunk_type: str  # 'function', 'class', 'method', 'module_level'


class CodeChunker:
    """
    Split code files into meaningful chunks for embedding.

    Strategy:
    - Functions: Each function is a chunk
    - Classes: Each class with its methods is a chunk (or split if too large)
    - Module-level code: Group imports, constants, etc. as separate chunks

    Supports:
    - Python: AST-based parsing
    - C++/CUDA: tree-sitter-based parsing
    """

    def __init__(
        self,
        max_chars: int = 2000,
        min_chars: int = 100,
        language: str = "python"
    ):
        """
        Args:
            max_chars: Maximum characters per chunk
            min_chars: Minimum characters per chunk (merge small chunks)
            language: Programming language ('python', 'cpp', 'cuda')
        """
        self.max_chars = max_chars
        self.min_chars = min_chars
        self.language = language

        # Initialize C++/CUDA parser if needed
        if language in ['cpp', 'cuda']:
            self.cpp_parser = CppCudaParser()
        else:
            self.cpp_parser = None

    def chunk_file(
        self,
        file_path: str,
        content: str,
        base_metadata: Optional[Dict[str, Any]] = None
    ) -> List[CodeChunk]:
        """
        Chunk a source file into meaningful pieces.

        Args:
            file_path: Path to the file
            content: File content as string
            base_metadata: Base metadata to attach to all chunks (e.g., imports)

        Returns:
            List of CodeChunk objects
        """
        if self.language == "python":
            return self._chunk_python(file_path, content, base_metadata or {})
        elif self.language in ['cpp', 'cuda']:
            return self._chunk_cpp_cuda(file_path, content, base_metadata or {})
        else:
            raise NotImplementedError(f"Language {self.language} not yet supported")

    def _chunk_python(
        self,
        file_path: str,
        content: str,
        base_metadata: Dict[str, Any]
    ) -> List[CodeChunk]:
        """Chunk Python code using AST parsing"""
        chunks = []
        lines = content.splitlines()

        try:
            tree = ast.parse(content)
        except SyntaxError as e:
            # If parsing fails, return whole file as one chunk
            return [CodeChunk(
                content=content,
                metadata={
                    **base_metadata,
                    'file': file_path,
                    'error': f'Syntax error: {str(e)}',
                    'type': 'file'
                },
                start_line=1,
                end_line=len(lines),
                chunk_type='module_level'
            )]

        # Extract top-level elements
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.FunctionDef):
                # Top-level function
                chunk = self._extract_function_chunk(
                    node, lines, file_path, base_metadata
                )
                if chunk:
                    chunks.append(chunk)

            elif isinstance(node, ast.ClassDef):
                # Class with methods
                class_chunks = self._extract_class_chunks(
                    node, lines, file_path, base_metadata
                )
                chunks.extend(class_chunks)

            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                # Skip imports - they'll be in base_metadata
                continue

            elif isinstance(node, ast.Assign):
                # Module-level constants/variables
                # These will be captured in module_level chunk below
                continue

        # Add module-level chunk (imports, docstring, constants)
        module_chunk = self._extract_module_level_chunk(
            tree, lines, file_path, base_metadata
        )
        if module_chunk:
            chunks.insert(0, module_chunk)  # Put at beginning

        # Merge very small chunks
        chunks = self._merge_small_chunks(chunks)

        return chunks

    def _extract_function_chunk(
        self,
        node: ast.FunctionDef,
        lines: List[str],
        file_path: str,
        base_metadata: Dict[str, Any]
    ) -> Optional[CodeChunk]:
        """Extract a function as a chunk"""
        start_line = node.lineno
        end_line = node.end_lineno or start_line

        # Get source code
        source_lines = lines[start_line - 1:end_line]
        source = '\n'.join(source_lines)

        if len(source) < self.min_chars:
            return None  # Too small, will be merged later

        # Extract function signature
        args = [arg.arg for arg in node.args.args]
        signature = f"def {node.name}({', '.join(args)})"

        # Extract docstring
        docstring = ast.get_docstring(node)

        # Build metadata
        metadata = {
            **base_metadata,
            'file': file_path,
            'type': 'function',
            'name': node.name,
            'signature': signature,
            'docstring': docstring,
            'line_start': start_line,
            'line_end': end_line,
        }

        # If too large, truncate (we'll split better in Phase 6)
        if len(source) > self.max_chars:
            source = source[:self.max_chars] + "\n# [TRUNCATED]"

        return CodeChunk(
            content=source,
            metadata=metadata,
            start_line=start_line,
            end_line=end_line,
            chunk_type='function'
        )

    def _extract_class_chunks(
        self,
        node: ast.ClassDef,
        lines: List[str],
        file_path: str,
        base_metadata: Dict[str, Any]
    ) -> List[CodeChunk]:
        """Extract class and its methods as chunks"""
        chunks = []

        start_line = node.lineno
        end_line = node.end_lineno or start_line

        # Get full class source
        source_lines = lines[start_line - 1:end_line]
        full_source = '\n'.join(source_lines)

        # Get class docstring and signature
        docstring = ast.get_docstring(node)
        bases = [self._get_name(base) for base in node.bases]
        signature = f"class {node.name}" + (f"({', '.join(bases)})" if bases else "")

        # If class is small enough, return as single chunk
        if len(full_source) <= self.max_chars:
            metadata = {
                **base_metadata,
                'file': file_path,
                'type': 'class',
                'name': node.name,
                'signature': signature,
                'docstring': docstring,
                'line_start': start_line,
                'line_end': end_line,
                'bases': bases
            }

            return [CodeChunk(
                content=full_source,
                metadata=metadata,
                start_line=start_line,
                end_line=end_line,
                chunk_type='class'
            )]

        # Class is large - split into class definition + methods
        # 1. Class definition chunk (without method bodies)
        class_def = self._get_class_definition_only(node, lines)
        if class_def:
            chunks.append(CodeChunk(
                content=class_def,
                metadata={
                    **base_metadata,
                    'file': file_path,
                    'type': 'class_definition',
                    'name': node.name,
                    'signature': signature,
                    'docstring': docstring,
                    'bases': bases
                },
                start_line=start_line,
                end_line=start_line + len(class_def.splitlines()),
                chunk_type='class'
            ))

        # 2. Each method as a separate chunk
        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                method_chunk = self._extract_method_chunk(
                    item, lines, file_path, node.name, base_metadata
                )
                if method_chunk:
                    chunks.append(method_chunk)

        return chunks

    def _extract_method_chunk(
        self,
        node: ast.FunctionDef,
        lines: List[str],
        file_path: str,
        class_name: str,
        base_metadata: Dict[str, Any]
    ) -> Optional[CodeChunk]:
        """Extract a class method as a chunk"""
        start_line = node.lineno
        end_line = node.end_lineno or start_line

        source_lines = lines[start_line - 1:end_line]
        source = '\n'.join(source_lines)

        if len(source) < self.min_chars:
            return None

        # Extract signature
        args = [arg.arg for arg in node.args.args]
        signature = f"def {node.name}({', '.join(args)})"

        docstring = ast.get_docstring(node)

        metadata = {
            **base_metadata,
            'file': file_path,
            'type': 'method',
            'class_name': class_name,
            'name': node.name,
            'signature': signature,
            'docstring': docstring,
            'full_name': f"{class_name}.{node.name}",
            'line_start': start_line,
            'line_end': end_line,
        }

        if len(source) > self.max_chars:
            source = source[:self.max_chars] + "\n# [TRUNCATED]"

        return CodeChunk(
            content=source,
            metadata=metadata,
            start_line=start_line,
            end_line=end_line,
            chunk_type='method'
        )

    def _extract_module_level_chunk(
        self,
        tree: ast.AST,
        lines: List[str],
        file_path: str,
        base_metadata: Dict[str, Any]
    ) -> Optional[CodeChunk]:
        """Extract module-level imports, docstring, and constants"""
        module_parts = []

        # Module docstring
        docstring = ast.get_docstring(tree)
        if docstring:
            module_parts.append(f'"""{docstring}"""')

        # Imports
        imports = []
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                import_line = lines[node.lineno - 1]
                imports.append(import_line)

        if imports:
            module_parts.append('\n'.join(imports))

        # Module-level constants (first few)
        constants = []
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.Assign) and len(constants) < 10:
                const_line = lines[node.lineno - 1]
                constants.append(const_line)

        if constants:
            module_parts.append('\n'.join(constants))

        if not module_parts:
            return None

        content = '\n\n'.join(module_parts)

        return CodeChunk(
            content=content,
            metadata={
                **base_metadata,
                'file': file_path,
                'type': 'module_level',
                'docstring': docstring,
            },
            start_line=1,
            end_line=len(module_parts),
            chunk_type='module_level'
        )

    def _get_class_definition_only(
        self,
        node: ast.ClassDef,
        lines: List[str]
    ) -> str:
        """Get class definition without method bodies"""
        parts = []

        # Class declaration
        bases = [self._get_name(base) for base in node.bases]
        class_decl = f"class {node.name}" + (f"({', '.join(bases)}):" if bases else ":")
        parts.append(class_decl)

        # Docstring
        docstring = ast.get_docstring(node)
        if docstring:
            parts.append(f'    """{docstring}"""')

        # Method signatures only
        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                args = [arg.arg for arg in item.args.args]
                method_sig = f"    def {item.name}({', '.join(args)}): ..."
                parts.append(method_sig)

        return '\n'.join(parts)

    def _get_name(self, node: ast.AST) -> str:
        """Get name from AST node"""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self._get_name(node.value)}.{node.attr}"
        return "unknown"

    def _merge_small_chunks(self, chunks: List[CodeChunk]) -> List[CodeChunk]:
        """Merge chunks that are too small"""
        if not chunks:
            return chunks

        merged = []
        buffer = []
        buffer_size = 0

        for chunk in chunks:
            chunk_size = len(chunk.content)

            if chunk_size >= self.min_chars:
                # Flush buffer first
                if buffer:
                    merged.append(self._combine_chunks(buffer))
                    buffer = []
                    buffer_size = 0

                # Add this chunk
                merged.append(chunk)
            else:
                # Add to buffer
                buffer.append(chunk)
                buffer_size += chunk_size

                # Flush if buffer is big enough
                if buffer_size >= self.min_chars:
                    merged.append(self._combine_chunks(buffer))
                    buffer = []
                    buffer_size = 0

        # Flush remaining buffer
        if buffer:
            merged.append(self._combine_chunks(buffer))

        return merged

    def _combine_chunks(self, chunks: List[CodeChunk]) -> CodeChunk:
        """Combine multiple small chunks into one"""
        if len(chunks) == 1:
            return chunks[0]

        combined_content = '\n\n'.join(chunk.content for chunk in chunks)
        combined_metadata = {
            **chunks[0].metadata,
            'combined_from': [chunk.metadata.get('name', 'unnamed') for chunk in chunks],
            'type': 'combined'
        }

        return CodeChunk(
            content=combined_content,
            metadata=combined_metadata,
            start_line=chunks[0].start_line,
            end_line=chunks[-1].end_line,
            chunk_type='combined'
        )

    # ==================== C++/CUDA Chunking (Phase 6) ====================

    def _chunk_cpp_cuda(
        self,
        file_path: str,
        content: str,
        base_metadata: Dict[str, Any]
    ) -> List[CodeChunk]:
        """Chunk C++ or CUDA code using tree-sitter parsing"""
        chunks = []

        # Parse entities (functions, classes, methods, templates, kernels)
        entities = self.cpp_parser.parse_file(file_path, content)

        # Convert each entity to a chunk
        for entity in entities:
            chunk = self._entity_to_chunk(entity, content, file_path, base_metadata)
            if chunk:
                chunks.append(chunk)

        # Merge very small chunks
        chunks = self._merge_small_chunks(chunks)

        return chunks

    def _entity_to_chunk(
        self,
        entity: CppEntity,
        content: str,
        file_path: str,
        base_metadata: Dict[str, Any]
    ) -> Optional[CodeChunk]:
        """Convert a C++/CUDA entity to a CodeChunk"""

        # Get source code for this entity
        source = self.cpp_parser.get_entity_source(entity, content)

        if len(source) < self.min_chars:
            return None  # Too small

        # Build metadata
        metadata = {
            **base_metadata,
            'file': file_path,
            'type': entity.type,
            'name': entity.name,
            'signature': entity.signature,
            'line_start': entity.start_line,
            'line_end': entity.end_line,
            'language': 'cuda' if entity.is_cuda_kernel else 'cpp'
        }

        # Add CUDA-specific metadata
        if entity.is_cuda_kernel:
            metadata['is_cuda_kernel'] = True

        # Add template metadata
        if entity.template_params:
            metadata['template_params'] = entity.template_params

        # Add docstring if available
        if entity.docstring:
            metadata['docstring'] = entity.docstring

        # Truncate if too large
        if len(source) > self.max_chars:
            source = source[:self.max_chars] + "\n// [TRUNCATED]"

        return CodeChunk(
            content=source,
            metadata=metadata,
            start_line=entity.start_line,
            end_line=entity.end_line,
            chunk_type=entity.type
        )


