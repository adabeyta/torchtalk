#!/usr/bin/env python3
"""
C++ and CUDA parser using tree-sitter.

Extracts functions, classes, methods, and templates from C++/CUDA files.
CUDA (.cu) files are parsed as C++ since CUDA is a C++ extension.
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
from tree_sitter import Parser
import tree_sitter_languages


@dataclass
class CppEntity:
    """Represents a C++/CUDA code entity"""
    name: str
    type: str  # 'function', 'class', 'method', 'template', 'kernel'
    signature: str
    start_line: int
    end_line: int
    file_path: str
    is_cuda_kernel: bool = False
    template_params: Optional[str] = None
    docstring: Optional[str] = None


class CppCudaParser:
    """
    Parse C++ and CUDA files using tree-sitter.

    Extracts:
    - Functions (including CUDA kernels with __global__)
    - Classes and structs
    - Methods
    - Templates
    - Namespaces
    """

    def __init__(self):
        """Initialize C++ parser"""
        # tree-sitter-languages API
        self.parser = Parser()
        self.parser.set_language(tree_sitter_languages.get_language('cpp'))
        print(" C++/CUDA parser initialized")

    def parse_file(self, file_path: str, content: str) -> List[CppEntity]:
        """
        Parse a C++ or CUDA file.

        Args:
            file_path: Path to the file
            content: File content as string

        Returns:
            List of CppEntity objects
        """
        tree = self.parser.parse(bytes(content, 'utf8'))
        root_node = tree.root_node

        entities = []

        # Walk the tree and extract entities
        self._extract_entities(root_node, content, file_path, entities)

        return entities

    def _extract_entities(
        self,
        node,
        content: str,
        file_path: str,
        entities: List[CppEntity],
        current_class: Optional[str] = None
    ):
        """Recursively extract entities from AST"""

        # Function definitions
        if node.type == 'function_definition':
            entity = self._extract_function(node, content, file_path, current_class)
            if entity:
                entities.append(entity)

        # Class definitions
        elif node.type in ['class_specifier', 'struct_specifier']:
            class_entity = self._extract_class(node, content, file_path)
            if class_entity:
                entities.append(class_entity)
                # Recursively process class members
                class_name = class_entity.name
                for child in node.children:
                    self._extract_entities(child, content, file_path, entities, current_class=class_name)

        # Template declarations
        elif node.type == 'template_declaration':
            template_entity = self._extract_template(node, content, file_path)
            if template_entity:
                entities.append(template_entity)

        # Recurse into children
        else:
            for child in node.children:
                self._extract_entities(child, content, file_path, entities, current_class)

    def _extract_function(
        self,
        node,
        content: str,
        file_path: str,
        current_class: Optional[str]
    ) -> Optional[CppEntity]:
        """Extract function definition"""

        # Get function declarator
        declarator = None
        for child in node.children:
            if child.type == 'function_declarator':
                declarator = child
                break

        if not declarator:
            return None

        # Get function name
        name_node = None
        for child in declarator.children:
            if child.type in ['identifier', 'qualified_identifier', 'field_identifier']:
                name_node = child
                break

        if not name_node:
            return None

        name = self._get_node_text(name_node, content)

        # Check if CUDA kernel (__global__ or __device__)
        is_cuda_kernel = False
        # Check all children for CUDA attributes
        full_func_text = self._get_node_text(node, content)
        if '__global__' in full_func_text or '__device__' in full_func_text:
            is_cuda_kernel = True

        # Get full signature
        signature = self._get_node_text(declarator, content)

        # Determine entity type
        if current_class:
            entity_type = 'method'
            full_name = f"{current_class}::{name}"
        elif is_cuda_kernel:
            entity_type = 'kernel'
            full_name = name
        else:
            entity_type = 'function'
            full_name = name

        return CppEntity(
            name=full_name,
            type=entity_type,
            signature=signature,
            start_line=node.start_point[0] + 1,
            end_line=node.end_point[0] + 1,
            file_path=file_path,
            is_cuda_kernel=is_cuda_kernel
        )

    def _extract_class(self, node, content: str, file_path: str) -> Optional[CppEntity]:
        """Extract class/struct definition"""

        # Get class name
        name_node = None
        for child in node.children:
            if child.type == 'type_identifier':
                name_node = child
                break

        if not name_node:
            return None

        name = self._get_node_text(name_node, content)

        # Get class signature (with base classes if any)
        signature = f"{node.type.replace('_specifier', '')} {name}"

        # Check for base classes
        for child in node.children:
            if child.type == 'base_class_clause':
                base_text = self._get_node_text(child, content)
                signature += f" {base_text}"
                break

        return CppEntity(
            name=name,
            type='class',
            signature=signature,
            start_line=node.start_point[0] + 1,
            end_line=node.end_point[0] + 1,
            file_path=file_path
        )

    def _extract_template(self, node, content: str, file_path: str) -> Optional[CppEntity]:
        """Extract template declaration"""

        # Get template parameters
        template_params = None
        for child in node.children:
            if child.type == 'template_parameter_list':
                template_params = self._get_node_text(child, content)
                break

        # Get the templated entity (function or class)
        for child in node.children:
            if child.type == 'function_definition':
                entity = self._extract_function(child, content, file_path, None)
                if entity:
                    entity.type = 'template_function'
                    entity.template_params = template_params
                    return entity
            elif child.type in ['class_specifier', 'struct_specifier']:
                entity = self._extract_class(child, content, file_path)
                if entity:
                    entity.type = 'template_class'
                    entity.template_params = template_params
                    return entity

        return None

    def _get_node_text(self, node, content: str) -> str:
        """Get text content of a node"""
        return content[node.start_byte:node.end_byte]

    def get_entity_source(self, entity: CppEntity, content: str) -> str:
        """Get full source code for an entity"""
        lines = content.splitlines()
        return '\n'.join(lines[entity.start_line - 1:entity.end_line])


