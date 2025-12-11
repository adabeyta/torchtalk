#!/usr/bin/env python3
"""
Parser for PyTorch native_functions.yaml and derivatives.yaml.

These YAML files are the authoritative source for ATen operators:
- native_functions.yaml: Defines all operators and their dispatch mappings
- derivatives.yaml: Defines backward (gradient) formulas

This module provides the missing link between the YAML definitions and actual
C++ implementations in aten/src/ATen/native/.
"""

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import yaml

log = logging.getLogger(__name__)


@dataclass
class NativeFunction:
    """Represents a function from native_functions.yaml"""
    name: str                           # Function name (e.g., "matmul", "add.Tensor")
    func_signature: str                 # Full signature (e.g., "matmul(Tensor self, Tensor other) -> Tensor")
    variants: List[str]                 # ["function", "method"]
    dispatch: Dict[str, str]            # {dispatch_key: implementation_name}
    structured: bool = False            # Whether it uses structured kernels
    structured_delegate: Optional[str] = None
    python_module: Optional[str] = None  # Module if specified
    tags: List[str] = field(default_factory=list)
    yaml_line: int = 0                  # Line number in YAML file

    @property
    def base_name(self) -> str:
        """Get base name without overload suffix (e.g., 'add' from 'add.Tensor')"""
        return self.name.split('.')[0]

    @property
    def overload(self) -> Optional[str]:
        """Get overload suffix if present (e.g., 'Tensor' from 'add.Tensor')"""
        parts = self.name.split('.', 1)
        return parts[1] if len(parts) > 1 else None


@dataclass
class Derivative:
    """Represents a derivative formula from derivatives.yaml"""
    name: str                           # Function name
    func_signature: str                 # Full signature
    gradients: Dict[str, str]           # {input_name: gradient_formula}
    output_differentiability: Optional[List[bool]] = None
    yaml_line: int = 0


@dataclass
class NativeImplementation:
    """Represents an actual C++ implementation in native/ directory"""
    function_name: str                  # C++ function name
    file_path: str                      # File where implemented
    line_number: int                    # Line number of definition
    signature: Optional[str] = None     # Full C++ signature
    dispatch_key: Optional[str] = None  # Which dispatch key this serves


class NativeFunctionsParser:
    """
    Parser for native_functions.yaml - the source of truth for ATen operators.

    Maps operator names to:
    1. Their dispatch configurations (CPU, CUDA, etc.)
    2. The actual C++ function names that implement them
    3. Links to source files in aten/src/ATen/native/
    """

    def __init__(self, pytorch_root: str):
        self.pytorch_root = Path(pytorch_root)
        self.native_functions_path = self.pytorch_root / "aten/src/ATen/native/native_functions.yaml"
        self.derivatives_path = self.pytorch_root / "tools/autograd/derivatives.yaml"
        self.native_dir = self.pytorch_root / "aten/src/ATen/native"

        # Parsed data
        self.functions: Dict[str, NativeFunction] = {}
        self.derivatives: Dict[str, Derivative] = {}
        self.implementations: Dict[str, List[NativeImplementation]] = {}

        # Cache for file content searches
        self._file_cache: Dict[str, str] = {}

    def parse_all(self) -> Tuple[Dict[str, NativeFunction], Dict[str, Derivative]]:
        """Parse all YAML files and find implementations."""
        log.info("Parsing native_functions.yaml...")
        self._parse_native_functions()
        log.info(f"Parsed {len(self.functions)} native functions")

        log.info("Parsing derivatives.yaml...")
        self._parse_derivatives()
        log.info(f"Parsed {len(self.derivatives)} derivative formulas")

        log.info("Finding native implementations...")
        self._find_implementations()
        log.info(f"Found implementations for {len(self.implementations)} functions")

        return self.functions, self.derivatives

    def _parse_native_functions(self):
        """Parse native_functions.yaml"""
        if not self.native_functions_path.exists():
            log.warning(f"native_functions.yaml not found at {self.native_functions_path}")
            return

        content = self.native_functions_path.read_text()

        # Track line numbers by finding "- func:" patterns
        line_map = {}
        for i, line in enumerate(content.split('\n'), 1):
            if line.strip().startswith('- func:'):
                # Extract function name from signature
                match = re.search(r'- func:\s*(\w+)', line)
                if match:
                    line_map[match.group(1)] = i

        try:
            docs = list(yaml.safe_load_all(content))
            # native_functions.yaml is a list of function definitions
            for doc in docs:
                if isinstance(doc, list):
                    for entry in doc:
                        self._parse_function_entry(entry, line_map)
        except yaml.YAMLError as e:
            log.error(f"YAML parse error: {e}")
            # Fall back to regex parsing
            self._parse_native_functions_regex(content)

    def _parse_function_entry(self, entry: dict, line_map: dict):
        """Parse a single function entry from native_functions.yaml"""
        if not isinstance(entry, dict) or 'func' not in entry:
            return

        func_sig = entry['func']

        # Extract name from signature: "name(args...) -> return"
        match = re.match(r'(\w+(?:\.\w+)?)\s*\(', func_sig)
        if not match:
            return

        name = match.group(1)
        base_name = name.split('.')[0]

        # Parse dispatch mapping
        dispatch = {}
        if 'dispatch' in entry:
            dispatch_entry = entry['dispatch']
            if isinstance(dispatch_entry, dict):
                for key, impl in dispatch_entry.items():
                    # Handle comma-separated keys like "SparseCPU, SparseCUDA"
                    for k in key.split(','):
                        dispatch[k.strip()] = impl

        # Parse variants
        variants = []
        if 'variants' in entry:
            var_str = entry['variants']
            if isinstance(var_str, str):
                variants = [v.strip() for v in var_str.split(',')]
            elif isinstance(var_str, list):
                variants = var_str

        func = NativeFunction(
            name=name,
            func_signature=func_sig,
            variants=variants,
            dispatch=dispatch,
            structured=entry.get('structured', False),
            structured_delegate=entry.get('structured_delegate'),
            python_module=entry.get('python_module'),
            tags=entry.get('tags', []) if isinstance(entry.get('tags'), list) else [],
            yaml_line=line_map.get(base_name, 0)
        )

        self.functions[name] = func

        # Also index by base name for easy lookup
        if name != base_name and base_name not in self.functions:
            self.functions[base_name] = func

    def _parse_native_functions_regex(self, content: str):
        """Fallback regex parser for native_functions.yaml"""
        # Pattern to match function entries
        entry_pattern = r'^- func:\s*(.+?)(?=^- func:|\Z)'

        for match in re.finditer(entry_pattern, content, re.MULTILINE | re.DOTALL):
            entry_text = match.group(0)
            line_num = content[:match.start()].count('\n') + 1

            # Extract function signature
            sig_match = re.search(r'- func:\s*(.+?)$', entry_text, re.MULTILINE)
            if not sig_match:
                continue

            func_sig = sig_match.group(1).strip()
            name_match = re.match(r'(\w+(?:\.\w+)?)', func_sig)
            if not name_match:
                continue

            name = name_match.group(1)

            # Extract dispatch
            dispatch = {}
            dispatch_match = re.search(r'dispatch:\s*\n((?:\s+.+\n)+)', entry_text)
            if dispatch_match:
                for line in dispatch_match.group(1).split('\n'):
                    kv_match = re.match(r'\s+(\S+):\s*(\S+)', line)
                    if kv_match:
                        dispatch[kv_match.group(1)] = kv_match.group(2)

            # Extract variants
            variants = []
            var_match = re.search(r'variants:\s*(.+)$', entry_text, re.MULTILINE)
            if var_match:
                variants = [v.strip() for v in var_match.group(1).split(',')]

            func = NativeFunction(
                name=name,
                func_signature=func_sig,
                variants=variants,
                dispatch=dispatch,
                yaml_line=line_num
            )
            self.functions[name] = func

    def _parse_derivatives(self):
        """Parse derivatives.yaml"""
        if not self.derivatives_path.exists():
            log.warning(f"derivatives.yaml not found at {self.derivatives_path}")
            return

        content = self.derivatives_path.read_text()

        try:
            data = yaml.safe_load(content)
            if not isinstance(data, list):
                return

            for i, entry in enumerate(data):
                if not isinstance(entry, dict) or 'name' not in entry:
                    continue

                name = entry['name']

                # Extract gradient formulas (keys that aren't special fields)
                special_keys = {'name', 'dispatch', 'output_differentiability'}
                gradients = {}
                for key, value in entry.items():
                    if key not in special_keys and isinstance(value, str):
                        gradients[key] = value

                # Extract signature from name field
                func_sig = name
                name_only = re.match(r'(\w+(?:\.\w+)?)', name)
                if name_only:
                    name = name_only.group(1)

                deriv = Derivative(
                    name=name,
                    func_signature=func_sig,
                    gradients=gradients,
                    output_differentiability=entry.get('output_differentiability'),
                    yaml_line=i
                )
                self.derivatives[name] = deriv

        except yaml.YAMLError as e:
            log.error(f"YAML parse error in derivatives.yaml: {e}")

    def _find_implementations(self):
        """Find actual C++ implementations for functions in aten/src/ATen/native/"""
        if not self.native_dir.exists():
            log.warning(f"Native directory not found: {self.native_dir}")
            return

        # Build a map of function names we're looking for
        target_funcs = set()
        for func in self.functions.values():
            target_funcs.add(func.base_name)
            # Also add dispatch implementation names
            for impl_name in func.dispatch.values():
                target_funcs.add(impl_name)

        # Scan native directory for C++ files
        cpp_files = list(self.native_dir.rglob('*.cpp'))
        cpp_files.extend(self.native_dir.rglob('*.h'))

        log.info(f"Scanning {len(cpp_files)} files in native/ for implementations...")

        for cpp_file in cpp_files:
            # Skip generated files
            if 'generated' in str(cpp_file).lower():
                continue

            try:
                content = cpp_file.read_text(encoding='utf-8', errors='replace')
                self._find_implementations_in_file(str(cpp_file), content, target_funcs)
            except Exception as e:
                log.debug(f"Error reading {cpp_file}: {e}")

    def _find_implementations_in_file(self, file_path: str, content: str, target_funcs: Set[str]):
        """Find function implementations in a single C++ file"""
        # Pattern to match C++ function definitions
        # Handles: Tensor func_name(...), void func_name(...), etc.
        # Also handles namespace prefixes like at::native::
        func_pattern = r'^(?:static\s+)?(?:inline\s+)?(?:TORCH_API\s+)?(?:C10_EXPORT\s+)?(?:[\w:]+\s+)+?(\w+)\s*\([^)]*\)\s*(?:const\s*)?\{'

        lines = content.split('\n')

        for i, line in enumerate(lines, 1):
            # Quick check if line might contain a function definition
            if not ('{' in line or (i < len(lines) and '{' in lines[i])):
                continue

            match = re.match(func_pattern, line)
            if match:
                func_name = match.group(1)

                if func_name in target_funcs:
                    # Extract full signature (may span multiple lines)
                    sig_end = line.find('{')
                    signature = line[:sig_end].strip() if sig_end > 0 else line.strip()

                    impl = NativeImplementation(
                        function_name=func_name,
                        file_path=file_path,
                        line_number=i,
                        signature=signature
                    )

                    if func_name not in self.implementations:
                        self.implementations[func_name] = []
                    self.implementations[func_name].append(impl)

    def get_function_info(self, name: str) -> Optional[Dict]:
        """
        Get comprehensive information about a function.

        Returns dict with:
        - native_function: The NativeFunction from YAML
        - derivative: The Derivative formula if exists
        - implementations: List of actual C++ implementations
        """
        # Try exact match first, then base name
        func = self.functions.get(name)
        if not func:
            # Try as base name
            for key, f in self.functions.items():
                if f.base_name == name:
                    func = f
                    break

        if not func:
            return None

        result = {
            'native_function': func,
            'derivative': self.derivatives.get(func.base_name),
            'implementations': [],
            'dispatch_implementations': {}
        }

        # Find implementations for base function
        base_impls = self.implementations.get(func.base_name, [])
        result['implementations'].extend(base_impls)

        # Find implementations for each dispatch key
        for dispatch_key, impl_name in func.dispatch.items():
            dispatch_impls = self.implementations.get(impl_name, [])
            if dispatch_impls:
                result['dispatch_implementations'][dispatch_key] = dispatch_impls

            # Also add to main implementations list
            result['implementations'].extend(dispatch_impls)

        return result

    def to_dict(self) -> Dict:
        """Convert all parsed data to a dictionary for serialization"""
        return {
            'functions': {
                name: {
                    'name': f.name,
                    'signature': f.func_signature,
                    'variants': f.variants,
                    'dispatch': f.dispatch,
                    'structured': f.structured,
                    'yaml_line': f.yaml_line,
                    'tags': f.tags,
                }
                for name, f in self.functions.items()
            },
            'derivatives': {
                name: {
                    'name': d.name,
                    'signature': d.func_signature,
                    'gradients': d.gradients,
                }
                for name, d in self.derivatives.items()
            },
            'implementations': {
                name: [
                    {
                        'function_name': impl.function_name,
                        'file_path': impl.file_path,
                        'line_number': impl.line_number,
                        'signature': impl.signature,
                    }
                    for impl in impls
                ]
                for name, impls in self.implementations.items()
            }
        }


def find_native_implementation(pytorch_root: str, function_name: str) -> List[NativeImplementation]:
    """
    Quick search for a function's native implementation.

    Searches aten/src/ATen/native/ for C++ files containing the function definition.
    Returns list of implementations found.
    """
    native_dir = Path(pytorch_root) / "aten/src/ATen/native"
    if not native_dir.exists():
        return []

    implementations = []

    # Pattern to find function definition
    func_pattern = rf'(?:Tensor|void|bool|int64_t|double|std::[\w<>]+)\s+{re.escape(function_name)}\s*\('

    for cpp_file in native_dir.rglob('*.cpp'):
        # Skip generated files
        if 'generated' in str(cpp_file).lower():
            continue

        try:
            content = cpp_file.read_text(encoding='utf-8', errors='replace')

            for match in re.finditer(func_pattern, content):
                line_num = content[:match.start()].count('\n') + 1

                # Get the full line for signature
                line_start = content.rfind('\n', 0, match.start()) + 1
                line_end = content.find('\n', match.end())
                signature = content[line_start:line_end].strip()

                impl = NativeImplementation(
                    function_name=function_name,
                    file_path=str(cpp_file),
                    line_number=line_num,
                    signature=signature
                )
                implementations.append(impl)

        except Exception:
            continue

    return implementations
