#!/usr/bin/env python3
"""
TorchTalk MCP Server - Cross-language binding information for Claude Code.

This MCP server provides STRUCTURAL knowledge about PyTorch-style codebases:
- Python → C++ → CUDA binding chains
- Dispatch keys (CPU, CUDA, etc.) for backend routing
- CUDA kernel mappings
- Function call relationships
- Native function definitions from native_functions.yaml
- Derivative formulas from derivatives.yaml

One-command setup:
    claude mcp add torchtalk -s user -- torchtalk mcp-serve --pytorch-source /path/to/pytorch

The server automatically builds and caches the index on first run.
"""

import argparse
import hashlib
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

from mcp.server.fastmcp import FastMCP

log = logging.getLogger(__name__)

# Global server instance
mcp = FastMCP("torchtalk")

# Global state for binding data (initialized on startup)
_bindings: List[Dict] = []
_cuda_kernels: List[Dict] = []
_by_python_name: Dict[str, List[Dict]] = {}
_by_cpp_name: Dict[str, List[Dict]] = {}
_by_file: Dict[str, List[Dict]] = {}
_by_dispatch_key: Dict[str, List[Dict]] = {}
_by_binding_type: Dict[str, List[Dict]] = {}
_kernels_by_name: Dict[str, Dict] = {}
_call_graph: Dict[str, List[str]] = {}
_import_graph: Dict[str, List[str]] = {}
_index_path: Optional[str] = None
_pytorch_source: Optional[str] = None

# Native functions data (from YAML files - authoritative source)
_native_functions: Dict[str, Dict] = {}
_derivatives: Dict[str, Dict] = {}
_native_implementations: Dict[str, List[Dict]] = {}

# Cache directory
CACHE_DIR = Path.home() / ".cache" / "torchtalk"


def _get_cache_path(source_path: str) -> Path:
    """Get cache file path for a given source directory."""
    # Create a hash of the source path for unique cache file
    path_hash = hashlib.md5(str(Path(source_path).resolve()).encode()).hexdigest()[:12]
    return CACHE_DIR / f"bindings_{path_hash}.json"


def _get_source_fingerprint(source_path: str) -> str:
    """Get a fingerprint of the source to detect changes."""
    source = Path(source_path)

    # Check a few key files that would change between versions
    key_files = [
        source / "version.txt",
        source / "torch/version.py",
        source / "CMakeLists.txt",
        source / ".git/HEAD",
    ]

    fingerprint_parts = []
    for f in key_files:
        if f.exists():
            stat = f.stat()
            fingerprint_parts.append(f"{f.name}:{stat.st_mtime}:{stat.st_size}")

    return hashlib.md5("|".join(fingerprint_parts).encode()).hexdigest()[:16]


def _is_cache_valid(cache_path: Path, source_path: str) -> bool:
    """Check if cached bindings are still valid."""
    if not cache_path.exists():
        return False

    try:
        with open(cache_path, 'r') as f:
            data = json.load(f)

        cached_fingerprint = data.get("metadata", {}).get("source_fingerprint", "")
        current_fingerprint = _get_source_fingerprint(source_path)

        return cached_fingerprint == current_fingerprint
    except Exception:
        return False


def _parse_native_functions_yaml(source_path: str) -> Tuple[Dict, Dict]:
    """
    Parse native_functions.yaml - the authoritative source for ATen operators.

    Returns:
        Tuple of (functions_dict, derivatives_dict)
    """
    import yaml

    source = Path(source_path)
    native_yaml = source / "aten/src/ATen/native/native_functions.yaml"
    derivatives_yaml = source / "tools/autograd/derivatives.yaml"

    functions = {}
    derivatives = {}

    # Parse native_functions.yaml
    if native_yaml.exists():
        log.info(f"Parsing {native_yaml}...")
        try:
            content = native_yaml.read_text()
            data = yaml.safe_load(content)

            if isinstance(data, list):
                for entry in data:
                    if not isinstance(entry, dict) or 'func' not in entry:
                        continue

                    func_sig = entry['func']
                    # Extract name from signature: "name(args...) -> return"
                    match = re.match(r'(\w+(?:\.\w+)?)\s*\(', func_sig)
                    if not match:
                        continue

                    name = match.group(1)
                    base_name = name.split('.')[0]

                    # Parse dispatch mapping
                    dispatch = {}
                    if 'dispatch' in entry:
                        dispatch_entry = entry['dispatch']
                        if isinstance(dispatch_entry, dict):
                            for key, impl in dispatch_entry.items():
                                # Handle comma-separated keys
                                for k in key.split(','):
                                    dispatch[k.strip()] = impl

                    func_data = {
                        'name': name,
                        'base_name': base_name,
                        'signature': func_sig,
                        'dispatch': dispatch,
                        'variants': entry.get('variants', ''),
                        'structured': entry.get('structured', False),
                        'structured_delegate': entry.get('structured_delegate'),
                        'tags': entry.get('tags', []),
                    }

                    functions[name] = func_data
                    # Also index by base name
                    if name != base_name:
                        if base_name not in functions:
                            functions[base_name] = func_data

                log.info(f"Parsed {len(functions)} native functions")

        except Exception as e:
            log.warning(f"Failed to parse native_functions.yaml: {e}")

    # Parse derivatives.yaml
    if derivatives_yaml.exists():
        log.info(f"Parsing {derivatives_yaml}...")
        try:
            content = derivatives_yaml.read_text()
            data = yaml.safe_load(content)

            if isinstance(data, list):
                for entry in data:
                    if not isinstance(entry, dict) or 'name' not in entry:
                        continue

                    name_sig = entry['name']
                    # Extract name from signature
                    match = re.match(r'(\w+(?:\.\w+)?)', name_sig)
                    if not match:
                        continue

                    name = match.group(1)

                    # Extract gradient formulas
                    special_keys = {'name', 'dispatch', 'output_differentiability'}
                    gradients = {}
                    for key, value in entry.items():
                        if key not in special_keys and isinstance(value, str):
                            gradients[key] = value

                    derivatives[name] = {
                        'name': name,
                        'signature': name_sig,
                        'gradients': gradients,
                    }

                log.info(f"Parsed {len(derivatives)} derivative formulas")

        except Exception as e:
            log.warning(f"Failed to parse derivatives.yaml: {e}")

    return functions, derivatives


def _find_native_implementations(source_path: str, functions: Dict) -> Dict[str, List[Dict]]:
    """
    Find actual C++ implementations in aten/src/ATen/native/.

    Uses an optimized approach: scan each file once and match against all target functions.
    """
    source = Path(source_path)
    native_dir = source / "aten/src/ATen/native"

    if not native_dir.exists():
        log.warning(f"Native directory not found: {native_dir}")
        return {}

    implementations: Dict[str, List[Dict]] = {}

    # Build set of function names to search for
    target_funcs = set()
    for func in functions.values():
        target_funcs.add(func['base_name'])
        # Add dispatch implementation names
        for impl_name in func.get('dispatch', {}).values():
            target_funcs.add(impl_name)

    log.info(f"Searching for {len(target_funcs)} function implementations...")

    # Scan native directory
    cpp_files = list(native_dir.rglob('*.cpp'))

    # Filter out generated files
    cpp_files = [f for f in cpp_files if 'generated' not in str(f).lower()]

    log.info(f"Scanning {len(cpp_files)} C++ files...")

    # Optimized approach: scan each file once, extract all function definitions
    # Pattern to match C++ function definitions - capture the function name
    func_def_pattern = re.compile(
        r'^(?:static\s+)?(?:inline\s+)?(?:TORCH_API\s+)?'
        r'(?:Tensor|TensorList|void|bool|int64_t|double|float|Scalar|c10::[\w<>:]+|at::[\w<>:]+|std::[\w<>:]+)\s+'
        r'(\w+)\s*\([^)]*\)\s*(?:const\s*)?\{',
        re.MULTILINE
    )

    for cpp_file in cpp_files:
        try:
            content = cpp_file.read_text(encoding='utf-8', errors='replace')

            # Find all function definitions in file
            for match in func_def_pattern.finditer(content):
                func_name = match.group(1)

                # Check if this is a function we're looking for
                if func_name in target_funcs:
                    line_num = content[:match.start()].count('\n') + 1

                    # Get signature (the matched text minus the opening brace)
                    sig_text = match.group(0).rstrip('{').strip()

                    impl = {
                        'function_name': func_name,
                        'file_path': str(cpp_file),
                        'line_number': line_num,
                        'signature': sig_text,
                    }

                    if func_name not in implementations:
                        implementations[func_name] = []
                    implementations[func_name].append(impl)

        except Exception as e:
            log.debug(f"Error reading {cpp_file}: {e}")

    log.info(f"Found implementations for {len(implementations)} functions")
    return implementations


def _build_and_cache_index(source_path: str) -> Path:
    """Build bindings from source and cache them."""
    from torchtalk.analysis.binding_detector import BindingDetector
    from torchtalk.analysis.repo_analyzer import RepoAnalyzer

    cache_path = _get_cache_path(source_path)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    log.info(f"Building binding index from {source_path}...")
    log.info("This may take a few minutes on first run...")

    # Detect bindings
    detector = BindingDetector()
    binding_graph = detector.detect_bindings_in_directory(source_path)

    # Convert to serializable format
    bindings = []
    for b in binding_graph.bindings:
        bindings.append({
            "python_name": b.python_name,
            "cpp_name": b.cpp_name,
            "type": b.binding_type,
            "file_path": b.file_path,
            "line": b.line_number,
            "dispatch_key": b.dispatch_key,
            "namespace": b.namespace,
            "signature": b.signature,
            "cuda_kernel": b.cuda_kernel,
        })

    cuda_kernels = []
    for k in binding_graph.cuda_kernels:
        cuda_kernels.append({
            "name": k.name,
            "file_path": k.file_path,
            "line": k.line_number,
            "parameters": k.parameters,
            "template_params": k.template_params,
            "called_by": k.called_by,
        })

    # Analyze repo for call/import graphs
    log.info("Analyzing repository structure...")
    try:
        analyzer = RepoAnalyzer(source_path)
        analyzer.analyze_repository()

        call_graph = {}
        for node in analyzer.call_graph.nodes():
            successors = list(analyzer.call_graph.successors(node))
            if successors:
                call_graph[str(node)] = [str(s) for s in successors]

        import_graph = {}
        for node in analyzer.import_graph.nodes():
            successors = list(analyzer.import_graph.successors(node))
            if successors:
                import_graph[str(node)] = [str(s) for s in successors]
    except Exception as e:
        log.warning(f"Graph analysis failed: {e}")
        call_graph = {}
        import_graph = {}

    # Parse native_functions.yaml and derivatives.yaml
    log.info("Parsing native_functions.yaml and derivatives.yaml...")
    native_funcs, deriv_funcs = _parse_native_functions_yaml(source_path)

    # Find actual implementations in native/ directory
    log.info("Finding native implementations...")
    native_impls = _find_native_implementations(source_path, native_funcs)

    # Save to cache
    data = {
        "bindings": bindings,
        "cuda_kernels": cuda_kernels,
        "call_graph": call_graph,
        "import_graph": import_graph,
        "native_functions": native_funcs,
        "derivatives": deriv_funcs,
        "native_implementations": native_impls,
        "metadata": {
            "source_path": str(Path(source_path).resolve()),
            "source_fingerprint": _get_source_fingerprint(source_path),
            "binding_count": len(bindings),
            "kernel_count": len(cuda_kernels),
            "native_function_count": len(native_funcs),
            "derivative_count": len(deriv_funcs),
            "native_impl_count": len(native_impls),
        }
    }

    with open(cache_path, 'w') as f:
        json.dump(data, f)

    file_size = cache_path.stat().st_size / (1024 * 1024)
    log.info(f"Cached {len(bindings)} bindings, {len(cuda_kernels)} kernels ({file_size:.1f} MB)")
    log.info(f"Cache location: {cache_path}")

    return cache_path


def _init_from_bindings_json(bindings_file: str):
    """Load from lightweight bindings.json (fast, ~10MB)."""
    global _bindings, _cuda_kernels, _by_python_name, _by_cpp_name, _by_file
    global _by_dispatch_key, _by_binding_type, _kernels_by_name
    global _call_graph, _import_graph
    global _native_functions, _derivatives, _native_implementations

    log.info(f"Loading bindings from {bindings_file}...")

    with open(bindings_file, 'r') as f:
        data = json.load(f)

    _bindings = data.get("bindings", [])
    _cuda_kernels = data.get("cuda_kernels", [])
    _call_graph = data.get("call_graph", {})
    _import_graph = data.get("import_graph", {})
    _native_functions = data.get("native_functions", {})
    _derivatives = data.get("derivatives", {})
    _native_implementations = data.get("native_implementations", {})

    # Build indexes
    for binding in _bindings:
        py_name = binding.get("python_name", "")
        cpp_name = binding.get("cpp_name", "")
        file_path = binding.get("file_path", "")
        dispatch_key = binding.get("dispatch_key")
        binding_type = binding.get("type")

        if py_name:
            _by_python_name.setdefault(py_name, []).append(binding)
        if cpp_name:
            _by_cpp_name.setdefault(cpp_name, []).append(binding)
        if file_path:
            _by_file.setdefault(file_path, []).append(binding)
        if dispatch_key:
            _by_dispatch_key.setdefault(dispatch_key, []).append(binding)
        if binding_type:
            _by_binding_type.setdefault(binding_type, []).append(binding)

    for kernel in _cuda_kernels:
        _kernels_by_name[kernel.get("name", "")] = kernel

    log.info(f"Loaded {len(_bindings)} bindings, {len(_cuda_kernels)} CUDA kernels")


def _init_from_source(source_path: str):
    """Initialize from PyTorch source - builds/caches automatically."""
    global _index_path, _pytorch_source

    source = Path(source_path).resolve()
    if not source.exists():
        raise FileNotFoundError(f"Source path not found: {source_path}")

    # Check for cached index
    cache_path = _get_cache_path(str(source))

    if _is_cache_valid(cache_path, str(source)):
        log.info(f"Using cached index from {cache_path}")
        _init_from_bindings_json(str(cache_path))
    else:
        # Build and cache
        cache_path = _build_and_cache_index(str(source))
        _init_from_bindings_json(str(cache_path))

    _index_path = str(source)
    _pytorch_source = str(source)


def _init_from_index(index_path: str):
    """Load binding and graph data from existing index/bindings.json."""
    global _index_path

    index_path_obj = Path(index_path)

    # Check for bindings.json
    if index_path_obj.is_file() and index_path_obj.name == "bindings.json":
        _init_from_bindings_json(str(index_path_obj))
        _index_path = str(index_path_obj.parent)
        return

    bindings_file = index_path_obj / "bindings.json"
    if bindings_file.exists():
        _init_from_bindings_json(str(bindings_file))
        _index_path = index_path
        return

    raise FileNotFoundError(f"No bindings.json found at {index_path}")


def _auto_detect_pytorch() -> Optional[str]:
    """Try to auto-detect PyTorch source location."""
    candidates = [
        os.environ.get("PYTORCH_SOURCE"),
        os.environ.get("PYTORCH_PATH"),
        Path.cwd() / "pytorch",
        Path.cwd().parent / "pytorch",
        Path("/myworkspace/pytorch"),  # Common dev setup
    ]

    for candidate in candidates:
        if candidate:
            path = Path(candidate)
            # Check if it looks like PyTorch source
            if (path / "torch").exists() and (path / "aten").exists():
                return str(path.resolve())

    return None


def _ensure_loaded():
    """
    Auto-load binding data if not already loaded.
    Tries: env vars, cwd, auto-detect PyTorch source.
    """
    if _bindings:
        return True

    # Check TORCHTALK_INDEX env var (explicit index path)
    index_path = os.environ.get("TORCHTALK_INDEX")
    if index_path:
        path = Path(index_path)
        if (path / "bindings.json").exists():
            _init_from_index(index_path)
            return True
        elif path.is_file() and path.name == "bindings.json":
            _init_from_bindings_json(str(path))
            return True

    # Check current directory for existing index
    cwd = Path.cwd()
    for subdir in ["pytorch_index", "index", "."]:
        candidate = cwd / subdir / "bindings.json"
        if candidate.exists():
            _init_from_index(str(candidate.parent))
            return True

    # Try to auto-detect and build from PyTorch source
    pytorch_source = _auto_detect_pytorch()
    if pytorch_source:
        log.info(f"Auto-detected PyTorch source at {pytorch_source}")
        _init_from_source(pytorch_source)
        return True

    return False


# ============================================================================
# Helper Functions
# ============================================================================

def _is_generated_file(file_path: str) -> bool:
    """Check if a file is auto-generated (less useful for developers)."""
    path_lower = file_path.lower()
    generated_markers = [
        '/build/',
        '/generated/',
        'register',  # RegisterCompositeImplicitAutograd_0.cpp etc.
        'variabletype',  # VariableType_0.cpp etc.
        'variabletypemanual',
    ]
    return any(marker in path_lower for marker in generated_markers)


def _rank_bindings(bindings: List[Dict]) -> List[Dict]:
    """
    Rank bindings by usefulness - prefer hand-written native code over generated wrappers.
    """
    def score(b: Dict) -> int:
        file_path = b.get("file_path", "")
        s = 0

        # Strongly prefer native implementations
        if "aten/src/ATen/native" in file_path and not _is_generated_file(file_path):
            s += 100

        # Prefer CUDA kernels
        if file_path.endswith(('.cu', '.cuh')):
            s += 50

        # Penalize generated files
        if _is_generated_file(file_path):
            s -= 50

        # Bonus for having line numbers
        if b.get("line") or b.get("line_number"):
            s += 10

        return s

    return sorted(bindings, key=score, reverse=True)


def _get_native_function_info(function_name: str) -> Optional[Dict]:
    """
    Get comprehensive info for a function from native_functions.yaml.
    This is the authoritative source.
    """
    if not _native_functions:
        return None

    # Try exact match first
    func = _native_functions.get(function_name)
    if func:
        return func

    # Try base name match
    for key, f in _native_functions.items():
        if f.get('base_name') == function_name:
            return f

    # Try partial match
    function_lower = function_name.lower()
    for key, f in _native_functions.items():
        if function_lower in key.lower() or function_lower in f.get('base_name', '').lower():
            return f

    return None


def _make_relative_path(file_path: str) -> str:
    """Convert absolute path to relative path from PyTorch root for cleaner output."""
    if _pytorch_source and file_path.startswith(_pytorch_source):
        return file_path[len(_pytorch_source):].lstrip('/')
    return file_path


# ============================================================================
# MCP Tools
# ============================================================================

@mcp.tool()
async def get_binding_chain(function_name: str) -> str:
    """
    Get the cross-language implementation chain for a PyTorch function.

    Shows the complete path from Python API to native C++ implementation:
    1. Native function definition (from native_functions.yaml)
    2. Dispatch configuration (CPU, CUDA, etc.)
    3. Actual implementation files with line numbers
    4. Derivative formula for backward pass (if available)

    Args:
        function_name: Function to trace (e.g., "matmul", "add", "conv2d")

    Returns:
        Complete binding chain with native implementations, dispatch info, and file locations.

    Example:
        get_binding_chain("matmul") returns the native function definition,
        dispatch mapping, and the actual implementation in LinearAlgebra.cpp
    """
    _ensure_loaded()
    if not _bindings and not _native_functions:
        return "Error: No binding data loaded. Provide --pytorch-source or set PYTORCH_SOURCE env var."

    output = [f"## Binding chain for `{function_name}`\n"]

    # ========== 1. Native Function Definition (authoritative source) ==========
    native_func = _get_native_function_info(function_name)

    if native_func:
        output.append("### Native Function Definition (from native_functions.yaml)\n")
        output.append(f"**Name:** `{native_func.get('name', function_name)}`")
        output.append(f"**Signature:** `{native_func.get('signature', 'N/A')}`")

        if native_func.get('variants'):
            output.append(f"**Variants:** {native_func['variants']}")

        if native_func.get('tags'):
            tags = native_func['tags']
            if isinstance(tags, list):
                output.append(f"**Tags:** {', '.join(str(t) for t in tags)}")

        # Dispatch configuration
        dispatch = native_func.get('dispatch', {})
        if dispatch:
            output.append("\n**Dispatch Configuration:**")
            for key, impl in sorted(dispatch.items()):
                output.append(f"  - {key}: `{impl}`")
        output.append("")

    # ========== 2. Native Implementations (actual code locations) ==========
    # First check our implementation cache
    impl_found = False
    base_name = native_func.get('base_name', function_name) if native_func else function_name

    native_impls = []

    # Get implementations for base function name
    if base_name in _native_implementations:
        native_impls.extend(_native_implementations[base_name])

    # Also get implementations for dispatch target names
    if native_func:
        for impl_name in native_func.get('dispatch', {}).values():
            if impl_name in _native_implementations:
                native_impls.extend(_native_implementations[impl_name])

    if native_impls:
        impl_found = True
        output.append("### Native C++ Implementations\n")
        output.append("*Hand-written implementations in aten/src/ATen/native/:*\n")

        seen = set()
        for impl in native_impls:
            key = (impl['file_path'], impl['line_number'])
            if key in seen:
                continue
            seen.add(key)

            rel_path = _make_relative_path(impl['file_path'])
            output.append(f"**`{impl['function_name']}`**")
            output.append(f"  → File: `{rel_path}:{impl['line_number']}`")
            if impl.get('signature'):
                sig = impl['signature'][:80] + '...' if len(impl.get('signature', '')) > 80 else impl.get('signature', '')
                output.append(f"  → Signature: `{sig}`")
            output.append("")

    # ========== 3. Registered Bindings (from TORCH_LIBRARY_IMPL) ==========
    # Find matching bindings from the binding detector
    matches = []
    for binding in _bindings:
        py_name = binding.get("python_name", "").lower()
        cpp_name = binding.get("cpp_name", "").lower()

        if function_name.lower() in py_name or function_name.lower() in cpp_name:
            matches.append(binding)

    if matches:
        # Rank and filter
        ranked = _rank_bindings(matches)

        # Separate native from generated
        native_bindings = [b for b in ranked if not _is_generated_file(b.get('file_path', ''))]
        generated_bindings = [b for b in ranked if _is_generated_file(b.get('file_path', ''))]

        if native_bindings:
            output.append("### Registered Implementations (TORCH_LIBRARY_IMPL)\n")

            # Group by dispatch key
            by_dispatch: Dict[str, List[Dict]] = {}
            for b in native_bindings:
                key = b.get("dispatch_key") or "default"
                by_dispatch.setdefault(key, []).append(b)

            for dispatch_key in ["CPU", "CUDA", "Meta", "default"]:
                if dispatch_key in by_dispatch:
                    output.append(f"**{dispatch_key}:**")
                    for b in by_dispatch[dispatch_key][:3]:
                        rel_path = _make_relative_path(b['file_path'])
                        line_info = f":{b['line']}" if b.get('line') else ""
                        output.append(f"  - `{b['cpp_name']}` → `{rel_path}{line_info}`")
                    output.append("")

            # Other dispatch keys
            for dispatch_key, binds in by_dispatch.items():
                if dispatch_key not in ["CPU", "CUDA", "Meta", "default"]:
                    output.append(f"**{dispatch_key}:**")
                    for b in binds[:2]:
                        rel_path = _make_relative_path(b['file_path'])
                        line_info = f":{b['line']}" if b.get('line') else ""
                        output.append(f"  - `{b['cpp_name']}` → `{rel_path}{line_info}`")
                    output.append("")

        # Only mention generated files briefly
        if generated_bindings and not native_bindings:
            output.append("### Generated Wrappers\n")
            output.append("*Auto-generated dispatch wrappers (less useful for understanding implementation):*\n")
            for b in generated_bindings[:3]:
                file_name = b['file_path'].split('/')[-1]
                output.append(f"  - `{b['cpp_name']}` → `{file_name}`")
            output.append("")

    # ========== 4. CUDA Kernels ==========
    related_kernels = [k for k in _cuda_kernels if function_name.lower() in k.get("name", "").lower()]
    if related_kernels:
        output.append("### CUDA Kernels\n")
        for k in related_kernels[:5]:
            rel_path = _make_relative_path(k['file_path'])
            line_info = f":{k['line']}" if k.get('line') else ""
            output.append(f"**`{k['name']}`**")
            output.append(f"  → File: `{rel_path}{line_info}`")
            if k.get("called_by"):
                output.append(f"  → Called by: {', '.join(k['called_by'][:3])}")
            output.append("")

    # ========== 5. Derivative Formula ==========
    deriv = _derivatives.get(base_name) or _derivatives.get(function_name)
    if deriv:
        output.append("### Backward Pass (from derivatives.yaml)\n")
        gradients = deriv.get('gradients', {})
        if gradients:
            for input_name, formula in gradients.items():
                formula_short = formula[:100] + '...' if len(formula) > 100 else formula
                output.append(f"  - `{input_name}`: `{formula_short}`")
        output.append("")

    # ========== Summary ==========
    if not native_func and not matches and not native_impls:
        return f"No binding information found for '{function_name}'. Try a broader search term or check spelling."

    output.append("---")
    output.append("**Next step:** Use Claude's Read tool to examine the implementation files listed above.")

    return "\n".join(output)


@mcp.tool()
async def get_native_function(function_name: str) -> str:
    """
    Get the native function definition from native_functions.yaml.

    This is the AUTHORITATIVE source for ATen operator definitions.
    Shows the official signature, dispatch configuration, and implementation function names.

    Args:
        function_name: Function name (e.g., "matmul", "add", "conv2d")

    Returns:
        Complete native function definition including dispatch mapping
    """
    _ensure_loaded()
    if not _native_functions:
        return "Error: No native function data loaded. Ensure PyTorch source is available."

    native_func = _get_native_function_info(function_name)

    if not native_func:
        # Try to find similar functions
        similar = []
        fn_lower = function_name.lower()
        for key in _native_functions.keys():
            if fn_lower in key.lower():
                similar.append(key)

        if similar:
            return f"Function '{function_name}' not found. Similar functions:\n" + "\n".join(f"  - {s}" for s in similar[:10])
        return f"Function '{function_name}' not found in native_functions.yaml."

    output = [f"## Native Function: `{native_func['name']}`\n"]
    output.append(f"**Signature:** `{native_func['signature']}`\n")

    if native_func.get('variants'):
        output.append(f"**Variants:** {native_func['variants']}")

    if native_func.get('structured'):
        output.append(f"**Structured:** Yes")
        if native_func.get('structured_delegate'):
            output.append(f"**Structured Delegate:** `{native_func['structured_delegate']}`")

    if native_func.get('tags'):
        tags = native_func['tags']
        if isinstance(tags, list):
            output.append(f"**Tags:** {', '.join(str(t) for t in tags)}")

    # Dispatch configuration - the most important part
    dispatch = native_func.get('dispatch', {})
    if dispatch:
        output.append("\n### Dispatch Configuration\n")
        output.append("*Maps dispatch keys to C++ implementation functions:*\n")
        output.append("| Dispatch Key | Implementation Function |")
        output.append("|-------------|------------------------|")
        for key, impl in sorted(dispatch.items()):
            output.append(f"| {key} | `{impl}` |")

        output.append("\n*Use `get_binding_chain` to find the actual file locations.*")
    else:
        output.append("\n*No explicit dispatch - uses default implementation.*")

    # Show implementation locations if we have them
    base_name = native_func.get('base_name', function_name)
    impls = []
    if base_name in _native_implementations:
        impls.extend(_native_implementations[base_name])
    for impl_name in dispatch.values():
        if impl_name in _native_implementations:
            impls.extend(_native_implementations[impl_name])

    if impls:
        output.append("\n### Implementation Locations\n")
        seen = set()
        for impl in impls:
            key = (impl['file_path'], impl['line_number'])
            if key in seen:
                continue
            seen.add(key)
            rel_path = _make_relative_path(impl['file_path'])
            output.append(f"- `{impl['function_name']}` → `{rel_path}:{impl['line_number']}`")

    # Derivative info
    deriv = _derivatives.get(base_name) or _derivatives.get(function_name)
    if deriv:
        output.append("\n### Backward Pass\n")
        gradients = deriv.get('gradients', {})
        if gradients:
            for input_name, formula in gradients.items():
                formula_short = formula[:80] + '...' if len(formula) > 80 else formula
                output.append(f"- `{input_name}`: `{formula_short}`")

    return "\n".join(output)


@mcp.tool()
async def get_dispatch_implementations(function_name: str) -> str:
    """
    Get all backend implementations (CPU, CUDA, etc.) for a function.

    PyTorch dispatches operations to different backends based on tensor device.
    This shows which backends have implementations and where they are.

    Args:
        function_name: Function name (e.g., "add", "matmul", "conv2d")

    Returns:
        Table of dispatch keys and their implementation files
    """
    _ensure_loaded()

    output = [f"## Dispatch implementations for `{function_name}`\n"]

    # First try native_functions.yaml (authoritative)
    native_func = _get_native_function_info(function_name)
    if native_func:
        dispatch = native_func.get('dispatch', {})
        if dispatch:
            output.append("### From native_functions.yaml (authoritative)\n")
            output.append("| Dispatch Key | Implementation | File Location |")
            output.append("|-------------|----------------|---------------|")

            for key, impl in sorted(dispatch.items()):
                # Try to find the actual file
                file_loc = "N/A"
                if impl in _native_implementations:
                    impls = _native_implementations[impl]
                    if impls:
                        rel_path = _make_relative_path(impls[0]['file_path'])
                        file_loc = f"`{rel_path}:{impls[0]['line_number']}`"
                output.append(f"| {key} | `{impl}` | {file_loc} |")

            output.append("")

    # Also show registered bindings
    if not _bindings:
        if not native_func:
            return "Error: No binding data loaded."
        return "\n".join(output)

    # Find matching bindings grouped by dispatch key
    dispatch_map: Dict[str, List[Dict]] = {}

    for binding in _bindings:
        py_name = binding.get("python_name", "").lower()
        cpp_name = binding.get("cpp_name", "").lower()

        if function_name.lower() in py_name or function_name.lower() in cpp_name:
            # Filter out generated files
            if _is_generated_file(binding.get('file_path', '')):
                continue
            key = binding.get("dispatch_key") or "default"
            dispatch_map.setdefault(key, []).append(binding)

    if dispatch_map:
        output.append("### From TORCH_LIBRARY_IMPL registrations\n")
        output.append("| Dispatch Key | C++ Function | File |")
        output.append("|-------------|--------------|------|")

        for key in ["CPU", "CUDA", "Meta", "CompositeImplicitAutograd", "CompositeExplicitAutograd", "default"]:
            if key in dispatch_map:
                for b in dispatch_map[key][:3]:
                    rel_path = _make_relative_path(b["file_path"])
                    line_info = f":{b['line']}" if b.get('line') else ""
                    output.append(f"| {key} | `{b['cpp_name']}` | `{rel_path}{line_info}` |")

        # Any other keys
        for key, bindings in dispatch_map.items():
            if key not in ["CPU", "CUDA", "Meta", "CompositeImplicitAutograd", "CompositeExplicitAutograd", "default"]:
                for b in bindings[:2]:
                    rel_path = _make_relative_path(b["file_path"])
                    line_info = f":{b['line']}" if b.get('line') else ""
                    output.append(f"| {key} | `{b['cpp_name']}` | `{rel_path}{line_info}` |")

    if not dispatch_map and not native_func:
        return f"No dispatch implementations found for '{function_name}'."

    return "\n".join(output)


@mcp.tool()
async def get_cuda_kernels(function_name: str = "") -> str:
    """
    Get CUDA kernel information for a function or list all kernels.

    CUDA kernels are __global__ functions that run on GPU.
    This shows kernel names, files, and what C++ functions call them.

    Args:
        function_name: Optional filter (empty = list all kernels)

    Returns:
        CUDA kernel details including callers
    """
    _ensure_loaded()
    # Get kernels from both dedicated list and bindings with cuda_kernel type
    all_kernels = list(_cuda_kernels)

    # Also check bindings for cuda_kernel type
    cuda_bindings = _by_binding_type.get("cuda_kernel", [])
    for b in cuda_bindings:
        kernel_dict = {
            "name": b.get("python_name", b.get("cpp_name", "")),
            "file_path": b.get("file_path", ""),
            "line": b.get("line"),
            "parameters": b.get("signature"),
            "called_by": [b.get("cuda_kernel")] if b.get("cuda_kernel") else [],
        }
        all_kernels.append(kernel_dict)

    if not all_kernels:
        return "No CUDA kernels found in index."

    if function_name:
        kernels = [k for k in all_kernels
                   if function_name.lower() in k.get("name", "").lower()
                   or any(function_name.lower() in c.lower() for c in k.get("called_by", []))]
    else:
        # Deduplicate by name
        seen = set()
        kernels = []
        for k in all_kernels:
            name = k.get("name", "")
            if name and name not in seen:
                seen.add(name)
                kernels.append(k)
                if len(kernels) >= 30:
                    break

    if not kernels:
        return f"No CUDA kernels found matching '{function_name}'."

    output = [f"## CUDA Kernels" + (f" for '{function_name}'" if function_name else "") + "\n"]

    for k in kernels[:30]:
        output.append(f"### `{k.get('name', 'unknown')}`")
        output.append(f"- File: `{k.get('file_path', 'unknown')}`" + (f" (line {k['line']})" if k.get('line') else ""))
        if k.get("parameters"):
            params = k["parameters"][:100] + "..." if len(k.get("parameters", "")) > 100 else k.get("parameters", "")
            output.append(f"- Parameters: `{params}`")
        if k.get("called_by"):
            output.append(f"- Called by: {', '.join(str(c) for c in k['called_by'][:5])}")
        output.append("")

    total = len(all_kernels)
    if total > len(kernels):
        output.append(f"\n*Showing {len(kernels)} of {total} total kernels*")

    return "\n".join(output)


@mcp.tool()
async def get_implementation_files(function_name: str) -> str:
    """
    Get file paths that implement a function across all languages.

    Returns just FILE PATHS organized by type (Python, C++, CUDA).
    Use Claude's Read tool to examine the actual code.

    Args:
        function_name: Function name to find

    Returns:
        File paths grouped by language/backend
    """
    _ensure_loaded()
    if not _bindings:
        return "Error: No binding data loaded."

    files: Dict[str, set] = {
        "cpp_binding": set(),
        "cpp_implementation": set(),
        "cuda": set(),
    }

    for binding in _bindings:
        py_name = binding.get("python_name", "").lower()
        cpp_name = binding.get("cpp_name", "").lower()

        if function_name.lower() in py_name or function_name.lower() in cpp_name:
            fp = binding.get("file_path", "")
            dispatch = binding.get("dispatch_key", "")

            if fp.endswith(('.cu', '.cuh')):
                files["cuda"].add(fp)
            elif dispatch == "CUDA":
                files["cuda"].add(fp)
            elif "pybind" in binding.get("type", ""):
                files["cpp_binding"].add(fp)
            else:
                files["cpp_implementation"].add(fp)

    # Add CUDA kernel files
    for kernel in _cuda_kernels:
        if function_name.lower() in kernel["name"].lower():
            files["cuda"].add(kernel["file_path"])

    if not any(files.values()):
        return f"No implementation files found for '{function_name}'."

    output = [f"## Implementation files for '{function_name}'\n"]

    if files["cpp_binding"]:
        output.append("### Python-C++ Bindings")
        for f in sorted(files["cpp_binding"])[:10]:
            output.append(f"- `{f}`")
        output.append("")

    if files["cpp_implementation"]:
        output.append("### C++ Implementations")
        for f in sorted(files["cpp_implementation"])[:10]:
            output.append(f"- `{f}`")
        output.append("")

    if files["cuda"]:
        output.append("### CUDA Files")
        for f in sorted(files["cuda"])[:10]:
            output.append(f"- `{f}`")
        output.append("")

    output.append("**Next step:** Use Claude's Read tool to examine these files.")
    return "\n".join(output)


@mcp.tool()
async def get_call_graph(function_name: str) -> str:
    """
    Get call relationships for a function.

    Shows what a function calls and what calls it.
    Useful for understanding code flow and dependencies.

    Args:
        function_name: Function to analyze (partial match)

    Returns:
        Callers and callees for matching functions
    """
    _ensure_loaded()
    if not _call_graph:
        return "No call graph data available."

    calls_out = {}  # what this function calls
    calls_in = {}   # what calls this function

    for func, called in _call_graph.items():
        if function_name.lower() in func.lower():
            calls_out[func] = called

        for c in called:
            if function_name.lower() in c.lower():
                calls_in.setdefault(c, []).append(func)

    if not calls_out and not calls_in:
        return f"No call graph data for '{function_name}'."

    output = [f"## Call graph for '{function_name}'\n"]

    if calls_out:
        output.append("### Outgoing calls (what it calls):\n")
        for func, called in list(calls_out.items())[:10]:
            output.append(f"**{func}** →")
            for c in called[:8]:
                output.append(f"  - {c}")
            if len(called) > 8:
                output.append(f"  - ... and {len(called) - 8} more")
            output.append("")

    if calls_in:
        output.append("### Incoming calls (what calls it):\n")
        for func, callers in list(calls_in.items())[:10]:
            output.append(f"**{func}** ←")
            for c in callers[:8]:
                output.append(f"  - {c}")
            output.append("")

    return "\n".join(output)


@mcp.tool()
async def list_binding_types() -> str:
    """
    Get summary of all binding types in the codebase.

    Shows counts and examples for each type:
    - pybind_function: m.def() bindings
    - pybind_class: py::class_ bindings
    - torch_library: TORCH_LIBRARY operators
    - cuda_kernel: __global__ functions
    - at_dispatch: AT_DISPATCH macros
    """
    _ensure_loaded()
    if not _by_binding_type:
        return "No binding type data available."

    output = ["## Binding Types Summary\n"]

    for btype, bindings in sorted(_by_binding_type.items(), key=lambda x: -len(x[1])):
        output.append(f"### {btype} ({len(bindings)} bindings)")

        # Show a few examples
        examples = bindings[:3]
        for b in examples:
            output.append(f"  - `{b['python_name']}` → `{b['cpp_name']}`")

        if len(bindings) > 3:
            output.append(f"  - ... and {len(bindings) - 3} more")
        output.append("")

    # Dispatch key summary
    if _by_dispatch_key:
        output.append("### Dispatch Keys\n")
        for key, bindings in sorted(_by_dispatch_key.items(), key=lambda x: -len(x[1])):
            output.append(f"- **{key}**: {len(bindings)} implementations")

    return "\n".join(output)


@mcp.tool()
async def search_bindings(query: str, limit: int = 20) -> str:
    """
    Search all bindings by name (Python or C++).

    Matches partial names in both Python and C++ function names.

    Args:
        query: Search term (case-insensitive)
        limit: Maximum results (default 20)

    Returns:
        Matching bindings with dispatch info
    """
    _ensure_loaded()
    if not _bindings:
        return "No bindings loaded."

    matches = []
    query_lower = query.lower()

    for binding in _bindings:
        py_name = binding.get("python_name", "").lower()
        cpp_name = binding.get("cpp_name", "").lower()

        if query_lower in py_name or query_lower in cpp_name:
            matches.append(binding)

    if not matches:
        return f"No bindings found matching '{query}'."

    # Deduplicate
    seen = set()
    unique = []
    for m in matches:
        key = (m.get("python_name"), m.get("cpp_name"))
        if key not in seen:
            seen.add(key)
            unique.append(m)

    output = [f"## Search results for '{query}'\n"]
    output.append(f"Found {len(unique)} unique binding(s):\n")

    for b in unique[:limit]:
        dispatch = f" [{b['dispatch_key']}]" if b.get('dispatch_key') else ""
        output.append(f"- **{b['python_name']}**{dispatch}")
        output.append(f"  → `{b['cpp_name']}`")
        output.append(f"  → `{b['file_path']}`")

    if len(unique) > limit:
        output.append(f"\n*Showing {limit} of {len(unique)} results*")

    return "\n".join(output)


# ============================================================================
# Public API
# ============================================================================

def load_binding_data(index_path: str):
    """Load binding data from index (for testing/external use)."""
    _init_from_index(index_path)


def extract_bindings_to_json(repo_path: str, output_file: str):
    """
    Extract bindings directly from repo (skips LlamaIndex entirely).
    Creates a lightweight JSON file for fast MCP server startup.
    """
    from torchtalk.analysis.binding_detector import BindingDetector
    from torchtalk.analysis.repo_analyzer import RepoAnalyzer

    log.info(f"Extracting bindings from {repo_path}...")

    # Detect bindings
    detector = BindingDetector()
    binding_graph = detector.detect_bindings_in_directory(repo_path)

    # Convert to serializable format
    bindings = []
    for b in binding_graph.bindings:
        bindings.append({
            "python_name": b.python_name,
            "cpp_name": b.cpp_name,
            "type": b.binding_type,
            "file_path": b.file_path,
            "line": b.line_number,
            "dispatch_key": b.dispatch_key,
            "namespace": b.namespace,
            "signature": b.signature,
            "cuda_kernel": b.cuda_kernel,
        })

    cuda_kernels = []
    for k in binding_graph.cuda_kernels:
        cuda_kernels.append({
            "name": k.name,
            "file_path": k.file_path,
            "line": k.line_number,
            "parameters": k.parameters,
            "template_params": k.template_params,
            "called_by": k.called_by,
        })

    # Analyze repo for call/import graphs
    log.info("Analyzing repository structure...")
    try:
        analyzer = RepoAnalyzer(repo_path)
        analyzer.analyze_repository()

        call_graph = {}
        for node in analyzer.call_graph.nodes():
            successors = list(analyzer.call_graph.successors(node))
            if successors:
                call_graph[str(node)] = [str(s) for s in successors]

        import_graph = {}
        for node in analyzer.import_graph.nodes():
            successors = list(analyzer.import_graph.successors(node))
            if successors:
                import_graph[str(node)] = [str(s) for s in successors]
    except Exception as e:
        log.warning(f"Graph analysis failed: {e}")
        call_graph = {}
        import_graph = {}

    # Save to JSON
    data = {
        "bindings": bindings,
        "cuda_kernels": cuda_kernels,
        "call_graph": call_graph,
        "import_graph": import_graph,
        "metadata": {
            "repo_path": repo_path,
            "binding_count": len(bindings),
            "kernel_count": len(cuda_kernels),
        }
    }

    # Ensure output directory exists
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump(data, f)

    file_size = Path(output_file).stat().st_size / (1024 * 1024)
    log.info(f"Saved {len(bindings)} bindings, {len(cuda_kernels)} kernels to {output_file} ({file_size:.1f} MB)")


def run_server(pytorch_source: Optional[str] = None, index_path: Optional[str] = None, transport: str = "stdio"):
    """
    Run the MCP server.

    Args:
        pytorch_source: Path to PyTorch source (auto-builds index)
        index_path: Path to existing bindings.json or index directory
        transport: MCP transport type
    """
    log.info(f"Initializing TorchTalk MCP server...")

    try:
        if pytorch_source:
            _init_from_source(pytorch_source)
        elif index_path:
            _init_from_index(index_path)
        else:
            # Try auto-detection
            if not _ensure_loaded():
                log.error("No PyTorch source found. Use --pytorch-source or set PYTORCH_SOURCE env var.")
                sys.exit(1)
    except Exception as e:
        log.error(f"Failed to initialize: {e}")
        sys.exit(1)

    log.info("Starting MCP server...")
    mcp.run(transport=transport)


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="TorchTalk MCP Server - Cross-language binding data for Claude Code",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-detect PyTorch source and build index
  torchtalk mcp-serve --pytorch-source /path/to/pytorch

  # Use existing index
  torchtalk mcp-serve --index ./pytorch_index

  # Let it auto-detect (looks for ../pytorch, PYTORCH_SOURCE env var, etc.)
  torchtalk mcp-serve

One-command Claude Code setup:
  claude mcp add torchtalk -s user -- torchtalk mcp-serve --pytorch-source /path/to/pytorch
        """
    )
    parser.add_argument(
        "--pytorch-source", "-p",
        help="Path to PyTorch source code (auto-builds and caches index)"
    )
    parser.add_argument(
        "--index", "-i",
        help="Path to existing bindings.json or index directory"
    )
    parser.add_argument(
        "--transport",
        default="stdio",
        choices=["stdio", "streamable-http"],
        help="MCP transport type (default: stdio)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )

    args = parser.parse_args()

    # Setup logging (to stderr, so it doesn't interfere with MCP stdio)
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        stream=sys.stderr,
    )

    run_server(
        pytorch_source=args.pytorch_source,
        index_path=args.index,
        transport=args.transport
    )


if __name__ == "__main__":
    main()
