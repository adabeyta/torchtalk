"""TorchTalk MCP Server - Cross-language binding analysis for PyTorch."""

import hashlib
import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, List, Any, Tuple

from mcp.server.fastmcp import FastMCP

from .formatting import Markdown, relative_path
from .analysis.helpers import (
    levenshtein_distance,
    fuzzy_match,
    safe_sort_key,
    dedupe_by_key,
    truncate,
)
from .analysis.config import (
    CPP_SEARCH_DIRS,
    PYTHON_SEARCH_DIRS,
    TEST_SEARCH_DIRS,
    TEST_UTILITY_MODULES,
    should_exclude as _should_exclude,
    has_test_patterns as _has_test_patterns,
)

log = logging.getLogger(__name__)
mcp = FastMCP("torchtalk")

CACHE_DIR = Path.home() / ".cache" / "torchtalk"


@dataclass
class ServerState:
    """Server state container."""
    bindings: List[Dict] = field(default_factory=list)
    cuda_kernels: List[Dict] = field(default_factory=list)
    native_functions: Dict[str, Dict] = field(default_factory=dict)
    derivatives: Dict[str, Dict] = field(default_factory=dict)
    native_implementations: Dict[str, List[Dict]] = field(default_factory=dict)

    by_python_name: Dict[str, List[Dict]] = field(default_factory=dict)
    by_cpp_name: Dict[str, List[Dict]] = field(default_factory=dict)
    by_dispatch_key: Dict[str, List[Dict]] = field(default_factory=dict)

    py_modules: Dict[str, Any] = field(default_factory=dict)
    py_classes: Dict[str, List[Any]] = field(default_factory=dict)
    py_functions: Dict[str, List[Any]] = field(default_factory=dict)
    nn_modules: List[Any] = field(default_factory=list)

    test_files: Dict[str, Dict] = field(default_factory=dict)
    test_classes: Dict[str, List[Dict]] = field(default_factory=dict)
    test_functions: Dict[str, List[Dict]] = field(default_factory=dict)
    test_utilities: Dict[str, Dict] = field(default_factory=dict)
    opinfo_registry: Dict[str, Dict] = field(default_factory=dict)

    pytorch_source: Optional[str] = None
    cpp_extractor: Any = None
    cpp_building: bool = False
    cpp_thread: Any = None


_state = ServerState()


# --- Caching ---


def _cache_path(source: str) -> Path:
    """Get cache file path for source directory."""
    path_hash = hashlib.md5(str(Path(source).resolve()).encode()).hexdigest()[:12]
    return CACHE_DIR / f"bindings_{path_hash}.json"


def _source_fingerprint(source: str) -> str:
    """Fingerprint source to detect changes."""
    src = Path(source)
    files = [
        src / "version.txt",
        src / "torch/version.py",
        src / ".git/HEAD",
    ]
    parts = []
    for f in files:
        if f.exists():
            s = f.stat()
            parts.append(f"{f.name}:{s.st_mtime}:{s.st_size}")
    return hashlib.md5("|".join(parts).encode()).hexdigest()[:16]


def _cache_valid(cache: Path, source: str) -> bool:
    """Check if cache is valid."""
    if not cache.exists():
        return False
    try:
        with open(cache) as f:
            data = json.load(f)
        return data.get("metadata", {}).get(
            "source_fingerprint"
        ) == _source_fingerprint(source)
    except Exception:
        return False


# --- YAML Parsing ---


def _parse_native_functions(source: str) -> Tuple[Dict, Dict]:
    """Parse native_functions.yaml and derivatives.yaml."""
    import yaml

    src = Path(source)
    functions, derivatives = {}, {}

    # native_functions.yaml
    nf_yaml = src / "aten/src/ATen/native/native_functions.yaml"
    if nf_yaml.exists():
        try:
            data = yaml.safe_load(nf_yaml.read_text())
            for entry in data or []:
                if not isinstance(entry, dict) or "func" not in entry:
                    continue

                sig = entry["func"]
                match = re.match(r"(\w+(?:\.\w+)?)\s*\(", sig)
                if not match:
                    continue

                name = match.group(1)
                base = name.split(".")[0]

                dispatch = {}
                if isinstance(entry.get("dispatch"), dict):
                    for key, impl in entry["dispatch"].items():
                        for k in key.split(","):
                            dispatch[k.strip()] = impl

                func = {
                    "name": name,
                    "base_name": base,
                    "signature": sig,
                    "dispatch": dispatch,
                    "variants": entry.get("variants", ""),
                    "structured": entry.get("structured", False),
                    "structured_delegate": entry.get("structured_delegate"),
                    "tags": entry.get("tags", []),
                }
                functions[name] = func
                if name != base and base not in functions:
                    functions[base] = func

            log.info(f"Parsed {len(functions)} native functions")
        except Exception as e:
            log.warning(f"Failed to parse native_functions.yaml: {e}")

    # derivatives.yaml
    deriv_yaml = src / "tools/autograd/derivatives.yaml"
    if deriv_yaml.exists():
        try:
            data = yaml.safe_load(deriv_yaml.read_text())
            for entry in data or []:
                if not isinstance(entry, dict) or "name" not in entry:
                    continue

                match = re.match(r"(\w+(?:\.\w+)?)", entry["name"])
                if not match:
                    continue

                name = match.group(1)
                grads = {
                    k: v
                    for k, v in entry.items()
                    if k not in {"name", "dispatch", "output_differentiability"}
                    and isinstance(v, str)
                }

                derivatives[name] = {
                    "name": name,
                    "signature": entry["name"],
                    "gradients": grads,
                }

            log.info(f"Parsed {len(derivatives)} derivatives")
        except Exception as e:
            log.warning(f"Failed to parse derivatives.yaml: {e}")

    return functions, derivatives


# --- Implementation Finding ---


def _find_implementations(source: str, functions: Dict) -> Dict[str, List[Dict]]:
    """Find C++ implementations in source using configured search directories."""
    src = Path(source)

    # Use configured search directories
    search_dirs = [src / d for d in CPP_SEARCH_DIRS]

    # Target function names
    targets = set()
    for f in functions.values():
        targets.add(f["base_name"])
        targets.update(f.get("dispatch", {}).values())

    log.info(f"Searching for {len(targets)} implementations in {len(search_dirs)} directories...")

    # Patterns for function definitions
    patterns = [
        re.compile(
            r"^(?:static\s+)?(?:inline\s+)?(?:TORCH_API\s+)?"
            r"(?:Tensor&?|void|bool|int64_t|double|std::tuple<[^>]+>|at::Tensor)\s+"
            r"(\w+)\s*\([^;]*\)\s*(?:const\s*)?{",
            re.MULTILINE,
        ),
        re.compile(
            r"^(?:Tensor&?|void)\s+(?:\w+::)+(\w+)\s*\([^;]*\)\s*{", re.MULTILINE
        ),
    ]

    impls: Dict[str, List[Dict]] = {}
    file_count = 0

    for search_dir in search_dirs:
        if not search_dir.exists():
            continue

        for cpp_file in search_dir.rglob("*.cpp"):
            file_str = str(cpp_file)

            # Apply exclusion patterns
            if _should_exclude(file_str):
                continue

            try:
                content = cpp_file.read_text(errors="ignore")
            except Exception:
                continue

            file_count += 1
            for pattern in patterns:
                for match in pattern.finditer(content):
                    func_name = match.group(1)
                    if func_name not in targets:
                        continue

                    line = content[: match.start()].count("\n") + 1
                    sig_end = content.find("\n", match.start())
                    sig = (
                        content[match.start() : sig_end].strip() if sig_end > 0 else ""
                    )

                    impl = {
                        "function_name": func_name,
                        "file_path": str(cpp_file),
                        "line_number": line,
                        "signature": truncate(sig, 100),
                    }
                    impls.setdefault(func_name, []).append(impl)

    log.info(
        f"Found {sum(len(v) for v in impls.values())} implementations in {file_count} files"
    )
    return impls


def _fuzzy_find(name: str, data: Dict[str, Any]) -> Optional[List[Any]]:
    """Fuzzy match function name in a dict."""
    if name in data:
        return data[name] if isinstance(data[name], list) else [data[name]]

    name_lower = name.lower()

    # Suffix match
    for key in data:
        if key.lower().endswith(name_lower) or key.lower().endswith(f"_{name_lower}"):
            return data[key] if isinstance(data[key], list) else [data[key]]

    # Contains match
    matches = [(k, v) for k, v in data.items() if name_lower in k.lower()]
    if matches:
        matches.sort(key=lambda x: len(x[0]))
        v = matches[0][1]
        return v if isinstance(v, list) else [v]

    # Levenshtein distance match (for typos)
    if len(name) >= 3:
        candidates = []
        for key in data:
            if abs(len(key) - len(name)) <= 3:
                dist = levenshtein_distance(name_lower, key.lower())
                max_dist = max(2, len(name) // 3)
                if dist <= max_dist:
                    candidates.append((dist, key, data[key]))

        if candidates:
            candidates.sort(key=lambda x: x[0])
            v = candidates[0][2]
            return v if isinstance(v, list) else [v]

    return None


# --- Index Building ---


def _build_index(source: str) -> Dict[str, Any]:
    """Build binding index from source."""
    from .analysis.binding_detector import BindingDetector

    log.info(f"Building index from {source}...")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Parse YAML files
    functions, derivatives = _parse_native_functions(source)

    # Find implementations
    implementations = _find_implementations(source, functions)

    # Detect bindings
    from dataclasses import asdict

    detector = BindingDetector()
    graph = detector.detect_bindings_in_directory(source)

    data = {
        "bindings": [b.to_dict() for b in graph.bindings],
        "cuda_kernels": [asdict(k) for k in graph.cuda_kernels],
        "native_functions": functions,
        "derivatives": derivatives,
        "native_implementations": implementations,
        "metadata": {
            "source_path": source,
            "source_fingerprint": _source_fingerprint(source),
        },
    }

    # Save cache
    cache = _cache_path(source)
    with open(cache, "w") as f:
        json.dump(data, f)
    log.info(f"Cached index to {cache}")

    return data


def _build_indexes(state: ServerState):
    """Build lookup indexes."""
    for binding in state.bindings:
        if py_name := binding.get("python_name"):
            state.by_python_name.setdefault(py_name, []).append(binding)
        if cpp_name := binding.get("cpp_name"):
            state.by_cpp_name.setdefault(cpp_name, []).append(binding)
        if dispatch := binding.get("dispatch_key"):
            state.by_dispatch_key.setdefault(dispatch, []).append(binding)


def _load_from_json(path: str):
    """Load bindings from JSON file."""
    global _state

    log.info(f"Loading bindings from {path}...")
    with open(path) as f:
        data = json.load(f)

    _state.bindings = data.get("bindings", [])
    _state.cuda_kernels = data.get("cuda_kernels", [])
    _state.native_functions = data.get("native_functions", {})
    _state.derivatives = data.get("derivatives", {})
    _state.native_implementations = data.get("native_implementations", {})

    _build_indexes(_state)

    log.info(
        f"Loaded {len(_state.bindings)} bindings, {len(_state.cuda_kernels)} CUDA kernels"
    )


# --- C++ Call Graph ---


def _init_cpp_call_graph(source: str):
    """Initialize C++ call graph (background thread on first run)."""
    global _state

    try:
        from .analysis.cpp_call_graph import CppCallGraphExtractor, LIBCLANG_AVAILABLE

        if not LIBCLANG_AVAILABLE:
            log.info("libclang not available - C++ call graph disabled")
            return

        src = Path(source)
        path_hash = hashlib.md5(str(src.resolve()).encode()).hexdigest()[:12]
        cache_key = f"pytorch_callgraph_parallel_{path_hash}"

        extractor = CppCallGraphExtractor()
        if extractor.load_cache(cache_key):
            _state.cpp_extractor = extractor
            log.info(
                f"Loaded C++ call graph from cache ({len(extractor.function_locations)} functions)"
            )
            return

        # Build in background
        import threading

        def build():
            global _state
            try:
                ext = CppCallGraphExtractor()
                ext.extract_from_pytorch_parallel(source)
                ext.save_cache(cache_key)
                _state.cpp_extractor = ext
                log.info(
                    f"C++ call graph ready: {len(ext.function_locations)} functions"
                )
            except Exception as e:
                log.warning(f"C++ call graph build failed: {e}")
            finally:
                _state.cpp_building = False

        _state.cpp_building = True
        log.info("Building C++ call graph in background (~1-2 min)...")
        _state.cpp_thread = threading.Thread(target=build, daemon=True)
        _state.cpp_thread.start()

    except Exception as e:
        log.warning(f"Failed to init C++ call graph: {e}")


def _cpp_status() -> str:
    """Get C++ call graph status. Empty string if ready."""
    if _state.cpp_building:
        return "C++ call graph building in background (~1-2 min). Try again shortly."

    if not _state.cpp_extractor:
        if _state.pytorch_source:
            src = Path(_state.pytorch_source)
            if (
                not (src / "compile_commands.json").exists()
                and not (src / "build/compile_commands.json").exists()
            ):
                return (
                    "C++ call graph unavailable - `compile_commands.json` not found.\n\n"
                    "**To enable:** Build PyTorch once:\n"
                    f"```\ncd {_state.pytorch_source}\npython setup.py develop\n```"
                )
        return "C++ call graph unavailable. Install libclang or build PyTorch."

    return ""


# --- Initialization ---


def _ensure_loaded(component: str = "bindings"):
    """Ensure required data is loaded.

    Args:
        component: What to check - "bindings" (default), "test", "python", "cpp"
    """
    checks = {
        "bindings": (_state.bindings or _state.native_functions, "No data loaded"),
        "test": (_state.test_files, "Test infrastructure not loaded"),
        "python": (_state.py_modules, "Python module analysis not available"),
        "cpp": (_state.cpp_extractor, "C++ call graph not available"),
    }
    loaded, msg = checks.get(component, (True, ""))
    if not loaded:
        raise RuntimeError(f"{msg}. Start server with --pytorch-source.")


def _init_python_modules(source: str):
    """Initialize Python module analysis using configured search directories."""
    global _state

    try:
        from .analysis.python_analyzer import PythonAnalyzer, build_module_index

        analyzer = PythonAnalyzer()
        src = Path(source)

        # Use configured Python search directories
        dirs_to_analyze = [src / d for d in PYTHON_SEARCH_DIRS]

        all_modules: Dict[str, Any] = {}
        analyzed_count = 0
        for dir_path in dirs_to_analyze:
            if dir_path.exists():
                modules = analyzer.analyze_directory(str(dir_path), skip_tests=True)
                all_modules.update(modules)
                analyzed_count += 1

        log.info(f"Analyzed {analyzed_count}/{len(dirs_to_analyze)} Python directories")

        if all_modules:
            index = build_module_index(all_modules)
            _state.py_modules = all_modules
            _state.py_classes = index["by_class"]
            _state.py_functions = index["by_function"]
            _state.nn_modules = index["nn_modules"]
            log.info(
                f"Loaded {len(all_modules)} Python modules, "
                f"{len(_state.nn_modules)} nn.Module classes"
            )
    except Exception as e:
        log.warning(f"Failed to analyze Python modules: {e}")


def _init_from_source(source: str):
    """Initialize from PyTorch source."""
    global _state

    src = Path(source).resolve()
    if not src.exists():
        raise FileNotFoundError(f"Source not found: {source}")

    cache = _cache_path(str(src))
    if _cache_valid(cache, str(src)):
        log.info(f"Using cached index from {cache}")
        _load_from_json(str(cache))
    else:
        data = _build_index(str(src))
        _state.bindings = data.get("bindings", [])
        _state.cuda_kernels = data.get("cuda_kernels", [])
        _state.native_functions = data.get("native_functions", {})
        _state.derivatives = data.get("derivatives", {})
        _state.native_implementations = data.get("native_implementations", {})
        _build_indexes(_state)

    _state.pytorch_source = str(src)
    _init_cpp_call_graph(str(src))
    _init_python_modules(str(src))
    _init_test_infrastructure(str(src))


def _init_test_infrastructure(source: str):
    """Initialize test infrastructure analysis."""
    global _state

    try:
        import ast
        src = Path(source)

        log.info("Analyzing test infrastructure...")

        # Scan test directories
        for test_dir in TEST_SEARCH_DIRS:
            dir_path = src / test_dir
            if not dir_path.exists():
                continue

            for py_file in dir_path.rglob("*.py"):
                file_str = str(py_file)

                # Skip __pycache__
                if "__pycache__" in file_str:
                    continue

                try:
                    content = py_file.read_text(encoding="utf-8", errors="replace")

                    # For test/ directory, be inclusive - index all .py files
                    # For torch/testing/, use pattern filtering
                    is_main_test_dir = "/test/" in file_str and "/testing/" not in file_str
                    if not is_main_test_dir and not _has_test_patterns(content):
                        continue

                    # Parse AST to extract test classes and functions
                    try:
                        tree = ast.parse(content, filename=str(py_file))
                    except SyntaxError:
                        continue

                    rel_path = str(py_file.relative_to(src))
                    file_info = {
                        "path": rel_path,
                        "full_path": file_str,
                        "classes": [],
                        "functions": [],
                        "imports": [],
                    }

                    for node in ast.walk(tree):
                        # Extract test classes
                        if isinstance(node, ast.ClassDef):
                            bases = [
                                _get_ast_name(b) for b in node.bases
                            ]
                            is_test_class = any(
                                "TestCase" in b or "TestBase" in b for b in bases
                            )
                            class_info = {
                                "name": node.name,
                                "file": rel_path,
                                "line": node.lineno,
                                "bases": bases,
                                "is_test_class": is_test_class,
                            }
                            file_info["classes"].append(class_info)

                            # Index test classes
                            _state.test_classes.setdefault(node.name, []).append(class_info)

                            # Extract test methods
                            for item in node.body:
                                if isinstance(item, ast.FunctionDef) and item.name.startswith("test_"):
                                    func_info = {
                                        "name": item.name,
                                        "class": node.name,
                                        "file": rel_path,
                                        "line": item.lineno,
                                    }
                                    file_info["functions"].append(func_info)
                                    _state.test_functions.setdefault(item.name, []).append(func_info)

                        # Extract standalone test functions
                        elif isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
                            if node.col_offset == 0:  # Top-level function
                                func_info = {
                                    "name": node.name,
                                    "class": None,
                                    "file": rel_path,
                                    "line": node.lineno,
                                }
                                file_info["functions"].append(func_info)
                                _state.test_functions.setdefault(node.name, []).append(func_info)

                    _state.test_files[rel_path] = file_info

                except Exception as e:
                    log.debug(f"Error parsing {py_file.name}: {e}")

        # Index test utilities
        for util_path in TEST_UTILITY_MODULES:
            full_path = src / util_path
            if full_path.exists():
                _state.test_utilities[util_path] = {
                    "path": util_path,
                    "full_path": str(full_path),
                    "exists": True,
                }

        # Parse OpInfo registry - scan all opinfo definition files
        opinfo_dirs = [
            src / "torch/testing/_internal/opinfo",
            src / "torch/testing/_internal/opinfo/definitions",
        ]
        for opinfo_dir in opinfo_dirs:
            if opinfo_dir.exists():
                for opinfo_file in opinfo_dir.glob("*.py"):
                    _parse_opinfo_registry(str(opinfo_file))

        log.info(
            f"Test infrastructure: {len(_state.test_files)} files, "
            f"{len(_state.test_classes)} test classes, "
            f"{len(_state.test_functions)} test functions"
        )

    except Exception as e:
        log.warning(f"Failed to analyze test infrastructure: {e}")


def _get_ast_name(node) -> str:
    """Get string name from AST node."""
    import ast
    try:
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{_get_ast_name(node.value)}.{node.attr}"
        elif isinstance(node, ast.Subscript):
            return _get_ast_name(node.value)
        elif isinstance(node, ast.Call):
            return _get_ast_name(node.func)
        elif isinstance(node, ast.Constant):
            return str(node.value) if node.value else ""
    except Exception:
        pass
    return ""


def _parse_opinfo_registry(opinfo_path: str):
    """Parse OpInfo definitions to map operators to their test info."""
    global _state

    try:
        path = Path(opinfo_path)
        content = path.read_text()

        # Find OpInfo definitions - pattern: OpInfo('op_name', ...)
        import re
        pattern = r"OpInfo\s*\(\s*['\"]([^'\"]+)['\"]"

        # Get relative path for display
        if _state.pytorch_source:
            try:
                rel_path = str(path.relative_to(_state.pytorch_source))
            except ValueError:
                rel_path = str(path)
        else:
            rel_path = str(path)

        count = 0
        for match in re.finditer(pattern, content):
            op_name = match.group(1)
            line = content[:match.start()].count("\n") + 1
            _state.opinfo_registry[op_name] = {
                "name": op_name,
                "file": rel_path,
                "line": line,
            }
            count += 1

        if count > 0:
            log.debug(f"Found {count} OpInfo definitions in {path.name}")

    except Exception as e:
        log.debug(f"Failed to parse OpInfo from {opinfo_path}: {e}")


def _auto_detect_pytorch() -> Optional[str]:
    """Auto-detect PyTorch source."""
    candidates = [
        os.environ.get("PYTORCH_SOURCE"),
        os.environ.get("PYTORCH_PATH"),
        Path.cwd() / "pytorch",
        Path.cwd().parent / "pytorch",
        Path("/myworkspace/pytorch"),
    ]
    for c in candidates:
        if c and Path(c).exists() and (Path(c) / "torch").exists():
            return str(c)
    return None


# --- Helpers ---


def _get_native_func(name: str) -> Optional[Dict]:
    """Get native function info with fuzzy matching."""
    if name in _state.native_functions:
        return _state.native_functions[name]

    name_lower = name.lower()
    for key, func in _state.native_functions.items():
        if key.lower() == name_lower or func.get("base_name", "").lower() == name_lower:
            return func

    # Partial match
    matches = [
        (k, f) for k, f in _state.native_functions.items() if name_lower in k.lower()
    ]
    if matches:
        matches.sort(key=lambda x: len(x[0]))
        return matches[0][1]

    return None


def _similar_functions(name: str, limit: int = 10) -> List[str]:
    """Find similar function names using substring match and Levenshtein distance."""
    name_lower = name.lower()
    results = []  # (priority, length, name)

    # Substring matches (priority 0)
    substring_matches = {k for k in _state.native_functions if name_lower in k.lower()}
    results.extend((0, len(m), m) for m in substring_matches)

    # Levenshtein matches for typos (priority = distance)
    if len(name) >= 3:
        max_dist = max(3, len(name) // 2)
        for key in _state.native_functions:
            if key in substring_matches or abs(len(key) - len(name)) > 5:
                continue
            dist = levenshtein_distance(name_lower, key.lower())
            if dist <= max_dist:
                results.append((dist, len(key), key))

    # Sort by priority, then length; dedupe
    results.sort(key=lambda x: (x[0], x[1]))
    seen = set()
    return [n for _, _, n in results if not (n in seen or seen.add(n))][:limit]


def _rel_path(path: str) -> str:
    """Get relative path."""
    return relative_path(path, _state.pytorch_source)


def _format_call_item(
    md: Markdown, item: Dict, name_key: str, file_key: str, line_key: str
):
    """Format a caller/callee item with optional file:line."""
    name = item[name_key]
    if file_path := item.get(file_key):
        line = f":{item[line_key]}" if item.get(line_key) else ""
        md.item(f"`{name}` → `{_rel_path(file_path)}{line}`")
    else:
        md.item(f"`{name}`")


# --- MCP Tools ---


@mcp.tool()
async def get_status() -> str:
    """Get TorchTalk status and available tools."""
    md = Markdown()
    md.h2("TorchTalk Status")

    # Source
    if _state.pytorch_source:
        md.code("PyTorch Source", _state.pytorch_source)
    else:
        md.bold("PyTorch Source", "Not configured")
    md.blank()

    # Data stats (folded in from list_binding_types)
    if _state.bindings:
        md.bold("Bindings", f"{len(_state.bindings):,} loaded")
        md.item(f"CUDA kernels: {len(_state.cuda_kernels):,}", 1)
        md.item(f"Dispatch keys: {len(_state.by_dispatch_key)}", 1)
        # Top dispatch keys
        if _state.by_dispatch_key:
            top_keys = sorted(_state.by_dispatch_key.items(), key=lambda x: -len(x[1]))[
                :5
            ]
            for key, bindings in top_keys:
                md.item(f"{key}: {len(bindings)}", 2)
    else:
        md.bold("Bindings", "Not loaded")
    md.blank()

    if _state.native_functions:
        md.bold("Native Functions", f"{len(_state.native_functions):,} operators")
    if _state.derivatives:
        md.bold("Derivatives", f"{len(_state.derivatives):,} formulas")
    md.blank()

    # C++ call graph
    md.h3("C++ Call Graph")
    if _state.cpp_building:
        md.bold("Status", "Building (~1-2 min)")
    elif _state.cpp_extractor:
        stats = _state.cpp_extractor.get_call_graph_data()["stats"]
        md.bold("Status", "Ready")
        md.item(f"Functions: {stats['total_functions']:,}", 1)
        md.item(f"Call edges: {stats['total_call_edges']:,}", 1)
    else:
        md.bold("Status", "Not available")
        if _state.pytorch_source:
            src = Path(_state.pytorch_source)
            if not (src / "compile_commands.json").exists():
                md.item("Missing compile_commands.json", 1)
                md.item("Fix: Build PyTorch with `python setup.py develop`", 1)
    md.blank()

    # Python modules
    md.h3("Python Modules")
    if _state.py_modules:
        md.bold("Status", "Ready")
        md.item(f"Modules: {len(_state.py_modules):,}", 1)
        md.item(f"Classes: {sum(len(v) for v in _state.py_classes.values()):,}", 1)
        md.item(f"nn.Module subclasses: {len(_state.nn_modules):,}", 1)
    else:
        md.bold("Status", "Not loaded")
    md.blank()

    # Test infrastructure
    md.h3("Test Infrastructure")
    if _state.test_files:
        md.bold("Status", "Ready")
        md.item(f"Test files: {len(_state.test_files):,}", 1)
        md.item(f"Test classes: {len(_state.test_classes):,}", 1)
        md.item(f"Test functions: {len(_state.test_functions):,}", 1)
        md.item(f"OpInfo definitions: {len(_state.opinfo_registry):,}", 1)
    else:
        md.bold("Status", "Not loaded")
    md.blank()

    # Tools
    md.h3("Available Tools")
    ready = "Ready" if _state.bindings else "Not ready"
    cpp_ready = (
        "Ready"
        if _state.cpp_extractor
        else ("Building..." if _state.cpp_building else "Not ready")
    )
    py_ready = "Ready" if _state.py_modules else "Not ready"
    test_ready = "Ready" if _state.test_files else "Not ready"

    md.table(
        ["Tool", "Status", "Description"],
        [
            ["`trace`", ready, "ATen op: Python → C++ → file:line"],
            ["`search`", ready, "Find ATen bindings by name, optional backend"],
            ["`trace_module`", py_ready, "Python module: torch.nn.Linear, etc."],
            ["`list_modules`", py_ready, "List nn.Module classes, optimizers"],
            ["`impact`", cpp_ready, "Transitive callers + Python entry points"],
            ["`calls`", cpp_ready, "Functions this function invokes (outbound)"],
            ["`called_by`", cpp_ready, "Functions that invoke this (inbound)"],
            ["`cuda_kernels`", ready, "GPU kernel launches with locations"],
            ["`find_similar_tests`", test_ready, "Find tests for an operator/concept"],
            ["`list_test_utils`", test_ready, "List test utilities and patterns"],
            ["`test_file_info`", test_ready, "Details about a specific test file"],
        ],
    )

    return md.build()


@mcp.tool()
async def trace(function_name: str, focus: str = "full") -> str:
    """
    Trace a PyTorch function from Python API to C++ implementation.

    Args:
        function_name: Function to trace (e.g., "add", "matmul", "softmax")
        focus: Level of detail - "full" (default), "yaml" (definition only), "dispatch" (backends only)

    Returns:
        Binding chain with file:line locations for each layer.
    """
    _ensure_loaded()

    md = Markdown()
    md.h2(f"Trace: `{function_name}`")

    native = _get_native_func(function_name)
    base_name = native.get("base_name", function_name) if native else function_name

    # === YAML section (always shown for "full" and "yaml") ===
    if focus in ("full", "yaml") and native:
        md.h3("Definition (native_functions.yaml)")
        md.code("Signature", native.get("signature", "N/A"))

        if native.get("variants"):
            md.bold("Variants", native["variants"])
        if native.get("structured"):
            md.bold("Structured", "Yes")
        if native.get("structured_delegate"):
            md.bold("Delegate", native["structured_delegate"])
        if native.get("tags"):
            md.bold("Tags", ", ".join(str(t) for t in native["tags"]))

        dispatch = native.get("dispatch", {})
        if dispatch:
            md.blank().text("**Dispatch Config:**")
            # Safe sort handling None keys
            for key, impl in sorted(
                dispatch.items(), key=lambda x: safe_sort_key(x[0])
            ):
                md.item(f"{key or 'default'}: `{impl}`")
        md.blank()

        # Derivative formula
        deriv = _state.derivatives.get(function_name) or _state.derivatives.get(
            base_name
        )
        if deriv and deriv.get("gradients"):
            md.text("**Derivative:**")
            for input_name, formula in deriv["gradients"].items():
                md.item(f"`{input_name}`: `{truncate(formula, 60)}`")
            md.blank()

    # === Dispatch section (always shown for "full" and "dispatch") ===
    if focus in ("full", "dispatch"):
        # From binding detector (TORCH_LIBRARY_IMPL registrations)
        bindings = _state.by_python_name.get(function_name, [])
        if not bindings:
            bindings = _state.by_cpp_name.get(function_name, [])
        if not bindings:
            found = _fuzzy_find(function_name, _state.by_python_name)
            if found:
                bindings = found

        if bindings:
            md.h3("Dispatch Registrations (TORCH_LIBRARY_IMPL)")
            rows = []
            seen = set()
            for b in bindings:
                key = (b.get("dispatch_key"), b.get("cpp_name"), b.get("file_path"))
                if key in seen:
                    continue
                seen.add(key)
                dispatch = b.get("dispatch_key") or "default"
                cpp = b.get("cpp_name") or "N/A"
                path = _rel_path(b.get("file_path") or "")
                line = f":{b['line_number']}" if b.get("line_number") else ""
                rows.append([dispatch, f"`{cpp}`", f"`{path}{line}`"])

            if rows:
                md.table(["Backend", "C++ Function", "File"], rows[:20])
            md.blank()

        # From native_functions.yaml dispatch config
        if native and native.get("dispatch") and not bindings:
            md.h3("Dispatch Config (from YAML)")
            # Safe sort handling None keys
            for key, impl in sorted(
                native["dispatch"].items(), key=lambda x: safe_sort_key(x[0])
            ):
                md.item(f"**{key or 'default'}**: `{impl}`")
            md.blank()

    # === Implementation locations (only for "full") ===
    impls = []  # Initialize for "not found" check later
    if focus == "full":
        if base_name in _state.native_implementations:
            impls.extend(_state.native_implementations[base_name])
        else:
            found = _fuzzy_find(base_name, _state.native_implementations)
            if found:
                impls.extend(found)

        # Also check dispatch targets
        if native:
            for impl_name in native.get("dispatch", {}).values():
                if impl_name in _state.native_implementations:
                    impls.extend(_state.native_implementations[impl_name])

        if impls:
            md.h3("C++ Implementations")
            seen = set()
            for impl in impls[:10]:
                key = (
                    impl["function_name"],
                    impl.get("file_path"),
                    impl.get("line_number"),
                )
                if key in seen:
                    continue
                seen.add(key)

                path = _rel_path(impl.get("file_path", ""))
                line = impl.get("line_number", "")
                md.item(f"`{impl['function_name']}` → `{path}:{line}`")
            md.blank()

    # Not found handling - check if we actually found anything useful
    bindings_found = _state.by_python_name.get(function_name) or _state.by_cpp_name.get(
        function_name
    )
    found_anything = native or bindings_found or impls

    if not found_anything:
        similar = _similar_functions(function_name)
        if similar:
            md.h3("Function Not Found")
            md.text(f"No exact match for `{function_name}`. Did you mean:")
            for s in similar[:8]:
                md.item(f"`{s}`")
        else:
            md.text(f"Function `{function_name}` not found in PyTorch bindings.")
            md.text("\nTry using `search()` to find related functions.")

    return md.build()


@mcp.tool()
async def cuda_kernels(function_name: str = "") -> str:
    """
    Find CUDA kernel launches (<<<grid, block>>>) in PyTorch.

    Args:
        function_name: Optional filter - search for kernels matching this name

    Returns:
        CUDA kernel locations with file:line and caller information.
    """
    _ensure_loaded()

    md = Markdown()
    title = f"CUDA Kernels: '{function_name}'" if function_name else "CUDA Kernels"
    md.h2(title)

    kernels = _state.cuda_kernels
    if function_name:
        name_lower = function_name.lower()
        kernels = [k for k in kernels if name_lower in (k.get("name") or "").lower()]

    if not kernels:
        if function_name:
            return f"No CUDA kernels found matching '{function_name}'."
        return "No CUDA kernels found."

    md.text(f"Found {len(kernels)} kernel(s)\n")

    for kernel in kernels[:15]:
        name = kernel.get("name", "unnamed")
        path = _rel_path(kernel.get("file_path", ""))
        line = kernel.get("line_number", "")
        md.item(f"`{name}` → `{path}:{line}`")

        callers = kernel.get("callers", [])
        if callers:
            md.item(f"Called by: {', '.join(f'`{c}`' for c in callers[:3])}", 1)

    if len(kernels) > 15:
        md.text(f"\n*Showing 15 of {len(kernels)} kernels*")

    return md.build()


@mcp.tool()
async def search(query: str, backend: str = "", limit: int = 20) -> str:
    """
    Search PyTorch bindings by name with optional backend filter.

    Args:
        query: Search term (fuzzy matches Python and C++ names)
        backend: Optional filter - "CPU", "CUDA", "Meta", etc.
        limit: Max results to return (default 20)

    Returns:
        Matching bindings with dispatch keys and file locations.
    """
    _ensure_loaded()

    query_lower = query.lower()
    backend_lower = backend.lower() if backend else ""

    matches = []
    for b in _state.bindings:
        # Name match
        name_match = (
            query_lower in (b.get("python_name") or "").lower()
            or query_lower in (b.get("cpp_name") or "").lower()
        )
        if not name_match:
            continue

        # Backend filter
        if backend_lower:
            dispatch_key = (b.get("dispatch_key") or "").lower()
            if backend_lower not in dispatch_key:
                continue

        matches.append(b)

    if not matches:
        filter_msg = f" with backend '{backend}'" if backend else ""
        return f"No bindings found matching '{query}'{filter_msg}."

    # Dedupe
    seen = set()
    unique = []
    for m in matches:
        key = (m.get("python_name"), m.get("cpp_name"), m.get("dispatch_key"))
        if key not in seen:
            seen.add(key)
            unique.append(m)

    md = Markdown()
    filter_msg = f" (backend: {backend})" if backend else ""
    md.h2(f"Search: '{query}'{filter_msg}")
    md.text(f"Found {len(unique)} binding(s)\n")

    for b in unique[:limit]:
        dispatch = f" [{b['dispatch_key']}]" if b.get("dispatch_key") else ""
        py_name = b.get("python_name", "N/A")
        cpp_name = b.get("cpp_name", "N/A")
        path = _rel_path(b.get("file_path", ""))
        line = f":{b['line_number']}" if b.get("line_number") else ""
        md.item(f"**{py_name}**{dispatch} → `{cpp_name}` (`{path}{line}`)")

    if len(unique) > limit:
        md.text(f"\n*Showing {limit} of {len(unique)} results*")

    return md.build()


@mcp.tool()
async def calls(function_name: str) -> str:
    """
    Find functions that this function calls (outbound dependencies).

    Args:
        function_name: C++ function to analyze

    Returns:
        List of called functions with file:line locations.
    """
    _ensure_loaded()
    if status := _cpp_status():
        return status

    callees = _state.cpp_extractor.get_callees(function_name, fuzzy=True)
    if not callees:
        return f"No outbound calls found for '{function_name}'."

    results = dedupe_by_key(callees, "callee")

    md = Markdown()
    md.h2(f"Calls: `{function_name}`")
    md.text("*Functions this calls (outbound dependencies):*\n")

    for item in results[:30]:
        _format_call_item(md, item, "callee", "callee_file", "callee_line")

    if len(results) > 30:
        md.text(f"\n*Showing 30 of {len(results)} calls*")

    return md.build()


@mcp.tool()
async def called_by(function_name: str) -> str:
    """
    Find functions that call this function (inbound dependents).

    Args:
        function_name: C++ function to analyze

    Returns:
        List of calling functions with file:line locations.
    """
    _ensure_loaded()
    if status := _cpp_status():
        return status

    callers = _state.cpp_extractor.get_callers(function_name, fuzzy=True)
    if not callers:
        return f"No inbound callers found for '{function_name}'."

    results = dedupe_by_key(callers, "caller")

    md = Markdown()
    md.h2(f"Called by: `{function_name}`")
    md.text("*Functions that call this (inbound dependents):*\n")

    for item in results[:30]:
        _format_call_item(md, item, "caller", "caller_file", "caller_line")

    if len(results) > 30:
        md.text(f"\n*Showing 30 of {len(results)} callers*")

    return md.build()


@mcp.tool()
async def impact(function_name: str, depth: int = 3) -> str:
    """
    Analyze the impact of modifying a function (transitive callers).

    Traces all code paths that depend on this function, useful for:
    - Security: Understanding vulnerability exposure
    - Refactoring: Knowing what might break
    - Testing: Identifying affected test coverage

    Args:
        function_name: C++ function to analyze
        depth: How many levels of callers to trace (default 3, max 5)

    Returns:
        Transitive callers grouped by depth, plus Python entry points if found.
    """
    _ensure_loaded()
    if status := _cpp_status():
        return status

    depth = min(max(depth, 1), 5)

    # BFS to find transitive callers
    visited = set()
    current_level = {function_name}
    callers_by_depth: Dict[int, List[Dict]] = {}

    for level in range(1, depth + 1):
        next_level = set()
        level_callers = []

        for func in current_level:
            for item in _state.cpp_extractor.get_callers(func, fuzzy=(level == 1)):
                caller = item["caller"]
                if caller not in visited and caller != function_name:
                    visited.add(caller)
                    next_level.add(caller)
                    level_callers.append(item)

        if level_callers:
            callers_by_depth[level] = level_callers
        current_level = next_level
        if not current_level:
            break

    if not callers_by_depth:
        return f"No callers found for '{function_name}'."

    md = Markdown()
    md.h2(f"Impact Analysis: `{function_name}`")
    md.text(f"*Tracing callers up to {depth} levels deep*\n")

    # Output by depth
    total = 0
    for level, callers in callers_by_depth.items():
        unique = dedupe_by_key(callers, "caller")
        total += len(unique)
        md.h3(f"Depth {level} ({len(unique)} callers)")

        for item in unique[:15]:
            _format_call_item(md, item, "caller", "caller_file", "caller_line")

        if len(unique) > 15:
            md.item(f"*... and {len(unique) - 15} more*")
        md.blank()

    # Find Python entry points
    python_entries = [
        {
            "python": b.get("python_name", c),
            "cpp": c,
            "dispatch": b.get("dispatch_key", ""),
        }
        for c in visited
        if c in _state.by_cpp_name
        for b in _state.by_cpp_name[c][:1]
    ]

    if python_entries:
        md.h3(f"Python Entry Points ({len(python_entries)} found)")
        md.text("*These Python APIs eventually call the target function:*\n")
        for entry in python_entries[:10]:
            dispatch = f" [{entry['dispatch']}]" if entry["dispatch"] else ""
            md.item(f"`{entry['python']}`{dispatch} → `{entry['cpp']}`")
        if len(python_entries) > 10:
            md.item(f"*... and {len(python_entries) - 10} more*")

    md.blank()
    md.text(
        f"**Total impact:** {total} functions across {len(callers_by_depth)} levels"
    )

    return md.build()


# --- Python Module Tools ---


@mcp.tool()
async def trace_module(module_name: str) -> str:
    """
    Trace a Python module (torch.nn.Linear, torch.optim.Adam, etc.)

    Args:
        module_name: Module or class name to trace (e.g., "Linear", "torch.nn.Linear")

    Returns:
        Module definition with methods, base classes, and file location.
    """
    _ensure_loaded()

    if not _state.py_classes:
        return "Python module analysis not available. Ensure PyTorch source is loaded."

    # Extract class name from qualified name
    search_name = module_name.split(".")[-1]

    # Find matching classes
    matches = []
    for name, classes in _state.py_classes.items():
        if search_name.lower() == name.lower():
            matches.extend(classes)
        elif search_name.lower() in name.lower():
            matches.extend(classes)

    if not matches:
        # Try fuzzy match
        similar = fuzzy_match(
            search_name, list(_state.py_classes.keys()), max_results=5
        )
        if similar:
            return (
                f"Module `{module_name}` not found. Did you mean: {', '.join(similar)}?"
            )
        return f"Module `{module_name}` not found."

    md = Markdown()
    md.h2(f"Module: `{module_name}`")

    for cls in matches[:3]:  # Limit to 3 matches
        md.h3(f"{cls.qualified_name}")
        path = _rel_path(cls.file_path)
        md.text(f"**File:** `{path}:{cls.line_number}`")

        if cls.bases:
            md.text(f"**Bases:** {', '.join(cls.bases)}")

        if cls.is_module:
            md.text("**Type:** torch.nn.Module")

        if cls.docstring:
            # Truncate docstring
            doc = (
                cls.docstring[:300] + "..."
                if len(cls.docstring) > 300
                else cls.docstring
            )
            md.blank().text(f"*{doc}*")

        if cls.methods:
            md.blank().text("**Methods:**")
            for method in cls.methods[:10]:
                sig = method.signature or "()"
                md.item(f"`{method.name}{sig}`")
            if len(cls.methods) > 10:
                md.item(f"*... and {len(cls.methods) - 10} more*")

        md.blank()

    if len(matches) > 3:
        md.text(f"*Showing 3 of {len(matches)} matches*")

    return md.build()


@mcp.tool()
async def list_modules(category: str = "nn") -> str:
    """
    List available PyTorch modules by category.

    Args:
        category: Category to list - "nn" (neural network layers), "optim" (optimizers),
                  "all" (all classes), or a search query

    Returns:
        List of available modules in the category.
    """
    _ensure_loaded()

    if not _state.py_classes:
        return "Python module analysis not available."

    md = Markdown()

    if category == "nn":
        md.h2("Neural Network Modules (torch.nn)")
        modules = sorted(_state.nn_modules, key=lambda x: x.name)
        md.text(f"Found {len(modules)} nn.Module subclasses\n")

        # Group by type
        layers = [
            m
            for m in modules
            if not any(x in m.name for x in ["Loss", "Container", "Sequential"])
        ]
        losses = [m for m in modules if "Loss" in m.name]
        containers = [
            m
            for m in modules
            if any(
                x in m.name
                for x in ["Container", "Sequential", "ModuleList", "ModuleDict"]
            )
        ]

        if layers:
            md.h3(f"Layers ({len(layers)})")
            for m in layers[:30]:
                md.item(f"`{m.name}` - {m.qualified_name}")
            if len(layers) > 30:
                md.item(f"*... and {len(layers) - 30} more*")

        if losses:
            md.h3(f"Loss Functions ({len(losses)})")
            for m in losses:
                md.item(f"`{m.name}`")

        if containers:
            md.h3(f"Containers ({len(containers)})")
            for m in containers:
                md.item(f"`{m.name}`")

    elif category == "optim":
        md.h2("Optimizers (torch.optim)")
        optim_classes = [
            cls
            for name, classes in _state.py_classes.items()
            for cls in classes
            if "optim" in cls.qualified_name.lower() and not name.startswith("_")
        ]
        for cls in sorted(optim_classes, key=lambda x: x.name)[:20]:
            md.item(f"`{cls.name}` - {cls.qualified_name}")

    elif category == "all":
        md.h2("All Python Classes")
        md.text(f"Total: {sum(len(v) for v in _state.py_classes.values())} classes\n")
        for name in sorted(_state.py_classes.keys())[:50]:
            md.item(f"`{name}`")
        if len(_state.py_classes) > 50:
            md.text(f"\n*Showing 50 of {len(_state.py_classes)} classes*")

    else:
        # Search query
        md.h2(f"Search: '{category}'")
        matches = []
        query_lower = category.lower()
        for name, classes in _state.py_classes.items():
            if query_lower in name.lower():
                matches.extend(classes)

        if matches:
            for cls in matches[:20]:
                path = _rel_path(cls.file_path)
                md.item(f"`{cls.name}` - `{path}:{cls.line_number}`")
            if len(matches) > 20:
                md.text(f"\n*Showing 20 of {len(matches)} matches*")
        else:
            md.text("No matches found.")

    return md.build()


# --- Test Infrastructure Tools ---


@mcp.tool()
async def find_similar_tests(query: str, limit: int = 20) -> str:
    """
    Find tests related to an operator, function, or concept.

    Searches test files, test classes, and test functions for matches.
    Useful for finding example tests when implementing new tests.

    Args:
        query: Search term (e.g., "add", "softmax", "backward", "cuda")
        limit: Maximum results to return (default 20)

    Returns:
        Matching tests with file locations and context.
    """
    _ensure_loaded("test")

    query_lower = query.lower()
    md = Markdown()
    md.h2(f"Tests matching: `{query}`")

    # Search test functions
    matching_funcs = []
    for func_name, locations in _state.test_functions.items():
        if query_lower in func_name.lower():
            for loc in locations:
                matching_funcs.append({
                    "name": func_name,
                    "class": loc.get("class"),
                    "file": loc["file"],
                    "line": loc["line"],
                })

    # Search test classes
    matching_classes = []
    for class_name, locations in _state.test_classes.items():
        if query_lower in class_name.lower():
            for loc in locations:
                matching_classes.append({
                    "name": class_name,
                    "file": loc["file"],
                    "line": loc["line"],
                    "bases": loc.get("bases", []),
                })

    # Search test files by path
    matching_files = []
    for file_path, info in _state.test_files.items():
        if query_lower in file_path.lower():
            matching_files.append({
                "path": file_path,
                "classes": len(info.get("classes", [])),
                "functions": len(info.get("functions", [])),
            })

    # Check OpInfo registry
    matching_opinfo = []
    for op_name, info in _state.opinfo_registry.items():
        if query_lower in op_name.lower():
            matching_opinfo.append(info)

    # Output results
    total = len(matching_funcs) + len(matching_classes) + len(matching_files) + len(matching_opinfo)

    if total == 0:
        return f"No tests found matching `{query}`. Try a broader search term."

    md.text(f"Found {total} matches\n")

    if matching_opinfo:
        md.h3(f"OpInfo Definitions ({len(matching_opinfo)})")
        md.text("*Operators with official test metadata:*\n")
        for info in matching_opinfo[:5]:
            md.item(f"`{info['name']}` → `{info['file']}:{info['line']}`")
        if len(matching_opinfo) > 5:
            md.item(f"*... and {len(matching_opinfo) - 5} more*")
        md.blank()

    if matching_funcs:
        md.h3(f"Test Functions ({len(matching_funcs)})")
        for func in matching_funcs[:limit]:
            class_prefix = f"{func['class']}." if func.get("class") else ""
            md.item(f"`{class_prefix}{func['name']}` → `{func['file']}:{func['line']}`")
        if len(matching_funcs) > limit:
            md.item(f"*... and {len(matching_funcs) - limit} more*")
        md.blank()

    if matching_classes:
        md.h3(f"Test Classes ({len(matching_classes)})")
        for cls in matching_classes[:10]:
            bases = f" ({', '.join(cls['bases'][:2])})" if cls.get("bases") else ""
            md.item(f"`{cls['name']}`{bases} → `{cls['file']}:{cls['line']}`")
        if len(matching_classes) > 10:
            md.item(f"*... and {len(matching_classes) - 10} more*")
        md.blank()

    if matching_files:
        md.h3(f"Test Files ({len(matching_files)})")
        for f in matching_files[:10]:
            md.item(f"`{f['path']}` ({f['classes']} classes, {f['functions']} tests)")
        if len(matching_files) > 10:
            md.item(f"*... and {len(matching_files) - 10} more*")

    return md.build()


@mcp.tool()
async def list_test_utils(category: str = "all") -> str:
    """
    List available PyTorch test utilities and infrastructure.

    Args:
        category: Filter by category - "all", "fixtures", "assertions", "decorators"

    Returns:
        Available test utilities with descriptions and file locations.
    """
    _ensure_loaded("test")

    md = Markdown()
    md.h2("PyTorch Test Utilities")

    # Key utility modules with descriptions
    utility_info = {
        "torch/testing/_internal/common_utils.py": {
            "name": "common_utils",
            "description": "Core test utilities: TestCase base, device helpers, dtype utilities",
            "key_items": ["TestCase", "run_tests", "instantiate_parametrized_tests", "IS_CUDA"],
        },
        "torch/testing/_internal/common_device_type.py": {
            "name": "common_device_type",
            "description": "Device-agnostic testing infrastructure",
            "key_items": ["instantiate_device_type_tests", "ops", "onlyCPU", "onlyCUDA"],
        },
        "torch/testing/_internal/common_dtype.py": {
            "name": "common_dtype",
            "description": "Data type testing utilities",
            "key_items": ["floating_types", "integral_types", "all_types_and_complex"],
        },
        "torch/testing/_internal/common_cuda.py": {
            "name": "common_cuda",
            "description": "CUDA-specific test utilities",
            "key_items": ["TEST_CUDA", "TEST_MULTIGPU", "TEST_CUDNN"],
        },
        "torch/testing/_internal/opinfo/core.py": {
            "name": "opinfo",
            "description": "Operator test info registry (OpInfo)",
            "key_items": ["OpInfo", "SampleInput", "DecorateInfo"],
        },
        "torch/testing/_comparison.py": {
            "name": "comparison",
            "description": "Tensor comparison and assertions",
            "key_items": ["assert_close", "assert_equal"],
        },
        "torch/testing/_internal/hypothesis_utils.py": {
            "name": "hypothesis_utils",
            "description": "Property-based testing with Hypothesis",
            "key_items": ["tensor_strategy", "dtype_strategy"],
        },
    }

    md.h3("Core Utilities")
    for path, info in utility_info.items():
        exists = path in _state.test_utilities or Path(_state.pytorch_source or "") / path
        status = "✓" if exists else "?"
        md.item(f"**{info['name']}** {status}")
        md.item(f"*{info['description']}*", 1)
        md.item(f"Key: `{', '.join(info['key_items'][:4])}`", 1)
        md.item(f"Path: `{path}`", 1)
        md.blank()

    # Statistics
    md.h3("Test Infrastructure Stats")
    if _state.test_files:
        md.item(f"Test files indexed: {len(_state.test_files)}")
        md.item(f"Test classes: {len(_state.test_classes)}")
        md.item(f"Test functions: {len(_state.test_functions)}")
        md.item(f"OpInfo definitions: {len(_state.opinfo_registry)}")
    else:
        md.text("*Test infrastructure not yet indexed*")

    # Common test patterns
    md.h3("Common Test Patterns")
    patterns = [
        ("Device-type tests", "`@instantiate_device_type_tests`", "Run tests across CPU/CUDA"),
        ("Parametrized tests", "`@parametrize`", "Run tests with multiple inputs"),
        ("OpInfo tests", "`@ops(op_db)`", "Test operators using OpInfo metadata"),
        ("Gradient check", "`gradcheck(fn, inputs)`", "Verify autograd correctness"),
        ("Assert close", "`torch.testing.assert_close(a, b)`", "Compare tensors with tolerance"),
    ]
    for name, code, desc in patterns:
        md.item(f"**{name}**: {code}")
        md.item(f"*{desc}*", 1)

    return md.build()


@mcp.tool()
async def test_file_info(file_path: str) -> str:
    """
    Get detailed information about a specific test file.

    Args:
        file_path: Path to test file (e.g., "test/test_torch.py" or just "test_torch")

    Returns:
        Test classes, functions, and infrastructure used in the file.
    """
    _ensure_loaded("test")

    # Find matching file
    query = file_path.lower()
    matches = []
    for path, info in _state.test_files.items():
        if query in path.lower():
            matches.append((path, info))

    if not matches:
        return f"No test file found matching `{file_path}`."

    md = Markdown()

    for path, info in matches[:3]:
        md.h2(f"Test File: `{path}`")

        if info.get("classes"):
            md.h3(f"Test Classes ({len(info['classes'])})")
            for cls in info["classes"][:10]:
                bases = f" extends {', '.join(cls['bases'][:2])}" if cls.get("bases") else ""
                md.item(f"`{cls['name']}`{bases} (line {cls['line']})")

        if info.get("functions"):
            md.h3(f"Test Functions ({len(info['functions'])})")
            for func in info["functions"][:20]:
                class_prefix = f"{func['class']}." if func.get("class") else ""
                md.item(f"`{class_prefix}{func['name']}` (line {func['line']})")
            if len(info["functions"]) > 20:
                md.item(f"*... and {len(info['functions']) - 20} more*")

        md.blank()

    if len(matches) > 3:
        md.text(f"*Showing 3 of {len(matches)} matching files*")

    return md.build()


# --- Server Entry Point ---


def run_server(
    pytorch_source: Optional[str] = None,
    index_path: Optional[str] = None,
    transport: str = "stdio",
):
    """Start MCP server."""
    source = pytorch_source or _auto_detect_pytorch()
    if source:
        _init_from_source(source)
    elif index_path:
        _load_from_json(index_path)
    else:
        log.warning(
            "No PyTorch source specified. Tools will return errors until data is loaded."
        )

    log.info("Starting TorchTalk MCP server...")
    mcp.run(transport=transport)
