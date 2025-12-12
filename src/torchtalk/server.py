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

from .formatting import Markdown, relative_path, truncate

log = logging.getLogger(__name__)
mcp = FastMCP("torchtalk")

CACHE_DIR = Path.home() / ".cache" / "torchtalk"


# =============================================================================
# State Management
# =============================================================================


@dataclass
class ServerState:
    """Server state container."""

    bindings: List[Dict] = field(default_factory=list)
    cuda_kernels: List[Dict] = field(default_factory=list)
    native_functions: Dict[str, Dict] = field(default_factory=dict)
    derivatives: Dict[str, Dict] = field(default_factory=dict)
    native_implementations: Dict[str, List[Dict]] = field(default_factory=dict)

    # Indexes
    by_python_name: Dict[str, List[Dict]] = field(default_factory=dict)
    by_cpp_name: Dict[str, List[Dict]] = field(default_factory=dict)
    by_file: Dict[str, List[Dict]] = field(default_factory=dict)
    by_dispatch_key: Dict[str, List[Dict]] = field(default_factory=dict)
    by_binding_type: Dict[str, List[Dict]] = field(default_factory=dict)
    kernels_by_name: Dict[str, Dict] = field(default_factory=dict)

    # Paths
    pytorch_source: Optional[str] = None
    index_path: Optional[str] = None

    # C++ call graph
    cpp_extractor: Any = None
    cpp_building: bool = False
    cpp_thread: Any = None


_state = ServerState()


def _reset_state():
    """Reset server state."""
    global _state
    _state = ServerState()


# =============================================================================
# Caching
# =============================================================================


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


# =============================================================================
# YAML Parsing
# =============================================================================


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


# =============================================================================
# Implementation Finding
# =============================================================================


def _find_implementations(source: str, functions: Dict) -> Dict[str, List[Dict]]:
    """Find C++ implementations in source."""
    src = Path(source)
    search_dirs = [src / "aten/src/ATen/native", src / "torch/csrc"]

    # Target function names
    targets = set()
    for f in functions.values():
        targets.add(f["base_name"])
        targets.update(f.get("dispatch", {}).values())

    log.info(f"Searching for {len(targets)} implementations...")

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
            if "generated" in str(cpp_file).lower():
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


def _grep_fallback(source: str, name: str) -> List[Dict]:
    """Fallback grep search for function."""
    import subprocess

    src = Path(source)
    search_dirs = [src / "aten/src/ATen/native", src / "torch/csrc"]

    results = []
    pattern = rf"\b{re.escape(name)}\s*\([^;]*\)\s*{{"

    for search_dir in search_dirs:
        if not search_dir.exists():
            continue

        try:
            proc = subprocess.run(
                ["grep", "-rn", "-E", pattern, "--include=*.cpp", str(search_dir)],
                capture_output=True,
                text=True,
                timeout=10,
            )
            for line in proc.stdout.strip().split("\n")[:5]:
                if ":" not in line:
                    continue
                parts = line.split(":", 2)
                if len(parts) >= 2:
                    results.append(
                        {
                            "function_name": name,
                            "file_path": parts[0],
                            "line_number": int(parts[1]) if parts[1].isdigit() else 0,
                            "signature": (
                                truncate(parts[2], 80) if len(parts) > 2 else ""
                            ),
                        }
                    )
        except Exception:
            pass

    return results


def _fuzzy_find(name: str, data: Dict[str, Any]) -> Optional[List[Any]]:
    """Fuzzy match function name."""
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

    return None


# =============================================================================
# Index Building
# =============================================================================


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
        py_name = binding.get("python_name", "")
        cpp_name = binding.get("cpp_name", "")
        file_path = binding.get("file_path", "")
        dispatch = binding.get("dispatch_key", "")
        btype = binding.get("binding_type", "")

        if py_name:
            state.by_python_name.setdefault(py_name, []).append(binding)
        if cpp_name:
            state.by_cpp_name.setdefault(cpp_name, []).append(binding)
        if file_path:
            state.by_file.setdefault(file_path, []).append(binding)
        if dispatch:
            state.by_dispatch_key.setdefault(dispatch, []).append(binding)
        if btype:
            state.by_binding_type.setdefault(btype, []).append(binding)

    for kernel in state.cuda_kernels:
        name = kernel.get("name", "")
        if name:
            state.kernels_by_name[name] = kernel


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


# =============================================================================
# C++ Call Graph
# =============================================================================


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


# =============================================================================
# Initialization
# =============================================================================


def _ensure_loaded():
    """Ensure data is loaded."""
    if not _state.bindings and not _state.native_functions:
        raise RuntimeError("No data loaded. Start server with --pytorch-source.")


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
    _state.index_path = str(src)
    _init_cpp_call_graph(str(src))


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


# =============================================================================
# Helper Functions
# =============================================================================


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
    """Find similar function names."""
    name_lower = name.lower()
    matches = [k for k in _state.native_functions if name_lower in k.lower()]
    matches.sort(key=len)
    return matches[:limit]


def _rel_path(path: str) -> str:
    """Get relative path."""
    return relative_path(path, _state.pytorch_source)


# =============================================================================
# MCP Tools
# =============================================================================


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

    # Data
    if _state.bindings:
        md.bold("Bindings", f"{len(_state.bindings):,} loaded")
        md.item(f"CUDA kernels: {len(_state.cuda_kernels):,}", 1)
        md.item(f"Dispatch keys: {len(_state.by_dispatch_key)}", 1)
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

    # Tools
    md.h3("Available Tools")
    ready = "Ready" if _state.bindings else "Not ready"
    cpp_ready = (
        "Ready"
        if _state.cpp_extractor
        else ("Building..." if _state.cpp_building else "Not ready")
    )

    md.table(
        ["Tool", "Status", "Use For"],
        [
            ["`get_binding_chain`", ready, "Python→C++ mapping"],
            ["`get_native_function`", ready, "Operator definitions"],
            ["`get_dispatch_implementations`", ready, "CPU/CUDA backends"],
            ["`get_cuda_kernels`", ready, "GPU kernel info"],
            ["`search_bindings`", ready, "Find functions"],
            ["`get_cpp_callers`", cpp_ready, "Impact analysis"],
            ["`get_cpp_callees`", cpp_ready, "Dependency tracing"],
        ],
    )

    return md.build()


@mcp.tool()
async def get_binding_chain(function_name: str) -> str:
    """Get Python → C++ → file mapping for a function."""
    _ensure_loaded()

    md = Markdown()
    md.h2(f"Binding chain for `{function_name}`")

    # Native function definition
    native = _get_native_func(function_name)
    if native:
        md.h3("Native Function Definition (from native_functions.yaml)")
        md.code("Name", native.get("name", function_name))
        md.code("Signature", native.get("signature", "N/A"))

        if native.get("variants"):
            md.bold("Variants", native["variants"])
        if native.get("tags"):
            md.bold("Tags", ", ".join(str(t) for t in native["tags"]))

        dispatch = native.get("dispatch", {})
        if dispatch:
            md.blank().text("**Dispatch Configuration:**")
            for key, impl in sorted(dispatch.items()):
                md.item(f"{key}: `{impl}`")
        md.blank()

    # Native implementations
    base_name = native.get("base_name", function_name) if native else function_name
    impls = []

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
        md.h3("Native C++ Implementations")
        md.text("*Hand-written implementations in aten/src/ATen/native/:*\n")

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

            md.code(impl["function_name"], "")
            path = _rel_path(impl.get("file_path", ""))
            line = impl.get("line_number", "")
            md.item(f"File: `{path}:{line}`", 1)
            if impl.get("signature"):
                md.item(f"Signature: `{truncate(impl['signature'], 80)}`", 1)
            md.blank()

    # Registered implementations (from binding detector)
    bindings = _state.by_python_name.get(function_name, [])
    if not bindings:
        bindings = _state.by_cpp_name.get(function_name, [])
    if not bindings:
        found = _fuzzy_find(function_name, _state.by_python_name)
        if found:
            bindings = found

    if bindings:
        md.h3("Registered Implementations (TORCH_LIBRARY_IMPL)")
        by_dispatch: Dict[str, List] = {}
        for b in bindings:
            key = b.get("dispatch_key", "default")
            by_dispatch.setdefault(key, []).append(b)

        for dispatch_key, group in sorted(by_dispatch.items()):
            md.text(f"**{dispatch_key}:**")
            for b in group[:5]:
                path = _rel_path(b.get("file_path", ""))
                line = f":{b['line_number']}" if b.get("line_number") else ""
                md.item(f"`{b.get('cpp_name', 'N/A')}` → `{path}{line}`")
            md.blank()

    # Derivative formula
    deriv = _state.derivatives.get(function_name) or _state.derivatives.get(base_name)
    if deriv and deriv.get("gradients"):
        md.h3("Derivative Formula (from derivatives.yaml)")
        for input_name, formula in deriv["gradients"].items():
            md.item(f"`{input_name}`: `{truncate(formula, 60)}`")

    if not native and not impls and not bindings:
        similar = _similar_functions(function_name)
        if similar:
            md.text(f"\nFunction '{function_name}' not found. Similar:\n")
            for s in similar[:8]:
                md.item(s)

    return md.build()


@mcp.tool()
async def get_native_function(function_name: str) -> str:
    """Get native function definition from native_functions.yaml."""
    _ensure_loaded()

    native = _get_native_func(function_name)
    if not native:
        similar = _similar_functions(function_name)
        if similar:
            return f"Function '{function_name}' not found. Similar:\n" + "\n".join(
                f"  - {s}" for s in similar
            )
        return f"Function '{function_name}' not found."

    md = Markdown()
    md.h2(f"Native Function: `{native['name']}`")
    md.code("Signature", native["signature"])
    md.blank()

    if native.get("variants"):
        md.bold("Variants", native["variants"])
    if native.get("structured"):
        md.bold("Structured", "Yes")
    if native.get("structured_delegate"):
        md.bold("Structured Delegate", native["structured_delegate"])
    if native.get("tags"):
        md.bold("Tags", ", ".join(str(t) for t in native["tags"]))
    md.blank()

    dispatch = native.get("dispatch", {})
    if dispatch:
        md.h3("Dispatch Configuration")
        for key, impl in sorted(dispatch.items()):
            md.item(f"**{key}**: `{impl}`")
        md.blank()

    # Derivative
    deriv = _state.derivatives.get(native["name"]) or _state.derivatives.get(
        native.get("base_name", "")
    )
    if deriv:
        md.h3("Derivative Formula")
        md.code("Signature", deriv.get("signature", "N/A"))
        for k, v in deriv.get("gradients", {}).items():
            md.item(f"`{k}`: `{truncate(v, 60)}`")

    return md.build()


@mcp.tool()
async def get_dispatch_implementations(function_name: str) -> str:
    """Get backend implementations (CPU, CUDA, etc.) for a function."""
    _ensure_loaded()

    md = Markdown()
    md.h2(f"Dispatch implementations for `{function_name}`")

    # From binding detector
    bindings = _state.by_python_name.get(function_name, [])
    if not bindings:
        bindings = _state.by_cpp_name.get(function_name, [])
    if not bindings:
        found = _fuzzy_find(function_name, _state.by_python_name)
        if found:
            bindings = found

    if bindings:
        md.h3("From TORCH_LIBRARY_IMPL registrations")
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
            md.table(["Dispatch Key", "C++ Function", "File"], rows[:20])
        md.blank()

    # From native_functions.yaml
    native = _get_native_func(function_name)
    if native and native.get("dispatch"):
        md.h3("From native_functions.yaml dispatch config")
        for key, impl in sorted(native["dispatch"].items()):
            md.item(f"**{key}**: `{impl}`")

    if not bindings and not (native and native.get("dispatch")):
        return f"No dispatch implementations found for '{function_name}'."

    return md.build()


@mcp.tool()
async def get_cuda_kernels(function_name: str = "") -> str:
    """Get CUDA kernel information."""
    _ensure_loaded()

    md = Markdown()
    title = f"CUDA Kernels for '{function_name}'" if function_name else "CUDA Kernels"
    md.h2(title)

    kernels = _state.cuda_kernels
    if function_name:
        name_lower = function_name.lower()
        kernels = [k for k in kernels if name_lower in k.get("name", "").lower()]

    if not kernels:
        if function_name:
            return f"No CUDA kernels found matching '{function_name}'."
        return "No CUDA kernels found."

    md.text(f"Found {len(kernels)} kernel(s)\n")

    for kernel in kernels[:15]:
        md.code(kernel.get("name", "unnamed"), "")
        path = _rel_path(kernel.get("file_path", ""))
        line = kernel.get("line_number", "")
        md.item(f"File: `{path}:{line}`", 1)

        callers = kernel.get("callers", [])
        if callers:
            md.item(f"Called by: {', '.join(f'`{c}`' for c in callers[:3])}", 1)
        md.blank()

    if len(kernels) > 15:
        md.text(f"\n*Showing 15 of {len(kernels)} kernels*")

    return md.build()


@mcp.tool()
async def search_bindings(query: str, limit: int = 20) -> str:
    """Search bindings by name."""
    _ensure_loaded()

    query_lower = query.lower()
    matches = [
        b
        for b in _state.bindings
        if query_lower in b.get("python_name", "").lower()
        or query_lower in b.get("cpp_name", "").lower()
    ]

    if not matches:
        return f"No bindings found matching '{query}'."

    # Dedupe
    seen = set()
    unique = []
    for m in matches:
        key = (m.get("python_name"), m.get("cpp_name"))
        if key not in seen:
            seen.add(key)
            unique.append(m)

    md = Markdown()
    md.h2(f"Search results for '{query}'")
    md.text(f"Found {len(unique)} binding(s)\n")

    for b in unique[:limit]:
        dispatch = f" [{b['dispatch_key']}]" if b.get("dispatch_key") else ""
        md.bold(b.get("python_name", "N/A"), dispatch)
        md.item(f"`{b.get('cpp_name', 'N/A')}`", 1)
        md.item(f"`{_rel_path(b.get('file_path', ''))}`", 1)
        md.blank()

    if len(unique) > limit:
        md.text(f"\n*Showing {limit} of {len(unique)} results*")

    return md.build()


@mcp.tool()
async def get_cpp_callees(function_name: str) -> str:
    """Get C++ functions called by the given function."""
    _ensure_loaded()

    status = _cpp_status()
    if status:
        return status

    callees = _state.cpp_extractor.get_callees(function_name, fuzzy=True)
    if not callees:
        return f"No call data found for '{function_name}'."

    md = Markdown()
    md.h2(f"C++ functions called by '{function_name}'")

    by_caller: Dict[str, List] = {}
    for item in callees:
        by_caller.setdefault(item["caller"], []).append(item)

    for caller, items in list(by_caller.items())[:10]:
        md.h3(f"`{caller}`")
        md.text("Calls:")
        for item in items[:15]:
            callee = item["callee"]
            if item.get("callee_file"):
                path = _rel_path(item["callee_file"])
                line = f":{item['callee_line']}" if item.get("callee_line") else ""
                md.item(f"`{callee}` → `{path}{line}`")
            else:
                md.item(f"`{callee}`")
        if len(items) > 15:
            md.item(f"... and {len(items) - 15} more")
        md.blank()

    stats = _state.cpp_extractor.get_call_graph_data()["stats"]
    md.text(
        f"\n*Call graph: {stats['total_functions']} functions, {stats['total_call_edges']} edges*"
    )

    return md.build()


@mcp.tool()
async def get_cpp_callers(function_name: str) -> str:
    """Get C++ functions that call the given function (impact analysis)."""
    _ensure_loaded()

    status = _cpp_status()
    if status:
        return status

    callers = _state.cpp_extractor.get_callers(function_name, fuzzy=True)
    if not callers:
        return f"No callers found for '{function_name}'."

    md = Markdown()
    md.h2(f"C++ functions that call '{function_name}'")

    by_callee: Dict[str, List] = {}
    for item in callers:
        by_callee.setdefault(item["callee"], []).append(item)

    for callee, items in list(by_callee.items())[:10]:
        md.h3(f"`{callee}`")
        md.text("Called by:")
        for item in items[:15]:
            caller = item["caller"]
            if item.get("caller_file"):
                path = _rel_path(item["caller_file"])
                line = f":{item['caller_line']}" if item.get("caller_line") else ""
                md.item(f"`{caller}` → `{path}{line}`")
            else:
                md.item(f"`{caller}`")
        if len(items) > 15:
            md.item(f"... and {len(items) - 15} more")
        md.blank()

    stats = _state.cpp_extractor.get_call_graph_data()["stats"]
    md.text(
        f"\n*Call graph: {stats['total_functions']} functions, {stats['total_call_edges']} edges*"
    )

    return md.build()


@mcp.tool()
async def list_binding_types() -> str:
    """Get summary of all binding types."""
    _ensure_loaded()

    md = Markdown()
    md.h2("Binding Types Summary")
    md.bold("Total bindings", str(len(_state.bindings)))
    md.bold("CUDA kernels", str(len(_state.cuda_kernels)))
    md.blank()

    for btype, bindings in sorted(
        _state.by_binding_type.items(), key=lambda x: -len(x[1])
    ):
        md.h3(f"{btype} ({len(bindings)} bindings)")
        for b in bindings[:3]:
            md.item(f"`{b.get('python_name', 'N/A')}` → `{b.get('cpp_name', 'N/A')}`")
        if len(bindings) > 3:
            md.item(f"... and {len(bindings) - 3} more")
        md.blank()

    if _state.by_dispatch_key:
        md.h3("Dispatch Keys")
        for key, bindings in sorted(
            _state.by_dispatch_key.items(), key=lambda x: -len(x[1])
        ):
            md.item(f"**{key}**: {len(bindings)} implementations")

    return md.build()


@mcp.tool()
async def get_call_graph(function_name: str) -> str:
    """Get call relationships for a function."""
    _ensure_loaded()

    md = Markdown()
    md.h2(f"Call graph for '{function_name}'")

    # Check C++ call graph first
    if _state.cpp_extractor:
        callees = _state.cpp_extractor.get_callees(function_name, fuzzy=True)
        callers = _state.cpp_extractor.get_callers(function_name, fuzzy=True)

        if callees:
            md.h3("Calls (what this function calls)")
            seen = set()
            for item in callees[:20]:
                if item["callee"] not in seen:
                    seen.add(item["callee"])
                    md.item(f"`{item['callee']}`")

        if callers:
            md.h3("Called by (what calls this function)")
            seen = set()
            for item in callers[:20]:
                if item["caller"] not in seen:
                    seen.add(item["caller"])
                    md.item(f"`{item['caller']}`")

        if callees or callers:
            return md.build()

    return f"No call graph data found for '{function_name}'."


# =============================================================================
# Server Entry Point
# =============================================================================


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
