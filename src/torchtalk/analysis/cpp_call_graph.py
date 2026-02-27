#!/usr/bin/env python3
"""C++ call graph extraction using libclang with parallel processing."""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any
from collections import defaultdict
from multiprocessing import Pool, cpu_count

from .patterns import CPP_SEARCH_DIRS, should_exclude, should_include_dir
from .helpers import levenshtein_distance

log = logging.getLogger(__name__)


try:
    import clang.cindex  # noqa: F401
    LIBCLANG_AVAILABLE = True
except ImportError:
    LIBCLANG_AVAILABLE = False
    log.warning("libclang not available - C++ call graph extraction disabled")


def _parse_single_file(args: Tuple[str, List[str]]) -> Dict[str, Any]:
    """Parse a single file and extract call graph data (runs in subprocess)."""
    file_path, compile_args = args
    result = {"file": file_path, "callees": {}, "callers": {}, "function_locations": {}, "success": False, "error": None}

    try:
        from clang.cindex import Index, CursorKind, TranslationUnit
        index = Index.create()
        filtered_args = [a for a in compile_args if a.startswith(("-I", "-D", "-std"))]
        tu = index.parse(file_path, args=filtered_args, options=TranslationUnit.PARSE_INCOMPLETE)

        if tu is None:
            result["error"] = "parse failed"
            return result

        callees, callers, function_locations = defaultdict(set), defaultdict(set), {}

        def get_qualified_name(cursor) -> str:
            parts = []
            c = cursor
            while c is not None and c.kind != CursorKind.TRANSLATION_UNIT:
                if c.spelling:
                    parts.append(c.spelling)
                c = c.semantic_parent
            return "::".join(reversed(parts)) if parts else ""

        def extract_calls(cursor, current_function=None):
            if cursor.kind in (CursorKind.FUNCTION_DECL, CursorKind.CXX_METHOD) and cursor.is_definition():
                func_name = get_qualified_name(cursor)
                if func_name and cursor.location.file:
                    current_function = func_name
                    function_locations[func_name] = (str(cursor.location.file), cursor.location.line)

            if cursor.kind == CursorKind.CALL_EXPR and current_function:
                called = cursor.referenced
                if called:
                    called_name = get_qualified_name(called)
                    if called_name and called_name != current_function:
                        callees[current_function].add(called_name)
                        callers[called_name].add(current_function)

            for child in cursor.get_children():
                extract_calls(child, current_function)

        extract_calls(tu.cursor)
        result["callees"] = {k: list(v) for k, v in callees.items()}
        result["callers"] = {k: list(v) for k, v in callers.items()}
        result["function_locations"] = function_locations
        result["success"] = True
    except Exception as e:
        result["error"] = str(e)

    return result


class CppCallGraphExtractor:
    """Extracts C++ call graph using libclang with multiprocessing."""

    def __init__(self, cache_dir: Optional[Path] = None):
        if not LIBCLANG_AVAILABLE:
            raise RuntimeError(
                "libclang is not available. Install with: pip install libclang"
            )

        self.cache_dir = (
            cache_dir or Path.home() / ".cache" / "torchtalk" / "call_graph"
        )
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.callees: Dict[str, Set[str]] = defaultdict(set)
        self.callers: Dict[str, Set[str]] = defaultdict(set)
        self.function_locations: Dict[str, Tuple[str, int]] = {}
        self.processed_files: Set[str] = set()

    def extract_from_pytorch_parallel(
        self,
        pytorch_source: str,
        num_workers: Optional[int] = None,
        include_dirs: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Extract call graph from PyTorch using parallel processing.

        Args:
            pytorch_source: Path to PyTorch source directory
            num_workers: Number of parallel workers (default: 80% of CPU count)
            include_dirs: List of directory patterns to include (default: CPP_SEARCH_DIRS)
        """
        source = Path(pytorch_source)

        # Use default directories if not specified
        if include_dirs is None:
            include_dirs = CPP_SEARCH_DIRS

        # Find compile_commands.json
        compile_commands = source / "compile_commands.json"
        if not compile_commands.exists():
            compile_commands = source / "build" / "compile_commands.json"

        if not compile_commands.exists():
            raise FileNotFoundError(
                f"compile_commands.json not found in {pytorch_source}"
            )

        # Load compile database
        with open(compile_commands) as f:
            compile_db = json.load(f)

        # Filter to relevant files using configurable patterns
        entries = []
        skipped_excluded = 0
        for entry in compile_db:
            file_path = entry.get("file", "")

            # Must be a C++ file
            if not file_path.endswith(".cpp"):
                continue

            # Check inclusion based on directory patterns
            if not should_include_dir(file_path, include_dirs):
                continue

            # Apply exclusion patterns (tests, third_party, etc.)
            if should_exclude(file_path):
                skipped_excluded += 1
                continue

            command = entry.get("command", "")
            directory = entry.get("directory", "")

            if command:
                args = command.split()[1:]
            else:
                args = entry.get("arguments", [])[1:]

            if not os.path.isabs(file_path) and directory:
                file_path = os.path.join(directory, file_path)

            entries.append((file_path, args))

        log.info(f"Filtered to {len(entries)} files (excluded {skipped_excluded} test/third_party files)")

        log.info(f"Processing {len(entries)} C++ files with parallel libclang...")

        # Determine number of workers
        if num_workers is None:
            num_workers = max(1, int(cpu_count() * 0.8))

        log.info(f"Using {num_workers} parallel workers")

        # Process files in parallel
        with Pool(processes=num_workers) as pool:
            results = pool.map(_parse_single_file, entries)

        # Merge results
        success_count = 0
        for result in results:
            if result["success"]:
                success_count += 1

                # Merge callees
                for func, called in result["callees"].items():
                    self.callees[func].update(called)

                # Merge callers
                for func, callers_list in result["callers"].items():
                    self.callers[func].update(callers_list)

                # Merge function locations
                self.function_locations.update(result["function_locations"])
                self.processed_files.add(result["file"])

        log.info(f"Completed: {success_count}/{len(entries)} files succeeded")
        log.info(
            f"Extracted {len(self.function_locations)} functions, "
            f"{sum(len(v) for v in self.callees.values())} call edges"
        )

        return self.get_call_graph_data()

    def get_call_graph_data(self) -> Dict[str, Any]:
        return {
            "callees": {k: list(v) for k, v in self.callees.items()},
            "callers": {k: list(v) for k, v in self.callers.items()},
            "function_locations": self.function_locations,
            "stats": {
                "total_functions": len(self.function_locations),
                "total_call_edges": sum(len(v) for v in self.callees.values()),
                "files_processed": len(self.processed_files),
            },
        }

    def _get_relations(
        self, function_name: str, direction: str, fuzzy: bool = True
    ) -> List[Dict[str, Any]]:
        """Get call relations (callees or callers) for a function.

        Args:
            function_name: Function to look up
            direction: "callees" or "callers"
            fuzzy: Whether to use fuzzy matching
        """
        results = []
        matches = self._find_matching_functions(function_name, fuzzy)
        source_dict = self.callees if direction == "callees" else self.callers

        # Field names depend on direction
        if direction == "callees":
            source_key, target_key, file_key, line_key = "caller", "callee", "callee_file", "callee_line"
        else:
            source_key, target_key, file_key, line_key = "callee", "caller", "caller_file", "caller_line"

        for func in matches:
            for target in source_dict.get(func, set()):
                loc = self.function_locations.get(target, (None, None))
                results.append({
                    source_key: func,
                    target_key: target,
                    file_key: loc[0],
                    line_key: loc[1],
                })
        return results

    def get_callees(self, function_name: str, fuzzy: bool = True) -> List[Dict[str, Any]]:
        """Get functions that this function calls (outbound)."""
        return self._get_relations(function_name, "callees", fuzzy)

    def get_callers(self, function_name: str, fuzzy: bool = True) -> List[Dict[str, Any]]:
        """Get functions that call this function (inbound)."""
        return self._get_relations(function_name, "callers", fuzzy)

    def _find_matching_functions(self, name: str, fuzzy: bool) -> List[str]:
        all_funcs = (
            set(self.callees.keys())
            | set(self.callers.keys())
            | set(self.function_locations.keys())
        )

        # 1. Exact match
        if name in all_funcs:
            return [name]

        if not fuzzy:
            return []

        # 2. Namespace suffix match (at::native::gemm matches "gemm")
        matches = [f for f in all_funcs if f.endswith("::" + name) or f == name]
        if matches:
            return matches

        # 3. Case-insensitive substring match
        name_lower = name.lower()
        substring_matches = [f for f in all_funcs if name_lower in f.lower()]

        # 4. Extract base name from namespaced functions for better matching
        # e.g., "gemm" should match "at::native::cpublas::gemm"
        base_name_matches = []
        for f in all_funcs:
            # Get the last part after ::
            base = f.split("::")[-1] if "::" in f else f
            if base.lower() == name_lower:
                base_name_matches.append(f)

        # 5. Levenshtein distance for typo tolerance (only if name is long enough)
        levenshtein_matches = []
        if len(name) >= 4 and not substring_matches and not base_name_matches:
            for f in all_funcs:
                base = f.split("::")[-1] if "::" in f else f
                if abs(len(base) - len(name)) <= 3:
                    dist = levenshtein_distance(name_lower, base.lower())
                    if dist <= max(2, len(name) // 3):
                        levenshtein_matches.append((dist, f))
            levenshtein_matches.sort(key=lambda x: x[0])
            levenshtein_matches = [f for _, f in levenshtein_matches[:10]]

        # Combine results with priority: base_name > substring > levenshtein
        result = []
        seen = set()
        for matches_list in [base_name_matches, substring_matches, levenshtein_matches]:
            for m in matches_list:
                if m not in seen:
                    seen.add(m)
                    result.append(m)

        # Sort by length (prefer shorter/more specific matches)
        result.sort(key=len)
        return result[:20]

    def save_cache(self, cache_key: str) -> Path:
        cache_path = self.cache_dir / f"{cache_key}.json"
        data = self.get_call_graph_data()
        with open(cache_path, "w") as f:
            json.dump(data, f)
        log.info(f"Saved call graph cache to {cache_path}")
        return cache_path

    def load_cache(self, cache_key: str) -> bool:
        cache_path = self.cache_dir / f"{cache_key}.json"
        if not cache_path.exists():
            return False

        try:
            with open(cache_path) as f:
                data = json.load(f)

            self.callees = defaultdict(
                set, {k: set(v) for k, v in data.get("callees", {}).items()}
            )
            self.callers = defaultdict(
                set, {k: set(v) for k, v in data.get("callers", {}).items()}
            )
            self.function_locations = data.get("function_locations", {})
            self.function_locations = {
                k: tuple(v) if isinstance(v, list) else v
                for k, v in self.function_locations.items()
            }
            log.info(f"Loaded call graph cache from {cache_path}")
            return True
        except Exception as e:
            log.warning(f"Failed to load cache: {e}")
            return False
