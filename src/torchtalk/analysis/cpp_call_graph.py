#!/usr/bin/env python3
"""C++ call graph extraction using libclang with parallel processing."""

import json
import logging
import os
from collections import defaultdict
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Any

from .helpers import levenshtein_distance
from .patterns import CPP_SEARCH_DIRS, should_exclude, should_include_dir

log = logging.getLogger(__name__)


try:
    import clang.cindex  # noqa: F401

    LIBCLANG_AVAILABLE = True
except ImportError:
    LIBCLANG_AVAILABLE = False
    log.warning("libclang not available - C++ call graph extraction disabled")


def _rel_to_root(path: str, source_root: str) -> str | None:
    """Return path relative to source_root, or None if outside the tree."""
    if not path:
        return None
    try:
        resolved = os.path.realpath(path)
        root = os.path.realpath(source_root)
    except OSError:
        return None
    if resolved == root or resolved.startswith(root + os.sep):
        return os.path.relpath(resolved, root)
    return None


def _collect_include_dirs(compile_db: list[dict], source_root: str) -> list[str]:
    """Union of `-I` dirs under source_root, repo-relative, sorted.

    Handles both `-Ipath` and `-I path`. External / absolute-not-under-root
    entries are dropped. Output feeds diagnostics only — does not invalidate.
    """
    dirs: set[str] = set()
    for entry in compile_db:
        cmd = entry.get("command", "")
        args = cmd.split() if cmd else entry.get("arguments", [])
        i = 0
        while i < len(args):
            a = args[i]
            if a == "-I" and i + 1 < len(args):
                candidate = args[i + 1]
                i += 2
            elif a.startswith("-I") and len(a) > 2:
                candidate = a[2:]
                i += 1
            else:
                i += 1
                continue
            if not candidate:
                continue
            rel = _rel_to_root(candidate, source_root)
            if rel is not None:
                dirs.add(rel)
    return sorted(dirs)


def _parse_single_file(args: tuple[str, list[str]]) -> dict[str, Any]:
    """Parse a single file and extract call graph data (runs in subprocess)."""
    file_path, compile_args = args
    result = {
        "file": file_path,
        "callees": {},
        "callers": {},
        "function_locations": {},
        "includes": [],
        "success": False,
        "error": None,
    }

    try:
        from clang.cindex import CursorKind, Index, TranslationUnit

        index = Index.create()
        filtered_args = [a for a in compile_args if a.startswith(("-I", "-D", "-std"))]
        tu = index.parse(
            file_path, args=filtered_args, options=TranslationUnit.PARSE_INCOMPLETE
        )

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
            if (
                cursor.kind in (CursorKind.FUNCTION_DECL, CursorKind.CXX_METHOD)
                and cursor.is_definition()
            ):
                func_name = get_qualified_name(cursor)
                if func_name and cursor.location.file:
                    current_function = func_name
                    function_locations[func_name] = (
                        str(cursor.location.file),
                        cursor.location.line,
                    )

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

        includes = []
        seen = set()
        for fi in tu.get_includes():
            inc_file = fi.include
            if inc_file is None:
                continue
            name = str(inc_file.name)
            if name and name not in seen:
                seen.add(name)
                includes.append(name)

        result["callees"] = {k: list(v) for k, v in callees.items()}
        result["callers"] = {k: list(v) for k, v in callers.items()}
        result["function_locations"] = function_locations
        result["includes"] = includes
        result["success"] = True
    except Exception as e:
        result["error"] = str(e)

    return result


class CppCallGraphExtractor:
    """Extracts C++ call graph using libclang with multiprocessing."""

    def __init__(self, cache_dir: Path | None = None):
        if not LIBCLANG_AVAILABLE:
            raise RuntimeError(
                "libclang is not available. Install with: pip install libclang"
            )

        self.cache_dir = (
            cache_dir or Path.home() / ".cache" / "torchtalk" / "call_graph"
        )
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.callees: dict[str, set[str]] = defaultdict(set)
        self.callers: dict[str, set[str]] = defaultdict(set)
        self.function_locations: dict[str, tuple[str, int]] = {}
        self.processed_files: set[str] = set()

        # Per-file attribution. Lets incremental updates evict contributions
        # from one file without losing edges still supplied by another.
        self.file_records: dict[str, dict] = {}

        # Per-TU include sets (repo-relative paths). Powers cross-TU
        # invalidation when a header changes: find every TU whose includes
        # intersect the changed header set, and re-parse those TUs.
        # Keys are repo-relative TU paths (for checkout portability).
        self.tu_includes: dict[str, list[str]] = {}

        # Repo-relative TU path → one of:
        # ok | parse_failed | unsupported_language | filtered
        self.tu_status: dict[str, str] = {}

        # Union of -I dirs seen in compile_commands.json, repo-relative, sorted.
        self.include_dirs: list[str] = []

    def extract_from_pytorch_parallel(
        self,
        pytorch_source: str,
        num_workers: int | None = None,
        include_dirs: list[str] | None = None,
    ) -> dict[str, Any]:
        """Extract call graph from PyTorch using parallel processing.

        Args:
            pytorch_source: Path to PyTorch source directory
            num_workers: Number of parallel workers (default: 80% of CPU count)
            include_dirs: Directory patterns to include
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

        self.include_dirs = _collect_include_dirs(compile_db, str(source))

        # Filter to relevant files using configurable patterns
        entries = []
        for entry in compile_db:
            file_path = entry.get("file", "")
            directory = entry.get("directory", "")
            if not os.path.isabs(file_path) and directory:
                file_path = os.path.join(directory, file_path)

            tu_rel = _rel_to_root(file_path, str(source))
            if tu_rel is None:
                continue

            if not file_path.endswith(".cpp"):
                self.tu_status[tu_rel] = "unsupported_language"
                continue
            if not should_include_dir(file_path, include_dirs):
                self.tu_status[tu_rel] = "filtered"
                continue
            if should_exclude(file_path):
                self.tu_status[tu_rel] = "filtered"
                continue

            command = entry.get("command", "")
            args = command.split()[1:] if command else entry.get("arguments", [])[1:]
            entries.append((file_path, args))

        filtered_count = sum(1 for s in self.tu_status.values() if s == "filtered")
        log.info(
            f"Filtered to {len(entries)} files "
            f"(excluded {filtered_count} test/third_party)"
        )

        log.info(f"Processing {len(entries)} C++ files with parallel libclang...")

        if num_workers is None:
            num_workers = max(1, int(cpu_count() * 0.8))
        log.info(f"Using {num_workers} parallel workers")

        with Pool(processes=num_workers) as pool:
            results = pool.map(_parse_single_file, entries)

        # Merge results, attributing each record to the file where the
        # function (or caller) is defined — not the TU that observed it.
        # Without this, inline defs in headers get duplicated once per TU.
        success_count = 0
        for result in results:
            tu_rel = _rel_to_root(result.get("file", ""), str(source))
            if result["success"]:
                success_count += 1
                if tu_rel:
                    self.tu_status[tu_rel] = "ok"
                self._merge_parse_result(result, source_root=str(source))
            elif tu_rel:
                self.tu_status[tu_rel] = "parse_failed"

        self._rebuild_aggregates()

        log.info(f"Completed: {success_count}/{len(entries)} files succeeded")
        log.info(
            f"Extracted {len(self.function_locations)} functions, "
            f"{sum(len(v) for v in self.callees.values())} call edges"
        )

        return self.get_call_graph_data()

    def get_call_graph_data(self) -> dict[str, Any]:
        # Intern tu_includes paths: PyTorch has ~8k unique headers but ~2M
        # TU→header references. Storing indices into a shared path list cuts
        # the cache from ~120 MB to ~15 MB on PyTorch.
        path_to_idx: dict[str, int] = {}
        paths_list: list[str] = []
        tu_includes_idx: dict[str, list[int]] = {}
        for tu, includes in self.tu_includes.items():
            indices = []
            for p in includes:
                idx = path_to_idx.get(p)
                if idx is None:
                    idx = len(paths_list)
                    path_to_idx[p] = idx
                    paths_list.append(p)
                indices.append(idx)
            tu_includes_idx[tu] = indices

        return {
            "callees": {k: list(v) for k, v in self.callees.items()},
            "callers": {k: list(v) for k, v in self.callers.items()},
            "function_locations": self.function_locations,
            "file_records": self.file_records,
            "tu_includes_paths": paths_list,
            "tu_includes_idx": tu_includes_idx,
            "tu_status": dict(self.tu_status),
            "include_dirs": list(self.include_dirs),
            "stats": {
                "total_functions": len(self.function_locations),
                "total_call_edges": sum(len(v) for v in self.callees.values()),
                "files_processed": len(self.processed_files),
                "coverage": self.coverage_summary(),
                "include_dirs_count": len(self.include_dirs),
            },
        }

    def _record_for(self, owner_file: str) -> dict[str, Any]:
        """Get or create the file_records entry for a defining file."""
        return self.file_records.setdefault(
            owner_file,
            {"callees": {}, "callers": {}, "function_locations": {}},
        )

    def _merge_parse_result(
        self, result: dict[str, Any], source_root: str | None = None
    ) -> None:
        """Attribute a TU's parse output to the file where each function is defined.

        Each edge (caller → callee) lives in the record of the file that
        DEFINES the caller. Function locations live in their own defining
        file's record. This keeps file_records evictable per-file without
        duplicating header inline definitions across every TU that includes
        them.

        When source_root is given, the TU's include set is stored (repo-relative)
        so a later header change can invalidate this TU.
        """
        locations = result.get("function_locations", {})
        callees = result.get("callees", {})

        for func, loc in locations.items():
            if not loc:
                continue
            loc_tuple = tuple(loc) if isinstance(loc, list) else loc
            owner = loc_tuple[0]
            self._record_for(owner)["function_locations"][func] = loc_tuple

        for caller, callee_list in callees.items():
            loc = locations.get(caller)
            if not loc:
                continue
            owner = loc[0]
            rec = self._record_for(owner)

            out = rec["callees"].setdefault(caller, [])
            out_set = set(out)
            for c in callee_list:
                if c not in out_set:
                    out.append(c)
                    out_set.add(c)

            for c in callee_list:
                inc = rec["callers"].setdefault(c, [])
                if caller not in inc:
                    inc.append(caller)

        if source_root:
            tu_rel = _rel_to_root(result.get("file", ""), source_root)
            if tu_rel is not None:
                self.tu_includes[tu_rel] = sorted(
                    {
                        rel
                        for h in result.get("includes", [])
                        if (rel := _rel_to_root(h, source_root)) is not None
                    }
                )

    def _rebuild_aggregates(self) -> None:
        """Rebuild aggregates and processed_files from file_records."""
        self.callees = defaultdict(set)
        self.callers = defaultdict(set)
        self.function_locations = {}
        self.processed_files = set()

        for file_path, rec in self.file_records.items():
            for func, called in rec.get("callees", {}).items():
                self.callees[func].update(called)
            for func, callers_list in rec.get("callers", {}).items():
                self.callers[func].update(callers_list)
            self.function_locations.update(rec.get("function_locations", {}))
            self.processed_files.add(file_path)

    def update_files(
        self,
        entries: list[tuple[str, list[str]]],
        removed: list[str] | None = None,
        num_workers: int | None = None,
        source_root: str | None = None,
    ) -> dict[str, int]:
        """Re-parse the given files and evict stale per-file contributions.

        entries: (file_path, compile_args) tuples for files to re-parse.
        removed: absolute paths whose contributions should be evicted with no
                 re-parse (for deleted files).
        source_root: if given, per-TU include sets are updated for the re-parsed
                 TUs so subsequent header changes can invalidate them.

        Raises RuntimeError if the extractor has no per-file attribution
        (loaded from a legacy cache) — caller should fall back to a full build.
        """
        if self.file_records is None or (
            not self.file_records and self.function_locations
        ):
            raise RuntimeError(
                "Call graph has no per-file attribution (legacy cache). "
                "Rebuild with 'torchtalk index build'."
            )

        for f in removed or []:
            self.file_records.pop(f, None)
            if source_root and (rel := _rel_to_root(f, source_root)):
                self.tu_includes.pop(rel, None)
                self.tu_status.pop(rel, None)

        for file, _ in entries:
            self.file_records.pop(file, None)
            if source_root and (rel := _rel_to_root(file, source_root)):
                self.tu_includes.pop(rel, None)

        results: list[dict[str, Any]] = []
        if entries:
            if num_workers is None:
                num_workers = max(1, int(cpu_count() * 0.8))
            if len(entries) > 1 and num_workers > 1:
                with Pool(processes=num_workers) as pool:
                    results = pool.map(_parse_single_file, entries)
            else:
                results = [_parse_single_file(e) for e in entries]

            for result in results:
                rel = (
                    _rel_to_root(result.get("file", ""), source_root)
                    if source_root
                    else None
                )
                if result["success"]:
                    if rel:
                        self.tu_status[rel] = "ok"
                    self._merge_parse_result(result, source_root=source_root)
                elif rel:
                    self.tu_status[rel] = "parse_failed"

        self._rebuild_aggregates()

        return {
            "files_updated": sum(1 for r in results if r["success"]),
            "files_failed": sum(1 for r in results if not r["success"]),
            "files_removed": len(removed or []),
            "total_functions": len(self.function_locations),
        }

    def find_affected_tus(self, changed_files: set[str]) -> set[str]:
        """Return repo-relative TU paths whose include set intersects changed_files.

        changed_files must be repo-relative paths. Returns TU paths whose
        recorded include set contains any of them. Use this to pick TUs to
        re-parse when a header file changes.
        """
        if not changed_files:
            return set()
        affected: set[str] = set()
        for tu_rel, includes in self.tu_includes.items():
            if any(h in changed_files for h in includes):
                affected.add(tu_rel)
        return affected

    def known_headers(self) -> set[str]:
        """Return every header path present in any TU's recorded include set.

        Used to detect incomplete baseline coverage: a changed header outside
        this set can't be resolved to an affected TU, which may indicate a
        baseline parse failure, a generated header added since baseline, or a
        truly unused header.
        """
        result: set[str] = set()
        for v in self.tu_includes.values():
            result.update(v)
        return result

    def coverage_summary(self) -> dict[str, int]:
        """Count TUs per status bucket from tu_status."""
        summary: dict[str, int] = defaultdict(int)
        for status in self.tu_status.values():
            summary[status] += 1
        return dict(summary)

    def load_from_path(self, cache_path: Path) -> bool:
        """Load a call graph JSON from an arbitrary path (not in cache_dir)."""
        if not cache_path.exists():
            return False
        try:
            with open(cache_path) as f:
                data = json.load(f)
            self._apply_loaded_data(data)
            log.info(f"Loaded call graph from {cache_path}")
            return True
        except Exception as e:
            log.warning(f"Failed to load call graph from {cache_path}: {e}")
            return False

    def _get_relations(
        self, function_name: str, direction: str, fuzzy: bool = True
    ) -> list[dict[str, Any]]:
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
            source_key, target_key, file_key, line_key = (
                "caller",
                "callee",
                "callee_file",
                "callee_line",
            )
        else:
            source_key, target_key, file_key, line_key = (
                "callee",
                "caller",
                "caller_file",
                "caller_line",
            )

        for func in matches:
            for target in source_dict.get(func, set()):
                loc = self.function_locations.get(target, (None, None))
                results.append(
                    {
                        source_key: func,
                        target_key: target,
                        file_key: loc[0],
                        line_key: loc[1],
                    }
                )
        return results

    def get_callees(
        self, function_name: str, fuzzy: bool = True
    ) -> list[dict[str, Any]]:
        """Get functions that this function calls (outbound)."""
        return self._get_relations(function_name, "callees", fuzzy)

    def get_callers(
        self, function_name: str, fuzzy: bool = True
    ) -> list[dict[str, Any]]:
        """Get functions that call this function (inbound)."""
        return self._get_relations(function_name, "callers", fuzzy)

    def _find_matching_functions(self, name: str, fuzzy: bool) -> list[str]:
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
            self._apply_loaded_data(data)
            log.info(f"Loaded call graph cache from {cache_path}")
            return True
        except Exception as e:
            log.warning(f"Failed to load cache: {e}")
            return False

    def _apply_loaded_data(self, data: dict[str, Any]) -> None:
        """Populate aggregates + file_records from a loaded JSON payload."""
        self.callees = defaultdict(
            set, {k: set(v) for k, v in data.get("callees", {}).items()}
        )
        self.callers = defaultdict(
            set, {k: set(v) for k, v in data.get("callers", {}).items()}
        )
        self.function_locations = {
            k: tuple(v) if isinstance(v, list) else v
            for k, v in data.get("function_locations", {}).items()
        }
        raw_records = data.get("file_records") or {}
        self.file_records = {
            path: {
                "callees": rec.get("callees", {}),
                "callers": rec.get("callers", {}),
                "function_locations": {
                    k: tuple(v) if isinstance(v, list) else v
                    for k, v in rec.get("function_locations", {}).items()
                },
            }
            for path, rec in raw_records.items()
        }
        # Prefer interned form (current); fall back to flat list (early caches).
        paths_list = data.get("tu_includes_paths")
        idx_map = data.get("tu_includes_idx")
        if paths_list is not None and idx_map is not None:
            self.tu_includes = {
                tu: [paths_list[i] for i in indices] for tu, indices in idx_map.items()
            }
        else:
            self.tu_includes = {
                k: list(v) for k, v in (data.get("tu_includes") or {}).items()
            }
        self.tu_status = dict(data.get("tu_status") or {})
        self.include_dirs = list(data.get("include_dirs") or [])
        self.processed_files = set(self.file_records)
