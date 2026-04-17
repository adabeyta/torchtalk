"""Tests for CppCallGraphExtractor's per-file attribution and incremental update."""

from __future__ import annotations

import json

import pytest

from torchtalk.analysis import cpp_call_graph
from torchtalk.analysis.cpp_call_graph import (
    LIBCLANG_AVAILABLE,
    CppCallGraphExtractor,
)

pytestmark = pytest.mark.skipif(not LIBCLANG_AVAILABLE, reason="libclang not available")


@pytest.fixture
def extractor(tmp_path):
    return CppCallGraphExtractor(cache_dir=tmp_path)


def _seed(ext: CppCallGraphExtractor, records: dict[str, dict]) -> None:
    ext.file_records = {
        path: {
            "callees": rec.get("callees", {}),
            "callers": rec.get("callers", {}),
            "function_locations": rec.get("function_locations", {}),
        }
        for path, rec in records.items()
    }
    ext._rebuild_aggregates()


class TestRebuildAggregates:
    def test_merges_records(self, extractor):
        _seed(
            extractor,
            {
                "/a.cpp": {
                    "callees": {"foo": ["bar"]},
                    "callers": {"bar": ["foo"]},
                    "function_locations": {"foo": ("/a.cpp", 1)},
                },
                "/b.cpp": {
                    "callees": {"foo": ["baz"]},
                    "callers": {"baz": ["foo"]},
                    "function_locations": {"baz": ("/b.cpp", 5)},
                },
            },
        )
        assert extractor.callees["foo"] == {"bar", "baz"}
        assert extractor.callers["bar"] == {"foo"}
        assert extractor.callers["baz"] == {"foo"}
        assert extractor.function_locations["foo"] == ("/a.cpp", 1)
        assert extractor.function_locations["baz"] == ("/b.cpp", 5)
        assert extractor.processed_files == {"/a.cpp", "/b.cpp"}

    def test_removing_a_file_keeps_edges_from_other_contributors(self, extractor):
        """If two files both record the same edge, removing one must keep the edge."""
        _seed(
            extractor,
            {
                "/a.cpp": {"callees": {"foo": ["shared"]}},
                "/b.cpp": {"callees": {"foo": ["shared"]}},
            },
        )
        assert "shared" in extractor.callees["foo"]

        extractor.file_records.pop("/a.cpp")
        extractor._rebuild_aggregates()
        assert "shared" in extractor.callees["foo"]


class TestUpdateFiles:
    def test_removed_file_evicts_edges(self, extractor, monkeypatch):
        _seed(
            extractor,
            {
                "/gone.cpp": {
                    "callees": {"dead": ["target"]},
                    "callers": {"target": ["dead"]},
                    "function_locations": {"dead": ("/gone.cpp", 1)},
                },
                "/keep.cpp": {
                    "callees": {"alive": ["helper"]},
                    "function_locations": {"alive": ("/keep.cpp", 2)},
                },
            },
        )
        stats = extractor.update_files(entries=[], removed=["/gone.cpp"])
        assert stats["files_removed"] == 1
        assert "dead" not in extractor.callees
        assert "target" not in extractor.callers
        assert "alive" in extractor.callees

    def test_changed_file_rewrites_entries(self, extractor, monkeypatch):
        _seed(
            extractor,
            {
                "/x.cpp": {
                    "callees": {"old": ["target"]},
                    "function_locations": {"old": ("/x.cpp", 1)},
                },
            },
        )

        def fake_parse(args):
            path, _ = args
            return {
                "file": path,
                "callees": {"renamed": ["newtarget"]},
                "callers": {"newtarget": ["renamed"]},
                "function_locations": {"renamed": (path, 10)},
                "success": True,
                "error": None,
            }

        monkeypatch.setattr(cpp_call_graph, "_parse_single_file", fake_parse)

        stats = extractor.update_files(entries=[("/x.cpp", [])], removed=[])
        assert stats["files_updated"] == 1
        assert "old" not in extractor.callees
        assert extractor.callees["renamed"] == {"newtarget"}
        assert extractor.function_locations["renamed"] == ("/x.cpp", 10)

    def test_failed_parse_leaves_file_evicted(self, extractor, monkeypatch):
        _seed(
            extractor,
            {
                "/x.cpp": {
                    "callees": {"old": ["target"]},
                    "function_locations": {"old": ("/x.cpp", 1)},
                },
            },
        )

        def fake_parse(args):
            return {
                "file": args[0],
                "callees": {},
                "callers": {},
                "function_locations": {},
                "success": False,
                "error": "parse failed",
            }

        monkeypatch.setattr(cpp_call_graph, "_parse_single_file", fake_parse)

        stats = extractor.update_files(entries=[("/x.cpp", [])])
        assert stats["files_updated"] == 0
        assert stats["files_failed"] == 1
        assert "old" not in extractor.callees

    def test_tu_includes_recorded_on_update(self, extractor, monkeypatch, tmp_path):
        """Per-TU include sets are stored as repo-relative paths after a re-parse."""
        source_root = tmp_path / "src"
        (source_root / "aten").mkdir(parents=True)
        tu_path = source_root / "aten" / "Foo.cpp"
        tu_path.write_text("")
        hdr_path = source_root / "aten" / "Header.h"
        hdr_path.write_text("")

        def fake_parse(args):
            return {
                "file": str(tu_path),
                "callees": {"foo": ["bar"]},
                "callers": {"bar": ["foo"]},
                "function_locations": {"foo": (str(tu_path), 1)},
                "includes": [str(hdr_path)],
                "success": True,
                "error": None,
            }

        monkeypatch.setattr(cpp_call_graph, "_parse_single_file", fake_parse)
        extractor.update_files(
            entries=[(str(tu_path), [])], source_root=str(source_root)
        )

        assert "aten/Foo.cpp" in extractor.tu_includes
        assert extractor.tu_includes["aten/Foo.cpp"] == ["aten/Header.h"]

    def test_find_affected_tus_matches_header_changes(self, extractor):
        extractor.tu_includes = {
            "A.cpp": ["common.h", "a_only.h"],
            "B.cpp": ["common.h"],
            "C.cpp": ["unrelated.h"],
        }
        assert extractor.find_affected_tus({"common.h"}) == {"A.cpp", "B.cpp"}
        assert extractor.find_affected_tus({"a_only.h"}) == {"A.cpp"}
        assert extractor.find_affected_tus({"missing.h"}) == set()
        assert extractor.find_affected_tus(set()) == set()

    def test_known_headers_returns_union_across_tus(self, extractor):
        extractor.tu_includes = {
            "A.cpp": ["common.h", "a_only.h"],
            "B.cpp": ["common.h", "b_only.h"],
        }
        assert extractor.known_headers() == {"common.h", "a_only.h", "b_only.h"}

    def test_known_headers_empty_for_empty_cache(self, extractor):
        assert extractor.known_headers() == set()

    def test_header_inline_defs_attribute_to_header_not_tu(
        self, extractor, monkeypatch
    ):
        """When TU A.cpp observes an inline def in B.h, attribute it to B.h."""

        def fake_parse(args):
            tu, _ = args
            return {
                "file": tu,
                "callees": {
                    "caller_in_cpp": ["x"],  # defined in A.cpp
                    "inline_helper": ["y"],  # defined in B.h (inline)
                },
                "callers": {},
                "function_locations": {
                    "caller_in_cpp": ("/A.cpp", 1),
                    "inline_helper": ("/B.h", 5),
                },
                "success": True,
                "error": None,
            }

        monkeypatch.setattr(cpp_call_graph, "_parse_single_file", fake_parse)
        extractor.update_files(entries=[("/A.cpp", [])])

        assert "/A.cpp" in extractor.file_records
        assert "/B.h" in extractor.file_records
        assert "caller_in_cpp" in extractor.file_records["/A.cpp"]["callees"]
        assert "inline_helper" in extractor.file_records["/B.h"]["callees"]
        # Evicting A.cpp should NOT remove B.h's edges
        extractor.update_files(entries=[], removed=["/A.cpp"])
        assert "inline_helper" in extractor.callees
        assert "caller_in_cpp" not in extractor.callees

    def test_legacy_cache_raises(self, extractor):
        # Legacy: aggregates populated but no file_records
        extractor.callees["foo"].add("bar")
        extractor.function_locations["foo"] = ("/legacy.cpp", 1)
        extractor.file_records = {}

        with pytest.raises(RuntimeError, match="legacy cache"):
            extractor.update_files(entries=[], removed=["/foo.cpp"])


class TestPersistence:
    def test_save_and_load_round_trip_preserves_file_records(self, extractor, tmp_path):
        _seed(
            extractor,
            {
                "/a.cpp": {
                    "callees": {"foo": ["bar"]},
                    "callers": {"bar": ["foo"]},
                    "function_locations": {"foo": ("/a.cpp", 1)},
                },
            },
        )
        extractor.tu_includes = {"a.cpp": ["common.h"]}
        extractor.save_cache("test-key")

        ext2 = CppCallGraphExtractor(cache_dir=tmp_path)
        assert ext2.load_cache("test-key")
        assert ext2.file_records == extractor.file_records
        assert ext2.tu_includes == {"a.cpp": ["common.h"]}
        assert ext2.callees["foo"] == {"bar"}
        assert ext2.function_locations["foo"] == ("/a.cpp", 1)

    def test_load_from_path_reads_arbitrary_file(self, tmp_path):
        data = {
            "callees": {"foo": ["bar"]},
            "callers": {"bar": ["foo"]},
            "function_locations": {"foo": ["/a.cpp", 1]},
            "file_records": {
                "/a.cpp": {
                    "callees": {"foo": ["bar"]},
                    "callers": {"bar": ["foo"]},
                    "function_locations": {"foo": ["/a.cpp", 1]},
                }
            },
        }
        payload = tmp_path / "snapshot-cg.json"
        payload.write_text(json.dumps(data))

        ext = CppCallGraphExtractor(cache_dir=tmp_path)
        assert ext.load_from_path(payload)
        assert "/a.cpp" in ext.file_records
        assert ext.function_locations["foo"] == ("/a.cpp", 1)

    def test_load_legacy_cache_without_file_records(self, extractor, tmp_path):
        legacy_path = tmp_path / "legacy.json"
        legacy_path.write_text(
            json.dumps(
                {
                    "callees": {"foo": ["bar"]},
                    "callers": {"bar": ["foo"]},
                    "function_locations": {"foo": ["/a.cpp", 1]},
                }
            )
        )
        assert extractor.load_from_path(legacy_path)
        # Aggregates work
        assert extractor.callees["foo"] == {"bar"}
        # But file_records empty → update_files should raise
        assert extractor.file_records == {}
        with pytest.raises(RuntimeError, match="legacy cache"):
            extractor.update_files(entries=[], removed=["/x.cpp"])


def _extractor_contains_edges(ext, pairs):
    return all(callee in ext.callees[caller] for caller, callee in pairs)
