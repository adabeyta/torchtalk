"""Tests for torchtalk.cli helpers."""

from __future__ import annotations

import json

from torchtalk.cli import (
    _format_coverage,
    _read_cache_stats,
    _read_coverage_from_cache,
)


class TestFormatCoverage:
    def test_orders_known_buckets_stably(self):
        cov = {"filtered": 3, "ok": 5, "parse_failed": 1, "unsupported_language": 2}
        assert (
            _format_coverage(cov)
            == "5 ok / 1 parse_failed / 2 unsupported_language / 3 filtered"
        )

    def test_appends_unknown_buckets_after_known(self):
        cov = {"ok": 1, "mystery": 7}
        assert _format_coverage(cov) == "1 ok / 7 mystery"

    def test_empty_returns_unknown(self):
        assert _format_coverage({}) == "unknown"

    def test_large_numbers_use_thousands_separator(self):
        assert _format_coverage({"ok": 12345}) == "12,345 ok"


class TestReadCoverageFromCache:
    def test_returns_coverage_when_present(self, tmp_path):
        path = tmp_path / "cg.json"
        path.write_text(json.dumps({"stats": {"coverage": {"ok": 10}}}))
        assert _read_coverage_from_cache(path) == {"ok": 10}

    def test_returns_none_for_missing_stats(self, tmp_path):
        path = tmp_path / "cg.json"
        path.write_text(json.dumps({"other": "data"}))
        assert _read_coverage_from_cache(path) is None

    def test_returns_none_for_missing_coverage(self, tmp_path):
        path = tmp_path / "cg.json"
        path.write_text(json.dumps({"stats": {"total_functions": 5}}))
        assert _read_coverage_from_cache(path) is None

    def test_returns_none_for_corrupt_json(self, tmp_path):
        path = tmp_path / "cg.json"
        path.write_text("{not valid json")
        assert _read_coverage_from_cache(path) is None

    def test_returns_none_for_missing_file(self, tmp_path):
        assert _read_coverage_from_cache(tmp_path / "missing.json") is None


class TestIndexUpdateExitCode:
    """cmd_index_update must translate uncovered_fail → non-zero exit."""

    def _fake_stats(self, uncovered_fail: bool) -> dict:
        cg = {
            "files_updated": 0,
            "header_affected_tus": 0,
            "files_removed": 0,
            "total_functions": 0,
            "uncovered_headers": 33 if uncovered_fail else 0,
            "uncovered_sample": [],
            "on_uncovered": "fail" if uncovered_fail else "warn",
        }
        if uncovered_fail:
            cg["uncovered_fail"] = True
        return {
            "cpp_files_changed": 0,
            "cpp_files_removed": 0,
            "headers_changed": 0,
            "yaml_changed": False,
            "bindings_total": 0,
            "cuda_kernels_total": 0,
            "baseline_snapshot": "foo",
            "baseline_commit": "abc1234",
            "call_graph": cg,
        }

    def test_returns_one_when_uncovered_fail(self, monkeypatch):
        from argparse import Namespace

        import torchtalk.cli as cli_mod
        from torchtalk import indexer

        monkeypatch.setattr(
            "torchtalk.config.resolve_pytorch_source", lambda: "/tmp/fake"
        )
        monkeypatch.setattr(
            indexer, "update_index", lambda *a, **kw: self._fake_stats(True)
        )
        args = Namespace(since="baseline", pytorch_source=None, on_uncovered="fail")
        assert cli_mod.cmd_index_update(args) == 1

    def test_returns_zero_when_no_uncovered_fail(self, monkeypatch):
        from argparse import Namespace

        import torchtalk.cli as cli_mod
        from torchtalk import indexer

        monkeypatch.setattr(
            "torchtalk.config.resolve_pytorch_source", lambda: "/tmp/fake"
        )
        monkeypatch.setattr(
            indexer, "update_index", lambda *a, **kw: self._fake_stats(False)
        )
        args = Namespace(since="baseline", pytorch_source=None, on_uncovered="warn")
        assert cli_mod.cmd_index_update(args) == 0


class TestReadCacheStats:
    def test_returns_full_stats_dict(self, tmp_path):
        path = tmp_path / "cg.json"
        path.write_text(
            json.dumps(
                {
                    "stats": {
                        "coverage": {"ok": 5},
                        "include_dirs_count": 42,
                        "total_functions": 100,
                    }
                }
            )
        )
        stats = _read_cache_stats(path)
        assert stats == {
            "coverage": {"ok": 5},
            "include_dirs_count": 42,
            "total_functions": 100,
        }

    def test_returns_none_for_missing_stats_key(self, tmp_path):
        path = tmp_path / "cg.json"
        path.write_text(json.dumps({"callees": {}}))
        assert _read_cache_stats(path) is None

    def test_returns_none_for_corrupt_json(self, tmp_path):
        path = tmp_path / "cg.json"
        path.write_text("not json")
        assert _read_cache_stats(path) is None
