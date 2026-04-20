"""Tests for indexer data structures and fuzzy matching."""

import json
from pathlib import Path

import pytest

from torchtalk import indexer, snapshots
from torchtalk.indexer import ServerState, _build_indexes, _fuzzy_find, update_index


class TestServerState:
    def test_default_empty(self):
        state = ServerState()
        assert state.bindings == []
        assert state.native_functions == {}
        assert state.pytorch_source is None

    def test_build_indexes(self):
        state = ServerState()
        state.bindings = [
            {"python_name": "add", "cpp_name": "at::add", "dispatch_key": "CPU"},
            {"python_name": "add", "cpp_name": "at::add", "dispatch_key": "CUDA"},
            {"python_name": "mul", "cpp_name": "at::mul", "dispatch_key": "CPU"},
        ]
        _build_indexes(state)

        assert "add" in state.by_python_name
        assert len(state.by_python_name["add"]) == 2
        assert "at::add" in state.by_cpp_name
        assert "CPU" in state.by_dispatch_key
        assert "CUDA" in state.by_dispatch_key


class TestFuzzyFind:
    def test_exact_match(self):
        data = {"relu": [{"name": "relu"}]}
        result = _fuzzy_find("relu", data)
        assert result is not None
        assert result[0]["name"] == "relu"

    def test_suffix_match(self):
        data = {"at::native::relu": [{"name": "relu"}]}
        result = _fuzzy_find("relu", data)
        assert result is not None

    def test_contains_match(self):
        data = {"mkldnn_relu": [{"name": "mkldnn_relu"}]}
        result = _fuzzy_find("relu", data)
        assert result is not None

    def test_no_match(self):
        data = {"softmax": [{"name": "softmax"}]}
        result = _fuzzy_find("nonexistent_function_xyz", data)
        assert result is None

    def test_returns_list(self):
        data = {"relu": {"name": "relu"}}
        result = _fuzzy_find("relu", data)
        assert isinstance(result, list)

    def test_levenshtein_match(self):
        data = {"softmax": [{"name": "softmax"}]}
        result = _fuzzy_find("sofmax", data)
        assert result is not None


class TestUpdateIndex:
    def test_raises_when_snapshot_has_no_commit(self, tmp_path, monkeypatch):
        cache = tmp_path / "cache"
        snap_dir = cache / "snapshots" / "baseline"
        snap_dir.mkdir(parents=True)
        (snap_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "name": "baseline",
                    "created": "2026-01-01T00:00:00+00:00",
                    "pytorch_source": str(tmp_path / "src"),
                    "source_fingerprint": "deadbeef",
                    "git_commit": None,
                    "bindings_size": 0,
                    "bindings_sha256": "x",
                    "content_fingerprint": None,
                    "schema_version": 2,
                }
            )
        )
        (snap_dir / "bindings.json").write_text(json.dumps({"bindings": []}))
        monkeypatch.setattr(snapshots, "SNAPSHOTS_DIR", cache / "snapshots")

        with pytest.raises(ValueError, match="no git_commit"):
            update_index(str(tmp_path / "src"), since="baseline")

    def test_drops_and_reindexes_changed_files(self, tmp_path, monkeypatch):
        """Stale bindings for changed files are dropped; re-detected entries added."""
        import subprocess

        src = tmp_path / "src"
        src.mkdir()
        cache = tmp_path / "cache"
        snap_dir = cache / "snapshots" / "baseline"
        snap_dir.mkdir(parents=True)

        baseline_bindings = {
            "bindings": [
                {
                    "python_name": "stale",
                    "cpp_name": "at::stale",
                    "dispatch_key": "CPU",
                    "file_path": str(src / "changed.cpp"),
                    "line_number": 1,
                },
                {
                    "python_name": "stable",
                    "cpp_name": "at::stable",
                    "dispatch_key": "CPU",
                    "file_path": str(src / "untouched.cpp"),
                    "line_number": 2,
                },
            ],
            "cuda_kernels": [],
            "native_functions": {},
            "derivatives": {},
            "native_implementations": {},
        }
        (snap_dir / "bindings.json").write_text(json.dumps(baseline_bindings))
        (snap_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "name": "baseline",
                    "created": "2026-01-01T00:00:00+00:00",
                    "pytorch_source": str(src),
                    "source_fingerprint": "deadbeef",
                    "git_commit": "abc1234",
                    "bindings_size": 0,
                    "bindings_sha256": "x",
                    "content_fingerprint": None,
                    "schema_version": 2,
                }
            )
        )
        (src / "changed.cpp").write_text("// new content")

        monkeypatch.setattr(snapshots, "SNAPSHOTS_DIR", cache / "snapshots")
        monkeypatch.setattr(indexer, "CACHE_DIR", cache)
        monkeypatch.setattr(indexer, "_cache_path", lambda _s: cache / "bindings.json")
        monkeypatch.setattr(indexer, "_source_fingerprint", lambda _s: "newfp")

        def fake_diff(cmd, **kwargs):
            class R:
                stdout = "M\tchanged.cpp\n"

            return R()

        monkeypatch.setattr(subprocess, "run", fake_diff)

        class FakeBindingGraph:
            def __init__(self):
                class B:
                    def to_dict(self):
                        return {
                            "python_name": "reindexed",
                            "cpp_name": "at::reindexed",
                            "dispatch_key": "CPU",
                            "file_path": str(src / "changed.cpp"),
                            "line_number": 5,
                        }

                self.bindings = [B()]
                self.cuda_kernels = []

        class FakeDetector:
            def detect_bindings(self, _path, _content):
                return FakeBindingGraph()

        monkeypatch.setattr(
            "torchtalk.analysis.binding_detector.BindingDetector", FakeDetector
        )

        stats = update_index(str(src), since="baseline")

        assert stats["cpp_files_changed"] == 1
        assert stats["bindings_total"] == 2  # stable + reindexed, stale dropped

        written = json.loads(Path(cache / "bindings.json").read_text())
        names = {b["python_name"] for b in written["bindings"]}
        assert names == {"stable", "reindexed"}


class TestWidenReparseSet:
    def test_returns_empty_for_no_uncovered(self, tmp_path):
        assert indexer._widen_reparse_set(tmp_path, set(), {}) == set()

    def test_filters_grep_results_to_compile_db(self, tmp_path, monkeypatch):
        import subprocess as sp

        def fake_run(cmd, **kw):
            class R:
                returncode = 0
                stdout = "a.cpp\nb.cpp\nnot_compiled.cpp\n"

            return R()

        monkeypatch.setattr(sp, "run", fake_run)

        cc_index = {
            str(tmp_path / "a.cpp"): {},
            str(tmp_path / "b.cpp"): {},
        }
        result = indexer._widen_reparse_set(tmp_path, {"foo.h"}, cc_index)
        assert result == {"a.cpp", "b.cpp"}

    def test_git_grep_missing_skips_silently(self, tmp_path, monkeypatch):
        import subprocess as sp

        def fake_run(cmd, **kw):
            raise FileNotFoundError("git missing")

        monkeypatch.setattr(sp, "run", fake_run)
        assert indexer._widen_reparse_set(tmp_path, {"foo.h"}, {}) == set()

    def test_git_grep_timeout_skips_header(self, tmp_path, monkeypatch):
        import subprocess as sp

        def fake_run(cmd, **kw):
            raise sp.TimeoutExpired(cmd=cmd, timeout=30)

        monkeypatch.setattr(sp, "run", fake_run)
        assert indexer._widen_reparse_set(tmp_path, {"foo.h"}, {}) == set()

    def test_header_with_empty_basename_skipped(self, tmp_path):
        assert indexer._widen_reparse_set(tmp_path, {""}, {}) == set()
