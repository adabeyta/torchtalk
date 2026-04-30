"""Tests for indexer data structures and fuzzy matching."""

import ast
import json
from pathlib import Path

import pytest

from torchtalk import indexer, snapshots
from torchtalk.indexer import (
    ServerState,
    _build_indexes,
    _classify_rhs,
    _collect_test_attr_hits,
    _fuzzy_find,
    _infer_local_types,
    update_index,
)


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


class TestClassifyRhs:
    def _rhs(self, code: str):
        return ast.parse(code, mode="eval").body

    def test_dict_literal(self):
        assert _classify_rhs(self._rhs("{}")) == "dict"

    def test_list_literal(self):
        assert _classify_rhs(self._rhs("[1, 2]")) == "list"

    def test_str_literal(self):
        assert _classify_rhs(self._rhs("'hello'")) == "str"

    def test_int_literal(self):
        assert _classify_rhs(self._rhs("42")) == "number"

    def test_bool_literal(self):
        assert _classify_rhs(self._rhs("True")) == "bool"

    def test_torch_call_is_tensor(self):
        assert _classify_rhs(self._rhs("torch.randn(3, 3)")) == "tensor"

    def test_F_call_is_tensor(self):
        assert _classify_rhs(self._rhs("F.relu(x)")) == "tensor"

    def test_unrelated_call_is_unknown(self):
        assert _classify_rhs(self._rhs("foo(x)")) is None

    def test_method_call_is_unknown(self):
        assert _classify_rhs(self._rhs("self.helper()")) is None


class TestInferLocalTypes:
    def _func(self, code: str):
        tree = ast.parse(code)
        return tree.body[0]

    def test_tracks_torch_assignments(self):
        func = self._func("def f():\n    t = torch.randn(3); d = {}; n = 5\n")
        types = _infer_local_types(func)
        assert types == {"t": "tensor", "d": "dict", "n": "number"}

    def test_skips_unknown_rhs(self):
        func = self._func("def f():\n    x = some_helper()\n")
        types = _infer_local_types(func)
        assert types == {}


class TestParseOpInfoRegistry:
    def test_extracts_name_aliases_aten_name(self, tmp_path, monkeypatch):
        # Reset state
        from torchtalk.indexer import _state

        _state.opinfo_registry = {}
        _state.opinfo_alias_map = {}
        _state.pytorch_source = str(tmp_path)

        opinfo_file = tmp_path / "opinfos.py"
        opinfo_file.write_text(
            "OpInfo('nn.functional.conv2d',\n"
            "       aliases=('conv2d',),\n"
            "       aten_name='conv2d')\n"
            "BinaryUfuncInfo('add',\n"
            "                aten_name='add')\n"
        )
        indexer._parse_opinfo_registry(str(opinfo_file))

        assert "nn.functional.conv2d" in _state.opinfo_registry
        entry = _state.opinfo_registry["nn.functional.conv2d"]
        assert entry["aliases"] == ["conv2d"]
        assert entry["aten_name"] == "conv2d"

        assert "add" in _state.opinfo_registry
        # Both aliases and aten_name register into alias_map
        assert "conv2d" in _state.opinfo_alias_map
        assert _state.opinfo_alias_map["conv2d"][0]["name"] == "nn.functional.conv2d"
        assert "add" in _state.opinfo_alias_map


class TestCollectTestAttrHits:
    def _walk(self, code: str, interesting: set[str]):
        tree = ast.parse(code)
        func = tree.body[0]
        index: dict[str, list[dict]] = {}
        _collect_test_attr_hits(func, "test/test_x.py", "TestX", interesting, index)
        return index

    def test_records_receiver_type_for_known_local(self):
        code = "def test_copy(self):\n    t = torch.randn(3)\n    t.copy_(other)\n"
        index = self._walk(code, {"copy_"})
        assert index["copy_"][0]["receiver_type"] == "tensor"

    def test_records_dict_receiver(self):
        code = "def test_copy(self):\n    d = {}\n    d.copy()\n"
        index = self._walk(code, {"copy"})
        assert index["copy"][0]["receiver_type"] == "dict"

    def test_unknown_receiver_records_none(self):
        code = "def test_copy(self):\n    t.copy_(other)\n"
        index = self._walk(code, {"copy_"})
        assert index["copy_"][0]["receiver_type"] is None
