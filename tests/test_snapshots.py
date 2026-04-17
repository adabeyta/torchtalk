"""Tests for the snapshots module."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from torchtalk import snapshots
from torchtalk.snapshots import (
    SCHEMA_VERSION,
    SnapshotCorruptError,
    SnapshotError,
    SnapshotExistsError,
    SnapshotManifest,
    SnapshotNotFoundError,
    _migrate,
    _validate_name,
    delete_snapshot,
    diff_snapshots,
    export_snapshot,
    find_nearest_snapshot,
    import_snapshot,
    list_snapshots,
    load_snapshot,
    save_snapshot,
)


@pytest.fixture
def fake_cache(tmp_path, monkeypatch):
    """Redirect CACHE_DIR and cache_paths into tmp_path, seed with minimal data."""
    cache = tmp_path / "cache"
    (cache / "call_graph").mkdir(parents=True)

    fingerprint = "deadbeef1234"
    bindings_file = cache / f"bindings_{fingerprint}.json"
    callgraph_file = (
        cache / "call_graph" / f"pytorch_callgraph_parallel_{fingerprint}.json"
    )

    bindings_payload = {
        "metadata": {"source_fingerprint": fingerprint},
        "bindings": [
            {
                "python_name": "torch.add",
                "cpp_name": "at::add",
                "dispatch_key": "CPU",
                "file_path": "aten/src/ATen/native/BinaryOps.cpp",
                "line_number": 10,
            }
        ],
    }
    bindings_file.write_text(json.dumps(bindings_payload))
    callgraph_file.write_text(json.dumps({"functions": {}, "edges": []}))

    source = str(tmp_path / "fake_pytorch")
    Path(source).mkdir()

    def fake_cache_paths(_source):
        return {"bindings": bindings_file, "callgraph": callgraph_file}

    monkeypatch.setattr(snapshots, "CACHE_DIR", cache)
    monkeypatch.setattr(snapshots, "SNAPSHOTS_DIR", cache / "snapshots")
    monkeypatch.setattr(snapshots, "cache_paths", fake_cache_paths)
    monkeypatch.setattr(snapshots, "resolve_pytorch_source", lambda: source)

    return {
        "cache": cache,
        "source": source,
        "fingerprint": fingerprint,
        "bindings": bindings_file,
        "callgraph": callgraph_file,
        "payload": bindings_payload,
    }


class TestValidateName:
    def test_accepts_simple(self):
        _validate_name("v1")
        _validate_name("v2.4-baseline")
        _validate_name("release_2026_04")

    def test_rejects_path_traversal(self):
        with pytest.raises(SnapshotError):
            _validate_name("../evil")

    def test_rejects_empty(self):
        with pytest.raises(SnapshotError):
            _validate_name("")

    def test_rejects_leading_dash(self):
        with pytest.raises(SnapshotError):
            _validate_name("-looks-like-flag")

    def test_rejects_too_long(self):
        with pytest.raises(SnapshotError):
            _validate_name("x" * 65)

    def test_accepts_namespaced(self):
        _validate_name("main/abc1234/v1")
        _validate_name("release/2_4")

    def test_rejects_too_many_components(self):
        with pytest.raises(SnapshotError, match="components"):
            _validate_name("a/b/c/d")

    def test_rejects_empty_component(self):
        with pytest.raises(SnapshotError):
            _validate_name("main//v1")

    def test_rejects_component_path_traversal(self):
        with pytest.raises(SnapshotError):
            _validate_name("main/../evil")


class TestSaveSnapshot:
    def test_creates_manifest_and_payload(self, fake_cache):
        manifest = save_snapshot("v1")
        snap_dir = fake_cache["cache"] / "snapshots" / "v1"

        assert snap_dir.is_dir()
        assert (snap_dir / "manifest.json").exists()
        assert (snap_dir / "bindings.json").exists()
        assert (snap_dir / "callgraph.json").exists()
        assert manifest.name == "v1"
        assert manifest.schema_version == SCHEMA_VERSION
        assert manifest.bindings_sha256 != ""

    def test_duplicate_raises(self, fake_cache):
        save_snapshot("v1")
        with pytest.raises(SnapshotExistsError):
            save_snapshot("v1")

    def test_no_source_raises(self, fake_cache, monkeypatch):
        monkeypatch.setattr(snapshots, "resolve_pytorch_source", lambda: None)
        with pytest.raises(SnapshotError, match="No PyTorch source"):
            save_snapshot("v1")

    def test_no_bindings_raises(self, fake_cache):
        fake_cache["bindings"].unlink()
        with pytest.raises(SnapshotError, match="No bindings cache"):
            save_snapshot("v1")


class TestLoadSnapshot:
    def test_roundtrip(self, fake_cache):
        save_snapshot("v1")
        fake_cache["bindings"].write_text(json.dumps({"modified": True}))
        load_snapshot("v1")
        # After load, live cache has original contents again
        restored = json.loads(fake_cache["bindings"].read_text())
        assert restored == fake_cache["payload"]

    def test_missing_raises(self, fake_cache):
        with pytest.raises(SnapshotNotFoundError):
            load_snapshot("ghost")

    def test_corrupt_bindings_raises(self, fake_cache):
        save_snapshot("v1")
        snap_dir = fake_cache["cache"] / "snapshots" / "v1"
        (snap_dir / "bindings.json").write_text('{"evil":true}')
        with pytest.raises(SnapshotCorruptError):
            load_snapshot("v1")

    def test_fingerprint_mismatch_refused(self, fake_cache, monkeypatch):
        save_snapshot("v1")
        # Swap the live cache to a different fingerprint
        other_fp = "cafebabe9999"
        other_bindings = fake_cache["cache"] / f"bindings_{other_fp}.json"
        other_bindings.write_text("{}")
        monkeypatch.setattr(
            snapshots,
            "cache_paths",
            lambda _: {
                "bindings": other_bindings,
                "callgraph": other_bindings.parent
                / "call_graph"
                / f"pytorch_callgraph_parallel_{other_fp}.json",
            },
        )
        with pytest.raises(SnapshotError, match="Fingerprint mismatch"):
            load_snapshot("v1")

    def test_fingerprint_mismatch_force_loads(self, fake_cache, monkeypatch):
        save_snapshot("v1")
        other_fp = "cafebabe9999"
        other_bindings = fake_cache["cache"] / f"bindings_{other_fp}.json"
        other_bindings.write_text("{}")
        monkeypatch.setattr(
            snapshots,
            "cache_paths",
            lambda _: {
                "bindings": other_bindings,
                "callgraph": other_bindings.parent
                / "call_graph"
                / f"pytorch_callgraph_parallel_{other_fp}.json",
            },
        )
        manifest = load_snapshot("v1", force=True)
        assert manifest.name == "v1"


class TestListAndDelete:
    def test_empty(self, fake_cache):
        assert list_snapshots() == []

    def test_list_sorted_newest_first(self, fake_cache):
        save_snapshot("old")
        # Mutate created to simulate an older timestamp
        old_dir = fake_cache["cache"] / "snapshots" / "old"
        mf = json.loads((old_dir / "manifest.json").read_text())
        mf["created"] = "2020-01-01T00:00:00+00:00"
        (old_dir / "manifest.json").write_text(json.dumps(mf))

        save_snapshot("new")

        names = [m.name for m in list_snapshots()]
        assert names == ["new", "old"]

    def test_list_skips_corrupt(self, fake_cache):
        save_snapshot("good")
        # Create a dir with an unreadable manifest
        bad_dir = fake_cache["cache"] / "snapshots" / "bad"
        bad_dir.mkdir()
        (bad_dir / "manifest.json").write_text("{not json")
        names = [m.name for m in list_snapshots()]
        assert names == ["good"]

    def test_delete(self, fake_cache):
        save_snapshot("v1")
        delete_snapshot("v1")
        assert not (fake_cache["cache"] / "snapshots" / "v1").exists()

    def test_delete_missing_raises(self, fake_cache):
        with pytest.raises(SnapshotNotFoundError):
            delete_snapshot("ghost")


class TestMigrate:
    def test_adds_missing_version(self):
        out = _migrate({})
        assert out["schema_version"] == SCHEMA_VERSION
        assert "bindings_sha256" in out
        assert "content_fingerprint" in out

    def test_preserves_current_version(self):
        data = {"schema_version": SCHEMA_VERSION, "bindings_sha256": "abc"}
        out = _migrate(data)
        assert out["bindings_sha256"] == "abc"

    def test_ignores_unknown_fields_on_load(self, fake_cache):
        save_snapshot("v1")
        mf = fake_cache["cache"] / "snapshots" / "v1" / "manifest.json"
        data = json.loads(mf.read_text())
        data["future_field"] = "value"
        mf.write_text(json.dumps(data))
        # Should still load (unknown field filtered by _load_manifest)
        manifests = list_snapshots()
        assert manifests[0].name == "v1"


class TestDiff:
    def test_identical_snapshots(self, fake_cache):
        save_snapshot("a")
        save_snapshot("b")
        d = diff_snapshots("a", "b")
        assert d.is_empty()

    def test_file_modified(self, fake_cache):
        save_snapshot("before")

        # Modify bindings in the live cache, then snapshot again
        payload = fake_cache["payload"].copy()
        payload["bindings"] = [
            {
                "python_name": "torch.add",
                "cpp_name": "at::add",
                "dispatch_key": "CUDA",  # changed
                "file_path": "aten/src/ATen/native/BinaryOps.cpp",
                "line_number": 10,
            }
        ]
        fake_cache["bindings"].write_text(json.dumps(payload))
        save_snapshot("after")

        d = diff_snapshots("before", "after")
        assert not d.is_empty()
        assert "aten/src/ATen/native/BinaryOps.cpp" in d.files_modified
        assert d.dispatch_keys_added == ["CUDA"]
        assert d.dispatch_keys_removed == ["CPU"]

    def test_file_added(self, fake_cache):
        save_snapshot("before")

        payload = fake_cache["payload"].copy()
        payload["bindings"] = payload["bindings"] + [
            {
                "python_name": "torch.mul",
                "cpp_name": "at::mul",
                "dispatch_key": "CPU",
                "file_path": "aten/src/ATen/native/NewOp.cpp",
                "line_number": 42,
            }
        ]
        fake_cache["bindings"].write_text(json.dumps(payload))
        save_snapshot("after")

        d = diff_snapshots("before", "after")
        assert "aten/src/ATen/native/NewOp.cpp" in d.files_added
        assert d.bindings_added == 1


class TestAtomicWrite:
    def test_partial_write_cleanup(self, fake_cache, monkeypatch):
        # Force an error mid-save to ensure no partial snapshot remains
        original_copy = snapshots.shutil.copy2
        calls = {"n": 0}

        def flaky_copy(src, dst):
            calls["n"] += 1
            if calls["n"] == 2:  # fail on the callgraph copy
                raise OSError("disk full")
            return original_copy(src, dst)

        monkeypatch.setattr(snapshots.shutil, "copy2", flaky_copy)
        with pytest.raises(OSError):
            save_snapshot("v1")

        # The snapshot dir should not exist after the failed save
        assert not (fake_cache["cache"] / "snapshots" / "v1").exists()


class TestManifestSchema:
    def test_manifest_to_from_json(self):
        m = SnapshotManifest(
            name="x",
            created="2026-01-01T00:00:00+00:00",
            pytorch_source="/p",
            source_fingerprint="abc",
            git_commit=None,
            bindings_size=100,
            bindings_sha256="hash",
        )
        # Round-trip through JSON
        from dataclasses import asdict

        data = asdict(m)
        m2 = SnapshotManifest(**data)
        assert m == m2


class TestExportImport:
    def test_export_creates_tarball(self, fake_cache, tmp_path):
        save_snapshot("v1")
        out = tmp_path / "v1.tar.gz"
        export_snapshot("v1", out)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_export_missing_raises(self, fake_cache, tmp_path):
        with pytest.raises(SnapshotNotFoundError):
            export_snapshot("ghost", tmp_path / "out.tar.gz")

    def test_roundtrip(self, fake_cache, tmp_path):
        save_snapshot("v1")
        out = tmp_path / "v1.tar.gz"
        export_snapshot("v1", out)
        delete_snapshot("v1")

        manifest = import_snapshot(out)
        assert manifest.name == "v1"
        assert (fake_cache["cache"] / "snapshots" / "v1").is_dir()

    def test_import_rename(self, fake_cache, tmp_path):
        save_snapshot("v1")
        out = tmp_path / "v1.tar.gz"
        export_snapshot("v1", out)
        delete_snapshot("v1")

        manifest = import_snapshot(out, name="v1-imported")
        assert manifest.name == "v1-imported"
        assert (fake_cache["cache"] / "snapshots" / "v1-imported").is_dir()

    def test_import_existing_name_raises(self, fake_cache, tmp_path):
        save_snapshot("v1")
        out = tmp_path / "v1.tar.gz"
        export_snapshot("v1", out)
        # v1 still exists in snapshots dir, import should refuse
        with pytest.raises(SnapshotExistsError):
            import_snapshot(out)

    def test_import_missing_archive(self, fake_cache, tmp_path):
        with pytest.raises(SnapshotError):
            import_snapshot(tmp_path / "nonexistent.tar.gz")

    def test_import_rejects_unsafe_paths(self, fake_cache, tmp_path):
        """Archive containing ../ paths must be rejected."""
        import tarfile

        archive = tmp_path / "evil.tar.gz"
        payload = tmp_path / "payload"
        payload.mkdir()
        (payload / "file").write_text("x")
        with tarfile.open(archive, "w:gz") as tar:
            tar.add(payload, arcname="../../etc/evil")

        with pytest.raises(SnapshotCorruptError, match="rejected"):
            import_snapshot(archive)


class TestNamespacedNames:
    def test_save_and_list_namespaced(self, fake_cache):
        save_snapshot("main/abc1234/v1")
        snap_dir = fake_cache["cache"] / "snapshots" / "main" / "abc1234" / "v1"
        assert snap_dir.is_dir()
        assert (snap_dir / "manifest.json").exists()

        names = [m.name for m in list_snapshots()]
        assert "main/abc1234/v1" in names

    def test_delete_namespaced(self, fake_cache):
        save_snapshot("main/abc/v1")
        delete_snapshot("main/abc/v1")
        assert not (fake_cache["cache"] / "snapshots" / "main" / "abc" / "v1").exists()

    def test_list_skips_in_progress_tempdirs(self, fake_cache):
        save_snapshot("v1")
        # Simulate an in-flight save tempdir alongside real snapshots
        tmp = fake_cache["cache"] / "snapshots" / ".v2-inprogress"
        tmp.mkdir()
        (tmp / "manifest.json").write_text("{}")

        names = [m.name for m in list_snapshots()]
        assert names == ["v1"]

    def test_export_import_namespaced_roundtrip(self, fake_cache, tmp_path):
        save_snapshot("main/abc/v1")
        out = tmp_path / "snap.tar.gz"
        export_snapshot("main/abc/v1", out)
        delete_snapshot("main/abc/v1")

        manifest = import_snapshot(out)
        assert manifest.name == "main/abc/v1"
        assert (fake_cache["cache"] / "snapshots" / "main" / "abc" / "v1").is_dir()


class TestContentFingerprint:
    def test_saved_in_manifest(self, fake_cache, monkeypatch):
        monkeypatch.setattr(snapshots, "_content_fingerprint", lambda _: "c0ffee")
        m = save_snapshot("v1")
        assert m.content_fingerprint == "c0ffee"

    def test_load_accepts_content_match_despite_path_mismatch(
        self, fake_cache, monkeypatch
    ):
        monkeypatch.setattr(snapshots, "_content_fingerprint", lambda _: "c0ffee")
        save_snapshot("v1")

        # Swap live cache to a different path fingerprint, keep content match
        other_fp = "cafebabe9999"
        other_bindings = fake_cache["cache"] / f"bindings_{other_fp}.json"
        other_bindings.write_text("{}")
        monkeypatch.setattr(
            snapshots,
            "cache_paths",
            lambda _: {
                "bindings": other_bindings,
                "callgraph": other_bindings.parent
                / "call_graph"
                / f"pytorch_callgraph_parallel_{other_fp}.json",
            },
        )
        m = load_snapshot("v1")
        assert m.name == "v1"

    def test_load_refuses_when_both_fingerprints_mismatch(
        self, fake_cache, monkeypatch
    ):
        monkeypatch.setattr(snapshots, "_content_fingerprint", lambda _: "c0ffee")
        save_snapshot("v1")

        monkeypatch.setattr(snapshots, "_content_fingerprint", lambda _: "different")
        other_fp = "cafebabe9999"
        other_bindings = fake_cache["cache"] / f"bindings_{other_fp}.json"
        other_bindings.write_text("{}")
        monkeypatch.setattr(
            snapshots,
            "cache_paths",
            lambda _: {
                "bindings": other_bindings,
                "callgraph": other_bindings.parent
                / "call_graph"
                / f"pytorch_callgraph_parallel_{other_fp}.json",
            },
        )
        with pytest.raises(SnapshotError, match="Fingerprint mismatch"):
            load_snapshot("v1")


class TestMigrateV2:
    def test_v1_manifest_gets_null_content_fingerprint(self, fake_cache):
        save_snapshot("v1")
        mf = fake_cache["cache"] / "snapshots" / "v1" / "manifest.json"
        data = json.loads(mf.read_text())
        data["schema_version"] = 1
        data.pop("content_fingerprint", None)
        mf.write_text(json.dumps(data))

        m = list_snapshots()[0]
        assert m.schema_version == SCHEMA_VERSION
        assert m.content_fingerprint is None


class TestFindNearestSnapshot:
    def test_exact_commit_match(self, fake_cache, monkeypatch):
        save_snapshot("v1")
        snap_dir = fake_cache["cache"] / "snapshots" / "v1"
        mf = json.loads((snap_dir / "manifest.json").read_text())
        mf["git_commit"] = "abc1234"
        (snap_dir / "manifest.json").write_text(json.dumps(mf))

        monkeypatch.setattr(snapshots, "_git_commit", lambda _: "abc1234")
        picked = find_nearest_snapshot()
        assert picked is not None
        assert picked.name == "v1"

    def test_ancestor_match_when_no_exact(self, fake_cache, monkeypatch):
        save_snapshot("old")
        snap_dir = fake_cache["cache"] / "snapshots" / "old"
        mf = json.loads((snap_dir / "manifest.json").read_text())
        mf["git_commit"] = "older01"
        (snap_dir / "manifest.json").write_text(json.dumps(mf))

        monkeypatch.setattr(snapshots, "_git_commit", lambda _: "newer99")
        monkeypatch.setattr(
            snapshots,
            "_is_ancestor",
            lambda _src, ancestor, descendant: ancestor == "older01",
        )
        picked = find_nearest_snapshot()
        assert picked is not None
        assert picked.name == "old"

    def test_none_when_no_candidates(self, fake_cache, monkeypatch):
        save_snapshot("v1")
        snap_dir = fake_cache["cache"] / "snapshots" / "v1"
        mf = json.loads((snap_dir / "manifest.json").read_text())
        mf["git_commit"] = None
        (snap_dir / "manifest.json").write_text(json.dumps(mf))

        monkeypatch.setattr(snapshots, "_git_commit", lambda _: "abc1234")
        assert find_nearest_snapshot() is None

    def test_none_when_not_git_repo(self, fake_cache, monkeypatch):
        save_snapshot("v1")
        monkeypatch.setattr(snapshots, "_git_commit", lambda _: None)
        assert find_nearest_snapshot() is None

    def test_none_when_no_source(self, fake_cache, monkeypatch):
        monkeypatch.setattr(snapshots, "resolve_pytorch_source", lambda: None)
        assert find_nearest_snapshot() is None

    def test_prefers_exact_over_ancestor(self, fake_cache, monkeypatch):
        save_snapshot("ancestor")
        save_snapshot("exact")
        for name, commit in [("ancestor", "older01"), ("exact", "head9999")]:
            mf_path = fake_cache["cache"] / "snapshots" / name / "manifest.json"
            mf = json.loads(mf_path.read_text())
            mf["git_commit"] = commit
            mf_path.write_text(json.dumps(mf))

        monkeypatch.setattr(snapshots, "_git_commit", lambda _: "head9999")
        monkeypatch.setattr(snapshots, "_is_ancestor", lambda *_: True)
        picked = find_nearest_snapshot()
        assert picked is not None
        assert picked.name == "exact"

    def test_prefers_content_fingerprint_over_git_commit(self, fake_cache, monkeypatch):
        monkeypatch.setattr(snapshots, "_content_fingerprint", lambda _: "c0ffee")
        save_snapshot("by_content")
        monkeypatch.setattr(snapshots, "_content_fingerprint", lambda _: "different")
        save_snapshot("by_commit")

        # 'by_commit' matches git HEAD, 'by_content' matches content hash
        for name, commit in [("by_content", "stale99"), ("by_commit", "head9999")]:
            mf_path = fake_cache["cache"] / "snapshots" / name / "manifest.json"
            mf = json.loads(mf_path.read_text())
            mf["git_commit"] = commit
            mf_path.write_text(json.dumps(mf))

        monkeypatch.setattr(snapshots, "_content_fingerprint", lambda _: "c0ffee")
        monkeypatch.setattr(snapshots, "_git_commit", lambda _: "head9999")
        picked = find_nearest_snapshot()
        assert picked is not None
        assert picked.name == "by_content"
