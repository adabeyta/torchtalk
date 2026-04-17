"""Named snapshots of the TorchTalk index cache.

A snapshot is a captured copy of the bindings index and call graph for a
given PyTorch source. Use cases: CI artifacts, version baselines, rollback
before risky rebuilds.

Layout:
    ~/.cache/torchtalk/snapshots/<name>/
        manifest.json     Metadata, schema version, checksums
        bindings.json     Binding index
        callgraph.json    C++ call graph (if built)
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import CACHE_DIR, cache_paths, resolve_pytorch_source

SNAPSHOTS_DIR = CACHE_DIR / "snapshots"
MANIFEST_NAME = "manifest.json"
SCHEMA_VERSION = 2

_NAME_COMPONENT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_NAME_MAX_COMPONENTS = 3


class SnapshotError(Exception):
    """Base class for snapshot operation failures."""


class SnapshotNotFoundError(SnapshotError):
    """Raised when a named snapshot does not exist."""


class SnapshotExistsError(SnapshotError):
    """Raised when a snapshot with the same name already exists."""


class SnapshotCorruptError(SnapshotError):
    """Raised when a snapshot's manifest or payload fails integrity checks."""


@dataclass
class SnapshotManifest:
    """Metadata for a snapshot directory."""

    name: str
    created: str
    pytorch_source: str
    source_fingerprint: str
    git_commit: str | None
    bindings_size: int
    bindings_sha256: str
    callgraph_size: int = 0
    callgraph_sha256: str = ""
    content_fingerprint: str | None = None
    schema_version: int = SCHEMA_VERSION


@dataclass
class SnapshotDiff:
    """Structural delta between two snapshots, produced by diff_snapshots()."""

    left: str
    right: str
    files_added: list[str] = field(default_factory=list)
    files_removed: list[str] = field(default_factory=list)
    files_modified: list[str] = field(default_factory=list)
    bindings_added: int = 0
    bindings_removed: int = 0
    dispatch_keys_added: list[str] = field(default_factory=list)
    dispatch_keys_removed: list[str] = field(default_factory=list)

    def is_empty(self) -> bool:
        return not (
            self.files_added
            or self.files_removed
            or self.files_modified
            or self.bindings_added
            or self.bindings_removed
            or self.dispatch_keys_added
            or self.dispatch_keys_removed
        )


def _validate_name(name: str) -> None:
    """Accept up to 3 slash-separated components (e.g. 'main/abc1234/v1').

    Each component must start with a letter or digit and use only letters,
    digits, dots, dashes, or underscores, max 64 chars.
    """
    components = name.split("/")
    if len(components) > _NAME_MAX_COMPONENTS:
        raise SnapshotError(
            f"Snapshot name '{name}' has {len(components)} components "
            f"(max {_NAME_MAX_COMPONENTS})."
        )
    for c in components:
        if not _NAME_COMPONENT_RE.match(c):
            raise SnapshotError(
                f"Invalid snapshot name component '{c}' in '{name}'. "
                "Use letters, digits, dots, dashes, underscores (max 64 chars)."
            )


def _hash_file(path: Path) -> str:
    """BLAKE2b hex digest of a file."""
    if hasattr(hashlib, "file_digest"):
        with path.open("rb") as f:
            return hashlib.file_digest(f, "blake2b").hexdigest()
    h = hashlib.blake2b()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_commit(source: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", source, "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        return result.stdout.strip() or None
    except (
        subprocess.CalledProcessError,
        FileNotFoundError,
        subprocess.TimeoutExpired,
    ):
        return None


def _content_fingerprint(source: str) -> str | None:
    """Merkle-style hash over HEAD tree + any uncommitted diff.

    Uses git's own tree hash (a content-addressed Merkle over all tracked files)
    combined with a hash of `git diff HEAD` to cover dirty trees. Two checkouts
    with identical content produce the same fingerprint regardless of path.
    Returns None when source is not a git working tree.
    """
    try:
        tree = subprocess.run(
            ["git", "-C", source, "rev-parse", "HEAD^{tree}"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        ).stdout.strip()
        diff = subprocess.run(
            ["git", "-C", source, "diff", "HEAD"],
            capture_output=True,
            check=True,
            timeout=15,
        ).stdout
    except (
        subprocess.CalledProcessError,
        FileNotFoundError,
        subprocess.TimeoutExpired,
    ):
        return None

    h = hashlib.blake2b(tree.encode(), digest_size=16)
    h.update(diff)
    return h.hexdigest()


def _is_ancestor(source: str, ancestor: str, descendant: str) -> bool:
    """Return True if ancestor commit is reachable from descendant in source's git."""
    try:
        result = subprocess.run(
            ["git", "-C", source, "merge-base", "--is-ancestor", ancestor, descendant],
            capture_output=True,
            timeout=5,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _atomic_write_text(path: Path, text: str) -> None:
    """Write text atomically via tempfile + fsync + os.replace."""
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except Exception:
        Path(tmp).unlink(missing_ok=True)
        raise


def _migrate(data: dict[str, Any]) -> dict[str, Any]:
    """Upgrade older manifest schemas to the current one in-place."""
    data.setdefault("schema_version", 0)
    if data["schema_version"] < 1:
        data.setdefault("bindings_sha256", "")
        data.setdefault("callgraph_sha256", "")
        data["schema_version"] = 1
    if data["schema_version"] < 2:
        data.setdefault("content_fingerprint", None)
        data["schema_version"] = 2
    return data


def _load_manifest(path: Path) -> SnapshotManifest:
    """Read, migrate, and validate a manifest file."""
    try:
        raw = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as e:
        raise SnapshotCorruptError(f"Cannot read manifest {path}: {e}") from e

    raw = _migrate(raw)
    known = {f.name for f in fields(SnapshotManifest)}
    try:
        return SnapshotManifest(**{k: v for k, v in raw.items() if k in known})
    except TypeError as e:
        raise SnapshotCorruptError(
            f"Manifest {path} missing required fields: {e}"
        ) from e


def _snapshot_dir(name: str) -> Path:
    """Directory path for a named snapshot (name validated)."""
    _validate_name(name)
    return SNAPSHOTS_DIR / name


def list_snapshots() -> list[SnapshotManifest]:
    """Return all readable snapshot manifests, newest first.

    Walks the snapshot tree recursively so namespaced names (e.g. `main/abc/v1`)
    show up. Hidden dirs (in-progress saves/imports) are skipped. Corrupt
    manifests are skipped silently; use read_manifest() to surface errors.
    """
    if not SNAPSHOTS_DIR.exists():
        return []

    manifests: list[SnapshotManifest] = []
    for mf in SNAPSHOTS_DIR.rglob(MANIFEST_NAME):
        rel = mf.relative_to(SNAPSHOTS_DIR).parts
        if any(p.startswith(".") for p in rel):
            continue
        try:
            manifests.append(_load_manifest(mf))
        except SnapshotCorruptError:
            continue

    manifests.sort(key=lambda m: m.created, reverse=True)
    return manifests


def read_manifest(name: str) -> SnapshotManifest:
    """Read a named snapshot's manifest, raising on missing/corrupt."""
    d = _snapshot_dir(name)
    if not d.exists():
        raise SnapshotNotFoundError(f"Snapshot '{name}' not found")
    return _load_manifest(d / MANIFEST_NAME)


def find_nearest_snapshot(source: str | None = None) -> SnapshotManifest | None:
    """Return the snapshot that best matches source.

    Tiered match: exact content_fingerprint (tree + dirty diff) > exact git_commit
    > most recent ancestor commit. Returns None if nothing matches or source is
    not a git working tree.
    """
    source = source or resolve_pytorch_source()
    if not source:
        return None

    candidates = list_snapshots()
    content_fp = _content_fingerprint(source)
    if content_fp:
        for m in candidates:
            if m.content_fingerprint and m.content_fingerprint == content_fp:
                return m

    head = _git_commit(source)
    if not head:
        return None

    with_commit = [m for m in candidates if m.git_commit]
    for m in with_commit:
        if m.git_commit == head:
            return m
    for m in with_commit:
        if _is_ancestor(source, m.git_commit, head):
            return m
    return None


def save_snapshot(name: str) -> SnapshotManifest:
    """Capture the active cache as a named snapshot.

    Uses a tempdir+rename to make the directory creation atomic.
    """
    _validate_name(name)

    source = resolve_pytorch_source()
    if not source:
        raise SnapshotError("No PyTorch source configured. Run 'torchtalk init' first.")

    paths = cache_paths(source)
    if not paths["bindings"].exists():
        raise SnapshotError(
            f"No bindings cache to snapshot for {source}. "
            "Start the MCP server once to build it."
        )

    dest = _snapshot_dir(name)
    if dest.exists():
        raise SnapshotExistsError(f"Snapshot '{name}' already exists")

    SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)
    dest.parent.mkdir(parents=True, exist_ok=True)

    safe_prefix = name.replace("/", "_")
    with tempfile.TemporaryDirectory(
        dir=SNAPSHOTS_DIR, prefix=f".{safe_prefix}-"
    ) as tmp:
        tmp_path = Path(tmp)

        bindings_tmp = tmp_path / "bindings.json"
        shutil.copy2(paths["bindings"], bindings_tmp)
        bindings_size = bindings_tmp.stat().st_size
        bindings_sha256 = _hash_file(bindings_tmp)

        callgraph_size = 0
        callgraph_sha256 = ""
        if paths["callgraph"].exists():
            cg_tmp = tmp_path / "callgraph.json"
            shutil.copy2(paths["callgraph"], cg_tmp)
            callgraph_size = cg_tmp.stat().st_size
            callgraph_sha256 = _hash_file(cg_tmp)

        fingerprint = paths["bindings"].stem.removeprefix("bindings_")
        manifest = SnapshotManifest(
            name=name,
            created=datetime.now(timezone.utc).isoformat(),
            pytorch_source=source,
            source_fingerprint=fingerprint,
            git_commit=_git_commit(source),
            bindings_size=bindings_size,
            bindings_sha256=bindings_sha256,
            callgraph_size=callgraph_size,
            callgraph_sha256=callgraph_sha256,
            content_fingerprint=_content_fingerprint(source),
        )
        _atomic_write_text(
            tmp_path / MANIFEST_NAME, json.dumps(asdict(manifest), indent=2)
        )

        os.rename(tmp_path, dest)

    return manifest


def load_snapshot(name: str, force: bool = False) -> SnapshotManifest:
    """Restore a snapshot into the active cache.

    Verifies checksums before touching the live cache. Refuses when neither
    path fingerprint nor content fingerprint match the current source, unless
    force=True.
    """
    manifest = read_manifest(name)
    src = _snapshot_dir(name)

    bindings_file = src / "bindings.json"
    if _hash_file(bindings_file) != manifest.bindings_sha256:
        raise SnapshotCorruptError(f"Snapshot '{name}' bindings.json failed checksum")

    callgraph_file = src / "callgraph.json"
    if (
        manifest.callgraph_sha256
        and callgraph_file.exists()
        and _hash_file(callgraph_file) != manifest.callgraph_sha256
    ):
        raise SnapshotCorruptError(f"Snapshot '{name}' callgraph.json failed checksum")

    source = resolve_pytorch_source() or manifest.pytorch_source
    paths = cache_paths(source)
    current_fp = paths["bindings"].stem.removeprefix("bindings_")
    current_content_fp = _content_fingerprint(source)

    path_match = current_fp == manifest.source_fingerprint
    content_match = (
        manifest.content_fingerprint is not None
        and current_content_fp == manifest.content_fingerprint
    )

    if not (force or path_match or content_match):
        raise SnapshotError(
            f"Fingerprint mismatch: snapshot is for '{manifest.pytorch_source}' "
            f"({manifest.source_fingerprint}), current source is '{source}' "
            f"({current_fp}). Pass force=True to override."
        )

    paths["bindings"].parent.mkdir(parents=True, exist_ok=True)
    paths["callgraph"].parent.mkdir(parents=True, exist_ok=True)

    shutil.copy2(bindings_file, paths["bindings"])
    if callgraph_file.exists():
        shutil.copy2(callgraph_file, paths["callgraph"])

    return manifest


def delete_snapshot(name: str) -> None:
    """Delete a named snapshot."""
    target = _snapshot_dir(name)
    if not target.exists():
        raise SnapshotNotFoundError(f"Snapshot '{name}' not found")
    shutil.rmtree(target)


def export_snapshot(name: str, output: Path) -> Path:
    """Package a snapshot into a gzipped tarball at output.

    Uses the leaf component of a namespaced name as the archive's top-level
    entry. The snapshot's canonical name is recorded inside manifest.json, so
    nesting is preserved on import.
    """
    import tarfile

    src = _snapshot_dir(name)
    if not src.exists():
        raise SnapshotNotFoundError(f"Snapshot '{name}' not found")

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    arcname = name.rsplit("/", 1)[-1]
    with tarfile.open(output, "w:gz") as tar:
        tar.add(src, arcname=arcname)
    return output


def import_snapshot(archive: Path, name: str | None = None) -> SnapshotManifest:
    """Extract a snapshot tarball into the snapshots directory.

    The archive must contain a single top-level directory holding manifest.json
    and bindings.json (the format produced by export_snapshot). If name is given,
    the snapshot is renamed on import. Verifies checksums after extraction.
    """
    import tarfile

    archive = Path(archive)
    if not archive.exists():
        raise SnapshotError(f"Archive not found: {archive}")

    SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(dir=SNAPSHOTS_DIR, prefix=".import-") as tmp:
        tmp_path = Path(tmp)
        with tarfile.open(archive, "r:gz") as tar:
            try:
                tar.extractall(path=tmp_path, filter="data")
            except tarfile.FilterError as e:
                raise SnapshotCorruptError(f"Archive rejected: {e}") from e

        roots = [p for p in tmp_path.iterdir() if p.is_dir()]
        if len(roots) != 1:
            raise SnapshotError(
                f"Archive must have one top-level directory, got {len(roots)}"
            )
        extracted = roots[0]

        if not (extracted / MANIFEST_NAME).exists():
            raise SnapshotCorruptError(f"Archive is missing {MANIFEST_NAME}")

        manifest = _load_manifest(extracted / MANIFEST_NAME)

        bindings_file = extracted / "bindings.json"
        if _hash_file(bindings_file) != manifest.bindings_sha256:
            raise SnapshotCorruptError("Archive bindings.json failed checksum")
        cg_file = extracted / "callgraph.json"
        if (
            manifest.callgraph_sha256
            and cg_file.exists()
            and _hash_file(cg_file) != manifest.callgraph_sha256
        ):
            raise SnapshotCorruptError("Archive callgraph.json failed checksum")

        final_name = name or manifest.name
        _validate_name(final_name)
        dest = SNAPSHOTS_DIR / final_name
        if dest.exists():
            raise SnapshotExistsError(f"Snapshot '{final_name}' already exists")
        dest.parent.mkdir(parents=True, exist_ok=True)

        if final_name != manifest.name:
            manifest.name = final_name
            _atomic_write_text(
                extracted / MANIFEST_NAME, json.dumps(asdict(manifest), indent=2)
            )

        os.rename(extracted, dest)

    return manifest


def diff_snapshots(left: str, right: str) -> SnapshotDiff:
    """Structural delta between two snapshots' binding indexes.

    File paths are normalized to be relative to each snapshot's recorded
    pytorch_source, so diffs work across checkouts at different locations.
    Used by Phase 2 partial-update logic to decide which files to re-index.
    """
    left_bindings, left_source = _load_snapshot_payload(left)
    right_bindings, right_source = _load_snapshot_payload(right)

    left_sigs = _file_signatures(left_bindings, left_source)
    right_sigs = _file_signatures(right_bindings, right_source)

    files_added = sorted(set(right_sigs) - set(left_sigs))
    files_removed = sorted(set(left_sigs) - set(right_sigs))
    files_modified = sorted(
        fp for fp in set(left_sigs) & set(right_sigs) if left_sigs[fp] != right_sigs[fp]
    )

    left_keys = {_binding_key(b) for b in left_bindings}
    right_keys = {_binding_key(b) for b in right_bindings}

    left_dk = {b.get("dispatch_key") for b in left_bindings if b.get("dispatch_key")}
    right_dk = {b.get("dispatch_key") for b in right_bindings if b.get("dispatch_key")}

    return SnapshotDiff(
        left=left,
        right=right,
        files_added=files_added,
        files_removed=files_removed,
        files_modified=files_modified,
        bindings_added=len(right_keys - left_keys),
        bindings_removed=len(left_keys - right_keys),
        dispatch_keys_added=sorted(right_dk - left_dk),
        dispatch_keys_removed=sorted(left_dk - right_dk),
    )


def _load_snapshot_payload(name: str) -> tuple[list[dict], str]:
    """Load bindings + pytorch_source from a snapshot. Returns (bindings, source).

    Source is used to normalize absolute file paths back to repo-relative form,
    so diffs work across checkouts at different locations.
    """
    d = _snapshot_dir(name)
    if not d.exists():
        raise SnapshotNotFoundError(f"Snapshot '{name}' not found")

    manifest = _load_manifest(d / MANIFEST_NAME)

    path = d / "bindings.json"
    if not path.exists():
        raise SnapshotCorruptError(f"Snapshot '{name}' is missing bindings.json")
    try:
        bindings = json.loads(path.read_text()).get("bindings", [])
    except json.JSONDecodeError as e:
        raise SnapshotCorruptError(f"Snapshot '{name}' bindings.json: {e}") from e
    return bindings, manifest.pytorch_source


def _relpath(fp: str, source: str) -> str:
    """Strip the PyTorch source prefix from a file path, if present."""
    if not fp:
        return fp
    prefix = source.rstrip("/") + "/"
    if fp.startswith(prefix):
        return fp[len(prefix) :]
    return fp


def _binding_key(b: dict) -> tuple[str, str, str]:
    return (
        b.get("python_name") or "",
        b.get("cpp_name") or "",
        b.get("dispatch_key") or "",
    )


def _file_signatures(bindings: list[dict], source: str) -> dict[str, str]:
    """Map relative-file-path -> hash of its bindings, for fast file-level diff."""
    per_file: dict[str, list] = {}
    for b in bindings:
        fp = _relpath(b.get("file_path") or "", source)
        if not fp:
            continue
        entry = (*_binding_key(b), b.get("line_number") or 0)
        per_file.setdefault(fp, []).append(entry)

    sigs: dict[str, str] = {}
    for fp, entries in per_file.items():
        entries.sort()
        sigs[fp] = hashlib.blake2b(repr(entries).encode(), digest_size=16).hexdigest()
    return sigs
