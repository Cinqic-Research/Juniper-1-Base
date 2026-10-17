"""Acquisition of pinned upstream revisions.

Hugging Face files are listed at the pinned commit, downloaded with HTTP Range
resumption into ``<name>.part`` and promoted only after the byte count and the
upstream LFS SHA-256 both match. Git sources are recorded by commit and read
with ``git show`` at processing time, so no working-tree state leaks in.

The acquisition manifest records, per file, the pinned URL, size, SHA-256 and
retrieval time. Re-running is idempotent: verified files are re-hashed and kept.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import subprocess
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote

from j1data.hashing import sha256_file, sha256_json
from j1data.paths import DataLayout, safe_child
from j1data.registry import Source

VERSION = "j1.acquire.v1"
HF_ENDPOINT = "https://huggingface.co"
_CHUNK = 1 << 22


def _token() -> str | None:
    token = os.environ.get("HF_TOKEN")
    if token:
        return token
    path = Path.home() / ".cache" / "huggingface" / "token"
    return path.read_text().strip() if path.exists() else None


def _request(url: str, headers: dict[str, str] | None = None) -> urllib.request.Request:
    request = urllib.request.Request(url, headers={"User-Agent": "j1data/0.1", **(headers or {})})
    token = _token()
    if token and url.startswith(HF_ENDPOINT):
        request.add_header("Authorization", f"Bearer {token}")
    return request


def list_hf_files(repo: str, revision: str) -> list[dict[str, Any]]:
    url = f"{HF_ENDPOINT}/api/datasets/{repo}/tree/{revision}?recursive=true"
    entries: list[dict[str, Any]] = []
    while url:
        with urllib.request.urlopen(_request(url), timeout=60) as response:
            entries.extend(json.load(response))
            link = response.headers.get("Link", "")
        url = ""
        for part in link.split(","):
            if 'rel="next"' in part:
                url = part.split(";")[0].strip().strip("<>")
    return [e for e in entries if e.get("type") == "file"]


def download_file(url: str, target: Path, size: int, sha256: str | None, attempts: int = 8) -> str:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink():
        raise ValueError(f"refusing to download through a symlink: {target}")
    if target.exists() and target.stat().st_size == size:
        digest = sha256_file(target)
        if sha256 is None or digest == sha256:
            return digest
        target.unlink()
    partial = target.with_name(target.name + ".part")
    if partial.is_symlink():
        raise ValueError(f"refusing to resume through a symlink: {partial}")
    for attempt in range(attempts):
        offset = partial.stat().st_size if partial.exists() else 0
        if offset > size:
            partial.unlink()
            offset = 0
        if offset < size:
            try:
                headers = {"Range": f"bytes={offset}-"} if offset else {}
                with urllib.request.urlopen(_request(url, headers), timeout=120) as response:
                    if offset and response.status != 206:
                        offset = 0
                    with partial.open("ab" if offset else "wb") as handle:
                        while chunk := response.read(_CHUNK):
                            handle.write(chunk)
            except (urllib.error.URLError, TimeoutError, ConnectionError) as error:
                time.sleep(min(60, 2**attempt))
                if attempt == attempts - 1:
                    raise RuntimeError(f"download failed after {attempts} attempts: {url}") from error
                continue
        if partial.stat().st_size != size:
            continue
        digest = sha256_file(partial)
        if sha256 is not None and digest != sha256:
            partial.unlink()
            raise RuntimeError(f"checksum mismatch for {url}")
        partial.replace(target)
        return digest
    raise RuntimeError(f"incomplete download: {url}")


def acquire_hf(source: Source, layout: DataLayout, workers: int = 3) -> dict[str, Any]:
    listing = {e["path"]: e for e in list_hf_files(source.repo, source.revision)}
    selected = source.selection().apply(list(listing))
    if not selected:
        raise RuntimeError(f"{source.source_id}: selection matched no files")
    raw_dir = layout.raw_dir(source.source_id, source.revision)

    def fetch(path: str) -> dict[str, Any]:
        entry = listing[path]
        upstream = (entry.get("lfs") or {}).get("oid")
        url = f"{HF_ENDPOINT}/datasets/{source.repo}/resolve/{source.revision}/{quote(path, safe='/')}"
        target = safe_child(raw_dir, path)
        digest = download_file(url, target, int(entry["size"]), upstream)
        git_oid = entry.get("oid")
        if upstream is None and (not isinstance(git_oid, str) or git_blob_oid(target, git_oid) != git_oid):
            target.unlink(missing_ok=True)
            raise RuntimeError(f"Git blob mismatch for {source.source_id}/{path}")
        return {
            "path": path,
            "url": url,
            "size": int(entry["size"]),
            "sha256": digest,
            "upstream_sha256": upstream,
            "git_blob_oid": git_oid,
            "retrieved_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        }

    with ThreadPoolExecutor(max_workers=workers) as pool:
        files = sorted(pool.map(fetch, selected), key=lambda f: f["path"])
    return _write_manifest(source, layout, files, upstream_file_count=len(listing))


def acquire_git(source: Source, layout: DataLayout, clone: Path) -> dict[str, Any]:
    subprocess.run(["git", "-C", str(clone), "cat-file", "-e", f"{source.revision}^{{commit}}"], check=True)
    names = subprocess.run(
        [
            "git",
            "-C",
            str(clone),
            "ls-tree",
            "-r",
            "--format=%(objectname) %(objectsize) %(path)",
            source.revision,
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    blobs = {}
    for line in names:
        oid, size, path = line.split(" ", 2)
        blobs[path] = (oid, int(size))
    selected = source.selection().apply(list(blobs))
    files = [{"path": p, "git_blob_oid": blobs[p][0], "size": blobs[p][1]} for p in selected]
    return _write_manifest(source, layout, files, upstream_file_count=len(blobs))


def _write_manifest(
    source: Source, layout: DataLayout, files: list[dict[str, Any]], upstream_file_count: int
) -> dict[str, Any]:
    manifest = {
        "schema": "j1.acquisition-manifest.v1",
        "acquire_version": VERSION,
        "source_id": source.source_id,
        "mechanism": source.raw["acquisition"]["mechanism"],
        "repo": source.repo,
        "revision": source.revision,
        "selection": source.raw["acquisition"]["files"],
        "upstream_file_count": upstream_file_count,
        "selected_file_count": len(files),
        "selected_bytes": sum(f["size"] for f in files),
        "acquired_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "files": files,
    }
    validate_manifest(source, manifest)
    path = layout.acquisition_manifest(source.source_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    tmp.replace(path)
    return manifest


def validate_manifest(source: Source, manifest: dict[str, Any]) -> None:
    """Check the manifest's logical source identity and selected-file inventory."""
    expected = {
        "schema": "j1.acquisition-manifest.v1",
        "acquire_version": VERSION,
        "source_id": source.source_id,
        "mechanism": source.raw["acquisition"]["mechanism"],
        "repo": source.repo,
        "revision": source.revision,
        "selection": source.raw["acquisition"]["files"],
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"{source.source_id}: acquisition manifest {key} does not match registry")
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError(f"{source.source_id}: acquisition manifest has no files")
    paths: set[str] = set()
    selected = source.selection()
    for entry in files:
        if not isinstance(entry, dict):
            raise ValueError(f"{source.source_id}: malformed acquisition file entry")
        path = entry.get("path")
        if not isinstance(path, str) or path in paths:
            raise ValueError(f"{source.source_id}: invalid or duplicate acquisition path {path!r}")
        paths.add(path)
        safe_child(Path("/validation-root"), path)
        if not selected.apply([path]):
            raise ValueError(f"{source.source_id}: unselected path in acquisition manifest: {path!r}")
        size = entry.get("size")
        if not isinstance(size, int) or size < 0:
            raise ValueError(f"{source.source_id}: invalid size for {path!r}")
        if source.raw["acquisition"]["mechanism"] == "huggingface":
            digest = entry.get("sha256")
            if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
                raise ValueError(f"{source.source_id}: missing or invalid SHA-256 for {path!r}")
            upstream = entry.get("upstream_sha256")
            if upstream is not None and upstream != digest:
                raise ValueError(f"{source.source_id}: upstream SHA-256 mismatch in manifest for {path!r}")
            git_oid = entry.get("git_blob_oid")
            if not isinstance(git_oid, str) or not re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", git_oid):
                raise ValueError(f"{source.source_id}: missing or invalid upstream Git blob id for {path!r}")
        else:
            oid = entry.get("git_blob_oid")
            if not isinstance(oid, str) or not re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", oid):
                raise ValueError(f"{source.source_id}: missing or invalid Git blob id for {path!r}")
    if manifest.get("selected_file_count") != len(files):
        raise ValueError(f"{source.source_id}: acquisition selected-file count does not match inventory")
    upstream_count = manifest.get("upstream_file_count")
    if type(upstream_count) is not int or upstream_count < len(files):
        raise ValueError(f"{source.source_id}: invalid upstream file count")
    if manifest.get("selected_bytes") != sum(entry["size"] for entry in files):
        raise ValueError(f"{source.source_id}: acquisition selected-byte count does not match inventory")


def manifest_identity(source: Source, manifest: dict[str, Any]) -> str:
    """Hash stable source content identities, excluding retrieval timestamps and URLs."""
    validate_manifest(source, manifest)
    files = [
        {key: entry.get(key) for key in ("path", "size", "sha256", "upstream_sha256", "git_blob_oid")}
        for entry in manifest["files"]
    ]
    files.sort(key=lambda entry: entry["path"] or "")
    return sha256_json(
        {
            "schema": manifest["schema"],
            "acquire_version": manifest.get("acquire_version"),
            "source_id": source.source_id,
            "mechanism": source.raw["acquisition"]["mechanism"],
            "repo": source.repo,
            "revision": source.revision,
            "selection": source.raw["acquisition"]["files"],
            "files": files,
        }
    )


def verify_acquired_source(
    source: Source, layout: DataLayout, manifest: dict[str, Any], clone: Path | None = None
) -> dict[str, Any]:
    """Recheck pinned source inventory and local bytes before processing starts."""
    validate_manifest(source, manifest)
    mechanism = source.raw["acquisition"]["mechanism"]
    checked_bytes = 0
    if mechanism == "huggingface":
        raw_dir = layout.raw_dir(source.source_id, source.revision)
        for position, entry in enumerate(manifest["files"], 1):
            path = safe_child(raw_dir, entry["path"])
            # Reject symlinks and non-regular objects, even when their resolved
            # targets happen to remain beneath the data root.
            current = raw_dir
            for part in Path(entry["path"]).parts:
                current = current / part
                mode = current.lstat().st_mode
                if stat.S_ISLNK(mode):
                    raise ValueError(f"{source.source_id}: symlink in acquired path {entry['path']!r}")
            if not stat.S_ISREG(path.lstat().st_mode):
                raise ValueError(
                    f"{source.source_id}: acquired path is not a regular file: {entry['path']!r}"
                )
            actual_size = path.stat().st_size
            if actual_size != entry["size"]:
                raise ValueError(
                    f"{source.source_id}: acquired size mismatch for {entry['path']!r}: "
                    f"{actual_size} != {entry['size']}"
                )
            digest = sha256_file(path)
            if digest != entry["sha256"]:
                raise ValueError(f"{source.source_id}: acquired SHA-256 mismatch for {entry['path']!r}")
            if entry.get("upstream_sha256") is None:
                expected_oid = entry["git_blob_oid"]
                if git_blob_oid(path, expected_oid) != expected_oid:
                    raise ValueError(f"{source.source_id}: acquired Git blob mismatch for {entry['path']!r}")
            checked_bytes += actual_size
            if position % 10 == 0 or position == len(manifest["files"]):
                print(
                    f"{source.source_id}: verified {position}/{len(manifest['files'])} files "
                    f"({checked_bytes / 1e9:.2f} GB)",
                    flush=True,
                )
    else:
        if clone is None:
            raise ValueError(f"{source.source_id}: Git clone is required to verify its pinned source")
        selected = source.selection()
        commit = subprocess.run(
            ["git", "-C", str(clone), "rev-parse", "--verify", f"{source.revision}^{{commit}}"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if commit != source.revision:
            raise ValueError(f"{source.source_id}: clone does not contain the exact pinned commit")
        tree = subprocess.run(
            [
                "git",
                "-C",
                str(clone),
                "ls-tree",
                "-r",
                "--format=%(objectname) %(objectsize) %(path)",
                source.revision,
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
        blobs = {line.split(" ", 2)[2]: tuple(line.split(" ", 2)[:2]) for line in tree}
        expected_paths = set(selected.apply(list(blobs)))
        recorded = {entry["path"]: entry for entry in manifest["files"]}
        if set(recorded) != expected_paths:
            raise ValueError(f"{source.source_id}: Git manifest file selection differs from pinned tree")
        for path, entry in recorded.items():
            oid, size = blobs[path]
            if entry["git_blob_oid"] != oid or entry["size"] != int(size):
                raise ValueError(f"{source.source_id}: Git blob identity mismatch for {path!r}")
            checked_bytes += int(size)
    return {
        "files": len(manifest["files"]),
        "bytes": checked_bytes,
        "manifest_identity": manifest_identity(source, manifest),
    }


def git_blob_oid(path: Path, expected_oid: str) -> str:
    """Compute a Git blob object id for a non-LFS file from its raw bytes."""
    if not isinstance(expected_oid, str) or not re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", expected_oid):
        raise ValueError("invalid upstream Git blob id")
    algorithm = "sha1" if len(expected_oid) == 40 else "sha256"
    digest = hashlib.new(algorithm, usedforsecurity=False)
    size = path.stat().st_size
    digest.update(f"blob {size}\0".encode())
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 22):
            digest.update(chunk)
    return digest.hexdigest()
