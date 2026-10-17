"""Content-addressed artifact preservation across the FLOWBOX HDD and Hugging Face.

An *artifact* is a directory tree described by an immutable manifest (relative
path, size, SHA-256 per file). Destinations hold copies; a *receipt* records
what was proven about one copy, at one of these levels:

* ``COMMITTED``        the bytes were sent and the destination acknowledged them.
* ``REMOTE_HASH``      the destination's own record of size and SHA-256 matches the
                       manifest (Hugging Face LFS/Xet metadata). The bytes were not read back.
* ``BYTES_VERIFIED``   the bytes were read back from the destination into a fresh
                       location and re-hashed by this tool.

Only ``BYTES_VERIFIED`` proves recoverability. Every operation is idempotent:
matching bytes are verified, never rewritten, and conflicting bytes at a path are
refused. Local writes go to a ``.partial`` sibling, are fsynced and re-hashed, then
renamed into place. Each receipt is also appended to a hash-chained audit log
(see ``replicate.audit_append``).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import socket
import sys
import tempfile
import time
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from .replicate import audit_append

SCHEMA = "j1.artifact-manifest.v1"
RECEIPT_SCHEMA = "j1.storage-receipt.v1"
LEVELS = ("COMMITTED", "REMOTE_HASH", "BYTES_VERIFIED")
CLASSES = ("PRESERVE", "RETAIN_UNTIL", "RECOMPUTABLE_TEMPORARY", "QUARANTINE")
HASH_CHUNK = 8 << 20


class StoreError(RuntimeError):
    """A refusal that must stop the caller (conflict, corruption, capacity, access)."""


def now() -> str:
    return datetime.now(UTC).isoformat()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(HASH_CHUNK):
            h.update(chunk)
    return h.hexdigest()


def _canonical(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()


# ---------------------------------------------------------------- manifests


def build_manifest(
    root: Path,
    artifact_id: str,
    artifact_class: str,
    retain_until: str | None = None,
    exclude: Iterable[str] = (),
    description: str = "",
) -> dict:
    if artifact_class not in CLASSES:
        raise StoreError(f"unknown artifact class {artifact_class!r}")
    if artifact_class == "RETAIN_UNTIL" and not retain_until:
        raise StoreError("RETAIN_UNTIL requires retain_until")
    skip = set(exclude)
    files: list[dict[str, Any]] = []
    for p in sorted(root.rglob("*")):
        rel = p.relative_to(root).as_posix()
        if p.is_symlink():
            raise StoreError(f"symlink in artifact tree: {rel}")
        if not p.is_file() or rel in skip or any(part.endswith(".partial") for part in p.parts):
            continue
        files.append({"path": rel, "size": p.stat().st_size, "sha256": sha256_file(p)})
    if not files:
        raise StoreError(f"no files under {root}")
    body = {
        "schema": SCHEMA,
        "artifact_id": artifact_id,
        "class": artifact_class,
        "retain_until": retain_until,
        "description": description,
        "files": files,
        "n_files": len(files),
        "total_bytes": sum(int(f["size"]) for f in files),
    }
    return {**body, "manifest_sha256": hashlib.sha256(_canonical(body)).hexdigest()}


def check_manifest(m: dict) -> dict:
    """Validates schema, path safety and the manifest self-hash; returns ``m``."""
    if m.get("schema") != SCHEMA:
        raise StoreError(f"unsupported manifest schema {m.get('schema')!r}")
    body = {k: v for k, v in m.items() if k != "manifest_sha256"}
    if hashlib.sha256(_canonical(body)).hexdigest() != m.get("manifest_sha256"):
        raise StoreError("manifest self-hash mismatch")
    for f in m["files"]:
        p = Path(f["path"])
        if p.is_absolute() or ".." in p.parts or f["path"] != p.as_posix():
            raise StoreError(f"unsafe path in manifest: {f['path']!r}")
    return m


def load_manifest(path: Path) -> dict:
    return check_manifest(json.loads(path.read_text()))


def verify_tree(m: dict, root: Path) -> list[str]:
    """Re-hashes every manifest file under ``root``; returns problems (empty = intact)."""
    problems = []
    for f in m["files"]:
        p = root / f["path"]
        if not p.is_file():
            problems.append(f"missing {f['path']}")
        elif p.stat().st_size != f["size"]:
            problems.append(f"size mismatch {f['path']}")
        elif sha256_file(p) != f["sha256"]:
            problems.append(f"sha256 mismatch {f['path']}")
    return problems


# ---------------------------------------------------------------- destinations


@dataclass
class RemoteFile:
    size: int
    sha256: str | None  # None when the destination keeps no content hash


class Destination(Protocol):
    label: str

    def stat(self, rel: str) -> RemoteFile | None: ...
    def put(self, files: list[tuple[Path, str]], message: str) -> str: ...
    def fetch(self, rel: str, dst: Path) -> None: ...
    def free_bytes(self) -> int | None: ...


class LocalDir:
    """A directory on a mounted filesystem (the FLOWBOX HDD archive)."""

    def __init__(self, root: Path, reserve_bytes: int = 20 << 30):
        self.root = root
        self.reserve = reserve_bytes
        self.label = f"local:{socket.gethostname()}:{root}"

    def _require_mounted(self) -> None:
        if not self.root.is_dir():
            raise StoreError(f"destination root unavailable: {self.root}")

    def stat(self, rel: str) -> RemoteFile | None:
        self._require_mounted()
        p = self.root / rel
        if not p.is_file():
            return None
        return RemoteFile(p.stat().st_size, sha256_file(p))

    def free_bytes(self) -> int | None:
        self._require_mounted()
        return shutil.disk_usage(self.root).free

    def put(self, files: list[tuple[Path, str]], message: str) -> str:
        self._require_mounted()
        need = sum(src.stat().st_size for src, _ in files)
        free = self.free_bytes() or 0
        if free - need < self.reserve:
            raise StoreError(f"insufficient capacity: need {need} + reserve {self.reserve}, free {free}")
        for src, rel in files:
            dst = self.root / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            tmp = dst.with_name(dst.name + ".partial")
            with src.open("rb") as fi, tmp.open("wb") as fo:
                shutil.copyfileobj(fi, fo, HASH_CHUNK)
                fo.flush()
                os.fsync(fo.fileno())
            if sha256_file(tmp) != sha256_file(src):
                tmp.unlink()
                raise StoreError(f"copy verification failed for {rel}")
            tmp.replace(dst)
        return f"local-copy {message!r} @{now()}"

    def fetch(self, rel: str, dst: Path) -> None:
        self._require_mounted()
        src = self.root / rel
        if not src.is_file():
            raise StoreError(f"missing at destination: {rel}")
        shutil.copyfile(src, dst)


class HFRepo:
    """A path prefix in a Hugging Face repository (private by default).

    Uploads go through ``create_commit`` in batches; large files use the client's
    Xet/LFS transfer, so an interrupted batch is retried by re-running the push,
    which skips files whose remote SHA-256 already matches.
    """

    def __init__(self, repo_id: str, repo_type: str = "dataset", prefix: str = "", private: bool = True, api=None):
        from huggingface_hub import HfApi

        self.api = api or HfApi()
        self.repo_id, self.repo_type, self.private = repo_id, repo_type, private
        self.prefix = prefix.strip("/")
        self.label = f"hf:{repo_type}:{repo_id}/{self.prefix}"
        self._ensured = False
        self.attempts, self.backoff_s = 4, 10.0
        self.retries: list[dict] = []

    def _remote(self, rel: str) -> str:
        return f"{self.prefix}/{rel}" if self.prefix else rel

    def ensure_repo(self) -> None:
        if self._ensured:
            return
        self.api.create_repo(self.repo_id, repo_type=self.repo_type, private=self.private, exist_ok=True)
        info = self.api.repo_info(self.repo_id, repo_type=self.repo_type)
        if self.private and not info.private:
            raise StoreError(f"{self.repo_id} is public; refusing to store private artifacts there")
        self._ensured = True

    def stat_many(self, rels: list[str]) -> dict[str, RemoteFile]:
        from huggingface_hub.errors import RepositoryNotFoundError

        out: dict[str, RemoteFile] = {}
        try:
            for i in range(0, len(rels), 200):
                infos = self.api.get_paths_info(
                    self.repo_id, [self._remote(r) for r in rels[i : i + 200]], repo_type=self.repo_type, expand=True
                )
                for info in infos:
                    if not hasattr(info, "size"):
                        continue  # a folder
                    rel = info.path[len(self.prefix) + 1 :] if self.prefix else info.path
                    lfs = getattr(info, "lfs", None)
                    out[rel] = RemoteFile(info.size, lfs.sha256 if lfs else None)
        except RepositoryNotFoundError:
            return {}
        return out

    def stat(self, rel: str) -> RemoteFile | None:
        return self.stat_many([rel]).get(rel)

    def free_bytes(self) -> int | None:
        return None  # the Hub exposes no quota API; ``hf_bytes`` tracks what this tool stored

    def put(self, files: list[tuple[Path, str]], message: str) -> str:
        from huggingface_hub import CommitOperationAdd

        self.ensure_repo()
        commit = ""
        for i in range(0, len(files), 50):
            for attempt in range(1, self.attempts + 1):
                ops = [CommitOperationAdd(self._remote(rel), str(src)) for src, rel in files[i : i + 50]]
                try:
                    info = self.api.create_commit(
                        self.repo_id, ops, commit_message=f"{message} [{i // 50 + 1}]", repo_type=self.repo_type
                    )
                    break
                except (ConnectionError, TimeoutError, OSError) as e:
                    # A batch commit is atomic on the Hub, so retrying the same batch is idempotent.
                    self.retries.append({"batch": i // 50 + 1, "attempt": attempt, "error": repr(e)[:300]})
                    if attempt == self.attempts:
                        raise
                    time.sleep(self.backoff_s * 2 ** (attempt - 1))
            commit = info.oid
        return commit

    def fetch(self, rel: str, dst: Path) -> None:
        from huggingface_hub import hf_hub_download

        with tempfile.TemporaryDirectory(dir=dst.parent) as td:
            got = hf_hub_download(
                self.repo_id, self._remote(rel), repo_type=self.repo_type, local_dir=td, force_download=True
            )
            Path(got).replace(dst)


def parse_destination(spec: str, reserve_bytes: int) -> Destination:
    """``local:/path`` or ``hf:<dataset|model>:<org/name>[/prefix]``."""
    kind, _, rest = spec.partition(":")
    if kind == "local":
        return LocalDir(Path(rest), reserve_bytes)
    if kind == "hf":
        repo_type, _, path = rest.partition(":")
        parts = path.split("/", 2)
        if repo_type not in ("dataset", "model") or len(parts) < 2:
            raise StoreError(f"bad hf destination {spec!r}")
        return HFRepo("/".join(parts[:2]), repo_type, parts[2] if len(parts) > 2 else "")
    raise StoreError(f"unknown destination kind in {spec!r}")


# ---------------------------------------------------------------- operations


def _stat_all(dest: Destination, rels: list[str]) -> dict[str, RemoteFile]:
    if isinstance(dest, HFRepo):
        return dest.stat_many(rels)
    out = {}
    for r in rels:
        s = dest.stat(r)
        if s is not None:
            out[r] = s
    return out


def _receipt(m: dict, dest: Destination, level: str, op: str, detail: dict, started: float) -> dict:
    return {
        "schema": RECEIPT_SCHEMA,
        "artifact_id": m["artifact_id"],
        "manifest_sha256": m["manifest_sha256"],
        "class": m["class"],
        "destination": dest.label,
        "operation": op,
        "level": level,
        "n_files": m["n_files"],
        "total_bytes": m["total_bytes"],
        "host": socket.gethostname(),
        "verified_utc": now(),
        "seconds": round(time.time() - started, 3),
        **detail,
    }


def write_receipt(receipts: Path, rec: dict) -> Path:
    d = receipts / rec["artifact_id"]
    d.mkdir(parents=True, exist_ok=True)
    slug = rec["destination"].replace("/", "_").replace(":", "_")
    path = d / f"{slug}.{rec['operation']}.{rec['level']}.{int(time.time() * 1000)}.json"
    path.write_text(json.dumps(rec, indent=1, sort_keys=True))
    audit_append(receipts, {"event": "RECEIPT", "file": path.name, "sha256": sha256_file(path), **rec})
    return path


def push(m: dict, src_root: Path, dest: Destination, receipts: Path) -> dict:
    """Copies the artifact to ``dest`` and proves the strongest level the destination allows."""
    started = time.time()
    if isinstance(dest, HFRepo):
        dest.retries = []
    problems = verify_tree(m, src_root)
    if problems:
        raise StoreError(f"source does not match manifest: {problems[:5]}")
    rels = [f["path"] for f in m["files"]]
    remote = _stat_all(dest, rels)
    todo, present = [], 0
    for f in m["files"]:
        r = remote.get(f["path"])
        if r is None:
            todo.append((src_root / f["path"], f["path"]))
        elif r.size != f["size"] or (r.sha256 is not None and r.sha256 != f["sha256"]):
            raise StoreError(f"conflicting bytes already at destination: {f['path']}")
        else:
            present += 1
    commit = dest.put(todo, f"{m['artifact_id']} {m['manifest_sha256'][:12]}") if todo else ""
    # The manifest travels with the artifact so a restore needs nothing else.
    man_rel = f"_manifests/{m['artifact_id']}.{m['manifest_sha256'][:16]}.json"
    if dest.stat(man_rel) is None:
        with tempfile.TemporaryDirectory() as td:
            mp = Path(td) / "m.json"
            mp.write_text(json.dumps(m, indent=1, sort_keys=True))
            dest.put([(mp, man_rel)], f"manifest {m['artifact_id']}")
    remote = _stat_all(dest, rels)
    unhashed = []
    for f in m["files"]:
        r = remote.get(f["path"])
        if r is None or r.size != f["size"] or (r.sha256 is not None and r.sha256 != f["sha256"]):
            raise StoreError(f"post-push check failed for {f['path']}")
        if r.sha256 is None:
            unhashed.append(f["path"])
    level = "REMOTE_HASH" if not unhashed else "COMMITTED"
    if isinstance(dest, LocalDir):
        level = "BYTES_VERIFIED"  # LocalDir.stat re-hashes the bytes on disk
    rec = _receipt(
        m,
        dest,
        level,
        "push",
        {
            "uploaded_files": len(todo),
            "already_present": present,
            "commit": commit,
            "unhashed_paths": unhashed[:20],
            "retries": getattr(dest, "retries", []),
        },
        started,
    )
    write_receipt(receipts, rec)
    return rec


def pull(m: dict, dest: Destination, out_root: Path, receipts: Path, reserve_bytes: int = 20 << 30) -> dict:
    """Restores the artifact from ``dest`` into ``out_root`` (must not exist), re-hashing every file."""
    started = time.time()
    if out_root.exists():
        raise StoreError(f"restore target exists: {out_root}")
    out_root.parent.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(out_root.parent).free
    if free - m["total_bytes"] < reserve_bytes:
        raise StoreError(f"insufficient capacity for restore: need {m['total_bytes']}, free {free}")
    partial = out_root.with_name(out_root.name + ".partial")
    partial.mkdir(exist_ok=True)  # an earlier interrupted pull is resumed, not trusted
    reused = 0
    for f in m["files"]:
        dst = partial / f["path"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.is_file() and dst.stat().st_size == f["size"] and sha256_file(dst) == f["sha256"]:
            reused += 1
            continue
        dest.fetch(f["path"], dst)
        if dst.stat().st_size != f["size"] or sha256_file(dst) != f["sha256"]:
            bad = dst.with_name(dst.name + ".corrupt")
            dst.replace(bad)
            raise StoreError(f"downloaded bytes do not match manifest: {f['path']} (kept as {bad.name})")
    (partial / "_manifest.json").write_text(json.dumps(m, indent=1, sort_keys=True))
    partial.rename(out_root)
    rec = _receipt(m, dest, "BYTES_VERIFIED", "pull", {"restored_to": str(out_root), "reused_partial": reused}, started)
    write_receipt(receipts, rec)
    return rec


def status(receipts: Path) -> dict[str, dict[str, str]]:
    """Best proven level per artifact manifest and destination, from the receipts on disk."""
    best: dict[str, dict[str, str]] = {}
    for p in sorted(receipts.glob("*/*.json")):
        r = json.loads(p.read_text())
        key = f"{r['artifact_id']}@{r['manifest_sha256'][:12]}"
        cur = best.setdefault(key, {}).get(r["destination"])
        if cur is None or LEVELS.index(r["level"]) > LEVELS.index(cur):
            best[key][r["destination"]] = r["level"]
    return best


def hf_bytes(receipts: Path) -> dict[str, int]:
    """Bytes this tool has stored per Hugging Face namespace (one count per manifest and repo)."""
    seen: dict[tuple[str, str], int] = {}
    for p in receipts.glob("*/*.json"):
        r = json.loads(p.read_text())
        if r["destination"].startswith("hf:") and r["operation"] == "push":
            repo = "/".join(r["destination"].split(":", 2)[2].split("/")[:2])
            seen[(repo, r["manifest_sha256"])] = r["total_bytes"]
    out: dict[str, int] = {}
    for (repo, _), n in seen.items():
        ns = repo.split("/")[0]
        out[ns] = out.get(ns, 0) + n
    return out


# ---------------------------------------------------------------- CLI


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m j1bench.store", description=__doc__.split("\n\n")[0])
    ap.add_argument("--receipts", type=Path, required=False, default=None)
    ap.add_argument("--reserve-gb", type=float, default=20.0)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("manifest")
    b.add_argument("root", type=Path)
    b.add_argument("--id", required=True)
    b.add_argument("--class", dest="cls", required=True, choices=CLASSES)
    b.add_argument("--retain-until")
    b.add_argument("--description", default="")
    b.add_argument("--out", type=Path, required=True)
    v = sub.add_parser("verify")
    v.add_argument("manifest", type=Path)
    v.add_argument("root", type=Path)
    for name in ("push", "pull"):
        s = sub.add_parser(name)
        s.add_argument("manifest", type=Path)
        s.add_argument("dest")
        s.add_argument("root", type=Path)
    sub.add_parser("status")
    a = ap.parse_args(argv)
    reserve = int(a.reserve_gb * (1 << 30))
    try:
        if a.cmd == "manifest":
            if a.out.exists():
                raise StoreError(f"manifest exists: {a.out}")
            m = build_manifest(a.root, a.id, a.cls, a.retain_until, description=a.description)
            a.out.write_text(json.dumps(m, indent=1, sort_keys=True))
            print(json.dumps({k: m[k] for k in ("artifact_id", "n_files", "total_bytes", "manifest_sha256")}))
            return 0
        if a.cmd == "verify":
            problems = verify_tree(load_manifest(a.manifest), a.root)
            print(json.dumps({"ok": not problems, "problems": problems[:50]}))
            return 0 if not problems else 2
        if a.receipts is None:
            raise StoreError("--receipts is required")
        if a.cmd == "status":
            print(
                json.dumps({"levels": status(a.receipts), "hf_bytes": hf_bytes(a.receipts)}, indent=1, sort_keys=True)
            )
            return 0
        m = load_manifest(a.manifest)
        dest = parse_destination(a.dest, reserve)
        rec = push(m, a.root, dest, a.receipts) if a.cmd == "push" else pull(m, dest, a.root, a.receipts, reserve)
        print(json.dumps(rec, sort_keys=True))
        return 0
    except Exception as e:  # every failure is logged, then reported through the exit status
        if a.receipts is not None and a.cmd in ("push", "pull"):
            a.receipts.mkdir(parents=True, exist_ok=True)
            audit_append(
                a.receipts, {"event": "FAILED", "cmd": a.cmd, "dest": a.dest, "error": repr(e)[:500], "utc": now()}
            )
        print(json.dumps({"error": repr(e)[:500]}), file=sys.stderr)
        return 2 if isinstance(e, StoreError) else 3


if __name__ == "__main__":
    sys.exit(main())
