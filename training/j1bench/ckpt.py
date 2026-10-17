"""Full-state resumable checkpoints with manifest-last publication.

The primary destination (RunPod global volume, object-backed via geesefs) has no
atomic rename and is eventually consistent, so publication cannot rely on
rename. Instead:

1. Every checkpoint gets a fresh directory name (step + random uid); nothing is
   ever overwritten.
2. Payload files are serialized to local staging, hashed, copied to the
   destination, and re-hashed from the destination.
3. ``manifest.json`` is written after all payload files verify. It lists every
   file's size and sha256 and carries a hash of its own canonical body.
4. After the manifest itself verifies, a ``COMPLETE`` marker holding the
   manifest hash is written. This is the publication point.

A checkpoint is valid only if the COMPLETE marker names the manifest hash, the
manifest parses, its self-hash matches, and every listed file exists with the
recorded size and hash. Directories without a
valid manifest are incomplete and are never selected for restore, ``latest``
or ``best``. ``latest.json`` is an advisory hint only; selection always
re-verifies.
"""

from __future__ import annotations

import contextlib
import errno
import hashlib
import json
import os
import re
import secrets
import shutil
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

MANIFEST = "manifest.json"
COMPLETE = "COMPLETE"
SCHEMA = "j1bench.ckpt.v1"
PAYLOAD = ("model.pt", "optim.pt", "trainer.pt")


class CheckpointError(RuntimeError):
    pass


def sha256_file(path: Path, bufsize: int = 16 << 20) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while b := f.read(bufsize):
            h.update(b)
    return h.hexdigest()


def canonical(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()


def _fault() -> str:
    return os.environ.get("J1_FAULT", "")


def _copy_verified(src: Path, dst: Path, expect_sha: str, *, file_index: int) -> None:
    """Copies src to dst, fsyncs, and re-hashes dst. Fault hooks simulate crashes and ENOSPC."""
    fault = _fault()
    limit = None
    if fault.startswith("enospc_after_bytes:"):
        limit = int(fault.split(":")[1])
    written = 0
    with Path(src).open("rb") as fi, Path(dst).open("wb") as fo:
        while b := fi.read(16 << 20):
            if limit is not None and written + len(b) > limit:
                raise OSError(errno.ENOSPC, "No space left on device (injected)", str(dst))
            fo.write(b)
            written += len(b)
            if fault == f"die_mid_file:{file_index}" and written >= 16 << 20:
                fo.flush()
                os._exit(137)
        fo.flush()
        os.fsync(fo.fileno())
    got = sha256_file(dst)
    if got != expect_sha:
        raise CheckpointError(f"post-copy hash mismatch for {dst}: {got} != {expect_sha}")


@dataclass
class SaveResult:
    path: Path
    manifest: dict
    timings: dict = field(default_factory=dict)


def save_checkpoint(root: Path, staging: Path, *, step: int, payload: dict, meta: dict, torch_save) -> SaveResult:
    """Serializes ``payload`` (name -> object) and publishes it under ``root``.

    ``meta`` must contain run_id and run_contract_sha256; the rest is recorded verbatim.
    """
    t0 = time.perf_counter()
    uid = secrets.token_hex(4)
    name = f"ckpt-{step:09d}-{uid}"
    stage_dir = staging / name
    stage_dir.mkdir(parents=True, exist_ok=False)
    files: dict[str, dict[str, Any]] = {}
    for fname in PAYLOAD:
        p = stage_dir / fname
        torch_save(payload[fname], p)
        with Path(p).open("rb") as f:
            os.fsync(f.fileno())
    t_ser = time.perf_counter()
    for fname in PAYLOAD:
        p = stage_dir / fname
        files[fname] = {"size": p.stat().st_size, "sha256": sha256_file(p)}
    t_hash = time.perf_counter()

    dest = Path(root) / meta["run_id"] / name
    dest.mkdir(parents=True, exist_ok=False)
    for i, fname in enumerate(PAYLOAD):
        if _fault() == f"die_after_files:{i}":
            os._exit(137)
        _copy_verified(stage_dir / fname, dest / fname, files[fname]["sha256"], file_index=i)
    t_copy = time.perf_counter()
    if _fault() == f"die_after_files:{len(PAYLOAD)}":
        os._exit(137)

    body = {
        "schema": SCHEMA,
        "name": name,
        "step": step,
        "created_utc": datetime.now(UTC).isoformat(),
        "files": files,
        **meta,
    }
    manifest = dict(body, manifest_sha256=hashlib.sha256(canonical(body)).hexdigest())
    mpath = dest / MANIFEST
    with Path(mpath).open("wb") as f:
        f.write(json.dumps(manifest, indent=1, sort_keys=True).encode())
        f.flush()
        os.fsync(f.fileno())
    ok, why = verify_checkpoint(dest, require_complete=False)
    if not ok:
        raise CheckpointError(f"checkpoint failed verification before publication: {why}")
    if _fault() == "die_before_complete":
        os._exit(137)
    with Path(dest / COMPLETE).open("wb") as f:
        f.write(manifest["manifest_sha256"].encode())
        f.flush()
        os.fsync(f.fileno())
    ok, why = verify_checkpoint(dest, check_hashes=False)
    if not ok:
        raise CheckpointError(f"published checkpoint failed verification: {why}")
    t_pub = time.perf_counter()
    _write_hint(Path(root) / meta["run_id"], manifest)
    shutil.rmtree(stage_dir, ignore_errors=True)
    timings = {
        "serialize_s": t_ser - t0,
        "hash_s": t_hash - t_ser,
        "copy_verify_s": t_copy - t_hash,
        "publish_verify_s": t_pub - t_copy,
        "total_s": time.perf_counter() - t0,
        "bytes": sum(v["size"] for v in files.values()),
    }
    return SaveResult(dest, manifest, timings)


def _write_hint(run_dir: Path, manifest: dict) -> None:
    # Advisory only: a stale or torn hint is harmless because selection re-verifies.
    with contextlib.suppress(OSError):
        (run_dir / "latest.json").write_text(json.dumps({"name": manifest["name"], "step": manifest["step"]}))


def read_manifest(ckpt_dir: Path) -> dict | None:
    try:
        manifest = json.loads((Path(ckpt_dir) / MANIFEST).read_text())
        return manifest if isinstance(manifest, dict) else None
    except (OSError, ValueError):
        return None


def verify_checkpoint(ckpt_dir: Path, *, check_hashes: bool = True, require_complete: bool = True) -> tuple[bool, str]:
    ckpt_dir = Path(ckpt_dir)
    m = read_manifest(ckpt_dir)
    if m is None:
        return False, "manifest missing or unparseable (incomplete checkpoint)"
    claimed = m.get("manifest_sha256")
    body = {k: v for k, v in m.items() if k != "manifest_sha256"}
    if claimed != hashlib.sha256(canonical(body)).hexdigest():
        return False, "manifest self-hash mismatch"
    if require_complete:
        try:
            marker = (ckpt_dir / COMPLETE).read_text().strip()
        except OSError:
            return False, "completion marker missing (unpublished checkpoint)"
        if marker != claimed:
            return False, "completion marker does not match manifest"
    if m.get("schema") != SCHEMA:
        return False, f"unknown schema {m.get('schema')!r}"
    if (
        m.get("name") != ckpt_dir.name
        or not isinstance(m.get("step"), int)
        or isinstance(m.get("step"), bool)
        or m["step"] < 0
        or not isinstance(m.get("created_utc"), str)
        or not isinstance(m.get("run_id"), str)
        or not isinstance(m.get("run_contract_sha256"), str)
        or not isinstance(m.get("files"), dict)
        or set(m["files"]) != set(PAYLOAD)
    ):
        return False, "malformed checkpoint manifest"
    for fname, rec in m["files"].items():
        if (
            not isinstance(rec, dict)
            or not isinstance(rec.get("size"), int)
            or isinstance(rec.get("size"), bool)
            or rec["size"] < 0
            or not isinstance(rec.get("sha256"), str)
            or re.fullmatch(r"[0-9a-f]{64}", rec["sha256"]) is None
        ):
            return False, f"malformed file record {fname}"
        p = ckpt_dir / fname
        if not p.exists():
            return False, f"missing file {fname}"
        if p.stat().st_size != rec["size"]:
            return False, f"size mismatch {fname}: {p.stat().st_size} != {rec['size']}"
        if check_hashes and sha256_file(p) != rec["sha256"]:
            return False, f"sha256 mismatch {fname}"
    return True, "ok"


def scan(run_dir: Path) -> list[Path]:
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        return []
    return sorted(
        (p for p in run_dir.iterdir() if p.is_dir() and p.name.startswith("ckpt-")), key=lambda p: p.name, reverse=True
    )


def select_latest_valid(
    run_dirs: list[Path], *, contract_sha: str | None = None, log=print
) -> tuple[Path | None, list[dict]]:
    """Returns the newest verified checkpoint across the given run directories.

    Candidates are ordered by (step, created_utc). A checkpoint whose run contract
    differs from ``contract_sha`` is reported as incompatible and skipped.
    """
    report, cands = [], []
    for rd in run_dirs:
        for p in scan(rd):
            m = read_manifest(p)
            # An unreadable manifest still sorts by the step in its directory name,
            # so it is reported (and rejected) in order rather than silently skipped.
            directory_step = p.name.split("-")[1]
            step = m.get("step") if m and isinstance(m.get("step"), int) else None
            if step is None or isinstance(step, bool):
                step = int(directory_step) if directory_step.isdecimal() else -1
            created = m.get("created_utc") if m and isinstance(m.get("created_utc"), str) else ""
            cands.append((step, created, p))
    cands.sort(key=lambda c: (c[0], c[1]), reverse=True)
    for step, _, p in cands:
        ok, why = verify_checkpoint(p)
        if ok and contract_sha is not None:
            got = (read_manifest(p) or {}).get("run_contract_sha256")
            if got != contract_sha:
                ok, why = False, f"incompatible run contract {got} (expected {contract_sha})"
        report.append({"path": str(p), "step": step, "valid": ok, "reason": why})
        log(f"[ckpt] candidate {p} step={step}: {why}")
        if ok:
            return p, report
    return None, report


# --- retention ---------------------------------------------------------------


@dataclass(frozen=True)
class RetentionPolicy:
    """v0.3 defaults. Reducing any of them needs a recorded owner-approved change."""

    keep_last: int = 3
    hourly_active_hours: float = 24.0  # newest checkpoint per active-training hour, preceding 24 active hours
    daily_for_run: bool = True  # newest checkpoint per UTC day for the whole run
    protected_tags: tuple = (
        "init",
        "stage_boundary",
        "best",
        "release_candidate",
        "published_result",
        "pinned",
        "incident",
        "planned_shutdown",
    )


DEFAULT_RETENTION = RetentionPolicy()


def retention_plan(entries: list[dict], policy: RetentionPolicy = DEFAULT_RETENTION) -> dict:
    """entries: [{name, step, created_utc, active_s, tags, valid}] -> {name: [keep reasons]}.

    Names absent from the result are prunable, but only after the caller has
    confirmed that every kept checkpoint and its replica verify. Invalid or
    incomplete directories are kept as quarantined evidence and never become
    latest-valid or best.
    """
    keep: dict[str, list[str]] = {}
    valid = sorted((e for e in entries if e["valid"]), key=lambda e: (e["step"], e["created_utc"]), reverse=True)
    for e in entries:
        if not e["valid"]:
            keep.setdefault(e["name"], []).append("quarantined_invalid")
    for e in valid[: policy.keep_last]:
        keep.setdefault(e["name"], []).append("latest_n")
    if valid:
        keep.setdefault(valid[0]["name"], []).append("latest_valid")
        now_active = valid[0]["active_s"]
    for e in valid:
        for t in e.get("tags", []):
            if t in policy.protected_tags:
                keep.setdefault(e["name"], []).append(t)
    seen_h, seen_d = set(), set()
    for e in valid:
        hb = int(e["active_s"] // 3600)
        if now_active - e["active_s"] <= policy.hourly_active_hours * 3600 and hb not in seen_h:
            seen_h.add(hb)
            keep.setdefault(e["name"], []).append("hourly")
        db = datetime.fromisoformat(e["created_utc"]).strftime("%Y%m%d")
        if policy.daily_for_run and db not in seen_d:
            seen_d.add(db)
            keep.setdefault(e["name"], []).append("daily")
    return keep


def best_checkpoint(entries: list[dict], metric: str = "eval_loss") -> str | None:
    vals = [e for e in entries if e["valid"] and e.get(metric) is not None]
    return min(vals, key=lambda e: e[metric])["name"] if vals else None


def prune_pilot(run_dir: Path) -> list[str]:
    """Pilot retention (docs/prep/14 §2; review T-12): keep the initialization checkpoint (step 0 or tag
    ``init``), the latest hash-verified checkpoint, the newest ``best``, every ``final``, and every directory that
    does not verify by content hash (quarantined evidence); delete the other verified ones. Validity is decided by
    ``verify_checkpoint(check_hashes=True)``, so a checkpoint whose sizes match but whose bytes differ is never
    chosen as latest and never deleted. Nothing is deleted while fewer than two checkpoints verify by hash.
    Never applies to canonical training, whose retention is ``retention_plan``."""
    import shutil

    dirs = scan(run_dir)
    valid: list[tuple[Path, dict]] = []
    keep: set[Path] = set()
    for d in dirs:
        ok, _ = verify_checkpoint(d, check_hashes=True)
        m = read_manifest(d) if ok else None
        if m is None:
            keep.add(d)  # quarantined evidence
        else:
            valid.append((d, m))
    if len(valid) < 2:
        return []
    valid.sort(key=lambda e: e[1]["step"])
    keep.add(valid[-1][0])
    keep.update(d for d, m in valid if m["step"] == 0 or "init" in m.get("tags", []))
    best = [d for d, m in valid if "best" in m.get("tags", [])]
    if best:
        keep.add(best[-1])
    keep.update(d for d, m in valid if "final" in m.get("tags", []))
    deleted = []
    for d in dirs:
        if d not in keep:
            shutil.rmtree(d)
            deleted.append(d.name)
    return deleted
