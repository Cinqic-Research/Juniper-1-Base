"""FLOWBOX-initiated pull of published checkpoints into the local HDD archive.

Runs on FLOWBOX; the pod never gets credentials for FLOWBOX. For each remote
checkpoint that has a COMPLETE marker:

1. rsync (resumable, --partial) into ``<archive>/checkpoints/<run>/.incoming/<name>``
2. verify locally: completion marker, manifest self-hash, every file's size and sha256
3. atomically rename into ``<archive>/checkpoints/<run>/<name>`` (ext4 rename)
4. write a machine-readable receipt and append a hash-chained audit-log line

Existing archived checkpoints are re-verified, never overwritten. A conflicting
or corrupt transfer goes to ``quarantine/`` with a FAILED receipt. Exit status is
nonzero if any checkpoint failed, so the caller cannot mistake a failure for success.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .ckpt import MANIFEST, read_manifest, verify_checkpoint


def now() -> str:
    return datetime.now(UTC).isoformat()


def audit_append(archive: Path, rec: dict) -> None:
    """Appends one JSON line whose ``prev`` field is the sha256 of the previous line."""
    log = archive / "audit.log"
    prev = "0" * 64
    if log.exists():
        lines = log.read_bytes().splitlines()
        if lines:
            prev = hashlib.sha256(lines[-1]).hexdigest()
    line = json.dumps({"prev": prev, **rec}, sort_keys=True).encode()
    with Path(log).open("ab") as f:
        f.write(line + b"\n")
        f.flush()
        os.fsync(f.fileno())


def audit_verify(archive: Path) -> tuple[bool, int]:
    log = archive / "audit.log"
    if not log.exists():
        return True, 0
    prev = "0" * 64
    lines = log.read_bytes().splitlines()
    for i, line in enumerate(lines):
        if json.loads(line)["prev"] != prev:
            return False, i
        prev = hashlib.sha256(line).hexdigest()
    return True, len(lines)


def host_identity(archive: Path) -> dict:
    src = subprocess.run(
        ["findmnt", "-no", "SOURCE,UUID,FSTYPE", "--target", str(archive)], capture_output=True, text=True
    ).stdout.split()
    return {
        "host": socket.gethostname(),
        "mount_source": src[0] if src else None,
        "fs_uuid": src[1] if len(src) > 1 else None,
        "fstype": src[2] if len(src) > 2 else None,
        "archive_root": str(archive),
    }


def remote_published(host: str, remote_run_dir: str) -> list[str]:
    r = subprocess.run(
        [
            "ssh",
            host,
            f"ls -1 {remote_run_dir} 2>/dev/null | grep '^ckpt-' | while read d; do "
            f"[ -f {remote_run_dir}/$d/COMPLETE ] && echo $d; done",
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if r.returncode not in (0, 1):
        raise RuntimeError(f"remote listing failed: {r.stderr.strip()}")
    return sorted(r.stdout.split())


def pull_one(host: str, remote_run_dir: str, name: str, archive: Path, run_id: str, retries: int = 2) -> dict:
    base = archive / "checkpoints" / run_id
    final = base / name
    incoming = base / ".incoming" / name
    receipts = archive / "receipts" / run_id
    receipts.mkdir(parents=True, exist_ok=True)
    rec: dict[str, Any] = {
        "artifact": "checkpoint",
        "run_id": run_id,
        "checkpoint": name,
        "source": f"{host}:{remote_run_dir}/{name}",
        "local_path": str(final),
        "storage_class": "flowbox_hdd_archive",
        "host_identity": host_identity(archive),
        "attempts": [],
    }
    if final.exists():
        ok, why = verify_checkpoint(final)
        rec.update(
            status="ALREADY_ARCHIVED_VERIFIED" if ok else "ARCHIVED_COPY_INVALID", reason=why, verified_utc=now()
        )
        return rec
    incoming.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for attempt in range(1, retries + 2):
        # Payload first, manifest and completion marker last (rsync otherwise sends
        # files alphabetically, so COMPLETE would land before model.pt/optim.pt).
        src = f"{host}:{remote_run_dir}/{name}/"
        r = subprocess.run(
            [
                "rsync",
                "-a",
                "--partial",
                "--inplace",
                "--timeout=120",
                "-e",
                "ssh",
                "--exclude",
                "COMPLETE",
                "--exclude",
                MANIFEST,
                src,
                f"{incoming}/",
            ],
            capture_output=True,
            text=True,
        )
        if r.returncode == 0:
            r = subprocess.run(
                ["rsync", "-a", "--timeout=120", "-e", "ssh", src, f"{incoming}/"], capture_output=True, text=True
            )
        rec["attempts"].append(
            {"attempt": attempt, "rc": r.returncode, "stderr": r.stderr.strip()[-300:], "utc": now()}
        )
        if r.returncode == 0:
            break
    transfer_s = time.time() - t0
    m = read_manifest(incoming)
    ok, why = verify_checkpoint(incoming)
    nbytes = sum(f.stat().st_size for f in incoming.iterdir() if f.is_file())
    rec.update(transfer_s=transfer_s, bytes=nbytes, MBps=nbytes / 1e6 / max(transfer_s, 1e-9))
    if m:
        rec["expected"] = dict(m["files"])
        rec["observed"] = {
            f: {
                "size": (incoming / f).stat().st_size,
                "sha256": hashlib.sha256((incoming / f).read_bytes()).hexdigest(),
            }
            for f in m["files"]
            if (incoming / f).exists()
        }
        rec["manifest_sha256"] = m.get("manifest_sha256")
        rec["step"] = m.get("step")
        created = datetime.fromisoformat(m["created_utc"])
        rec["replication_lag_s"] = (datetime.now(UTC) - created).total_seconds()
    if ok:
        final.parent.mkdir(parents=True, exist_ok=True)
        incoming.rename(final)
        fd = os.open(final.parent, os.O_RDONLY)
        os.fsync(fd)
        os.close(fd)
        ok2, why2 = verify_checkpoint(final, check_hashes=False)
        rec.update(status="VERIFIED" if ok2 else "FAILED", reason=why2, verified_utc=now())
    else:
        q = archive / "quarantine" / run_id / f"{name}.{int(time.time())}"
        q.parent.mkdir(parents=True, exist_ok=True)
        if r.returncode == 0:
            shutil.move(str(incoming), q)
            rec["quarantine_path"] = str(q)
        rec.update(status="FAILED", reason=why, verified_utc=now())
    (receipts / f"{name}.{rec['status']}.{int(time.time())}.json").write_text(json.dumps(rec, indent=1))
    audit_append(
        archive,
        {
            k: rec.get(k)
            for k in (
                "utc",
                "artifact",
                "run_id",
                "checkpoint",
                "status",
                "reason",
                "manifest_sha256",
                "bytes",
                "replication_lag_s",
            )
        }
        | {"utc": now()},
    )
    return rec


def main(argv=None):
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument("--host", default="j1pod")
    a.add_argument("--remote-root", required=True, help="remote checkpoint root, e.g. /workspace/j1ckpt")
    a.add_argument("--run-id", required=True)
    a.add_argument("--archive", default=os.environ.get("FLOWBOX_ARCHIVE_ROOT"), required=False)
    a.add_argument("--only", action="append", default=[], help="pull only these checkpoint names")
    a.add_argument("--retries", type=int, default=2)
    args = a.parse_args(argv)
    if not args.archive:
        sys.exit("set --archive or FLOWBOX_ARCHIVE_ROOT")
    archive = Path(args.archive)
    if not archive.is_dir():
        sys.exit(f"archive root {archive} is not mounted or missing; refusing to create it implicitly")
    remote_run_dir = f"{args.remote_root}/{args.run_id}"
    try:
        names = args.only or remote_published(args.host, remote_run_dir)
    except (RuntimeError, subprocess.TimeoutExpired) as e:
        audit_append(
            archive,
            {
                "utc": now(),
                "artifact": "listing",
                "run_id": args.run_id,
                "status": "SOURCE_UNREACHABLE",
                "reason": str(e)[:300],
                "source": f"{args.host}:{remote_run_dir}",
            },
        )
        sys.exit(f"source unreachable; nothing verified: {e}")
    failed = 0
    for n in names:
        r = pull_one(args.host, remote_run_dir, n, archive, args.run_id, args.retries)
        failed += r["status"] not in ("VERIFIED", "ALREADY_ARCHIVED_VERIFIED")
        print(
            json.dumps(
                {
                    k: r.get(k)
                    for k in ("checkpoint", "status", "reason", "bytes", "transfer_s", "MBps", "replication_lag_s")
                }
            ),
            flush=True,
        )
    ok, n = audit_verify(archive)
    print(json.dumps({"audit_chain_ok": ok, "audit_lines": n, "failed": failed}))
    sys.exit(1 if failed or not ok else 0)


if __name__ == "__main__":
    main()
