"""Juniper 1 time logging and temporal traceability (spec v0.6 §8).

Event streams
    One append-only UTF-8 JSONL segment per producer (one process) under
    ``<log_root>/<run_id>/``. Every event carries the schema fields of §8 "Event schema
    and identity". Inapplicable fields are ``null`` with a reason in ``null_reasons``.

Clocks
    ``timestamp_utc`` is RFC 3339 with milliseconds and a ``Z`` suffix, from the realtime
    clock. Durations come only from ``CLOCK_MONOTONIC`` readings (``monotonic_ns``) taken in
    the same clock domain ``<host_id>:<boot_id>:CLOCK_MONOTONIC``. On Linux,
    CLOCK_MONOTONIC excludes suspend time. A span's elapsed time is never computed across
    clock domains; the terminal event refuses it.

Durability
    Start, terminal, incident, checkpoint, transfer, budget and ledger events are written
    and fsynced immediately. Routine events (heartbeat, progress) are written immediately
    and fsynced at least every ``flush_interval_s`` (10 s) by the heartbeat thread. A
    write failure raises ``TimeLogError``. Callers must stop at their next safe boundary
    and must not continue unlogged.

Spans
    ``run``, ``stage`` and ``operation`` spans each emit ``<kind>_start`` and exactly one
    ``<kind>_end`` whose ``outcome`` is completed, failed, cancelled or interrupted. A
    retry is a new attempt whose ``retry_of`` names the predecessor attempt. Heartbeats run
    at most every 60 s while a span is open and report progress separately from liveness.

The supervisor (``audit``) and the rollup live in ``timeaudit.py``.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import resource as rusage
import signal
import socket
import subprocess
import sys
import threading
import time
import uuid
from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCHEMA = "j1.time-event.v1"
PROJECT = "juniper-1"
OUTCOMES = ("completed", "failed", "cancelled", "interrupted")
# Frozen top-level time categories (§8 "Time categories"); spans declare one.
CATEGORIES = (
    "initialization",
    "compilation_warmup",
    "cpu_data_processing",
    "gpu_compute",
    "data_io_wait",
    "evaluation",
    "checkpoint",
    "replication_archival",
    "restoration",
    "transfer",
    "manual_review",
    "planned_idle",
    "unexpected_interruption",
    "supervision",
)
IMMEDIATE = {
    "checkpoint",
    "restore",
    "transfer",
    "verification",
    "budget",
    "resource_state",
    "billing_snapshot",
    "clock_anomaly",
    "clock_check",
    "failure",
    "retry",
    "interruption",
    "cancellation",
    "decision",
    "alert",
}
HEARTBEAT_MAX_S = 60.0


class TimeLogError(RuntimeError):
    """Logging failed; the caller must stop at its next consistent boundary."""


def utc_ms(ns: int | None = None) -> str:
    ns = time.time_ns() if ns is None else ns
    return datetime.fromtimestamp(ns / 1e9, UTC).strftime("%Y-%m-%dT%H:%M:%S.") + f"{(ns // 1_000_000) % 1000:03d}Z"


def boot_id() -> str:
    try:
        return Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    except OSError:
        return "unknown-boot"


def host_id() -> str:
    try:
        mid = Path("/etc/machine-id").read_text().strip()[:12]
    except OSError:
        mid = "no-machine-id"
    return f"{socket.gethostname()}-{mid}"


class TimeLog:
    """Writer for one producer. Thread-safe. ``wall_ns``/``mono_ns`` are injectable for tests."""

    def __init__(
        self,
        log_root: Path,
        run_id: str,
        producer_name: str = "proc",
        flush_interval_s: float = 10.0,
        wall_ns: Callable[[], int] = time.time_ns,
        mono_ns: Callable[[], int] = time.monotonic_ns,
        boot: str | None = None,
        host: str | None = None,
    ):
        self.run_id = run_id
        self.host_id = host or host_id()
        self.boot_id = boot or boot_id()
        self.clock_domain_id = f"{self.host_id}:{self.boot_id}:CLOCK_MONOTONIC"
        self.wall_ns, self.mono_ns = wall_ns, mono_ns
        self.producer_id = f"{producer_name}:{self.host_id}:{os.getpid()}:{uuid.uuid4().hex[:8]}"
        self.dir = Path(log_root) / run_id
        try:
            self.dir.mkdir(parents=True, exist_ok=True)
            self.path = self.dir / f"{self.producer_id.replace(':', '_')}.jsonl"
            self._f = self.path.open("a", encoding="utf-8")
        except OSError as e:
            raise TimeLogError(f"cannot open time log: {e}") from e
        self.flush_interval_s = flush_interval_s
        self.seq = 0
        self._lock = threading.Lock()
        self._last_sync = mono_ns()
        self._last_wall = wall_ns()
        self._last_mono = self._last_sync
        self.closed = False
        res = time.get_clock_info("monotonic")
        self.event(
            "producer_start",
            durable=True,
            pid=os.getpid(),
            argv=sys.argv[:8],
            monotonic_clock={
                "implementation": res.implementation,
                "resolution_s": res.resolution,
                "includes_suspend": False,
            },
            wall_clock_resolution_s=time.get_clock_info("time").resolution,
        )

    # -- core ---------------------------------------------------------------
    def event(self, event_type: str, durable: bool | None = None, **fields: Any) -> dict:
        with self._lock:
            if self.closed:
                raise TimeLogError("time log is closed")
            w, m = self.wall_ns(), self.mono_ns()
            # Wall-clock jump detection: realtime and monotonic must advance together.
            dw, dm = w - self._last_wall, m - self._last_mono
            jump = abs(dw - dm) > 2_000_000_000
            self._last_wall, self._last_mono = w, m
            rec = {
                "schema_version": SCHEMA,
                "event_id": uuid.uuid4().hex,
                "project_id": PROJECT,
                "run_id": self.run_id,
                "producer_id": self.producer_id,
                "seq": self.seq,
                "event_type": event_type,
                "timestamp_utc": utc_ms(w),
                "host_id": self.host_id,
                "clock_domain_id": self.clock_domain_id,
                "monotonic": m,
                "monotonic_unit": "ns",
                "status": fields.pop("status", "ok"),
            }
            nulls = {}
            for k in ("stage_id", "attempt_id", "span_id", "parent_span_id"):
                v = fields.pop(k, None)
                rec[k] = v
                if v is None:
                    nulls[k] = fields.pop(f"{k}_null_reason", "not within a span" if k != "parent_span_id" else "root")
            if "elapsed_from_mono" in fields:
                # A span's duration is measured from the SAME monotonic reading its terminal event carries, so the
                # audit's end.monotonic - start.monotonic equals elapsed_seconds exactly even when another thread
                # (a heartbeat fsync) held the lock between the caller's clock read and this record.
                rec["elapsed_seconds"] = round((m - fields.pop("elapsed_from_mono")) / 1e9, 6)
            rec.update(fields)
            if nulls:
                rec["null_reasons"] = nulls
            line = json.dumps(rec, sort_keys=True, ensure_ascii=False)
            try:
                self._f.write(line + "\n")
                self._f.flush()
                if (
                    durable
                    or (durable is None and (event_type.endswith(("_start", "_end")) or event_type in IMMEDIATE))
                    or m - self._last_sync >= self.flush_interval_s * 1e9
                ):
                    os.fsync(self._f.fileno())
                    self._last_sync = m
            except OSError as e:
                raise TimeLogError(f"time log write failed: {e}") from e
            self.seq += 1
        if jump:
            self.event(
                "clock_anomaly",
                kind="wall_clock_jump",
                wall_delta_ns=dw,
                monotonic_delta_ns=dm,
                note="prior timestamps are not rewritten; durations use the monotonic clock",
            )
        return rec

    def sync(self) -> None:
        with self._lock:
            if not self.closed:
                try:
                    os.fsync(self._f.fileno())
                except OSError as e:
                    raise TimeLogError(f"time log fsync failed: {e}") from e
                self._last_sync = self.mono_ns()

    def close(self) -> None:
        if self.closed:
            return
        self.event("producer_end", durable=True, events_written=self.seq + 1)
        with self._lock:
            self._f.close()
            self.closed = True
        digest = hashlib.sha256(self.path.read_bytes()).hexdigest()
        with (self.dir / "seals.jsonl").open("a") as f:
            f.write(json.dumps({"segment": self.path.name, "sha256": digest, "sealed_utc": utc_ms()}) + "\n")
            f.flush()
            os.fsync(f.fileno())

    # -- spans --------------------------------------------------------------
    def span(
        self,
        kind: str,
        name: str,
        *,
        stage_id: str,
        attempt_id: str,
        category: str,
        resource: str | None = None,
        parent: Span | None = None,
        exclusive: bool = True,
        **fields: Any,
    ) -> Span:
        if kind not in ("run", "stage", "operation"):
            raise ValueError(kind)
        if category not in CATEGORIES:
            raise ValueError(f"unknown category {category!r}")
        return Span(self, kind, name, stage_id, attempt_id, category, resource, parent, exclusive, fields)


class Span:
    def __init__(
        self,
        log: TimeLog,
        kind: str,
        name: str,
        stage_id: str,
        attempt_id: str,
        category: str,
        resource: str | None,
        parent: Span | None,
        exclusive: bool,
        fields: dict,
    ):
        self.log, self.kind, self.name = log, kind, name
        self.stage_id, self.attempt_id, self.category = stage_id, attempt_id, category
        self.resource, self.parent, self.exclusive = resource, parent, exclusive
        self.span_id = uuid.uuid4().hex
        self.counters: dict[str, float] = {}
        self.last_progress_utc: str | None = None
        self.ended = False
        self._hb: threading.Thread | None = None
        self._hb_stop = threading.Event()
        ids: dict[str, Any] = {
            "stage_id": stage_id,
            "attempt_id": attempt_id,
            "span_id": self.span_id,
            "parent_span_id": parent.span_id if parent else None,
        }
        self._ids = ids
        rec = log.event(
            f"{kind}_start", name=name, category=category, resource=resource, exclusive=exclusive, **ids, **fields
        )
        self.start_mono, self.start_domain = rec["monotonic"], rec["clock_domain_id"]

    def event(self, event_type: str, **fields: Any) -> dict:
        return self.log.event(event_type, **self._ids, **fields)

    def progress(self, **counters: float) -> None:
        self.counters.update(counters)
        self.last_progress_utc = utc_ms(self.log.wall_ns())

    def start_heartbeat(self, interval_s: float = 30.0, probe: Callable[[], dict] | None = None) -> None:
        interval_s = min(interval_s, HEARTBEAT_MAX_S, self.log.flush_interval_s)

        def beat() -> None:
            while not self._hb_stop.wait(interval_s):
                extra = {}
                if probe is not None:
                    try:
                        extra = probe()
                    except Exception as e:  # a probe failure is reported, not fatal
                        extra = {"probe_error": repr(e)[:200]}
                try:
                    self.event(
                        "heartbeat",
                        stage=self.stage_id,
                        counters=dict(self.counters),
                        last_progress_utc=self.last_progress_utc,
                        interval_s=interval_s,
                        **extra,
                    )
                    self.log.sync()
                except TimeLogError:
                    return  # the owner of the span sees the failure on its next write

        self._hb = threading.Thread(target=beat, daemon=True)
        self._hb.start()

    def end(
        self,
        outcome: str = "completed",
        *,
        exit_code: int | None = None,
        failure_reason: str | None = None,
        artifacts: list | None = None,
        evidence: list | None = None,
        **fields: Any,
    ) -> dict:
        if self.ended:
            raise TimeLogError(f"span {self.span_id} already has a terminal outcome")
        if outcome not in OUTCOMES:
            raise ValueError(outcome)
        self._hb_stop.set()
        if self._hb is not None:
            self._hb.join(timeout=5)
        m = self.log.mono_ns()
        if self.log.clock_domain_id != self.start_domain:
            raise TimeLogError("span ended in a different clock domain; record an interruption instead")
        elapsed = (m - self.start_mono) / 1e9
        if elapsed < 0:
            raise TimeLogError("negative monotonic duration")
        nulls = {}
        if exit_code is None:
            nulls["exit_code"] = "in-process span" if outcome == "completed" else "no process exit code"
        if failure_reason is None and outcome != "completed":
            failure_reason = "unspecified"
        self.ended = True
        return self.event(
            f"{self.kind}_end",
            name=self.name,
            category=self.category,
            resource=self.resource,
            outcome=outcome,
            elapsed_from_mono=self.start_mono,
            exit_code=exit_code,
            failure_reason=failure_reason,
            counters=dict(self.counters),
            artifacts=artifacts or [],
            evidence=evidence or [],
            terminal_null_reasons=nulls,
            **fields,
        )

    def __enter__(self) -> Span:
        return self

    def __exit__(self, et, ev, tb) -> None:
        if self.ended:
            return
        if et is None:
            self.end("completed")
        elif issubclass(et, KeyboardInterrupt):
            self.end("cancelled", failure_reason="KeyboardInterrupt")
        else:
            self.end("failed", failure_reason=f"{et.__name__}: {ev}"[:500])


# ---------------------------------------------------------------- clock checks


def clock_check_local() -> dict:
    """NTP status from systemd-timesyncd (offset and root distance), if available."""
    try:
        sync = subprocess.run(
            ["timedatectl", "show", "-p", "NTPSynchronized", "--value"], capture_output=True, text=True, timeout=10
        ).stdout.strip()
        status = subprocess.run(["timedatectl", "timesync-status"], capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.TimeoutExpired) as e:
        return {"method": "timedatectl", "synchronized": None, "error": repr(e)[:200]}

    def field(name: str) -> str | None:
        for line in status.splitlines():
            if line.strip().startswith(name + ":"):
                return line.split(":", 1)[1].strip()
        return None

    def ms(v: str | None) -> float | None:
        if not v:
            return None
        v = v.split()[0].replace("+", "")
        for unit, f in (("ms", 1.0), ("us", 1e-3), ("ns", 1e-6), ("s", 1e3)):
            if v.endswith(unit):
                try:
                    return float(v[: -len(unit)]) * f
                except ValueError:
                    return None
        return None

    off, root = ms(field("Offset")), ms(field("Root distance"))
    return {
        "method": "systemd-timesyncd",
        "synchronized": sync == "yes",
        "offset_ms": off,
        "root_distance_ms": root,
        "server": field("Server"),
        "uncertainty_ms": None if off is None or root is None else abs(off) + root,
    }


def clock_check_remote(ssh_host: str) -> dict:
    """Offset of a remote host against this host, bounded by half the SSH round trip."""
    t0 = time.time_ns()
    try:
        out = subprocess.run(
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", ssh_host, "date +%s%N"],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as e:
        return {"method": "ssh-date", "host": ssh_host, "synchronized": None, "error": repr(e)[:200]}
    t1 = time.time_ns()
    try:
        remote = int(out.stdout.strip())
    except ValueError:
        return {"method": "ssh-date", "host": ssh_host, "synchronized": None, "error": out.stderr[-200:]}
    offset_ms = (remote - (t0 + t1) / 2) / 1e6
    half_rtt_ms = (t1 - t0) / 2e6
    return {
        "method": "ssh-date vs local NTP clock",
        "host": ssh_host,
        "offset_ms": round(offset_ms, 3),
        "uncertainty_ms": round(half_rtt_ms, 3),
        "synchronized": abs(offset_ms) + half_rtt_ms <= 1000.0,
    }


def record_clock_check(log: TimeLog, ssh_host: str | None = None, max_offset_ms: float = 1000.0) -> dict:
    c = clock_check_remote(ssh_host) if ssh_host else clock_check_local()
    ok = c.get("synchronized") is True and (c.get("offset_ms") is None or abs(c["offset_ms"]) <= max_offset_ms)
    c["permitted_offset_ms"] = max_offset_ms
    c["utc_ordering"] = "trusted" if ok else "uncertain"
    log.event("clock_check", status="ok" if ok else "warning", **c)
    return c


# ---------------------------------------------------------------- command wrapper


def run_command(
    argv: list[str],
    log_root: Path,
    run_id: str,
    stage_id: str,
    attempt_id: str,
    category: str,
    resource: str | None,
    retry_of: str | None,
    progress_file: Path | None,
    heartbeat_s: float = 30.0,
) -> int:
    """Runs ``argv`` as a child inside an operation span. Forwards SIGTERM/SIGINT, records the
    exit, and stops the child if the time log itself fails."""
    log = TimeLog(log_root, run_id, producer_name="wrap")
    record_clock_check(log)
    if retry_of:
        log.event("retry", stage_id=stage_id, attempt_id=attempt_id, retry_of=retry_of)
    span = log.span(
        "operation",
        " ".join(argv)[:300],
        stage_id=stage_id,
        attempt_id=attempt_id,
        category=category,
        resource=resource,
        argv=argv,
        cwd=str(Path.cwd()),
        retry_of=retry_of,
        code_identity=_git_identity(Path.cwd()),
    )
    child = subprocess.Popen(argv)
    span.event("child_started", child_pid=child.pid)
    stop_reason: list[str] = []

    def forward(sig: int, _frame: Any) -> None:
        stop_reason.append(signal.Signals(sig).name)
        child.send_signal(sig)

    signal.signal(signal.SIGTERM, forward)
    signal.signal(signal.SIGINT, forward)

    def probe() -> dict:
        out: dict[str, Any] = {"child_alive": child.poll() is None}
        if progress_file is not None and progress_file.exists():
            st = progress_file.stat()
            out["progress_file_bytes"] = st.st_size
            out["progress_file_mtime_utc"] = utc_ms(int(st.st_mtime_ns))
        return out

    span.start_heartbeat(heartbeat_s, probe)
    log_failed = False
    while True:
        try:
            rc = child.wait(timeout=1.0)
            break
        except subprocess.TimeoutExpired:
            if span._hb is not None and not span._hb.is_alive() and not span.ended:
                # The heartbeat thread exits only on a log failure: stop the child safely.
                log_failed = True
                child.send_signal(signal.SIGTERM)
    usage = rusage.getrusage(rusage.RUSAGE_CHILDREN)
    outcome = "completed" if rc == 0 else ("cancelled" if stop_reason else "failed")
    try:
        span.end(
            outcome,
            exit_code=rc,
            failure_reason=None
            if rc == 0
            else (f"signal forwarded: {stop_reason}" if stop_reason else f"exit code {rc}"),
            child_cpu_user_s=usage.ru_utime,
            child_cpu_sys_s=usage.ru_stime,
            child_max_rss_kib=usage.ru_maxrss,
            log_failure_stop=log_failed,
        )
        log.close()
    except TimeLogError as e:
        print(f"[timelog] terminal event could not be written: {e}", file=sys.stderr)
        return 70
    return rc


def _git_identity(path: Path) -> dict:
    try:
        rev = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10)
        dirty = subprocess.run(
            ["git", "-C", str(path), "status", "--porcelain"], capture_output=True, text=True, timeout=30
        )
    except (OSError, subprocess.TimeoutExpired):
        return {"git": None}
    if rev.returncode != 0:
        return {"git": None}
    return {"git": rev.stdout.strip(), "dirty": bool(dirty.stdout.strip())}


# ---------------------------------------------------------------- resource ledger


def ledger_event(log_root: Path, run_id: str, kind: str, **fields: Any) -> dict:
    """Appends a paid-resource ledger record (state observation, request, rate, billing snapshot)."""
    if kind not in ("resource_state", "resource_request", "rate", "billing_snapshot", "budget"):
        raise ValueError(kind)
    log = TimeLog(log_root, run_id, producer_name="ledger")
    try:
        return log.event(kind, durable=True, **fields)
    finally:
        log.close()


@contextlib.contextmanager
def maybe_span(log: TimeLog | None, *args: Any, **kwargs: Any) -> Iterator[Span | None]:
    if log is None:
        yield None
        return
    with log.span(*args, **kwargs) as s:
        yield s


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m j1bench.timelog")
    ap.add_argument("--log-root", type=Path, required=True)
    ap.add_argument("--run-id", required=True)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="wrap a command in an operation span")
    r.add_argument("--stage", required=True)
    r.add_argument("--attempt", default=None)
    r.add_argument("--retry-of", default=None)
    r.add_argument("--category", required=True, choices=CATEGORIES)
    r.add_argument("--resource", default=None)
    r.add_argument("--progress-file", type=Path, default=None)
    r.add_argument("--heartbeat-s", type=float, default=30.0)
    r.add_argument("argv", nargs=argparse.REMAINDER)
    c = sub.add_parser("clock-check")
    c.add_argument("--ssh-host", default=None)
    lg = sub.add_parser("ledger", help="append a paid-resource ledger record")
    lg.add_argument("kind", choices=["resource_state", "resource_request", "rate", "billing_snapshot", "budget"])
    lg.add_argument("--json", required=True, help="record fields as a JSON object")
    ev = sub.add_parser("event", help="append a decision/manual-review/backfill record")
    ev.add_argument("event_type")
    ev.add_argument("--json", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "run":
        cmd = a.argv[1:] if a.argv[:1] == ["--"] else a.argv
        attempt = a.attempt or f"{a.stage}-{uuid.uuid4().hex[:8]}"
        return run_command(
            cmd,
            a.log_root,
            a.run_id,
            a.stage,
            attempt,
            a.category,
            a.resource,
            a.retry_of,
            a.progress_file,
            a.heartbeat_s,
        )
    if a.cmd == "clock-check":
        log = TimeLog(a.log_root, a.run_id, producer_name="clock")
        print(json.dumps(record_clock_check(log, a.ssh_host)))
        log.close()
        return 0
    if a.cmd == "ledger":
        print(json.dumps(ledger_event(a.log_root, a.run_id, a.kind, **json.loads(a.json))))
        return 0
    log = TimeLog(a.log_root, a.run_id, producer_name="manual")
    print(json.dumps(log.event(a.event_type, durable=True, **json.loads(a.json))))
    log.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
