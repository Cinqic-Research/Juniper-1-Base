"""Spec v0.6 §8 acceptance cases for the time logger, supervisor, audit and rollup."""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from j1bench import timeaudit, timelog
from j1bench.timelog import TimeLog, TimeLogError

RUN = "r1"


class Clock:
    """Controllable wall and monotonic clocks (ns)."""

    def __init__(self):
        self.wall = 1_791_500_000_000_000_000
        self.mono = 5_000_000_000

    def tick(self, s, wall_jump_s=0.0):
        self.mono += int(s * 1e9)
        self.wall += int((s + wall_jump_s) * 1e9)


def _log(tmp_path, clock=None, **kw):
    c = clock or Clock()
    return TimeLog(tmp_path, RUN, wall_ns=lambda: c.wall, mono_ns=lambda: c.mono, **kw), c


def _events(tmp_path):
    ev, probs = timeaudit.load(tmp_path / RUN)
    assert not probs
    return ev


def test_successful_and_failed_spans_have_one_terminal_and_monotonic_elapsed(tmp_path):
    log, c = _log(tmp_path)
    with log.span("stage", "s", stage_id="s", attempt_id="a1", category="cpu_data_processing", resource="cpu") as sp:
        c.tick(2.5)
        sp.progress(units=3)
    with (
        pytest.raises(ValueError),
        log.span("operation", "x", stage_id="s", attempt_id="a1", category="evaluation", resource="cpu", parent=sp),
    ):
        c.tick(1.0)
        raise ValueError("boom")
    log.close()
    ends = [e for e in _events(tmp_path) if e["event_type"] in ("stage_end", "operation_end")]
    assert sorted(e["outcome"] for e in ends) == ["completed", "failed"]
    ends.sort(key=lambda e: e["event_type"] != "stage_end")
    assert ends[0]["elapsed_seconds"] == 2.5 and ends[0]["counters"] == {"units": 3}
    assert "ValueError: boom" in ends[1]["failure_reason"]
    assert ends[0]["timestamp_utc"].endswith("Z") and len(ends[0]["timestamp_utc"].split(".")[1]) == 4
    rep = timeaudit.audit(tmp_path / RUN)
    assert rep["ok"], rep["problems"]


def test_double_terminal_refused(tmp_path):
    log, _ = _log(tmp_path)
    sp = log.span("operation", "o", stage_id="s", attempt_id="a", category="evaluation")
    sp.end()
    with pytest.raises(TimeLogError):
        sp.end()


def test_null_fields_carry_reasons(tmp_path):
    log, _ = _log(tmp_path)
    e = log.event("decision", choice="x")
    assert e["stage_id"] is None and "stage_id" in e["null_reasons"]


def test_wall_clock_jump_recorded_and_durations_unaffected(tmp_path):
    log, c = _log(tmp_path)
    sp = log.span("operation", "o", stage_id="s", attempt_id="a", category="gpu_compute", resource="gpu")
    c.tick(1.0, wall_jump_s=-3600)  # NTP step backwards by an hour
    end = sp.end()
    assert end["elapsed_seconds"] == 1.0
    ev = _events(tmp_path)
    assert any(e["event_type"] == "clock_anomaly" for e in ev)
    assert any("clock anomaly" in p for p in timeaudit.audit(tmp_path / RUN)["problems"])


def test_clock_domain_change_refuses_cross_domain_duration(tmp_path):
    log, _ = _log(tmp_path)
    sp = log.span("operation", "o", stage_id="s", attempt_id="a", category="gpu_compute")
    log.clock_domain_id = "other-host:other-boot:CLOCK_MONOTONIC"  # reboot / host change
    with pytest.raises(TimeLogError, match="clock domain"):
        sp.end()


def test_disk_full_raises_instead_of_silently_dropping(tmp_path, monkeypatch):
    log, _ = _log(tmp_path)

    def nospace(_fd):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(timelog.os, "fsync", nospace)
    with pytest.raises(TimeLogError, match="write failed"):
        log.event("checkpoint", step=1)


def test_missing_duplicate_and_tampered_events_detected(tmp_path):
    log, c = _log(tmp_path)
    for i in range(5):
        c.tick(1)
        log.event("decision", i=i)
    log.close()
    seg = log.path
    lines = seg.read_text().splitlines()
    seg.write_text("\n".join(lines[:2] + lines[3:] + [lines[4]]) + "\n")  # drop seq 2, duplicate seq 4
    probs = timeaudit.audit(tmp_path / RUN)["problems"]
    assert any("missing sequence" in p for p in probs)
    assert any("duplicate event_id" in p for p in probs)
    assert any("sealed segment modified" in p for p in probs)


def test_overlap_accounting_and_unclassified_time(tmp_path):
    log, c = _log(tmp_path)
    run = log.span("run", "r", stage_id="s", attempt_id="a", category="supervision", resource="gpu")
    tr = log.span(
        "operation", "train", stage_id="s", attempt_id="a", category="gpu_compute", resource="gpu", parent=run
    )
    c.tick(10)
    xfer = log.span(
        "operation",
        "upload",
        stage_id="s",
        attempt_id="a",
        category="transfer",
        resource="net",
        exclusive=False,
        parent=run,
    )
    c.tick(5)
    tr.end()
    c.tick(3)  # gap with no exclusive span on gpu
    ck = log.span("operation", "ckpt", stage_id="s", attempt_id="a", category="checkpoint", resource="gpu", parent=run)
    c.tick(2)
    ck.end()
    xfer.end()
    run.end()
    log.close()
    r = timeaudit.rollup(tmp_path / RUN)
    gpu = r["per_resource"]["gpu"]
    assert gpu["categories_s"] == {"checkpoint": 2.0, "gpu_compute": 15.0}
    assert gpu["unclassified_s"] == pytest.approx(3.0, abs=0.01)
    up = r["overlapping_services"][0]
    assert up["service_s"] == 10.0 and up["overlap_with_exclusive_s"] == pytest.approx(7.0, abs=0.01)
    assert r["wall_time_union_s"] == pytest.approx(20.0, abs=0.01)


def test_overlapping_exclusive_spans_flagged(tmp_path):
    log, c = _log(tmp_path)
    a = log.span("operation", "a", stage_id="s", attempt_id="a", category="gpu_compute", resource="gpu")
    c.tick(1)
    b = log.span("operation", "b", stage_id="s", attempt_id="a", category="evaluation", resource="gpu")
    c.tick(1)
    b.end()
    a.end()
    assert any("overlap" in p for p in timeaudit.audit(tmp_path / RUN)["problems"])


def test_ledger_bounds_and_stop_delay(tmp_path):
    t0 = 1_791_500_000.0
    clock = Clock()
    clock.wall = int(t0 * 1e9)
    log, c = _log(tmp_path, clock)
    log.event("resource_state", resource="pod:x", state="EXITED", usd_per_hour=2.0)
    c.tick(60)
    log.event("resource_state", resource="pod:x", state="RUNNING", usd_per_hour=2.0)
    c.tick(3540)
    log.event("resource_request", resource="pod:x", action="stop")
    log.event("resource_state", resource="pod:x", state="RUNNING", usd_per_hour=2.0)  # stop not yet effective
    c.tick(60)
    log.event("resource_state", resource="pod:x", state="EXITED", usd_per_hour=2.0)
    log.event("billing_snapshot", total_usd=3.55)
    c.tick(600)
    log.event("billing_snapshot", total_usd=5.60)
    log.close()
    led = timeaudit.ledger_costs(_events(tmp_path))
    res = led["resources"]["pod:x"]
    assert res["billed_hours_bounds"] == [round(3540 / 3600, 4), round(3660 / 3600, 4)]
    assert res["estimated_usd_bounds"][1] == pytest.approx(2.033, abs=0.001)
    assert led["provider_reconciliation"]["provider_delta_usd"] == pytest.approx(2.05)


def _wrap(tmp_path, *argv, extra=(), resource="cpu"):
    return [
        sys.executable,
        "-m",
        "j1bench.timelog",
        "--log-root",
        str(tmp_path),
        "--run-id",
        RUN,
        "run",
        "--stage",
        "s",
        "--category",
        "cpu_data_processing",
        "--resource",
        resource,
        "--heartbeat-s",
        "0.3",
        *extra,
        "--",
        *argv,
    ]


ROOT = Path(__file__).resolve().parents[1]


def test_wrapper_success_failure_retry_and_concurrency(tmp_path):
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    ok = subprocess.run(_wrap(tmp_path, sys.executable, "-c", "import time; time.sleep(1)"), env=env, cwd=ROOT)
    bad = subprocess.run(
        _wrap(tmp_path, sys.executable, "-c", "raise SystemExit(7)", extra=["--attempt", "a1"]), env=env, cwd=ROOT
    )
    retry = subprocess.run(_wrap(tmp_path, "true", extra=["--attempt", "a2", "--retry-of", "a1"]), env=env, cwd=ROOT)
    # Concurrent workers each own a resource timeline; sharing one exclusive timeline is an audit error.
    procs = [
        subprocess.Popen(_wrap(tmp_path, "sleep", "0.5", resource=f"cpu:w{i}"), env=env, cwd=ROOT) for i in range(4)
    ]
    assert [p.wait() for p in procs] == [0] * 4
    assert (ok.returncode, bad.returncode, retry.returncode) == (0, 7, 0)
    ev = _events(tmp_path)
    ends = [e for e in ev if e["event_type"] == "operation_end"]
    assert sorted(e["outcome"] for e in ends) == ["completed"] * 6 + ["failed"]
    assert any(e["event_type"] == "retry" and e["retry_of"] == "a1" for e in ev)
    assert sum(e["event_type"] == "heartbeat" for e in ev) >= 2
    assert sum(e["event_type"] == "clock_check" for e in ev) == 7
    rep = timeaudit.audit(tmp_path / RUN)
    assert rep["ok"], rep["problems"]


def test_abrupt_kill_supervisor_appends_honest_interruption(tmp_path):
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    p = subprocess.Popen(_wrap(tmp_path, "sleep", "30"), env=env, cwd=ROOT, start_new_session=True)
    time.sleep(2.0)
    os.killpg(p.pid, signal.SIGKILL)  # wrapper and child die without terminal events
    p.wait()
    rep = timeaudit.audit(tmp_path / RUN)
    assert rep["open_spans"]
    appended = timeaudit.repair(tmp_path / RUN, tmp_path, heartbeat_s=0.3, now=time.time() + 5)
    assert len(appended) == 1 and appended[0]["outcome"] == "interrupted"
    assert appended[0]["elapsed_seconds"] is None and appended[0]["elapsed_lower_bound_s"] > 1.0
    assert timeaudit.repair(tmp_path / RUN, tmp_path, heartbeat_s=0.3, now=time.time() + 5) == []
    after = timeaudit.audit(tmp_path / RUN)
    assert not after["open_spans"]


def test_sigterm_forwarded_and_recorded_as_cancelled(tmp_path):
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    p = subprocess.Popen(_wrap(tmp_path, "sleep", "30"), env=env, cwd=ROOT)
    time.sleep(1.5)
    p.send_signal(signal.SIGTERM)
    assert p.wait(timeout=20) != 0
    end = next(e for e in _events(tmp_path) if e["event_type"] == "operation_end")
    assert end["outcome"] == "cancelled" and "SIGTERM" in end["failure_reason"]


def test_buffered_tail_loss_is_bounded_by_flush_interval(tmp_path):
    """Routine events are written per line; only the OS page cache can lose them, and the
    heartbeat thread fsyncs at least every flush interval. Here we verify that the bytes are
    in the file before any explicit sync (process death loses nothing written)."""
    log, c = _log(tmp_path, flush_interval_s=10.0)
    for i in range(20):
        c.tick(0.1)
        log.event("progress", i=i, durable=False)
    lines = (tmp_path / RUN / log.path.name).read_text().splitlines()
    assert len(lines) == 21  # producer_start + 20 progress events, readable without close()


def test_clock_check_unavailable_marks_ordering_uncertain(tmp_path, monkeypatch):
    log, _ = _log(tmp_path)
    monkeypatch.setattr(timelog, "clock_check_local", lambda: {"method": "timedatectl", "synchronized": None})
    c = timelog.record_clock_check(log)
    assert c["utc_ordering"] == "uncertain"


def test_cli_ledger_and_event(tmp_path):
    rc = timelog.main(
        [
            "--log-root",
            str(tmp_path),
            "--run-id",
            RUN,
            "ledger",
            "billing_snapshot",
            "--json",
            json.dumps({"total_usd": 3.55, "source": "mcp:list-billing"}),
        ]
    )
    assert rc == 0
    ev = _events(tmp_path)
    assert any(e["event_type"] == "billing_snapshot" and e["total_usd"] == 3.55 for e in ev)


def test_timesyncd_parsing(monkeypatch):
    out = (
        "       Server: 2620:2d:4000:1::41 (ntp.ubuntu.com)\nRoot distance: 7.964ms (max: 5s)\n"
        "       Offset: +1.047ms\n"
    )

    class R:
        def __init__(self, stdout):
            self.stdout = stdout

    calls = iter([R("yes\n"), R(out)])
    monkeypatch.setattr(timelog.subprocess, "run", lambda *_a, **_k: next(calls))
    c = timelog.clock_check_local()
    assert c["synchronized"] and c["offset_ms"] == 1.047 and c["root_distance_ms"] == 7.964
    assert c["uncertainty_ms"] == pytest.approx(9.011)
