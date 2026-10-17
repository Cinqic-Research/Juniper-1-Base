"""Time-logger qualification on a host (spec v0.6 §8 "Implementation and acceptance gates").

Frozen before running (tolerances):

* Known-duration control at d = 0.25, 1 and 5 s. ``R`` is the operation's own interval
  (``time.perf_counter_ns`` right after ``span()`` returns, until right before ``end()`` is
  called). ``W`` is the measured cost of opening the span (writing and fsyncing its start
  event). The span's ``elapsed_seconds`` E must satisfy **R <= E <= R + W + 5 ms**: the span
  encloses the operation, and its excess is fully explained by persisting the start event.
  The excess E - R is reported as logger overhead per span.
* Wrapper control: a wrapped ``sleep 2`` child must report a span within **[2.0, 2.0 + 0.5] s**.
  The upper bound covers process start-up.
* Overhead: the median and p99 cost of a durable (fsynced) event and of a routine event on the log
  device are reported. No threshold applies; the numbers feed the throughput and cost accounting.
* Audit of everything produced must be clean.

Writes a JSON report and exits nonzero if a frozen tolerance fails.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from j1bench import timeaudit
from j1bench.timelog import TimeLog, clock_check_local


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-dir", type=Path, default=None, help="directory on the device to qualify")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    root = a.log_dir or Path(tempfile.mkdtemp(prefix="j1tlq-"))
    run = f"timelog-qual-{int(time.time())}"
    log = TimeLog(root, run, producer_name="qual")
    controls, ok = [], True
    for d in (0.25, 1.0, 5.0):
        t0 = time.perf_counter_ns()
        sp = log.span(
            "operation", f"sleep-{d}", stage_id="qual", attempt_id="a", category="planned_idle", resource="cpu"
        )
        t1 = time.perf_counter_ns()
        time.sleep(d)
        t2 = time.perf_counter_ns()
        end = sp.end()
        w, r = (t1 - t0) / 1e9, (t2 - t1) / 1e9
        e = end["elapsed_seconds"]
        passed = r - 1e-6 <= e <= r + w + 0.005
        ok &= passed
        controls.append(
            {
                "target_s": d,
                "operation_s": round(r, 6),
                "span_elapsed_s": e,
                "start_write_s": round(w, 6),
                "logger_excess_s": round(e - r, 6),
                "pass": passed,
            }
        )
    lat_durable, lat_routine = [], []
    for i in range(200):
        t = time.perf_counter_ns()
        log.event("verification", durable=True, i=i)
        lat_durable.append((time.perf_counter_ns() - t) / 1e6)
        t = time.perf_counter_ns()
        log.event("progress", durable=False, i=i)
        lat_routine.append((time.perf_counter_ns() - t) / 1e6)
    log.close()
    env = {"PYTHONPATH": str(Path(__file__).resolve().parents[1]), "PATH": "/usr/bin:/bin"}
    t0 = time.perf_counter()
    rc = subprocess.run(
        [
            sys.executable,
            "-m",
            "j1bench.timelog",
            "--log-root",
            str(root),
            "--run-id",
            run,
            "run",
            "--stage",
            "qual",
            "--category",
            "planned_idle",
            "--resource",
            "cpu:wrap",
            "--",
            "sleep",
            "2",
        ],
        env=env,
    ).returncode
    wall = time.perf_counter() - t0
    wend = next(
        e
        for e in timeaudit.load(root / run)[0]
        if e["event_type"] == "operation_end" and e.get("resource") == "cpu:wrap"
    )
    wrap_pass = rc == 0 and 2.0 <= wend["elapsed_seconds"] <= 2.5
    ok &= wrap_pass
    aud = timeaudit.audit(root / run)
    ok &= aud["ok"]

    def q(xs: list[float], p: float) -> float:
        return round(sorted(xs)[int(p * (len(xs) - 1))], 3)

    rep = {
        "host": platform.node(),
        "log_device_path": str(root),
        "python": platform.python_version(),
        "clock": clock_check_local(),
        "controls": controls,
        "wrapper": {"elapsed_s": wend["elapsed_seconds"], "outer_wall_s": round(wall, 3), "pass": wrap_pass},
        "event_cost_ms": {
            "durable_median": q(lat_durable, 0.5),
            "durable_p99": q(lat_durable, 0.99),
            "routine_median": q(lat_routine, 0.5),
            "routine_p99": q(lat_routine, 0.99),
            "durable_mean": round(statistics.fmean(lat_durable), 3),
        },
        "audit": {"ok": aud["ok"], "problems": aud["problems"], "events": aud["events"]},
        "pass": bool(ok),
    }
    a.out.write_text(json.dumps(rep, indent=1) + "\n")
    print(json.dumps(rep, indent=1))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
