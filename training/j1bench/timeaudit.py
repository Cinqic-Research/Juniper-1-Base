"""Supervisor, audit, rollup and billing reconciliation for j1.time-event.v1 logs (spec v0.6 §8).

``audit``    reports the following:
             - duplicate event IDs
             - missing or duplicate per-producer sequence numbers
             - orphaned spans and missing or repeated outcomes
             - invalid or negative durations and clock-domain misuse
             - wall-clock jumps and stale heartbeats (gap > 3x the declared interval)
             - missing required fields
             - sealed-segment hash mismatches
             - overlapping exclusive spans on one resource timeline
``repair``   appends an ``interrupted`` outcome, written by a supervisor producer, for each
             open span whose producer is dead or silent for > 3 heartbeat intervals. It
             records the last observed time and the detection time. It never invents a crash
             time, and it never edits earlier entries.
``rollup``   produces per-run totals:
             - wall time as the union of span intervals
             - per-resource exclusive category totals plus unclassified time
             - overlapping (non-exclusive) service durations and their overlap
             - open spans, and replayed vs retained work
             - estimated paid cost from the resource ledger, reconciled with provider snapshots
             Measured, estimated and unknown values are labelled.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from .timelog import SCHEMA, TimeLog, utc_ms

REQUIRED = (
    "schema_version",
    "event_id",
    "project_id",
    "run_id",
    "producer_id",
    "seq",
    "event_type",
    "timestamp_utc",
    "host_id",
    "clock_domain_id",
    "monotonic",
    "monotonic_unit",
    "status",
    "stage_id",
    "attempt_id",
    "span_id",
    "parent_span_id",
)
TERMINAL_REQUIRED = ("elapsed_seconds", "outcome", "exit_code", "failure_reason", "counters", "artifacts", "evidence")


def ts(s: str) -> float:
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def load(run_dir: Path) -> tuple[list[dict], list[str]]:
    events, problems = [], []
    for p in sorted(run_dir.glob("*.jsonl")):
        if p.name == "seals.jsonl":
            continue
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines()):
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                problems.append(f"{p.name}:{i + 1}: unparsable line (torn write or corruption)")
    return events, problems


def spans(events: list[dict]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for e in events:
        t = e["event_type"]
        if t.endswith("_start") and t[:-6] in ("run", "stage", "operation"):
            out.setdefault(e["span_id"], {"ends": []})["start"] = e
        elif t.endswith("_end") and t[:-4] in ("run", "stage", "operation"):
            out.setdefault(e["span_id"], {"ends": []})["ends"].append(e)
    return out


def audit(run_dir: Path) -> dict[str, Any]:
    events, problems = load(run_dir)
    seen: set[str] = set()
    by_producer: dict[str, list[int]] = defaultdict(list)
    for e in events:
        missing = [k for k in REQUIRED if k not in e]
        if missing or e.get("schema_version") != SCHEMA:
            problems.append(f"event {e.get('event_id')}: missing fields {missing} or wrong schema")
        if e.get("event_id") in seen:
            problems.append(f"duplicate event_id {e['event_id']}")
        seen.add(e.get("event_id", ""))
        by_producer[e.get("producer_id", "?")].append(e.get("seq", -1))
        if e["event_type"] == "clock_anomaly":
            problems.append(f"clock anomaly at {e['timestamp_utc']}: {e.get('kind')}")
    for prod, seqs in by_producer.items():
        s = sorted(seqs)
        if len(set(s)) != len(s):
            problems.append(f"{prod}: duplicate sequence numbers")
        expected = set(range(s[0], s[-1] + 1)) if s else set()
        gaps = sorted(expected - set(s))
        if s and s[0] != 0:
            gaps = list(range(0, s[0])) + gaps
        if gaps:
            problems.append(f"{prod}: missing sequence numbers {gaps[:10]}{'...' if len(gaps) > 10 else ''}")
    open_spans = []
    hb: dict[str, list[dict]] = defaultdict(list)
    for e in events:
        if e["event_type"] == "heartbeat" and e.get("span_id"):
            hb[e["span_id"]].append(e)
    for sid, sp in spans(events).items():
        st, ends = sp.get("start"), sp["ends"]
        if st is None:
            problems.append(f"span {sid}: terminal without start")
            continue
        if len(ends) > 1:
            problems.append(f"span {sid}: {len(ends)} terminal outcomes")
        if not ends:
            open_spans.append(sid)
            continue
        end = ends[0]
        if end.get("supervisor_appended"):
            continue
        miss = [k for k in TERMINAL_REQUIRED if k not in end]
        if miss:
            problems.append(f"span {sid}: terminal missing {miss}")
        if end["clock_domain_id"] != st["clock_domain_id"]:
            problems.append(
                f"span {sid}: clock-domain misuse (start {st['clock_domain_id']}, end {end['clock_domain_id']})"
            )
            continue
        el = end.get("elapsed_seconds")
        mono = (end["monotonic"] - st["monotonic"]) / 1e9
        if el is None or el < 0 or abs(el - mono) > 1e-3:
            problems.append(f"span {sid}: invalid elapsed {el} vs monotonic {mono:.6f}")
        beats = sorted([st, *hb[sid], end], key=lambda e: e["monotonic"])
        interval = max((b.get("interval_s") or 0 for b in hb[sid]), default=0)
        if interval:
            gap = max((b["monotonic"] - a["monotonic"]) / 1e9 for a, b in itertools.pairwise(beats))
            if gap > 3 * interval:
                problems.append(f"span {sid}: stale heartbeat gap {gap:.1f}s > 3x{interval}s")
    seals = run_dir / "seals.jsonl"
    if seals.exists():
        for line in seals.read_text().splitlines():
            s = json.loads(line)
            p = run_dir / s["segment"]
            if not p.exists():
                problems.append(f"sealed segment missing: {s['segment']}")
            elif hashlib.sha256(p.read_bytes()).hexdigest() != s["sha256"]:
                problems.append(f"sealed segment modified: {s['segment']}")
    problems += _exclusive_overlaps(events)
    return {
        "run_dir": str(run_dir),
        "events": len(events),
        "producers": len(by_producer),
        "open_spans": open_spans,
        "problems": problems,
        "ok": not problems,
    }


def _intervals(events: list[dict]) -> list[dict]:
    out = []
    for sid, sp in spans(events).items():
        st = sp.get("start")
        if not st or not sp["ends"] or sp["ends"][0].get("elapsed_seconds") is None:
            continue
        s = ts(st["timestamp_utc"])
        out.append(
            {
                "span_id": sid,
                "start": s,
                "end": s + sp["ends"][0]["elapsed_seconds"],
                "category": st.get("category"),
                "resource": st.get("resource"),
                "exclusive": st.get("exclusive"),
                "kind": st["event_type"][:-6],
                "name": st.get("name"),
                "outcome": sp["ends"][0]["outcome"],
                "parent": st.get("parent_span_id"),
                "replayed": sp["ends"][0].get("replayed", False),
            }
        )
    return out


def _leaf_exclusive(iv: list[dict]) -> list[dict]:
    parents = {i["parent"] for i in iv}
    return [i for i in iv if i["exclusive"] and i["resource"] and i["span_id"] not in parents]


def _exclusive_overlaps(events: list[dict], tol: float = 0.005) -> list[str]:
    probs = []
    by_res: dict[str, list[dict]] = defaultdict(list)
    for i in _leaf_exclusive(_intervals(events)):
        by_res[i["resource"]].append(i)
    for res, xs in by_res.items():
        xs.sort(key=lambda i: i["start"])
        for a, b in itertools.pairwise(xs):
            if b["start"] < a["end"] - tol:
                probs.append(f"resource {res}: exclusive spans overlap ({a['name']} / {b['name']})")
    return probs


def _union(iv: list[tuple[float, float]]) -> float:
    total, cur = 0.0, None
    for s, e in sorted(iv):
        if cur is None or s > cur[1]:
            if cur:
                total += cur[1] - cur[0]
            cur = [s, e]
        else:
            cur[1] = max(cur[1], e)
    return total + (cur[1] - cur[0] if cur else 0.0)


def repair(run_dir: Path, log_root: Path, heartbeat_s: float = 30.0, now: float | None = None) -> list[dict]:
    """Appends ``interrupted`` outcomes for open spans whose producer is gone."""
    events, _ = load(run_dir)
    now = now if now is not None else datetime.now().timestamp()
    last_by_prod: dict[str, dict] = {}
    for e in events:
        if e["producer_id"] not in last_by_prod or e["seq"] > last_by_prod[e["producer_id"]]["seq"]:
            last_by_prod[e["producer_id"]] = e
    out = []
    log = None
    for sid, sp in spans(events).items():
        st = sp.get("start")
        if not st or sp["ends"]:
            continue
        last = last_by_prod[st["producer_id"]]
        silent = now - ts(last["timestamp_utc"])
        pid = int(st["producer_id"].split(":")[-2])
        same_boot = st["clock_domain_id"].split(":")[1] == Path("/proc/sys/kernel/random/boot_id").read_text().strip()
        alive = same_boot and Path(f"/proc/{pid}").exists()
        if alive or silent <= 3 * heartbeat_s:
            continue
        log = log or TimeLog(log_root, st["run_id"], producer_name="supervisor")
        out.append(
            log.event(
                f"{st['event_type'][:-6]}_end",
                durable=True,
                stage_id=st["stage_id"],
                attempt_id=st["attempt_id"],
                span_id=sid,
                parent_span_id=st.get("parent_span_id"),
                name=st.get("name"),
                category=st.get("category"),
                resource=st.get("resource"),
                outcome="interrupted",
                supervisor_appended=True,
                last_observed_utc=last["timestamp_utc"],
                detection_utc=utc_ms(),
                elapsed_seconds=None,
                elapsed_lower_bound_s=round(ts(last["timestamp_utc"]) - ts(st["timestamp_utc"]), 3),
                exit_code=None,
                failure_reason="producer silent and not alive; crash time unknown",
                counters={},
                artifacts=[],
                evidence=[],
                terminal_null_reasons={"elapsed_seconds": "producer died; only a lower bound is known"},
            )
        )
    if log:
        log.close()
    return out


def ledger_costs(events: list[dict]) -> dict[str, Any]:
    """Estimated billed intervals per resource from observed states; bounds from observation spacing."""
    states: dict[str, list[dict]] = defaultdict(list)
    snaps = []
    for e in events:
        if e["event_type"] == "resource_state":
            states[e["resource"]].append(e)
        elif e["event_type"] == "billing_snapshot":
            snaps.append(e)
    res_out = {}
    for res, obs in states.items():
        obs.sort(key=lambda e: ts(e["timestamp_utc"]))
        low = high = 0.0
        running_since: tuple[float, float] | None = None  # (last non-running obs, first running obs)
        prev_t = None
        for e in obs:
            t = ts(e["timestamp_utc"])
            run = e["state"] == "RUNNING"
            if run and running_since is None:
                running_since = (prev_t if prev_t is not None else t, t)
            elif not run and running_since is not None and prev_t is not None:
                low += max(0.0, prev_t - running_since[1])  # last RUNNING obs - first RUNNING obs
                high += t - running_since[0]
                running_since = None
            prev_t = t
        if running_since is not None and prev_t is not None:
            low += prev_t - running_since[1]
            high += prev_t - running_since[0]
        rate = next((e["usd_per_hour"] for e in reversed(obs) if e.get("usd_per_hour") is not None), None)
        res_out[res] = {
            "observations": len(obs),
            "billed_hours_bounds": [round(low / 3600, 4), round(high / 3600, 4)],
            "usd_per_hour": rate,
            "estimated_usd_bounds": None
            if rate is None
            else [round(low / 3600 * rate, 3), round(high / 3600 * rate, 3)],
            "basis": "estimate from observed provider states; not an invoice",
        }
    recon = None
    if len(snaps) >= 2:
        snaps.sort(key=lambda e: ts(e["timestamp_utc"]))
        recon = {
            "provider_delta_usd": round(snaps[-1]["total_usd"] - snaps[0]["total_usd"], 4),
            "from_snapshot": snaps[0]["timestamp_utc"],
            "to_snapshot": snaps[-1]["timestamp_utc"],
            "note": "provider billing lags; the delta is pending until a later snapshot stops changing",
        }
    return {"resources": res_out, "provider_reconciliation": recon}


def rollup(run_dir: Path) -> dict[str, Any]:
    events, _ = load(run_dir)
    iv = _intervals(events)
    leaves = _leaf_exclusive(iv)
    per_res: dict[str, dict] = {}
    by_res: dict[str, list[dict]] = defaultdict(list)
    for i in leaves:
        by_res[i["resource"]].append(i)
    for res, xs in by_res.items():
        observed = max(i["end"] for i in xs) - min(i["start"] for i in xs)
        cats: dict[str, float] = defaultdict(float)
        for i in xs:
            cats[i["category"]] += i["end"] - i["start"]
        classified = sum(cats.values())
        per_res[res] = {
            "observed_interval_s": round(observed, 3),
            "categories_s": {k: round(v, 3) for k, v in sorted(cats.items())},
            "unclassified_s": round(max(0.0, observed - classified), 3),
        }
    overl = [i for i in iv if not i["exclusive"]]
    excl_union = [(i["start"], i["end"]) for i in leaves]
    overlap_rows = []
    for o in overl:
        ov = sum(max(0.0, min(o["end"], e) - max(o["start"], s)) for s, e in excl_union)
        overlap_rows.append(
            {
                "name": o["name"],
                "category": o["category"],
                "service_s": round(o["end"] - o["start"], 3),
                "overlap_with_exclusive_s": round(ov, 3),
            }
        )
    clock = [e for e in events if e["event_type"] == "clock_check"]
    return {
        "run_dir": str(run_dir),
        "wall_time_union_s": round(_union([(i["start"], i["end"]) for i in iv]), 3),
        "per_resource": per_res,
        "overlapping_services": overlap_rows,
        "open_or_interrupted": [
            {"span_id": s, "name": sp["start"].get("name")}
            for s, sp in spans(events).items()
            if sp.get("start") and (not sp["ends"] or sp["ends"][0]["outcome"] == "interrupted")
        ],
        "replayed_s": round(sum(i["end"] - i["start"] for i in iv if i["replayed"]), 3),
        "clock_checks": [
            {k: c.get(k) for k in ("timestamp_utc", "host_id", "method", "offset_ms", "uncertainty_ms", "utc_ordering")}
            for c in clock
        ],
        "ledger": ledger_costs(events),
        "labels": {
            "durations": "measured (monotonic, per clock domain)",
            "cross_host_placement": "UTC; bounded by clock-check uncertainty",
            "costs": "estimated unless provider_reconciliation is final",
        },
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m j1bench.timeaudit")
    ap.add_argument("cmd", choices=["audit", "repair", "rollup"])
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--heartbeat-s", type=float, default=30.0)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args(argv)
    if a.cmd == "audit":
        res: Any = audit(a.run_dir)
    elif a.cmd == "repair":
        res = repair(a.run_dir, a.run_dir.parent, a.heartbeat_s)
    else:
        res = rollup(a.run_dir)
    text = json.dumps(res, indent=1)
    if a.out:
        a.out.write_text(text + "\n")
    print(text)
    return 0 if a.cmd != "audit" or res["ok"] else 2


if __name__ == "__main__":
    sys.exit(main())
