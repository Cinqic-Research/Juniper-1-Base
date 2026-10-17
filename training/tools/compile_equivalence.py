"""Eager versus compiled learning equivalence (schema ``j1.compile-equivalence.v2``).

Compares two trainer runs that differ only in ``--compile`` under a protocol registered BEFORE the runs
(``--protocol``): the comparison range, the evaluation milestones, the registered tolerances (0.5 % mean relative
loss difference, 0.3 % final validation bpb), the stability criterion, the per-domain criterion and the compile
options. The tolerances are read from the protocol file only; the record carries the protocol's SHA-256.

Validation (INVALID, exit 2) before any comparison: both summaries are trainer summaries with ``compile`` false /
true; every other config key is identical (except run ids and output paths); ``data_identity``, ``code``
(package SHA-256), ``model``, ``n_params`` and the evaluation-set identity are identical; both runs completed
(``final_step == steps_run == stop_at``, sibling ``.status`` exit code 0, no ``restored`` event); the metrics
JSONL has every update from 1 to the final step exactly once with finite, positive losses; identical ``batch_fp``
at every update (same data order); evaluation records at every registered milestone with finite, positive bpb (also
per domain), the last one at the final step (a zero or negative loss/bpb is malformed, not a denominator: review
R-08). PASS (exit 0) iff every registered criterion holds, else FAIL (exit 1).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

SCHEMA = "j1.compile-equivalence.v2"
# Config keys allowed to differ between the pair (the compared variable plus per-run outputs).
EXCLUDED_CONFIG_KEYS = frozenset({"compile", "compile_mode", "run_id", "metrics", "summary"})


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _positive(v) -> bool:
    """A finite number strictly above zero (losses and bpb are relative-difference denominators)."""
    return isinstance(v, int | float) and not isinstance(v, bool) and math.isfinite(v) and v > 0


def read_metrics(p: Path) -> tuple[dict[int, dict], list[dict], list[str]]:
    """Update records by step, eval_detail records, and problems (duplicates, non-finite, resumes)."""
    updates: dict[int, dict] = {}
    evals: list[dict] = []
    problems: list[str] = []
    for n, line in enumerate(p.read_text().splitlines(), 1):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            problems.append(f"{p.name}:{n}: unparseable record")
            continue
        ev = r.get("event")
        if ev == "restored" or ev == "resume_scan":
            problems.append(f"{p.name}:{n}: run was resumed ({ev})")
        if ev == "eval_detail":
            evals.append(r)
        elif ev is None and isinstance(r.get("step"), int) and "loss" in r:
            s = r["step"]
            if s in updates:
                problems.append(f"{p.name}:{n}: duplicate update record for step {s}")
            if not _positive(r["loss"]):
                problems.append(f"{p.name}:{n}: non-finite or non-positive loss {r['loss']!r} at step {s}")
            updates[s] = r
    return updates, evals, problems


def validate_pair(e: dict, c: dict, eager: Path, compiled: Path, proto: dict) -> tuple[list[str], dict]:
    problems: list[str] = []
    if e.get("config", {}).get("compile") is not False or c.get("config", {}).get("compile") is not True:
        problems.append("runs must be eager (compile false) and compiled (compile true)")
    ce, cc = e.get("config", {}), c.get("config", {})
    keys = (set(ce) | set(cc)) - EXCLUDED_CONFIG_KEYS
    diff = sorted(k for k in keys if ce.get(k) != cc.get(k))
    if diff:
        problems.append(f"runs differ in config {diff}")
    for k in ("data_identity", "model", "n_params"):
        if e.get(k) != c.get(k):
            problems.append(f"runs differ in {k}")
    if e.get("code", {}).get("package_sha256") != c.get("code", {}).get("package_sha256"):
        problems.append("runs differ in code identity")
    if e.get("eval_set") != c.get("eval_set") or not e.get("eval_set"):
        problems.append("runs differ in evaluation-set identity (or it is missing)")
    stop_at = proto["stop_at"]
    for name, s, p in (("eager", e, eager), ("compiled", c, compiled)):
        if s.get("final_step") != stop_at or s.get("steps_run") != stop_at:
            problems.append(f"{name}: final_step/steps_run {s.get('final_step')}/{s.get('steps_run')} != {stop_at}")
        if s.get("config", {}).get("stop_at") != stop_at:
            problems.append(f"{name}: config.stop_at {s.get('config', {}).get('stop_at')} != protocol {stop_at}")
        st = p.with_suffix(".status")
        try:
            rc = json.loads(st.read_text()).get("exit_code")
        except (OSError, json.JSONDecodeError):
            rc = None
        if rc != 0:
            problems.append(f"{name}: status file {st.name} missing or exit_code {rc!r} != 0")
        led = s.get("last_eval_detail") or {}
        if led.get("step") != stop_at or not _positive(led.get("bpb")):
            problems.append(f"{name}: last evaluation is not a finite positive bpb at step {stop_at}")
        bad_domains = sorted(k for k, v in (led.get("by_source") or {}).items() if not _positive((v or {}).get("bpb")))
        if bad_domains:
            problems.append(f"{name}: non-finite or non-positive domain bpb for {bad_domains}")
    compile_mode = c.get("config", {}).get("compile_mode")
    if compile_mode != proto["compile_options"]["mode"]:
        problems.append(f"compiled run mode {compile_mode!r} != registered {proto['compile_options']['mode']!r}")
    for k, want in proto.get("required_config", {}).items():
        if ce.get(k) != want:
            problems.append(f"config.{k} = {ce.get(k)!r} != registered {want!r}")
    ident = {
        "preset": ce.get("preset"),
        "shards_json_sha256": (e.get("data_identity") or {}).get("shards_json_sha256"),
        "tokenizer_json_sha256": (e.get("data_identity") or {}).get("tokenizer_json_sha256"),
        "build": (e.get("data_identity") or {}).get("build"),
        "limit_tokens": (e.get("data_identity") or {}).get("limit_tokens"),
        "seed": ce.get("seed"),
        "steps": ce.get("steps"),
        "stop_at": ce.get("stop_at"),
        "eval_documents": (e.get("eval_set") or {}).get("documents"),
        "eval_scored_bytes": (e.get("eval_set") or {}).get("scored_bytes"),
        "code_package_sha256": {
            "eager": e.get("code", {}).get("package_sha256"),
            "compiled": c.get("code", {}).get("package_sha256"),
        },
        "n_params": e.get("n_params"),
        "compile_mode": compile_mode,
        "run_ids": {"eager": ce.get("run_id"), "compiled": cc.get("run_id")},
    }
    return problems, ident


def compare(e: dict, c: dict, ue: dict, uc: dict, ee: list, ec: list, proto: dict) -> dict:
    lo, hi = proto["compare_range"]
    tol = proto["tolerances"]
    rel = {s: abs(ue[s]["loss"] - uc[s]["loss"]) / abs(ue[s]["loss"]) for s in range(lo, hi + 1)}
    mean_rel = sum(rel.values()) / len(rel)
    max_rel = max(rel.values())
    be, bc = e["last_eval_detail"]["bpb"], c["last_eval_detail"]["bpb"]
    bpb_rel = abs(be - bc) / be
    # Milestones: finite bpb at every registered step for both runs.
    me = {r["step"]: r for r in ee}
    mc = {r["step"]: r for r in ec}
    milestones = {}
    for s in proto["milestones"]:
        a, b = me.get(s, {}).get("bpb"), mc.get(s, {}).get("bpb")
        milestones[str(s)] = {"eager": a, "compiled": b, "rel_diff": (abs(a - b) / a) if a and b else None}
    # Domains: final per-source bpb; critical domains (by scored bytes) must be within the registered tolerance.
    de, dc = e["last_eval_detail"].get("by_source", {}), c["last_eval_detail"].get("by_source", {})
    domains = {}
    for name in sorted(set(de) | set(dc)):
        a, b = de.get(name, {}), dc.get(name, {})
        if "bpb" in a and "bpb" in b:
            domains[name] = {
                "bytes": a.get("bytes"),
                "critical": bool(a.get("bytes", 0) >= proto["domains"]["critical_min_bytes"]),
                "eager": a["bpb"],
                "compiled": b["bpb"],
                "rel_diff": abs(a["bpb"] - b["bpb"]) / a["bpb"],
            }
    crit_ok = all(d["rel_diff"] <= proto["domains"]["bpb_rel"] for d in domains.values() if d["critical"])
    criteria = {
        "mean_rel_loss_within": mean_rel <= tol["loss_rel"],
        "final_bpb_within": bpb_rel <= tol["bpb_rel"],
        "stability_max_rel_within": max_rel <= proto["stability"]["max_rel_loss_diff"],
        "critical_domains_within": crit_ok and any(d["critical"] for d in domains.values()),
        "milestones_finite": all(
            isinstance(m["eager"], int | float) and isinstance(m["compiled"], int | float) for m in milestones.values()
        ),
    }
    return {
        "criteria": criteria,
        "mean_rel_loss_diff": mean_rel,
        "max_rel_loss_diff": max_rel,
        "range": [lo, hi],
        "final_bpb": {"eager": be, "compiled": bc, "rel_diff": bpb_rel},
        "milestones": milestones,
        "domains": domains,
        "tolerances": {"loss_rel": tol["loss_rel"], "bpb_rel": tol["bpb_rel"]},
        "stability": {"max_rel_loss_diff": proto["stability"]["max_rel_loss_diff"]},
        "throughput": {"eager": e.get("tokens_per_s_median"), "compiled": c.get("tokens_per_s_median")},
        "first_step_s": {"eager": e.get("first_step_s"), "compiled": c.get("first_step_s")},
        "startup": {k: (s.get("startup") or {}).get("process_elapsed_s") for k, s in (("eager", e), ("compiled", c))},
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("eager", type=Path, help="summary JSON of the eager run (metrics .jsonl and .status beside it)")
    ap.add_argument("compiled", type=Path)
    ap.add_argument("--protocol", type=Path, required=True, help="registered protocol JSON (tolerances, range)")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    proto = json.loads(a.protocol.read_text())
    if proto.get("tolerances") != {"loss_rel": 0.005, "bpb_rel": 0.003}:
        rec = {"schema": SCHEMA, "status": "INVALID", "problems": ["protocol tolerances are not the registered ones"]}
        a.out.write_text(json.dumps(rec, indent=1))
        print(json.dumps(rec))
        return 2
    e, c = json.loads(a.eager.read_text()), json.loads(a.compiled.read_text())
    problems, ident = validate_pair(e, c, a.eager, a.compiled, proto)
    ue, ee, pe = read_metrics(a.eager.with_suffix(".jsonl"))
    uc, ec, pc = read_metrics(a.compiled.with_suffix(".jsonl"))
    problems += pe + pc
    stop_at = proto["stop_at"]
    for name, u in (("eager", ue), ("compiled", uc)):
        missing = [s for s in range(1, stop_at + 1) if s not in u]
        if missing:
            problems.append(f"{name}: missing update records {missing[:5]}{'...' if len(missing) > 5 else ''}")
    if not problems:
        fp_diff = [s for s in range(1, stop_at + 1) if ue[s].get("batch_fp") != uc[s].get("batch_fp")]
        if fp_diff:
            problems.append(
                f"batch fingerprints differ at {len(fp_diff)} updates (first {fp_diff[0]}): data order differs"
            )
    lo, hi = proto["compare_range"]
    if not (1 <= lo <= hi <= stop_at):
        problems.append(f"compare_range {proto['compare_range']} outside 1..{stop_at}")
    for name, evs in (("eager", ee), ("compiled", ec)):
        have = {r.get("step") for r in evs if _positive(r.get("bpb"))}
        missing = [s for s in proto["milestones"] if s not in have]
        if missing:
            problems.append(
                f"{name}: no finite positive evaluation at milestones {missing} (coverage differs from the protocol)"
            )
    base = {
        "schema": SCHEMA,
        "protocol": {"path": str(a.protocol), "sha256": sha256_file(a.protocol)},
        "identity": ident,
        "inputs_sha256": {
            "eager_summary": sha256_file(a.eager),
            "compiled_summary": sha256_file(a.compiled),
            "eager_metrics": sha256_file(a.eager.with_suffix(".jsonl")),
            "compiled_metrics": sha256_file(a.compiled.with_suffix(".jsonl")),
        },
    }
    if problems:
        rec = {**base, "status": "INVALID", "problems": problems}
        a.out.write_text(json.dumps(rec, indent=1))
        print(json.dumps({"status": "INVALID", "problems": problems}))
        return 2
    cmp = compare(e, c, ue, uc, ee, ec, proto)
    ok = all(cmp["criteria"].values())
    rec = {**base, "status": "PASS" if ok else "FAIL", **cmp}
    a.out.write_text(json.dumps(rec, indent=1))
    print(json.dumps({k: rec[k] for k in ("status", "criteria", "mean_rel_loss_diff", "final_bpb", "throughput")}))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
