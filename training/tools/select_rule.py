"""Applies the pre-registered selection rules of docs/prep/08_evaluation_framework.md §3.

``lr``           (lr-v1.1): lowest final validation loss; excludes non-finite runs and runs whose
                 training loss spiked above 1.5x its running minimum after warmup. v1.1 adds
                 fail-closed validity checks (incomplete or mismatched runs are rejected).
``tokenizer``    (tokenizer-v1.2): rule v1.1 unchanged (lowest validation bits per byte; 0.5 %
                 relative tie band; ties to the highest training bytes/s, then the smaller vocabulary;
                 hard requirements on round trip, control IDs and byte fallback). v1.2 changes only
                 measurement: full-split validation (identical scored bytes), exact one-epoch training
                 consumption (consumed source bytes within 0.1 %), training bytes/s from the bytes
                 actually consumed, and the mean over the pre-registered seeds.
``architecture-v2`` (protocol 18): core vs wide with three paired seeds; equal-token noninferiority/superiority,
                 dedicated equal-compute runs, critical-domain gates, registered equivalence criterion.
``ec-budget``    (protocol 18 amendment 18.1): faster shape's equal-compute update count from screening throughput.
``architecture`` (architecture-v1): keep ``j1-core-28x640`` unless an alternative improves BOTH the
                 equal-token and the equal-compute validation bpb by more than the noise band
                 (max(0.5 %, 2 x pooled relative seed SD of the small-model tokenizer pilots)).

Every rule fails closed: a missing, non-finite, incomplete, stale or mismatched input yields
``decision: null`` with ``status: INVALID`` and the reasons, never a silent winner.
The decision and every input value are printed as JSON so the record is reproducible.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import statistics
import sys
from datetime import datetime
from pathlib import Path

STEM_PROBES = ("python", "json_tools", "structured", "paths_urls", "math", "science", "english", "identifiers_rare")
TIE_REL = 0.005
CONSUMED_BYTES_REL_TOL = 0.001
CORE = "j1-core-28x640"


class Invalid(Exception):
    pass


def _finite(x) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(x)


def _status(path: Path) -> dict | None:
    st = path.with_suffix(".status")
    return json.loads(st.read_text()) if st.is_file() else None


def _diverged(path: Path) -> str | None:
    """Reason when a run ended on the trainer's non-finite abort (exit != 0 and an ``abort`` event)."""
    st = _status(path)
    m = path.with_suffix(".jsonl")
    if st is None or st.get("exit_code") in (0, None) or not m.is_file():
        return None
    for line in m.read_text().splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("event") == "abort" and str(r.get("reason", "")).startswith("non_finite"):
            return str(r["reason"])
    return None


def _load(path: Path) -> dict:
    if not path.is_file():
        raise Invalid(f"missing summary {path}")
    st = _status(path)
    if st is None or st.get("exit_code") != 0:
        raise Invalid(f"{path.name}: run status missing or nonzero ({st})")
    d = json.loads(path.read_text())
    c = d.get("config") or {}
    if d.get("run_id") != path.stem or c.get("run_id") != path.stem:
        raise Invalid(f"{path.name}: summary run_id {d.get('run_id')} does not match the file")
    if d.get("final_step") != c.get("steps"):
        raise Invalid(f"{path.name}: incomplete run (final_step {d.get('final_step')} != steps {c.get('steps')})")
    if not _finite(d.get("last_eval")):
        raise Invalid(f"{path.name}: non-finite or missing final evaluation")
    m = path.with_suffix(".jsonl")
    if m.is_file():
        steps = []
        for line in m.read_text().splitlines():
            r = json.loads(line)
            if r.get("event") == "restored":
                d["_resumed"] = True
            if r.get("event") == "eval":
                steps.append(r["step"])
        if any(b <= a for a, b in itertools.pairwise(steps)):
            raise Invalid(f"{m.name}: evaluation steps not strictly increasing (mixed attempts?)")
    return d


def _same(rows: list[dict], key, what: str) -> None:
    vals = {json.dumps(key(r), sort_keys=True) for r in rows}
    if len(vals) != 1:
        raise Invalid(f"runs disagree on {what}: {sorted(vals)}")


def _spiked(metrics: Path, warmup: int) -> bool:
    best = math.inf
    for line in metrics.read_text().splitlines():
        r = json.loads(line)
        if "loss" not in r or "step" not in r or r.get("event"):
            continue
        if not math.isfinite(r["loss"]):
            return True
        if r["step"] > warmup and r["loss"] > 1.5 * best:
            return True
        best = min(best, r["loss"])
    return False


def _fail(rule: str, e: Exception, rows=None) -> dict:
    return {"rule": rule, "status": "INVALID", "decision": None, "reason": str(e), "rows": rows or []}


CAUGHT = (Invalid, KeyError, TypeError, ValueError)


def select_lr(runs: list[Path]) -> dict:
    """Every listed run must be complete or have ended on the trainer's non-finite abort (excluded)."""
    rows: list[dict] = []
    try:
        ds = []
        for s in runs:
            why = None if s.is_file() else _diverged(s)
            if why:
                rows.append({"run": s.stem, "lr": None, "val_loss": None, "eligible": False, "diverged": why})
                continue
            ds.append((s, _load(s)))
        if not ds:
            raise Invalid("no completed run")
        _same(
            [d for _, d in ds],
            lambda d: (d["config"]["preset"], d["config"]["data"], d["config"]["steps"], d["config"]["seed"]),
            "preset/data/steps/seed",
        )
        _same([d for _, d in ds], lambda d: d.get("device_name"), "device (no mixed hardware in one comparison)")
        _same([d for _, d in ds], lambda d: d["code"]["package_sha256"], "code identity")
        _same(
            [d for _, d in ds],
            lambda d: (
                (d.get("eval_set") or {}).get("kind"),
                {k: v for k, v in (d.get("data_identity") or {}).items() if k != "seed"},
            ),
            "evaluation set and data identity",
        )
        for s, d in ds:
            metrics = s.with_suffix(".jsonl")
            if not metrics.is_file():
                raise Invalid(f"missing metrics {metrics}")
            spiked = _spiked(metrics, d["config"]["warmup"])
            rows.append(
                {
                    "run": s.stem,
                    "lr": d["config"]["lr"],
                    "val_loss": d["last_eval"],
                    "eligible": not spiked,
                    "spiked": spiked,
                }
            )
    except CAUGHT as e:
        return _fail("lr-v1.1", e, rows)
    eligible = [r for r in rows if r["eligible"]]
    if not eligible:
        return {"rule": "lr-v1.1", "status": "NO_ELIGIBLE", "decision": None, "rows": rows, "reason": "no eligible run"}
    best = min(eligible, key=lambda r: r["val_loss"])
    return {"rule": "lr-v1.1", "status": "DECIDED", "decision": best["lr"], "rows": rows}


def _rel_sd(xs: list[float]) -> float:
    return statistics.stdev(xs) / statistics.fmean(xs) if len(xs) > 1 else math.nan


def pooled_rel_sd(groups: dict[str, list[float]]) -> tuple[float, int]:
    """Pooled relative SD over groups (each group: one configuration, several seeds)."""
    ss, dof = 0.0, 0
    for xs in groups.values():
        if len(xs) > 1:
            m = statistics.fmean(xs)
            ss += sum(((x - m) / m) ** 2 for x in xs)
            dof += len(xs) - 1
    return (math.sqrt(ss / dof) if dof else math.nan), dof


def _tok_runs(runs: list[Path]) -> dict[str, list[dict]]:
    """Groups tokenizer-pilot summaries by candidate (run ids ``tok-<cand>-s<seed>``)."""
    by: dict[str, list[dict]] = {}
    for s in runs:
        d = _load(s)
        rid = d["config"]["run_id"]
        if "-s" not in rid:
            raise Invalid(f"{rid}: run id lacks a seed suffix")
        cand = rid.rsplit("-s", 1)[0]
        det = d.get("last_eval_detail") or {}
        if not _finite(det.get("bpb")):
            raise Invalid(f"{rid}: no finite validation bpb")
        if (d.get("eval_set") or {}).get("kind") != "full_split_eval_v1":
            raise Invalid(f"{rid}: not evaluated on the full validation split")
        if not d.get("consumed") or "target_bytes" not in d["consumed"]:
            raise Invalid(f"{rid}: no exact-epoch consumption record")
        by.setdefault(cand, []).append(d)
    return by


def _throughput(tp: dict[str, Path]) -> dict[str, dict]:
    """A100 reference throughput runs (amendment 15): complete 300-update runs on one device type."""
    out = {}
    for cand, p in tp.items():
        if not p.is_file():
            raise Invalid(f"missing throughput summary {p}")
        st = _status(p)
        if st is None or st.get("exit_code") != 0:
            raise Invalid(f"{p.name}: throughput run status missing or nonzero")
        d = json.loads(p.read_text())
        if d.get("final_step") != 300 or not _finite(d.get("tokens_per_s_median")):
            raise Invalid(f"{p.name}: throughput run incomplete or non-finite")
        out[cand] = d
    _same(list(out.values()), lambda d: d.get("device_name"), "throughput device")
    if "A100" not in str(next(iter(out.values())).get("device_name")):
        raise Invalid("tie-break throughput must come from the A100 reference (amendment 15)")
    _same(list(out.values()), lambda d: d["code"]["package_sha256"], "throughput code identity")
    return out


def select_tokenizer(
    runs: list[Path], evals: dict[str, Path], seeds: list[int], tp: dict[str, Path] | None = None
) -> dict:
    rows: list[dict] = []
    try:
        by = _tok_runs(runs)
        allrun = [d for ds in by.values() for d in ds]
        _same(allrun, lambda d: d.get("device_name"), "device (no mixed hardware in one comparison)")
        tpd = _throughput(tp) if tp else None
        if tpd is None and "A100" not in str(allrun[0].get("device_name")):
            raise Invalid("pilots ran off the A100: the A100 throughput runs (--tp) are required (amendment 15)")
        if tpd is not None and set(tpd) != set(by):
            raise Invalid(f"throughput candidates {sorted(tpd)} != pilot candidates {sorted(by)}")
        _same(allrun, lambda d: d["code"]["package_sha256"], "code identity")
        _same(
            allrun,
            lambda d: (
                d["config"]["preset"],
                d["config"]["seq_len"],
                d["config"]["micro_bs"] * d["config"]["accum"],
                d["config"]["lr"],
            ),
            "model/batch/lr",
        )
        _same(
            allrun,
            lambda d: (d["eval_set"]["documents"], d["eval_set"]["scored_bytes"]),
            "validation documents and scored bytes",
        )
        _same(allrun, lambda d: d["consumed"]["prefix_bytes"], "declared prefix source bytes")
        tb = [d["consumed"]["target_bytes"] for d in allrun]
        if max(tb) / min(tb) - 1 > CONSUMED_BYTES_REL_TOL:
            raise Invalid(
                f"consumed training bytes differ by more than {CONSUMED_BYTES_REL_TOL:.1%}: {min(tb)}..{max(tb)}"
            )
        for cand, ds in by.items():
            got = sorted(d["config"]["seed"] for d in ds)
            if got != sorted(seeds):
                raise Invalid(f"{cand}: seeds {got} != pre-registered {sorted(seeds)}")
            if cand not in evals:
                raise Invalid(f"{cand}: no tokenizer eval.json supplied")
            ev = json.loads(evals[cand].read_text())
            tok_sha = {d["data_identity"]["tokenizer_json_sha256"] for d in ds}
            if len(tok_sha) != 1 or ev.get("tokenizer_json_sha256") not in tok_sha:
                raise Invalid(
                    f"{cand}: eval.json tokenizer {ev.get('tokenizer_json_sha256')} != trained shards {sorted(tok_sha)}"
                )
            hard = (
                sum(x["roundtrip_failures"] for x in ev["domains"].values()) == 0
                and sum(x["control_ids_from_text"] for x in ev["domains"].values()) == 0
                and sum(p["roundtrip_failures"] for p in ev["probes"].values()) == 0
                and sum(p.get("control_ids_from_text", 0) for p in ev["probes"].values()) == 0
                and all(k in ev["probes"] and ev["probes"][k].get("byte_fallback_tokens", 1) == 0 for k in STEM_PROBES)
                and ev["byte_fallback_rate"] <= 0.005
            )
            bpbs = [d["last_eval_detail"]["bpb"] for d in ds]
            rows.append(
                {
                    "candidate": cand,
                    "vocab": ev["total_vocab"],
                    "seeds": sorted(d["config"]["seed"] for d in ds),
                    "val_bpb_by_seed": {d["config"]["seed"]: d["last_eval_detail"]["bpb"] for d in ds},
                    "val_bpb": statistics.fmean(bpbs),
                    "val_loss_per_token": statistics.fmean(d["last_eval"] for d in ds),
                    "train_bytes_per_s": (
                        # A100 tokens/s x consumed source bytes per target token (a data property of the prefix)
                        tpd[cand]["tokens_per_s_median"]
                        * ds[0]["consumed"]["target_bytes"]
                        / ds[0]["consumed"]["target_tokens"]
                        if tpd is not None
                        else statistics.fmean(d["train_bytes_per_s_median"] for d in ds)
                    ),
                    "throughput_device": tpd[cand].get("device_name") if tpd is not None else ds[0].get("device_name"),
                    "train_tokens_per_s": statistics.fmean(d["tokens_per_s_median"] for d in ds),
                    "consumed_target_bytes": ds[0]["consumed"]["target_bytes"],
                    "byte_fallback_rate": ev["byte_fallback_rate"],
                    "n_params": ds[0]["n_params"],
                    "hard_requirements": hard,
                }
            )
    except CAUGHT as e:
        return _fail("tokenizer-v1.2", e, rows)
    sd, dof = pooled_rel_sd({r["candidate"]: list(r["val_bpb_by_seed"].values()) for r in rows})
    ok = [r for r in rows if r["hard_requirements"]]
    if not ok:
        return {
            "rule": "tokenizer-v1.2",
            "status": "NO_ELIGIBLE",
            "decision": None,
            "rows": rows,
            "reason": "no candidate passed hard requirements",
        }
    best = min(r["val_bpb"] for r in ok)
    tied = [r for r in ok if r["val_bpb"] <= best * (1 + TIE_REL)]
    choice = sorted(tied, key=lambda r: (-r["train_bytes_per_s"], r["vocab"]))[0]
    # Pre-registered uncertainty flag: an untied runner-up within 2 pooled seed SDs could change the result.
    untied = sorted(r["val_bpb"] for r in ok if r not in tied)
    gap = (untied[0] / best - 1) if untied else math.inf
    needs = _finite(sd) and gap < 2 * sd
    return {
        "rule": "tokenizer-v1.2",
        "status": "DECIDED",
        "decision": choice["candidate"],
        "best_bpb": best,
        "tied": [r["candidate"] for r in tied],
        "pooled_rel_seed_sd": sd,
        "pooled_dof": dof,
        "runner_up_gap_rel": gap if math.isfinite(gap) else None,
        "uncertainty_could_change_result": needs,
        "rows": rows,
    }


def _curve(metrics: Path, first_step_s: float) -> list[tuple[float, float]]:
    """(training-only seconds excluding the first update, validation bpb) at each evaluation."""
    pts = []
    for line in metrics.read_text().splitlines():
        r = json.loads(line)
        if r.get("event") == "eval":
            if not (_finite(r.get("train_seconds")) and _finite(r.get("eval_bpb"))):
                raise Invalid(f"{metrics.name}: eval event without train_seconds/eval_bpb")
            pts.append((r["train_seconds"] - first_step_s, r["eval_bpb"]))
    if len(pts) < 2:
        raise Invalid(f"{metrics.name}: fewer than two evaluations")
    return pts


def _interp(pts: list[tuple[float, float]], t: float) -> float:
    if t < pts[0][0] - 1e-9 or t > pts[-1][0] + 1e-9:
        raise Invalid(
            f"equal-compute time {t:.1f}s outside observed curve [{pts[0][0]:.1f}, {pts[-1][0]:.1f}] (no extrapolation)"
        )
    for (t0, y0), (t1, y1) in itertools.pairwise(pts):
        if t0 <= t <= t1:
            return y0 if t1 == t0 else y0 + (y1 - y0) * (t - t0) / (t1 - t0)
    return pts[-1][1]


def select_architecture(runs: list[Path], noise_runs: list[Path], required: list[str] | None = None) -> dict:
    rows: list[dict] = []
    try:
        noise = _tok_runs(noise_runs)
        sd, dof = pooled_rel_sd({k: [d["last_eval_detail"]["bpb"] for d in v] for k, v in noise.items()})
        if not _finite(sd) or dof < 3:
            raise Invalid("noise band needs at least three degrees of freedom of small-model seed repeats")
        band = max(TIE_REL, 2 * sd)
        ds = []
        for s in runs:
            d = _load(s)
            if d.get("_resumed"):
                raise Invalid(f"{s.name}: resumed run; its training-time curve is not comparable")
            if not _finite((d.get("last_eval_detail") or {}).get("bpb")):
                raise Invalid(f"{s.name}: no finite validation bpb")
            if not d.get("consumed"):
                raise Invalid(f"{s.name}: no exact-epoch consumption record")
            m = s.with_suffix(".jsonl")
            if not m.is_file():
                raise Invalid(f"missing metrics {m}")
            ds.append((d, _curve(m, d["first_step_s"])))
        _same([d for d, _ in ds], lambda d: d["code"]["package_sha256"], "code identity")
        _same([d for d, _ in ds], lambda d: d.get("device_name"), "device (no mixed hardware in one comparison)")
        _same(
            [d for d, _ in ds],
            lambda d: (
                {k: v for k, v in d["data_identity"].items() if k != "seed"},
                d["consumed"]["target_bytes"],
                d["config"]["steps"],
            ),
            "training data/steps",
        )
        _same(
            [d for d, _ in ds], lambda d: (d["eval_set"]["documents"], d["eval_set"]["scored_bytes"]), "validation set"
        )
        _same(
            [d for d, _ in ds],
            lambda d: (d["config"]["seq_len"], d["config"]["micro_bs"] * d["config"]["accum"]),
            "tokens per update",
        )
        presets = {d["config"]["preset"] for d, _ in ds}
        if CORE not in presets or len(presets) < 2:
            raise Invalid(f"need {CORE} and at least one alternative, got {sorted(presets)}")
        if required and set(required) != presets:
            raise Invalid(f"required candidates {sorted(required)} != completed {sorted(presets)}")
        by: dict[str, list] = {}
        for d, c in ds:
            by.setdefault(d["config"]["preset"], []).append((d, c))
        seeds = {p: sorted(d["config"]["seed"] for d, _ in v) for p, v in by.items()}
        t_star = min(c[-1][0] for _, c in ds)
        for p, v in sorted(by.items()):
            rows.append(
                {
                    "preset": p,
                    "seeds": seeds[p],
                    "n_params": v[0][0]["n_params"],
                    "lr": v[0][0]["config"]["lr"],
                    "equal_token_bpb": statistics.fmean(d["last_eval_detail"]["bpb"] for d, _ in v),
                    "equal_compute_bpb": statistics.fmean(_interp(c, t_star) for _, c in v),
                    "train_seconds_excl_first_update": statistics.fmean(c[-1][0] for _, c in v),
                    "first_update_seconds": statistics.fmean(d["first_step_s"] for d, _ in v),
                    "wall_seconds": statistics.fmean(d["wall_s"] for d, _ in v),
                    "tokens_per_s_median": statistics.fmean(d["tokens_per_s_median"] for d, _ in v),
                    "peak_vram_gib": max(d["peak_vram_gib"] for d, _ in v),
                }
            )
    except CAUGHT as e:
        return _fail("architecture-v1", e, rows)
    core = next(r for r in rows if r["preset"] == CORE)
    for r in rows:
        r["equal_token_improvement_rel"] = (core["equal_token_bpb"] - r["equal_token_bpb"]) / core["equal_token_bpb"]
        r["equal_compute_improvement_rel"] = (core["equal_compute_bpb"] - r["equal_compute_bpb"]) / core[
            "equal_compute_bpb"
        ]
        r["beats_band_both"] = (
            r["preset"] != CORE
            and r["equal_token_improvement_rel"] > band
            and r["equal_compute_improvement_rel"] > band
        )
        r["beats_band_one"] = r["preset"] != CORE and (r["equal_token_improvement_rel"] > band) != (
            r["equal_compute_improvement_rel"] > band
        )
    winners = [r for r in rows if r["beats_band_both"]]
    res = {
        "rule": "architecture-v1",
        "noise_band_rel": band,
        "pooled_rel_seed_sd": sd,
        "pooled_dof": dof,
        "equal_compute_seconds": t_star,
        "equal_compute_boundary": (
            "cumulative synchronized optimizer-update time, excluding the first (compile) update; "
            "evaluation and checkpoint time excluded"
        ),
        "rows": rows,
    }
    if winners:
        w = max(winners, key=lambda r: min(r["equal_token_improvement_rel"], r["equal_compute_improvement_rel"]))
        if len(seeds[w["preset"]]) < 2 or len(seeds[CORE]) < 2:
            res.update(
                status="NEEDS_SEED_REPEATS",
                decision=None,
                repeat=[CORE, w["preset"]],
                reason="one seed clears the band on both comparisons; pre-registered repeats needed",
            )
        else:
            res.update(status="DECIDED", decision=w["preset"])
        return res
    if any(r["beats_band_one"] for r in rows):
        res.update(
            status="INCONCLUSIVE",
            decision=CORE,
            provisional=True,
            reason="an alternative clears the band on only one comparison; core retained provisionally",
        )
        return res
    res.update(
        status="DECIDED",
        decision=CORE,
        provisional=True,
        reason="no alternative clears the noise band on both comparisons; short-run evidence only",
    )
    return res


# ---------------------------------------------------------------------------------------------------------------
# architecture-v2 and ec-budget (docs/prep/18_architecture_followup_protocol.md, registered before any result;
# review fixes F1-F16 applied before any run)

WIDE = "j1-wide-20x768"
NI_REL = 0.005
M_REL = 0.005
EFF_MIN_REL = 0.03
DOM_LIMIT_REL = -0.015
CRITICAL = ("arxiv-papers", "fineweb-edu-dedup", "finemath-4plus", "pressbooks", "stackexchange", "stackv2-edu",
            "wikimedia")  # fmt: skip
CRITICAL_MIN_BYTES = 700_000
T90 = {2: 1.8856, 4: 1.5332}  # one-sided 90 % t quantiles (alpha = 0.10) by degrees of freedom
REGISTERED_SEEDS = ((21, 22, 23), (21, 22, 23, 24, 25))
EC_TIME_TOL = 0.02
R_BOUNDS = (0.80, 1.25)
ET_UPDATES = 2792
SCREEN_UPDATES = 1396
SCREEN_SEED = 11
VAL_DOCS, VAL_BYTES = 2576, 12_311_466  # protocol 18b: successor validation split (v0.2.2)
FIXED_CONFIG = ("seq_len", "micro_bs", "accum", "warmup", "min_lr_ratio", "wd", "wd_policy", "clip", "opt",
                "precision", "compile", "compile_mode", "sdpa", "exact_epoch", "eval_set", "preload_data",
                "deterministic", "data", "tokenizer", "gqa_kernel", "verify_data")  # fmt: skip


def _paired(by_seed_x: dict[int, float], by_seed_y: dict[int, float]) -> list[float]:
    """Improvement of X over Y per seed: (y - x) / y (positive = X has lower bpb)."""
    return [(by_seed_y[s] - by_seed_x[s]) / by_seed_y[s] for s in sorted(by_seed_x)]


def _stats(xs: list[float]) -> dict:
    n = len(xs)
    if n not in (3, 5):
        raise Invalid(f"registered seed counts are 3 or 5, got {n}")
    if not all(_finite(x) for x in xs):
        raise Invalid("non-finite paired difference")
    m = statistics.fmean(xs)
    sd = statistics.stdev(xs)
    h = T90[n - 1] * sd / math.sqrt(n)
    return {"n": n, "values": xs, "mean": m, "sd": sd, "lower90": m - h, "upper90": m + h}


def _host(s: Path) -> dict:
    h = s.with_suffix(".host.json")
    if not h.is_file():
        raise Invalid(f"{s.name}: missing host record {h.name}")
    d = json.loads(h.read_text())
    if not d.get("gpu_uuid") or not d.get("pod_id"):
        raise Invalid(f"{h.name}: host record lacks gpu_uuid/pod_id")
    return {"gpu_uuid": d["gpu_uuid"], "pod_id": d["pod_id"]}


def _paid_seconds(s: Path) -> float:
    st = _status(s) or {}
    try:
        a = datetime.fromisoformat(st["start_utc"].replace("Z", "+00:00"))
        b = datetime.fromisoformat(st["end_utc"].replace("Z", "+00:00"))
    except (KeyError, ValueError, AttributeError) as e:
        raise Invalid(f"{s.name}: status lacks start/end times") from e
    sec = (b - a).total_seconds()
    if not sec > 0:
        raise Invalid(f"{s.name}: non-positive paid run time")
    return sec


def _arch_run(s: Path, stage: str) -> dict:
    d = _load(s)
    if not s.with_suffix(".jsonl").is_file():
        raise Invalid(f"{s.name}: metrics JSONL required (resume detection)")
    if d.get("_resumed") or d.get("steps_run") != d.get("final_step"):
        raise Invalid(f"{s.name}: resumed run; its training-time record is not comparable")
    det = d.get("last_eval_detail") or {}
    if not _finite(det.get("bpb")):
        raise Invalid(f"{s.name}: no finite validation bpb")
    if det.get("step") != d["config"]["steps"]:
        raise Invalid(f"{s.name}: final evaluation at step {det.get('step')} != {d['config']['steps']}")
    es = d.get("eval_set") or {}
    if (es.get("kind"), es.get("documents"), es.get("scored_bytes")) != ("full_split_eval_v1", VAL_DOCS, VAL_BYTES):
        raise Invalid(f"{s.name}: evaluation set is not the registered full validation split")
    for k, v in (det.get("by_source") or {}).items():
        if not _finite(v.get("bpb")):
            raise Invalid(f"{s.name}: non-finite bpb for source {k}")
    if d["config"].get("stage_id") != stage:
        raise Invalid(f"{s.name}: stage id {d['config'].get('stage_id')} != {stage}")
    if not d.get("consumed"):
        raise Invalid(f"{s.name}: no exact-epoch consumption record")
    for k in ("train_seconds", "first_step_s", "tokens_per_s_median"):
        if not _finite(d.get(k)):
            raise Invalid(f"{s.name}: missing or non-finite {k}")
    if d["consumed"].get("target_tokens") != d["config"]["steps"] * 65536:
        raise Invalid(f"{s.name}: consumed targets differ from steps x 65,536")
    d["_host"] = _host(s)
    d["_paid_s"] = _paid_seconds(s)
    return d


def _train_s(d: dict) -> float:
    """Synchronized optimizer-update seconds excluding the first (compile) update."""
    return float(d["train_seconds"]) - float(d["first_step_s"])


def _per_update_s(d: dict) -> float:
    return _train_s(d) / (d["config"]["steps"] - 1)


def select_architecture_v2(et_runs: list[Path], ec_runs: list[Path], seeds: list[int], amendment: Path | None) -> dict:
    rows: dict = {}
    try:
        if tuple(sorted(seeds)) not in REGISTERED_SEEDS:
            raise Invalid(f"seed set {sorted(seeds)} is not a registered confirmation seed set")
        if amendment is None:
            raise Invalid("amendment 18.1 record required (--amendment)")
        am = json.loads(amendment.read_text())
        if am.get("amendment") != "18.1":
            raise Invalid("amendment record is not 18.1")
        frozen_lr = {p: float(v) for p, v in am["selected_lr"].items()}
        fast, slow, u_fast = am["ec_budget"]["faster"], am["ec_budget"]["slower"], int(am["ec_budget"]["decision"])
        et = [(s, _arch_run(s, "fu3-confirm-et")) for s in et_runs]
        ec = [(s, _arch_run(s, "fu3-confirm-ec")) for s in ec_runs]
        alld = [d for _, d in et + ec]
        _same(alld, lambda d: d["code"]["package_sha256"], "code identity")
        _same(alld, lambda d: (d.get("device_name"), d["_host"]), "host/GPU (no host change within a comparison)")
        if am["ec_budget"].get("host") != alld[0]["_host"]:
            raise Invalid("confirmation host/GPU differs from the screening host that produced amendment 18.1")
        _same(alld, lambda d: {k: d["config"].get(k) for k in FIXED_CONFIG}, "fixed settings")
        _same(
            alld,
            lambda d: {k: v for k, v in d["data_identity"].items() if k not in ("seed", "limit_tokens")},
            "training data identity",
        )
        if {d["config"]["steps"] for _, d in et} != {ET_UPDATES}:
            raise Invalid(f"equal-token runs must use the registered {ET_UPDATES} updates")
        by: dict[str, dict[int, dict]] = {}
        for _, d in et:
            p, sd_ = d["config"]["preset"], d["config"]["seed"]
            if sd_ in by.setdefault(p, {}):
                raise Invalid(f"duplicate equal-token run {p} seed {sd_}")
            by[p][sd_] = d
        if set(by) != {CORE, WIDE} or {fast, slow} != {CORE, WIDE}:
            raise Invalid(f"equal-token presets {sorted(by)} / amendment shapes do not match {[CORE, WIDE]}")
        for p in by:
            if sorted(by[p]) != sorted(seeds):
                raise Invalid(f"{p}: equal-token seeds {sorted(by[p])} != registered {sorted(seeds)}")
            for d in by[p].values():
                if not math.isclose(d["config"]["lr"], frozen_lr[p], rel_tol=1e-9):
                    raise Invalid(f"{p}: LR {d['config']['lr']} differs from the frozen {frozen_lr[p]}")
        for s_ in seeds:  # paired seeds see the same ordered windows
            _same([by[CORE][s_], by[WIDE][s_]], lambda d: d["data_identity"].get("seed"), f"data order seed {s_}")
        ecd: dict[int, dict] = {}
        for _, d in ec:
            if d["config"]["preset"] != fast:
                raise Invalid(f"equal-compute run of {d['config']['preset']}; only the faster shape {fast} gets EC")
            if d["config"]["steps"] != u_fast:
                raise Invalid(f"equal-compute run has {d['config']['steps']} updates, amendment 18.1 says {u_fast}")
            if not math.isclose(d["config"]["lr"], frozen_lr[fast], rel_tol=1e-9):
                raise Invalid("equal-compute run uses a different LR from the frozen faster-shape LR")
            if d["config"]["seed"] in ecd:
                raise Invalid(f"duplicate equal-compute seed {d['config']['seed']}")
            ecd[d["config"]["seed"]] = d
        if sorted(ecd) != sorted(seeds):
            raise Invalid(f"equal-compute seeds {sorted(ecd)} != registered {sorted(seeds)}")
        tps = {p: statistics.median(1.0 / _per_update_s(d) for d in by[p].values()) for p in by}
        if tps[fast] < tps[slow]:
            raise Invalid("faster/slower order observed in confirmation contradicts amendment 18.1")
        t_star = statistics.median(_train_s(d) for d in by[slow].values())
        mism = {s_: _train_s(d) / t_star - 1 for s_, d in ecd.items()}
        ec_status = "MATCHED" if all(abs(v) <= EC_TIME_TOL for v in mism.values()) else "UNMATCHED"

        def q(d: dict) -> float:
            return float(d["last_eval_detail"]["bpb"])

        et_q = {p: {s_: q(d) for s_, d in by[p].items()} for p in by}
        ec_q = {slow: et_q[slow], fast: {s_: q(d) for s_, d in ecd.items()}}
        for d in alld:
            src = d["last_eval_detail"].get("by_source") or {}
            got = tuple(sorted(k for k, v in src.items() if v["bytes"] >= CRITICAL_MIN_BYTES))
            if got != tuple(sorted(CRITICAL)):
                raise Invalid(f"{d['run_id']}: critical domains {got} != registered {CRITICAL}")
        res_c: dict = {}
        for x, y in ((CORE, WIDE), (WIDE, CORE)):
            et_s = _stats(_paired(et_q[x], et_q[y]))
            ec_s = _stats(_paired(ec_q[x], ec_q[y])) if ec_status == "MATCHED" else None
            dom = {}
            for dname in CRITICAL:
                xs = {s_: by[x][s_]["last_eval_detail"]["by_source"][dname]["bpb"] for s_ in seeds}
                ys = {s_: by[y][s_]["last_eval_detail"]["by_source"][dname]["bpb"] for s_ in seeds}
                dom[dname] = statistics.fmean(_paired(xs, ys))
            res_c[x] = {
                "equal_token": et_s,
                "equal_compute": ec_s,
                "domain_mean_improvement": dom,
                "NI_ET": et_s["lower90"] >= -NI_REL,
                "SUP_ET": et_s["mean"] >= M_REL and et_s["lower90"] > 0,
                "SUP_EC": bool(ec_s and ec_s["mean"] >= M_REL and ec_s["lower90"] > 0),
                "DOM": all(v >= DOM_LIMIT_REL for v in dom.values()),
            }
        eff = {p: statistics.fmean(d["consumed"]["target_tokens"] / d["_paid_s"] for d in by[p].values()) for p in by}
        rows = {
            "candidates": res_c,
            "et_bpb": et_q,
            "ec_bpb": ec_q,
            "critical_domains": list(CRITICAL),
            "slower": slow,
            "faster": fast,
            "updates_per_s_median": tps,
            "paid_targets_per_s": eff,
            "paid_time_basis": "run process start to end (bench status): data load/verify, compile, training, "
            "evaluations and checkpoint saves",
            "equal_compute_seconds": t_star,
            "equal_compute_status": ec_status,
            "ec_time_mismatch_rel": mism,
            "lr": frozen_lr,
            "u_fast": u_fast,
            "host": alld[0]["_host"],
        }
    except (*CAUGHT, OSError, json.JSONDecodeError) as e:
        return _fail("architecture-v2", e, [rows] if rows else [])
    res = {
        "rule": "architecture-v2",
        "margins": {"NI": NI_REL, "M": M_REL, "DOM": DOM_LIMIT_REL, "EFF": EFF_MIN_REL, "alpha_one_sided": 0.10},
        **rows,
    }
    c = res_c
    sup = [x for x in (CORE, WIDE) if c[x]["SUP_ET"] and c[x]["DOM"]]
    if sup:
        return {**res, "status": "SELECTED", "decision": sup[0], "basis": "equal-token superiority with domain gates"}
    ni_ec = [x for x in (CORE, WIDE) if c[x]["NI_ET"] and c[x]["DOM"] and c[x]["SUP_EC"]]
    if len(ni_ec) == 1:
        return {
            **res,
            "status": "SELECTED",
            "decision": ni_ec[0],
            "basis": "equal-token noninferiority, domain gates, material equal-compute improvement",
        }
    if all(c[x]["NI_ET"] and c[x]["DOM"] for x in (CORE, WIDE)):
        hi, lo = sorted((CORE, WIDE), key=lambda p: eff[p], reverse=True)
        gain = eff[hi] / eff[lo] - 1
        if gain >= EFF_MIN_REL:
            return {
                **res,
                "status": "ENGINEERING_CHOICE_UNDER_UNCERTAINTY",
                "decision": hi,
                "paid_efficiency_gain_rel": gain,
                "basis": "equal-token quality equivalent within +/-0.5 % (TOST, alpha 0.10 each side, i.e. a "
                "two-sided 80 % interval) and domain gates pass; paid valid targets per second higher by >= 3 % "
                "(registered acceptable-risk criterion)",
            }
        return {
            **res,
            "status": "INCONCLUSIVE",
            "decision": None,
            "paid_efficiency_gain_rel": gain,
            "basis": "quality equivalent but paid-efficiency difference below the registered 3 % materiality",
        }
    return {**res, "status": "INCONCLUSIVE", "decision": None, "basis": "no registered selection condition holds"}


def ec_budget(core_runs: list[Path], wide_runs: list[Path]) -> dict:
    """Amendment 18.1: the faster shape's equal-compute update count from same-host screening step times."""
    try:
        rate: dict[str, float] = {}
        alld = []
        for name, runs in ((CORE, core_runs), (WIDE, wide_runs)):
            ds = [_load(s) for s in runs]
            for s, d in zip(runs, ds, strict=True):
                d["_host"] = _host(s)
                if not _finite(d.get("train_seconds")) or not _finite(d.get("first_step_s")):
                    raise Invalid(f"{s.name}: missing training-time record")
                if not s.with_suffix(".jsonl").is_file():
                    raise Invalid(f"{s.name}: metrics JSONL required (resume detection)")
                if d.get("_resumed") or d.get("steps_run") != d.get("final_step"):
                    raise Invalid(f"{s.name}: resumed run; its training-time record is not comparable")
                if d["config"].get("stage_id") != "fu3-lr-screen":
                    raise Invalid(f"{s.name}: stage id {d['config'].get('stage_id')} != fu3-lr-screen")
            if len(ds) < 3:
                raise Invalid(f"{name}: need at least 3 completed screening runs, got {len(ds)}")
            if {d["config"]["preset"] for d in ds} != {name}:
                raise Invalid(f"{name}: screening runs of another preset supplied")
            rate[name] = statistics.median(1.0 / _per_update_s(d) for d in ds)
            alld += ds
        _same(alld, lambda d: (d["config"]["steps"], d["config"]["seed"]), "screening steps/seed")
        _same(alld, lambda d: {k: d["config"].get(k) for k in FIXED_CONFIG}, "screening fixed settings")
        if {d["config"]["steps"] for d in alld} != {SCREEN_UPDATES} or {d["config"]["seed"] for d in alld} != {
            SCREEN_SEED
        }:
            raise Invalid("screening runs must use the registered 1,396 updates and seed 11")
        _same(alld, lambda d: (d.get("device_name"), d["_host"]), "screening host/GPU")
        _same(alld, lambda d: d["code"]["package_sha256"], "screening code identity")
        slow, fast = (CORE, WIDE) if rate[CORE] <= rate[WIDE] else (WIDE, CORE)
        r = rate[fast] / rate[slow]
        if not (R_BOUNDS[0] <= r <= R_BOUNDS[1]):
            raise Invalid(f"throughput ratio {r:.4f} outside registered bounds {R_BOUNDS}; protocol stops")
    except (*CAUGHT, OSError, json.JSONDecodeError) as e:
        return _fail("ec-budget-v1", e)
    return {
        "rule": "ec-budget-v1",
        "status": "DECIDED",
        "decision": round(ET_UPDATES * r),
        "faster": fast,
        "slower": slow,
        "ratio": r,
        "ratio_basis": "median updates per second over training seconds excluding the first (compile) update",
        "updates_per_s_median": rate,
        "et_updates": ET_UPDATES,
        "host": alld[0]["_host"],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["lr", "tokenizer", "architecture", "architecture-v2", "ec-budget"])
    ap.add_argument("summaries", nargs="+", type=Path)
    ap.add_argument("--eval", action="append", default=[], help="CANDIDATE=path/to/eval.json (tokenizer rule)")
    ap.add_argument("--seeds", default="1,2", help="pre-registered seeds per tokenizer candidate")
    ap.add_argument("--tp", action="append", default=[], help="CANDIDATE=A100 throughput summary (amendment 15)")
    ap.add_argument(
        "--noise", nargs="*", type=Path, default=[], help="small-model seed-repeat summaries (architecture rule)"
    )
    ap.add_argument("--required", default="", help="comma-separated presets that must all be present (architecture)")
    ap.add_argument("--ec", nargs="*", type=Path, default=[], help="equal-compute run summaries (architecture-v2)")
    ap.add_argument("--wide", nargs="*", type=Path, default=[], help="wide screening summaries (ec-budget)")
    ap.add_argument("--confirm-seeds", default="21,22,23", help="registered confirmation seeds (architecture-v2)")
    ap.add_argument("--amendment", type=Path, help="amendment_18_1.json (architecture-v2)")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    if a.kind == "lr":
        res = select_lr(a.summaries)
    elif a.kind == "tokenizer":
        res = select_tokenizer(
            a.summaries,
            {k: Path(v) for k, v in (e.split("=", 1) for e in a.eval)},
            [int(x) for x in a.seeds.split(",")],
            {k: Path(v) for k, v in (e.split("=", 1) for e in a.tp)} or None,
        )
    elif a.kind == "architecture":
        res = select_architecture(a.summaries, a.noise, [x for x in a.required.split(",") if x])
    elif a.kind == "architecture-v2":
        res = select_architecture_v2(a.summaries, a.ec, [int(x) for x in a.confirm_seeds.split(",")], a.amendment)
    else:
        res = ec_budget(a.summaries, a.wide)
    text = json.dumps(res, indent=1)
    if a.out:
        a.out.write_text(text)
    print(text)
    return 0 if res["decision"] is not None else 2


if __name__ == "__main__":
    sys.exit(main())
