"""SFT v2.1 decisions (``docs/posttrain/sft_v21_protocol_v1.json``): the single-seed diagnostic and the final gate.

``--mode diagnostic`` scores seed 1 against the v1.2-recipe baseline of seed 1 and the SFT v2 candidate
of seed 1 on the historical v1.3 test bundle (a predeclared diagnostic set, not untouched confirmation).
Outcomes: ``ADVANCE`` (run the registered replication seeds), ``STOP_FAILED`` or ``STOP_INSUFFICIENT``.

``--mode final`` applies the same rules to every registered seed (seed means, each seed must improve on
its primary endpoints) on the historical bundle, and the primary and protected-skill rules once more on the
sealed confirmation bundle. Outcomes: ``QUALIFIED_FOR_REVIEW``, ``FAILED`` or ``INSUFFICIENT_EVIDENCE``.

Rules (all fixed by the protocol before any v2.1 output exists):

* primary generalization vs v1.2: point improvement >= ``min_gain``, paired item-bootstrap CI lower bound
  > ``ci_lower_min``, Holm-adjusted exact McNemar p < alpha (largest per-seed p), every seed improves;
* clean slices (paraphrase:clean, compose_heldout:unseen_function) vs v1.2: CI lower bound > 0;
* new capability vs v2: CI lower bound of (v2.1 - v2) > -margin on each primary endpoint;
* protected skills vs v1.2 (affirmative non-inferiority): PASS iff CI lower bound > -margin; FAIL iff the
  point estimate or CI upper bound < -margin; otherwise INSUFFICIENT. Gains elsewhere never offset;
* knowledge per relation: the same affirmative rule on chat answers vs v1.2, and the candidate's own
  few-shot answers not significantly worse than the frozen base's (exact McNemar, any seed);
* gates: protocol validity, uncertainty, false abstention, full-split bpb ceiling, checkpoint verification,
  smoke/recovery pass and the dev early-stop rule.

Inputs are read fail-closed with the SFT v2 tool's strict loaders; every input file's SHA-256 and the
score files' bundle, scorer and model identities are checked against the protocol and recorded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.sft_decide import early_stop_triggered
from tools.sft_seed_report import holm, mcnemar_p
from tools.sft_v2_decide import _finite_tree, _member, _read, _score_file, paired_seed_mean, resolve_margin

Items = dict[str, dict]
# Rates here are ratios of item counts; a registered boundary (e.g. a margin of exactly 1/12 against a
# difference of exactly -2/24) must not flip on float rounding. Comparisons apply the exact rule with this
# tolerance: a value within TOL of the boundary counts as equal to it.
TOL = 1e-9
IDENTITY = ("run_id", "checkpoint", "manifest_sha256", "model_state_sha256")


def _gt(a: float, b: float) -> bool:
    return a > b + TOL


def _lt(a: float, b: float) -> bool:
    return a < b - TOL


def _chat(d: dict) -> Items:
    return {r["id"]: r for r in d["items"] if r["split"] == "test" and r["variant"] == "chat" and not r.get("unscored")}


def _fewshot(d: dict) -> Items:
    rows = [r for r in d["items"] if r["split"] == "test" and r["variant"] == "base_fewshot"]
    out = {r["id"].removesuffix("#fewshot"): r for r in rows}
    if len(out) != len(rows):
        raise ValueError("duplicate few-shot item IDs")
    return out


def _status(r: dict, margin: float) -> str:
    lo, hi = r["bootstrap_ci95"]
    if _gt(lo, -margin):
        return "PASS"
    return "FAIL" if _lt(hi, -margin) or _lt(r["difference"], -margin) else "INSUFFICIENT"


def evaluate(
    proto: dict,
    cand: dict[int, Items],
    v12: dict[int, Items],
    v2: dict[int, Items] | None,
    sensitivity: dict[str, list[str]],
    margins_from: dict[int, Items],
    *,
    knowledge: bool,
    exposed: frozenset[str] = frozenset(),
    min_n: int = 0,
) -> dict:
    """Rows and verdicts for one bundle. ``margins_from`` holds every baseline seed (auto margins use all three)."""
    seeds = sorted(cand)
    if sorted(v12) != seeds or (v2 is not None and sorted(v2) != seeds):
        raise ValueError("candidate, v1.2 and v2 seeds differ")
    ref = cand[seeds[0]]
    others = [*cand.values(), *v12.values(), *(v2.values() if v2 else [])]
    if any(set(x) != set(ref) for x in others):
        raise ValueError("score files do not cover the same items")
    for x in others:
        if any(any(r[k] != ref[i][k] for k in ("category", "family", "split", "variant")) for i, r in x.items()):
            raise ValueError("paired item metadata differs")
    boot = proto["statistics"]["bootstrap"]
    rs, bs = boot["resamples"], boot["seed"]

    def ids_for(group: str) -> list[str]:
        ids = sorted(i for i, r in ref.items() if _member(r, group))
        if not ids:
            raise ValueError(f"required evaluation group is empty: {group}")
        return ids

    rows: dict[str, dict] = {}
    prim = proto["primary"]
    pv = {}
    for g in prim["endpoints"]:
        r = paired_seed_mean(cand, v12, ids_for(g), rs, bs)
        r["max_seed_p"] = max(x["mcnemar_p"] for x in r["per_seed"].values())
        pv[g] = r["max_seed_p"]
        rows[f"primary:{g}"] = r
    adj = holm(pv)
    for g in prim["endpoints"]:
        r = rows[f"primary:{g}"]
        r["max_seed_p_holm"] = adj[g]
        r["pass"] = (
            not _lt(r["difference"], prim["min_gain"])
            and _gt(r["bootstrap_ci95"][0], prim["ci_lower_min"])
            and all(x["improved"] for x in r["per_seed"].values())
            and adj[g] < prim["alpha"]
        )
    for cls in prim["clean_slices"]:
        ids = sorted(i for i in sensitivity.get(cls, []) if i in ref)
        if not ids:
            if knowledge:  # the historical bundle must carry every registered slice
                raise ValueError(f"clean slice {cls} is empty")
            continue
        r = paired_seed_mean(cand, v12, ids, rs, bs)
        r["pass"] = _gt(r["bootstrap_ci95"][0], 0.0)
        rows[f"clean_slice:{cls}"] = r
    if v2 is not None:
        m = proto["vs_v2"]["margin"]
        for g in proto["vs_v2"]["endpoints"]:
            r = paired_seed_mean(cand, v2, ids_for(g), rs, bs)
            r["margin"] = m
            r["status"] = _status(r, m)
            rows[f"vs_v2:{g}"] = r
    ni = proto["noninferiority"]
    for g, spec in ni["groups"].items():
        ids = ids_for(g)
        r = paired_seed_mean(cand, v12, ids, rs, bs)
        r["margin"] = resolve_margin(spec, ni["auto"], margins_from, ids)
        r["status"] = _status(r, r["margin"])
        if len(ids) < min_n:  # registered minimum size for a gated group on this bundle
            r["status"] = "INSUFFICIENT" if r["status"] == "PASS" else r["status"]
            r["below_min_n"] = min_n
        rows[f"noninferiority:{g}"] = r
        fresh = [i for i in ids if i not in exposed]
        if exposed and fresh and len(fresh) < len(ids):
            # Reported, not gated (amendment 1): items whose tasks the v1.2 baseline trained on removed.
            rows[f"reported_fresh:{g}"] = paired_seed_mean(cand, v12, fresh, rs, bs)
    for g in proto.get("reported_groups", []):
        ids = sorted(i for i, r in ref.items() if _member(r, g))
        if ids:
            rows[f"reported:{g}"] = paired_seed_mean(cand, v12, ids, rs, bs)
    if knowledge:
        kn = proto["knowledge"]
        for g, spec in kn["relations"].items():
            ids = ids_for(g)
            r = paired_seed_mean(cand, v12, ids, rs, bs)
            r["margin"] = resolve_margin(spec, kn["auto"], margins_from, ids)
            r["chat_status"] = _status(r, r["margin"])
            rows[f"knowledge:{g}"] = r
    return rows


def _gates(proto: dict, cand: dict[int, Items], evidence: dict[int, dict], smoke: dict) -> dict:
    g = proto["gates"]
    per_seed = {}
    for s, items in sorted(cand.items()):
        e = evidence[s]
        allv = [r for r in e["scores"]["items"] if r["split"] == "test" and r["variant"] == "chat"]
        fa = [i for i, r in items.items() if r["category"] == "factuality"]
        unc = [i for i, r in items.items() if r["category"] == "uncertainty"]
        hist = e["dev"]["dev_history"]
        completed = bool(hist) and hist[-1]["update"] == e["dev"]["updates"] == proto["training"]["updates"]
        per_seed[s] = {
            "completed_registered_updates": completed,
            "protocol_valid": sum(r["protocol_valid"] for r in allv) / len(allv),
            "uncertainty_strict": sum(items[i]["strict"] for i in unc) / len(unc),
            "false_abstention_on_facts": sum(items[i]["abstained"] for i in fa) / len(fa),
            "bpb_full": e["retention"]["bpb"],
            "retention_mode_ok": e["retention"]["mode"] == "full" and e["retention"]["precision"] == "fp32",
            "checkpoint_verified": e["check"]["verify"][0] is True and e["check"]["exact"] is True,
            "early_stop": early_stop_triggered(hist),
        }
    ps = per_seed.values()
    return {
        "per_seed": per_seed,
        "G-PROTO": all(p["protocol_valid"] >= g["protocol_valid_min"] for p in ps),
        "G-UNC": all(
            p["uncertainty_strict"] >= g["uncertainty_strict_min"]
            and p["false_abstention_on_facts"] <= g["false_abstention_max"]
            for p in ps
        ),
        "G-RET": all(p["retention_mode_ok"] and p["bpb_full"] <= g["full_split_bpb_max"] for p in ps),
        "G-CKPT": all(p["checkpoint_verified"] and p["completed_registered_updates"] for p in ps),
        "G-REC": smoke.get("schema") == "j1.sft-v21-smoke.v1"
        and smoke.get("ab_bitwise") is True
        and all(smoke.get("rollback", {}).get(k, {}).get("final_equals_A") is True for k in ("corrupt", "torn"))
        and smoke.get("pass") is True,
        "EARLY-STOP": not any(p["early_stop"] for p in ps),
    }


def _fewshot_retention(proto: dict, base_few: Items, cand_few: dict[int, Items], rows: dict) -> None:
    alpha = proto["knowledge"]["alpha"]
    for g in proto["knowledge"]["relations"]:
        r = rows[f"knowledge:{g}"]
        ids = r["ids"]
        if any(i not in base_few or any(i not in f for f in cand_few.values()) for i in ids):
            raise ValueError("missing paired few-shot knowledge items")
        worse = {}
        for s, few in sorted(cand_few.items()):
            b_only = sum(1 for i in ids if base_few[i]["strict"] and not few[i]["strict"])
            c_only = sum(1 for i in ids if few[i]["strict"] and not base_few[i]["strict"])
            p = mcnemar_p(b_only, c_only)
            worse[s] = {"candidate_fewshot": sum(few[i]["strict"] for i in ids) / len(ids), "p": p}
            worse[s]["significantly_worse"] = b_only > c_only and p < alpha
        r["base_fewshot"] = sum(base_few[i]["strict"] for i in ids) / len(ids)
        r["fewshot_vs_base"] = worse
        r["retention_pass"] = not any(w["significantly_worse"] for w in worse.values())
        r["status"] = "FAIL" if not r["retention_pass"] else r["chat_status"]


def prior_rows(
    proto: dict, cand: dict[int, Items], prior: dict[int, Items], v12_all: dict[int, Items], exposed: frozenset[str]
) -> dict[str, dict]:
    """Reported, never gated: candidate minus predecessor on the listed groups (all and fresh items), with the
    v1.2 baseline's seed-to-seed accuracy SD on each group as a yardstick for single-seed differences."""
    boot = proto["statistics"]["bootstrap"]
    ref = cand[sorted(cand)[0]]
    out: dict[str, dict] = {}
    for g in proto["prior"]["groups"]:
        ids = sorted(i for i, r in ref.items() if _member(r, g))
        if not ids:
            continue
        out[f"reported_vs_prior:{g}"] = paired_seed_mean(cand, prior, ids, boot["resamples"], boot["seed"])
        fresh = [i for i in ids if i not in exposed]
        if fresh and len(fresh) < len(ids):
            out[f"reported_vs_prior_fresh:{g}"] = paired_seed_mean(cand, prior, fresh, boot["resamples"], boot["seed"])
        accs = [sum(v12_all[s][i]["strict"] for i in ids) / len(ids) for s in sorted(v12_all)]
        mean = sum(accs) / len(accs)
        sd = (sum((a - mean) ** 2 for a in accs) / (len(accs) - 1)) ** 0.5
        out[f"yardstick_v12_seed_sd:{g}"] = {"n": len(ids), "seed_accuracies": accs, "sample_sd": sd}
    return out


def verdict(mode: str, rows: dict, gates: dict, confirm_rows: dict | None = None) -> str:
    gate_ok = all(v for k, v in gates.items() if k != "per_seed")
    prim_ok = all(r["pass"] for k, r in rows.items() if k.startswith(("primary:", "clean_slice:")))
    statuses = [r["status"] for k, r in rows.items() if k.startswith(("vs_v2:", "noninferiority:", "knowledge:"))]
    if confirm_rows is not None:
        prim_ok = prim_ok and all(r["pass"] for k, r in confirm_rows.items() if k.startswith("primary:"))
        statuses += [r["status"] for k, r in confirm_rows.items() if k.startswith(("vs_v2:", "noninferiority:"))]
    if not gate_ok or not prim_ok or "FAIL" in statuses:
        return "STOP_FAILED" if mode == "diagnostic" else "FAILED"
    if "INSUFFICIENT" in statuses:
        if mode == "diagnostic":
            # Small knowledge relations may stay INSUFFICIENT at one seed; protected skills may not.
            skill = [r["status"] for k, r in rows.items() if k.startswith(("vs_v2:", "noninferiority:"))]
            return "STOP_INSUFFICIENT" if "INSUFFICIENT" in skill else "ADVANCE"
        return "INSUFFICIENT_EVIDENCE"
    return "ADVANCE" if mode == "diagnostic" else "QUALIFIED_FOR_REVIEW"


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _load_scores(path: Path, proto: dict, bundle_key: str, expect_model: tuple[str, str], inputs: dict) -> dict:
    d = _score_file(path)
    if d.get("bundle_sha256") != proto[bundle_key] or d.get("scorer") != proto["scorer"]:
        raise ValueError(f"{path}: bundle or scorer differs from the protocol")
    key, value = expect_model
    model = d["model"]
    got = {
        "run_id": model.get("run_id"),
        "freeze_id": model.get("freeze_id"),
        "checkpoint": f"{model.get('run_id')}/{model.get('checkpoint')}",
    }[key]
    if got != value:
        raise ValueError(f"{path}: model {got!r} is not the registered {value!r}")
    if key != "freeze_id" and model.get("updates") != proto["training"]["updates"]:
        raise ValueError(f"{path}: model is not a final ({proto['training']['updates']}-update) checkpoint")
    inputs[str(path)] = _sha(path)
    return d


def _identity(model: dict) -> tuple:
    return tuple(model.get(k) for k in IDENTITY)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--protocol", type=Path, required=True)
    ap.add_argument("--mode", choices=("diagnostic", "final"), required=True)
    ap.add_argument("--eval-root", type=Path, required=True, help="directory holding the registered layout")
    ap.add_argument("--smoke", type=Path, required=True)
    ap.add_argument("--sensitivity", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    proto = _read(a.protocol)
    _finite_tree(proto)
    if proto.get("schema") != "j1.sft-v21-protocol.v1":
        raise SystemExit("not an SFT v2.1 protocol")
    inputs: dict[str, str] = {str(a.protocol): _sha(a.protocol)}
    lay = proto["evaluation_layout"]
    seeds = [proto["diagnostic_seed"]] if a.mode == "diagnostic" else proto["seeds"]
    root = a.eval_root
    ex = proto["exposed_items"]
    ex_path = a.protocol.resolve().parents[2] / ex["file"]
    if _sha(ex_path) != ex["sha256"]:
        raise SystemExit("exposed-items file differs from the protocol")
    exposed = frozenset(_read(ex_path)["item_ids"])
    inputs[str(ex_path)] = ex["sha256"]
    raw: dict[str, dict[int, dict]] = {}

    def bundle_scores(bundle: str, sub: str) -> tuple[dict, dict, dict | None, dict]:
        key = "eval_bundle_sha256" if bundle == "historical" else "confirmation_bundle_sha256"
        cand, v12, v2 = {}, {}, {}
        raw[bundle] = {}
        for s in proto["seeds"]:
            v12[s] = _chat(
                _load_scores(
                    root / lay[bundle]["v12"].format(seed=s) / sub,
                    proto,
                    key,
                    ("checkpoint", proto["baseline"]["checkpoints"][str(s)]),
                    inputs,
                )
            )
        for s in seeds:
            d = _load_scores(
                root / lay[bundle]["v21"].format(seed=s) / sub, proto, key, ("run_id", proto["runs"][str(s)]), inputs
            )
            raw[bundle][s] = d
            cand[s] = _chat(d)
            v2[s] = _chat(
                _load_scores(
                    root / lay[bundle]["v2"].format(seed=s) / sub,
                    proto,
                    key,
                    ("checkpoint", proto["v2"]["checkpoints"][str(s)]),
                    inputs,
                )
            )
        return cand, {s: v12[s] for s in seeds}, v2, v12

    cand, v12, v2, v12_all = bundle_scores("historical", "cand_scores.json")
    sens: dict[str, list[str]] = {}
    for i, v in _read(a.sensitivity)["items"].items():
        sens.setdefault(f"{v['family'].split('.')[1]}:{v['class']}", []).append(i)
    inputs[str(a.sensitivity)] = _sha(a.sensitivity)
    rows = evaluate(proto, cand, v12, v2, sens, v12_all, knowledge=True, exposed=exposed)
    for k in rows:
        if k.startswith("knowledge:"):
            rows[k]["ids"] = sorted(i for i, r in cand[seeds[0]].items() if _member(r, k.split(":", 1)[1]))
    base_few = _fewshot(
        _load_scores(
            root / lay["historical"]["base_fewshot"],
            proto,
            "eval_bundle_sha256",
            ("freeze_id", proto["base"]["freeze_id"]),
            inputs,
        )
    )
    cand_few = {}
    for s in seeds:
        d = _load_scores(
            root / lay["historical"]["v21"].format(seed=s) / "cand_scores_fewshot_facts.json",
            proto,
            "eval_bundle_sha256",
            ("run_id", proto["runs"][str(s)]),
            inputs,
        )
        if _identity(d["model"]) != _identity(raw["historical"][s]["model"]):
            raise ValueError(f"seed {s}: few-shot scores come from a different checkpoint")
        cand_few[s] = _fewshot(d)
    _fewshot_retention(proto, base_few, cand_few, rows)
    for k in rows:
        rows[k].pop("ids", None)
    evidence = {}
    for s in seeds:
        d = root / lay["historical"]["v21"].format(seed=s)
        ident = _identity(raw["historical"][s]["model"])
        evidence[s] = {
            "scores": raw["historical"][s],  # the hash-recorded parse, not a second read
            "retention": _read(d / "cand_ret_full.json"),
            "check": _read(d / "pilot_recovery_check.json"),
            "dev": _read(d / "pilot_dev_history.json"),
        }
        e = evidence[s]
        if (
            _identity(e["retention"]["model"]) != ident
            or _identity(e["check"]["checkpoint"]) != ident
            or e["dev"].get("checkpoint") != ident[1]
        ):
            raise ValueError(f"seed {s}: retention, checkpoint check or dev history describe a different checkpoint")
        for f in ("cand_ret_full.json", "pilot_recovery_check.json", "pilot_dev_history.json"):
            inputs[str(d / f)] = _sha(d / f)
    smoke = _read(a.smoke)
    inputs[str(a.smoke)] = _sha(a.smoke)
    gates = _gates(proto, cand, evidence, smoke)
    if "prior" in proto:  # reported only: this stage against its predecessor's candidate, same seeds
        pr = proto["prior"]
        prior_seeds = [s for s in seeds if str(s) in pr["runs"]]
        prior = {
            s: _chat(
                _load_scores(
                    root / pr["layout"].format(seed=s) / "cand_scores.json",
                    proto,
                    "eval_bundle_sha256",
                    ("checkpoint", pr["checkpoints"][str(s)]),
                    inputs,
                )
            )
            for s in prior_seeds
        }
        if prior:
            rows.update(prior_rows(proto, {s: cand[s] for s in prior_seeds}, prior, v12_all, exposed))
    confirm_rows = None
    if a.mode == "final":
        c_cand, c_v12, c_v2, c_all = bundle_scores("confirmation", "cand_scores.json")
        for s in seeds:
            if _identity(raw["confirmation"][s]["model"]) != _identity(raw["historical"][s]["model"]):
                raise ValueError(f"seed {s}: confirmation scores come from a different checkpoint")
        conf = proto["confirmation"]
        ni = proto["noninferiority"]
        cproto = {
            **proto,
            "noninferiority": {**ni, "groups": {g: ni["groups"][g] for g in conf["noninferiority_groups"]}},
            "reported_groups": conf["reported_groups"],
        }
        confirm_rows = evaluate(cproto, c_cand, c_v12, c_v2, {}, c_all, knowledge=False, min_n=conf["min_n"])
    out = {
        "schema": "j1.sft-v21-decision.v1",
        "mode": a.mode,
        "seeds": seeds,
        "outcome": verdict(a.mode, rows, gates, confirm_rows),
        "gates": gates,
        "rows": rows,
        "confirmation_rows": confirm_rows,
        "inputs_sha256": inputs,
        "note": "a qualified research candidate is not a production release and is not promoted by this tool",
    }
    a.out.write_text(
        json.dumps(
            out, indent=1, sort_keys=True, allow_nan=False, default=lambda x: list(x) if isinstance(x, tuple) else x
        )
    )
    print(
        json.dumps({"outcome": out["outcome"], "gates": {k: v for k, v in gates.items() if k != "per_seed"}}, indent=1)
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
