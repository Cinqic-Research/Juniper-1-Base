"""Apply the SFT v2 decision rule (docs/posttrain/sft_v2_protocol_v1.json) to recorded results.

Every threshold comes from the protocol; every value from per-item score files. Inputs, all on the
same sealed evaluation bundle and scored by the same scorer:

* ``--v2 SEED DIR``: a v2 run's evaluation directory (``cand_scores.json``, ``cand_scores_fewshot_facts.json``,
  ``cand_ret_full.json``, ``pilot_recovery_check.json``, ``pilot_dev_history.json``);
* ``--baseline SEED DIR``: the replicated v1.2-recipe candidate with the same seed number, evaluated
  on this bundle (``cand_scores.json``);
* ``--base-scores``: the frozen base (chat and few-shot) on this bundle;
* ``--recovery-ab``: the stage's smoke recovery A/B result.

Aggregation is fixed in advance: a metric's *seed mean* is the mean over the registered seeds; the
paired item bootstrap resamples items (the same draw for every seed) and recomputes the seed-mean
difference; per-seed exact McNemar compares each v2 seed with the baseline of the same seed number.
The outcome is ``QUALIFIED_FOR_REVIEW``, ``FAILED`` or ``INSUFFICIENT_EVIDENCE``; nothing is promoted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.sft_decide import early_stop_triggered
from tools.sft_seed_report import holm, mcnemar_p, slice_of


def _canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def _read(path: Path) -> dict:
    def reject(value):
        raise ValueError(f"non-finite JSON constant {value}")

    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    value = json.loads(path.read_text(), parse_constant=reject, object_pairs_hook=unique)
    _finite_tree(value)
    if not isinstance(value, dict):
        raise ValueError("evidence JSON must contain an object")
    return value


def _finite_tree(value) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("non-finite numeric evidence")
    if isinstance(value, dict):
        for child in value.values():
            _finite_tree(child)
    elif isinstance(value, list):
        for child in value:
            _finite_tree(child)


def _validate_rows(rows: list[dict]) -> None:
    if not isinstance(rows, list) or not rows:
        raise ValueError("score items must be a nonempty list")
    ids = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("malformed score item")
        for key in ("id", "split", "variant", "family", "category"):
            if not isinstance(row.get(key), str) or not row[key]:
                raise ValueError(f"missing or invalid item {key}")
        if row["id"] in ids:
            raise ValueError("duplicated score item ID")
        ids.add(row["id"])
        for key in ("strict", "protocol_valid", "abstained"):
            if type(row.get(key)) is not bool:
                raise ValueError(f"item {key} must be boolean")
        if "unscored" in row and type(row["unscored"]) is not bool:
            raise ValueError("item unscored must be boolean")


def _metadata_sha(rows: list[dict]) -> str:
    keys = ("id", "split", "variant", "family", "category")
    return hashlib.sha256(_canonical(sorted(tuple(r[k] for k in keys) for r in rows))).hexdigest()


def _score_file(path: Path) -> dict:
    d = _read(path)
    if d.get("schema") != "j1.sft-scores.v1":
        raise ValueError("unknown score schema")
    if (
        d.get("report_sha256")
        != hashlib.sha256(_canonical({k: v for k, v in d.items() if k != "report_sha256"})).hexdigest()
    ):
        raise ValueError("score report self-hash mismatch")
    _validate_rows(d["items"])
    return d


def _items(path: Path) -> dict[str, dict]:
    d = _score_file(path)
    return {r["id"]: r for r in d["items"] if r["split"] == "test" and r["variant"] == "chat" and not r.get("unscored")}


class EvidenceBinding:
    """Explicit audited provenance for historical reports that omit the score-engine revision.

    A caller pins the binding's file hash. The binding is an audit assertion, not proof that the
    original scorer recorded identities it omitted. Its item metadata digest must come from the
    sealed bundle's expected selection; its model identities from verified checkpoint manifests.
    """

    def __init__(self, path: Path, sha256: str, protocol_path: Path, proto: dict):
        if hashlib.sha256(path.read_bytes()).hexdigest() != sha256:
            raise ValueError("evidence binding differs from the pinned file hash")
        self.d = _read(path)
        self.proto = proto
        if (
            self.d.get("schema") != "j1.sft-v2-audited-bindings.v1"
            or self.d.get("protocol_sha256") != hashlib.sha256(protocol_path.read_bytes()).hexdigest()
            or self.d.get("eval_bundle_sha256") != proto["eval_bundle_sha256"]
            or self.d.get("data_code_revision") != proto["data_code_revision"]
            or self.d.get("score_engine") not in proto["scorer"].split()
            or self.d.get("scorer") not in proto["scorer"].split()
            or not self.d.get("audit_note")
        ):
            raise ValueError("audited binding does not describe the registered protocol/scorer")
        self.used: set[str] = set()

    def read(self, role: str, path: Path, *, scores: bool = False, model: dict | None = None) -> dict:
        rec = self.d["files"].get(role)
        if rec is None or rec.get("sha256") != hashlib.sha256(path.read_bytes()).hexdigest():
            raise ValueError(f"unbound or changed evidence: {role}")
        self.used.add(role)
        d = _score_file(path) if scores else _read(path)
        if scores and (
            d.get("bundle_sha256") != self.proto["eval_bundle_sha256"]
            or d.get("scorer") != self.d["scorer"]
            or _metadata_sha(d["items"]) != rec.get("item_metadata_sha256")
            or d.get("model") != rec.get("model")
        ):
            raise ValueError(f"score identities or expected bundle selection differ: {role}")
        if model is not None:
            actual = d.get("model") if "model" in d else d.get("checkpoint")
            if not isinstance(actual, dict) or {k: v for k, v in actual.items() if k != "kind"} != {
                k: v for k, v in model.items() if k != "kind"
            }:
                raise ValueError(f"candidate identity differs: {role}")
        return d


def _member(r: dict, group: str) -> bool:
    if group.startswith("category:"):
        return r["category"] == group.split(":", 1)[1]
    if group.startswith("family_prefix:"):
        return r["family"].startswith(group.split(":", 1)[1])
    return group in slice_of(r["family"])


def paired_seed_mean(v2: dict[int, dict], base: dict[int, dict], ids: list[str], resamples: int, seed: int) -> dict:
    seeds = sorted(v2)
    diffs = {s: [int(v2[s][i]["strict"]) - int(base[s][i]["strict"]) for i in ids] for s in seeds}
    n = len(ids)

    def mean_diff(sample: list[int]) -> float:
        return sum(sum(diffs[s][j] for j in sample) / n for s in seeds) / len(seeds)

    rng = random.Random(seed)
    boots = sorted(mean_diff([rng.randrange(n) for _ in range(n)]) for _ in range(resamples))
    per_seed = {}
    for s in seeds:
        b_only = sum(1 for i in ids if base[s][i]["strict"] and not v2[s][i]["strict"])
        c_only = sum(1 for i in ids if v2[s][i]["strict"] and not base[s][i]["strict"])
        per_seed[s] = {
            "v2": sum(v2[s][i]["strict"] for i in ids) / n,
            "baseline": sum(base[s][i]["strict"] for i in ids) / n,
            "mcnemar_p": mcnemar_p(b_only, c_only),
            "improved": c_only > b_only,
        }
    return {
        "n": n,
        "difference": mean_diff(list(range(n))),
        "bootstrap_ci95": (boots[int(0.025 * resamples)], boots[int(0.975 * resamples) - 1]),
        "per_seed": per_seed,
    }


def resolve_margin(spec, rule: dict, baseline: dict[int, dict], ids: list[str]) -> float:
    """A numeric margin, or "auto": max(floor, k * sample SD of the baseline seeds' accuracy on ``ids``)
    (registered as a formula before the baseline was measured on this bundle)."""
    if spec != "auto":
        return float(spec)
    accs = [sum(baseline[s][i]["strict"] for i in ids) / max(1, len(ids)) for s in sorted(baseline)]
    return max(float(rule["floor"]), float(rule["k"]) * statistics.stdev(accs))


def decide(proto: dict, v2: dict[int, dict], baseline: dict[int, dict], base: dict, extra: dict) -> dict:
    _finite_tree(proto)
    _finite_tree(extra)
    if type(extra["recovery_ab"].get("bitwise")) is not bool:
        raise ValueError("recovery bitwise evidence must be boolean")
    seeds = proto["seeds"]
    if sorted(v2) != sorted(seeds) or sorted(baseline) != sorted(seeds):
        raise SystemExit(f"need v2 and baseline results for seeds {seeds}")
    ref = v2[seeds[0]]
    if any(set(x) != set(ref) for x in [*v2.values(), *baseline.values()]):
        raise SystemExit("score files do not cover the same test items")
    for items in [*v2.values(), *baseline.values()]:
        _validate_rows([{**r, "id": i} for i, r in items.items()])
        if any(any(r[k] != ref[i][k] for k in ("category", "family", "split", "variant")) for i, r in items.items()):
            raise ValueError("paired item metadata differs")
    boot = proto["statistics"]["bootstrap"]
    rows: dict[str, dict] = {}

    def ids_for(group: str) -> list[str]:
        ids = sorted(i for i, r in ref.items() if _member(r, group))
        if not ids:
            raise ValueError(f"required evaluation group is empty: {group}")
        return ids

    prim = proto["primary"]
    pv = {}
    for g in prim["endpoints"]:
        res = paired_seed_mean(v2, baseline, ids_for(g), boot["resamples"], boot["seed"])
        res["max_seed_p"] = max(x["mcnemar_p"] for x in res["per_seed"].values())
        pv[g] = res["max_seed_p"]
        rows[f"primary:{g}"] = res
    adj = holm(pv)
    for g in prim["endpoints"]:
        r = rows[f"primary:{g}"]
        r["max_seed_p_holm"] = adj[g]
        r["pass"] = (
            r["difference"] >= prim["margin"]
            and r["bootstrap_ci95"][0] > 0
            and all(x["improved"] for x in r["per_seed"].values())
            and adj[g] < prim["alpha"]
        )
        r["insufficient"] = not r["pass"] and r["bootstrap_ci95"][1] >= prim["margin"]
    ni = proto["noninferiority"]
    for g, spec in ni["groups"].items():
        r = paired_seed_mean(v2, baseline, ids_for(g), boot["resamples"], boot["seed"])
        margin = resolve_margin(spec, ni.get("auto", {}), baseline, ids_for(g))
        lo, hi = r["bootstrap_ci95"]
        r["margin"] = margin
        r["status"] = (
            "PASS" if lo > -margin else ("FAIL" if hi < -margin or r["difference"] < -margin else "INSUFFICIENT")
        )
        rows[f"noninferiority:{g}"] = r
    kn = proto["knowledge"]
    bf = {
        r["id"].removesuffix("#fewshot"): r
        for r in base["items"]
        if r["split"] == "test" and r["variant"] == "base_fewshot"
    }
    few_rows = [r for r in base["items"] if r["split"] == "test" and r["variant"] == "base_fewshot"]
    if len(bf) != len(few_rows):
        raise ValueError("duplicate base few-shot IDs")
    kp = {}
    for g, spec in kn["relations"].items():
        ids = ids_for(g)
        if any(i not in bf or any(i not in extra[s]["fewshot"] for s in seeds) for i in ids):
            raise ValueError("missing paired knowledge items")
        if any(type(bf[i].get("strict")) is not bool for i in ids) or any(
            type(extra[s]["fewshot"][i].get("strict")) is not bool for s in seeds for i in ids
        ):
            raise ValueError("few-shot strict scores must be boolean")
        margin = resolve_margin(spec, kn.get("auto", {}), baseline, ids)
        r = paired_seed_mean(v2, baseline, ids, boot["resamples"], boot["seed"])
        few = sum(bf[i]["strict"] for i in ids) / max(1, len(ids))
        worse = {}
        for s in seeds:
            # Retention: the candidate's own few-shot answers against the base's on the same items (the
            # knowledge diagnostic showed chat accuracy measures elicitation, not retained knowledge).
            fs = extra[s]["fewshot"]
            b_only = sum(1 for i in ids if bf[i]["strict"] and not fs[i]["strict"])
            c_only = sum(1 for i in ids if fs[i]["strict"] and not bf[i]["strict"])
            worse[s] = {
                "candidate_fewshot": sum(fs[i]["strict"] for i in ids) / max(1, len(ids)),
                "p": mcnemar_p(b_only, c_only),
                "worse": b_only > c_only,
            }
        kp[g] = max(w["p"] for w in worse.values() if w["worse"]) if any(w["worse"] for w in worse.values()) else 1.0
        r |= {"base_fewshot": few, "candidate_fewshot_vs_base_fewshot": worse, "chat_margin_vs_baseline": margin}
        rows[f"knowledge:{g}"] = r
    kadj = holm(kp) if kp else {}
    for g in kn["relations"]:
        r = rows[f"knowledge:{g}"]
        r["worse_than_base_fewshot_p_holm"] = kadj[g]
        r["retention_pass"] = not (
            all(w["worse"] for w in r["candidate_fewshot_vs_base_fewshot"].values()) and kadj[g] < kn["alpha"]
        )
        r["chat_noninferior"] = r["difference"] >= -r["chat_margin_vs_baseline"]
        r["pass"] = r["retention_pass"] and r["chat_noninferior"]
    # Reported, not gated (predeclared): extra groups and the rebuild review's sensitivity classes.
    for g in proto.get("reported_groups", []):
        ids = sorted(i for i, r in ref.items() if _member(r, g))  # reported only: an empty group is skipped
        if ids:
            rows[f"reported:{g}"] = paired_seed_mean(v2, baseline, ids, boot["resamples"], boot["seed"])
    for cls, ids in sorted(extra.get("sensitivity", {}).items()):
        ids = sorted(i for i in ids if i in ref)
        if ids:
            rows[f"reported:sensitivity:{cls}"] = paired_seed_mean(v2, baseline, ids, boot["resamples"], boot["seed"])
    gates = proto["gates"]
    per_seed = {}
    for s in seeds:
        e = extra[s]
        items = v2[s]
        fa_ids = [i for i, r in items.items() if r["category"] == "factuality"]
        unc_ids = [i for i, r in items.items() if r["category"] == "uncertainty"]
        # The hash-checked parse from main(); re-reading the path could see different bytes.
        allv = e["score_items"] if "score_items" in e else _read(Path(e["scores_path"]))["items"]
        _validate_rows(allv)
        scored = {r["id"]: r for r in allv if r["split"] == "test" and r["variant"] == "chat" and not r.get("unscored")}
        if set(scored) != set(items) or any(
            any(scored[i][k] != r[k] for k in ("category", "family", "strict", "protocol_valid", "abstained"))
            for i, r in items.items()
        ):
            raise ValueError("score file differs from the paired in-memory scores")
        if (
            not e["retention"]["bpb"] >= 0
            or type(e["check"]["verify"][0]) is not bool
            or type(e["check"]["exact"]) is not bool
        ):
            raise ValueError("invalid retention or checkpoint evidence")
        history = e["dev"]["dev_history"]
        if (
            type(e["dev"]["updates"]) is not int
            or not history
            or any(type(r["update"]) is not int or r["dev_loss"] < 0 for r in history)
            or [r["update"] for r in history] != sorted({r["update"] for r in history})
            or history[-1]["update"] != e["dev"]["updates"]
        ):
            raise ValueError("invalid development history")
        valid = [r for r in allv if r["split"] == "test" and r["variant"] == "chat"]
        per_seed[s] = {
            "protocol_valid": sum(r["protocol_valid"] for r in valid) / len(valid),
            "uncertainty_strict": sum(items[i]["strict"] for i in unc_ids) / max(1, len(unc_ids)),
            "false_abstention_on_facts": sum(items[i]["abstained"] for i in fa_ids) / max(1, len(fa_ids)),
            "bpb_full": e["retention"]["bpb"],
            "retention_mode_ok": e["retention"]["mode"] == "full" and e["retention"]["precision"] == "fp32",
            "checkpoint_verified": bool(e["check"]["verify"][0]) and bool(e["check"]["exact"]),
            "updates": e["dev"]["updates"],
            "early_stop": early_stop_triggered(e["dev"]["dev_history"]),
        }
    g_rows = {
        "G-PROTO": all(p["protocol_valid"] >= gates["protocol_valid_min"] for p in per_seed.values()),
        "G-RET": all(
            p["retention_mode_ok"] and p["bpb_full"] <= gates["full_split_bpb_max"] for p in per_seed.values()
        ),
        "G-UNC": all(
            p["uncertainty_strict"] >= gates["uncertainty_strict_min"]
            and p["false_abstention_on_facts"] <= gates["false_abstention_max"]
            for p in per_seed.values()
        ),
        "G-REC": bool(extra["recovery_ab"].get("bitwise")) and all(p["checkpoint_verified"] for p in per_seed.values()),
        "EARLY-STOP": all(
            p["updates"] == proto["training"]["updates"] and not p["early_stop"] for p in per_seed.values()
        ),
    }
    primary_pass = all(rows[f"primary:{g}"]["pass"] for g in prim["endpoints"])
    ni_status = [rows[f"noninferiority:{g}"]["status"] for g in ni["groups"]]
    knowledge_pass = all(rows[f"knowledge:{g}"]["pass"] for g in kn["relations"])
    if not all(g_rows.values()) or "FAIL" in ni_status or not knowledge_pass:
        outcome = "FAILED"
    elif primary_pass and all(s == "PASS" for s in ni_status):
        outcome = "QUALIFIED_FOR_REVIEW"
    else:
        outcome = "INSUFFICIENT_EVIDENCE"
    return {
        "schema": "j1.sft-v2-decision.v1",
        "protocol": proto["schema"],
        "outcome": outcome,
        "candidate": proto["candidate"],
        "gates": g_rows,
        "per_seed": per_seed,
        "rows": rows,
        "note": "a qualified research candidate is not a production release; the engineer does not promote it",
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--protocol", type=Path, required=True)
    ap.add_argument("--v2", nargs=2, action="append", metavar=("SEED", "DIR"), required=True)
    ap.add_argument("--baseline", nargs=2, action="append", metavar=("SEED", "DIR"), required=True)
    ap.add_argument("--base-scores", type=Path, required=True)
    ap.add_argument("--recovery-ab", type=Path, required=True)
    ap.add_argument("--sensitivity", type=Path, help="tools/sft_v2_sensitivity.py output (reported breakdowns)")
    ap.add_argument("--evidence-bindings", type=Path, required=True, help="additive independent provenance audit")
    ap.add_argument("--evidence-bindings-sha256", required=True, help="pinned file hash of that audit")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    proto = _read(a.protocol)
    audit = EvidenceBinding(a.evidence_bindings, a.evidence_bindings_sha256, a.protocol, proto)
    for args in (a.v2, a.baseline):
        if len({int(s) for s, _ in args}) != len(args):
            raise ValueError("duplicate seed arguments")
    v2, baseline, models, parsed = {}, {}, {}, {}
    for label, args, target in (("v2", a.v2, v2), ("baseline", a.baseline, baseline)):
        for s, d in args:
            seed = int(s)
            scores = audit.read(f"{label}:{seed}:scores", Path(d) / "cand_scores.json", scores=True)
            model = scores["model"]
            expected = proto["runs"][str(seed)] if label == "v2" else proto["baseline"]["checkpoints"][str(seed)]
            observed = model["run_id"] if label == "v2" else f"{model['run_id']}/{model['checkpoint']}"
            if observed != expected or model["updates"] != proto["training"]["updates"]:
                raise ValueError("score model differs from the registered candidate")
            target[seed] = {
                r["id"]: r
                for r in scores["items"]
                if r["split"] == "test" and r["variant"] == "chat" and not r.get("unscored")
            }
            if label == "v2":
                models[seed] = model
                parsed[seed] = scores["items"]
    extra: dict = {"recovery_ab": audit.read("recovery_ab", a.recovery_ab)}
    if a.sensitivity:
        sens: dict[str, list[str]] = {}
        for i, v in audit.read("sensitivity", a.sensitivity)["items"].items():
            sens.setdefault(f"{v['family'].split('.')[1]}:{v['class']}", []).append(i)
        extra["sensitivity"] = sens
    for s, d in a.v2:
        p = Path(d)
        fs = audit.read(f"v2:{s}:fewshot", p / "cand_scores_fewshot_facts.json", scores=True, model=models[int(s)])
        extra[int(s)] = {
            "scores_path": str(p / "cand_scores.json"),
            "score_items": parsed[int(s)],
            "retention": audit.read(f"v2:{s}:retention", p / "cand_ret_full.json", model=models[int(s)]),
            "check": audit.read(f"v2:{s}:check", p / "pilot_recovery_check.json", model=models[int(s)]),
            "dev": audit.read(f"v2:{s}:dev", p / "pilot_dev_history.json"),
            "fewshot": {
                r["id"].removesuffix("#fewshot"): r
                for r in fs["items"]
                if r["split"] == "test" and r["variant"] == "base_fewshot"
            },
        }
        if extra[int(s)]["dev"].get("checkpoint") != models[int(s)]["checkpoint"]:
            raise ValueError("development history describes a different checkpoint")
    base = audit.read("base:scores", a.base_scores, scores=True)
    if base["model"].get("freeze_id") != proto["base"]["freeze_id"]:
        raise ValueError("base score identity differs from the registered freeze")
    if set(audit.d["files"]) != audit.used:
        raise ValueError("unused bindings: invocation does not match the audited evidence set")
    res = decide(proto, v2, baseline, base, extra)
    res["evidence_binding_sha256"] = a.evidence_bindings_sha256
    res["evidence_qualification"] = audit.d["audit_note"]
    a.out.write_text(
        json.dumps(
            res, indent=1, sort_keys=True, allow_nan=False, default=lambda x: list(x) if isinstance(x, tuple) else x
        )
    )
    print(json.dumps({"outcome": res["outcome"], "gates": res["gates"]}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
