"""Seed-variance report for the SFT replication (docs/posttrain/sft_replication_protocol_v1.json).

Inputs per seed: an evaluation directory with ``cand_scores.json``, ``comparison.json``,
``decision.json``, ``cand_ret_full.json`` and ``pilot_dev_history.json`` (seed 1: the preserved pilot
evaluation). Everything is recomputed from per-item scores where possible:

* per seed: every category and slice (strict test score with a Wilson 95% interval), protocol validity,
  abstention metrics, full-split bpb, final dev loss, the v3 decision;
* across seeds: min, max, mean, sample standard deviation (n = 3: descriptive only);
* paired: each seed against the base few-shot view (from its comparison) and each pair of seeds on the
  same items (exact McNemar; Holm over the seed pairs within a slice);
* the registered consistency check.

Seed spread and item-level uncertainty are reported separately; pooled items are not treated as
independent training replications. ``--label`` marks a scoring view (original or corrected rescoring);
different views are never mixed in one report.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.sft_decide import wilson_lower


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, mid - half), min(1.0, mid + half))


def mcnemar_p(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2**n
    return float(min(1.0, 2 * tail))


def holm(p: dict[str, float]) -> dict[str, float]:
    out, running = {}, 0.0
    for rank, k in enumerate(sorted(p, key=p.get)):
        running = max(running, min(1.0, (len(p) - rank) * p[k]))
        out[k] = running
    return out


def slice_of(family: str) -> list[str]:
    """Slices a test item belongs to (besides its category)."""
    out = [f"family:{family}"]
    head = family.split(".")[0]
    if family.startswith(("heldout.if", "heldout.ops")):
        out.append("if_heldout_families")
    elif family.startswith(("if.paraphrase", "if.compose_heldout")):
        pass  # evaluation-only paraphrases/compositions: their own categories, never in distribution
    elif head == "if":
        out.append("if_in_distribution")
    if family.startswith("heldout.json"):
        out.append("json_heldout")
    if family in ("facts.country_capital", "facts.us_state_capital"):
        out.append("facts_capitals")
    if family.startswith("arith."):
        out.append("math_arith")
    if family == "ext.gsm8k":
        out.append("math_gsm8k")
    if family in ("ext.mbpp", "ext.humaneval"):
        out.append("python_benchmarks")
    return out


def load_seed(d: Path) -> dict:
    scores = json.loads((d / "cand_scores.json").read_text())
    items = {r["id"]: r for r in scores["items"] if r["split"] == "test" and r["variant"] == "chat"}
    groups: dict[str, list[str]] = defaultdict(list)
    for i, r in items.items():
        groups[f"category:{r['category']}"].append(i)
        groups["all"].append(i)
        for s in slice_of(r["family"]):
            groups[s].append(i)
    metrics = {}
    for g, ids in sorted(groups.items()):
        k = sum(items[i]["strict"] for i in ids)
        metrics[g] = {"n": len(ids), "strict": k / len(ids), "wilson95": wilson(k, len(ids))}
    comp = json.loads((d / "comparison.json").read_text())["test"]
    dec = json.loads((d / "decision.json").read_text())
    ret = json.loads((d / "cand_ret_full.json").read_text())
    dev = json.loads((d / "pilot_dev_history.json").read_text())
    return {
        "dir": str(d),
        "model": scores["model"],
        "scorer": scores["scorer"],
        "items": items,
        "metrics": metrics,
        "protocol_valid": comp["protocol_valid"]["candidate"],
        "uncertainty_strict": comp["abstention"]["uncertainty_strict"],
        "false_abstention_on_facts": comp["abstention"]["false_abstention_on_facts"],
        "versus_base_fewshot": comp["versus_base_fewshot"],
        "significant_regressions_vs_fewshot": comp["significant_regressions_vs_fewshot"],
        "bpb_full": ret["bpb"],
        "final_dev_loss": dev["dev_history"][-1]["dev_loss"] if dev["dev_history"] else None,
        "decision": dec["decision"],
        "gates": {k: v["pass"] for k, v in dec["gates"].items()},
    }


def describe(values: list[float]) -> dict:
    return {
        "min": min(values),
        "max": max(values),
        "mean": statistics.fmean(values),
        "sd": statistics.stdev(values) if len(values) > 1 else None,
        "range": max(values) - min(values),
    }


def report(seeds: dict[str, dict], label: str, rep: dict) -> dict:
    names = sorted(seeds)
    common = set.intersection(*(set(s["metrics"]) for s in seeds.values()))
    scalar = ("protocol_valid", "uncertainty_strict", "false_abstention_on_facts", "bpb_full", "final_dev_loss")
    summary = {g: describe([seeds[n]["metrics"][g]["strict"] for n in names]) for g in sorted(common)}
    summary |= {
        k: describe([seeds[n][k] for n in names]) for k in scalar if all(seeds[n][k] is not None for n in names)
    }
    pairs: dict[str, dict] = {}
    for g in sorted(common):
        ids = sorted(i for i in seeds[names[0]]["items"] if g == "all" or _in_group(seeds[names[0]]["items"][i], g))
        raw = {}
        for a, b in itertools.combinations(names, 2):
            sa = [seeds[a]["items"][i]["strict"] for i in ids]
            sb = [seeds[b]["items"][i]["strict"] for i in ids]
            a_only = sum(x and not y for x, y in zip(sa, sb, strict=True))
            b_only = sum(y and not x for x, y in zip(sa, sb, strict=True))
            raw[f"{a}-{b}"] = {
                "n": len(ids),
                "a_only": a_only,
                "b_only": b_only,
                "mcnemar_p": mcnemar_p(a_only, b_only),
            }
        adj = holm({k: v["mcnemar_p"] for k, v in raw.items()})
        for k, v in raw.items():
            v["mcnemar_p_holm"] = adj[k]
        pairs[g] = raw
    p1 = "if_in_distribution"
    check = rep["variance_report"]["consistency_check"]
    decisions = {n: seeds[n]["decision"] for n in names}
    findings = {
        "same_decision": len(set(decisions.values())) == 1,
        "p1_pairwise_holm_min_p": min(v["mcnemar_p_holm"] for v in pairs[p1].values()),
        "all_pass_G-RET": all(seeds[n]["gates"].get("G-RET") for n in names),
    }
    findings["inconsistent"] = (
        not findings["same_decision"] or findings["p1_pairwise_holm_min_p"] < 0.05 or not findings["all_pass_G-RET"]
    )
    return {
        "schema": "j1.sft-seed-report.v1",
        "scoring_view": label,
        "scorers": sorted({seeds[n]["scorer"] for n in names}),
        "seeds": {
            n: {k: v for k, v in seeds[n].items() if k != "items"}
            | {"p1_wilson_lower": wilson_lower(seeds[n]["metrics"][p1]["strict"], seeds[n]["metrics"][p1]["n"])}
            for n in names
        },
        "across_seeds": summary,
        "pairwise_seeds": pairs,
        "consistency_check": {"rule": check, **findings},
        "note": "n = 3 seeds: spread is descriptive; item intervals are per seed; no pooling across seeds",
    }


def _in_group(r: dict, g: str) -> bool:
    if g.startswith("category:"):
        return r["category"] == g.split(":", 1)[1]
    return g in slice_of(r["family"])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", nargs=2, action="append", metavar=("NAME", "EVAL_DIR"), required=True)
    ap.add_argument("--label", required=True, choices=("original_scoring", "corrected_rescoring"))
    root = Path(__file__).resolve().parents[1]
    ap.add_argument("--replication", type=Path, default=root / "docs/posttrain/sft_replication_protocol_v1.json")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    seeds = {n: load_seed(Path(d)) for n, d in a.seed}
    if len({s["scorer"] for s in seeds.values()}) != 1:
        raise SystemExit("seed reports come from different scorers; never mix scoring views")
    res = report(seeds, a.label, json.loads(a.replication.read_text()))
    a.out.write_text(json.dumps(res, indent=1, sort_keys=True))
    keys = ("category:instruction_following", "if_in_distribution", "if_heldout_families", "facts_capitals")
    print(json.dumps({k: res["across_seeds"].get(k) for k in (*keys, "bpb_full")}, indent=1))
    print(json.dumps(res["consistency_check"], indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
