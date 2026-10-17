"""Distillation protocol v2 decision (Juniper-1-Data docs/distillation/gptoss_teacher_protocol_v2.json).

Per teacher arm (A1 answer-only, A2 concise rationale, A3 verified equations) against the matched no-teacher
control C0, all on the v1.3 bundle's test split:

* primary: GSM8K test (500) strict accuracy, exact McNemar per contrast, Holm over the three contrasts;
* claim: Holm-adjusted p < 0.05 AND gain >= 0.03 AND retention passes (full-split bpb <= 0.943872 and no
  Holm-significant regression versus C0 on any v1.3 test category);
* reported: generated arithmetic, and GSM8K answers whose working contains a false equation (re-evaluated
  with the Data verifier).

Outcome per arm: ``IMPROVEMENT`` or ``NO_IMPROVEMENT_DETECTED`` (which is not evidence of no effect).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.sft_seed_report import holm, mcnemar_p

BPB_MAX = 0.943872


def _items(d: Path) -> dict[str, dict]:
    rows = json.loads((d / "cand_scores.json").read_text())["items"]
    return {r["id"]: r for r in rows if r["split"] == "test" and r["variant"] == "chat" and not r.get("unscored")}


def compare(c0: dict[str, dict], arm: dict[str, dict], ids: list[str]) -> dict:
    b = sum(1 for i in ids if c0[i]["strict"] and not arm[i]["strict"])
    c = sum(1 for i in ids if arm[i]["strict"] and not c0[i]["strict"])
    n = len(ids)
    return {"n": n, "c0": sum(c0[i]["strict"] for i in ids) / n, "arm": sum(arm[i]["strict"] for i in ids) / n,
            "gain": (c - b) / n, "c0_only": b, "arm_only": c, "mcnemar_p": mcnemar_p(b, c)}  # fmt: skip


def false_equations(rows: dict[str, dict], ids: list[str], data_src: Path) -> dict:
    sys.path.insert(0, str(data_src))
    from j1data.teacherverify import check_rationale

    wrong = sum(1 for i in ids if check_rationale(rows[i]["output"]).wrong)
    return {"answers": len(ids), "with_false_equation": wrong}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", nargs=2, action="append", metavar=("NAME", "EVAL_DIR"), required=True)
    ap.add_argument("--data-src", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    dirs = {n: Path(d) for n, d in a.arm}
    if set(dirs) != {"C0", "A1", "A2", "A3"}:
        raise SystemExit("need C0, A1, A2 and A3")
    items = {n: _items(d) for n, d in dirs.items()}
    ref = items["C0"]
    if any(set(x) != set(ref) for x in items.values()):
        raise SystemExit("arms were scored on different items")
    gsm = sorted(i for i, r in ref.items() if r["family"] == "ext.gsm8k")
    arith = sorted(i for i, r in ref.items() if r["family"].startswith("arith."))
    cats = defaultdict(list)
    for i, r in ref.items():
        cats[r["category"]].append(i)
    primary = {n: compare(ref, items[n], gsm) for n in ("A1", "A2", "A3")}
    adj = holm({n: v["mcnemar_p"] for n, v in primary.items()})
    out = {"schema": "j1.distill-decision.v1", "arms": {}}
    for n in ("A1", "A2", "A3"):
        ret = json.loads((dirs[n] / "cand_ret_full.json").read_text())
        regress = {}
        for cat, ids in sorted(cats.items()):
            r = compare(ref, items[n], sorted(ids))
            regress[cat] = r | {"worse": r["c0_only"] > r["arm_only"]}
        radj = holm({k: v["mcnemar_p"] for k, v in regress.items()})
        sig = sorted(k for k, v in regress.items() if v["worse"] and radj[k] < 0.05)
        retention = ret["mode"] == "full" and ret["precision"] == "fp32" and ret["bpb"] <= BPB_MAX and not sig
        p = primary[n] | {"mcnemar_p_holm": adj[n]}
        claim = p["mcnemar_p_holm"] < 0.05 and p["gain"] >= 0.03 and retention
        out["arms"][n] = {
            "primary_gsm8k": p,
            "arithmetic": compare(ref, items[n], arith),
            "retention": {"bpb": ret["bpb"], "significant_regressions_vs_c0": sig, "pass": retention},
            "false_equations": false_equations(items[n], [i for i in gsm if items[n][i]["strict"]], a.data_src),
            "outcome": "IMPROVEMENT" if claim else "NO_IMPROVEMENT_DETECTED",
        }
    out["c0"] = {"gsm8k": sum(ref[i]["strict"] for i in gsm) / len(gsm),
                 "bpb": json.loads((dirs["C0"] / "cand_ret_full.json").read_text())["bpb"]}  # fmt: skip
    out["note"] = "one seed per arm: a positive result needs a replication seed before any adoption"
    a.out.write_text(json.dumps(out, indent=1, sort_keys=True))
    print(json.dumps({n: v["outcome"] for n, v in out["arms"].items()}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
