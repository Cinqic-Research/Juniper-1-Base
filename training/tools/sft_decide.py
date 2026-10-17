"""Apply the registered SFT pilot decision rule (protocol v3) to recorded results. No judgment here:
every threshold is read from the protocol JSON and every value from a result file.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from pathlib import Path


def wilson_lower(p: float, n: int, z: float = 1.959964) -> float:
    if n == 0:
        return 0.0
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    return max(0.0, mid - z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den)


def early_stop_triggered(dev_history: list[dict], rises: int = 3) -> bool:
    """Registered rule: dev loss rises at three consecutive dev evaluations."""
    losses = [d["dev_loss"] for d in sorted(dev_history, key=lambda d: d["update"])]
    run = 0
    for prev, cur in itertools.pairwise(losses):
        run = run + 1 if cur > prev else 0
        if run >= rises:
            return True
    return False


def decide(
    proto: dict, comparison: dict, retention: dict, recovery_ab: dict, pilot_check: dict, pilot_manifest: dict
) -> dict:
    names = {
        "retention": retention.get("model", {}).get("checkpoint"),
        "pilot_check": pilot_check.get("checkpoint", {}).get("checkpoint"),
        "dev_history": pilot_manifest.get("checkpoint"),
        "comparison": comparison.get("candidate_model", {}).get("checkpoint"),
    }
    if len(set(names.values())) != 1 or None in names.values():
        raise SystemExit(f"result files describe different candidates: {names}")
    t = comparison["test"]
    if_chat = t["versus_base_chat"]["instruction_following_in_distribution"]
    if_few = t["versus_base_fewshot"]["instruction_following_in_distribution"]
    p = proto["primary"]
    g = proto["gates"]
    cand = if_chat["candidate"]
    lb = wilson_lower(cand, if_chat["n"])
    rows = {
        "P1": {
            "candidate": cand,
            "wilson_lower": lb,
            "versus_fewshot_difference": if_few["difference"],
            "versus_fewshot_ci95": if_few["bootstrap_ci95"],
            "pass": cand >= p["absolute"]["strict_min"]
            and lb >= p["absolute"]["wilson_lower_min"]
            and if_few["bootstrap_ci95"][0] > p["versus_base_fewshot"]["ci_lower_gt"],
        },
        "G-PROTO": {
            "candidate": t["protocol_valid"]["candidate"],
            "pass": t["protocol_valid"]["candidate"] >= g["G-PROTO"]["protocol_valid_min"],
        },
        "G-REG": {
            "significant_regressions_vs_fewshot": t["significant_regressions_vs_fewshot"],
            "pass": not t["significant_regressions_vs_fewshot"],
        },
        "G-RET": {
            "candidate_bpb": retention["bpb"],
            "max": g["G-RET"]["full_split_bpb_max"],
            "pass": retention["mode"] == "full"
            and retention["precision"] == "fp32"
            and retention["bpb"] <= g["G-RET"]["full_split_bpb_max"],
        },
        "G-UNC": {
            "uncertainty_strict": t["abstention"]["uncertainty_strict"],
            "false_abstention_on_facts": t["abstention"]["false_abstention_on_facts"],
            "pass": t["abstention"]["uncertainty_strict"] >= g["G-UNC"]["uncertainty_strict_min"]
            and t["abstention"]["false_abstention_on_facts"] <= g["G-UNC"]["false_abstention_max"],
        },
        "G-REC": {
            "ab_bitwise": recovery_ab["bitwise"],
            "ab_numerical_only_with_full_evidence": bool(recovery_ab.get("numerical_only")),
            "pilot_checkpoint_verified": bool(pilot_check["verify"][0]),
            "pilot_checkpoint_reproduces_dev_loss": pilot_check["exact"],
            "pass": bool(pilot_check["verify"][0])
            and bool(pilot_check["exact"])
            and (bool(recovery_ab["bitwise"]) or bool(recovery_ab.get("numerical_only"))),
        },
        "EARLY-STOP": {
            "rule": "dev loss rises at three consecutive dev evaluations",
            "pilot_updates": pilot_manifest["updates"],
            "registered_updates": proto["pilot"]["updates"],
            "triggered": early_stop_triggered(pilot_manifest.get("dev_history", [])),
            "pass": pilot_manifest["updates"] == proto["pilot"]["updates"]
            and not early_stop_triggered(pilot_manifest.get("dev_history", [])),
        },
    }
    ok = all(r["pass"] for r in rows.values())
    return {
        "protocol": proto["schema"],
        "decision": proto["decision"]["pass"] if ok else proto["decision"]["fail"],
        "gates": rows,
        "note": proto["decision"]["note"],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ("protocol", "comparison", "retention", "recovery-ab", "pilot-check", "pilot-dev-history", "out"):
        ap.add_argument(f"--{name}", type=Path, required=True)
    a = ap.parse_args()
    loaded = [json.loads(p.read_text()) for p in (a.protocol, a.comparison, a.retention, a.recovery_ab, a.pilot_check)]
    res = decide(*loaded, json.loads(a.pilot_dev_history.read_text()))
    a.out.write_text(json.dumps(res, indent=1, sort_keys=True))
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
