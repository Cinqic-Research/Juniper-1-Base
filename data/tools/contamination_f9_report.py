"""Assemble the committed F9 results (``prohibited_f9.json``, ``summary_f9.json``) from archived run outputs.

Inputs are the outputs of ``tools/contamination_f9.py`` (scan report, pass-2 hits, adjudication, controls)
and the index directory. Identities are cross-checked and the tool stops on any mismatch: index, accepted
manifest, tool SHA-256, hit/decision sets. Test-split documents appear as IDs and counts only.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path
from typing import Any

BLIND_SPOTS = [
    "Field tokens outside every indexable 10-gram (layers-v1 needs >= 3 alphanumeric tokens per gram): "
    "HumanEval+ 13.7 %, MBPP+ 16.8 %, MBPP challenge 1.6 % of value tokens; 6,174 HumanEval+ and 1,124 "
    "MBPP+ items have no gram and no identifier (punctuation-dense inputs, e.g. lists of brackets).",
    "Representation changes of the same test data: EvalPlus JSONL form (base_input/plus_input as JSON: "
    "double quotes, lists for tuples, null for None), pretty-printed or reordered inputs, single inputs "
    "shorter than 10 tokens.",
    "Partial copies shorter than about 2 x 64 tokens that do not cover 0.6 of one window's non-common grams.",
    "Expected outputs without inputs, re-generated or paraphrased tests, renamed entry points.",
    "Items whose grams are all common are matched only via common_only; generic literal windows make that "
    "rule over-inclusive (see sensitivity_informational), not under-inclusive.",
]


def _load(p: Path) -> Any:
    return json.loads(p.read_text())


def assemble(
    scan: dict,
    hits: list[dict],
    decisions: list[dict],
    controls_unc: dict,
    controls_cal: dict,
    index_meta: dict,
    sources: dict,
    coverage: dict,
    prereg_commit: str,
) -> tuple[dict, dict]:
    ident = index_meta["index_identity"]
    for name, rec in (
        ("scan", scan),
        ("controls_uncalibrated", controls_unc),
        ("controls_calibrated", controls_cal),
    ):
        if rec.get("index_identity") != ident:
            raise SystemExit(f"{name}: index identity {rec.get('index_identity')} != {ident}")
    if controls_cal.get("common_sha256") != scan["common_sha256"]:
        raise SystemExit("calibrated controls did not use the scan's common set")
    if controls_unc.get("common_sha256") is not None:
        raise SystemExit("uncalibrated controls used a common set")
    if len({h["doc_id"] for h in hits}) != len(hits):
        raise SystemExit("duplicate doc_id in hits")
    if [h["doc_id"] for h in hits] != [d["doc_id"] for d in decisions]:
        raise SystemExit("decisions do not cover exactly the pass-2 hits")
    if scan["hit_documents_pass2"] != len(hits):
        raise SystemExit("scan report and hit file disagree")
    by: dict[str, dict[str, dict[str, list[str]]]] = {}
    counts: collections.Counter[tuple[str, str]] = collections.Counter()
    for d in decisions:
        counts[(d["split"], d["decision"])] += 1
        if d["decision"] == "PROHIBITED":
            by.setdefault(d["split"], {}).setdefault(d["source_id"], {}).setdefault(d["reason"], []).append(
                d["doc_id"]
            )
    for split in by.values():
        for src in split.values():
            for ids in src.values():
                ids.sort()
    label_counts: collections.Counter[str] = collections.Counter()
    for h in hits:
        for lab in (
            set(h["layer_a_benchmarks"])
            | set(h.get("common_only_benchmarks", []))
            | set(h["layer_c_benchmarks"])
        ):
            label_counts[lab] += 1
    splits = ("train", "validation", "test")
    prohibited = {
        "schema": "j1.contamination-adjudication.f9.v1",
        "criteria": f"docs/contamination_f9_criteria.md (pre-registration commit {prereg_commit}); "
        "docs/contamination_adjudication_criteria_v2.md unchanged",
        "index_identity": ident,
        "accepted_manifest_sha256": scan["accepted_manifest_sha256"],
        "build_identity": scan["build_identity"],
        "tool_sha256": scan["tool_sha256"],
        "layout": "prohibited[split][source_id][reason] = sorted doc_ids",
        "counts": {
            s: sum(len(ids) for src in by.get(s, {}).values() for ids in src.values()) for s in splits
        },
        "prohibited": {s: by.get(s, {}) for s in splits},
    }
    summary = {
        "schema": "j1.contamination-f9-summary.v1",
        "prereg_commit": prereg_commit,
        "index_identity": ident,
        "index_items": index_meta["items"],
        "index_distinct_grams": index_meta["distinct_grams"],
        "index_identifiers": index_meta["identifiers"],
        "fields": {
            lab: {k: v for k, v in rec.items() if k != "files"} | {"files": rec["files"]}
            for lab, rec in sources["labels"].items()
        },
        "index_coverage": {
            lab: index_meta["benchmarks"][lab] | coverage.get(lab, {}) for lab in sources["labels"]
        },
        "controls_uncalibrated": {"positive": controls_unc["positive"], "negative": controls_unc["negative"]},
        "controls_calibrated": {"positive": controls_cal["positive"], "negative": controls_cal["negative"]},
        "scan": {k: v for k, v in scan.items() if k != "version"},
        "hits_pass2": len(hits),
        "hits_by_label": dict(sorted(label_counts.items())),
        "decisions": {f"{s}:{dec}": n for (s, dec), n in sorted(counts.items())},
        "prohibited_by_split": prohibited["counts"],
        "prohibited_by_split_source": {
            s: {src: sum(len(ids) for ids in v.values()) for src, v in sorted(by.get(s, {}).items())}
            for s in splits
        },
    }
    return prohibited, summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--index", type=Path, required=True)
    ap.add_argument("--prereg-commit", required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()
    r = a.results
    prohibited, summary = assemble(
        _load(r / "scan_f9.json"),
        _load(r / "hits_pass2.json"),
        _load(r / "adjudication_f9.json"),
        _load(r / "controls_f9_uncalibrated.json"),
        _load(r / "controls_f9_calibrated.json"),
        _load(a.index / "index.json"),
        _load(a.index / "f9_sources.json"),
        _load(r / "item_coverage_f9.json"),
        a.prereg_commit,
    )
    sens = r / "sensitivity_f9.json"
    if sens.is_file():
        summary["sensitivity_informational"] = _load(sens)
    summary["residual_blind_spots"] = BLIND_SPOTS
    a.out_dir.mkdir(parents=True, exist_ok=True)
    (a.out_dir / "prohibited_f9.json").write_text(json.dumps(prohibited) + "\n")
    (a.out_dir / "summary_f9.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps({"prohibited_by_split": summary["prohibited_by_split"], "hits": summary["hits_pass2"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
