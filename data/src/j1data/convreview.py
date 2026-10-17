"""Review plumbing for conversational drafts under rubric ``j1.conv-rubric.v1`` (docs/sft/conv_rubric_v1.md).

``batches``  writes blind first-review batches: review id, bucket, the bucket instruction and the
             conversation only (no provenance, no route label).
``ingest``   validates first-review output against the rubric: exactly one review per draft; scores 0-2 on
             the eight items; the decision must equal the rubric rule; an accept in a factual bucket needs at
             least one non-blank source. A missing or invalid review is a reject.
``sample``   draws the stratified 25% second-review sample from first-accepted records (by hash within
             route and bucket) and writes it under salted ids, so the second reviewer cannot line its items
             up with first-review files.
``second``   ingests the blinded second review, bound to the first review's digest and to the exact sample.

Reviewers only accept or reject; no reviewer text is ever copied into a record (R13).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from j1data.hashing import canonical_json, sha256_bytes, sha256_file, sha256_json, unit_interval

VERSION = "j1.conv-review.v1"
RUBRIC = "j1.conv-rubric.v1"
ITEMS = (
    "correctness",
    "naturalness",
    "relevance",
    "continuity",
    "format",
    "uncertainty",
    "privacy",
    "target",
)
THRESHOLD = 13
SECOND_SHARE = 0.25
SECOND_MAX_REJECT = 0.10
# Buckets whose assistant turns state facts about the world: an accept must cite at least one source.
FACTUAL_BUCKETS = frozenset({"everyday_qa", "explanation", "answerable_contrast", "corrections"})


def read_rows(path: Path) -> list[dict[str, Any]]:
    # split on "\n" only: model text may contain U+2028/U+0085, which str.splitlines() would split on
    return [json.loads(x) for x in path.read_text().split("\n") if x.strip()]


def review_id(draft: dict[str, Any]) -> str:
    return str(draft["content_sha256"])[:16]


def salted(rid: str) -> str:
    return sha256_bytes(f"j1.conv-review.second:{rid}".encode())[:16]


def blind(draft: dict[str, Any], instruction: str, rid: str | None = None) -> dict[str, Any]:
    conv = draft["conversation"]
    return {
        "id": rid or review_id(draft),
        "bucket": draft["slot"]["bucket"],
        "bucket_instruction": instruction,
        "system": conv.get("system", ""),
        "turns": conv["turns"],
    }


def rule(scores: dict[str, int]) -> str:
    """Accept only with correctness 2, no zero anywhere and a total of at least 13."""
    ok = scores["correctness"] == 2 and min(scores.values()) > 0 and sum(scores.values()) >= THRESHOLD
    return "accept" if ok else "reject"


def validate(review: Any, bucket: str | None = None) -> list[str]:
    problems = []
    if not isinstance(review, dict):
        return ["not an object"]
    scores = review.get("scores")
    if not isinstance(scores, dict) or set(scores) != set(ITEMS):
        return [f"scores must have exactly {list(ITEMS)}"]
    if not all(isinstance(v, int) and not isinstance(v, bool) and 0 <= v <= 2 for v in scores.values()):
        problems.append("every score must be an integer 0-2")
    elif review.get("decision") != rule(scores):
        problems.append(
            f"decision {review.get('decision')!r} does not follow the rubric rule ({rule(scores)})"
        )
    for key in ("reasons", "checks"):
        if not isinstance(review.get(key), str):
            problems.append(f"{key} must be a string")
    sources = review.get("sources")
    if not isinstance(sources, list) or not all(isinstance(s, str) for s in sources):
        problems.append("sources must be a list of strings")
    elif (
        bucket in FACTUAL_BUCKETS
        and review.get("decision") == "accept"
        and not any(s.strip() for s in sources)
    ):
        problems.append(f"an accept in factual bucket {bucket} needs at least one source")
    return problems


def ingest(items: list[tuple[str, str]], reviews: list[dict[str, Any]], reviewer: str) -> dict[str, Any]:
    """``items`` is [(review id, bucket)]. Every item gets exactly one decision."""
    ids = [i for i, _ in items]
    if len(set(ids)) != len(ids):
        raise SystemExit("review ids collide; drafts are not distinct")
    by_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in reviews:
        by_id[str(r.get("id"))].append(r)
    unknown = sorted(set(by_id) - set(ids))
    if unknown:
        raise SystemExit(f"reviews for unknown ids: {unknown[:5]}")
    decisions: dict[str, dict[str, Any]] = {}
    for rid, bucket in items:
        got = by_id.get(rid, [])
        if len(got) != 1:
            decisions[rid] = {"decision": "reject", "invalid": f"{len(got)} reviews (need exactly 1)"}
            continue
        problems = validate(got[0], bucket)
        if problems:
            decisions[rid] = {"decision": "reject", "invalid": "; ".join(problems)}
            continue
        decisions[rid] = {k: got[0][k] for k in ("scores", "decision", "reasons", "checks", "sources")}
    counts = Counter(v["decision"] for v in decisions.values())
    return {
        "schema": VERSION,
        "rubric": RUBRIC,
        "reviewer": reviewer,
        "items": len(items),
        "accepted": counts["accept"],
        "rejected": counts["reject"],
        "invalid": sum(1 for v in decisions.values() if "invalid" in v),
        "decisions": dict(sorted(decisions.items())),
    }


def draft_items(drafts: list[dict[str, Any]]) -> list[tuple[str, str]]:
    return [(review_id(d), d["slot"]["bucket"]) for d in drafts]


def sample(drafts: list[dict[str, Any]], first: dict[str, Any]) -> list[str]:
    """25% of first-accepted ids per (route, bucket), at least one per stratum, lowest hash first."""
    accepted: dict[tuple[str, str], list[str]] = defaultdict(list)
    for d in drafts:
        rid = review_id(d)
        if first["decisions"][rid]["decision"] == "accept":
            accepted[(d["provenance"]["route"], d["slot"]["bucket"])].append(rid)
    out: list[str] = []
    for stratum in sorted(accepted):
        ids = sorted(accepted[stratum], key=lambda i: unit_interval("j1.conv-review.second", i))
        out += ids[: max(1, math.ceil(SECOND_SHARE * len(ids)))]
    return sorted(out)


def second(
    sample_ids: list[str],
    buckets: dict[str, str],
    reviews: list[dict[str, Any]],
    reviewer: str,
    first_sha: str,
) -> dict[str, Any]:
    """``reviews`` use salted ids; results are keyed by the real review id."""
    if not sample_ids:
        raise SystemExit("the second-review sample is empty")
    back = {salted(i): i for i in sample_ids}
    res = ingest([(salted(i), buckets[i]) for i in sample_ids], reviews, reviewer)
    res["decisions"] = dict(sorted((back[k], v) for k, v in res["decisions"].items()))
    rate = res["rejected"] / len(sample_ids)
    return res | {
        "first_ingest_sha256": first_sha,
        "sample_ids": sorted(sample_ids),
        "sample": len(sample_ids),
        "rejection_rate": round(rate, 4),
        "max_rejection_rate": SECOND_MAX_REJECT,
        "passed": rate <= SECOND_MAX_REJECT,
    }


def merge_rereview(first: dict[str, Any], rereview: dict[str, Any], reviewer: str) -> dict[str, Any]:
    """First review after a failed second review: re-review decisions replace the original accepts; the
    original rejects stand. The re-review must cover exactly the original accepts (it can only remove)."""
    accepted = {k for k, v in first["decisions"].items() if v["decision"] == "accept"}
    if set(rereview["decisions"]) != accepted:
        raise SystemExit("the re-review must cover exactly the original first-accepted records")
    decisions = {k: (rereview["decisions"][k] if k in accepted else v) for k, v in first["decisions"].items()}
    counts = Counter(v["decision"] for v in decisions.values())
    return {
        "schema": VERSION,
        "rubric": RUBRIC,
        "reviewer": reviewer,
        "items": len(decisions),
        "accepted": counts["accept"],
        "rejected": counts["reject"],
        "invalid": sum(1 for v in decisions.values() if "invalid" in v),
        "decisions": dict(sorted(decisions.items())),
        "merged_from": {
            "first_ingest_sha256": first["ingest_sha256"],
            "rereview_ingest_sha256": rereview["ingest_sha256"],
        },
    }


def self_hash(obj: dict[str, Any], key: str = "ingest_sha256") -> str:
    if sha256_json({k: v for k, v in obj.items() if k != key}) != obj.get(key):
        raise SystemExit(f"{key} self-hash mismatch")
    return str(obj[key])


def main(argv: list[str] | None = None) -> int:
    import tomllib

    ap = argparse.ArgumentParser(prog="python -m j1data.convreview", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("batches")
    b.add_argument("--drafts", type=Path, required=True)
    b.add_argument("--config", type=Path, required=True)
    b.add_argument("--size", type=int, default=40)
    b.add_argument("--out", type=Path, required=True)
    i = sub.add_parser("ingest")
    i.add_argument("--drafts", type=Path, required=True)
    i.add_argument("--reviews", type=Path, nargs="+", required=True)
    i.add_argument("--reviewer", required=True)
    i.add_argument("--out", type=Path, required=True)
    s = sub.add_parser("sample")
    s.add_argument("--drafts", type=Path, required=True)
    s.add_argument("--first", type=Path, required=True)
    s.add_argument("--config", type=Path, required=True)
    s.add_argument("--out", type=Path, required=True)
    m = sub.add_parser("merge")
    m.add_argument("--first", type=Path, required=True)
    m.add_argument("--rereview", type=Path, required=True)
    m.add_argument("--reviewer", required=True)
    m.add_argument("--out", type=Path, required=True)
    c = sub.add_parser("second")
    c.add_argument("--drafts", type=Path, required=True)
    c.add_argument("--first", type=Path, required=True)
    c.add_argument("--reviews", type=Path, nargs="+", required=True)
    c.add_argument("--reviewer", required=True)
    c.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    if a.cmd == "merge":
        f, r = json.loads(a.first.read_text()), json.loads(a.rereview.read_text())
        self_hash(f), self_hash(r)
        res = merge_rereview(f, r, a.reviewer)
        res["ingest_sha256"] = sha256_json(res)
        a.out.write_bytes(canonical_json(res) + b"\n")
        print(json.dumps({k: res[k] for k in ("items", "accepted", "rejected", "invalid")}))
        return 0
    if a.cmd in ("batches", "sample"):
        cfg = tomllib.loads(a.config.read_text())
        instr = {k: v["instruction"] for k, v in cfg["buckets"].items()}
    drafts = read_rows(a.drafts)
    if a.cmd == "batches":
        a.out.mkdir(parents=True, exist_ok=True)
        rows = [blind(d, instr[d["slot"]["bucket"]]) for d in drafts]
        for n in range(0, len(rows), a.size):
            (a.out / f"batch_{n // a.size:02d}.jsonl").write_bytes(
                b"".join(canonical_json(r) + b"\n" for r in rows[n : n + a.size])
            )
        print(json.dumps({"drafts": len(rows), "batches": math.ceil(len(rows) / a.size)}))
        return 0
    if a.cmd == "ingest":
        res = ingest(draft_items(drafts), [r for p in a.reviews for r in read_rows(p)], a.reviewer)
        res["inputs"] = {"drafts": sha256_file(a.drafts)} | {
            p.name: sha256_file(p) for p in sorted(a.reviews)
        }
        res["ingest_sha256"] = sha256_json(res)
        a.out.write_bytes(canonical_json(res) + b"\n")
        print(json.dumps({k: res[k] for k in ("items", "accepted", "rejected", "invalid")}))
        return 0
    first = json.loads(a.first.read_text())
    first_sha = self_hash(first)
    picked = sample(drafts, first)
    if a.cmd == "sample":
        chosen = set(picked)
        # Blind: salted ids, no first-review scores, no route or provenance.
        rows = [
            blind(d, instr[d["slot"]["bucket"]], salted(review_id(d)))
            for d in drafts
            if review_id(d) in chosen
        ]
        a.out.write_bytes(b"".join(canonical_json(r) + b"\n" for r in rows))
        print(json.dumps({"sample": len(rows)}))
        return 0
    buckets = dict(draft_items(drafts))
    res = second(picked, buckets, [r for p in a.reviews for r in read_rows(p)], a.reviewer, first_sha)
    res["inputs"] = {"drafts": sha256_file(a.drafts)} | {p.name: sha256_file(p) for p in sorted(a.reviews)}
    res["ingest_sha256"] = sha256_json(res)
    a.out.write_bytes(canonical_json(res) + b"\n")
    print(json.dumps({k: res[k] for k in ("sample", "rejected", "rejection_rate", "passed")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
