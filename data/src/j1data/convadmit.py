"""Admission build for the conversational seed corpus (j1.conv-admit.v1; contract section 5).

``python -m j1data.convadmit --config configs/conv_seed_v1.toml --plan DIR --drafts drafts.jsonl
--first first.json --second second.json --sealed DIR --bundle DIR [--bundle DIR ...] --tokenizer FILE
--data-root DIR --out DIR``

The second review must be bound to this first review (digest) and must cover exactly the stratified sample
that ``j1data.convreview.sample`` draws from it, and it must pass its >10% rule. Then a draft is admitted
only if all of these hold:
* the first review accepted it and the second review (if it sampled the record) did not reject it;
* its bucket is still under the bucket's registered count (records are taken in plan slot order);
* it is not a near-duplicate of an admitted record (word 5-gram Jaccard >= 0.5 over the conversation, or
  first user turns of >= 5 words with word-set Jaccard >= 0.8) and its assistant turns do not open with a
  six-word prefix already used by three admitted records (template repetition);
* no turn shares a word 8-gram with a decoded prompt of the required evaluation bundles or a sealed
  conversational confirmation prompt, no turn contains a whole short (3-7 word) prompt of those, and the
  layers-v1 benchmark index finds no match;
* it serializes as a valid ``juniper-chat-v1`` episode within 1,025 tokens (system text masked).
The build fails unless at least the contract's minimum is admitted and the median assistant turn is at most
40 words. The split is by whole conversation, 10% dev by hash.

Factuality is ``MODEL_JUDGED_ONLY``: a reviewer model, independent of the GPT-OSS author, checked the facts
and attests sources, which is not an oracle. Output follows ``j1.sft-dataset.v1``; mixing it needs a
conversation-source adapter in ``sftmix`` (not written yet). It is not a training authorization.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import tomllib
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

from j1data import convreview as cr
from j1data import jtok
from j1data.hashing import canonical_json, doc_id, sha256_bytes, sha256_file, sha256_json, unit_interval
from j1data.sft import IGNORE, causal_rows, encode_text_episode

VERSION = "j1.conv-admit.v1"
LICENSE = "LicenseRef-gpt-oss-20b-output-conv"
RIGHTS = (
    "R05 output of a locally run openai/gpt-oss-20b @6cee5e81 (Apache-2.0 weights), conversational seed "
    "corpus route B (owner approval 2026-10-14 1:39 PM EDT; contract docs/sft/conversation_contract_v1.md)"
)
NOTICE = (
    "Conversational seed corpus j1-conv-seed-v1.\n\nEvery conversation was drafted by a locally run "
    "openai/gpt-oss-20b (revision 6cee5e81ee83917806bbde320786a8fb61efebee; weights licensed Apache-2.0, "
    "https://www.apache.org/licenses/LICENSE-2.0) and admitted after automatic checks and a rubric review "
    "by separately prompted Claude models that could only accept or reject. No human reviewed the corpus. "
    "Records are labelled teacher/gpt-oss-20b-local.\n"
)
ROUTE, AUTHOR_LABEL = "B", "teacher/gpt-oss-20b-local"
MAX_TOKENS = 1025  # as j1-sft-v1.3: 1,024 trained positions per row
DEV_SHARE = 0.10
NEAR_DUP_5GRAM = 0.5
NEAR_DUP_FIRST_USER = 0.8
FIRST_USER_MIN_WORDS = 5
TEMPLATE_PREFIX_WORDS, TEMPLATE_MAX_REPEATS = 6, 3
CONTAM_NGRAM = 8
SHORT_MIN_WORDS = 3
MIN_ADMITTED = 300
MAX_MEDIAN_WORDS = 40
REQUIRED_BUNDLES = frozenset(
    {"j1-sft-eval-v1.2", "j1-sft-eval-v1.3", "j1-sft-confirm-v1", "j1-sft-confirm-v3"}
)


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower().replace("\u2019", "'"))


def _grams(text: str, n: int) -> set[str]:
    w = _words(text)
    return {" ".join(w[i : i + n]) for i in range(len(w) - n + 1)}


def _jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if a | b else 0.0


def conversation_text(conv: dict[str, Any]) -> str:
    return "\n".join([conv.get("system", "")] + [t["text"] for t in conv["turns"]])


class Protected:
    """Evaluation and sealed prompts: 8-gram overlap, plus whole-phrase containment for short prompts."""

    def __init__(self, texts: list[str]) -> None:
        self.grams: set[str] = set()
        self.short: set[str] = set()
        for t in texts:
            w = _words(t)
            if len(w) >= CONTAM_NGRAM:
                self.grams |= _grams(t, CONTAM_NGRAM)
            elif len(w) >= SHORT_MIN_WORDS:
                self.short.add(" ".join(w))

    def hit(self, text: str) -> bool:
        if _grams(text, CONTAM_NGRAM) & self.grams:
            return True
        w = _words(text)
        if len(w) < SHORT_MIN_WORDS:
            return False
        padded = f" {' '.join(w)} "
        return any(f" {s} " in padded for s in self.short)


def episode(draft: dict[str, Any], review: dict[str, Any], decon_identity: str) -> dict[str, Any]:
    conv = draft["conversation"]
    messages: list[dict[str, Any]] = []
    if conv.get("system", "").strip():
        messages.append(
            {
                "role": "system",
                "content": [{"type": "text", "text": conv["system"], "trust": "trusted"}],
                "train": False,
            }
        )
    for t in conv["turns"]:
        trust = "trusted" if t["role"] == "assistant" else "untrusted"
        m: dict[str, Any] = {
            "role": t["role"],
            "content": [{"type": "text", "text": t["text"], "trust": trust}],
            "train": t["role"] == "assistant",
        }
        if t["role"] == "assistant":
            m["channel"] = "final"
        messages.append(m)
    sources = [s for s in review["sources"] if s.strip()]
    checks = [
        {
            "name": "convdraft_automatic_checks",
            "oracle": "deterministic_rule",
            "passed": True,
            "detail": "j1.convdraft.v1 shape/banned-pattern/arithmetic/length/overlap checks",
        },
        {
            "name": "rubric_first_review",
            "oracle": "independent_model",
            "passed": True,
            "detail": f"j1.conv-rubric.v1 total {sum(review['scores'].values())}/16; Claude reviewer, "
            "independent of the GPT-OSS author",
        },
        {
            # Reviewer text stays in the review record (rubric): only a count and a digest here.
            "name": "reviewer_attested_sources",
            "oracle": "independent_model",
            "passed": True,
            "detail": f"{len(sources)} sources; sha256 {sha256_json(sources)}",
        },
    ]
    return {
        "schema": "j1.episode.v1",
        "episode_id": doc_id("conv-seed-v1", draft["content_sha256"]),
        "stage": "agent-instruction-sft",
        "task_family": f"conv.{draft['slot']['bucket']}",
        "behaviors": ["direct_answer", "no_tool"],
        "messages": messages,
        "provenance": {
            "origin": "teacher",
            "teacher_record_id": draft["content_sha256"],
            "source_prompt_id": draft["slot"]["slot"],
            "source_license": LICENSE,
            "created": VERSION,
        },
        "validation": {
            "validator_version": f"{VERSION}+j1.conv-rubric.v1",
            "checks": checks,
            "factuality": "MODEL_JUDGED_ONLY",
            "accepted": True,
            "acceptance_reason": "automatic checks, rubric review, second-review rule, near-duplicate, "
            "template and contamination checks",
        },
        "decontamination": {"index_identity": decon_identity, "best_coverage": 0.0, "passed": True},
    }


def bundle_prompts(bundles: list[Path], tok: Any) -> tuple[list[str], dict[str, str]]:
    out: list[str] = []
    seen: dict[str, str] = {}
    for b in bundles:
        m = json.loads((b / "manifest.json").read_text())
        if sha256_json({k: v for k, v in m.items() if k != "bundle_sha256"}) != m["bundle_sha256"]:
            raise SystemExit(f"{b}: bundle manifest self-hash mismatch")
        if sha256_file(b / "prompts.jsonl") != m["files"]["prompts.jsonl"]["sha256"]:
            raise SystemExit(f"{b}/prompts.jsonl: bytes differ from its manifest")
        seen[m["version"]] = m["bundle_sha256"]
        for row in cr.read_rows(b / "prompts.jsonl"):
            out.append(str(tok.decode([int(t) for t in row["prompt_ids"]], skip_special_tokens=True)))
    missing = REQUIRED_BUNDLES - set(seen)
    if missing:
        raise SystemExit(f"required evaluation bundles missing: {sorted(missing)}")
    return out, seen


Admitted = tuple[str, dict[str, Any], dict[str, Any], tuple[list[int], list[int]]]


def check_reviews(drafts: list[dict[str, Any]], first: dict[str, Any], second: dict[str, Any]) -> None:
    if second.get("first_ingest_sha256") != first.get("ingest_sha256"):
        raise SystemExit("the second review is not bound to this first review")
    expected = cr.sample(drafts, first)
    if (
        not expected
        or second.get("sample_ids") != expected
        or set(second.get("decisions", {})) != set(expected)
    ):
        raise SystemExit("the second review does not cover exactly the registered stratified sample")
    if not second.get("passed"):
        raise SystemExit("the second review did not pass its rule; everything goes back to first review")


def _near_duplicate(
    grams: set[str], first_user: list[str], kept_grams: list[set[str]], kept_first: list[set[str]]
) -> bool:
    if any(_jaccard(grams, g) >= NEAR_DUP_5GRAM for g in kept_grams):
        return True
    if len(first_user) < FIRST_USER_MIN_WORDS:
        return False
    fu = set(first_user)
    return any(len(f) >= FIRST_USER_MIN_WORDS and _jaccard(fu, f) >= NEAR_DUP_FIRST_USER for f in kept_first)


def admit(
    cfg: dict[str, Any],
    plan_obj: dict[str, Any],
    drafts: list[dict[str, Any]],
    first: dict[str, Any],
    second: dict[str, Any],
    protected: Protected,
    tok: Any,
    decon_identity: str,
    layers_match: Callable[[str], bool] | None = None,
) -> tuple[list[Admitted], list[dict[str, Any]]]:
    """Returns (admitted [(split, episode, draft, row)], rejections)."""
    check_reviews(drafts, first, second)
    order = {s["slot"]: i for i, s in enumerate(plan_obj["slots"])}
    caps = {b: int(v["count"]) for b, v in cfg["buckets"].items()}
    taken: dict[str, int] = dict.fromkeys(caps, 0)
    kept_grams: list[set[str]] = []
    kept_first: list[set[str]] = []
    prefixes: Counter[str] = Counter()
    admitted: list[Admitted] = []
    rejected: list[dict[str, Any]] = []
    for d in sorted(drafts, key=lambda x: order[x["slot"]["slot"]]):
        rid = cr.review_id(d)
        bucket = d["slot"]["bucket"]
        rev = first["decisions"].get(rid, {"decision": "reject"})
        conv = d["conversation"]
        text = conversation_text(conv)
        grams = _grams(text, 5)
        first_user = _words(conv["turns"][0]["text"])
        heads = [
            " ".join(_words(t["text"])[:TEMPLATE_PREFIX_WORDS])
            for t in conv["turns"]
            if t["role"] == "assistant"
        ]
        heads = [h for h in heads if len(h.split()) == TEMPLATE_PREFIX_WORDS]
        why = None
        if rev["decision"] != "accept":
            why = "first review rejected"
        elif second["decisions"].get(rid, {}).get("decision") == "reject":
            why = "second review rejected"
        elif taken[bucket] >= caps[bucket]:
            why = "bucket full"
        elif _near_duplicate(grams, first_user, kept_grams, kept_first):
            why = "near-duplicate of an admitted record"
        elif any(prefixes[h] >= TEMPLATE_MAX_REPEATS for h in heads):
            why = "template repetition"
        elif any(protected.hit(t["text"]) for t in conv["turns"]) or protected.hit(conv.get("system", "")):
            why = "overlaps an evaluation or sealed confirmation prompt"
        elif layers_match is not None and layers_match(text):
            why = "matches the layers-v1 benchmark index"
        if not why:
            ep = episode(d, rev, decon_identity)
            try:
                enc = encode_text_episode(
                    tok, ep, approved_source_licenses={LICENSE}, expected_decontamination_index=decon_identity
                )
            except ValueError as e:
                why = f"serialization: {e}"
            else:
                if len(enc.ids) > MAX_TOKENS:
                    why = f"{len(enc.ids)} tokens exceed {MAX_TOKENS}"
        if why:
            rejected.append({"id": rid, "slot": d["slot"]["slot"], "bucket": bucket, "reason": why})
            continue
        xs, ys = causal_rows([enc], len(enc.ids) - 1, 0, vocab_size=tok.get_vocab_size())
        split = "dev" if unit_interval("j1.conv-admit.split", rid) < DEV_SHARE else "train"
        admitted.append((split, ep, d, (xs[0], ys[0])))
        taken[bucket] += 1
        kept_grams.append(grams)
        kept_first.append(set(first_user))
        prefixes.update(heads)
    return admitted, rejected


def median_assistant_words(admitted: list[Admitted]) -> float:
    words = [
        len(t["text"].split())
        for _, _, d, _ in admitted
        for t in d["conversation"]["turns"]
        if t["role"] == "assistant"
    ]
    return float(statistics.median(words)) if words else 0.0


def record(
    split: str, ep: dict[str, Any], d: dict[str, Any], row: tuple[list[int], list[int]]
) -> dict[str, Any]:
    first_user = " ".join(_words(d["conversation"]["turns"][0]["text"]))
    key = sha256_bytes(f"conv-seed-v1:{first_user}".encode())
    rec = {
        "episode_id": ep["episode_id"],
        "split": split,
        "family": ep["task_family"],
        "category": "conversation",
        "route": ROUTE,
        "author_label": AUTHOR_LABEL,
        "source_id": "conv-seed-v1",
        "source_revision": d["provenance"]["artifact_sha256"],
        "source_record": d["slot"]["slot"],
        "license": LICENSE,
        "rights_decision": RIGHTS,
        "origin": "teacher",
        "task_key_sha256": key,
        "task_keys_sha256": [key],
        "content_sha256": sha256_bytes(canonical_json(ep["messages"])),
        "verification": [c["name"] + ":" + c["oracle"] for c in ep["validation"]["checks"]],
        "factuality": ep["validation"]["factuality"],
        "decontamination_best_coverage": 0.0,
        "notes": [],
        "tokens": len(row[0]) + 1,
        "target_tokens": sum(1 for v in row[1] if v != IGNORE),
    }
    rec["record_sha256"] = sha256_json({**rec, "episode": ep})
    return rec | {"episode": ep}


CODE_FILES = (
    "convadmit.py",
    "convreview.py",
    "convdraft.py",
    "sft.py",
    "sftbuild.py",
    "jtok.py",
    "episodes.py",
    "hashing.py",
    "protocol.py",
)


def main(argv: list[str] | None = None) -> int:
    from j1data import convdraft, sftgen
    from j1data.registry import REPO_ROOT
    from j1data.sftbuild import LAYERS_MIN_COVERAGE, LAYERS_REL, _layers_index, _save_rows

    ap = argparse.ArgumentParser(prog="python -m j1data.convadmit", description=__doc__)
    for name in ("config", "drafts", "first", "second", "sealed", "tokenizer", "data_root", "out"):
        ap.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    ap.add_argument("--plan", type=Path, action="append", required=True, help="round 1 first, then top-ups")
    ap.add_argument("--bundle", type=Path, action="append", required=True)
    ap.add_argument("--min-admitted", type=int, default=MIN_ADMITTED)
    ap.add_argument("--amendment", default="", help="contract section that authorizes a lower minimum")
    a = ap.parse_args(argv)
    if a.min_admitted < MIN_ADMITTED and not a.amendment.strip():
        raise SystemExit(f"a minimum below {MIN_ADMITTED} needs --amendment naming the contract section")
    cfg = tomllib.loads(a.config.read_text())
    plans = [json.loads((p / "plan.json").read_text()) for p in a.plan]
    plan_obj = plans[0]
    if plan_obj.get("round", 1) != 1 or any(
        p.get("base_plan_sha256") != sha256_json(plan_obj) or p.get("round") != n + 2
        for n, p in enumerate(plans[1:])
    ):
        raise SystemExit("--plan must be round 1 followed by its top-up rounds in order")
    combined = {"slots": [s for p in plans for s in p["slots"]]}
    tok = jtok.load(a.tokenizer.read_text())
    sealed_rows, seal = convdraft.load_sealed(a.sealed, plan_obj, cfg)
    if not convdraft.seal_committed(seal["seal_sha256"]):
        raise SystemExit("the seal digest is not committed under docs/sft")
    prompts, bundles = bundle_prompts(a.bundle, tok)
    protected = Protected(prompts + [r["user"] for r in sealed_rows])
    decon_identity = (
        "layers-v1:" + sha256_file(a.data_root / LAYERS_REL / "index.json") + "+keys:" + sftgen.VERSION
    )
    index = _layers_index(a.data_root)

    def layers_match(text: str) -> bool:
        return index.match(text, LAYERS_MIN_COVERAGE) is not None

    first, second = json.loads(a.first.read_text()), json.loads(a.second.read_text())
    first_sha, second_sha = cr.self_hash(first), cr.self_hash(second)
    drafts = cr.read_rows(a.drafts)
    admitted, rejected = admit(
        cfg, combined, drafts, first, second, protected, tok, decon_identity, layers_match
    )
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "rejections.jsonl").write_bytes(b"".join(canonical_json(r) + b"\n" for r in rejected))
    median = median_assistant_words(admitted)
    if len(admitted) < a.min_admitted or median > MAX_MEDIAN_WORDS:
        raise SystemExit(
            f"admitted {len(admitted)} (minimum {a.min_admitted}), median assistant words {median} "
            f"(maximum {MAX_MEDIAN_WORDS}); no dataset written (rejections.jsonl kept for diagnosis)"
        )
    files: dict[str, Any] = {}
    counts: dict[str, Any] = {}
    for split in ("train", "dev"):
        part = [x for x in admitted if x[0] == split]
        recs = [record(*x) for x in part]
        (a.out / f"{split}.jsonl").write_bytes(b"".join(canonical_json(r) + b"\n" for r in recs))
        npz = a.out / f"{split}_rows.npz"
        _save_rows(npz, [x[3] for x in part], [r["episode_id"] for r in recs])
        files[f"{split}.jsonl"] = {"rows": len(recs), "sha256": sha256_file(a.out / f"{split}.jsonl")}
        files[f"{split}_rows.npz"] = {"sha256": sha256_file(npz)}
        counts[split] = {
            "episodes": len(recs),
            "tokens": sum(r["tokens"] for r in recs),
            "target_tokens": sum(r["target_tokens"] for r in recs),
            "by_bucket": {
                b: sum(1 for r in recs if r["family"] == f"conv.{b}") for b in sorted(cfg["buckets"])
            },
        }
    (a.out / "NOTICE").write_text(NOTICE)
    files["NOTICE"] = {"sha256": sha256_file(a.out / "NOTICE")}
    files["rejections.jsonl"] = {"rows": len(rejected), "sha256": sha256_file(a.out / "rejections.jsonl")}
    here = Path(__file__).parent
    code = {n: sha256_file(here / n) for n in CODE_FILES}
    code["tools/contamination_layers.py"] = sha256_file(REPO_ROOT / "tools/contamination_layers.py")
    manifest = {
        "schema": "j1.sft-dataset.v1",
        "version": cfg["draft"]["version"],
        "builder": VERSION,
        "stage": "agent-instruction-sft",
        "protocol": "juniper-chat-v1",
        "max_tokens": MAX_TOKENS,
        "tokenizer_sha256": sha256_file(a.tokenizer),
        "decontamination_index_identity": decon_identity,
        "decontamination_note": "the layers-v1 part is real (same index, coverage 0.6, any hit rejects); the "
        "'+keys' part is vacuous here, since conversation records carry no generator task keys. The "
        "conversation-specific checks are the 8-gram and short-prompt checks listed under 'checks'.",
        "approved_licenses": [LICENSE],
        "row_format": "per-episode causal rows from j1data.sft.causal_rows: inputs = ids[:-1], targets = "
        "labels[1:] (-100 masked); one episode per row; no packing",
        "counts": counts,
        "median_assistant_words": median,
        "files": files,
        "review": {
            "rubric": cr.RUBRIC,
            "first_reviewer": first["reviewer"],
            "second_reviewer": second["reviewer"],
            "first": {k: first[k] for k in ("items", "accepted", "rejected", "invalid")},
            "second": {
                k: second[k] for k in ("sample", "rejected", "rejection_rate", "max_rejection_rate", "passed")
            },
            "statement": "both reviewers are separately prompted Claude subagents (same vendor, labelled); "
            "they could only accept or reject; no human reviewed the corpus",
        },
        "inputs": {
            "config_sha256": sha256_json(cfg),
            "plan_sha256": [sha256_json(p) for p in plans],
            "drafts_sha256": sha256_file(a.drafts),
            "first_review_sha256": first_sha,
            "second_review_sha256": second_sha,
            "seal_sha256": seal["seal_sha256"],
            "eval_bundles": dict(sorted(bundles.items())),
        },
        "checks": {
            "near_duplicate": f"word 5-gram Jaccard >= {NEAR_DUP_5GRAM}; first user turns of >= "
            f"{FIRST_USER_MIN_WORDS} words with Jaccard >= {NEAR_DUP_FIRST_USER}",
            "template_repetition": f"assistant {TEMPLATE_PREFIX_WORDS}-word openings used by "
            f"{TEMPLATE_MAX_REPEATS} admitted records",
            "contamination": f"word {CONTAM_NGRAM}-gram overlap and whole short-prompt containment against "
            "decoded evaluation prompts and sealed prompts; "
            f"layers-v1 index at coverage {LAYERS_MIN_COVERAGE}",
            "split": f"{DEV_SHARE:.0%} dev by hash of the whole conversation",
            "gates": f">= {a.min_admitted} admitted (registered {MIN_ADMITTED}); median assistant words "
            f"<= {MAX_MEDIAN_WORDS}",
            "minimum_amendment": a.amendment or None,
        },
        "code_sha256": code,
        "rejection_history": "rejections.jsonl (admission) and the drafting run's rejections.jsonl",
        "note": "not a training authorization; a stage needs a qualified parent and a registered protocol",
    }
    manifest["manifest_sha256"] = sha256_json(manifest)
    (a.out / "manifest.json").write_bytes(canonical_json(manifest) + b"\n")
    print(
        json.dumps(
            {"admitted": counts, "rejected": len(rejected), "median_assistant_words": median}, indent=1
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
