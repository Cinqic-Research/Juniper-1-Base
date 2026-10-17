"""Filtered OpenAssistant OASST1 conversations (rights entry R07) for instruction SFT.

OASST1 is volunteer-written and volunteer-ranked; nothing in it is fact-checked and its human-origin
metadata (``synthetic``/``model_name``) cannot prove that a volunteer never pasted model output.
This module therefore keeps a deliberately narrow subset and records why everything else was
dropped:

* structure: ``ready_for_export`` trees; every message on the path English, not deleted, review
  passed, not synthetic, no ``model_name``; each assistant step is the rank-0 reply, and a deeper
  user follow-up is taken only under that reply;
* label thresholds, model-origin heuristics, PII patterns, code, claim density / advice domains,
  stale "current" statements and the token budget (``configs/sft_oasst_v1.toml``).

A pair (user message plus the assistant reply) that fails a filter cuts the conversation before it
(whole-turn trimming); a tree whose first pair fails is rejected. Every cut and rejection lands in
the ledger. Assistant targets carry an ``open_ended`` spec scored by the declared rubric
``j1.open-ended-rubric.v1`` (``docs/sft/oasst_v1.md``); there is no deterministic oracle.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import tomllib
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from j1data import jtok, sftgen
from j1data.acquire import HF_ENDPOINT, download_file, list_hf_files
from j1data.hashing import canonical_json, sha256_bytes, sha256_file, unit_interval
from j1data.protocol import Message, serialize
from j1data.sftgen import Item, Turn

VERSION = "j1.sftoasst.v1"
SOURCE_ID = "oasst1"
LICENSE = "Apache-2.0"
RUBRIC = "j1.open-ended-rubric.v1"
OPEN_SPEC: dict[str, Any] = {"type": "open_ended", "rubric": RUBRIC}
SPLITS = ("train", "dev", "test")
STAGES = ("structure", "labels", "model_origin", "pii", "code", "claims", "temporal", "length", "dedup")
RUBRIC_DIMENSIONS = (
    "helpfulness",
    "instruction_adherence",
    "coherence",
    "factual_caution",
    "safety_privacy",
    "language_quality",
    "no_model_origin_signs",
)

Msg = dict[str, Any]
Pair = tuple[Msg, Msg]  # (prompter, assistant)


def load_config(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        return tomllib.load(handle)


# --- upstream files ------------------------------------------------------------------------------


def git_blob_sha1(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def fetch(cfg: dict[str, Any], root: Path) -> dict[str, Path]:
    """Pinned upstream files under ``sft/raw/oasst1/<revision>/``. Offline when every file is present
    and matches the lock; otherwise downloads at the pinned revision and verifies LFS SHA-256 (data)
    or the git blob id (LICENSE/README) before comparing with the lock."""
    src = cfg["source"]
    lock: dict[str, str] = src["lock"]
    base = root / "sft/raw" / SOURCE_ID / src["revision"]
    wanted = [*src["files"], *src["evidence"]]
    local = {rel: base / rel for rel in wanted}
    if all(p.is_file() and sha256_file(p) == lock[rel] for rel, p in local.items()):
        return local
    listing = {e["path"]: e for e in list_hf_files(src["repo"], src["revision"])}
    for rel in wanted:
        info = listing[rel]
        upstream = (info.get("lfs") or {}).get("oid")
        url = f"{HF_ENDPOINT}/datasets/{src['repo']}/resolve/{src['revision']}/{rel}"
        digest = download_file(url, local[rel], int(info["size"]), upstream)
        if upstream is None and git_blob_sha1(local[rel].read_bytes()) != info["oid"]:
            raise RuntimeError(f"{rel}: bytes differ from the git blob at the pinned revision")
        if digest != lock[rel]:
            raise RuntimeError(f"{rel}: bytes differ from the configured lock")
    return local


def read_messages(paths: Sequence[Path]) -> list[Msg]:
    import pyarrow.parquet as pq

    rows: list[Msg] = []
    for p in paths:
        rows.extend(pq.read_table(p).to_pylist())
    return rows


# --- per-message checks ----------------------------------------------------------------------------


def labels(m: Msg, min_count: int = 1) -> dict[str, float]:
    lab = m.get("labels") or {}
    return {
        n: float(v)
        for n, v, c in zip(lab.get("name") or [], lab.get("value") or [], lab.get("count") or [], strict=True)
        if c is not None and c >= min_count
    }


@dataclass
class Fail:
    stage: str
    reason: str
    margin: float | None = None  # distance past a numeric threshold (near-threshold review sampling)
    detail: str = ""


def structural(m: Msg, cfg: dict[str, Any], role: str) -> Fail | None:
    t = cfg["tree"]
    checks = (
        ("role", m.get("role") == role),
        ("tree_state", m.get("tree_state") == t["tree_state"]),
        ("lang", m.get("lang") == t["lang"]),
        ("deleted", m.get("deleted") is False),
        ("review_result", m.get("review_result") is True),
        ("synthetic", m.get("synthetic") is False),
        ("model_name", m.get("model_name") is None),
        ("rank", role != "assistant" or m.get("rank") == t["assistant_rank"]),
        ("empty", bool((m.get("text") or "").strip())),
    )
    for name, ok in checks:
        if not ok:
            return Fail("structure", f"{role}_{name}")
    return None


def label_fail(m: Msg, cfg: dict[str, Any]) -> Fail | None:
    lc = cfg["labels"]
    role = "assistant" if m["role"] == "assistant" else "prompter"
    counted = labels(m, int(lc["required_min_count"]))
    for name in lc[f"{role}_required"]:
        if name not in counted:
            return Fail("labels", f"{role}_unlabeled_{name}")
    lab = labels(m, 1)  # thresholds apply to any label with at least one annotation
    for name, cap in {**lc["flag_max"], **lc["likert_max"]}.items():
        if name in lab and lab[name] > cap:
            return Fail("labels", f"{role}_{name}", lab[name] - cap, f"{name}={lab[name]:.3f}>{cap}")
    for name, floor in lc[f"{role}_min"].items():
        if name in lab and lab[name] < floor:
            return Fail("labels", f"{role}_low_{name}", floor - lab[name], f"{name}={lab[name]:.3f}<{floor}")
    detox = m.get("detoxify") or {}
    worst = max(((v, k) for k, v in detox.items() if v is not None), default=(0.0, ""))
    if worst[0] > lc["detoxify_max"]:
        return Fail("labels", f"{role}_detoxify_{worst[1]}", worst[0] - lc["detoxify_max"])
    return None


def _compile(patterns: Sequence[str]) -> re.Pattern[str]:
    return re.compile("|".join(f"(?:{p})" for p in patterns), re.IGNORECASE)


@dataclass
class Rules:
    """Compiled text heuristics (patterns declared in the config; PII and code rules here)."""

    model_any: re.Pattern[str]
    model_assistant: re.Pattern[str]
    prompter_identity: re.Pattern[str]
    author_flag: re.Pattern[str]
    advice: dict[str, re.Pattern[str]]
    claim_phrases: re.Pattern[str]
    recency: re.Pattern[str]

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> Rules:
        mo = cfg["model_origin"]
        return cls(
            _compile(mo["any_role"]),
            _compile(mo["assistant"]),
            _compile(mo["prompter_identity"]),
            _compile(mo["author_flag"]),
            {k: re.compile(r"\b(?:" + "|".join(v) + r")", re.I) for k, v in cfg["claims"]["advice"].items()},
            _compile(cfg["claims"]["phrases"]),
            _compile(cfg["temporal"]["recency"]),
        )


EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)*\.[A-Za-z]{2,}")
PHONE = re.compile(r"(?<![\w.])(?:\+\d{1,3}[ .-]?)?(?:\(\d{2,4}\)|\d{3})[ .-]\d{3}[ .-]\d{3,4}(?![\w.])")
SSN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
STREET = re.compile(
    r"\b\d{1,5}\s+(?:[A-Z][a-z]+\s+){1,3}(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr"
    r"|Court|Ct|Way|Place|Pl|Terrace|Highway|Hwy)\b\.?"
)
URL = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
USER_URL = re.compile(r"[?&][\w-]+=|/(?:users?|u|profile|people|account|members?|@)[/\w.-]|[\w.-]+@", re.I)
IBAN = re.compile(r"\b[A-Z]{2}\d{2}(?: ?[A-Z0-9]{4}){2,7}(?: ?[A-Z0-9]{1,4})?\b")
DIGIT_RUN = re.compile(r"(?<!\d)\d(?:[ -]?\d){12,18}(?!\d)")
IPV4 = re.compile(r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b")


def luhn(digits: str) -> bool:
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch) * (2 if i % 2 else 1)
        total += d - 9 if d > 9 else d
    return total % 10 == 0


def pii_hits(text: str) -> list[str]:
    hits = [
        name
        for name, rx in (("email", EMAIL), ("phone", PHONE), ("ssn", SSN), ("street", STREET))
        if rx.search(text)
    ]
    if any(USER_URL.search(u) for u in URL.findall(text)):
        hits.append("url_user_data")
    if IBAN.search(text):
        hits.append("iban")
    if any(luhn(re.sub(r"\D", "", r)) for r in DIGIT_RUN.findall(text)):
        hits.append("card_like_digits")
    if IPV4.search(text):
        hits.append("ip_address")
    return hits


CODE_LINE = re.compile(
    r"^\s*(?:def |class |import |from \S+ import|#include|public |private |function\b|const |let |var |"
    r"return\b|if\s*\(|for\s*\(|while\s*\(|SELECT\b|\$ |>>> |<\w+[^>]*>|.*[;{}]\s*$|.*\)\s*:\s*$|"
    r"[\w.\[\]]+\s*[-+*/]?=\s*\S)"
)

# Equations, comparisons and math symbols (unverified derivations are treated like unverified code).
MATH = re.compile(r"[=≠≤≥≈∑∫√π∞±\u00d7\u00f7^]|\\(?:frac|sum|int|sqrt)|\b[a-z]\(\w\)|\d\s*[-+*/]\s*\d")


def code_reason(text: str, role: str, cfg: dict[str, Any]) -> str | None:
    c = cfg["code"]
    if "```" in text or "~~~" in text:
        return "fenced_code"
    if sum(1 for line in text.splitlines() if CODE_LINE.match(line)) >= c["max_code_lines"]:
        return "code_lines"
    if role == "assistant":
        if len(re.findall(r"`[^`\n]+`", text)) >= c["max_inline_code"]:
            return "inline_code"
        if len(MATH.findall(text)) >= c["max_math"]:
            return "math"
    return None


WORD = re.compile(r"\b[\w'-]+\b")
NUMBER = re.compile(r"(?<![\w.])\d[\d,]*(?:\.\d+)?")
YEAR = re.compile(r"\b(?:1[5-9]\d\d|20\d\d)s?\b")
LIST_MARKER = re.compile(r"^\s*\d+[.)]\s", re.M)
CAPITAL = re.compile(r"\b(?:[A-Z][a-z]+|[A-Z]{2,})\b")
CITATION = re.compile(
    r"\[\d+\]|\bet al\b|\bdoi\b|\bISBN\b|\(\s*[A-Z][A-Za-z&.,\s]{1,40},?\s*(?:1[5-9]|20)\d\d\s*\)"
    r"|^\s*(?:references|sources?|citations?|bibliography)\s*:",
    re.M | re.I,
)
NOT_PROPER = frozenset({"I", "I'm", "I'd", "I'll", "I've", "OK", "AI"})


@dataclass
class ClaimScore:
    score: float
    features: dict[str, int]


def claim_score(text: str, rules: Rules, cfg: dict[str, Any]) -> ClaimScore:
    """Declared claim-density heuristic: weighted count of numbers, years, mid-sentence capitalized
    words, citation markers, claim phrases and URLs per 100 words (denominator at least ``min_words``)."""
    w = cfg["claims"]["weights"]
    body = LIST_MARKER.sub(" ", text)
    proper = 0
    for mt in CAPITAL.finditer(body):
        before = body[: mt.start()].rstrip()
        if mt.group() not in NOT_PROPER and before and before[-1] not in ".!?:\n\"'*#-(":
            proper += 1
    years = len(YEAR.findall(body))
    feats = {
        "numbers": max(0, len(NUMBER.findall(body)) - years),
        "years": years,
        "proper": proper,
        "citations": len(CITATION.findall(text)),
        "phrases": len(rules.claim_phrases.findall(text)),
        "urls": len(URL.findall(text)),
    }
    words = len(WORD.findall(text))
    raw = sum(w[k] * v for k, v in feats.items())
    return ClaimScore(100.0 * raw / max(words, int(cfg["claims"]["min_words"])), feats)


def advice_domains(text: str, rules: Rules, min_terms: int) -> list[str]:
    out = []
    for name, rx in rules.advice.items():
        if len({m.group().lower() for m in rx.finditer(text)}) >= min_terms:
            out.append(name)
    return out


def temporal_reason(text: str, rules: Rules, cfg: dict[str, Any]) -> str | None:
    """Statements presented as current that were only current when OASST1 was written (to 2023-04)."""
    t = cfg["temporal"]
    for sentence in re.split(r"(?<=[.!?\n])\s+", text):
        years = [int(y[:4]) for y in YEAR.findall(sentence)]
        if any(y > t["future_after"] for y in years):
            return "future_year"
        if rules.recency.search(sentence) and any(y >= t["recent_from"] for y in years):
            return "dated_current_claim"
    return None


def classify(prompt: str) -> str:
    p = prompt.lower()
    if re.search(
        r"\b(write|compose|draft|poem|story|essay|lyrics|haiku|limerick|letter|email|rewrite|rephrase|"
        r"summari[sz]e|paraphrase|slogan|tweet|joke|song|script|dialogue)\b",
        p,
    ):
        return "writing"
    if re.search(
        r"\b(should i|how (can|do|should) i|advice|tips?|recommend|suggest|help me|ideas? for)\b", p
    ):
        return "advice"
    if re.search(r"\b(explain|what (is|are|does)|why|how (does|do|is|are)|difference between|meaning)\b", p):
        return "explanation"
    return "conversation"


def flagged_authors(rows: list[Msg], rules: Rules, cfg: dict[str, Any]) -> set[str]:
    """Authors whose assistant messages (any language) carry strong self-identification markers at
    least ``author_min_hits`` times, or at a rate of at least ``author_min_rate``."""
    mo = cfg["model_origin"]
    total: Counter[str] = Counter()
    hits: Counter[str] = Counter()
    for r in rows:
        if r["role"] == "assistant":
            total[r["user_id"]] += 1
            if rules.author_flag.search(r["text"] or ""):
                hits[r["user_id"]] += 1
    return {
        u
        for u, h in hits.items()
        if h >= int(mo["author_min_hits"]) or h / total[u] >= float(mo["author_min_rate"])
    }


# --- trees and paths -------------------------------------------------------------------------------


@dataclass
class Forest:
    by_id: dict[str, Msg]
    children: dict[str, list[Msg]]
    roots: list[Msg]


def forest(rows: list[Msg]) -> Forest:
    by_id = {r["message_id"]: r for r in rows}
    children: dict[str, list[Msg]] = defaultdict(list)
    roots = []
    for r in rows:
        if r["parent_id"] is None:
            roots.append(r)
        else:
            children[r["parent_id"]].append(r)
    for v in children.values():
        v.sort(key=lambda m: m["message_id"])
    roots.sort(key=lambda m: m["message_tree_id"])
    return Forest(by_id, dict(children), roots)


def _rank0_reply(prompter: Msg, f: Forest, cfg: dict[str, Any]) -> tuple[Msg | None, Fail | None]:
    ranked = [c for c in f.children.get(prompter["message_id"], []) if c["role"] == "assistant"]
    top = [c for c in ranked if c.get("rank") == cfg["tree"]["assistant_rank"]]
    if not top:
        return None, Fail("structure", "no_rank0_reply" if ranked else "no_reply")
    reply = top[0]
    return (None, bad) if (bad := structural(reply, cfg, "assistant")) else (reply, None)


def select_path(root: Msg, f: Forest, cfg: dict[str, Any]) -> tuple[list[Pair], Fail | None]:
    """Prompter -> rank-0 reply -> (best follow-up under that reply) -> its rank-0 reply -> ...

    Among follow-ups under a rank-0 reply only those that pass the structural checks and have a
    structurally valid rank-0 reply qualify; the one with the highest prompter quality label (then
    the lowest message id) is taken. Returns the pairs and, when there are none, the reason."""
    if bad := structural(root, cfg, "prompter"):
        return [], bad
    pairs: list[Pair] = []
    prompter = root
    while True:
        reply, stop = _rank0_reply(prompter, f, cfg)
        if reply is None:
            return pairs, stop
        pairs.append((prompter, reply))
        options = []
        for c in f.children.get(reply["message_id"], []):
            if c["role"] != "prompter" or structural(c, cfg, "prompter"):
                continue
            nxt, _ = _rank0_reply(c, f, cfg)
            if nxt is not None:
                options.append((-labels(c).get("quality", 0.0), c["message_id"], c))
        if not options:
            return pairs, None
        prompter = min(options, key=lambda o: (o[0], o[1]))[2]


# --- assembly --------------------------------------------------------------------------------------


@dataclass
class Candidate:
    tree_id: str
    pairs: list[Pair]
    trims: list[Fail] = field(default_factory=list)
    claim_scores: list[float] = field(default_factory=list)
    system: list[Turn] = field(default_factory=list)
    tokens: int = 0
    target_tokens: int = 0
    assistant_tokens: list[int] = field(default_factory=list)


def turns_of(c: Candidate, pairs: list[Pair] | None = None) -> list[Turn]:
    out = list(c.system)
    for p, a in c.pairs if pairs is None else pairs:
        out.append(Turn("user", p["text"].strip()))
        out.append(Turn("assistant", a["text"].strip(), dict(OPEN_SPEC)))
    return out


def messages_of(turns: list[Turn]) -> list[Message]:
    """Same mapping as the dataset builder: each user message opens a turn; assistant -> final."""
    msgs, turn = [], 0
    for t in turns:
        if t.role == "user":
            turn += 1
        msgs.append(Message(t.role, t.text, "final" if t.role == "assistant" else None, turn))
    return msgs


def render(tok: Any, turns: list[Turn]) -> tuple[int, int, list[int]]:
    """Total tokens, trained tokens and trained tokens per assistant turn (juniper-chat-v1)."""
    enc = serialize(tok, messages_of(turns))
    total, trained = len(enc.ids), sum(1 for x in enc.labels if x != -100)
    per = []
    for t in turns:
        if t.role == "assistant":
            per.append(len(tok.encode(t.text, add_special_tokens=False).ids) + 2)  # <|final|> ... <|eot|>
    return total, trained, per


def split_of(tree_id: str, cfg: dict[str, Any]) -> str:
    s = cfg["split"]
    u = unit_interval(s["salt"], tree_id)
    return "train" if u < s["train"] else "dev" if u < s["train"] + s["dev"] else "test"


def _norm(text: str) -> str:
    return " ".join(text.casefold().split())


@dataclass
class Result:
    items: dict[str, list[Item]]
    ledger: list[dict[str, Any]]
    stats: dict[str, Any]
    near: list[dict[str, Any]]  # rejected near-threshold conversations (review sampling only)


def _ledger_row(tree_id: str, fail: Fail, kind: str, pair: Pair | None, kept_pairs: int) -> dict[str, Any]:
    return {
        "family": "chat.oasst",
        "key_sha256": sha256_bytes(f"oasst:{tree_id}".encode()),
        "tree_id": tree_id,
        "source_id": SOURCE_ID,
        "source_record": ">".join(m["message_id"] for m in pair) if pair else None,
        "split": None,
        "kind": kind,  # "reject": whole tree dropped; "trim": this pair and everything after it dropped
        "stage": fail.stage,
        "reason": f"{fail.stage}:{fail.reason}",
        "detail": fail.detail,
        "margin": None if fail.margin is None else round(fail.margin, 4),
        "pairs_kept": kept_pairs,
    }


def build(cfg_path: Path, root: Path, tokenizer: Any | None = None) -> Result:
    cfg = load_config(cfg_path)
    src = cfg["source"]
    files = fetch(cfg, root)
    rows = read_messages([files[r] for r in src["files"]])
    if tokenizer is None:
        tok_path = root / cfg["render"]["tokenizer"]
        if sha256_file(tok_path) != cfg["render"]["tokenizer_sha256"]:
            raise RuntimeError("tokenizer bytes differ from the pinned SHA-256")
        tokenizer = jtok.load(tok_path.read_text())
    rules = Rules.from_config(cfg)
    f = forest(rows)
    ledger: list[dict[str, Any]] = []
    near: list[dict[str, Any]] = []
    funnel: dict[str, dict[str, int]] = {}

    def snap(name: str, cands: list[Candidate]) -> None:
        funnel[name] = {
            "trees": len(cands),
            "pairs": sum(len(c.pairs) for c in cands),
            "multi_turn": sum(len(c.pairs) > 1 for c in cands),
        }

    funnel["raw"] = {"trees": len(f.roots), "messages": len(rows)}
    ready = [r for r in f.roots if r["tree_state"] == cfg["tree"]["tree_state"]]
    funnel["ready_for_export"] = {"trees": len(ready)}
    english = [r for r in ready if r["lang"] == cfg["tree"]["lang"]]
    funnel["english_root"] = {
        "trees": len(english),
        "messages": sum(1 for r in rows if r["lang"] == cfg["tree"]["lang"]),
    }

    flagged = flagged_authors(rows, rules, cfg)

    cands: list[Candidate] = []
    for r in english:
        pairs, stop = select_path(r, f, cfg)
        if not pairs:
            assert stop is not None
            ledger.append(_ledger_row(r["message_tree_id"], stop, "reject", None, 0))
            continue
        cands.append(Candidate(r["message_tree_id"], pairs))
    snap("structure", cands)

    def pair_filter(stage: str, check: Callable[[Candidate, int, Pair], Fail | None]) -> None:
        nonlocal cands
        kept = []
        for c in cands:
            for i, pair in enumerate(c.pairs):
                bad = check(c, i, pair)
                if bad is None:
                    continue
                kind = "reject" if i == 0 else "trim"
                ledger.append(_ledger_row(c.tree_id, bad, kind, pair, i))
                if kind == "reject" and bad.margin is not None:
                    near.append({"tree_id": c.tree_id, "fail": bad, "turns": turns_of(c, c.pairs[: i + 1])})
                c.pairs = c.pairs[:i]
                c.trims.append(bad)
                break
            if c.pairs:
                kept.append(c)
        cands = kept
        snap(stage, cands)

    def by_labels(_c: Candidate, _i: int, pair: Pair) -> Fail | None:
        return label_fail(pair[0], cfg) or label_fail(pair[1], cfg)

    def by_model(_c: Candidate, _i: int, pair: Pair) -> Fail | None:
        p, a = pair
        for m, role in ((p, "prompter"), (a, "assistant")):
            if hit := rules.model_any.search(m["text"]):
                return Fail("model_origin", f"{role}_marker", detail=hit.group()[:60])
        if hit := rules.model_assistant.search(a["text"]):
            return Fail("model_origin", "assistant_marker", detail=hit.group()[:60])
        if hit := rules.prompter_identity.search(p["text"]):
            return Fail("model_origin", "prompter_identity_question", detail=hit.group()[:60])
        if a["user_id"] in flagged:
            return Fail("model_origin", "flagged_author")
        return None

    def by_pii(_c: Candidate, _i: int, pair: Pair) -> Fail | None:
        for m, role in zip(pair, ("prompter", "assistant"), strict=True):
            if hits := pii_hits(m["text"]):
                return Fail("pii", f"{role}_{hits[0]}", detail=",".join(hits))
        return None

    def by_code(_c: Candidate, _i: int, pair: Pair) -> Fail | None:
        for m, role in zip(pair, ("prompter", "assistant"), strict=True):
            if why := code_reason(m["text"], role, cfg):
                return Fail("code", f"{role}_{why}")
        return None

    cl = cfg["claims"]

    def by_claims(c: Candidate, i: int, pair: Pair) -> Fail | None:
        p, a = pair
        if dom := advice_domains(p["text"] + "\n" + a["text"], rules, int(cl["advice_min_terms"])):
            return Fail("claims", f"advice_{dom[0]}")
        cs = claim_score(a["text"], rules, cfg)
        if cs.features["citations"] > cl["max_citations"]:
            return Fail("claims", "citations", detail=json.dumps(cs.features))
        if cs.score > cl["heavy"]:
            return Fail("claims", "claim_density", cs.score - cl["heavy"], f"{cs.score:.2f}")
        del c.claim_scores[i:]
        c.claim_scores.append(cs.score)
        return None

    def by_temporal(_c: Candidate, _i: int, pair: Pair) -> Fail | None:
        why = temporal_reason(pair[1]["text"], rules, cfg)
        return Fail("temporal", why) if why else None

    pair_filter("labels", by_labels)
    pair_filter("model_origin", by_model)
    pair_filter("pii", by_pii)
    pair_filter("code", by_code)
    pair_filter("claims", by_claims)
    pair_filter("temporal", by_temporal)

    # Length: optional system prompt (sftgen convention, seeded per tree), then drop whole trailing
    # pairs until the rendered conversation fits.
    rc = cfg["render"]
    max_tokens = int(rc["max_tokens"])
    kept = []
    for c in cands:
        rng = random.Random(int.from_bytes(hashlib.sha256(f"{rc['seed']}:{c.tree_id}".encode()).digest()[:8]))
        c.system = sftgen._system(rng, float(rc["system_probability"]))
        n = len(c.pairs)
        while n:
            total, trained, per = render(tokenizer, turns_of(c, c.pairs[:n]))
            if total <= max_tokens:
                break
            n -= 1
        if n == 0:
            bad = Fail("length", "first_pair_over_budget", total - max_tokens, f"{total}>{max_tokens}")
            ledger.append(_ledger_row(c.tree_id, bad, "reject", c.pairs[0], 0))
            near.append({"tree_id": c.tree_id, "fail": bad, "turns": turns_of(c, c.pairs[:1])})
            continue
        if n < len(c.pairs):
            bad = Fail("length", "trailing_pairs_over_budget")
            ledger.append(_ledger_row(c.tree_id, bad, "trim", c.pairs[n], n))
            c.trims.append(bad)
            c.pairs = c.pairs[:n]
        c.tokens, c.target_tokens, c.assistant_tokens = total, trained, per
        kept.append(c)
    cands = kept
    snap("length", cands)

    # Split by tree; an exact (normalized) first prompt shared by several trees is kept once, in the
    # most held-out split (test > dev > train), then by lowest tree id.
    order = {"test": 0, "dev": 1, "train": 2}
    first: dict[str, list[tuple[int, str, Candidate]]] = defaultdict(list)
    for c in cands:
        first[_norm(c.pairs[0][0]["text"])].append((order[split_of(c.tree_id, cfg)], c.tree_id, c))
    items: dict[str, list[Item]] = {s: [] for s in SPLITS}
    survivors = []
    for group in first.values():
        group.sort(key=lambda g: (g[0], g[1]))
        survivors.append(group[0][2])
        for g in group[1:]:
            row = _ledger_row(g[1], Fail("dedup", "duplicate_first_prompt"), "reject", g[2].pairs[0], 0)
            row["split"] = split_of(g[1], cfg)
            ledger.append(row)
    survivors.sort(key=lambda c: c.tree_id)
    snap("dedup", survivors)
    light = float(cl["light"])
    for c in survivors:
        split = split_of(c.tree_id, cfg)
        multi = len(c.pairs) > 1
        scores = c.claim_scores[: len(c.pairs)]
        tag = "UNVERIFIED_CLAIMS" if max(scores) > light else "CLAIM_LIGHT"
        prompt = c.pairs[0][0]["text"]
        it = Item(
            family="chat.oasst.multi" if multi else "chat.oasst",
            key=f"oasst:{c.tree_id}",
            turns=turns_of(c),
            behaviors=["open_ended", classify(prompt)],
            factuality="NOT_VERIFIED",
            difficulty="medium",
            source_id=SOURCE_ID,
            source_revision=src["revision"],
            source_record=">".join(m["message_id"] for pair in c.pairs for m in pair),
            license=LICENSE,
            origin="human",
            split=split,
            meta={
                "factuality_tag": tag,
                "claim_scores": [round(s, 3) for s in scores],
                "tokens": c.tokens,
                "target_tokens": c.target_tokens,
                "assistant_target_tokens": c.assistant_tokens,
                "trimmed": [f"{t.stage}:{t.reason}" for t in c.trims],
                "rights_entry": src["rights_entry"],
                "transform": VERSION,
            },
            extra_keys=[f"oasst-prompt:{sha256_bytes(_norm(prompt).encode())}"],
        )
        items[split].append(it)
    for row in ledger:
        if row["split"] is None and row["kind"] == "trim":
            row["split"] = split_of(row["tree_id"], cfg)
    stats = statistics(funnel, items, ledger, flagged_authors=len(flagged), max_tokens=max_tokens)
    return Result(items, ledger, stats, near)


def oasst_items(cfg_path: Path, root: Path) -> tuple[dict[str, list[Item]], list[dict[str, Any]]]:
    """Admitted OASST1 conversations per split and the rejection/trim ledger (see ``build``)."""
    res = build(cfg_path, root)
    return res.items, res.ledger


# --- statistics and review sample ------------------------------------------------------------------


def _dist(values: list[int]) -> dict[str, Any]:
    if not values:
        return {"n": 0}
    v = sorted(values)

    def q(p: float) -> int:
        return v[min(len(v) - 1, int(p * (len(v) - 1) + 0.5))]

    return {
        "n": len(v),
        "sum": sum(v),
        "min": v[0],
        "p50": q(0.5),
        "p90": q(0.9),
        "p99": q(0.99),
        "max": v[-1],
    }


def statistics(
    funnel: dict[str, dict[str, int]],
    items: dict[str, list[Item]],
    ledger: list[dict[str, Any]],
    *,
    flagged_authors: int,
    max_tokens: int,
) -> dict[str, Any]:
    allit = [it for s in SPLITS for it in items[s]]
    per_split = {}
    for s in SPLITS:
        its = items[s]
        per_split[s] = {
            "conversations": len(its),
            "multi_turn": sum(it.family == "chat.oasst.multi" for it in its),
            "assistant_turns": sum(1 for it in its for t in it.turns if t.role == "assistant"),
            "tokens": sum(it.meta["tokens"] for it in its),
            "target_tokens": sum(it.meta["target_tokens"] for it in its),
        }
    toks = [it.meta["tokens"] for it in allit]
    buckets = {
        f"<={b}": sum(t <= b for t in toks)
        for b in (256, 512, 1024, 1025, 2048)
        if b <= max_tokens or b == 2048
    }
    return {
        "version": VERSION,
        "funnel": funnel,
        "flagged_authors": flagged_authors,
        "rejections_by_reason": dict(
            sorted(Counter(r["reason"] for r in ledger if r["kind"] == "reject").items())
        ),
        "trims_by_reason": dict(sorted(Counter(r["reason"] for r in ledger if r["kind"] == "trim").items())),
        "per_split": per_split,
        "turn_pairs": dict(sorted(Counter(sum(t.role == "user" for t in it.turns) for it in allit).items())),
        "tokens": _dist(toks),
        "target_tokens": _dist([it.meta["target_tokens"] for it in allit]),
        "assistant_turn_target_tokens": _dist(
            [n for it in allit for n in it.meta["assistant_target_tokens"]]
        ),
        "length_buckets": buckets,
        "behaviors": dict(sorted(Counter(it.behaviors[1] for it in allit).items())),
        "factuality_tag": dict(sorted(Counter(it.meta["factuality_tag"] for it in allit).items())),
        "with_system": sum(it.turns[0].role == "system" for it in allit),
    }


def review_sample(res: Result, cfg: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Blinded, seeded review sample: admitted conversations stratified by (behavior, single/multi)
    and rejected near-threshold conversations (the closest to each reason's threshold). Returns
    (sample, key); the key maps review ids to status and must not be shown to the reviewer first."""
    rc = cfg["review"]
    rng = random.Random(int(rc["seed"]))
    admitted = [it for s in SPLITS for it in res.items[s]]
    strata: dict[str, list[Item]] = defaultdict(list)
    for it in admitted:
        strata[f"{it.behaviors[1]}/{it.family}"].append(it)
    want = int(rc["admitted"])
    names = sorted(strata)
    quota = {n: max(1, round(want * len(strata[n]) / len(admitted))) for n in names}
    while sum(quota.values()) > want:
        quota[max(names, key=lambda n: quota[n])] -= 1
    while sum(quota.values()) < want:
        quota[max(names, key=lambda n: len(strata[n]) - quota[n])] += 1
    picks: list[tuple[str, str, Item | dict[str, Any]]] = []
    for n in names:
        for it in rng.sample(strata[n], min(quota[n], len(strata[n]))):
            picks.append(("admitted", n, it))
    # Rejected: equal quota per stage; within a stage, the closest few per reason. PII-flagged
    # conversations are never copied into the review file.
    by_reason: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in res.near:
        if not r["fail"].reason.endswith("_pii"):
            by_reason[f"{r['fail'].stage}:{r['fail'].reason}"].append(r)
    pools: dict[str, list[tuple[str, dict[str, Any]]]] = defaultdict(list)
    per = int(rc["near_pool_per_reason"])
    for reason in sorted(by_reason):
        group = sorted(by_reason[reason], key=lambda r: (r["fail"].margin, r["tree_id"]))[:per]
        pools[reason.split(":")[0]] += [(reason, r) for r in group]
    left = int(rc["rejected"])
    for done, stage in enumerate(sorted(pools, key=lambda st: len(pools[st]))):
        take = min(len(pools[stage]), left // (len(pools) - done))
        left -= take
        for why, r in rng.sample(pools[stage], take):
            picks.append(("rejected", why, r))
    rng.shuffle(picks)
    sample, key = [], []
    for status, stratum, obj in picks:
        if isinstance(obj, Item):
            turns, tree_id, cause, margin = obj.turns, obj.key.split(":", 1)[1], None, None
        else:
            turns, tree_id = obj["turns"], obj["tree_id"]
            cause, margin = f"{obj['fail'].stage}:{obj['fail'].reason}", obj["fail"].margin
        rid = sha256_bytes(f"{rc['seed']}:{tree_id}".encode())[:16]
        sample.append(
            {
                "review_id": rid,
                "rubric": RUBRIC,
                "turns": [{"role": t.role, "text": t.text} for t in turns],
                "scores": dict.fromkeys(RUBRIC_DIMENSIONS),
                "pass": None,
                "reviewer": None,
                "notes": None,
            }
        )
        key.append(
            {
                "review_id": rid,
                "status": status,
                "stratum": stratum,
                "tree_id": tree_id,
                "reason": cause,
                "margin": None if margin is None else round(margin, 4),
            }
        )
    return sample, key


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"".join(canonical_json(r) + b"\n" for r in rows))
    return sha256_file(path)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m j1data.sftoasst")
    ap.add_argument(
        "--config", type=Path, default=Path(__file__).resolve().parents[2] / "configs/sft_oasst_v1.toml"
    )
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--review", action="store_true", help="write the blinded review sample under the root")
    a = ap.parse_args(argv)
    res = build(a.config, a.root)
    out = dict(res.stats)
    if a.review:
        cfg = load_config(a.config)
        sample, key = review_sample(res, cfg)
        base = a.root / cfg["review"]["dir"]
        out["review"] = {
            "sample": str(base / "review_sample.jsonl"),
            "sample_sha256": _write_jsonl(base / "review_sample.jsonl", sample),
            "key_sha256": _write_jsonl(base / "review_key.jsonl", key),
            "rows": len(sample),
        }
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
