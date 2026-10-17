"""Conversational seed drafts from the pinned local GPT-OSS-20B (j1.convdraft.v1; route B of the contract).

Three steps, in this order:

``python -m j1data.convdraft plan --config configs/conv_seed_v1.toml --out DIR`` writes the seeded slot plan.
Topics are split by a hash into training and confirmation pools before anything is drafted.

``python -m j1data.convdraft confirm --config ... --plan DIR --artifact GGUF --out DIR`` drafts the
conversational confirmation prompts (user turns only, own meta-prompt and seed), dedupes them, requires at
least the configured minimum and seals them: ``confirmation_prompts.sealed.jsonl`` plus ``seal.json``. The
seal digest is also written into the plan directory (one seal per plan) and is committed to this repository
before any training draft; every draft's provenance carries it.

``python -m j1data.convdraft draft --config ... --plan DIR --sealed DIR --artifact GGUF --out DIR`` refuses to
run without a valid seal for the same plan and config, matching the plan's seal record. It drafts every
training slot, runs the automatic checks and writes ``drafts.jsonl`` (passed) and ``rejections.jsonl``
(failed, with reasons). Drafts that overlap a sealed prompt are rejected.

Both drafting steps refuse an endpoint that is not loopback, a GGUF whose bytes are not the pinned teacher
artifact, a port not owned by a local llama-server process, a server whose ``/props`` reports another model
file or build, and a dirty checkout (untracked files included). The served model is re-checked during the
run and the GGUF is re-hashed after it. Nothing here is
admitted: drafts go to independent review (accept or reject only, never edit) and then to a separate
admission build, which owns the 1,024-token limit, near-duplicate detection and contamination checks.
Records are labelled ``teacher/gpt-oss-20b-local`` (rights decision R05); none is ever labelled human.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import tomllib
import urllib.parse
import urllib.request
from collections import deque
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from j1data.hashing import canonical_json, sha256_file, sha256_json, unit_interval
from j1data.teacher import Completion, TeacherConfig, openai_chat

VERSION = "j1.convdraft.v1"
LABEL = "teacher/gpt-oss-20b-local"
RIGHTS_DECISION = "R05"
LOOPBACK = frozenset({"127.0.0.1", "localhost", "::1"})
TEACHER_SYSTEM = (
    "You write realistic example chats for training a small assistant. Follow the instructions exactly."
)

RULES = (
    "Write natural contemporary American English. The assistant is a small, honest personal assistant: it"
    " answers directly, keeps answers short (a median of about 40 words, never over {max_words} words), "
    "follows any requested format or length exactly, never flatters, never opens with filler such as "
    "'Great question', never introduces itself, never cites sources or URLs, and never claims to have "
    "searched, browsed, looked anything up, saved, remembered across chats, sent, scheduled or run "
    "anything. It says it does not know only when the answer really cannot be known. Every fact and every"
    " number must be correct, and the assistant never makes a mistake on purpose. Use no real person's "
    "private details."
)
SYSTEM_LAYER = (
    "Also write a one-sentence system instruction such as an app would give (for example a preferred tone"
    " or length); the assistant follows it without mentioning it."
)

# Typographic quotes, the Unicode minus and spaced en dashes are folded before any check.
_FOLD = str.maketrans({"\u2019": "'", "\u2018": "'", "\u201c": '"', "\u201d": '"', "\u2212": "-"})

BANNED = [
    (
        re.compile(
            r"https?://|www\.|\b[a-z0-9][a-z0-9-]*\.(?:com|org|net|gov|edu|io|co|us|uk|ca|info|app|dev|ai|me)\b",
            re.I,
        ),
        "url",
    ),
    (
        re.compile(
            r"\bas an ai\b|\bi(?: am|'m) (?:just |only )?an? (?:ai|artificial intelligence)\b"
            r"|\b(?:as|i(?: am|'m)) (?:just |only )?an? (?:large |ai )?language model\b"
            r"|\bi(?: am|'m) (?:just |only )?an? (?:computer )?program\b",
            re.I,
        ),
        "identity boilerplate",
    ),
    (
        re.compile(
            r"^\s*(?:(?:oh|ooh|wow|ah),?\s+)?(?:(?:what|that's|that is|such) an? )?(?:great|excellent|good|"
            r"fantastic|wonderful|awesome|terrific|interesting) (?:question|idea|point|choice)"
            r"\s*(?:[!.,;:\u2014\u2013]|-+|$)|^\s*i love (?:this|that) question",
            re.I,
        ),
        "flattery opener",
    ),
]
# Claims that the assistant did, is doing or will do something outside the chat. Verbs that only ever
# mean an outside action (book, email, look up, remind you) match alone; general verbs (add, set, put,
# send, check, ...) match only with an outside object (your calendar, a reminder, online, the forecast),
# so in-chat phrasing such as "I'll add a tip at the end" or "I'll put them in order" passes.
_ADV = r"(?:\s+(?:\w+ly|just|already|now|gone ahead and|go ahead and))*"
_EXT = (
    r"(?:to|on|in|into|from|for) (?:your|the) (?:calendar|reminders?|to-?do list|notes app|contacts|"
    r"cart|inbox|account|phone|log|tracker|app)"
    r"|(?:a|an|the|your) (?:\w+ )?(?:reminder|alarm|timer|appointment|reservation|booking|event)s?\b"
    r"|online|the (?:internet|web)"
    r"|the (?:weather|forecast|news|website|site|prices?|reviews|menu|hours|traffic|score)"
)
_INHERENT = (
    r"search(?:ed|ing)?|googl(?:e|ed|ing)|brows(?:e|ed|ing)|look(?:ed|ing)? (?:it |that |this |them )?up"
    r"|book(?:ed|ing)?|reserv(?:e|ed|ing)|email(?:ed|ing)?|text(?:ed|ing)? (?:him|her|them|you|it)"
    r"|messag(?:e|ed|ing) (?:him|her|them)|remind(?:ed|ing)? you|log(?:ged|ging)? (?:it|that|this|your)"
    r"|call(?:ed|ing)? (?:him|her|them|the (?:restaurant|store|office|doctor|clinic|shop))"
)
_GENERAL = (
    r"(?:(?:add|put|save|stor|set|creat|sen[dt]|plac|mak|log|updat|check|verif|order|schedul|remember)\w*"
    r"|not(?:e|ed|es|ing))"  # not "not": "I'm not able to check the weather" is an honest answer
)
_SUBJ = r"\bI(?:'ve| have|'ll| will|'m| am)?"
ACTION = re.compile(
    rf"{_SUBJ}{_ADV} (?:{_INHERENT})\b"
    rf"|{_SUBJ}{_ADV} {_GENERAL}\b[^.!?\n:]{{0,40}}?\b(?:{_EXT})"
    rf"|\bI(?:'ve| have)?{_ADV} (?:ran the numbers|noted (?:that|it|down)"
    r"|found (?:it |that |this |out )?online)\b"
    r"|\bI(?:'ll| will)\s+remember\b"
    r"(?! (?:that |this |it )?(?:for|during) the rest of (?:this|our) (?:chat|conversation))"
    r"|\bkeep (?:that|this|it) in mind for next time\b"
    r"|\bI (?:just|already) checked\b(?! (?:the |your )?(?:math|numbers|arithmetic|calculation|work))"
    rf"|\blet me{_ADV} (?:{_INHERENT}|pull up|find out|check (?:{_EXT}))\b"
    r"|(?:^|[.!?\u2014\u2013-]\s*)"
    r"(?:setting|adding|booking|scheduling|sending|saving|logging|searching)\b[^.!?\n]{0,40}?\bnow\b"
    r"|\b(?:your|the) (?:\w+ )?(?:reminder|alarm|timer|appointment|reservation|booking|order|email"
    r"|message|text|table|event|workout)s? (?:is|are|was|were|has been|have been|is now|are now) (?:all )?"
    r"(?:set|booked|scheduled|sent|placed|confirmed|created|saved|logged|added|reserved)\b"
    r"|\b(?:reminder|alarm|timer|table|appointment|event) (?:set|booked|scheduled|saved|added|logged)\b"
    r"|\baccording to (?:my|a quick) search\b",
    re.I,
)
# In these buckets the assistant reproduces the user's own text, so an action phrase whose verb the user wrote
# ("I booked the venue") is passed to review as a flag instead of being rejected.
QUOTING_BUCKETS = frozenset({"writing_revision", "summarization"})

_N = r"\$?-?\d[\d,]*(?:\.\d+)?%?"
_OP = r"(?:\s*[+*/\u00d7\u00f7]\s*|\s+[xX]\s+|\s+-\s+|\s+(?:times|multiplied by|divided by|plus|minus|of)\s+)"
_EQ = r"(?:\s*=\s*|\s+(?:is|equals|comes to|makes)\s+)"
EQUATION = re.compile(rf"(?<![\w.:/(])(?P<lhs>{_N}(?:{_OP}{_N})+){_EQ}(?P<rhs>{_N}(?:/\d+)?)(?![\w:])")
_OPS = {
    "+": "+", "plus": "+", "-": "-", "minus": "-", "x": "*", "X": "*", "\u00d7": "*", "*": "*",
    "times": "*", "multiplied by": "*", "/": "/", "\u00f7": "/", "divided by": "/", "of": "of",
}  # fmt: skip


def load_config(path: Path) -> dict[str, Any]:
    return tomllib.loads(path.read_text())


def split_topics(cfg: dict[str, Any]) -> tuple[list[str], list[str]]:
    share = float(cfg["confirmation"]["topic_share"])
    topics = sorted(set(cfg["topics"]["everyday"]))
    confirm = [t for t in topics if unit_interval("j1.convdraft.topic-split", t) < share]
    return [t for t in topics if t not in confirm], confirm


ROUND_SEED_STEP = 7919


def plan(
    cfg: dict[str, Any],
    round_no: int = 1,
    oversample: float | None = None,
    base_plan_sha256: str = "",
    slot_counts: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Round 1 is the registered plan. A later round (a top-up after automatic or review rejections) keeps the
    config, topic split and confirmation slots, so it reuses round 1's seal, and draws new training slots
    from seed + 7919 x (round - 1) with its own slot names and oversample factor."""
    if round_no == 1 and (oversample is not None or base_plan_sha256 or slot_counts):
        raise ValueError("round 1 is the registered plan; it takes no overrides")
    if round_no > 1 and not base_plan_sha256:
        raise ValueError("a later round must name the round-1 plan it extends")
    train_topics, confirm_topics = split_topics(cfg)
    rng = random.Random(int(cfg["draft"]["seed"]) + ROUND_SEED_STEP * (round_no - 1))
    over = float(cfg["draft"]["oversample"]) if oversample is None else oversample
    tag = "" if round_no == 1 else f"r{round_no}-"
    slots = []
    if slot_counts and set(slot_counts) - set(cfg["buckets"]):
        raise ValueError(f"unknown buckets {sorted(set(slot_counts) - set(cfg['buckets']))}")
    for bucket in sorted(cfg["buckets"]):
        spec = cfg["buckets"][bucket]
        # A targeted top-up drafts only the named buckets, with explicit slot counts (others get none).
        n = slot_counts.get(bucket, 0) if slot_counts else round(int(spec["count"]) * over)
        lo, hi = spec["turns"]
        for i in range(n):
            slots.append(
                {
                    "slot": f"{bucket}-{tag}{i:03d}",
                    "bucket": bucket,
                    "topic": rng.choice(train_topics),
                    "user_turns": rng.randint(lo, hi),
                    "system_layer": rng.random() < float(cfg["draft"]["system_layer_share"]),
                    "seed": rng.randrange(1, 2**31),
                }
            )
    crng = random.Random(int(cfg["confirmation"]["seed"]))
    buckets = sorted(cfg["buckets"])
    confirm = [
        {
            "slot": f"confirm-{i:03d}",
            "bucket": buckets[i % len(buckets)],
            "topic": crng.choice(confirm_topics),
            "seed": crng.randrange(1, 2**31),
        }
        for i in range(int(cfg["confirmation"]["candidates"]))
    ]
    out: dict[str, Any] = {
        "schema": VERSION + ".plan",
        "version": cfg["draft"]["version"],
        "train_topics": train_topics,
        "confirm_topics": confirm_topics,
        "slots": slots,
        "confirmation_slots": confirm,
    }
    if round_no > 1:
        out |= {"round": round_no, "oversample": over, "base_plan_sha256": base_plan_sha256}
        if slot_counts:
            out["slot_counts"] = dict(sorted(slot_counts.items()))
    return out


CONFIRM_KEYS = ("train_topics", "confirm_topics", "confirmation_slots")


def meta_prompt(cfg: dict[str, Any], slot: dict[str, Any]) -> str:
    spec = cfg["buckets"][slot["bucket"]]
    parts = [
        f"Write one realistic chat between a user and an assistant about: {slot['topic']}.",
        spec["instruction"],
        f"The user speaks exactly {slot['user_turns']} time(s); the assistant replies after each user turn.",
        RULES.format(max_words=cfg["draft"]["max_response_words"]),
    ]
    if slot["system_layer"]:
        parts.append(SYSTEM_LAYER)
    parts.append(
        'Return only JSON: {"system": "<instruction or empty>", "turns": [{"role": "user", "text": "..."}, '
        '{"role": "assistant", "text": "..."}, ...]}'
    )
    return "\n".join(parts)


def confirm_prompt(cfg: dict[str, Any], slot: dict[str, Any]) -> str:
    spec = cfg["buckets"][slot["bucket"]]
    return "\n".join(
        [
            "Write one realistic first message that a user might send to a personal assistant about: "
            f"{slot['topic']}.",
            "It should fit this kind of request: " + spec["instruction"].split(".")[0] + ".",
            "Write only the user's message, in natural American English, at most 80 words. Return only JSON: "
            '{"user": "..."}',
        ]
    )


def _json(text: str) -> Any:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`").split("\n", 1)[-1]
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("no JSON object")
    return json.loads(text[start : end + 1])


def _norm(text: str) -> str:
    return " ".join(text.translate(_FOLD).lower().split())


def _shingles(text: str, n: int = 8) -> set[str]:
    words = re.findall(r"[a-z0-9']+", _norm(text))
    return {" ".join(words[i : i + n]) for i in range(len(words) - n + 1)}


def _jaccard(a: str, b: str) -> float:
    x, y = set(re.findall(r"[a-z0-9']+", _norm(a))), set(re.findall(r"[a-z0-9']+", _norm(b)))
    return len(x & y) / len(x | y) if x | y else 1.0


def _value(tok: str) -> tuple[float, bool]:
    pct = tok.endswith("%")
    num, _, den = tok.replace("$", "").replace(",", "").rstrip("%").partition("/")
    v = float(num) / float(den) if den else float(num)
    return (v / 100 if pct else v), pct


def _results(lhs: str) -> tuple[float, float] | None:
    """(normal precedence, left to right) values of ``lhs``; None when it is not checkable arithmetic."""
    parts = re.split(f"({_OP})", lhs)
    nums = [_value(p) for p in parts[0::2]]
    ops = [_OPS[" ".join(p.split())] for p in parts[1::2]]
    for i, op in enumerate(ops):
        if op == "of" and not nums[i][1]:
            return None  # "3 of 4" is not arithmetic
        if op in "+-" and (nums[i][1] or nums[i + 1][1]):
            return None  # "80 + 15%" is shorthand for a markup, not a sum
    ops = ["*" if op == "of" else op for op in ops]
    vals = [v for v, _ in nums]

    def apply(a: float, op: str, b: float) -> float:
        if op == "/":
            return a / b if b else float("nan")
        return {"+": a + b, "-": a - b, "*": a * b}[op]

    ltr = vals[0]
    for op, v in zip(ops, vals[1:], strict=True):
        ltr = apply(ltr, op, v)
    terms, signs, cur = [], ["+"], vals[0]
    for op, v in zip(ops, vals[1:], strict=True):
        if op in "*/":
            cur = apply(cur, op, v)
        else:
            terms.append(cur)
            signs.append(op)
            cur = v
    terms.append(cur)
    return sum(t if s == "+" else -t for s, t in zip(signs, terms, strict=True)), ltr


def arithmetic_errors(text: str, quoted: str = "") -> tuple[list[str], list[str]]:
    """(errors, review flags) for every ``a op b [op c ...] = d`` (also "times", "is", "% of", fractions).

    The result must match d at d's own precision (a half is not "close enough"). An equation that is right
    only when read left to right ("2 + 3 x 4 = 20") is an error. Division by zero is an error. An equation
    that also occurs in ``quoted`` (the user's own text, such as a quoted wrong sum) is not checked.
    """
    text = text.translate(_FOLD).replace(" \u2013 ", " - ")
    quoted = " ".join(quoted.translate(_FOLD).replace(" \u2013 ", " - ").lower().split())
    errors: list[str] = []
    pos = 0
    while m := EQUATION.search(text, pos):
        pos = m.start("rhs")
        shown = " ".join(m.group(0).split()).rstrip(",")
        if re.search(rf"(?<![\w.]){re.escape(shown.lower())}(?![\w])", quoted):
            continue
        got = _results(m.group("lhs"))
        if got is None:
            continue
        rhs, pct = _value(m.group("rhs"))
        digits = m.group("rhs").replace(",", "").rstrip("%").partition("/")[0].partition(".")[2]
        exact = "/" in m.group("rhs")
        tol = (1e-9 if exact else 0.5 * 10.0 ** -len(digits) / (100 if pct else 1)) - 1e-12
        prec, _ltr = got
        if not abs(prec - rhs) < max(tol, 1e-9 * max(1.0, abs(rhs))):
            errors.append(f"arithmetic error: {shown}")
    return errors, []


def _stem(word: str) -> str:
    return re.sub(r"(?:ed|ing|s)$", "", word.lower())


_FILLER = frozenset(
    {
        "i",
        "i'll",
        "i've",
        "i'm",
        "have",
        "will",
        "am",
        "just",
        "already",
        "now",
        "gone",
        "go",
        "ahead",
        "and",
        "let",
        "me",
    }
)


def _action_verb(match: str) -> str:
    words = [w for w in re.findall(r"[a-z']+", match.lower()) if w not in _FILLER and not w.endswith("ly")]
    return _stem(words[0]) if words else ""


def check(cfg: dict[str, Any], slot: dict[str, Any], obj: Any) -> tuple[list[str], list[str]]:
    """Automatic checks: (problems, review flags). No problems means the draft passes to review."""
    problems: list[str] = []
    flags: list[str] = []
    if not isinstance(obj, dict) or not isinstance(obj.get("turns"), list):
        return ["malformed JSON shape"], []
    turns = obj["turns"]
    roles = [t.get("role") if isinstance(t, dict) else None for t in turns]
    if not turns or roles != ["user", "assistant"] * (len(turns) // 2):
        problems.append("turns must alternate user/assistant and end with the assistant")
    if len(turns) // 2 != slot["user_turns"]:
        problems.append(f"expected {slot['user_turns']} user turns, got {len(turns) // 2}")
    if not isinstance(obj.get("system", ""), str):
        problems.append("system must be a string")
    elif slot["system_layer"] and not obj.get("system", "").strip():
        problems.append("missing system instruction")
    max_words = int(cfg["draft"]["max_response_words"])
    user_text = " ".join(
        t["text"]
        for t in turns
        if isinstance(t, dict) and t.get("role") == "user" and isinstance(t.get("text"), str)
    )
    user_stems = {_stem(w) for w in re.findall(r"[a-z']+", _norm(user_text))}
    for t in turns:
        text = t.get("text") if isinstance(t, dict) else None
        if not isinstance(text, str) or not text.strip():
            problems.append("empty turn")
            continue
        if t.get("role") != "assistant":
            continue  # user turns may carry false premises and the user's own claims
        folded = text.translate(_FOLD)
        if len(folded.split()) > max_words:
            problems.append(f"assistant turn over {max_words} words")
        for pattern, why in BANNED:
            if pattern.search(folded):
                problems.append(why)
        for m in ACTION.finditer(folded):
            verb = _action_verb(m.group(0))
            if slot["bucket"] in QUOTING_BUCKETS and verb and verb in user_stems:
                flags.append(f"action phrase from the user's text: {m.group(0)}")
            else:
                problems.append("fabricated action claim")
        errors, aflags = arithmetic_errors(folded, user_text)
        problems.extend(errors)
        flags.extend(aflags)
    return sorted(set(problems)), sorted(set(flags))


def _first_user(obj: Any) -> str:
    turns = obj.get("turns") if isinstance(obj, dict) else None
    if isinstance(turns, list) and turns and isinstance(turns[0], dict):
        return _norm(str(turns[0].get("text", "")))
    return ""


def code_identity() -> dict[str, Any]:
    """Commit and file hashes; ``dirty`` also counts untracked files under src and configs."""
    import subprocess

    from j1data.build import code_revision
    from j1data.registry import REPO_ROOT

    here = Path(__file__).resolve().parent
    untracked = subprocess.run(
        [
            "git",
            "-C",
            str(REPO_ROOT),
            "status",
            "--porcelain",
            "--untracked-files=all",
            "--",
            "src",
            "configs",
        ],
        capture_output=True,
        text=True,
    ).stdout.strip()
    rev = code_revision()
    return rev | {
        "dirty": bool(rev["dirty"] or untracked),
        "convdraft.py": sha256_file(here / "convdraft.py"),
        "teacher.py": sha256_file(here / "teacher.py"),
    }


def _get_json(url: str) -> dict[str, Any]:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))  # never through a proxy
    with opener.open(url, timeout=30) as r:
        out: dict[str, Any] = json.load(r)
    return out


def _listener(port: int) -> str:
    """The `ss` line for the local TCP listener on ``port`` (shows the owning process)."""
    import subprocess

    return subprocess.run(
        ["ss", "-ltnpH", f"sport = :{port}"], capture_output=True, text=True, check=True
    ).stdout.strip()


def verify_served(
    teacher: TeacherConfig,
    artifact: Path,
    fetch: Callable[[str], dict[str, Any]] = _get_json,
    listener: Callable[[int], str] = _listener,
) -> dict[str, Any]:
    """The loopback server is a local llama-server process serving ``artifact`` with the pinned build."""
    url = urllib.parse.urlparse(teacher.endpoint)
    if url.hostname not in LOOPBACK:
        raise SystemExit(f"teacher endpoint must be loopback, got {url.hostname!r}")
    owner = listener(url.port or 80)
    if '"llama-server"' not in owner:
        raise SystemExit(f"port {url.port} is not served by a local llama-server process: {owner!r}")
    props = fetch(teacher.endpoint.rstrip("/") + "/props")
    served = props.get("model_path")
    if not isinstance(served, str) or not Path(served).is_absolute():
        raise SystemExit(f"server must report an absolute model_path, got {served!r}")
    if Path(served).resolve() != artifact.resolve():
        raise SystemExit(f"server reports model {served!r}, not {artifact}")
    build = props.get("build_info")
    tag = re.search(r"\bb\d+\b", teacher.runtime)
    if (
        not tag
        or not isinstance(build, str)
        or not re.search(rf"(?<![0-9A-Za-z]){tag.group(0)}(?![0-9])", build)
    ):
        raise SystemExit(
            f"server build {build!r} is not runtime {tag.group(0) if tag else teacher.runtime!r}"
        )
    return {
        "endpoint": teacher.endpoint,
        "served_model_file": Path(served).name,
        "build_info": build,
        "total_slots": props.get("total_slots"),  # server-side batching; outputs can depend on it
    }


def verify_endpoint(
    teacher: TeacherConfig,
    artifact: Path,
    fetch: Callable[[str], dict[str, Any]] = _get_json,
    listener: Callable[[int], str] = _listener,
) -> dict[str, Any]:
    """Refuse anything but the pinned local GPT-OSS-20B served on loopback (rights decision R13)."""
    if sha256_file(artifact) != teacher.artifact_sha256:
        raise SystemExit(f"{artifact}: bytes are not the pinned teacher artifact")
    served = verify_served(teacher, artifact, fetch, listener)
    return served | {"artifact_sha256_verified": teacher.artifact_sha256}


def teacher_for(cfg: dict[str, Any], teacher: TeacherConfig) -> TeacherConfig:
    d = cfg["draft"]
    return replace(
        teacher,
        temperature=float(d["temperature"]),
        top_p=float(d["top_p"]),
        max_tokens=int(d["max_tokens"]),
        reasoning_effort=str(d["reasoning_effort"]),
        system=TEACHER_SYSTEM,
        developer="",
    )


def provenance(
    cfg: dict[str, Any],
    plan_obj: dict[str, Any],
    tcfg: TeacherConfig,
    served: dict[str, Any],
    code: dict[str, Any],
) -> dict[str, Any]:
    return {
        "route": "B",
        "label": LABEL,
        "rights_decision": RIGHTS_DECISION,
        "model": tcfg.model,
        "upstream_revision": tcfg.upstream_revision,
        "artifact_sha256": tcfg.artifact_sha256,
        "runtime": tcfg.runtime,
        "served": served,
        "sampling": {
            "temperature": tcfg.temperature,
            "top_p": tcfg.top_p,
            "max_tokens": tcfg.max_tokens,
            "reasoning_effort": tcfg.reasoning_effort,
        },
        "teacher_system_sha256": sha256_json(tcfg.system),
        "rules_sha256": sha256_json({"rules": RULES, "system_layer": SYSTEM_LAYER}),
        "config_sha256": sha256_json(cfg),
        "plan_sha256": sha256_json(plan_obj),
        "code": code,
        "tool": VERSION,
    }


class _Outage:
    """Stops a run after three endpoint failures in a row instead of rejecting every remaining slot."""

    def __init__(self) -> None:
        self.n = 0

    def __call__(self, failed: bool) -> None:
        self.n = self.n + 1 if failed else 0
        if self.n >= 3:
            raise SystemExit("teacher endpoint failed three times in a row; stopping")


_BAD_DRAFT = (ValueError, KeyError, IndexError, AttributeError, TypeError)


def _in_order(
    chat: Callable[[TeacherConfig, str, int], Completion],
    tcfg: TeacherConfig,
    jobs: Iterable[tuple[str, int]],
    workers: int,
) -> Iterator[Completion | OSError]:
    """Completions for (prompt, seed) jobs, yielded in job order with up to ``workers`` requests in flight.

    Results are consumed in plan order, so dedupe, stopping and rejection decisions are the same as a
    sequential run's. Up to ``workers`` requests may already be in flight when the consumer stops; queued ones
    are cancelled and the results of in-flight ones are discarded.
    """

    def call(job: tuple[str, int]) -> Completion | OSError:
        try:
            return chat(tcfg, job[0], job[1])
        except OSError as e:
            return e

    it = iter(jobs)
    with ThreadPoolExecutor(max(1, workers)) as ex:
        pending = deque(ex.submit(call, job) for _, job in zip(range(max(1, workers)), it, strict=False))
        try:
            while pending:
                yield pending.popleft().result()
                nxt = next(it, None)  # only once the consumer wants more: no request past a stop
                if nxt is not None:
                    pending.append(ex.submit(call, nxt))
        finally:
            for f in pending:
                f.cancel()


RECHECK_EVERY = 25  # slots between checks that the server still serves the pinned model file


def draft_confirmation(
    cfg: dict[str, Any],
    plan_obj: dict[str, Any],
    teacher: TeacherConfig,
    served: dict[str, Any],
    code: dict[str, Any],
    chat: Callable[[TeacherConfig, str, int], Completion] = openai_chat,
    recheck: Callable[[], object] | None = None,
    workers: int = 1,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Draft, dedupe and seal the confirmation prompts. Returns (rows, seal); fails below the minimum."""
    tcfg = teacher_for(cfg, teacher)
    prov = provenance(cfg, plan_obj, tcfg, served, code) | {"concurrent_requests": workers}
    target, minimum = int(cfg["confirmation"]["count"]), int(cfg["confirmation"]["minimum"])
    rows: list[dict[str, Any]] = []
    dropped: dict[str, int] = {}
    outage = _Outage()
    cslots = plan_obj["confirmation_slots"]
    results = _in_order(chat, tcfg, ((confirm_prompt(cfg, x), int(x["seed"])) for x in cslots), workers)
    for i, slot in enumerate(cslots):
        if len(rows) >= target:
            break  # checked before the next result is requested
        out = next(results)
        if recheck and i % RECHECK_EVERY == 0:
            recheck()
        prompt = confirm_prompt(cfg, slot)
        if isinstance(out, OSError):
            outage(True)
            dropped["endpoint error"] = dropped.get("endpoint error", 0) + 1
            continue
        outage(False)
        why = None
        user = ""
        if out.finish_reason != "stop":
            why = f"finish_reason {out.finish_reason or 'missing'}"
        else:
            try:
                value = _json(out.output)["user"]
                if not isinstance(value, str):
                    raise TypeError("user is not a string")
                user = value.strip()
            except _BAD_DRAFT:
                why = "unparseable"
        if not why and not user:
            why = "empty"
        if not why and len(user.split()) > 80:
            why = "over 80 words"
        if not why and any(_norm(user) == _norm(r["user"]) or _jaccard(user, r["user"]) >= 0.7 for r in rows):
            why = "duplicate"
        if why:
            dropped[why] = dropped.get(why, 0) + 1
            continue
        rows.append(
            {
                "slot": slot,
                "user": user,
                "meta_prompt_sha256": sha256_json(prompt),
                "finish_reason": out.finish_reason,
                "raw_output": out.output,
                "reasoning": out.reasoning,
                "provenance": prov,
            }
        )
    if len(rows) < minimum:
        raise SystemExit(f"only {len(rows)} confirmation prompts survived; the minimum is {minimum}")
    seal = {
        "schema": VERSION + ".seal",
        "count": len(rows),
        "dropped": dict(sorted(dropped.items())),
        "rows_sha256": sha256_json(rows),
        "plan_sha256": prov["plan_sha256"],
        "config_sha256": prov["config_sha256"],
        "minimum": minimum,
        "sealed_at": datetime.now(ZoneInfo("America/New_York")).isoformat(timespec="seconds"),
    }
    seal["seal_sha256"] = sha256_json(seal)
    return rows, seal


def load_sealed(
    sealed: Path, plan_obj: dict[str, Any], cfg: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Rows and seal, verified against the plan, the config and the minimum."""
    seal = json.loads((sealed / "seal.json").read_text())
    if sha256_json({k: v for k, v in seal.items() if k != "seal_sha256"}) != seal["seal_sha256"]:
        raise SystemExit("seal.json self-hash mismatch")
    text = (sealed / "confirmation_prompts.sealed.jsonl").read_text()
    rows = [json.loads(x) for x in text.split("\n") if x.strip()]
    if sha256_json(rows) != seal["rows_sha256"] or len(rows) != seal["count"]:
        raise SystemExit("sealed confirmation prompts differ from their seal")
    if plan_obj.get("round", 1) > 1:
        # A top-up round reuses round 1's seal: same base plan, identical topic split and confirmation slots.
        reference = plan(cfg)
        if seal["plan_sha256"] != plan_obj["base_plan_sha256"] or any(
            plan_obj[k] != reference[k] for k in CONFIRM_KEYS
        ):
            raise SystemExit("the seal belongs to another plan")
    elif seal["plan_sha256"] != sha256_json(plan_obj):
        raise SystemExit("the seal belongs to another plan")
    if seal["config_sha256"] != sha256_json(cfg):
        raise SystemExit("the seal belongs to another config")
    if seal["count"] < int(cfg["confirmation"]["minimum"]):
        raise SystemExit("the seal holds fewer prompts than the configured minimum")
    return rows, seal


def draft(
    cfg: dict[str, Any],
    plan_obj: dict[str, Any],
    teacher: TeacherConfig,
    sealed: list[dict[str, Any]],
    seal: dict[str, Any],
    served: dict[str, Any],
    code: dict[str, Any],
    chat: Callable[[TeacherConfig, str, int], Completion] = openai_chat,
    recheck: Callable[[], object] | None = None,
    workers: int = 1,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Draft every training slot. Requires the sealed confirmation prompts, which drafts must not overlap."""
    if not sealed or not seal.get("seal_sha256"):
        raise SystemExit("seal the confirmation prompts before drafting training conversations")
    tcfg = teacher_for(cfg, teacher)
    prov = provenance(cfg, plan_obj, tcfg, served, code) | {
        "concurrent_requests": workers,
        "seal_sha256": seal["seal_sha256"],
        "sealed_at": seal["sealed_at"],
        "drafting_started_at": datetime.now(ZoneInfo("America/New_York")).isoformat(timespec="seconds"),
    }
    sealed_text = [r["user"] for r in sealed]
    sealed_shingles = set().union(*(_shingles(u) for u in sealed_text))
    seen: set[str] = set()
    drafts: list[dict[str, Any]] = []
    rejections: list[dict[str, Any]] = []
    outage = _Outage()
    slots = plan_obj["slots"]
    results = _in_order(chat, tcfg, ((meta_prompt(cfg, x), int(x["seed"])) for x in slots), workers)
    for i, (slot, out) in enumerate(zip(slots, results, strict=False)):
        if recheck and i % RECHECK_EVERY == 0:
            recheck()
        prompt = meta_prompt(cfg, slot)
        rec: dict[str, Any] = {"slot": slot, "meta_prompt_sha256": sha256_json(prompt), "provenance": prov}
        if isinstance(out, OSError):
            outage(True)
            rejections.append({**rec, "reasons": [f"endpoint {type(out).__name__}: {str(out)[:120]}"]})
            continue
        outage(False)
        rec |= {"finish_reason": out.finish_reason, "raw_output": out.output, "reasoning": out.reasoning}
        if out.finish_reason != "stop":
            rejections.append({**rec, "reasons": [f"finish_reason {out.finish_reason or 'missing'}"]})
            continue
        try:
            obj = _json(out.output)
            problems, flags = check(cfg, slot, obj)
        except _BAD_DRAFT as e:
            rejections.append({**rec, "reasons": [f"{type(e).__name__}: {str(e)[:120]}"]})
            continue
        key = _first_user(obj)
        if key and key in seen:
            problems.append("duplicate first user turn")
        seen.add(key)
        users = [
            str(t.get("text", "")) for t in obj["turns"] if isinstance(t, dict) and t.get("role") == "user"
        ]
        if any(_shingles(u) & sealed_shingles for u in users) or any(
            _jaccard(key, s) >= 0.7 for s in sealed_text
        ):
            problems.append("overlaps a sealed confirmation prompt")
        if not problems:
            body = {"system": obj.get("system", ""), "turns": obj["turns"]}
            drafts.append(
                rec | {"conversation": body, "content_sha256": sha256_json(body), "review_flags": flags}
            )
        else:
            rejections.append(
                rec | {"conversation": obj, "reasons": sorted(set(problems)), "review_flags": flags}
            )
    return drafts, rejections


def _rows_bytes(rows: list[dict[str, Any]]) -> bytes:
    return b"".join(canonical_json(r) + b"\n" for r in rows)


def _write(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_bytes(_rows_bytes(rows))


def _publish(files: dict[Path, bytes], last: Path | None = None) -> None:
    """Write every file under a temporary name, then rename them into place (``last`` renamed last)."""
    tmp = {path: path.with_name(path.name + ".partial") for path in files}
    for path, data in files.items():
        tmp[path].write_bytes(data)
    for path in sorted(files, key=lambda q: q == last):
        tmp[path].replace(path)


def seal_committed(digest: str) -> bool:
    """The seal digest appears in a file under docs/sft at the committed HEAD of this repository."""
    import subprocess

    from j1data.registry import REPO_ROOT

    found = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "grep", "-q", "-F", digest, "HEAD", "--", "docs/sft"],
        capture_output=True,
    )
    return found.returncode == 0


def _teacher(endpoint: str | None) -> TeacherConfig:
    from j1data.registry import REPO_ROOT

    t = tomllib.loads((REPO_ROOT / "configs" / "teacher.toml").read_text())["teacher"]
    return TeacherConfig(
        endpoint=endpoint or t["endpoint"],
        model=t["model"],
        upstream_revision=t["upstream_revision"],
        artifact_sha256=t["artifact_sha256"],
        runtime=t["runtime"],
        temperature=0.0,
        top_p=1.0,
        max_tokens=1,
        reasoning_effort="low",
        candidates=1,
        base_seed=0,
        template_version=VERSION,
        system="",
        developer="",
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m j1data.convdraft", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--round", type=int, default=1)
    p.add_argument("--oversample", type=float)
    p.add_argument("--base-plan", type=Path, help="round-1 plan directory (later rounds only)")
    p.add_argument("--slot-counts", help="targeted top-up: bucket=n,... (later rounds only)")
    for name in ("confirm", "draft"):
        s = sub.add_parser(name)
        s.add_argument("--config", type=Path, required=True)
        s.add_argument("--plan", type=Path, required=True)
        s.add_argument(
            "--artifact", type=Path, required=True, help="the pinned GPT-OSS-20B GGUF being served"
        )
        s.add_argument("--out", type=Path, required=True)
        s.add_argument("--endpoint")
        s.add_argument(
            "--workers", type=int, default=1, help="concurrent requests (match llama-server --parallel)"
        )
        if name == "draft":
            s.add_argument("--sealed", type=Path, required=True)
            s.add_argument("--base-plan", type=Path, help="round-1 plan directory, for a top-up round")
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    a.out.mkdir(parents=True, exist_ok=True)
    if a.cmd == "plan":
        base = ""
        if a.round > 1:
            base_obj = json.loads((a.base_plan / "plan.json").read_text())
            if base_obj.get("round", 1) != 1 or any(base_obj[k] != plan(cfg)[k] for k in CONFIRM_KEYS):
                raise SystemExit("the base plan is not round 1 of this config")
            base = sha256_json(base_obj)
        counts = (
            {k: int(v) for k, v in (x.split("=") for x in a.slot_counts.split(","))}
            if a.slot_counts
            else None
        )
        obj = plan(cfg, a.round, a.oversample, base, counts)
        (a.out / "plan.json").write_bytes(canonical_json(obj) + b"\n")
        print(json.dumps({"slots": len(obj["slots"]), "confirmation_slots": len(obj["confirmation_slots"])}))
        return 0
    os.environ["NO_PROXY"] = os.environ["no_proxy"] = "*"  # teacher calls never go through a proxy
    code = code_identity()
    if code["dirty"] or not code["commit"]:
        raise SystemExit("refusing to draft from a dirty or unidentified checkout")
    plan_obj = json.loads((a.plan / "plan.json").read_text())
    teacher = _teacher(a.endpoint)
    served = verify_endpoint(teacher, a.artifact)

    def recheck() -> object:
        return verify_served(teacher, a.artifact)

    # one seal per plan: a re-seal elsewhere cannot go unnoticed; a top-up round uses round 1's record
    if plan_obj.get("round", 1) > 1:
        if a.cmd != "draft" or not a.base_plan:
            raise SystemExit("a top-up round only drafts, and needs --base-plan")
        if sha256_json(json.loads((a.base_plan / "plan.json").read_text())) != plan_obj["base_plan_sha256"]:
            raise SystemExit("--base-plan is not this round's base plan")
    record = (getattr(a, "base_plan", None) or a.plan) / "seal_record.json"
    staged: dict[Path, bytes] = {}
    if a.cmd == "confirm":
        if (a.out / "seal.json").exists() or record.exists():
            raise SystemExit(f"this plan is already sealed ({record})")
        rows, seal = draft_confirmation(
            cfg, plan_obj, teacher, served, code, recheck=recheck, workers=a.workers
        )
        staged[a.out / "confirmation_prompts.sealed.jsonl"] = _rows_bytes(rows)
        staged[a.out / "seal.json"] = canonical_json(seal) + b"\n"
        staged[record] = (
            canonical_json({"seal_sha256": seal["seal_sha256"], "sealed_at": seal["sealed_at"]}) + b"\n"
        )
        result: dict[str, Any] = {
            "confirmation": len(rows),
            "dropped": seal["dropped"],
            "seal": seal["seal_sha256"],
        }
    else:
        sealed, seal = load_sealed(a.sealed, plan_obj, cfg)
        if not record.exists() or json.loads(record.read_text())["seal_sha256"] != seal["seal_sha256"]:
            raise SystemExit("the sealed set is not the one recorded for this plan")
        if not seal_committed(seal["seal_sha256"]):
            raise SystemExit(f"commit the seal digest {seal['seal_sha256']} under docs/sft before drafting")
        drafts, rejections = draft(
            cfg, plan_obj, teacher, sealed, seal, served, code, recheck=recheck, workers=a.workers
        )
        staged[a.out / "drafts.jsonl"] = _rows_bytes(drafts)
        staged[a.out / "rejections.jsonl"] = _rows_bytes(rejections)
        result = {"drafts": len(drafts), "rejected": len(rejections), "seal": seal["seal_sha256"]}
    # Nothing is published until the server and the GGUF are verified once more after the run.
    recheck()
    if sha256_file(a.artifact) != teacher.artifact_sha256:
        raise SystemExit(f"{a.artifact} changed during the run; nothing was written")
    result["artifact_rehashed_after_run"] = True
    staged[a.out / f"{a.cmd}_run.json"] = canonical_json(result | {"served": served, "code": code}) + b"\n"
    _publish(staged, last=record if a.cmd == "confirm" else None)
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
