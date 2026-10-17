"""Compositional instruction families for SFT dataset v1.3 (generalization; computed targets only).

The v1.2 pilot learned its 17 trained instruction templates but transferred to 3 of 272 held-out
instruction items. This module separates three kinds of generalization and reserves part of each for
evaluation only:

* paraphrase: every operation has many instruction phrasings and several prompt frames; about one in
  five phrasings and two of the frames are never used in training (``if.paraphrase`` test family);
* composition: operations chain ("sort these words, then make them uppercase"); a fixed quarter of the
  operation pairs is never trained (``if.compose_heldout``);
* new operations: the six v1.2 held-out operations stay untrained (so their old score remains an
  unseen-family measure) and four more are added (``heldout.if2``).

Targets are computed by the operation functions, upstream text or the single approved phrase
"I don't know." (rights ledger R00/R13): prompt wording is templated and loss-masked, and no
natural-language target sentence is written here. Every split decision comes from the task key.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from j1data.sftgen import (
    ABSTAIN_TARGET,
    ADJECTIVES,
    ANIMALS,
    CITIES,
    COLORS,
    FRUITS,
    JOBS,
    NAMES,
    NOUNS,
    OBJECTS,
    Item,
    Turn,
    _article,
    _pick,
    _system,
)

VERSION = "j1.sftops.v1"
WORD_POOL = sorted(set(NOUNS + ADJECTIVES + FRUITS + ANIMALS + COLORS))


def _h(text: str) -> int:
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:4], "big")


# --- operations on a list of words -------------------------------------------------------------


@dataclass(frozen=True)
class Op:
    name: str
    out: str  # "words" (chainable), "number", "word"
    phrasings: tuple[str, ...]  # "{it}" = reference to the input; parameters as {p}, {q}
    fn: Callable[[list[str], dict[str, Any]], Any]
    params: Callable[[random.Random, list[str]], dict[str, Any]] = lambda _rng, _w: {}
    needs: Callable[[list[str]], bool] = lambda _w: True


def _longer(n: int) -> Callable[[list[str]], list[str]]:
    return lambda w: [x for x in w if len(x) > n]


def _p_word(rng: random.Random, w: list[str]) -> dict[str, Any]:
    return {"p": _pick(rng, w)}


def _p_replace(rng: random.Random, w: list[str]) -> dict[str, Any]:
    return {"p": _pick(rng, w), "q": _pick(rng, [x for x in WORD_POOL if x not in w])}


def _p_k(rng: random.Random, w: list[str]) -> dict[str, Any]:
    return {"p": rng.randint(1, max(1, len(w) - 1))}


def _p_len(rng: random.Random, _w: list[str]) -> dict[str, Any]:
    return {"p": rng.randint(3, 6)}


def _p_letter(rng: random.Random, w: list[str]) -> dict[str, Any]:
    return {"p": _pick(rng, sorted({x[0] for x in w}))}


def _p_nth(rng: random.Random, w: list[str]) -> dict[str, Any]:
    # p >= 2: "the first word" is the separate `first` operation (and one of its phrasings is reserved)
    return {"p": rng.randint(2, len(w))}


def _ordinal(n: int) -> str:
    return {1: "first", 2: "second", 3: "third", 4: "fourth", 5: "fifth", 6: "sixth", 7: "seventh"}.get(
        n, f"{n}th"
    )


TRAIN_OPS: tuple[Op, ...] = (
    Op(
        "upper",
        "words",
        (
            "convert {it} to uppercase",
            "write {it} in all capital letters",
            "make {it} uppercase",
            "change every letter of {it} to a capital letter",
            "turn {it} into upper case",
            "rewrite {it} using only uppercase letters",
            "put {it} in all caps",
            "capitalize every letter of {it}",
            "use capital letters for all of {it}",
            "set {it} in uppercase",
        ),
        lambda w, _p: [x.upper() for x in w],
    ),
    Op(
        "lower",
        "words",
        (
            "convert {it} to lowercase",
            "write {it} in lowercase letters",
            "make {it} all lowercase",
            "change every letter of {it} to lowercase",
            "turn {it} into lower case",
            "rewrite {it} without any capital letters",
            "use only small letters for {it}",
            "set {it} in lowercase",
        ),
        lambda w, _p: [x.lower() for x in w],
        needs=lambda w: any(x != x.lower() for x in w),
    ),
    Op(
        "title",
        "words",
        (
            "capitalize the first letter of each word in {it}",
            "start every word of {it} with a capital letter",
            "write {it} in title case, capitalizing each word",
            "make the first letter of every word in {it} uppercase",
            "give each word of {it} an initial capital",
            "capitalize each word of {it}",
            "uppercase only the first letter of every word in {it}",
            "put a capital letter at the start of each word in {it}",
        ),
        lambda w, _p: [x[:1].upper() + x[1:] for x in w],
    ),
    Op(
        "reverse",
        "words",
        (
            "reverse the order of {it}",
            "write {it} in reverse order",
            "list {it} backwards, last word first",
            "put {it} in the opposite order",
            "flip the order of {it}",
            "rearrange {it} so the last word comes first and the first comes last",
            "reverse {it}",
            "give {it} from the last word to the first",
            "invert the order of {it}",
        ),
        lambda w, _p: list(reversed(w)),
    ),
    Op(
        "sort",
        "words",
        (
            "sort {it} alphabetically",
            "put {it} in alphabetical order",
            "arrange {it} from A to Z",
            "order {it} alphabetically",
            "alphabetize {it}",
            "sort {it} in dictionary order",
            "rearrange {it} into alphabetical order",
            "list {it} in A-to-Z order",
            "put {it} in order from A to Z",
        ),
        lambda w, _p: sorted(w),
    ),
    Op(
        "remove",
        "words",
        (
            "remove the word '{p}' from {it}",
            "delete '{p}' from {it}",
            "take '{p}' out of {it}",
            "drop the word '{p}' from {it}",
            "leave out '{p}' from {it}",
            "write {it} without the word '{p}'",
            "omit '{p}' from {it}",
            "cut the word '{p}' from {it}",
        ),
        lambda w, p: [x for x in w if x != p["p"]],
        _p_word,
        needs=lambda w: len(w) >= 3,
    ),
    Op(
        "replace",
        "words",
        (
            "replace '{p}' with '{q}' in {it}",
            "swap the word '{p}' for '{q}' in {it}",
            "change '{p}' to '{q}' in {it}",
            "substitute '{q}' for '{p}' in {it}",
            "in {it}, use '{q}' instead of '{p}'",
            "put '{q}' where '{p}' is in {it}",
            "exchange '{p}' for '{q}' in {it}",
            "turn every '{p}' in {it} into '{q}'",
        ),
        lambda w, p: [p["q"] if x == p["p"] else x for x in w],
        _p_replace,
    ),
    Op(
        "dedupe",
        "words",
        (
            "remove the duplicate words from {it}, keeping the first occurrence",
            "delete repeated words in {it}, keeping each word's first appearance",
            "keep only the first copy of each word in {it}",
            "drop any word in {it} that already appeared earlier",
            "write {it} with every repeated word removed after its first use",
            "deduplicate {it}, keeping the original order",
        ),
        lambda w, _p: list(dict.fromkeys(w)),
        needs=lambda w: len(set(w)) < len(w),
    ),
    Op(
        "first_k",
        "words",
        (
            "keep only the first {p} words of {it}",
            "write just the first {p} words of {it}",
            "give the first {p} words of {it}",
            "take the first {p} words of {it}",
            "list only the opening {p} words of {it}",
            "show the first {p} words from {it}",
            "cut {it} down to its first {p} words",
        ),
        lambda w, p: w[: p["p"]],
        _p_k,
        needs=lambda w: len(w) >= 3,
    ),
    Op(
        "last_k",
        "words",
        (
            "keep only the last {p} words of {it}",
            "write just the last {p} words of {it}",
            "give the final {p} words of {it}",
            "take the last {p} words of {it}",
            "list only the closing {p} words of {it}",
            "show the last {p} words from {it}",
        ),
        lambda w, p: w[-p["p"] :],
        _p_k,
        needs=lambda w: len(w) >= 3,
    ),
    Op(
        "drop_first",
        "words",
        (
            "remove the first word of {it}",
            "delete the first word from {it}",
            "drop the opening word of {it}",
            "write {it} without its first word",
            "leave out the first word of {it}",
            "skip the first word of {it}",
        ),
        lambda w, _p: w[1:],
        needs=lambda w: len(w) >= 3,
    ),
    Op(
        "drop_last",
        "words",
        (
            "remove the last word of {it}",
            "delete the final word from {it}",
            "drop the last word of {it}",
            "write {it} without its last word",
            "leave out the final word of {it}",
            "skip the last word of {it}",
        ),
        lambda w, _p: w[:-1],
        needs=lambda w: len(w) >= 3,
    ),
    Op(
        "longer_than",
        "words",
        (
            "keep only the words in {it} that are longer than {p} letters",
            "remove every word of {it} with {p} or fewer letters",
            "filter {it} to words with more than {p} letters",
            "list the words from {it} that have more than {p} letters",
            "select the words of {it} longer than {p} letters",
            "drop the words in {it} that have at most {p} letters",
        ),
        lambda w, p: [x for x in w if len(x) > p["p"]],
        _p_len,
    ),
    Op(
        "starts_with",
        "words",
        (
            "keep only the words in {it} that start with '{p}'",
            "list the words from {it} beginning with the letter '{p}'",
            "select the words of {it} whose first letter is '{p}'",
            "filter {it} to words starting with '{p}'",
            "pick out the words in {it} that begin with '{p}'",
        ),
        lambda w, p: [x for x in w if x.startswith(p["p"])],
        _p_letter,
    ),
    Op(
        "count",
        "number",
        (
            "count {it}",
            "how many words are in {it}?",
            "tell me the number of words in {it}",
            "give the word count of {it}",
            "count how many words {it} contains",
            "what is the number of words in {it}?",
            "find how many words there are in {it}",
        ),
        lambda w, _p: len(w),
    ),
    Op(
        "count_letters",
        "number",
        (
            "count the total number of letters in {it}",
            "how many letters are there in {it} altogether?",
            "add up the letters in all of {it}",
            "give the total letter count of {it}",
            "find the combined number of letters in {it}",
        ),
        lambda w, _p: sum(len(x) for x in w),
        needs=lambda w: len(w) <= 5,
    ),
    Op(
        "first",
        "word",
        (
            "what is the first word of {it}?",
            "give the first word of {it}",
            "tell me which word comes first in {it}",
            "write only the first word of {it}",
            "name the opening word of {it}",
            "which word starts {it}?",
        ),
        lambda w, _p: w[0],
    ),
    Op(
        "last",
        "word",
        (
            "what is the last word of {it}?",
            "give the final word of {it}",
            "tell me which word comes last in {it}",
            "write only the last word of {it}",
            "name the closing word of {it}",
            "which word ends {it}?",
        ),
        lambda w, _p: w[-1],
    ),
    Op(
        "nth",
        "word",
        (
            "what is the {p_ord} word of {it}?",
            "give the {p_ord} word in {it}",
            "write only the {p_ord} word of {it}",
            "tell me the {p_ord} word of {it}",
            "which word is in position {p} in {it}?",
        ),
        lambda w, p: w[p["p"] - 1],
        _p_nth,
        needs=lambda w: len(w) >= 2,
    ),
    Op(
        "longest",
        "word",
        (
            "which word in {it} is the longest? If several tie, give the first of them",
            "find the longest word in {it} (the first one if there is a tie)",
            "give the word with the most letters in {it}; on a tie, the earliest",
            "name the longest word of {it}, choosing the first on a tie",
        ),
        lambda w, _p: max(w, key=len),
    ),
    Op(
        "shortest",
        "word",
        (
            "which word in {it} is the shortest? If several tie, give the first of them",
            "find the shortest word in {it} (the first one if there is a tie)",
            "give the word with the fewest letters in {it}; on a tie, the earliest",
            "name the shortest word of {it}, choosing the first on a tie",
        ),
        lambda w, _p: min(w, key=len),
    ),
)

# Unseen operations (evaluation only). The first six are the v1.2 held-out instructions.
HELDOUT_OPS: tuple[Op, ...] = (
    Op(
        "reverse_sort",
        "words",
        ("sort {it} in reverse alphabetical order",),
        lambda w, _p: sorted(w, reverse=True),
    ),
    Op(
        "every_other",
        "words",
        ("write every other word of {it}, starting with the first",),
        lambda w, _p: w[::2],
    ),
    Op(
        "last_letters",
        "words",
        ("write the last letter of each word of {it}",),
        lambda w, _p: [x[-1] for x in w],
    ),
    Op(
        "count_vowels",
        "number",
        ("how many of the vowels a, e, i, o and u are in {it} altogether?",),
        lambda w, _p: sum(c in "aeiou" for x in w for c in x),
        needs=lambda w: len(w) <= 4,
    ),
    Op(
        "swap_first_last",
        "words",
        ("swap the first and last words of {it}",),
        lambda w, _p: [w[-1], *w[1:-1], w[0]],
    ),
    Op(
        "word_after",
        "word",
        ("which word comes right after '{p}' in {it}?",),
        lambda w, p: w[w.index(p["p"]) + 1],
        lambda rng, w: {"p": _pick(rng, w[:-1])},
    ),
    Op(
        "acronym",
        "word",
        ("make an acronym from the first letters of {it}, in capitals",),
        lambda w, _p: "".join(x[0] for x in w).upper(),
    ),
    Op(
        "sort_by_length",
        "words",
        ("sort {it} from shortest to longest, keeping the original order for equal lengths",),
        lambda w, _p: sorted(w, key=len),
    ),
    Op(
        "duplicate_each",
        "words",
        ("write each word of {it} twice in a row",),
        lambda w, _p: [y for x in w for y in (x, x)],
    ),
    Op(
        "position_of",
        "number",
        ("at which position (counting from 1) does the word '{p}' appear in {it}?",),
        lambda w, p: w.index(p["p"]) + 1,
        _p_word,
    ),
)
V12_HELDOUT = ("reverse_sort", "every_other", "last_letters", "count_vowels", "swap_first_last", "word_after")
OPS = {op.name: op for op in TRAIN_OPS}
HOPS = {op.name: op for op in HELDOUT_OPS}


# Phrasings that (near-)duplicate a v1.2 training template ("How many words are in this sentence?",
# "Rewrite the following in all capital letters"): never reserved, since training already contains them.
SHARED_WITH_V12 = frozenset(
    {("count", "how many words are in {it}?"), ("upper", "write {it} in all capital letters")}
)


def reserved_phrasing(op: Op, i: int) -> bool:
    """About one phrasing in five per operation is evaluation-only (never the first)."""
    if (op.name, op.phrasings[i]) in SHARED_WITH_V12:
        return False
    return i > 0 and _h(f"phr:{op.name}:{op.phrasings[i]}") % 5 == 0


# Ordered pairs whose composition equals a held-out operation (sort then reverse = reverse_sort).
EQUIVALENT_TO_HELDOUT_OP = frozenset({("sort", "reverse")})


def heldout_pair(a: str, b: str) -> bool:
    """A quarter of the ordered operation pairs is evaluation-only, plus every pair equivalent to a held-out
    operation. Any training use of two operations in sequence (compositions, chained conversation turns) must
    avoid these pairs."""
    return (a, b) in EQUIVALENT_TO_HELDOUT_OP or _h(f"pair:{a}>{b}") % 4 == 0


# --- presentation: input display, output format, frames ----------------------------------------

DISPLAYS = ("spaces", "commas", "lines")
FORMATS: dict[str, tuple[str, ...]] = {  # format -> instruction suffixes stating it
    "spaces": ("Separate the words with single spaces.", "Write them on one line separated by spaces."),
    "commas": ("Separate the words with commas.", "Use a comma and a space between words."),
    "lines": ("Put each word on its own line.", "Write one word per line."),
    "numbered": ("Give the result as a numbered list.", "Number each word, one per line, like 1. word."),
    "bullets": (
        "Give the result as a bulleted list with dashes.",
        "Write each word on its own line after '- '.",
    ),
    "json": ("Answer with a JSON array of strings.", "Return the words as a JSON list."),
}


def show(words: list[str], how: str) -> str:
    return {"spaces": " ".join, "commas": ", ".join, "lines": "\n".join}[how](words)


def render(words: list[str], fmt: str) -> str:
    if fmt in ("spaces", "commas", "lines"):
        return show(words, fmt)
    if fmt == "numbered":
        return "\n".join(f"{i}. {x}" for i, x in enumerate(words, 1))
    if fmt == "bullets":
        return "\n".join(f"- {x}" for x in words)
    return json.dumps(words)


def spec_for(value: Any, out: str, fmt: str) -> dict[str, Any]:
    if out == "number":
        return {"type": "number", "value": str(value)}
    if out == "word":
        return {"type": "exact", "value": str(value), "strip_period": True}
    if fmt == "json":
        return {"type": "json", "value": value}
    return {"type": "exact", "value": render(value, fmt)}


# Frames place the instruction and the words; ``it`` is how the instruction refers to the input.
# Frames 0-7 are used in training; 8 and 9 are evaluation-only. Question-form instructions ("which
# word ...?") only go into frames that do not wrap them in an imperative ("Please ...").
@dataclass(frozen=True)
class Frame:
    text: str
    it: str
    questions: bool
    colon: bool = False  # the words follow the instruction on the same line


FRAMES: tuple[Frame, ...] = (
    Frame("{Instr}{p} {x}", "these words", True, colon=True),
    Frame("{Instr}{p}\n\n{x}", "the following words", True),
    Frame("{x}\n\n{Instr}{p}", "the words above", True),
    Frame("Please {instr}: {x}", "these words", False),
    Frame("Could you {instr}?\n{x}", "these words", False),
    Frame("I need you to {instr}.\nWords: {x}", "the words", False),
    Frame("Task: {Instr}{p}\nInput: {x}", "the input", True),
    Frame("{Instr}, please.\n```\n{x}\n```", "the words in the block", False),
    Frame("Here are some words: {x}\n{Instr}{p}", "them", True),
    Frame('Words: "{x}"\nInstruction: {instr}', "the quoted words", True),
)
TRAIN_FRAMES = tuple(range(8))
TEST_FRAMES = (8, 9)


def _cap(s: str) -> str:
    return s[:1].upper() + s[1:]


def _clause(op: Op, i: int, it: str, params: dict[str, Any]) -> str:
    fill = {"it": it, **params}
    if "p" in params and isinstance(params["p"], int):
        fill["p_ord"] = _ordinal(params["p"])
    return op.phrasings[i].format(**fill)


def _frame(frame: Frame, instr: str, x: str) -> str:
    question = instr.endswith("?")
    body = instr.rstrip("?")
    end = "?" if question else (":" if frame.colon else ".")
    return frame.text.format(Instr=_cap(body), instr=body, x=x, p=end)


def _words(rng: random.Random, op: Op, lo: int = 3, hi: int = 7) -> list[str]:
    for _ in range(200):
        w = rng.sample(WORD_POOL, rng.randint(lo, hi))
        if op.name == "dedupe":
            w = w + rng.sample(w, rng.randint(1, 2))
            rng.shuffle(w)
        if op.name == "lower":
            w = [_pick(rng, (x.upper(), x.title(), x)) for x in w]
        if op.needs(w):
            return w
    raise RuntimeError(f"no input for {op.name}")


def _pick_phrasing(rng: random.Random, op: Op, reserved: bool) -> int:
    pool = [i for i in range(len(op.phrasings)) if reserved_phrasing(op, i) == reserved]
    return _pick(rng, pool)


def _single(rng: random.Random, op: Op, *, test: bool, family: str) -> Item:
    w = _words(rng, op)
    params = op.params(rng, w)
    display = _pick(rng, DISPLAYS)
    has_reserved = any(reserved_phrasing(op, i) for i in range(len(op.phrasings)))
    mode = rng.randrange(3) if test else -1  # test: reserved phrasing (0), reserved frame (1), both (2)
    reserved_phr = test and has_reserved and mode != 1
    i = _pick_phrasing(rng, op, reserved_phr)
    question = op.phrasings[i].endswith("?")
    frames = TEST_FRAMES if test and (mode != 0 or not reserved_phr) else TRAIN_FRAMES
    frame = _pick(rng, [f for f in frames if FRAMES[f].questions or not question])
    instr = _clause(op, i, FRAMES[frame].it, params)
    value = op.fn(w, params)
    fmt = display
    user = _frame(FRAMES[frame], instr, show(w, display))
    if op.out == "words" and rng.random() < 0.4:
        fmt = _pick(rng, sorted(FORMATS))
        user += ("\n" if "\n" in user else " ") + _pick(rng, FORMATS[fmt])
    target = str(value) if op.out != "words" else render(value, fmt)
    turns = [*_system(rng), Turn("user", user), Turn("assistant", target, spec_for(value, op.out, fmt))]
    key = f"op:{op.name}:{json.dumps(params, sort_keys=True)}:{' '.join(w)}"
    meta = {"ops": [op.name], "frame": frame, "phrasing": i, "reserved_phrasing": reserved_phr}
    return Item(family, key, turns, ["direct_answer", "no_tool"], meta=meta)


def gen_op(rng: random.Random) -> Item:
    op = _pick(rng, TRAIN_OPS)
    return _single(rng, op, test=False, family=f"if.op.{op.name}")


def gen_paraphrase_test(rng: random.Random) -> Item:
    op = _pick(rng, TRAIN_OPS)
    return _single(rng, op, test=True, family=f"if.paraphrase.{op.name}")


def gen_heldout_op(rng: random.Random) -> Item:
    op = _pick(rng, HELDOUT_OPS)
    fam = "heldout.ops.v12" if op.name in V12_HELDOUT else "heldout.ops.new"
    item = _single(rng, op, test=False, family=f"{fam}.{op.name}")
    item.key = "h" + item.key
    return item


COMPOSE = (
    ("{A}, then {b}", "the result"),
    ("First {a}. Then {b}", "the result"),
    ("{A} and after that {b}", "the result"),
    ("Do two steps: 1) {a}; 2) {b}", "the result of step 1"),
    ("{A}. Next, {b}", "those words"),
    ("{A}; once that is done, {b}", "what you get"),  # evaluation-only
    ("Step one: {a}. Step two: {b}", "the output of step one"),  # evaluation-only
)
TRAIN_COMPOSE = (0, 1, 2, 3, 4)
TEST_COMPOSE = (5, 6)


def _compose(rng: random.Random, *, heldout: bool) -> Item:
    for _ in range(1000):
        a = _pick(rng, [o for o in TRAIN_OPS if o.out == "words"])
        b = _pick(rng, TRAIN_OPS)
        if a.name == b.name or heldout_pair(a.name, b.name) != heldout:
            continue
        w = _words(rng, a, 4, 7)
        pa = a.params(rng, w)
        mid = a.fn(w, pa)
        if not mid or not b.needs(mid) or mid == w:
            continue
        pb = b.params(rng, mid)
        value = b.fn(mid, pb)
        break
    else:
        raise RuntimeError("no composition")
    display = _pick(rng, DISPLAYS)
    ci = _pick(rng, TEST_COMPOSE if heldout and rng.random() < 0.3 else TRAIN_COMPOSE)
    tmpl, it2 = COMPOSE[ci]
    ia = _clause(a, _pick_phrasing(rng, a, False), "these words", pa).rstrip("?")
    b_imperative = [
        i for i in range(len(b.phrasings)) if not reserved_phrasing(b, i) and not b.phrasings[i].endswith("?")
    ]
    ib = _clause(b, _pick(rng, b_imperative), it2, pb)
    instr = tmpl.format(A=_cap(ia), a=ia, b=ib)
    fmt = display
    if b.out == "words" and rng.random() < 0.4:
        fmt = _pick(rng, sorted(FORMATS))
        instr += ". " + _pick(rng, FORMATS[fmt])[:-1]
    user = f"{instr}.\n\n{show(w, display)}" if rng.random() < 0.5 else f"{instr}: {show(w, display)}"
    target = str(value) if b.out != "words" else render(value, fmt)
    turns = [*_system(rng), Turn("user", user), Turn("assistant", target, spec_for(value, b.out, fmt))]
    family = "if.compose_heldout" if heldout else "if.compose"
    key = f"cmp:{a.name}>{b.name}:{json.dumps([pa, pb], sort_keys=True)}:{' '.join(w)}"
    return Item(family, key, turns, ["direct_answer", "no_tool"], meta={"ops": [a.name, b.name], "frame": ci})


def gen_compose(rng: random.Random) -> Item:
    return _compose(rng, heldout=False)


def gen_compose_heldout(rng: random.Random) -> Item:
    return _compose(rng, heldout=True)


# --- extraction and formatting -----------------------------------------------------------------


def _people(rng: random.Random, n: int) -> list[dict[str, Any]]:
    names = rng.sample(NAMES, n)
    return [
        {"name": x, "age": rng.randint(18, 80), "city": _pick(rng, CITIES), "job": _pick(rng, JOBS)}
        for x in names
    ]


def gen_format(rng: random.Random) -> Item:
    people = _people(rng, rng.randint(2, 4))
    cols = rng.sample(["name", "age", "city", "job"], rng.randint(2, 3))
    if "name" in cols:
        cols.remove("name")
    cols = ["name", *cols]
    lines = [
        f"{p['name']} is {p['age']} years old, works as {_article(p['job'])} {p['job']}"
        f" and lives in {p['city']}."
        for p in people
    ]
    text = " ".join(lines)
    kind = rng.randrange(3)
    header = [c.capitalize() for c in cols]
    rows = [[str(p[c]) for c in cols] for p in people]
    if kind == 0:
        user = _pick(
            rng,
            (
                f"Make a Markdown table with the columns {', '.join(header)} from this text:\n\n{text}",
                f"{text}\n\nPut this information in a Markdown table with columns {', '.join(header)}.",
                f"Create a table (Markdown) listing {', '.join(header)} for each person: {text}",
            ),
        )
        out = "\n".join(
            ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
            + ["| " + " | ".join(r) + " |" for r in rows]
        )
        spec: dict[str, Any] = {"type": "md_table", "header": header, "rows": rows}
        fam = "fmt.table"
    elif kind == 1:
        user = _pick(
            rng,
            (
                f"Convert this text to CSV with a header row {','.join(cols)}:\n\n{text}",
                f"{text}\n\nWrite the data as CSV. Header: {','.join(cols)}",
                f"Turn the following into comma-separated values with the columns {', '.join(cols)}"
                f" (header first): {text}",
            ),
        )
        out = "\n".join([",".join(cols)] + [",".join(r) for r in rows])
        spec = {"type": "lines", "value": out}
        fam = "fmt.csv"
    else:
        field = _pick(rng, ["city", "job", "name"])
        if field == "name":
            q = _pick(
                rng,
                (
                    "List the names of the people in this text, separated by commas",
                    "Who is mentioned in this text? Give the names separated by commas",
                ),
            )
        else:
            q = {
                "city": _pick(
                    rng,
                    (
                        "List the cities mentioned in this text, in order, separated by commas",
                        "Which cities appear in this text? Give them in order, comma-separated",
                    ),
                ),
                "job": _pick(
                    rng,
                    (
                        "List everyone's job from this text, in order, separated by commas",
                        "What jobs are mentioned in this text? Give them in order, comma-separated",
                    ),
                ),
            }[field]
        user = f"{q}.\n\n{text}" if rng.random() < 0.5 else f"{text}\n\n{q}."
        vals = [str(p[field]) for p in people]
        out = ", ".join(vals)
        spec = {"type": "exact", "value": out, "strip_period": True}
        fam = "extract.list"
    turns = [*_system(rng), Turn("user", user), Turn("assistant", out, spec)]
    return Item(fam, f"fmt:{kind}:{','.join(cols)}:{text}", turns, ["schema_following", "no_tool"])


# --- multi-turn conversations ------------------------------------------------------------------

FOLLOW_SAME = (
    "Now do the same for these: {x}",
    "Same thing with these words: {x}",
    "Do that again for: {x}",
    "And for this one? {x}",
    "Please repeat that with: {x}",
)
FOLLOW_CHAIN = ("Now {b}.", "Next, {b}.", "Great. Now {b}.", "Take that and {b}.", "Then {b}.")
CORRECTION = (
    "Sorry, I meant: {b}.",
    "Actually, {b} instead.",
    "Wait, I wanted you to {b}.",
    "My mistake. Please {b}.",
    "No, {b}.",
)


def _ask(
    rng: random.Random, op: Op, w: list[str], params: dict[str, Any], display: str, it: str = "these words"
) -> tuple[str, str, dict[str, Any], Any]:
    i = _pick_phrasing(rng, op, False)
    instr = _clause(op, i, it, params)
    if instr.endswith("?"):
        user = f"{_cap(instr)}\n{show(w, display)}"
    else:
        user = f"{_cap(instr)}: {show(w, display)}"
    value = op.fn(w, params)
    target = str(value) if op.out != "words" else render(value, display)
    return user, target, spec_for(value, op.out, display), value


class _Retry(Exception):
    """The sampled first step admits no allowed second step; draw again."""


def gen_conversation(rng: random.Random) -> Item:
    """Two to four computed turns: a repeated task on new input, a chained step on the previous
    result, or a corrected request on the original input. Every assistant turn is supervised."""
    for _ in range(100):
        try:
            return _conversation(rng)
        except _Retry:
            continue
    raise RuntimeError("no conversation")


def _conversation(rng: random.Random) -> Item:
    kind = rng.randrange(3)
    display = _pick(rng, ("spaces", "commas"))
    a = _pick(rng, [o for o in TRAIN_OPS if o.out == "words"])
    w = _words(rng, a, 4, 7)
    pa = a.params(rng, w)
    u1, t1, s1, v1 = _ask(rng, a, w, pa, display)
    turns = [*_system(rng), Turn("user", u1), Turn("assistant", t1, s1)]
    keys = [f"op:{a.name}:{json.dumps(pa, sort_keys=True)}:{' '.join(w)}"]
    if kind == 0:  # same task, new inputs (word-valued parameters do not carry over)
        for _ in range(rng.randint(1, 2)):
            w2 = _words(rng, a, 3, 6)
            if isinstance(pa.get("p"), str) or ("p" in pa and pa["p"] >= len(w2)):
                w2 = _words(rng, a, len(w) - 1, len(w) + 1) if isinstance(pa.get("p"), int) else w2
            p2 = a.params(rng, w2) if isinstance(pa.get("p"), str) else pa
            if "p" in p2 and isinstance(p2["p"], int) and p2["p"] >= len(w2):
                continue
            v2 = a.fn(w2, p2)
            t2 = render(v2, display)
            follow = _pick(rng, FOLLOW_SAME).format(x=show(w2, display))
            if p2 is not pa:  # the parameter changed, so restate the request
                follow = _ask(rng, a, w2, p2, display)[0]
            turns += [Turn("user", follow), Turn("assistant", t2, spec_for(v2, "words", display))]
            keys.append(f"op:{a.name}:{json.dumps(p2, sort_keys=True)}:{' '.join(w2)}")
        fam = "conv.repeat"
        ops = [a.name]
    elif kind == 1:  # chain on the previous result (a held-out pair never appears in training)
        allowed = [
            o
            for o in TRAIN_OPS
            if o.name != a.name and not heldout_pair(a.name, o.name) and v1 and o.needs(v1)
        ]
        if not allowed:
            raise _Retry
        b = _pick(rng, allowed)
        pb = b.params(rng, v1)
        imperative = [
            i
            for i in range(len(b.phrasings))
            if not reserved_phrasing(b, i) and not b.phrasings[i].endswith("?")
        ]
        ib = _clause(b, _pick(rng, imperative), "that", pb)
        v2 = b.fn(v1, pb)
        t2 = str(v2) if b.out != "words" else render(v2, display)
        turns += [
            Turn("user", _pick(rng, FOLLOW_CHAIN).format(b=ib.rstrip("?"))),
            Turn("assistant", t2, spec_for(v2, b.out, display)),
        ]
        fam = "conv.chain"
        ops = [a.name, b.name]
    else:  # correction: redo on the original input
        b = _pick(rng, [o for o in TRAIN_OPS if o.out == "words" and o.name != a.name and o.needs(w)])
        pb = b.params(rng, w)
        ib = _clause(b, _pick_phrasing(rng, b, False), "the words", pb)
        v2 = b.fn(w, pb)
        t2 = render(v2, display)
        turns += [
            Turn("user", _pick(rng, CORRECTION).format(b=ib.rstrip("?"))),
            Turn("assistant", t2, spec_for(v2, "words", display)),
        ]
        fam = "conv.correction"
        ops = [a.name, b.name]
    key = f"conv:{kind}:" + "|".join(keys) + f":{turns[-2].text}"
    return Item(
        fam,
        key,
        turns,
        ["direct_answer", "no_tool"],
        extra_keys=keys,
        meta={"ops": ops, "chained": kind == 1},
    )


def gen_context_qa(rng: random.Random) -> Item:
    """A short context about one fictional person, then answerable and unanswerable questions about it
    (computed answers or "I don't know."), sometimes across turns with a pronoun."""
    name, job, city = _pick(rng, NAMES), _pick(rng, JOBS), _pick(rng, CITIES)
    objs = rng.sample(OBJECTS, 3)
    n1, n2 = rng.randint(2, 30), rng.randint(2, 30)
    pron = _pick(rng, ("she", "he", "they"))
    have = "have" if pron == "they" else "has"
    live = "live" if pron == "they" else "lives"
    ctx = (
        f"{name} is {_article(job)} {job}. {_cap(pron)} {live} in {city}"
        f" and {have} {n1} {objs[0]} and {n2} {objs[1]}."
    )
    questions: list[tuple[str, str, dict[str, Any]]] = [
        (f"How many {objs[0]} does {name} have?", str(n1), {"type": "number", "value": str(n1)}),
        (f"How many {objs[1]} does {name} have?", str(n2), {"type": "number", "value": str(n2)}),
        (
            f"Where does {name} live?",
            city,
            {"type": "entity", "aliases": [city], "distractors": [c for c in CITIES if c != city]},
        ),
        (
            f"What is {name}'s job?",
            job,
            {"type": "entity", "aliases": [job], "distractors": [j for j in JOBS if j != job]},
        ),
        (
            f"How many {objs[2]} does {name} have?",
            ABSTAIN_TARGET,
            {"type": "abstain", "forbid_numbers": True},
        ),
        (f"What is {name}'s favorite color?", ABSTAIN_TARGET, {"type": "abstain", "forbidden": list(COLORS)}),
        (f"How old is {name}?", ABSTAIN_TARGET, {"type": "abstain", "forbid_numbers": True}),
    ]
    answerable, unanswerable = list(range(4)), list(range(4, len(questions)))
    picks = rng.sample(answerable, rng.randint(1, 2))
    if rng.random() < 0.4:  # abstention targets kept near the fact-bearing share (admission review F5)
        picks.insert(rng.randrange(len(picks) + 1), _pick(rng, unanswerable))
    lead = _pick(
        rng,
        (
            "Read this and answer the question.",
            "Answer using only this text.",
            "Use the text below to answer.",
            "",
        ),
    )
    first_q = questions[picks[0]]
    user1 = f"{lead}\n\n{ctx}\n\n{first_q[0]}".strip() if rng.random() < 0.5 else f"{ctx}\n\n{first_q[0]}"
    turns = [*_system(rng), Turn("user", user1), Turn("assistant", first_q[1], first_q[2])]
    for j in picks[1:]:
        q, t, s = questions[j]
        if rng.random() < 0.5:  # pronoun continuity
            q = q.replace(f"{name}'s", {"she": "her", "he": "his", "they": "their"}[pron])
            q = q.replace(f"does {name}", f"does {pron}" if pron != "they" else "do they").replace(
                f"is {name}", f"is {pron}" if pron != "they" else "are they"
            )
        turns += [Turn("user", q), Turn("assistant", t, s)]
    fam = "qa.context" if len(picks) == 1 else "qa.context.multi"
    return Item(fam, f"ctx:{ctx}", turns, ["direct_answer", "missing_evidence", "uncertainty", "no_tool"])


Generator = Callable[[random.Random], Item]
TRAIN_FAMILIES: dict[str, Generator] = {
    "if.op": gen_op,
    "if.compose": gen_compose,
    "fmt": gen_format,
    "conv": gen_conversation,
    "qa.context": gen_context_qa,
}
TEST_FAMILIES: dict[str, Generator] = {
    "if.paraphrase": gen_paraphrase_test,
    "if.compose_heldout": gen_compose_heldout,
    "heldout.ops": gen_heldout_op,
}


def coverage() -> dict[str, Any]:
    """Declared generalization splits (for the dataset card and tests)."""
    pairs = [(a.name, b.name) for a in TRAIN_OPS if a.out == "words" for b in TRAIN_OPS if a.name != b.name]
    return {
        "train_ops": len(TRAIN_OPS),
        "heldout_ops": [o.name for o in HELDOUT_OPS],
        "phrasings_total": sum(len(o.phrasings) for o in TRAIN_OPS),
        "phrasings_reserved": sum(
            reserved_phrasing(o, i) for o in TRAIN_OPS for i in range(len(o.phrasings))
        ),
        "pairs_total": len(pairs),
        "pairs_heldout": sum(heldout_pair(a, b) for a, b in pairs),
        "frames": {"train": list(TRAIN_FRAMES), "test": list(TEST_FRAMES)},
        "compose_frames": {"train": list(TRAIN_COMPOSE), "test": list(TEST_COMPOSE)},
    }


def _seq_check(seq: Sequence[Op]) -> None:
    names = [o.name for o in seq]
    if len(set(names)) != len(names):
        raise ValueError("duplicate operation names")


_seq_check(TRAIN_OPS + HELDOUT_OPS)
