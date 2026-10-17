"""Answer specifications and scorers shared by SFT admission and post-training evaluation.

A specification is a small JSON object. The dataset builder admits a target only if it scores 1
under its own specification, and the evaluation scorer grades model responses with the same
function, so a scorer defect shows up as rejected references, not as silently inflated results.
Scorers are deterministic and never call a model.

Types: ``exact``, ``number``, ``entity``, ``abstain``, ``json``, ``json_key``, ``stdout``,
``python_tests`` (sandboxed), ``lowercase``, ``uppercase``, ``max_words``, ``all``, ``lines``,
``md_table``, ``python_suite`` (out-of-process I/O suite, j1data.codecheck) and
``open_ended``. ``open_ended`` (human-written conversation targets) has no
deterministic oracle: :func:`score` raises :class:`Unscored` and callers record it as unscored
(rubric review), never as passed or failed.

v2 (independent review 2026-10-12): Python tests run in a harness that strips ``__main__``
blocks, executes the candidate in its own namespace, runs the tests and prints a per-run nonce
only after they all pass (exit status alone never counts); the code block that defines the entry
point is graded. Abstention requires an abstention phrase and no hedged guess. Entities fail when
the response also names a distractor (e.g. lists every continent); accents are folded. Numbers
prefer a ``#### n`` marker, then the clause after "answer is", then the last number.

v4 (finding S-09, dataset v1.3): element symbols that are also English words ("No", "As", "In",
"He", ...) count as a guess only where a symbol would stand (mid-sentence, alone as a sentence, quoted
or emphasized, or after an answer cue), not as the first word of an ordinary sentence. More honest
uncertainty phrasings are recognized; a committed answer after a refusal ("I don't know, but its
capital is X") fails even when X is not a known label; "might be fictional" is not a hedged guess.
"""

from __future__ import annotations

import ast
import json
import re
import secrets
import unicodedata
from decimal import Decimal, InvalidOperation
from typing import Any

from j1data import sandbox

VERSION = "j1.sftscore.v4"

_APOSTROPHES = str.maketrans({"’": "'", "‘": "'", "ʼ": "'", "−": "-"})
_UNSURE = (
    r"i don't know|i do not know|i'm not sure|i am not sure|i'm uncertain|i am uncertain|i have no idea"
    r"|i've no idea|i can't (?:know|tell|say|answer|determine|be sure|verify)"
    r"|i cannot (?:know|tell|say|answer|determine|be sure|verify)"
    r"|i'm (?:not able|unable) to (?:know|tell|say|answer|determine|verify)"
    r"|i am (?:not able|unable) to (?:know|tell|say|answer|determine|verify)"
)
ABSTAIN_RE = re.compile(
    rf"\b(?:{_UNSURE}|i'm not aware|i am not aware|i'm not familiar|i am not familiar"
    r"|(?:i have|i've got|there is) no (?:information|data|record)"
    r"|i (?:don't|do not) have (?:any |that |this |enough |reliable )?information"
    r"|you haven't told me|you have not told me"
    r"|there(?: is|'s) no (?:country|element|state|place|record) (?:called|named)"
    r"|no such (?:country|element|place|state)|(?:isn't|is not) a real|not a real"
    r"|(?:doesn't|does not) (?:seem to )?exist|(?:hasn't|has not) (?:happened|taken place) yet"
    r"|there(?: is|'s) no way (?:for me )?to know|(?:it's|it is) impossible (?:for me )?to know)\b",
    re.IGNORECASE,
)
# Phrases that commit to an answer after hedging ("I don't know, but it is probably X"). "Might be
# fictional" and similar statements about the question itself are honest uncertainty, not guesses.
HEDGE_RE = re.compile(
    r"\b(?:probably|likely|maybe|perhaps|possibly|presumably"
    r"|(?:might|could|may) be"
    r"(?! (?:a )?(?:fictional|made[- ]up|imaginary|invented|fictitious|not real|unknown))"
    r"|i think|i believe|i guess|my (?:best )?guess|if i had to guess|i'd say|i would say|answer is)\b",
    re.IGNORECASE,
)
# A committed answer after a refusal names the asked-for value even when it is not a known label.
COMMIT_RE = re.compile(
    r"\b(?:(?:its|the) (?:capital|currency|chemical symbol|symbol|atomic number|winner|result|answer)"
    r" (?:is|was|would be)"
    r"|(?:it|they) (?:uses?|used) the|(?:was|were) won by|it is called|it's called)"
    r"\s+(?!(?:or|and|but|because|since|if|whether|unknown|not|a fictional|fictional)\b)[\w\"'“‘*]",
    re.IGNORECASE,
)
# Abstention phrases that make an *answer* to an answerable question void.
REFUSAL_CORE_RE = re.compile(rf"\b(?:{_UNSURE})\b", re.IGNORECASE)
# Element symbols that are also English words: ordinary sentence-initial use ("No, I don't know",
# "As far as I know") is not a guess. Measured on the v1.2 bundle's abstention specs.
WORD_SYMBOLS = frozenset(
    {"Am", "As", "At", "Be", "Ca", "Cf", "Cm", "Cs", "Cu", "Es", "Fl", "Fr", "He"}
    | {"Ho", "In", "La", "Mg", "Mo", "No", "Pa", "Pd", "Pt", "Re", "Ti", "Ts"}
)
_SENTENCE_START = re.compile(r"(?:^|[.!?]\s+|\n\s*|[-*•]\s+)$")
_ANSWER_CUE = re.compile(r"(?:\b(?:is|be|was|symbol|answer)|[:=])\s*$", re.IGNORECASE)
_MARKED_RE = re.compile(r"####\s*([-−]?[\d,]*\.?\d+)")
_ANSWER_CLAUSE_RE = re.compile(r"answer(?: is|:)[:\s]*([^\n]*)", re.IGNORECASE)
_NUMBER_RE = re.compile(r"-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?")
_FENCE_RE = re.compile(r"```[a-zA-Z0-9_+-]*\n(.*?)```", re.DOTALL)


class SpecError(ValueError):
    """The specification itself is malformed."""


class Unscored(ValueError):
    """The specification has no deterministic oracle (open-ended text)."""


def _table_rows(text: str) -> list[list[str]] | None:
    rows = []
    for line in strip_fence(text).strip().splitlines():
        line = line.strip()
        if not (line.startswith("|") and line.endswith("|")):
            return None
        cells = [c.strip() for c in line[1:-1].split("|")]
        if all(re.fullmatch(r":?-{3,}:?", c) for c in cells):
            continue  # alignment row
        rows.append(cells)
    return rows


def normalize(text: str) -> str:
    return text.translate(_APOSTROPHES)


def fold(text: str) -> str:
    """Case- and accent-insensitive form (San José == San Jose, Malé == Male)."""
    decomposed = unicodedata.normalize("NFKD", normalize(text))
    return "".join(c for c in decomposed if not unicodedata.combining(c)).casefold()


def _num(text: str) -> Decimal | None:
    try:
        return Decimal(text.replace(",", "").replace("−", "-").rstrip("."))
    except InvalidOperation:
        return None


def extract_number(response: str) -> Decimal | None:
    """``#### n``; else the first number after the last ``=`` in the "answer is" clause; else the last."""
    text = normalize(response).replace("**", "")
    marked = _MARKED_RE.findall(text)
    if marked:
        return _num(marked[-1])
    clause = _ANSWER_CLAUSE_RE.findall(text)
    if clause:
        sentence = clause[-1].split(". ")[0]
        after_eq = sentence.rsplit("=", 1)[-1]  # "answer is 5 + 3 = 8" -> 8; "answer is 18 (3 x 6)" -> 18
        first = _NUMBER_RE.findall(after_eq)
        if first:
            return _num(first[0])
    found = _NUMBER_RE.findall(text)
    return _num(found[-1]) if found else None


def strip_fence(response: str) -> str:
    blocks = _FENCE_RE.findall(response)
    return blocks[-1] if blocks else response


def select_code(response: str, entry_point: str | None) -> str:
    """The fenced block that defines ``entry_point`` (last such), else the last block, else the text."""
    blocks = _FENCE_RE.findall(response) or [response]
    if entry_point:
        defining = [
            b for b in blocks if re.search(rf"^\s*def\s+{re.escape(entry_point)}\s*\(", b, re.MULTILINE)
        ]
        if defining:
            return str(defining[-1])
    return str(blocks[-1])


def strip_main_blocks(code: str) -> str | None:
    """Remove top-level ``if __name__ == "__main__":`` blocks; None if the code does not parse."""
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError):
        return None

    def is_main(node: ast.stmt) -> bool:
        return isinstance(node, ast.If) and "__name__" in ast.unparse(node.test)

    tree.body = [n for n in tree.body if not is_main(n)]
    return ast.unparse(tree)


# Non-adversarial harness: it defeats early exits, __main__ blocks and exit-status tricks, but code that
# deliberately inspects interpreter frames can read the nonce. Use an out-of-process checker before any
# reward/RL use of this scorer.
_HARNESS = """
import sys
_ns = {{"__name__": "candidate", "__builtins__": __builtins__}}
exec(compile({code!r}, "<candidate>", "exec"), _ns)
exec(compile({setup!r}, "<setup>", "exec"), _ns)
exec(compile({tests!r}, "<tests>", "exec"), _ns)
sys.stdout.write("\\n{nonce}\\n")
sys.stdout.flush()
"""


def run_tests(
    code: str, setup: str, tests: str, *, purpose: str = "functional_diagnostic"
) -> tuple[bool, str]:
    """Run assert tests after ``code`` in the sandbox. Like j1data.codecheck, a functional diagnostic:
    the candidate shares the harness interpreter and can forge the completion nonce."""
    if purpose != "functional_diagnostic":
        raise SpecError(f"{purpose!r} refused: candidate interpreter can forge the completion nonce")
    cleaned = strip_main_blocks(code)
    if cleaned is None:
        return False, "candidate does not parse"
    nonce = secrets.token_hex(16)
    result = sandbox.run_python(_HARNESS.format(setup=setup, code=cleaned, tests=tests, nonce=nonce))
    passed = result.ok and result.stdout.rstrip("\n").endswith(nonce)
    return passed, ("timeout" if result.timed_out else result.stderr[-200:] or "no completion nonce")


def _abstained(text: str, core_only: bool = False) -> bool:
    return bool((REFUSAL_CORE_RE if core_only else ABSTAIN_RE).search(normalize(text)))


def _names_symbol(text: str, symbol: str) -> bool:
    """Whether ``text`` states the element symbol (case-sensitive). A word-like symbol counts only
    where a symbol would stand, never as the first word of an ordinary sentence."""
    for m in re.finditer(rf"(?<![A-Za-z]){re.escape(symbol)}(?![A-Za-z])", text):
        if symbol not in WORD_SYMBOLS:
            return True
        before, after = text[: m.start()], text[m.end() :]
        quoted = before[-1:] in "\"'“‘*`(" and after[:1] in "\"'”’*`)"
        starts = _SENTENCE_START.search(before) is not None
        alone = starts and re.match(r"[\s.!?*`]*(?:$|\n)", after) is not None
        if quoted or alone or not starts or _ANSWER_CUE.search(before):
            return True
    return False


def _mentions(text: str, phrase: str) -> bool:
    p = fold(phrase).strip()
    return bool(p) and re.search(r"(?<![\w])" + re.escape(p) + r"(?![\w])", fold(text)) is not None


def score(spec: dict[str, Any], response: str) -> tuple[bool, str]:
    """Return ``(passed, detail)``. Raises :class:`sandbox.SandboxUnavailable` for an executable
    specification when isolation is unavailable (the caller records *not executed*)."""
    kind = spec.get("type")
    if kind == "exact":
        want, got = str(spec["value"]).strip(), response.strip()
        if spec.get("strip_period"):
            want, got = want.rstrip(".").strip(), got.rstrip(".").strip()
        if not spec.get("case_sensitive", True):
            want, got = want.casefold(), got.casefold()
        return got == want, "exact"
    if kind == "number":
        want_n = _num(str(spec["value"]))
        if want_n is None:
            raise SpecError("number spec without a numeric value")
        got_n = extract_number(response)
        return got_n is not None and got_n == want_n, f"extracted {got_n}"
    if kind == "entity":
        aliases = spec.get("aliases") or []
        if not aliases:
            raise SpecError("entity spec without aliases")
        if _abstained(response, core_only=True):
            return False, "abstained on an answerable question"
        if not any(_mentions(response, a) for a in aliases):
            return False, "entity not named"
        named_wrong = [d for d in spec.get("distractors", []) if _mentions(response, d)]
        return not named_wrong, ("ok" if not named_wrong else f"also names {named_wrong[:3]}")
    if kind == "abstain":
        text = normalize(response)
        if not _abstained(text):
            return False, "no abstention"
        if HEDGE_RE.search(text):
            return False, "hedged guess"
        # A committed answer after the abstention ("I don't know. Asia.") names a candidate label.
        named = [lab for lab in spec.get("forbidden", []) if _mentions(text, lab)]
        named += [s for s in spec.get("forbidden_symbols", []) if _names_symbol(text, s)]
        if named:
            return False, f"names {named[:3]}"
        if COMMIT_RE.search(text):
            return False, "commits to an answer"
        if spec.get("forbid_numbers") and re.search(r"\d", text):
            return False, "states a number"
        return True, "abstention"
    if kind == "json":
        try:
            got = json.loads(strip_fence(response).strip())
        except (json.JSONDecodeError, ValueError):
            return False, "not JSON"
        return _json_equal(got, spec["value"]), "json"
    if kind == "json_key":
        try:
            got = json.loads(strip_fence(response).strip())
        except (json.JSONDecodeError, ValueError):
            return False, "not JSON"
        if not isinstance(got, dict) or set(got) != {spec["key"]}:
            return False, "wrong JSON shape"
        value = got[spec["key"]]
        if isinstance(value, (dict, list, bool)) or value is None:
            return False, "answer value must be a string or number"
        return score(spec["inner"], str(value))
    if kind == "stdout":
        want_s = spec["value"]
        if want_s is None:
            raise SpecError("stdout spec has no recorded output")
        a = [line.rstrip() for line in strip_fence(response).strip("\n").splitlines()]
        b = [line.rstrip() for line in str(want_s).strip("\n").splitlines()]
        return a == b, "stdout"
    if kind == "python_tests":
        entry = str(spec.get("entry_point") or "")
        code = select_code(response, entry)
        prefix = spec.get("prefix", "")
        if prefix:
            # HumanEval style: the prompt's imports, helpers and signature always precede the answer;
            # a whole-function answer simply redefines the stub.
            code = prefix + "\n" + code
        return run_tests(
            code,
            spec.get("setup", ""),
            spec["tests"],
            purpose=str(spec.get("purpose", "functional_diagnostic")),
        )
    if kind == "python_suite":
        # Out-of-process graded I/O suite (j1data.codecheck): expected values come from the trusted
        # reference; the candidate never runs in this process. Protected evaluation material only.
        from j1data import codecheck

        code = select_code(response, str(spec["entry_point"]))
        if spec.get("prefix"):
            code = spec["prefix"] + "\n" + code
        cleaned = strip_main_blocks(code)
        if cleaned is None:
            return False, "candidate does not parse"
        return codecheck.check_against_suite(
            spec["suite"],
            cleaned,
            spec["tiers"],
            purpose=str(spec.get("purpose", "functional_diagnostic")),
        )
    if kind == "lines":
        a = [line.strip() for line in strip_fence(response).strip().splitlines() if line.strip()]
        b = [line.strip() for line in str(spec["value"]).strip().splitlines() if line.strip()]
        return a == b, "lines"
    if kind == "md_table":
        rows = _table_rows(response)
        if rows is None:
            return False, "not a Markdown table"
        want_rows = [list(map(str, spec["header"])), *[list(map(str, r)) for r in spec["rows"]]]
        return rows == want_rows, "md_table"
    if kind == "open_ended":
        raise Unscored("open-ended target: rubric review, no deterministic oracle")
    if kind == "lowercase":
        return any(c.isalpha() for c in response) and response == response.lower(), "lowercase"
    if kind == "uppercase":
        return any(c.isalpha() for c in response) and response == response.upper(), "uppercase"
    if kind == "max_words":
        return len(response.split()) <= int(spec["n"]), "max_words"
    if kind == "all":
        details = []
        for part in spec["parts"]:
            ok, detail = score(part, response)
            details.append(detail)
            if not ok:
                return False, f"failed {part.get('type')}: {detail}"
        return True, "; ".join(details)
    raise SpecError(f"unknown spec type {kind!r}")


def _json_equal(a: Any, b: Any) -> bool:
    """Structural equality that keeps JSON types apart (true != 1, 1 != "1")."""
    if isinstance(a, bool) or isinstance(b, bool):
        return isinstance(a, bool) and isinstance(b, bool) and a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return a == b
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_json_equal(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_json_equal(x, y) for x, y in zip(a, b, strict=True))
    return type(a) is type(b) and a == b
