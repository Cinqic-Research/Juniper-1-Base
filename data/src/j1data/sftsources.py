"""External sources for the first instruction-SFT dataset and its evaluation suite.

Each loader returns :class:`j1data.sftgen.Item` candidates together with explicit rejections.
Upstream files are read from pinned revisions and hashed by the builder; nothing here
decides rights (see ``configs/sft_sources.toml`` and the rights ledger).

* GSM8K ``main`` (MIT, human-written). Every calculator annotation ``<<expr=result>>`` is
  re-evaluated with a restricted arithmetic evaluator, and the final ``#### answer`` must
  be a number. Solutions without annotations cannot be execution-checked and are rejected.
  Only the arithmetic is verified; the solution logic remains the human author's.
* MBPP ``full`` (CC-BY-4.0, human-written). The reference code is normalized with
  ``ast.unparse`` (comments and layout change; recorded as a modification) and must pass all
  of its tests in the sandbox. A bug-fix variant applies one deterministic AST mutation that
  makes at least one test fail.
* HumanEval (MIT) and the GSM8K/MBPP test splits are evaluation-only.
* Wikidata (CC0) facts come from a pinned snapshot file produced by :func:`fetch_wikidata`.
"""

from __future__ import annotations

import ast
import json
import operator
import random
import re
import time
import urllib.parse
import urllib.request
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, ClassVar

import pyarrow.parquet as pq

from j1data import sandbox
from j1data.hashing import sha256_bytes
from j1data.sftgen import Item, Turn

VERSION = "j1.sftsources.v1"

# --- GSM8K ------------------------------------------------------------------------------------

_ANNOT_RE = re.compile(r"<<([^<>=]*)=([^<>]*)>>")
_FINAL_RE = re.compile(r"\n?####\s*(.+?)\s*$")
_ALLOWED_BIN = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}


class ArithmeticError_(ValueError):
    pass


def safe_eval(expr: str) -> float:
    """Evaluate +,-,*,/,//,%,** over numeric literals and parentheses only."""
    tree = ast.parse(expr.replace("^", "**").replace(",", ""), mode="eval")

    def ev(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, (int, float))
            and not isinstance(node.value, bool)
        ):
            return float(node.value)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            v = ev(node.operand)
            return -v if isinstance(node.op, ast.USub) else v
        if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BIN:
            left, right = ev(node.left), ev(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 64:
                raise ArithmeticError_("exponent too large")
            return float(_ALLOWED_BIN[type(node.op)](left, right))
        raise ArithmeticError_(f"disallowed syntax {type(node).__name__}")

    return ev(tree)


def annotation_matches(expr: str, result: str) -> bool:
    try:
        got = safe_eval(expr)
        want = float(result.replace(",", "").strip())
    except (ArithmeticError_, SyntaxError, ValueError, ZeroDivisionError, OverflowError):
        return False
    if abs(got - want) <= 1e-6 * max(1.0, abs(want)):
        return True
    decimals = len(result.split(".")[1]) if "." in result else 0
    return round(got, decimals) == round(want, decimals)


def gsm8k_items(path: Path, *, split: str, revision: str) -> tuple[list[Item], list[dict[str, Any]]]:
    rows = pq.read_table(path).to_pylist()
    items: list[Item] = []
    rejected: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        record = f"gsm8k/main/{path.stem.split('-')[0]}/{index}"
        question, solution = row["question"].strip(), row["answer"].strip()
        final = _FINAL_RE.search(solution)
        annotations = _ANNOT_RE.findall(solution)
        reason = None
        if final is None:
            reason = "no #### final answer"
        elif not annotations:
            reason = "no executable calculator annotations"
        else:
            bad = [f"{e}={r}" for e, r in annotations if not annotation_matches(e, r)]
            if bad:
                reason = f"calculator annotation mismatch: {bad[0]}"
        answer = final.group(1).replace(",", "").strip() if final else ""
        try:
            Decimal(answer)
        except ArithmeticError:
            reason = reason or f"final answer is not a number: {answer!r}"
        if reason:
            rejected.append({"source_record": record, "family": "gsm8k", "reason": reason})
            continue
        body = _ANNOT_RE.sub(lambda _m: "", solution[: final.start()]).strip()  # type: ignore[union-attr]
        target = f"{body}\n#### {answer}"  # GSM8K's own human-authored answer line
        items.append(
            Item(
                "gsm8k",
                f"gsm8k:{record}",
                [Turn("user", question), Turn("assistant", target, {"type": "number", "value": answer})],
                ["direct_answer", "concise_reasoning", "no_tool"],
                factuality="VERIFIED_EXECUTION",
                difficulty="medium",
                source_id="openai/gsm8k",
                source_revision=revision,
                source_record=record,
                license="MIT",
                origin="human",
                split=split,
                extra_checks=[
                    {
                        "name": "calculator_annotations_reexecuted",
                        "oracle": "execution",
                        "passed": True,
                        "detail": f"{len(annotations)} annotations re-evaluated by a restricted "
                        "arithmetic evaluator",
                    },
                    {
                        "name": "final_answer_matches_dataset_label",
                        "oracle": "known_answer",
                        "passed": True,
                        "detail": "the #### line equals the dataset's answer",
                    },
                ],
                meta={"question": question},
            )
        )
    return items, rejected


def gsm8k_eval_items(path: Path, *, revision: str) -> list[Item]:
    items = []
    for index, row in enumerate(pq.read_table(path).to_pylist()):
        final = _FINAL_RE.search(row["answer"])
        assert final is not None
        answer = final.group(1).replace(",", "").strip()
        record = f"gsm8k/main/test/{index}"
        items.append(
            Item(
                "ext.gsm8k",
                f"gsm8k:{record}",
                [
                    Turn("user", row["question"].strip()),
                    Turn("assistant", "", {"type": "number", "value": answer}),
                ],
                ["direct_answer"],
                source_id="openai/gsm8k",
                source_revision=revision,
                source_record=record,
                license="MIT",
                origin="human",
                split="test",
                meta={"question": row["question"]},
            )
        )
    return items


# --- MBPP / HumanEval -------------------------------------------------------------------------


_ENTRY_RE = re.compile(
    r"assert\s*(?:not\s+)?\(?\s*(?:math\.isclose\(\s*|set\(\s*|sorted\(\s*|round\(\s*)*([A-Za-z_]\w*)\s*\("
)


def mbpp_entry_point(tests: list[str]) -> str | None:
    m = _ENTRY_RE.search(tests[0]) if tests else None
    return m.group(1) if m else None


def _mbpp_prompt(text: str, tests: list[str]) -> str:
    return f"{text.strip()}\nYour code should pass these tests:\n\n" + "\n".join(tests)


def _mbpp_tests(row: dict[str, Any]) -> str:
    return "\n".join([*row["test_list"], *(row.get("challenge_test_list") or [])])


def normalize_code(code: str) -> str:
    return ast.unparse(ast.parse(code.replace("\r\n", "\n").replace("\t", "    ")))


class _Mutator(ast.NodeTransformer):
    SWAPS: ClassVar[dict[type, type]] = {
        ast.Add: ast.Sub,
        ast.Sub: ast.Add,
        ast.Mult: ast.Add,
        ast.Lt: ast.LtE,
        ast.LtE: ast.Lt,
        ast.Gt: ast.GtE,
        ast.GtE: ast.Gt,
        ast.Eq: ast.NotEq,
        ast.NotEq: ast.Eq,
        ast.And: ast.Or,
        ast.Or: ast.And,
    }

    def __init__(self, target: int) -> None:
        self.target, self.seen = target, -1

    def _hit(self) -> bool:
        self.seen += 1
        return self.seen == self.target

    def visit_BinOp(self, node: ast.BinOp) -> ast.AST:
        self.generic_visit(node)
        if type(node.op) in self.SWAPS and self._hit():
            node.op = self.SWAPS[type(node.op)]()
        return node

    def visit_Compare(self, node: ast.Compare) -> ast.AST:
        self.generic_visit(node)
        if type(node.ops[0]) in self.SWAPS and self._hit():
            node.ops[0] = self.SWAPS[type(node.ops[0])]()
        return node

    def visit_BoolOp(self, node: ast.BoolOp) -> ast.AST:
        self.generic_visit(node)
        if self._hit():
            node.op = self.SWAPS[type(node.op)]()
        return node

    def visit_Constant(self, node: ast.Constant) -> ast.AST:
        if isinstance(node.value, int) and not isinstance(node.value, bool) and self._hit():
            return ast.copy_location(ast.Constant(node.value + 1), node)
        return node


def mutation_sites(code: str) -> int:
    m = _Mutator(-1)
    m.visit(ast.parse(code))
    return m.seen + 1


def mutate(code: str, site: int) -> str:
    return ast.unparse(_Mutator(site).visit(ast.parse(code)))


def _passes(setup: str, code: str, tests: str) -> sandbox.Result:
    return sandbox.run_python("\n".join([setup, code, "", tests, ""]))


def mbpp_items(
    path: Path, *, split: str, revision: str, upstream_split: str, with_fix: bool, seed: int = 0
) -> tuple[list[Item], list[dict[str, Any]]]:
    items: list[Item] = []
    rejected: list[dict[str, Any]] = []
    for row in pq.read_table(path).to_pylist():
        record = f"mbpp/full/{upstream_split}/{row['task_id']}"
        try:
            code = normalize_code(row["code"])
        except SyntaxError as e:
            rejected.append(
                {"source_record": record, "family": "python.mbpp", "reason": f"reference does not parse: {e}"}
            )
            continue
        setup, tests = row.get("test_setup_code") or "", _mbpp_tests(row)
        result = _passes(setup, code, tests)
        if not result.ok:
            rejected.append(
                {
                    "source_record": record,
                    "family": "python.mbpp",
                    "reason": "reference fails its own tests in sandbox",
                }
            )
            continue
        spec = {
            "type": "python_tests",
            "setup": setup,
            "tests": tests,
            "entry_point": mbpp_entry_point(row["test_list"]),
        }
        check = {
            "name": "reference_passes_tests_in_sandbox",
            "oracle": "execution",
            "passed": True,
            "detail": f"{sandbox.VERSION}; {len(row['test_list'])} tests + "
            f"{len(row.get('challenge_test_list') or [])} challenge",
        }
        common = {
            "factuality": "VERIFIED_EXECUTION",
            "source_id": "google-research-datasets/mbpp",
            "source_revision": revision,
            "license": "CC-BY-4.0",
            "origin": "human",
            "split": split,
            "difficulty": "medium",
        }
        user = _mbpp_prompt(row["text"], row["test_list"])
        items.append(
            Item(
                "python.mbpp",
                f"mbpp:{row['task_id']}",
                [Turn("user", user), Turn("assistant", f"```python\n{code}\n```", spec)],
                ["python_use", "no_tool"],
                source_record=record,
                extra_checks=[check],
                meta={
                    "modification": "reference code normalized with ast.unparse",
                    "task_id": row["task_id"],
                    "task_text": row["text"],
                    "normalized_code": code,
                },
                **common,  # type: ignore[arg-type]
            )
        )
        if not with_fix:
            continue
        rng = random.Random(f"{seed}:{row['task_id']}")
        sites = list(range(mutation_sites(code)))
        rng.shuffle(sites)
        buggy = None
        for site in sites[:12]:
            candidate = mutate(code, site)
            if candidate != code and not _passes(setup, candidate, tests).ok:
                buggy = candidate
                break
        if buggy is None:
            rejected.append(
                {"source_record": record, "family": "python.mbpp_fix", "reason": "no failing mutation found"}
            )
            continue
        fix_user = (
            f"The following Python code is supposed to solve this task: {row['text'].strip()}\n"
            f"It has a bug. Fix it and return the corrected code.\n\n```python\n{buggy}\n```\n\n"
            "It should pass these tests:\n" + "\n".join(row["test_list"])
        )
        items.append(
            Item(
                "python.mbpp_fix",
                f"mbppfix:{row['task_id']}",
                [Turn("user", fix_user), Turn("assistant", f"```python\n{code}\n```", spec)],
                ["python_use", "error_correction", "no_tool"],
                source_record=record + "#fix",
                extra_checks=[
                    check,
                    {
                        "name": "mutated_code_fails_tests_in_sandbox",
                        "oracle": "execution",
                        "passed": True,
                        "detail": "one AST mutation; at least one test fails",
                    },
                ],
                meta={
                    "modification": "deterministic AST mutation of the normalized reference",
                    "task_id": row["task_id"],
                    "task_text": row["text"],
                    "normalized_code": code,
                },
                **common,  # type: ignore[arg-type]
            )
        )
    return items, rejected


def mbpp_eval_items(path: Path, *, revision: str) -> list[Item]:
    items = []
    for row in pq.read_table(path).to_pylist():
        record = f"mbpp/full/test/{row['task_id']}"
        spec = {
            "type": "python_tests",
            "setup": row.get("test_setup_code") or "",
            "tests": _mbpp_tests(row),
            "entry_point": mbpp_entry_point(row["test_list"]),
        }
        try:
            reference = normalize_code(row["code"])
        except SyntaxError:
            reference = ""
        items.append(
            Item(
                "ext.mbpp",
                f"mbpp:{row['task_id']}",
                [Turn("user", _mbpp_prompt(row["text"], row["test_list"])), Turn("assistant", "", spec)],
                ["python_use"],
                source_id="google-research-datasets/mbpp",
                source_revision=revision,
                source_record=record,
                license="CC-BY-4.0",
                origin="human",
                split="test",
                meta={"task_id": row["task_id"], "reference_code": reference, "task_text": row["text"]},
            )
        )
    return items


def humaneval_items(path: Path, *, revision: str) -> list[Item]:
    items = []
    for row in pq.read_table(path).to_pylist():
        user = (
            "Complete the following Python function. Return the whole function in a code block.\n\n"
            f"```python\n{row['prompt'].rstrip()}\n```"
        )
        spec = {
            "type": "python_tests",
            "setup": "",
            "prefix": row["prompt"],
            "tests": f"{row['test']}\ncheck({row['entry_point']})",
            "entry_point": row["entry_point"],
        }
        items.append(
            Item(
                "ext.humaneval",
                f"humaneval:{row['task_id']}",
                [Turn("user", user), Turn("assistant", "", spec)],
                ["python_use"],
                source_id="openai/openai_humaneval",
                source_revision=revision,
                source_record=row["task_id"],
                license="MIT",
                origin="human",
                split="test",
                meta={"canonical_solution": row["canonical_solution"]},
            )
        )
    return items


# --- Wikidata (CC0) ---------------------------------------------------------------------------

WDQS = "https://query.wikidata.org/sparql"
USER_AGENT = "Juniper1-SFTData/0.1 (https://github.com/Cinqic-Research; research dataset build)"
LABELS = 'SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }'
QUERIES = {
    "country_capital": (
        "SELECT ?s ?sLabel ?o ?oLabel ?rev (GROUP_CONCAT(DISTINCT ?alt; separator='|') AS ?alts) WHERE { "
        "?s wdt:P31 wd:Q3624078 . FILTER NOT EXISTS { ?s wdt:P576 ?end } "
        "?s wdt:P36 ?o . ?s schema:version ?rev . "
        "OPTIONAL { ?o skos:altLabel ?alt . FILTER(LANG(?alt) = 'en') } " + LABELS + " } "
        "GROUP BY ?s ?sLabel ?o ?oLabel ?rev"
    ),
    "country_continent": (
        "SELECT ?s ?sLabel ?o ?oLabel ?rev (GROUP_CONCAT(DISTINCT ?alt; separator='|') AS ?alts) WHERE { "
        "?s wdt:P31 wd:Q3624078 . FILTER NOT EXISTS { ?s wdt:P576 ?end } "
        "?s wdt:P30 ?o . ?s schema:version ?rev . "
        "OPTIONAL { ?o skos:altLabel ?alt . FILTER(LANG(?alt) = 'en') } " + LABELS + " } "
        "GROUP BY ?s ?sLabel ?o ?oLabel ?rev"
    ),
    "country_currency": (
        "SELECT ?s ?sLabel ?o ?oLabel ?rev (GROUP_CONCAT(DISTINCT ?alt; separator='|') AS ?alts) WHERE { "
        "?s wdt:P31 wd:Q3624078 . FILTER NOT EXISTS { ?s wdt:P576 ?end } "
        "?s wdt:P38 ?o . ?s schema:version ?rev . "
        "OPTIONAL { ?o skos:altLabel ?alt . FILTER(LANG(?alt) = 'en') } " + LABELS + " } "
        "GROUP BY ?s ?sLabel ?o ?oLabel ?rev"
    ),
    "us_state_capital": (
        "SELECT ?s ?sLabel ?o ?oLabel ?rev (GROUP_CONCAT(DISTINCT ?alt; separator='|') AS ?alts) WHERE { "
        "?s wdt:P31 wd:Q35657 . ?s wdt:P36 ?o . ?s schema:version ?rev . "
        "OPTIONAL { ?o skos:altLabel ?alt . FILTER(LANG(?alt) = 'en') } " + LABELS + " } "
        "GROUP BY ?s ?sLabel ?o ?oLabel ?rev"
    ),
    "element_symbol": (
        "SELECT ?s ?sLabel ?o ?rev WHERE { ?s wdt:P31 wd:Q11344 . ?s wdt:P246 ?o . "
        "?s schema:version ?rev . " + LABELS + " }"
    ),
    "element_number": (
        "SELECT ?s ?sLabel ?o ?rev WHERE { ?s wdt:P31 wd:Q11344 . ?s wdt:P1086 ?o . "
        "?s schema:version ?rev . " + LABELS + " }"
    ),
}


def fetch_wikidata(out: Path) -> dict[str, Any]:
    """Run the pinned queries once and write an immutable snapshot (raw responses + hashes)."""
    snapshot: dict[str, Any] = {
        "schema": "j1.wikidata-snapshot.v1",
        "endpoint": WDQS,
        "license": "CC0-1.0",
        "retrieved_utc": datetime.now(UTC).isoformat(),
        "queries": {},
    }
    for name, query in QUERIES.items():
        url = WDQS + "?" + urllib.parse.urlencode({"query": query, "format": "json"})
        req = urllib.request.Request(
            url, headers={"User-Agent": USER_AGENT, "Accept": "application/sparql-results+json"}
        )
        with urllib.request.urlopen(req, timeout=90) as resp:
            raw = resp.read()
        snapshot["queries"][name] = {
            "query": query,
            "response_sha256": sha256_bytes(raw),
            "bindings": json.loads(raw)["results"]["bindings"],
        }
        time.sleep(2)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(snapshot, indent=1, sort_keys=True, ensure_ascii=False))
    return snapshot


def _qid(uri: str) -> str:
    return uri.rsplit("/", 1)[-1]


FACT_TEMPLATES: dict[str, tuple[str, ...]] = {
    "country_capital": (
        "What is the capital of {s}?",
        "What city is the capital of {s}?",
        "Name the capital of {s}.",
    ),
    "country_continent": ("Which continent is {s} in?", "On what continent is {s} located?"),
    "country_currency": ("What is the currency of {s}?", "What currency is used in {s}?"),
    "us_state_capital": (
        "What is the capital of the state of {s}?",
        "What city is the capital of the U.S. state of {s}?",
    ),
    "element_symbol": ("What is the chemical symbol for {s}?",),
    "element_number": ("What is the atomic number of {s}?",),
}
_THE = {
    "Philippines",
    "Netherlands",
    "Bahamas",
    "Gambia",
    "Maldives",
    "Comoros",
    "Seychelles",
    "Czech Republic",
    "Dominican Republic",
    "Central African Republic",
    "Vatican City",
    "State of Palestine",
}


def with_article(label: str) -> str:
    """'the Philippines', 'the United States', 'the Marshall Islands'; other names unchanged."""
    if label.lower().startswith("the ") or not (
        label in _THE
        or label.startswith(("United ", "Republic of ", "Democratic Republic", "Federated States"))
        or label.endswith(" Islands")
    ):
        return label
    return "the " + label


def _fold(text: str) -> str:
    from j1data.sftscore import fold

    return fold(text)


def wikidata_facts(snapshot: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """One fact per (relation, subject) with exactly one object; ambiguous facts are rejected.

    Element facts are limited to the 118 confirmed elements (a single atomic number 1-118). Aliases
    that contain the subject's name or are generic parts of other objects' labels (e.g. "dollar")
    are dropped; every other object label of the relation becomes a distractor, so an answer that
    lists several candidates fails."""
    facts: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    numbers: dict[str, set[str]] = {}
    for b in snapshot["queries"]["element_number"]["bindings"]:
        numbers.setdefault(_qid(b["s"]["value"]), set()).add(b["o"]["value"])
    confirmed = {
        q
        for q, v in numbers.items()
        if len(v) == 1 and next(iter(v)).isdigit() and 1 <= int(next(iter(v))) <= 118
    }
    for rel, q in sorted(snapshot["queries"].items()):
        by_subject: dict[str, list[dict[str, Any]]] = {}
        for b in q["bindings"]:
            by_subject.setdefault(_qid(b["s"]["value"]), []).append(b)
        rel_facts = []
        for qid, rows in sorted(by_subject.items()):
            objects = {r["o"]["value"] for r in rows}
            record = f"wikidata/{rel}/{qid}"
            if rel.startswith("element_") and qid not in confirmed:
                rejected.append(
                    {
                        "source_record": record,
                        "family": "facts",
                        "reason": "not one of the 118 confirmed elements",
                    }
                )
                continue
            s_label = rows[0].get("sLabel", {}).get("value", "")
            if len(objects) != 1:
                rejected.append(
                    {
                        "source_record": record,
                        "family": "facts",
                        "reason": f"{len(objects)} objects (ambiguous)",
                    }
                )
                continue
            row = rows[0]
            o_value = row["o"]["value"]
            o_label = row.get("oLabel", {}).get("value", o_value) if o_value.startswith("http") else o_value
            if not s_label or re.fullmatch(r"Q\d+", s_label) or re.fullmatch(r"Q\d+", o_label):
                rejected.append(
                    {"source_record": record, "family": "facts", "reason": "missing English label"}
                )
                continue
            alts = [a for a in (row.get("alts", {}).get("value") or "").split("|") if len(a) >= 3]
            rel_facts.append(
                {
                    "relation": rel,
                    "subject_qid": qid,
                    "subject_label": s_label,
                    "object": o_value,
                    "object_label": o_label,
                    "alts": alts,
                    "subject_revision": int(row["rev"]["value"]),
                    "record": record,
                    "case_sensitive": rel == "element_symbol",
                }
            )
        labels = sorted({f["object_label"] for f in rel_facts})
        for f in rel_facts:
            own = _fold(f["object_label"])
            others = [lab for lab in labels if _fold(lab) != own]
            f["distractors"] = [lab for lab in others if own not in _fold(lab) and _fold(lab) not in own]
            f["aliases"] = [f["object_label"]] + [
                a
                for a in f.pop("alts")
                if _fold(f["subject_label"]) not in _fold(a) and not any(_fold(a) in _fold(o) for o in others)
            ]
            facts.append(f)
    return facts, rejected


def fact_leaks(item: Item) -> bool:
    """True when restating the question alone would pass the answer's scorer."""
    from j1data.sftscore import score

    spec = item.turns[-1].spec
    assert spec is not None
    return score(spec, item.meta["question"])[0]


def fact_spec(fact: dict[str, Any]) -> dict[str, Any]:
    if fact["relation"] == "element_number":
        return {"type": "number", "value": fact["object_label"]}
    if fact["case_sensitive"]:  # element symbols: "I", "In", "He" are also English words
        return {"type": "exact", "value": fact["object_label"], "strip_period": True}
    return {"type": "entity", "aliases": fact["aliases"], "distractors": fact["distractors"]}


def fact_item(fact: dict[str, Any], rng: random.Random, split: str, snapshot_sha: str) -> Item:
    subject = (
        fact["subject_label"]
        if fact["relation"].startswith(("element", "us_"))
        else with_article(fact["subject_label"])
    )
    q = rng.choice(FACT_TEMPLATES[fact["relation"]]).format(s=subject)
    return Item(
        f"facts.{fact['relation']}",
        f"fact:{fact['subject_qid']}:{fact['relation']}",
        [Turn("user", q), Turn("assistant", fact["object_label"], fact_spec(fact))],
        ["direct_answer", "no_tool"],
        factuality="VERIFIED_SOURCE",
        source_id="wikidata",
        source_revision=f"snapshot:{snapshot_sha}",
        source_record=f"{fact['record']}@rev{fact['subject_revision']}",
        license="CC0-1.0",
        origin="template",
        split=split,
        extra_checks=[
            {
                "name": "wikidata_statement_single_valued",
                "oracle": "source_evidence",
                "passed": True,
                "detail": f"{fact['record']} revision {fact['subject_revision']}",
            }
        ],
        meta={"subject_qid": fact["subject_qid"], "question": q},
    )
