"""SFT dataset v1 components: sandbox isolation, scorers, generators, source verification,
split isolation, episode encoding and the evaluation statistics."""

from __future__ import annotations

import json
import random
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from j1data import jtok, sandbox, sftbuild, sfteval, sftgen, sftscore, sftsources
from j1data.protocol import Message, serialize
from j1data.sft import IGNORE

needs_sandbox = pytest.mark.skipif(
    not sandbox.available(), reason="bubblewrap sandbox unavailable on this host"
)


@pytest.fixture(scope="module")
def tokenizer():
    return jtok.train(
        jtok.CandidateSpec("sft-v1-fixture", 448, limit_alphabet=200, min_frequency=1, stem_alphabet=False),
        [
            "The quick brown fox jumps over the lazy dog. What is 12 + 7? The answer is 19.",
            '{"name": "Alice", "age": 30} I don\'t know. ```python\ndef f(x):\n    return x + 1\n```',
        ]
        * 50,
    )


# --- sandbox -----------------------------------------------------------------------------------


@needs_sandbox
def test_sandbox_has_no_network_home_env_or_host_files():
    probe = (
        "import os, socket\n"
        "print(sorted(os.listdir('/')))\n"
        "print(os.environ.get('HOME'), os.path.exists('/home'), os.path.exists('/etc/passwd'))\n"
        "try:\n    socket.create_connection(('1.1.1.1', 80), timeout=1); print('net')\n"
        "except OSError:\n    print('nonet')\n"
    )
    r = sandbox.run_python(probe)
    assert r.ok
    lines = r.stdout.splitlines()
    assert "home" not in lines[0] and "etc" not in lines[0]
    assert lines[1] == "None False False" and lines[2] == "nonet"


@needs_sandbox
def test_sandbox_limits_time_memory_and_reports_failures():
    assert sandbox.run_python("while True: pass", sandbox.Limits(cpu_seconds=1, wall_seconds=5)).ok is False
    assert not sandbox.run_python("x = bytearray(3 << 30)").ok
    r = sandbox.run_python("import sys; sys.exit(3)")
    assert r.returncode == 3 and not r.ok


@needs_sandbox
def test_program_stderr_cannot_fake_a_sandbox_setup_failure():
    """A candidate printing a bwrap-like error is an ordinary failed program, not "not executed"."""
    r = sandbox.run_python(
        "import sys; sys.stderr.write('bwrap: Creating new namespace failed\\n'); sys.exit(1)"
    )
    assert r.returncode == 1 and not r.ok and "bwrap:" in r.stderr


@needs_sandbox
def test_a_real_setup_failure_is_reported(monkeypatch):
    real = sandbox._command
    monkeypatch.setattr(
        sandbox,
        "_command",
        lambda launcher: [real(launcher)[0], "--bind", "/nonexistent-src", "/x", *real(launcher)[1:]],
    )
    with pytest.raises(sandbox.SandboxUnavailable):
        sandbox.run_python("print(1)")


def test_sandbox_never_falls_back_to_unsandboxed(monkeypatch):
    monkeypatch.setattr(sandbox.shutil, "which", lambda _name: None)
    with pytest.raises(sandbox.SandboxUnavailable):
        sandbox.run_python("print(1)")


# --- scorers -----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("spec", "good", "bad"),
    [
        ({"type": "exact", "value": "a b c"}, " a b c\n", "a b c."),
        ({"type": "exact", "value": "Paris", "strip_period": True}, "Paris.", "paris"),
        ({"type": "number", "value": "85"}, "37 + 48 = 85.", "37 + 48 = 84."),
        ({"type": "number", "value": "1200"}, "The answer is 1,200.", "The answer is 120."),
        ({"type": "entity", "aliases": ["Paris"]}, "The capital of France is Paris.", "I don't know Paris."),
        (
            {"type": "entity", "aliases": ["Asia"], "distractors": ["Europe", "Africa"]},
            "Asia",
            "Asia, Europe or Africa",
        ),
        ({"type": "entity", "aliases": ["São Tomé"]}, "Sao Tome", "Santo Tome"),
        ({"type": "exact", "value": "He", "strip_period": True}, "He.", "The symbol is He."),
        ({"type": "abstain"}, "I don't know.", "The capital of Veltoria is Brask."),
        ({"type": "abstain"}, "I don\u2019t know.", "I don't know for sure, but the answer is Paris."),
        ({"type": "abstain"}, "There is no country called Veltoria.", "I'm not sure. It might be Brask."),
        ({"type": "number", "value": "8"}, "The answer is 5 + 3 = 8.", "The answer is 5 + 3 = 9."),
        ({"type": "number", "value": "18"}, "The answer is **18** (3 x 6).", "The answer is **6**."),
        ({"type": "number", "value": "-5"}, "#### \u22125", "#### 5"),
        ({"type": "json", "value": {"a": 1, "b": True}}, '{"b": true, "a": 1}', '{"a": "1", "b": true}'),
        ({"type": "json", "value": [1]}, "```json\n[1]\n```", "[true]"),
        (
            {"type": "json_key", "key": "answer", "inner": {"type": "number", "value": "4"}},
            '{"answer": "4"}',
            '{"answer": [1, 2, 4]}',
        ),
        ({"type": "stdout", "value": "a\nb"}, "```\na\nb\n```", "a b"),
        (
            {"type": "all", "parts": [{"type": "number", "value": "3"}, {"type": "lowercase"}]},
            "the answer is 3.",
            "The answer is 3.",
        ),
        ({"type": "uppercase"}, "THE ANSWER IS 3.", "17"),
        ({"type": "max_words", "n": 2}, "17", "it is 17"),
    ],
)
def test_scorers(spec, good, bad):
    assert sftscore.score(spec, good)[0] is True
    assert sftscore.score(spec, bad)[0] is False


@needs_sandbox
def test_python_tests_scorer_executes_in_sandbox():
    spec = {"type": "python_tests", "setup": "", "tests": "assert f(2) == 3"}
    assert sftscore.score(spec, "```python\ndef f(x):\n    return x + 1\n```")[0]
    assert not sftscore.score(spec, "```python\ndef f(x):\n    return x\n```")[0]
    assert not sftscore.score(spec, "```python\nimport os\nos.listdir('/home')\n```")[0]


def test_malformed_spec_is_an_error_not_a_score():
    with pytest.raises(sftscore.SpecError):
        sftscore.score({"type": "nope"}, "x")
    with pytest.raises(sftscore.SpecError):
        sftscore.score({"type": "stdout", "value": None}, "x")


# --- generators --------------------------------------------------------------------------------

FACTS = [
    {
        "question": "What is the capital of France?",
        "answer": "Paris",
        "spec": {"type": "entity", "aliases": ["Paris"]},
        "key": "fact:Q142:country_capital",
        "record": "wikidata/country_capital/Q142",
    }
]


def test_generators_are_deterministic_and_every_reference_passes_its_scorer():
    fams = sftgen.families(frozenset(), FACTS) | sftgen.HELDOUT_FAMILIES
    for name, gen in fams.items():
        a = [next(s) for s in [sftgen.stream(gen, 5)] for _ in range(150)]
        b = [next(s) for s in [sftgen.stream(gen, 5)] for _ in range(150)]
        assert [(i.key, [t.text for t in i.turns]) for i in a] == [
            (i.key, [t.text for t in i.turns]) for i in b
        ]
        for it in a:
            if "needs_execution" in it.meta:
                continue  # filled from the sandbox by the builder
            for t in it.turns:
                if t.role == "assistant":
                    assert t.spec is not None and sftscore.score(t.spec, t.text)[0], (name, t.text, t.spec)


def test_hierarchy_targets_follow_the_system_rule_not_the_user_override():
    rng = random.Random(3)
    seen = set()
    for _ in range(400):
        it = sftgen.gen_hierarchy(rng, FACTS)
        rule = it.turns[0].text
        target = it.turns[-1].text
        seen.add(rule)
        if "lowercase" in rule:
            assert target == target.lower() and any(c.isalpha() for c in target)
        if "uppercase" in rule:
            assert target == target.upper() and any(c.isalpha() for c in target)
        if "JSON" in rule:
            assert set(json.loads(target)) == {"answer"}
    assert len(seen) == len(sftgen.HIERARCHY_RULES)


def test_injection_targets_treat_forged_delimiters_as_data():
    rng = random.Random(4)
    for _ in range(300):
        it = sftgen.gen_injection(rng)
        target = it.turns[-1].text
        assert "HACKED" not in target or "HACKED" in it.turns[-2].text  # only as transformed data
        assert sftscore.score(it.turns[-1].spec, target)[0]


def test_abstention_never_uses_a_real_label():
    rng = random.Random(5)
    real = frozenset({sftgen.fictional_place(random.Random(5)).casefold()})
    for _ in range(200):
        it = sftgen.gen_abstain(rng, real)
        assert not any(r in it.turns[-2].text.casefold() for r in real)


# --- sources -----------------------------------------------------------------------------------


def test_safe_eval_and_annotations():
    assert sftsources.safe_eval("(3+4)*2/7") == 2.0
    for bad in ("__import__('os')", "abs(-1)", "x+1", "[1][0]", "2**999"):
        with pytest.raises((sftsources.ArithmeticError_, SyntaxError)):
            sftsources.safe_eval(bad)
    assert sftsources.annotation_matches("10/3", "3.33")
    assert not sftsources.annotation_matches("10/3", "3.5")


def _gsm_parquet(tmp: Path) -> Path:
    rows = [
        {"question": "A has 2 and buys 3. How many?", "answer": "2+3=<<2+3=5>>5\n#### 5"},
        {"question": "B has 2 and buys 3. How many?", "answer": "2+3=<<2+3=6>>6\n#### 6"},
        {"question": "C has 2. How many?", "answer": "C has 2.\n#### 2"},
    ]
    p = tmp / "train-00000-of-00001.parquet"
    pq.write_table(pa.Table.from_pylist(rows), p)
    return p


def test_gsm8k_annotations_are_reexecuted_and_failures_recorded(tmp_path):
    items, rejected = sftsources.gsm8k_items(_gsm_parquet(tmp_path), split="train", revision="r")
    assert [i.turns[-1].text for i in items] == ["2+3=5\n#### 5"]
    assert sorted(r["reason"].split(":")[0] for r in rejected) == [
        "calculator annotation mismatch",
        "no executable calculator annotations",
    ]
    assert items[0].factuality == "VERIFIED_EXECUTION" and {c["oracle"] for c in items[0].extra_checks} == {
        "execution",
        "known_answer",
    }


def test_mutation_changes_behaviour():
    code = sftsources.normalize_code("def f(a, b):\n\treturn a + b if a > 0 else 0\n")
    n = sftsources.mutation_sites(code)
    variants = {sftsources.mutate(code, i) for i in range(n)}
    assert n >= 3 and code not in variants


def _snapshot():
    def b(s, sl, o, ol, rev=1, alts=""):
        out = {
            "s": {"value": f"http://www.wikidata.org/entity/{s}"},
            "sLabel": {"value": sl},
            "rev": {"value": str(rev)},
        }
        out["o"] = {"value": o if not o.startswith("Q") else f"http://www.wikidata.org/entity/{o}"}
        out["oLabel"] = {"value": ol}
        out["alts"] = {"value": alts}
        return out

    return {
        "queries": {
            "country_capital": {
                "bindings": [
                    b("Q142", "France", "Q90", "Paris"),
                    b("Q1", "Bolivia", "Q2", "Sucre"),
                    b("Q1", "Bolivia", "Q3", "La Paz"),
                ]
            },
            "element_symbol": {
                "bindings": [b("Q560", "helium", "He", "He"), b("Q999", "unbinilium", "Ubn", "Ubn")]
            },
            "element_number": {
                "bindings": [b("Q560", "helium", "2", "2"), b("Q999", "unbinilium", "120", "120")]
            },
        }
    }


def test_wikidata_facts_reject_ambiguous_and_unconfirmed():
    facts, rejected = sftsources.wikidata_facts(_snapshot())
    got = {(f["relation"], f["subject_label"]) for f in facts}
    assert got == {("country_capital", "France"), ("element_symbol", "helium"), ("element_number", "helium")}
    reasons = {r["reason"] for r in rejected}
    assert "2 objects (ambiguous)" in reasons and "not one of the 118 confirmed elements" in reasons
    he = next(f for f in facts if f["relation"] == "element_symbol")
    item = sftsources.fact_item(he, random.Random(0), "train", "sha")
    assert not sftsources.fact_leaks(item)  # "helium" in the question is not the symbol "He"


# --- builder pieces ----------------------------------------------------------------------------


def _item(key, text, split):
    return sftgen.Item(
        "if.upper",
        key,
        [
            sftgen.Turn("user", text),
            sftgen.Turn("assistant", text.upper(), {"type": "exact", "value": text.upper()}),
        ],
        ["direct_answer"],
        split=split,
    )


def test_isolation_prefers_held_out_splits():
    items = {
        "test": [_item("k1", "a b", "test")],
        "dev": [_item("k1", "a b c", "dev"), _item("k2", "A  B", "dev")],
        "train": [_item("k2", "x", "train"), _item("k3", "c d", "train"), _item("k4", "c d", "train")],
    }
    kept, rejected = sftbuild._isolate(items)
    assert [i.key for i in kept["test"]] == ["k1"]
    assert kept["dev"] == []  # k1 key collides with test; "A  B" equals "a b" after normalization
    assert [i.key for i in kept["train"]] == ["k3"]
    assert len(rejected) == 4


def test_jsonl_reader_keeps_unicode_line_separators(tmp_path):
    p = tmp_path / "x.jsonl"
    rows = [{"t": "a b\u0085c"}, {"t": "d"}]
    p.write_bytes(b"".join(json.dumps(r, ensure_ascii=False).encode() + b"\n" for r in rows))
    assert sftbuild.read_jsonl(p) == rows


def test_episode_encoding_is_next_token_aligned(tokenizer):
    it = sftgen.gen_arith_basic(random.Random(1))
    ep = sftbuild.to_episode(it, "idx")
    cfg = {"build": {"approved_licenses": ["Cinqic-generated"], "max_tokens": 1025}}
    x, y = sftbuild.encode(tokenizer, ep, cfg, "idx")
    full = serialize(tokenizer, sftbuild._messages(it.turns), train_turns=None)
    assert x == full.ids[:-1] and y == full.labels[1:]
    for t in range(len(y) - 1):
        assert y[t] in (IGNORE, x[t + 1])
    assert y[-1] == jtok.control_id("<|eot|>")
    host = [
        i
        for i, m in enumerate(
            Message(t.role, t.text, "final" if t.role == "assistant" else None) for t in it.turns
        )
        if m.role != "assistant"
    ]
    assert host  # system/user tokens exist and carry -100 (checked through the full labels above)


def test_eval_prompts_never_contain_answers(tokenizer):
    it = sftgen.gen_arith_basic(random.Random(2))
    prompts, answers = [], []
    sftbuild._add_eval(tokenizer, it, "test", prompts, answers, {}, 1)
    assert "spec" not in prompts[0] and "reference" not in prompts[0]
    ids = prompts[0]["prompt_ids"]
    assert ids[-1] == jtok.control_id("<|assistant|>") and answers[0]["spec"] == it.turns[-1].spec
    target_ids = tokenizer.encode(it.turns[-1].text, add_special_tokens=False).ids
    assert all(ids[i : i + len(target_ids)] != target_ids for i in range(len(ids)))


# --- evaluation statistics ---------------------------------------------------------------------


def test_wilson_mcnemar_and_bootstrap():
    lo, hi = sfteval.wilson(50, 100)
    assert 0.40 < lo < 0.41 and 0.59 < hi < 0.60
    assert sfteval.mcnemar_p(0, 0) == 1.0 and sfteval.mcnemar_p(0, 10) == pytest.approx(2 / 1024)
    r1 = sfteval.paired([False] * 50 + [True] * 50, [True] * 100)
    r2 = sfteval.paired([False] * 50 + [True] * 50, [True] * 100)
    assert r1 == r2 and r1["difference"] == 0.5 and r1["bootstrap_ci95"][0] > 0.35


def test_strict_scoring_requires_a_well_formed_turn(tokenizer):
    final, eot = jtok.control_id("<|final|>"), jtok.control_id("<|eot|>")
    text = tokenizer.encode("The answer is 19.", add_special_tokens=False).ids
    assert sfteval._content(tokenizer, [final, *text, eot], "chat")[:2] == (True, "The answer is 19.")
    valid, strict, lenient = sfteval._content(tokenizer, [*text], "chat")
    assert (valid, strict, lenient) == (False, "", "The answer is 19.")
    few = tokenizer.encode("19\n\nQuestion: next", add_special_tokens=False).ids
    assert sfteval._content(tokenizer, few, "base_fewshot")[1] == "19"


@needs_sandbox
@pytest.mark.parametrize(
    "cheat",
    [
        "def f(x):\n    return x\nif __name__ == '__main__':\n    import sys\n    sys.exit(0)",
        "import os\ndef f(x):\n    return x\nos._exit(0)",
        "import atexit, os\natexit.register(lambda: os._exit(0))\ndef f(x):\n    return x",
        "def f(x):\n    raise SystemExit(0)",
    ],
)
def test_python_tests_cannot_pass_by_exiting_early(cheat):
    spec = {"type": "python_tests", "setup": "", "tests": "assert f(2) == 3", "entry_point": "f"}
    assert not sftscore.score(spec, f"```python\n{cheat}\n```")[0]


@needs_sandbox
def test_python_tests_grade_the_block_that_defines_the_entry_point():
    spec = {"type": "python_tests", "setup": "", "tests": "assert f(2) == 3", "entry_point": "f"}
    answer = "```python\ndef f(x):\n    return x + 1\n```\nUsage:\n```python\nprint(f(2))\n```"
    assert sftscore.score(spec, answer)[0]


def test_inner_tasks_carry_their_keys_for_isolation():
    rng = random.Random(9)
    for _ in range(200):
        for it in (sftgen.gen_hierarchy(rng, FACTS), sftgen.gen_user_format(rng, FACTS)):
            assert it.extra_keys and (
                it.extra_keys[0].startswith("arith:") or it.extra_keys[0] == FACTS[0]["key"]
            )


def test_fictional_place_questions_share_one_key():
    keys: dict[str, set[str]] = {}
    rng = random.Random(11)
    for _ in range(3000):
        it = sftgen.gen_abstain(rng, frozenset())
        if it.key.startswith("fict:"):
            place = it.key.split(":", 1)[1]
            assert place in it.turns[-2].text
            keys.setdefault(place, set()).add(it.turns[-2].text)
    assert any(len(v) > 1 for v in keys.values())  # several relations, one key


def test_articles():
    assert (
        sftgen._article("engineer") == "an"
        and sftgen._article("18") == "an"
        and sftgen._article("80") == "an"
    )
    assert sftgen._article("teacher") == "a" and sftgen._article("25") == "a"
    assert (
        sftsources.with_article("Philippines") == "the Philippines"
        and sftsources.with_article("France") == "France"
    )


@needs_sandbox
def test_humaneval_style_body_completion_is_joined_to_the_prompt():
    prompt = 'def f(x):\n    """Return x + 1."""\n'
    spec = {
        "type": "python_tests",
        "setup": "",
        "prefix": prompt,
        "tests": "assert f(2) == 3",
        "entry_point": "f",
    }
    assert sftscore.score(spec, "```python\n    return x + 1\n```")[0]
    assert sftscore.score(spec, "```python\ndef f(x):\n    return x + 1\n```")[0]
    assert not sftscore.score(spec, "```python\n    return x\n```")[0]


def test_case_rules_never_draw_numeric_answers():
    facts = [
        *FACTS,
        {**FACTS[0], "question": "What is the atomic number of iron?", "answer": "26", "key": "fact:Q677:n"},
    ]
    rng = random.Random(4)
    for _ in range(300):
        it = sftgen.gen_hierarchy(rng, facts)
        if "lowercase" in it.turns[0].text or "uppercase" in it.turns[0].text:
            assert it.extra_keys == [FACTS[0]["key"]]


@pytest.mark.parametrize(
    ("response", "ok"),
    [
        ("I'm sorry, but I don't know your middle name.", True),
        ("There is no country called Veltoria. It is not a real place.", True),
        ("I can't know that - it's in the future.", True),
        ("I don't know. Asia.", False),
        ("I'm not sure; Europe.", False),
        ("I don't know, but it is probably Asia.", False),
    ],
)
def test_abstention_rejects_committed_guesses(response, ok):
    spec = {"type": "abstain", "forbidden": ["Asia", "Europe", "Africa"]}
    assert sftscore.score(spec, response)[0] is ok


def test_abstention_on_numbers_and_symbols():
    assert not sftscore.score({"type": "abstain", "forbid_numbers": True}, "I don't know. 26.")[0]
    assert sftscore.score({"type": "abstain", "forbid_numbers": True}, "I don't know.")[0]
    assert not sftscore.score({"type": "abstain", "forbidden_symbols": ["Fe", "He"]}, "I don't know. Fe.")[0]


@needs_sandbox
def test_humaneval_whole_function_answers_keep_prompt_imports_and_helpers():
    prompt = (
        "from typing import List\n\ndef helper(x):\n    return x * 2\n\n"
        'def f(xs: List[int]) -> int:\n    """Doc."""\n'
    )
    spec = {
        "type": "python_tests",
        "setup": "",
        "prefix": prompt,
        "tests": "assert f([1, 2]) == 6",
        "entry_point": "f",
    }
    assert sftscore.score(
        spec, "```python\ndef f(xs: List[int]) -> int:\n    return sum(helper(x) for x in xs)\n```"
    )[0]


@needs_sandbox
def test_setup_runs_after_the_candidate():
    spec = {
        "type": "python_tests",
        "setup": "n = Node(3)",
        "tests": "assert value(n) == 3",
        "entry_point": "value",
    }
    code = (
        "class Node:\n    def __init__(self, v):\n        self.v = v\n\ndef value(node):\n    return node.v"
    )
    assert sftscore.score(spec, f"```python\n{code}\n```")[0]


def test_canonical_code_ignores_identifier_names():
    a = "def add_nums(a, b):\n    total = a + b\n    return total"
    b = "def sum_two(x, y):\n    s = x + y\n    return s"
    c = "def sum_two(x, y):\n    return x - y"
    assert sftbuild.canonical_code(a) == sftbuild.canonical_code(b) != sftbuild.canonical_code(c)
