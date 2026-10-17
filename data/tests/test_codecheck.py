"""j1.codecheck.v1: out-of-process I/O checker (incl. adversarial candidates), value encoding and comparison,
functional-twin detection, quarantine policy and wrong-code controls."""

from __future__ import annotations

import math

import pytest

from j1data import codecheck as cc
from j1data import sandbox

needs_sandbox = pytest.mark.skipif(
    not sandbox.available(), reason="bubblewrap sandbox unavailable on this host"
)

REFERENCE = "def total(xs, k):\n    return sum(xs) * k\n"
SPEC = {
    "type": "python_io",
    "entry_point": "total",
    "inputs": [[[1, 2, 3], 2], [[], 5], [[-4, 4, 7], 1], [[10], 0]],
    "reference": REFERENCE,
}


# --- encoding and comparison (pure) ----------------------------------------------------------------


def test_encoding_round_trips_and_keeps_types_apart():
    value = {
        "a": (1, 2.5, None),
        3: [True, "x", b"\x00\xff"],
        "s": {1, 2},
        "f": frozenset({"q"}),
        "c": 1 + 2j,
    }
    enc = cc.encode_value(value)
    assert cc.decode_value(enc) == value
    assert cc.encode_value((1, 2)) != cc.encode_value([1, 2])
    assert cc.encode_value(True) is True and cc.encode_value(1) == 1
    # set/dict encodings are canonical: insertion order does not matter
    assert cc.encode_value({3, 1, 2}) == cc.encode_value({2, 3, 1})
    assert cc.encode_value({"b": 1, "a": 2}) == cc.encode_value({"a": 2, "b": 1})


def test_subclasses_are_read_through_the_base_type():
    class Liar(int):
        def __eq__(self, other):
            return True

        __hash__ = int.__hash__

    class Stretchy(list):
        def __iter__(self):
            return iter([1, 2, 3, 4])

    assert cc.encode_value(Liar(7)) == 7
    assert cc.encode_value(Stretchy([9])) == ["L", 9]

    class Opaque:
        pass

    with pytest.raises(cc.Unserializable):
        cc.encode_value(Opaque())
    with pytest.raises(cc.Unserializable):
        cc.encode_value(x for x in range(3))


def test_values_equal_semantics():
    e = cc.encode_value
    assert cc.values_equal(e(3), e(3.0))
    assert cc.values_equal(e(True), e(1))  # Python == semantics for numbers
    assert not cc.values_equal(e([1, 2]), e((1, 2)))
    assert cc.values_equal(e(0.1 + 0.2), e(0.3))
    assert not cc.values_equal(e(0.3001), e(0.3))
    assert cc.values_equal(e(0.3001), e(0.3), cc.FloatTol(abs=1e-3))
    assert cc.values_equal(e(float("nan")), e(float("nan")))
    assert not cc.values_equal(e(float("inf")), e(1e308))
    assert cc.values_equal(e([3, 1, 2]), e([1, 2, 3]), compare="set")
    assert not cc.values_equal(e([1, 1, 2]), e([1, 2]), compare="multiset")
    assert cc.values_equal(e({"a": [1.0]}), e({"a": [1]}))
    assert not cc.values_equal(["H", "ab", 3], ["H", "cd", 3])


def test_safe_literal_accepts_literals_only():
    assert cc.safe_literal("[(1, -2.5), {'a': inf}, float('nan'), set()]")[0] == (1, -2.5)
    assert math.isinf(cc.safe_literal("float('-inf')"))
    assert cc.safe_literal("frozenset([1, 2])") == frozenset({1, 2})
    for bad in ("__import__('os')", "open('/etc/passwd')", "[x for x in y]", "a + b", "f(1)"):
        with pytest.raises(ValueError):
            cc.safe_literal(bad)


# --- task helpers (pure) -------------------------------------------------------------------------------


def test_harvest_inputs_from_mbpp_and_humaneval_style_tests():
    mbpp = {
        "entry_point": "f",
        "tests": "assert f([1, 2], 3) == 4\nassert set(f((5,), -1)) == {1}\nassert f(x, 1) == 2",
    }
    assert cc.harvest_inputs(mbpp) == [[[1, 2], 3], [(5,), -1]]
    he = {
        "entry_point": "g",
        "callee": "candidate",
        "tests": "def check(candidate):\n    assert candidate('ab') == 'ba'\n"
        "    assert abs(candidate('') - 1) < 1e-6\n",
    }
    assert cc.harvest_inputs(he) == [["ab"], [""]]


def test_mutated_inputs_are_deterministic_type_preserving_and_new():
    seeds = [[[1, 2, 3], "abc", (4, 5)], [[7], "z", (1,)]]
    a = cc.mutate_inputs(seeds, 30, "t1")
    assert a == cc.mutate_inputs(seeds, 30, "t1")
    assert a != cc.mutate_inputs(seeds, 30, "t2")
    assert len(a) == 30
    for x in a:
        assert isinstance(x[0], list) and isinstance(x[1], str) and isinstance(x[2], tuple)
        assert all(isinstance(v, int) for v in x[0])
        assert x not in seeds


def test_task_ids_for_dataset_records():
    assert cc.task_id_for_record("mbpp/full/train/601#fix") == "mbpp:601"
    assert cc.task_id_for_record("mbpp/full/validation/533") == "mbpp:533"
    assert cc.task_id_for_record("HumanEval/12") == "humaneval:12"
    assert cc.task_id_for_record("wikidata/country_capital/Q1") is None


def test_statement_and_structure_features():
    a = cc.statement_features("Write a function to find the sum of 3 numbers in a list.")
    b = cc.statement_features("Write a python function to find the sum of 5 numbers in a list.")
    assert cc._jaccard(a["w3n"], b["w3n"]) > cc._jaccard(a["w3"], b["w3"])
    assert cc._jaccard(a["c5"], b["c5"]) > 0.7
    s1 = cc.structure_keys("def area(l, b):\n    return l * b\n")
    s2 = cc.structure_keys("def multiply_int(x, y):\n    '''doc'''\n    return x * y\n")
    s3 = cc.structure_keys("def f(x, y):\n    return x * y * 2\n")
    assert s1["same_shape"] == s2["same_shape"]
    assert s1["same_code"] != s3["same_code"]


def _ev(own_ab, own_ba, mut_ab=None, mut_ba=None, stmt=0.0, same_code=False):
    ev = {
        "statement": {"w3": stmt, "w3n": stmt, "c5": stmt, "w1n": stmt},
        "structure": {"same_code": same_code, "same_shape": same_code},
        "compatible": True,
        "own_a_on_b": own_ab,
        "own_b_on_a": own_ba,
    }
    if mut_ab is not None:
        ev["mut_a_on_b"], ev["mut_b_on_a"] = mut_ab, mut_ba
    return ev


def test_verdict_rules():
    v = cc._verdict
    varied = {"nonmodal": 10, "nonmodal_agree": 10}
    assert v(_ev({"strict": 3}, {"loose": 3}, {"strict": 20, **varied}, {"strict": 20}))[0] == cc.TWIN
    # agreeing on both tasks' own tests is not enough without the mutated inputs
    assert v(_ev({"strict": 3, **varied}, {"strict": 3, **varied}))[0] != cc.TWIN
    # identical on every input but the outputs are (almost) constant: not enough to call it a twin
    assert v(_ev({"strict": 3}, {"strict": 3}, {"strict": 20}, {"strict": 20}))[0] != cc.TWIN
    # one mutated input where only one errors: partial domain overlap
    verdict, reasons = v(
        _ev({"strict": 3}, {"strict": 3}, {"strict": 20, "err": 1, **varied}, {"strict": 20})
    )
    assert verdict == cc.AMBIGUOUS and "a_passes_b_tests" in reasons
    # passes the other task's three weak tests but disagrees on its mutated inputs: coincidence
    verdict, reasons = v(_ev({"strict": 3}, {"differ": 3}, {"differ": 15, "strict": 5}, {"differ": 20}))
    assert verdict == cc.DISTINCT and reasons == ["coincidental_test_pass"]
    # a boolean predicate agreeing on mostly-False outputs is not evidence
    chance = {"strict": 45, "differ": 5, "nonmodal": 6, "nonmodal_agree": 1}
    assert v(_ev({"differ": 3}, {"differ": 3}, chance, {}))[0] == cc.DISTINCT
    assert v(_ev({"differ": 3}, {"differ": 3}))[0] == cc.DISTINCT
    assert v(_ev({"differ": 3}, {"differ": 3}, stmt=0.9)) == (
        cc.AMBIGUOUS,
        ["statement_near_duplicate"],
    )
    # new statement measures alone do not override behaviour that was compared and differs
    sibling = _ev({"differ": 3}, {"differ": 3}, {"differ": 9}, {"differ": 9})
    sibling["statement"] = {"w3": 0.2, "w3n": 0.2, "c5": 0.8, "w1n": 0.5}
    assert v(sibling) == (cc.DISTINCT, ["statement_similar"])
    assert v(_ev({"differ": 3}, {"differ": 3}, same_code=True))[0] == cc.AMBIGUOUS


def test_quarantine_policies():
    pairs = [
        {"a": "mbpp:1", "b": "mbpp:2", "verdict": cc.TWIN, "reasons": []},
        {"a": "mbpp:2", "b": "mbpp:500", "verdict": cc.AMBIGUOUS, "reasons": []},
        {"a": "mbpp:3", "b": "mbpp:2", "verdict": cc.AMBIGUOUS, "reasons": []},
        {"a": "mbpp:4", "b": "mbpp:501", "verdict": cc.DISTINCT, "reasons": []},
    ]
    open_ids, protected = ["mbpp:1", "mbpp:2", "mbpp:3", "mbpp:4"], ["mbpp:500", "mbpp:501"]
    assert cc.quarantine(pairs, open_ids, protected, transitive="none") == {"mbpp:2": ["mbpp:500"]}
    assert cc.quarantine(pairs, open_ids, protected) == {"mbpp:1": ["mbpp:500"], "mbpp:2": ["mbpp:500"]}
    assert set(cc.quarantine(pairs, open_ids, protected, transitive="all")) == {"mbpp:1", "mbpp:2", "mbpp:3"}
    assert cc.clusters(pairs) == [["mbpp:1", "mbpp:2", "mbpp:3", "mbpp:500"]]
    general = {"a": "humaneval:1", "b": "mbpp:2", "verdict": cc.AMBIGUOUS, "reasons": ["a_passes_b_tests"]}
    assert cc.link_kind(general, "mbpp:2") == "protected_reference_passes_open_tests"
    assert cc.link_kind(general, "humaneval:1") == "open_reference_passes_protected_tests"


def test_wrong_programs_cover_the_requested_kinds():
    record = {
        "id": "mbpp:9",
        "entry_point": "count_pos",
        "code": "def count_pos(xs):\n    n = 0\n    for x in xs:\n        if x > 0:\n            n += 1\n"
        "        else:\n            n += 0\n    return n\n",
    }
    kinds = {w["kind"] for w in cc.wrong_programs(record, 3, "s")}
    assert {"off_by_one", "return_constant", "return_first_expected", "swapped_branches"} <= kinds
    assert "empty_input_mishandled" in kinds


# --- out-of-process checker (sandbox) -----------------------------------------------------------------


@needs_sandbox
def test_checker_accepts_correct_and_rejects_wrong_code():
    assert cc.score_python_io(SPEC, f"```python\n{REFERENCE}```")[0]
    assert cc.score_python_io(SPEC, "def total(xs, k):\n    return sum(xs) + k")[0] is False
    literal = {
        "type": "python_io",
        "entry_point": "total",
        "inputs_literal": "[([1], 2), ([], 1)]",
        "expected": [2, 0],
    }
    assert cc.score_python_io(literal, REFERENCE)[0]
    # HumanEval style: the prompt prefix supplies imports and helpers
    he = {
        "entry_point": "g",
        "prefix": "import math\n\ndef g(x):\n    pass\n",
        "inputs": [[4]],
        "expected": [2.0],
    }
    assert cc.score_python_io(he, "def g(x):\n    return math.sqrt(x)")[0]
    assert cc.score_python_io(SPEC, "def total(xs, k) return")[1] == "candidate does not parse"


EXPLOITS = cc.EXPLOIT_PROGRAMS


@needs_sandbox
@pytest.mark.parametrize("name", sorted(EXPLOITS))
def test_known_in_process_exploits_do_not_pass(name):
    fast = cc.RunConfig(
        call_seconds=1.0, load_seconds=2.0, limits=sandbox.Limits(cpu_seconds=5, wall_seconds=8)
    )
    passed, detail = cc.score_python_io(SPEC, f"```python\n{EXPLOITS[name]}```", fast)
    assert passed is False, (name, detail)


@needs_sandbox
def test_exploit_prints_do_not_disturb_a_correct_candidate():
    noisy = (
        "import os\nprint('PASSED')\ndef total(xs, k):\n    os.write(1, b'garbage line\\n')\n"
        "    print('x')\n    return sum(xs) * k\n"
    )
    assert cc.score_python_io(SPEC, noisy)[0]


@needs_sandbox
def test_arguments_are_deep_copied_per_call():
    spec = {"entry_point": "f", "inputs": [[[1]], [[1]]], "reference": "def f(xs):\n    return len(xs)\n"}
    mutating = "def f(xs):\n    xs.append(0)\n    return len(xs) - 1\n"
    assert cc.score_python_io(spec, mutating)[0]
    shared = "STATE = []\ndef f(xs):\n    STATE.append(1)\n    return len(STATE)\n"
    assert not cc.score_python_io(spec, shared)[0]


@needs_sandbox
def test_floats_timeouts_recursion_and_unordered():
    spec = {
        "entry_point": "f",
        "inputs": [[0.1], [0.2]],
        "reference": "def f(x):\n    return x * 3\n",
        "float_tol": 1e-9,
    }
    assert cc.score_python_io(spec, "def f(x):\n    return x + x + x\n")[0]
    assert not cc.score_python_io(spec, "def f(x):\n    return round(x * 3, 1) + 1e-6\n")[0]
    slow = cc.RunConfig(call_seconds=0.5, limits=sandbox.Limits(cpu_seconds=5, wall_seconds=8))
    ok, detail = cc.score_python_io(spec, "def f(x):\n    while True:\n        pass\n", slow)
    assert not ok and "timeout" in detail
    deep = {
        "entry_point": "f",
        "inputs": [[50]],
        "reference": "def f(n):\n    return 0 if n == 0 else 1 + f(n - 1)\n",
    }
    assert cc.score_python_io(deep, deep["reference"])[0]
    assert not cc.score_python_io(deep, "def f(n):\n    return f(n + 1)\n")[0]
    unordered = {
        "entry_point": "f",
        "inputs": [[[3, 1, 2]]],
        "reference": "def f(xs):\n    return sorted(xs)\n",
        "compare": "set",
    }
    assert cc.score_python_io(unordered, "def f(xs):\n    return list(reversed(sorted(xs)))\n")[0]


@needs_sandbox
def test_large_outputs_compare_by_digest():
    spec = {"entry_point": "f", "inputs": [[5000]], "reference": "def f(n):\n    return list(range(n))\n"}
    assert cc.score_python_io(spec, "def f(n):\n    return [i for i in range(n)]\n")[0]
    assert not cc.score_python_io(spec, "def f(n):\n    return list(range(n - 1)) + [0]\n")[0]


@needs_sandbox
def test_one_crashing_call_does_not_hide_the_others():
    code = "import os\ndef f(x):\n    if x == 3:\n        os._exit(0)\n    return [x] * 300\n"
    out = cc.run_io(code, "f", [[i] for i in range(60)])
    assert [o.kind for o in out].count("crash") == 1 and out[3].kind == "crash"
    assert out[59].value == ["L", *([59] * 300)]  # far more output than the sandbox keeps per process


@needs_sandbox
def test_functional_twins_on_small_tasks():
    def rec(tid, split, text, code, entry, tests):
        return {
            "id": tid,
            "source": "mbpp",
            "split": split,
            "text": text,
            "code": code,
            "entry_point": entry,
            "callee": entry,
            "setup": "",
            "tests": tests,
            "executable": True,
        }

    a = rec(
        "mbpp:900",
        "train",
        "Write a function to add up a list.",
        "def s(xs):\n    return sum(xs)\n",
        "s",
        "assert s([1, 2]) == 3\nassert s([]) == 0\nassert s([5, 5, 5]) == 15",
    )
    b = rec(
        "mbpp:11",
        "test",
        "Total of the elements of a sequence.",
        "def total(ys):\n    t = 0\n    for y in ys:\n        t += y\n    return t\n",
        "total",
        "assert total([4]) == 4\nassert total([1, -1, 9]) == 9",
    )
    c = rec(
        "mbpp:12",
        "test",
        "Product of a list.",
        "def p(xs):\n    r = 1\n    for x in xs:\n        r *= x\n    return r\n",
        "p",
        "assert p([2, 3]) == 6\nassert p([1]) == 1",
    )
    pairs = cc.functional_twins([a], [b, c], mutations=12, workers=2)
    verdicts = {(p["a"], p["b"]): p["verdict"] for p in pairs}
    assert verdicts[("mbpp:11", "mbpp:900")] == cc.TWIN
    assert verdicts.get(("mbpp:12", "mbpp:900"), cc.DISTINCT) == cc.DISTINCT
    assert cc.quarantine(pairs, ["mbpp:900"], ["mbpp:11", "mbpp:12"]) == {"mbpp:900": ["mbpp:11"]}


@needs_sandbox
def test_strong_suite_verifies_inputs_and_grades_by_tier():
    record = {
        "id": "mbpp:77",
        "source": "mbpp",
        "split": "test",
        "text": "Return the largest element.",
        "code": "def biggest(xs):\n    return max(xs)\n",
        "entry_point": "biggest",
        "callee": "biggest",
        "setup": "",
        "tests": "assert biggest([1, 5, 2]) == 5\nassert biggest([3]) == 3",
        "executable": True,
    }
    plus = {
        "inputs": [[[-3, -9]], [[7, 7, 1]], [[0, 2]], [[]]],
        "results": [-3, 7, 99, None],  # the third recorded result disagrees with the reference: dropped
        "atol": 0.0,
        "unordered": False,
        "reference": "def biggest(xs):\n    return max(xs)\n",
        "entry_point": "biggest",
    }
    suite = cc.strong_suite(record, plus, mutation_inputs=10)
    tiers = suite["spec"]["tiers"]
    assert tiers.count("own") == 2 and tiers.count("plus") == 2
    assert suite["stats"]["plus_disagrees_with_evalplus"] == 1
    assert suite["stats"]["plus_reference_failed_or_unstable"] == 1  # max([]) raises
    assert all(ok for ok, _ in cc.tiered_check(suite, record["code"]).values())
    positive_only = (
        "def biggest(xs):\n    m = 0\n    for x in xs:\n        m = x if x > m else m\n    return m\n"
    )
    result = cc.tiered_check(suite, positive_only)
    assert result["own"][0] and not result["plus"][0] and not result["all"][0]
    kinds = [w["kind"] for w in cc.wrong_programs(record, cc.encode_value(5), "x")]
    assert "return_first_expected" in kinds
