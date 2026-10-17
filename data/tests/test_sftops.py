"""Invariants of the v1.3 compositional families (j1data.sftops)."""

import json
import random
import re

import pytest

from j1data import sftops as S
from j1data.sftscore import score


def _items(gen, n=600, seed=3):
    rng = random.Random(seed)
    return [gen(rng) for _ in range(n)]


@pytest.mark.parametrize("name", sorted(S.TRAIN_FAMILIES) + sorted(S.TEST_FAMILIES))
def test_every_reference_scores_1_and_generation_is_deterministic(name):
    gen = {**S.TRAIN_FAMILIES, **S.TEST_FAMILIES}[name]
    a, b = _items(gen, 300), _items(gen, 300)
    assert [(i.key, [t.text for t in i.turns]) for i in a] == [(i.key, [t.text for t in i.turns]) for i in b]
    for it in a:
        for t in it.turns:
            if t.role == "assistant":
                ok, detail = score(t.spec, t.text)
                assert ok, (it.family, t.text, detail)


def test_training_never_uses_reserved_phrasings_frames_or_pairs():
    for it in _items(S.gen_op, 3000):
        assert not it.meta["reserved_phrasing"]
        assert it.meta["frame"] in S.TRAIN_FRAMES
    for it in _items(S.gen_compose, 3000):
        a, b = it.meta["ops"]
        assert not S.heldout_pair(a, b)
        assert it.meta["frame"] in S.TRAIN_COMPOSE


def test_paraphrase_test_items_are_unseen_in_phrasing_or_frame():
    for it in _items(S.gen_paraphrase_test, 2000):
        assert it.meta["reserved_phrasing"] or it.meta["frame"] in S.TEST_FRAMES


def test_heldout_compositions_are_reserved_pairs():
    for it in _items(S.gen_compose_heldout, 2000):
        assert S.heldout_pair(*it.meta["ops"])


def test_heldout_operations_are_not_trained():
    assert not set(S.OPS) & set(S.HOPS)
    assert set(S.V12_HELDOUT) <= set(S.HOPS)
    for it in _items(S.gen_heldout_op, 1000):
        assert it.family.startswith(("heldout.ops.v12.", "heldout.ops.new."))
        assert it.meta["ops"][0] in S.HOPS


def test_behaviors_are_in_the_episode_schema():
    import json
    from pathlib import Path

    schema = json.loads((Path(__file__).resolve().parents[1] / "schemas/episode.schema.json").read_text())
    allowed = set(schema["properties"]["behaviors"]["items"]["enum"])
    for gen in {**S.TRAIN_FAMILIES, **S.TEST_FAMILIES}.values():
        for it in _items(gen, 100):
            assert set(it.behaviors) <= allowed, (it.family, it.behaviors)


def test_coverage_declares_real_reservations():
    c = S.coverage()
    assert c["phrasings_reserved"] >= 15 and c["pairs_heldout"] >= 40
    assert all(any(not S.reserved_phrasing(o, i) for i in range(len(o.phrasings))) for o in S.TRAIN_OPS)


def test_targets_are_never_free_text():
    """Rights R00/R13: targets are computed values, input words or the approved abstention phrase."""
    for gen in S.TRAIN_FAMILIES.values():
        for it in _items(gen, 400):
            users = " ".join(t.text for t in it.turns if t.role == "user")
            for t in it.turns:
                if t.role == "assistant" and t.text != "I don't know.":
                    text = re.sub(r"[\n,|\[\]\"]", " ", t.text)
                    words = [w.strip(".-").lower() for w in text.split()]
                    stray = [w for w in words if w and not w.isdigit() and w not in users.lower()]
                    assert not stray, (it.family, t.text, stray)


FACTS = [
    {"question": "What is the capital of France?", "answer": "Paris", "key": "f1", "record": "r1",
     "spec": {"type": "entity", "aliases": ["Paris"], "distractors": ["Lyon"]}},
    {"question": "Which continent is Peru on?", "answer": "South America", "key": "f2", "record": "r2",
     "spec": {"type": "entity", "aliases": ["South America"], "distractors": ["Asia"]}},
]  # fmt: skip


def _skeleton(template: str) -> re.Pattern[str]:
    parts = re.split(r"\{(?:it|p|q|p_ord)\}", template.rstrip("?"))
    return re.compile(".+?".join(re.escape(x) for x in parts), re.IGNORECASE)


def test_reserved_phrasings_never_occur_in_any_training_family():
    """Admission review F2: a reserved paraphrase must not reach training through another family."""
    from j1data import sftgen

    fams = sftgen.families(frozenset(), FACTS, {}) | S.TRAIN_FAMILIES
    users = []
    for gen in fams.values():
        users += [t.text for it in _items(gen, 1500, seed=11) for t in it.turns if t.role == "user"]
    reserved = [
        _skeleton(o.phrasings[i])
        for o in S.TRAIN_OPS
        for i in range(len(o.phrasings))
        if S.reserved_phrasing(o, i)
    ]
    hits = [(r.pattern, u) for r in reserved for u in users if r.search(u)]
    assert not hits, hits[:3]


def test_no_training_item_uses_a_heldout_pair():
    """Admission review F1/F3: chained turns and compositions avoid held-out pairs, incl. sort->reverse."""
    assert S.heldout_pair("sort", "reverse")
    for gen in (S.gen_conversation, S.gen_compose):
        for it in _items(gen, 4000, seed=5):
            ops = it.meta["ops"]
            if len(ops) == 2 and (it.family == "if.compose" or it.meta.get("chained")):
                assert not S.heldout_pair(*ops), (it.family, ops)


def test_nth_never_asks_for_the_first_word():
    for it in _items(S.gen_op, 3000, seed=9):
        if it.meta["ops"] == ["nth"]:
            assert json.loads(it.key.split(":", 2)[2].rsplit(":", 1)[0])["p"] >= 2


def test_context_items_hold_at_most_one_unanswerable_question():
    for it in _items(S.gen_context_qa, 2000, seed=4):
        assert sum(t.text == "I don't know." for t in it.turns if t.role == "assistant") <= 1
