"""j1data.convadmit: review binding, bucket caps, near-duplicates, templates, contamination, serialization."""

from __future__ import annotations

from typing import Any

import pytest

from j1data import convadmit as ca
from j1data import convreview as cr
from j1data import jtok
from j1data.hashing import sha256_json
from j1data.sft import IGNORE

GOOD = dict.fromkeys(cr.ITEMS, 2)


@pytest.fixture(scope="module")
def tok() -> Any:
    t = jtok.train(
        jtok.CandidateSpec("conv-fixture", 448, limit_alphabet=200, min_frequency=1, stem_alphabet=False),
        [
            "How long should green tea steep? About two minutes in hot water. Keep it short and friendly.",
            "The quick brown fox jumps over the lazy dog. Color, behavior, organization. 0123456789",
            "What is 2 + 2? Answer number one, two, three. Please list the steps for me now!",
        ]
        * 50,
    )
    return jtok.load(t.to_str())


def _draft(
    n: int, user: str, bucket: str = "writing_revision", system: str = "", answer: str = ""
) -> dict[str, Any]:
    return {
        "content_sha256": f"{n:016x}" + "0" * 48,
        "slot": {"bucket": bucket, "slot": f"{bucket}-{n:03d}"},
        "conversation": {
            "system": system,
            "turns": [
                {"role": "user", "text": user},
                {"role": "assistant", "text": answer or f"Answer number {n}."},
            ],
        },
        "provenance": {"artifact_sha256": "a" * 64, "route": "B"},
    }


def _reviews(
    drafts: list[dict[str, Any]], reject: tuple[int, ...] = ()
) -> tuple[dict[str, Any], dict[str, Any]]:
    revs = []
    for i, d in enumerate(drafts):
        s = GOOD | ({"correctness": 0} if i in reject else {})
        revs.append(
            {
                "id": cr.review_id(d),
                "scores": s,
                "decision": cr.rule(s),
                "reasons": "",
                "checks": "",
                "sources": ["s"],
            }
        )
    first = cr.ingest(cr.draft_items(drafts), revs, "first")
    first["ingest_sha256"] = sha256_json(first)
    ids = cr.sample(drafts, first)
    buckets = dict(cr.draft_items(drafts))
    srev = [{**r, "id": cr.salted(r["id"])} for r in revs if r["id"] in ids]
    second = cr.second(ids, buckets, srev, "second", first["ingest_sha256"])
    return first, second


CFG = {"buckets": {"writing_revision": {"count": 3}, "corrections": {"count": 2}}}


def _plan(drafts: list[dict[str, Any]]) -> dict[str, Any]:
    return {"slots": [d["slot"] for d in drafts]}


def test_admission_gates(tok: Any) -> None:
    ds = [
        _draft(0, "How long should green tea steep for the best taste?"),
        _draft(1, "How long should green tea steep for the best taste?"),  # near-duplicate of 0
        _draft(2, "What temperature for black tea?"),  # first review rejects
        _draft(4, "Is oolong closer to green or black tea in flavor?"),
        _draft(5, "Can I reuse tea leaves for a second cup at home?", system="Keep it short."),
        _draft(6, "Should I rinse loose leaves first?"),  # bucket full
        _draft(7, "You said tea has no caffeine, right?", bucket="corrections"),
        _draft(
            8, "Quick one: what is the capital of Lesotho anyway?", bucket="corrections"
        ),  # short eval prompt
    ]
    first, second = _reviews(ds, reject=(2,))
    protected = ca.Protected(
        ["What is the capital of Lesotho?", "a much longer evaluation prompt with many words in it"]
    )
    admitted, rejected = ca.admit(CFG, _plan(ds), ds, first, second, protected, tok, "ident")
    got = {d["slot"]["slot"] for _, _, d, _ in admitted}
    assert got == {"writing_revision-000", "writing_revision-004", "writing_revision-005", "corrections-007"}
    why = {r["slot"]: r["reason"] for r in rejected}
    assert why["writing_revision-001"] == "near-duplicate of an admitted record"
    assert why["writing_revision-002"] == "first review rejected"
    assert why["writing_revision-006"] == "bucket full"
    assert why["corrections-008"] == "overlaps an evaluation or sealed confirmation prompt"
    for split, ep, d, (x, y) in admitted:
        assert split in ("train", "dev")
        trained = [t for t in y if t != IGNORE]
        assert trained and len(x) == len(y)
        if d["conversation"]["system"]:
            assert ep["messages"][0]["role"] == "system" and ep["messages"][0]["train"] is False
        assert ep["validation"]["factuality"] == "MODEL_JUDGED_ONLY"
        assert ep["validation"]["checks"][-1]["detail"].startswith("1 sources; sha256 ")  # no reviewer text
        rec = ca.record(split, ep, d, (x, y))
        assert rec["license"] == ca.LICENSE and rec["target_tokens"] == len(trained) and rec["route"] == "B"


def test_short_first_turns_are_not_near_duplicates(tok: Any) -> None:
    ds = [_draft(0, "How many cups are in a quart?"), _draft(1, "How many cups are in a gallon?")]
    first, second = _reviews(ds)
    admitted, _ = ca.admit(CFG, _plan(ds), ds, first, second, ca.Protected([]), tok, "ident")
    assert len(admitted) == 2


def test_template_repetition(tok: Any) -> None:
    opener = "Sure thing here is the answer for you: "
    ds = [
        _draft(i, f"Distinct question {i} about item {i * 7}?", answer=opener + f"item {i}.")
        for i in range(5)
    ]
    cfg = {"buckets": {"writing_revision": {"count": 10}}}
    first, second = _reviews(ds)
    admitted, rejected = ca.admit(cfg, _plan(ds), ds, first, second, ca.Protected([]), tok, "ident")
    assert len(admitted) == ca.TEMPLATE_MAX_REPEATS
    assert {r["reason"] for r in rejected} == {"template repetition"}


def test_reviews_must_be_bound(tok: Any) -> None:
    ds = [_draft(0, "How long should green tea steep?"), _draft(1, "Why is the sky blue at noon?")]
    first, second = _reviews(ds)
    p = ca.Protected([])
    with pytest.raises(SystemExit, match="not bound"):
        ca.admit(CFG, _plan(ds), ds, first, second | {"first_ingest_sha256": "x"}, p, tok, "i")
    with pytest.raises(SystemExit, match="exactly the registered"):
        ca.admit(CFG, _plan(ds), ds, first, second | {"sample_ids": [], "decisions": {}}, p, tok, "i")
    with pytest.raises(SystemExit, match="did not pass"):
        ca.admit(CFG, _plan(ds), ds, first, second | {"passed": False}, p, tok, "i")
    admitted, rejected = ca.admit(CFG, _plan(ds), ds, first, second, p, tok, "i", lambda _t: True)
    assert not admitted and {r["reason"] for r in rejected} == {"matches the layers-v1 benchmark index"}


def test_model_judged_path_is_narrow(tok: Any) -> None:
    from j1data import sft

    ds = [_draft(0, "How long should green tea steep?")]
    first, _ = _reviews(ds)
    ep = ca.episode(ds[0], first["decisions"][cr.review_id(ds[0])], "i")
    assert sft._reviewed_teacher_conversation(ep)
    assert not sft._reviewed_teacher_conversation(ep | {"task_family": "facts.capital"})
    assert not sft._reviewed_teacher_conversation(
        ep | {"provenance": {**ep["provenance"], "origin": "human"}}
    )
    unreviewed = [c for c in ep["validation"]["checks"] if c["name"] != "rubric_first_review"]
    assert not sft._reviewed_teacher_conversation(
        ep | {"validation": {**ep["validation"], "checks": unreviewed}}
    )
