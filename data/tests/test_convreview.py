"""j1data.convreview: rubric rule, ingest validation, blinding, stratified second-review sample."""

from __future__ import annotations

from typing import Any

import pytest

from j1data import convreview as cr
from j1data.hashing import sha256_json

GOOD = dict.fromkeys(cr.ITEMS, 2)


def _draft(n: int, bucket: str = "writing_revision") -> dict[str, Any]:
    return {
        "content_sha256": f"{n:016x}" + "0" * 48,
        "slot": {"bucket": bucket, "slot": f"{bucket}-{n:03d}"},
        "conversation": {
            "system": "",
            "turns": [{"role": "user", "text": "q"}, {"role": "assistant", "text": "a"}],
        },
        "provenance": {"label": "teacher/gpt-oss-20b-local", "route": "B"},
    }


def _review(
    d: dict[str, Any], rid: str | None = None, sources: list[str] | None = None, **scores: int
) -> dict[str, Any]:
    s = GOOD | scores
    return {
        "id": rid or cr.review_id(d),
        "scores": s,
        "decision": cr.rule(s),
        "reasons": "r",
        "checks": "c",
        "sources": ["https://example.org/ref"] if sources is None else sources,
    }


def test_rule() -> None:
    assert cr.rule(GOOD) == "accept"
    assert cr.rule(GOOD | {"privacy": 0}) == "reject"  # any zero rejects
    assert cr.rule(GOOD | {"correctness": 1}) == "reject"  # every assistant turn is trained
    assert cr.rule(GOOD | {"naturalness": 1, "relevance": 1, "target": 1, "format": 1}) == "reject"  # 12 < 13
    assert cr.rule(GOOD | {"naturalness": 1, "relevance": 1, "target": 1}) == "accept"  # 13


def test_blind_batches_hide_provenance() -> None:
    row = cr.blind(_draft(1), "instruction")
    assert set(row) == {"id", "bucket", "bucket_instruction", "system", "turns"}
    assert cr.blind(_draft(1), "i", cr.salted("x"))["id"] == cr.salted("x") != "x"


def test_ingest_validates_and_defaults_to_reject() -> None:
    ds = [_draft(i) for i in range(5)] + [_draft(9, "everyday_qa")]
    bad_decision = _review(ds[1]) | {"decision": "reject"}
    reviews = [
        _review(ds[0]),
        bad_decision,
        _review(ds[2], correctness=0),
        _review(ds[3]),
        _review(ds[3]),
        _review(ds[5], sources=[" "]),  # factual bucket accept needs a non-blank source
    ]
    res = cr.ingest(cr.draft_items(ds), reviews, "first")
    d = res["decisions"]
    assert d[cr.review_id(ds[0])]["decision"] == "accept"
    assert "does not follow" in d[cr.review_id(ds[1])]["invalid"]
    assert d[cr.review_id(ds[2])]["decision"] == "reject"
    assert "2 reviews" in d[cr.review_id(ds[3])]["invalid"]
    assert "0 reviews" in d[cr.review_id(ds[4])]["invalid"]
    assert "needs at least one source" in d[cr.review_id(ds[5])]["invalid"]
    assert (res["accepted"], res["rejected"]) == (1, 5)
    with pytest.raises(SystemExit, match="unknown"):
        cr.ingest(cr.draft_items(ds[:1]), [_review(ds[1])], "first")
    assert cr.validate({"scores": GOOD | {"format": 3}, "decision": "accept"})
    assert cr.validate({"scores": GOOD | {"format": True}, "decision": "accept"})


def _first(ds: list[dict[str, Any]]) -> dict[str, Any]:
    res = cr.ingest(cr.draft_items(ds), [_review(d) for d in ds], "first")
    res["ingest_sha256"] = sha256_json(res)
    return res


def test_sample_is_stratified_salted_and_bound() -> None:
    ds = [_draft(i, "writing_revision") for i in range(8)] + [
        _draft(100 + i, "corrections") for i in range(3)
    ]
    first = _first(ds)
    ids = cr.sample(ds, first)
    assert len(ids) == 2 + 1 and ids == sorted(ids) and ids == cr.sample(ds, first)
    buckets = dict(cr.draft_items(ds))
    picked = {cr.review_id(d): d for d in ds if cr.review_id(d) in ids}
    ok = cr.second(ids, buckets, [_review(d, cr.salted(i)) for i, d in picked.items()], "second", "sha")
    assert ok["passed"] and ok["rejection_rate"] == 0 and ok["first_ingest_sha256"] == "sha"
    assert set(ok["decisions"]) == set(ids) and ok["sample_ids"] == ids
    reviews = [
        _review(d, cr.salted(i), correctness=0 if n == 0 else 2) for n, (i, d) in enumerate(picked.items())
    ]
    bad = cr.second(ids, buckets, reviews, "second", "sha")
    assert not bad["passed"] and bad["rejection_rate"] == pytest.approx(1 / 3, abs=1e-4)
    with pytest.raises(SystemExit, match="unknown"):  # unsalted ids are not accepted
        cr.second(ids, buckets, [_review(d) for d in picked.values()], "second", "sha")
    with pytest.raises(SystemExit, match="empty"):
        cr.second([], buckets, [], "second", "sha")


def test_self_hash() -> None:
    first = _first([_draft(1)])
    assert cr.self_hash(first) == first["ingest_sha256"]
    with pytest.raises(SystemExit):
        cr.self_hash(first | {"accepted": 99})


def test_read_rows_keeps_unicode_line_separators(tmp_path: Any) -> None:
    p = tmp_path / "x.jsonl"
    p.write_text(b'{"t": "a\\u2028b\\u0085c"}\n{"t": "d"}\n'.decode("unicode_escape"))
    assert [r["t"] for r in cr.read_rows(p)] == ["a b\u0085c", "d"]


def test_merge_rereview_can_only_remove() -> None:
    ds = [_draft(i) for i in range(4)]
    reviews = [_review(ds[0]), _review(ds[1]), _review(ds[2], correctness=0), _review(ds[3])]
    first = cr.ingest(cr.draft_items(ds), reviews, "first")
    first["ingest_sha256"] = sha256_json(first)
    acc = [d for d in ds if first["decisions"][cr.review_id(d)]["decision"] == "accept"]
    re = cr.ingest(cr.draft_items(acc), [_review(acc[0]), _review(acc[1], format=0), _review(acc[2])], "re")
    re["ingest_sha256"] = sha256_json(re)
    merged = cr.merge_rereview(first, re, "merged")
    assert (merged["accepted"], merged["rejected"]) == (2, 2)
    assert merged["decisions"][cr.review_id(ds[2])]["decision"] == "reject"  # original reject stands
    with pytest.raises(SystemExit, match="exactly"):
        cr.merge_rereview(
            first, cr.ingest(cr.draft_items(ds), [_review(d) for d in ds], "re") | {"ingest_sha256": "x"}, "m"
        )
