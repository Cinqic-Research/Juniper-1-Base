"""j1data.sftreplay: held-out phrasing patterns match evaluation-only text and not trained phrasings."""

from __future__ import annotations

from j1data import sftops, sftreplay


def _render(template: str) -> str:
    return sftreplay._norm(template.replace("{it}", "these words").replace("{p}", "3").replace("{q}", "x"))


def test_reserved_and_heldout_phrasings_match_and_trained_ones_do_not() -> None:
    pats = sftreplay.heldout_patterns()
    hit = lambda text: any(p.search(text) for _, p in pats)  # noqa: E731
    reserved = [
        (op, i)
        for op in sftops.TRAIN_OPS
        for i in range(len(op.phrasings))
        if sftops.reserved_phrasing(op, i)
    ]
    assert reserved
    assert all(hit(_render(op.phrasings[i])) for op, i in reserved)
    assert all(hit(_render(p)) for op in sftops.HELDOUT_OPS for p in op.phrasings)
    trained = [op.phrasings[0] for op in sftops.TRAIN_OPS]  # the first phrasing is never reserved
    assert sum(hit(_render(p)) for p in trained) <= len(trained) // 10
    assert hit('words: "a b c" instruction: sort them')
    assert hit("here are some words: a b c")


def test_unqualified_categories_are_refused(tmp_path) -> None:  # type: ignore[no-untyped-def]
    import pytest

    with pytest.raises(SystemExit, match="only qualified"):
        sftreplay.admit(tmp_path, tmp_path, tmp_path, tmp_path / "t.json", ["instruction_following"])
