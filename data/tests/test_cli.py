from types import SimpleNamespace

import pytest

from j1data.cli import _require_exploratory_build, _validate_process_sources


def test_process_rejects_explicit_inadmissible_source_before_dispatch() -> None:
    registry = {"blocked": SimpleNamespace(admissible=False, license_state="REJECTED")}
    with pytest.raises(SystemExit, match=r"processing refused.*REJECTED"):
        _validate_process_sources(registry, ["blocked"])


def test_process_rejects_unknown_explicit_source() -> None:
    with pytest.raises(SystemExit, match="unknown source"):
        _validate_process_sources({}, ["missing"])


def test_process_accepts_explicit_admissible_source() -> None:
    registry = {"approved": SimpleNamespace(admissible=True, license_state="APPROVED")}
    assert _validate_process_sources(registry, ["approved"]) == ["approved"]


def test_process_rejects_partial_work_in_release_build() -> None:
    with pytest.raises(SystemExit, match="partial source or unit processing"):
        _require_exploratory_build("release", "release", ["source-a"], ["source-a"], "unit-*", None)


def test_process_allows_partial_work_in_distinct_exploratory_build() -> None:
    _require_exploratory_build("scratch-run", "release", ["source-a"], ["source-a"], "unit-*", None)
