import json
from pathlib import Path

import pytest

from j1data import gates
from j1data.dedup import UnitRef
from j1data.finalize import validate_marker_inventory
from j1data.gates import (
    _accounting_passes,
    _current_test_record,
    _marker_inventory_matches,
    _test_record_passes,
)
from j1data.paths import DataLayout


@pytest.mark.parametrize(
    "record",
    [
        {"passed": 146, "failed": 0, "returncode": 23, "end_to_end_passed": True, "end_to_end_returncode": 0},
        {"passed": 146, "failed": 0, "returncode": 0, "end_to_end_passed": True, "end_to_end_returncode": 1},
        {"passed": 146, "failed": 0, "returncode": 0, "end_to_end_passed": False, "end_to_end_returncode": 0},
    ],
)
def test_test_gate_rejects_nonzero_or_failed_suite_evidence(record: dict[str, object]) -> None:
    assert not _test_record_passes(record)


def test_test_gate_requires_both_suites_to_pass() -> None:
    assert _test_record_passes(
        {"passed": 146, "failed": 0, "returncode": 0, "end_to_end_passed": True, "end_to_end_returncode": 0}
    )


def test_test_gate_ignores_evidence_from_a_different_commit_or_input_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    layout = DataLayout(tmp_path)
    build = layout.work("build") / "build.json"
    build.parent.mkdir(parents=True)
    build.write_text('{"identity":"build-identity"}')
    runs = layout.reports("build") / "test-runs"
    runs.mkdir(parents=True)
    stale = {
        "tested_commit": "old-head",
        "test_inputs_sha256": "current-inputs",
        "build_identity": "build-identity",
    }
    current = {
        "tested_commit": "current-head",
        "test_inputs_sha256": "current-inputs",
        "build_identity": "build-identity",
    }
    (runs / "1-stale.json").write_text(json.dumps(stale))
    (runs / "2-current.json").write_text(json.dumps(current))
    monkeypatch.setattr(gates, "_git_head", lambda: "current-head")
    monkeypatch.setattr(gates, "_test_inputs_sha256", lambda: "current-inputs")
    assert _current_test_record(layout, "build") == current


def test_training_gate_exposes_owner_decisions_as_pending_or_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = tmp_path / "configs"
    config.mkdir()
    decision_file = config / "owner_decisions.toml"
    decision_file.write_text(
        '[[decision]]\ndecision_id="fineweb-edu-training-use"\nstatus="PENDING"\n'
        'sources=["fineweb-edu-dedup"]\n'
    )
    monkeypatch.setattr(gates, "REPO_ROOT", tmp_path)
    status, evidence = gates._owner_decision_status({"fineweb-edu-dedup"})
    assert status == gates.NOT_VERIFIED and "fineweb-edu-training-use" in evidence
    decision_file.write_text(
        '[[decision]]\ndecision_id="fineweb-edu-training-use"\nstatus="REJECTED"\n'
        'sources=["fineweb-edu-dedup"]\n'
    )
    status, evidence = gates._owner_decision_status({"fineweb-edu-dedup"})
    assert status == gates.FAIL and "fineweb-edu-training-use" in evidence


def test_training_gate_requires_each_applicable_owner_decision_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = tmp_path / "configs"
    config.mkdir()
    (config / "owner_decisions.toml").write_text("")
    monkeypatch.setattr(gates, "REPO_ROOT", tmp_path)
    status, evidence = gates._owner_decision_status({"fineweb-edu-dedup"})
    assert status == gates.NOT_VERIFIED
    assert "missing required decisions" in evidence


def test_training_gate_rejects_duplicate_required_decision_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = tmp_path / "configs"
    config.mkdir()
    record = (
        '[[decision]]\ndecision_id="fineweb-edu-training-use"\nstatus="APPROVED"\n'
        'sources=["fineweb-edu-dedup"]\n'
    )
    (config / "owner_decisions.toml").write_text(record + record)
    monkeypatch.setattr(gates, "REPO_ROOT", tmp_path)
    status, evidence = gates._owner_decision_status({"fineweb-edu-dedup"})
    assert status == gates.NOT_VERIFIED
    assert "duplicate required decisions" in evidence


def test_finalize_rejects_stale_or_mismatched_marker_inventory() -> None:
    current_ref = UnitRef("source", "unit-current", 0, 0, Path("candidate.parquet"), Path("sig.parquet"))
    current = {"source_id": "source", "unit_id": "unit-current", "build_identity": "build-a"}
    stale = {"source_id": "source", "unit_id": "unit-old", "build_identity": "build-a"}
    with pytest.raises(RuntimeError, match="stale="):
        validate_marker_inventory([current_ref], [current, stale], "build-a")
    with pytest.raises(RuntimeError, match="identity_mismatch="):
        validate_marker_inventory([current_ref], [{**current, "build_identity": "build-old"}], "build-a")


def test_finalize_accepts_exact_current_marker_inventory() -> None:
    ref = UnitRef("source", "unit", 0, 0, Path("candidate.parquet"), Path("sig.parquet"))
    marker = {"source_id": "source", "unit_id": "unit", "build_identity": "build-a"}
    validate_marker_inventory([ref], [marker], "build-a")


def test_freeze_gates_reject_missing_accounting_and_incomplete_inventories() -> None:
    required = {
        "raw_equals_stage_total": True,
        "candidates_equal_candidate_rows": True,
        "dedup_saw_all_candidates": True,
        "candidates_equal_removed_plus_unselected_plus_leaked_plus_accepted": True,
        "finalize_dedup_agree": True,
    }
    assert _accounting_passes(required)
    assert not _accounting_passes({})
    assert not _accounting_passes({**required, "finalize_dedup_agree": False})

    expected = {("source-a", "unit-a"), ("source-a", "unit-b")}
    current = [
        {"source_id": "source-a", "unit_id": "unit-a", "build_identity": "build-a"},
        {"source_id": "source-a", "unit_id": "unit-b", "build_identity": "build-a"},
    ]
    assert _marker_inventory_matches(expected, expected, current, [], "build-a")
    assert not _marker_inventory_matches(expected, {("source-a", "unit-a")}, current[:1], [], "build-a")
    assert not _marker_inventory_matches(
        expected, expected, current, [], "build-a", exclusion_identity="new-index"
    )


def test_expected_finalize_units_follow_process_segments(tmp_path) -> None:  # type: ignore[no-untyped-def]
    import json as _json

    from j1data.gates import _expected_finalize_units
    from j1data.paths import DataLayout

    layout = DataLayout(tmp_path)
    units = layout.work("b") / "units"
    (units / "s").mkdir(parents=True)
    (units / "s" / "u1.json").write_text(
        _json.dumps({"source_id": "s", "unit_id": "u1", "segments": [{"index": 0}, {"index": 1}]})
    )
    (units / "s" / "u2.json").write_text(_json.dumps({"source_id": "s", "unit_id": "u2", "outputs": []}))
    expected = {("s", "u1"), ("s", "u2")}
    assert _expected_finalize_units(layout, "b", expected) == {
        ("s", "u1~part-000000"),
        ("s", "u1~part-000001"),
        ("s", "u2"),
    }
    assert _expected_finalize_units(layout, "b", {*expected, ("s", "u3")}) is None
