import hashlib
import json
from datetime import UTC, datetime, timedelta

import pytest
import torch

from j1bench import ckpt as C

META = {"run_id": "r", "run_contract_sha256": "abc"}


def _save(root, stage, step=1):
    payload = {"model.pt": {"w": torch.arange(10.0)}, "optim.pt": {"s": torch.ones(3)}, "trainer.pt": {"step": step}}
    return C.save_checkpoint(root, stage, step=step, payload=payload, meta=META, torch_save=torch.save)


def test_roundtrip_and_selection(tmp_path):
    r1 = _save(tmp_path / "p", tmp_path / "s", 1)
    r2 = _save(tmp_path / "p", tmp_path / "s", 2)
    assert C.verify_checkpoint(r1.path)[0] and C.verify_checkpoint(r2.path)[0]
    p, _ = C.select_latest_valid([tmp_path / "p" / "r"], contract_sha="abc")
    assert p == r2.path


@pytest.mark.parametrize("damage", ["flip", "delete", "truncate_manifest", "no_manifest"])
def test_damaged_latest_is_rejected_and_previous_selected(tmp_path, damage):
    r1 = _save(tmp_path / "p", tmp_path / "s", 1)
    r2 = _save(tmp_path / "p", tmp_path / "s", 2)
    if damage == "flip":
        f = r2.path / "model.pt"
        b = bytearray(f.read_bytes())
        b[len(b) // 2] ^= 1
        f.write_bytes(bytes(b))
    elif damage == "delete":
        (r2.path / "optim.pt").unlink()
    elif damage == "truncate_manifest":
        m = r2.path / C.MANIFEST
        m.write_text(m.read_text()[:50])
    else:
        (r2.path / C.MANIFEST).unlink()
    ok, why = C.verify_checkpoint(r2.path)
    assert not ok, why
    p, report = C.select_latest_valid([tmp_path / "p" / "r"])
    assert p == r1.path and any(not r["valid"] and r["path"] == str(r2.path) for r in report)


def test_tampered_manifest_body_fails_self_hash(tmp_path):
    r = _save(tmp_path / "p", tmp_path / "s", 1)
    m = json.loads((r.path / C.MANIFEST).read_text())
    m["step"] = 99
    (r.path / C.MANIFEST).write_text(json.dumps(m))
    assert C.verify_checkpoint(r.path) == (False, "manifest self-hash mismatch")


@pytest.mark.parametrize("damage", ["list", "missing_payload", "bad_record", "bad_step", "bad_created"])
def test_rehashed_malformed_manifest_is_rejected_without_breaking_selection(tmp_path, damage):
    r1 = _save(tmp_path / "p", tmp_path / "s", 1)
    r2 = _save(tmp_path / "p", tmp_path / "s", 2)
    m = C.read_manifest(r2.path)
    assert m is not None
    if damage == "list":
        (r2.path / C.MANIFEST).write_text("[]")
    else:
        if damage == "missing_payload":
            del m["files"]["optim.pt"]
        elif damage == "bad_record":
            m["files"]["optim.pt"]["size"] = "invalid"
        elif damage == "bad_step":
            m["step"] = "invalid"
        else:
            m["created_utc"] = []
        body = {k: v for k, v in m.items() if k != "manifest_sha256"}
        m["manifest_sha256"] = hashlib.sha256(C.canonical(body)).hexdigest()
        (r2.path / C.MANIFEST).write_text(json.dumps(m))
        (r2.path / C.COMPLETE).write_text(m["manifest_sha256"])
    ok, _ = C.verify_checkpoint(r2.path)
    assert not ok
    latest, report = C.select_latest_valid([tmp_path / "p" / "r"])
    assert latest == r1.path
    assert any(not item["valid"] and item["path"] == str(r2.path) for item in report)


def test_incompatible_contract_skipped(tmp_path):
    _save(tmp_path / "p", tmp_path / "s", 1)
    p, rep = C.select_latest_valid([tmp_path / "p" / "r"], contract_sha="other")
    assert p is None and "incompatible" in rep[0]["reason"]


def test_injected_enospc_leaves_no_valid_partial(tmp_path, monkeypatch):
    r1 = _save(tmp_path / "p", tmp_path / "s", 1)
    monkeypatch.setenv("J1_FAULT", "enospc_after_bytes:10")
    with pytest.raises(OSError):
        _save(tmp_path / "p", tmp_path / "s", 2)
    monkeypatch.delenv("J1_FAULT")
    dirs = C.scan(tmp_path / "p" / "r")
    assert len(dirs) == 2
    p, _ = C.select_latest_valid([tmp_path / "p" / "r"])
    assert p == r1.path


def test_never_overwrites_same_step(tmp_path):
    a = _save(tmp_path / "p", tmp_path / "s", 5)
    b = _save(tmp_path / "p", tmp_path / "s", 5)
    assert a.path != b.path and C.verify_checkpoint(a.path)[0]


def test_retention_plan():
    now = datetime(2026, 10, 10, 12, tzinfo=UTC)
    ents = []
    for i in range(160):  # one checkpoint per 15 active minutes = 40 active hours
        ts = now - timedelta(minutes=15 * (159 - i))
        ents.append(
            {
                "name": f"c{i:03d}",
                "step": 500 * i,
                "created_utc": ts.isoformat(),
                "active_s": 900.0 * i,
                "tags": [],
                "valid": True,
            }
        )
    ents[0]["tags"] = ["init"]
    ents[20]["tags"] = ["stage_boundary"]
    ents[30]["tags"] = ["best"]
    ents.append(
        {
            "name": "broken",
            "step": 10**9,
            "created_utc": now.isoformat(),
            "active_s": 1e9,
            "tags": ["best"],
            "valid": False,
        }
    )
    plan = C.retention_plan(ents)
    assert {"c159", "c158", "c157"} <= set(plan) and "latest_valid" in plan["c159"]
    assert "init" in plan["c000"] and "stage_boundary" in plan["c020"] and "best" in plan["c030"]
    assert plan["broken"] == ["quarantined_invalid"]
    hourly = sorted(n for n, r in plan.items() if "hourly" in r)
    # newest active_s = 143100 s; window covers active hours 15..39 -> 25 buckets
    assert len(hourly) == 25 and "c063" in hourly and "c062" not in plan
    days = {n for n, r in plan.items() if "daily" in r}
    assert len(days) == 3  # 39.75 h back from 12:00 UTC touches three UTC dates
    ents[-1]["eval_loss"] = 0.1
    ents[30]["eval_loss"] = 1.0
    assert C.best_checkpoint(ents) == "c030"


def test_missing_completion_marker_is_unpublished(tmp_path):
    r = _save(tmp_path / "p", tmp_path / "s", 1)
    (r.path / C.COMPLETE).unlink()
    assert C.verify_checkpoint(r.path) == (False, "completion marker missing (unpublished checkpoint)")


def _flip_byte(path):
    b = bytearray(path.read_bytes())
    b[len(b) // 2] ^= 1
    path.write_bytes(bytes(b))  # same size, different content


def _save_tagged(run_root, stage, step, tags):
    payload = {"model.pt": {"w": torch.arange(10.0) + step}, "optim.pt": {"s": torch.ones(3)}}
    payload["trainer.pt"] = {"step": step}
    meta = dict(META, tags=list(tags))
    return C.save_checkpoint(run_root, stage, step=step, payload=payload, meta=meta, torch_save=torch.save).path


def test_prune_pilot_keeps_init_latest_best_final_and_invalid(tmp_path):
    """Review T-12: the init checkpoint is retained; hash-invalid directories are quarantined, never latest."""
    run = tmp_path / "p" / "r"
    made = {}
    for step, tags in ((0, ["init"]), (5, ["eval", "best"]), (10, []), (15, ["eval", "best"]), (20, []), (25, [])):
        made[step] = _save_tagged(tmp_path / "p", tmp_path / "s", step, tags)
    _flip_byte(made[25] / "model.pt")  # sizes intact, bytes corrupt: not valid, not latest, kept as evidence
    assert C.verify_checkpoint(made[25], check_hashes=False)[0] and not C.verify_checkpoint(made[25])[0]
    deleted = C.prune_pilot(run)
    assert sorted(deleted) == [made[5].name, made[10].name]
    assert sorted(p.name for p in run.iterdir() if p.is_dir()) == sorted(made[s].name for s in (0, 15, 20, 25))


def test_prune_pilot_corrupt_newest_deletes_nothing(tmp_path):
    """Review T-12 reproducer: flip one byte in the newest payload (size unchanged) -> prune deletes nothing."""
    run = tmp_path / "p" / "r"
    init = _save_tagged(tmp_path / "p", tmp_path / "s", 0, ["init"])
    mid = _save_tagged(tmp_path / "p", tmp_path / "s", 5, [])
    newest = _save_tagged(tmp_path / "p", tmp_path / "s", 10, [])
    _flip_byte(newest / "optim.pt")
    assert C.prune_pilot(run) == []
    assert all(d.is_dir() for d in (init, mid, newest))
    # A second verified checkpoint makes the older non-init one prunable; init and the corrupt one stay.
    latest = _save_tagged(tmp_path / "p", tmp_path / "s", 15, [])
    assert C.prune_pilot(run) == [mid.name]
    assert all(d.is_dir() for d in (init, newest, latest))


def test_prune_pilot_refuses_with_fewer_than_two_verified(tmp_path):
    run = tmp_path / "p" / "r"
    only = _save_tagged(tmp_path / "p", tmp_path / "s", 7, [])
    assert C.prune_pilot(run) == [] and only.is_dir()
    broken = _save_tagged(tmp_path / "p", tmp_path / "s", 9, [])
    (broken / C.COMPLETE).unlink()
    assert C.prune_pilot(run) == [] and only.is_dir() and broken.is_dir()
    assert C.prune_pilot(tmp_path / "nowhere") == []
