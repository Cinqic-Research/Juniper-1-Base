"""End-to-end exact-resume on CPU with the tiny preset."""

import hashlib
import json
import pickle
from pathlib import Path

import pytest
import torch

from j1bench import ckpt as C
from j1bench.train import Trainer, build_parser


class UnexpectedGlobal:
    pass


def run(tmp, *extra):
    a = build_parser().parse_args(
        [
            "--preset",
            "tiny",
            "--device",
            "cpu",
            "--seq-len",
            "32",
            "--micro-bs",
            "2",
            "--accum",
            "2",
            "--steps",
            "12",
            "--warmup",
            "3",
            "--run-id",
            "t",
            "--stage-dir",
            str(tmp / "stage"),
            *extra,
        ]
    )
    return Trainer(a).run()


def losses(path):
    return [r["loss_hex"] for r in map(json.loads, Path(path).read_text().splitlines()) if "loss_hex" in r]


def test_interrupted_and_resumed_run_is_bitwise_identical(tmp_path):
    full = run(tmp_path, "--metrics", str(tmp_path / "full.jsonl"))
    root = tmp_path / "ck"
    run(
        tmp_path,
        "--ckpt-root",
        str(root),
        "--ckpt-every-steps",
        "4",
        "--stop-at",
        "6",
        "--metrics",
        str(tmp_path / "a.jsonl"),
    )
    res = run(
        tmp_path,
        "--ckpt-root",
        str(root),
        "--ckpt-every-steps",
        "4",
        "--resume",
        "auto",
        "--metrics",
        str(tmp_path / "b.jsonl"),
    )
    assert res["final_param_sha256"] == full["final_param_sha256"]
    assert res["final_optim_sha256"] == full["final_optim_sha256"]
    assert losses(tmp_path / "a.jsonl") + losses(tmp_path / "b.jsonl") == losses(tmp_path / "full.jsonl")
    steps = [C.read_manifest(p)["step"] for p in C.scan(root / "t")]
    # cadence restarts from the restored checkpoint: 0 init, 4, 6 (stop), 10, 12 final
    assert sorted(steps) == [0, 4, 6, 10, 12]


def test_resume_refuses_changed_contract_unless_corrective(tmp_path):
    root = tmp_path / "ck"
    run(tmp_path, "--ckpt-root", str(root), "--stop-at", "2")
    import pytest

    with pytest.raises(C.CheckpointError):
        run(tmp_path, "--ckpt-root", str(root), "--resume", str(C.scan(root / "t")[0]), "--lr", "1e-3")
    with pytest.raises(C.CheckpointError):  # auto-resume must not silently start fresh over incompatible history
        run(tmp_path, "--ckpt-root", str(root), "--resume", "auto", "--lr", "1e-3")
    with pytest.raises(C.CheckpointError):  # corrective continuation needs a new run id
        run(
            tmp_path,
            "--ckpt-root",
            str(root),
            "--resume",
            str(C.scan(root / "t")[0]),
            "--lr",
            "1e-3",
            "--corrective-continuation",
            "lr change test",
        )
    r = run(
        tmp_path,
        "--ckpt-root",
        str(root),
        "--resume",
        str(C.scan(root / "t")[0]),
        "--lr",
        "1e-3",
        "--corrective-continuation",
        "lr change test",
        "--run-id",
        "t2",
        "--metrics",
        str(tmp_path / "c.jsonl"),
    )
    assert r["final_step"] == 12
    ev = [json.loads(x) for x in Path(tmp_path / "c.jsonl").read_text().splitlines()]
    assert any(
        e.get("event") == "restored" and e["corrective"]["corrective_continuation"] == "lr change test" for e in ev
    )


def test_production_cadence_defaults_and_time_trigger(tmp_path):
    d = build_parser().parse_args([])
    assert (d.ckpt_every_steps, d.ckpt_every_seconds) == (500, 900.0)
    root = tmp_path / "ck"
    # Step trigger far away (500); a tiny time budget forces time-triggered saves,
    # which must land on optimizer boundaries (integer steps) only.
    run(tmp_path, "--ckpt-root", str(root), "--ckpt-every-seconds", "0.000001", "--metrics", str(tmp_path / "m.jsonl"))
    saves = [json.loads(x) for x in Path(tmp_path / "m.jsonl").read_text().splitlines() if '"checkpoint"' in x]
    assert [s["step"] for s in saves] == list(range(0, 13))
    for p in C.scan(root / "t"):
        assert C.read_manifest(p)["save_reason"] in ("init", "cadence", "final")


def test_restore_rejects_untrusted_global_even_with_matching_manifest_hashes(tmp_path):
    root = tmp_path / "ck"
    run(tmp_path, "--ckpt-root", str(root), "--stop-at", "2")
    p = C.scan(root / "t")[0]
    trainer_file = p / "trainer.pt"
    torch.save({"unexpected": UnexpectedGlobal()}, trainer_file)
    manifest = C.read_manifest(p)
    assert manifest is not None
    manifest["files"]["trainer.pt"] = {
        "size": trainer_file.stat().st_size,
        "sha256": C.sha256_file(trainer_file),
    }
    body = {k: v for k, v in manifest.items() if k != "manifest_sha256"}
    manifest["manifest_sha256"] = hashlib.sha256(C.canonical(body)).hexdigest()
    (p / C.MANIFEST).write_text(json.dumps(manifest))
    (p / C.COMPLETE).write_text(manifest["manifest_sha256"])
    assert C.verify_checkpoint(p) == (True, "ok")

    args = build_parser().parse_args(
        [
            "--preset",
            "tiny",
            "--device",
            "cpu",
            "--seq-len",
            "32",
            "--micro-bs",
            "2",
            "--accum",
            "2",
            "--steps",
            "12",
            "--warmup",
            "3",
            "--run-id",
            "t",
        ]
    )
    with pytest.raises(pickle.UnpicklingError):
        Trainer(args).restore(p)


def test_corrective_continuation_with_lower_lr_rebases_schedule_and_records_carryover(tmp_path):
    """Regression (A100 recovery suite T9, 2026-10-10): the restored scheduler kept the parent's peak LR, so a
    lower new LR tripped the [0, peak] guard. The schedule is now re-based and the carryover recorded."""
    root = tmp_path / "ck"
    run(tmp_path, "--ckpt-root", str(root), "--stop-at", "4", "--lr", "3e-3")
    r = run(
        tmp_path, "--ckpt-root", str(root), "--resume", str(C.scan(root / "t")[-1]), "--lr", "1e-4",
        "--corrective-continuation", "lower lr", "--run-id", "t3", "--metrics", str(tmp_path / "d.jsonl"),
    )  # fmt: skip
    assert r["final_step"] == 12
    ev = [json.loads(x) for x in Path(tmp_path / "d.jsonl").read_text().splitlines()]
    restored = next(e for e in ev if e.get("event") == "restored")
    co = restored["corrective"]["scheduler_carryover"]
    assert co["base_lr_from"][0] == 3e-3 and co["base_lr_to"][0] == 1e-4 and co["position"] == restored["step"]
    assert restored["lr"] <= 1e-4 and "optimizer_carryover" in restored["corrective"]
    assert all(e["lr"] <= 1e-4 * (1 + 1e-6) for e in ev if "lr" in e and "step" in e and not e.get("event"))
