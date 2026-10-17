"""SFT stage qualification on synthetic fixtures (tiny model, CPU, float64 where exactness matters).

The known-outcome task: a user message of three content tokens; the assistant must answer with
each token + 20. Episodes follow ``juniper-chat-v1`` token layout (IDs from j1tok's control block)
and the Juniper-1-Data shift (inputs = ids[:-1], targets = labels[1:]).
"""

import hashlib
import json
import os
import signal
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from j1bench import ckpt as C
from j1bench import sft as S
from j1bench import sftdata as D
from j1bench.frozen import DIGEST_KEY, FREEZE_SCHEMA, record_digest
from j1bench.model import ModelConfig, Transformer
from j1bench.train import tensor_tree_sha

USER, END, ASSISTANT, FINAL, EOT = 3, 9, 4, 7, 11
LO, HI, OFFSET = 40, 56, 20  # answer token = prompt token + 20 (disjoint ranges)
CFG = {
    "d_ff": 64, "d_model": 32, "head_dim": 16, "init_std": 0.02, "max_seq_len": 32, "n_head": 2, "n_kv_head": 1,
    "n_layer": 2, "norm_eps": 1e-5, "qk_norm": "rmsnorm_per_head", "rope_theta": 100000.0, "tie_embeddings": True,
    "vocab_size": 128,
}  # fmt: skip
ROOT = Path(__file__).resolve().parents[1]


def episode(prompt, shift=True):
    ids = [USER, *prompt, END, ASSISTANT, FINAL, *[t + OFFSET for t in prompt], EOT]
    labels = [-100] * (len(prompt) + 3) + [FINAL, *[t + OFFSET for t in prompt], EOT]
    if shift:
        return ids[:-1], labels[1:]
    return ids[:-1], labels[:-1]  # the A-01 defect: token-aligned labels (teaches copying)


def make_dataset(tmp, *, n_train=512, n_dev=64, shift=True, seed=0, name="ds"):
    rng = np.random.default_rng(seed)
    d = tmp / name
    d.mkdir()
    files = {}
    for split, n in (("train", n_train), ("dev", n_dev)):
        xs, ys, offs = [], [], [0]
        for _ in range(n):
            p = rng.integers(LO, HI, size=int(rng.integers(2, 5))).tolist()
            x, y = episode(p, shift)
            xs += x
            ys += y
            offs.append(len(xs))
        np.savez(
            d / f"{split}_rows.npz",
            inputs=np.array(xs, np.int32),
            targets=np.array(ys, np.int32),
            offsets=np.array(offs, np.int64),
            episode_ids=np.array(
                [list((i + (10**6 if split == "dev" else 0)).to_bytes(16, "big")) for i in range(n)], dtype=np.uint8
            ),
        )
        files[f"{split}_rows.npz"] = {"sha256": C.sha256_file(d / f"{split}_rows.npz")}
    tok = tmp / "tokenizer.json"
    if not tok.exists():
        tok.write_text("{}")
    m = {
        "schema": D.DATASET_SCHEMA,
        "version": f"synthetic-{name}",
        "stage": "agent-instruction-sft",
        "tokenizer_sha256": C.sha256_file(tok),
        "files": files,
        "eval_bundle_sha256": "none",
    }
    m["manifest_sha256"] = hashlib.sha256(D.canonical_json(m)).hexdigest()
    (d / "manifest.json").write_text(json.dumps(m))
    return d, m["manifest_sha256"], tok


def make_base(tmp):
    torch.manual_seed(0)
    model = Transformer(ModelConfig(**CFG))
    sd = model.state_dict()
    meta = {"run_id": "base", "run_contract_sha256": "base", "run_contract": {"model": CFG}}
    path = C.save_checkpoint(
        tmp / "base", tmp / "bst", step=1, payload={"model.pt": sd, "optim.pt": {}, "trainer.pt": {}}, meta=meta,
        torch_save=torch.save,
    ).path  # fmt: skip
    man = json.loads((path / "manifest.json").read_text())
    tok = tmp / "tokenizer.json"
    tok.write_text("{}")
    rec = {
        "schema": FREEZE_SCHEMA, "freeze_id": "juniper1-base-v1", "decision": "BASE_FREEZE_APPROVED",
        "source_checkpoint": {
            "name": path.name, "run_id": "base", "step": 1, "manifest_sha256": man["manifest_sha256"],
            "files": {"model.pt": {"sha256": man["files"]["model.pt"]["sha256"]}},
        },
        "model": {"config": CFG, "state_sha256": tensor_tree_sha(sd), "parameters": model.num_params()},
        "tokenizer": {"sha256": C.sha256_file(tok)},
    }  # fmt: skip
    rec[DIGEST_KEY] = record_digest(rec)
    rp = tmp / "freeze.json"
    rp.write_text(json.dumps(rec))
    return rp, path, tok, rec[DIGEST_KEY]


def contract(ds_sha, tok, base_digest, **kw):
    c = {
        "schema": S.CONTRACT_SCHEMA,
        "stage_id": "synthetic-sft",
        "run_id": kw.pop("run_id", "run"),
        "base": {"freeze_id": "juniper1-base-v1", "freeze_record_sha256": base_digest},
        "dataset": {"manifest_sha256": ds_sha, "tokenizer_sha256": C.sha256_file(tok)},
        "optimizer": {"name": "adamw", "lr": 3e-3, "betas": [0.9, 0.95], "eps": 1e-8, "weight_decay": 0.0, "clip": 1.0},
        "schedule": {"warmup_updates": 5, "total_updates": 20, "min_lr_ratio": 0.1},
        "batch": {"episodes_per_update": 16, "token_budget": 96},
        "precision": "fp32",
        "seed": 7,
        "checkpoint": {"every_updates": 500, "every_seconds": 900},
        "dev_every": 0,
        "deterministic": True,
    }
    for k, v in kw.items():
        if isinstance(v, dict) and k in c:
            c[k] = {**c[k], **v}
        else:
            c[k] = v
    return c


@pytest.fixture
def env(tmp_path):
    rp, base, tok, digest = make_base(tmp_path)
    ds, ds_sha, _ = make_dataset(tmp_path)
    return {"tmp": tmp_path, "record": rp, "base": base, "tok": tok, "digest": digest, "ds": ds, "ds_sha": ds_sha}


def trainer(env, c, out="out", **kw):
    return S.SFTTrainer(
        c, freeze_record=env["record"], base_checkpoint=env["base"], tokenizer=env["tok"], dataset_dir=env["ds"],
        out_root=env["tmp"] / out, staging=env["tmp"] / "stage", device="cpu", log=lambda *_: None, **kw,
    )  # fmt: skip


def greedy_accuracy(model, n=64, seed=123):
    rng = np.random.default_rng(seed)
    model.eval()
    ok = 0
    with torch.no_grad():
        for _ in range(n):
            p = rng.integers(LO, HI, size=3).tolist()
            seq = [USER, *p, END, ASSISTANT]
            for _ in range(5):
                logits = model(torch.tensor([seq]))
                seq.append(int(logits[0, -1].argmax()))
            ok += seq[-5:] == [FINAL, *[t + OFFSET for t in p], EOT]
    model.train()
    return ok / n


# --- shift and mask correctness ---------------------------------------------------------------


def test_alignment_rejects_unshifted_and_misaligned_labels(tmp_path):
    ds, sha, tok = make_dataset(tmp_path, shift=False, name="bad")
    with pytest.raises(D.DatasetError, match="not the next input token"):
        D.SFTDataset(ds, expected_manifest_sha256=sha, tokenizer_sha256=C.sha256_file(tok), vocab_size=128)
    x, y = episode([50, 60, 70])
    x, y = np.array(x), np.array(y)
    D.check_alignment(x, y, np.array([0, len(x)]))
    for bad in (np.roll(y, 1), np.roll(y, -1)):
        with pytest.raises(D.DatasetError):
            D.check_alignment(x, bad, np.array([0, len(x)]))
    unsup_last = y.copy()
    unsup_last[-1] = -100
    with pytest.raises(D.DatasetError, match="final target"):
        D.check_alignment(x, unsup_last, np.array([0, len(x)]))


def test_dataset_identity_is_pinned(env):
    with pytest.raises(D.DatasetError, match="pins"):
        D.SFTDataset(
            env["ds"], expected_manifest_sha256="0" * 64, tokenizer_sha256=C.sha256_file(env["tok"]), vocab_size=128
        )
    (env["ds"] / "dev_rows.npz").write_bytes((env["ds"] / "dev_rows.npz").read_bytes() + b"x")
    with pytest.raises(D.DatasetError, match="bytes differ"):
        D.SFTDataset(
            env["ds"],
            expected_manifest_sha256=env["ds_sha"],
            tokenizer_sha256=C.sha256_file(env["tok"]),
            vocab_size=128,
        )


def test_eval_bundle_can_never_be_training_data(tmp_path):
    d = tmp_path / "eval"
    d.mkdir()
    (d / "manifest.json").write_text(json.dumps({"schema": D.EVAL_SCHEMA, "bundle_sha256": "x"}))
    with pytest.raises(D.DatasetError, match="evaluation bundle"):
        D.SFTDataset(d, expected_manifest_sha256="x", tokenizer_sha256="x", vocab_size=128)


def test_masked_and_padded_positions_carry_no_gradient():
    torch.manual_seed(1)
    model = Transformer(ModelConfig(**CFG)).double()
    x, y = episode([50, 60, 70])
    xt, yt = torch.tensor([x]), torch.tensor([y])
    logits = model(xt).detach().requires_grad_(True)
    F.cross_entropy(logits.reshape(-1, 128), yt.reshape(-1), ignore_index=-100, reduction="sum").backward()
    masked = (yt == -100)[0]
    assert (
        masked.sum() == 5
        and torch.all(logits.grad[0, masked] == 0)
        and torch.all(logits.grad[0, ~masked].abs().sum(-1) > 0)
    )
    # right padding changes neither the loss nor any parameter gradient
    grads = []
    for pad in (0, 7):
        model.zero_grad()
        xp = torch.tensor([x + [0] * pad])
        yp = torch.tensor([y + [-100] * pad])
        _, ls = model(xp, yp)
        ls.backward()
        grads.append((ls.item(), [p.grad.clone() for p in model.parameters()]))
    assert grads[0][0] == pytest.approx(grads[1][0], abs=1e-12)
    for a, b in zip(grads[0][1], grads[1][1], strict=True):
        assert torch.allclose(a, b, atol=1e-12, rtol=0)


def test_micro_batches_hold_one_episode_per_row(env):
    ds = D.SFTDataset(
        env["ds"], expected_manifest_sha256=env["ds_sha"], tokenizer_sha256=C.sha256_file(env["tok"]), vocab_size=128
    )
    tr = ds.splits["train"]
    plan = D.update_plan(tr.n, 7, 0, 16)
    seen = []
    for x, y, idx, cnt in D.micro_batches(tr, plan[0], 40):
        assert x.numel() <= 40 and cnt == int((y != -100).sum())
        for r, e in enumerate(idx):
            xi, yi = tr.episode(e)
            assert x[r, : len(xi)].tolist() == xi.tolist() and torch.all(x[r, len(xi) :] == 0)
            assert y[r, : len(yi)].tolist() == yi.tolist() and torch.all(y[r, len(yi) :] == -100)
            seen.append(e)
    assert sorted(seen) == sorted(plan[0].tolist())


def test_update_is_invariant_to_micro_batch_split(env):
    states = []
    for budget in (32, 512):
        c = contract(env["ds_sha"], env["tok"], env["digest"], batch={"token_budget": budget}, run_id=f"b{budget}")
        t = trainer(env, c, out=f"o{budget}")
        t.model.double()
        t.opt = torch.optim.AdamW(t.model.param_groups(0.0), lr=1e-3, foreach=False)
        rec = t.train_update()
        states.append((rec["micro_batches"], [p.detach().clone() for p in t.model.parameters()]))
    assert states[0][0] > states[1][0] == 1
    for a, b in zip(states[0][1], states[1][1], strict=True):
        assert torch.allclose(a, b, atol=1e-12, rtol=0)


# --- learning on a known task, and sensitivity to a broken shift ------------------------------


def test_learns_known_task_and_dev_loss_falls(env):
    c = contract(
        env["ds_sha"], env["tok"], env["digest"], schedule={"warmup_updates": 20, "total_updates": 800}, dev_every=400
    )
    t = trainer(env, c)
    before = greedy_accuracy(t.model)
    assert t.run() == 0
    dev = [r["dev_loss"] for r in t.dev_history]
    assert before < 0.05 and greedy_accuracy(t.model) >= 0.95
    assert dev[-1] < 0.05 and dev[0] > dev[-1]


def test_broken_shift_is_caught_by_loader_and_by_the_learning_check(tmp_path, monkeypatch):
    rp, base, tok, digest = make_base(tmp_path)
    ds, sha, _ = make_dataset(tmp_path, shift=False, name="unshifted")
    env = {"tmp": tmp_path, "record": rp, "base": base, "tok": tok, "digest": digest, "ds": ds, "ds_sha": sha}
    c = contract(sha, tok, digest, schedule={"warmup_updates": 20, "total_updates": 800})
    with pytest.raises(D.DatasetError):
        trainer(env, c)
    monkeypatch.setattr(D, "check_alignment", lambda *_: None)  # deliberately disable the guard
    t = trainer(env, c)
    assert t.run() == 0
    final_loss = json.loads((tmp_path / "out/run/sft_log.jsonl").read_text().splitlines()[-3])
    assert greedy_accuracy(t.model) < 0.5  # copying was learned instead of the task
    assert final_loss is not None


# --- checkpoint cadence, recovery and fault injection -----------------------------------------


def _ckpts(run_dir):
    return sorted((C.read_manifest(p)["updates"], p) for p in C.scan(run_dir) if C.verify_checkpoint(p)[0])


def test_cadence_by_updates_and_final(env):
    c = contract(env["ds_sha"], env["tok"], env["digest"], checkpoint={"every_updates": 5})
    t = trainer(env, c)
    assert t.run() == 0
    assert [u for u, _ in _ckpts(t.run_dir)] == [5, 10, 15, 20]


def test_cadence_by_time_with_slow_updates(env):
    now = [0.0]

    def clock():
        now[0] += 200.0  # every call advances the clock: updates look slow
        return now[0]

    c = contract(env["ds_sha"], env["tok"], env["digest"], schedule={"total_updates": 8})
    t = trainer(env, c, clock=clock)
    assert t.run() == 0
    events = [json.loads(line) for line in (t.run_dir / "sft_log.jsonl").read_text().splitlines()]
    saves = [e for e in events if e.get("event") == "checkpoint"]
    assert saves and all(e["seconds_since_previous"] <= 900 + 400 for e in saves)
    assert {e["reason"] for e in saves} >= {"cadence_time"}


def test_contract_cadence_limits_enforced(env):
    for bad in ({"every_updates": 501}, {"every_seconds": 901}):
        with pytest.raises(S.ContractError, match="cadence"):
            S.validate_contract(contract(env["ds_sha"], env["tok"], env["digest"], checkpoint=bad))
    with pytest.raises(S.ContractError, match="single-device"):
        S.validate_contract(contract(env["ds_sha"], env["tok"], env["digest"], world_size=2))


def _final_state(t):
    return tensor_tree_sha(t.model.state_dict()), tensor_tree_sha(t.opt.state_dict())


def _fps(run_dir):
    out = {}
    for line in (run_dir / "sft_log.jsonl").read_text().splitlines():
        e = json.loads(line)
        if "batch_fp" in e:
            out[e["update"]] = (e["batch_fp"], e["loss"])
    return out


def test_clean_resume_is_bitwise_identical(env):
    c = contract(env["ds_sha"], env["tok"], env["digest"], checkpoint={"every_updates": 7})
    ref = trainer(env, c, out="ref")
    assert ref.run() == 0
    a = trainer(env, c, out="split")
    assert a.run(stop_after=10) == S.EXIT_STOPPED  # last checkpoint at 7; updates 8-10 are lost
    b = trainer(env, c, out="split")
    assert b.run() == 0 and b.exposure == ref.exposure
    assert _final_state(b) == _final_state(ref)
    assert {k: v for k, v in _fps(b.run_dir).items() if k > 7} == {k: v for k, v in _fps(ref.run_dir).items() if k > 7}


SCRIPT = textwrap.dedent(
    """
    import json, sys
    sys.path.insert(0, {root!r})
    from pathlib import Path
    from j1bench import sft as S
    env = json.loads(Path({envf!r}).read_text())
    c = json.loads(Path({cf!r}).read_text())
    t = S.SFTTrainer(c, freeze_record=Path(env["record"]), base_checkpoint=Path(env["base"]),
                     tokenizer=Path(env["tok"]), dataset_dir=Path(env["ds"]), out_root=Path(env["tmp"]) / "crash",
                     staging=Path(env["tmp"]) / "st2", device="cpu", log=lambda *_: None)
    sys.exit(t.run())
    """
)


def _run_child(env, c, fault_var, fault):
    envf, cf = env["tmp"] / "env.json", env["tmp"] / "c.json"
    envf.write_text(json.dumps({k: str(v) for k, v in env.items()}))
    cf.write_text(json.dumps(c))
    script = SCRIPT.format(root=str(ROOT), envf=str(envf), cf=str(cf))
    e = {**os.environ, fault_var: fault, "OMP_NUM_THREADS": "1"}
    return subprocess.run([sys.executable, "-c", script], env=e, capture_output=True, text=True, timeout=300)


@pytest.mark.parametrize(
    ("var", "fault"),
    [
        ("J1_SFT_FAULT", "kill_mid_update:9"),  # process dies with a partially accumulated gradient
        ("J1_FAULT", "die_after_files:1"),  # checkpoint torn: model.pt copied, optim.pt missing
        ("J1_FAULT", "die_before_complete"),  # manifest written, publication marker missing
    ],
)
def test_crash_then_resume_matches_uninterrupted(env, var, fault):
    c = contract(env["ds_sha"], env["tok"], env["digest"], checkpoint={"every_updates": 6})
    ref = trainer(env, c, out="ref")
    assert ref.run() == 0
    r = _run_child(env, c, var, fault)
    assert r.returncode == 137, r.stderr[-2000:]
    run_dir = env["tmp"] / "crash" / "run"
    valid = _ckpts(run_dir)
    if var == "J1_SFT_FAULT":
        assert [u for u, _ in valid] == [6]  # the update-9 crash lost 7-8 only
    else:
        assert valid == []  # the torn first checkpoint is never valid
        assert C.scan(run_dir)  # ...but it is there and refused, not deleted
    resumed = trainer(env, c, out="crash")
    assert resumed.run() == 0
    assert _final_state(resumed) == _final_state(ref)


def test_corrupted_optimizer_state_rolls_back_to_previous_verified(env):
    c = contract(env["ds_sha"], env["tok"], env["digest"], checkpoint={"every_updates": 6})
    ref = trainer(env, c, out="ref")
    assert ref.run() == 0
    a = trainer(env, c, out="rb")
    assert a.run(stop_after=13) == S.EXIT_STOPPED
    (u12, p12), (u6, _) = _ckpts(a.run_dir)[1], _ckpts(a.run_dir)[0]
    assert (u6, u12) == (6, 12)
    raw = bytearray((p12 / "optim.pt").read_bytes())
    raw[len(raw) // 2] ^= 0xFF  # same size, different bytes
    (p12 / "optim.pt").write_bytes(bytes(raw))
    assert not C.verify_checkpoint(p12)[0]
    b = trainer(env, c, out="rb")
    assert b.run() == 0
    log = [json.loads(x) for x in (b.run_dir / "sft_log.jsonl").read_text().splitlines()]
    restored = [e for e in log if e.get("event") == "restored"][-1]
    assert restored["update"] == 6 and _final_state(b) == _final_state(ref)


def test_explicit_rollback_to_an_earlier_checkpoint(env):
    c = contract(env["ds_sha"], env["tok"], env["digest"], checkpoint={"every_updates": 5})
    a = trainer(env, c, out="rbk")
    assert a.run() == 0
    (_, p5), *_ = _ckpts(a.run_dir)
    b = trainer(env, c, out="rbk")
    b.restore(p5)
    assert b.update == 5
    m = C.read_manifest(p5)
    assert tensor_tree_sha(b.model.state_dict()) == m["state_sha256"]["model"]


def test_changed_contract_dataset_or_precision_cannot_resume(env, tmp_path):
    c = contract(env["ds_sha"], env["tok"], env["digest"], checkpoint={"every_updates": 5})
    assert trainer(env, c, out="cc").run(stop_after=6) == S.EXIT_STOPPED
    for change in ({"optimizer": {"lr": 1e-3}}, {"seed": 8}, {"batch": {"episodes_per_update": 8}}):
        with pytest.raises(S.ContractError, match="new run_id"):
            trainer(
                env,
                contract(env["ds_sha"], env["tok"], env["digest"], checkpoint={"every_updates": 5}, **change),
                out="cc",
            ).run()
    ds2, sha2, _ = make_dataset(tmp_path, seed=9, name="ds2")
    env2 = {**env, "ds": ds2, "ds_sha": sha2}
    with pytest.raises(S.ContractError, match="new run_id"):
        trainer(env2, contract(sha2, env["tok"], env["digest"], checkpoint={"every_updates": 5}), out="cc").run()
    with pytest.raises(S.ContractError, match="refusing to restart"):
        trainer(env, c, out="cc").run(resume=False)


def test_changed_tokenizer_is_refused(env):
    c = contract(env["ds_sha"], env["tok"], env["digest"])
    env["tok"].write_text('{"changed": true}')
    with pytest.raises(Exception, match="tokenizer"):
        trainer(env, c)


def test_signal_saves_after_the_update_then_exits(env):
    c = contract(env["ds_sha"], env["tok"], env["digest"])
    t = trainer(env, c)
    t._on_signal(signal.SIGTERM, None)
    assert t.run() == S.EXIT_SIGNAL
    assert [u for u, _ in _ckpts(t.run_dir)] == [1]
    t2 = trainer(env, c)
    assert t2.run() == 0 and t2.update == 20


def test_nonfinite_loss_aborts_without_applying_or_saving(env):
    c = contract(env["ds_sha"], env["tok"], env["digest"])
    t = trainer(env, c)
    with torch.no_grad():
        t.model.embed.weight[50:60] = float("nan")
    before = tensor_tree_sha(t.model.state_dict())
    assert t.run() == S.EXIT_NUMERIC
    assert tensor_tree_sha(t.model.state_dict()) == before and t.update == 0 and _ckpts(t.run_dir) == []


def test_failed_checkpoint_write_stops_with_exit_3(env, monkeypatch):
    c = contract(env["ds_sha"], env["tok"], env["digest"], checkpoint={"every_updates": 3})
    t = trainer(env, c)

    def boom(*_a, **_k):
        raise OSError(28, "No space left on device (injected)")

    monkeypatch.setattr(C, "save_checkpoint", boom)
    assert t.run() == S.EXIT_CKPT_FAILED and t.update == 3
    log = [json.loads(x) for x in (t.run_dir / "sft_log.jsonl").read_text().splitlines()]
    assert [e["attempt"] for e in log if e.get("event") == "checkpoint_failed"] == [1, 2]
    assert not any(e.get("event") == "checkpoint" for e in log)


def test_frozen_base_bytes_are_never_modified(env):
    before = {p.name: C.sha256_file(p) for p in env["base"].iterdir()}
    c = contract(env["ds_sha"], env["tok"], env["digest"], checkpoint={"every_updates": 5})
    assert trainer(env, c).run() == 0
    assert {p.name: C.sha256_file(p) for p in env["base"].iterdir()} == before
    m = C.read_manifest(_ckpts(env["tmp"] / "out" / "run")[-1][1])
    assert m["lineage"]["root"] == "juniper1-base-v1" and m["base_freeze_record_sha256"] == env["digest"]
    assert m["exposure"]["target_tokens"] > 0 and m["data_cursor"]["epoch"] >= 0


# --- review fixes (independent trainer review 2026-10-12) -------------------------------------


def test_unqualified_precisions_are_refused(env):
    for p in ("bf16", "fp16"):
        with pytest.raises(S.ContractError, match="not qualified"):
            S.validate_contract(contract(env["ds_sha"], env["tok"], env["digest"], precision=p))


def test_torch_save_runtime_error_is_retried_then_exit_3(env, monkeypatch):
    c = contract(env["ds_sha"], env["tok"], env["digest"], checkpoint={"every_updates": 3})
    t = trainer(env, c)

    def boom(*_a, **_k):
        raise RuntimeError("[enforce fail at inline_container.cc] file write failed")

    monkeypatch.setattr(C, "save_checkpoint", boom)
    assert t.run() == S.EXIT_CKPT_FAILED
    log = [json.loads(x) for x in (t.run_dir / "sft_log.jsonl").read_text().splitlines()]
    assert [e["attempt"] for e in log if e.get("event") == "checkpoint_failed"] == [1, 2]


def test_one_process_per_run_directory_and_world_size(env, monkeypatch):
    c = contract(env["ds_sha"], env["tok"], env["digest"])
    first = trainer(env, c)
    with pytest.raises(S.ContractError, match="in use"):
        trainer(env, c)
    first.close()
    trainer(env, c).close()  # free again once the holder is closed
    monkeypatch.setenv("WORLD_SIZE", "2")
    with pytest.raises(S.ContractError, match="WORLD_SIZE"):
        trainer(env, c, out="ws")


def test_batch_contract_is_checked_at_startup(env):
    for batch in ({"token_budget": 8}, {"episodes_per_update": 10_000}):
        with pytest.raises(S.ContractError):
            trainer(env, contract(env["ds_sha"], env["tok"], env["digest"], batch=batch), out=f"b{len(batch)}{batch}")
    with pytest.raises(S.ContractError, match="log_every"):
        S.validate_contract(contract(env["ds_sha"], env["tok"], env["digest"], log_every=0))


def test_final_checkpoint_carries_the_final_dev_evaluation(env):
    c = contract(env["ds_sha"], env["tok"], env["digest"], checkpoint={"every_updates": 10}, dev_every=10)
    t = trainer(env, c)
    assert t.run() == 0
    (u, final), *_ = sorted(_ckpts(t.run_dir), reverse=True)
    m = C.read_manifest(final)
    assert u == 20 and m["cadence"]["reason"] == "final" and m["last_dev"]["update"] == 20


def test_sft_checkpoints_load_in_the_evaluation_tools(env):
    from j1bench.generate import load_checkpoint_model

    c = contract(env["ds_sha"], env["tok"], env["digest"])
    t = trainer(env, c)
    assert t.run() == 0
    model, ident = load_checkpoint_model(_ckpts(t.run_dir)[-1][1])
    assert ident["updates"] == 20
    assert tensor_tree_sha(model.state_dict()) == tensor_tree_sha({k: v.cpu() for k, v in t.model.state_dict().items()})


def test_loader_requires_stop_token_targets_and_unique_ids(tmp_path):
    x, y = episode([50, 60])
    bad = list(y)
    bad[-1] = 7  # supervised but not a stop token
    with pytest.raises(D.DatasetError, match="stop token"):
        D.check_alignment(np.array(x), np.array(bad), np.array([0, len(x)]))
    ds, _, tok = make_dataset(tmp_path, name="dup")
    with np.load(ds / "dev_rows.npz") as z:
        arrays = dict(z)
    arrays["episode_ids"][1] = arrays["episode_ids"][0]
    np.savez(ds / "dev_rows.npz", **arrays)
    m = json.loads((ds / "manifest.json").read_text())
    m["files"]["dev_rows.npz"]["sha256"] = C.sha256_file(ds / "dev_rows.npz")
    m.pop("manifest_sha256")
    m["manifest_sha256"] = hashlib.sha256(D.canonical_json(m)).hexdigest()
    (ds / "manifest.json").write_text(json.dumps(m))
    with pytest.raises(D.DatasetError, match="duplicated"):
        D.SFTDataset(
            ds, expected_manifest_sha256=m["manifest_sha256"], tokenizer_sha256=C.sha256_file(tok), vocab_size=128
        )


def test_recovery_ab_tool_and_decision_rule(env, tmp_path):
    import importlib

    ab = importlib.import_module("tools.sft_recovery_ab")
    decide = importlib.import_module("tools.sft_decide")
    c = contract(env["ds_sha"], env["tok"], env["digest"], checkpoint={"every_updates": 7}, dev_every=10)
    assert trainer(env, c, out="A").run() == 0
    b = trainer(env, c, out="B")
    b._on_signal(signal.SIGTERM, None)
    assert b.run() == S.EXIT_SIGNAL
    assert trainer(env, c, out="B").run() == 0
    out = tmp_path / "ab.json"
    sys_argv = ["x", "--run-a", str(env["tmp"] / "A/run"), "--run-b", str(env["tmp"] / "B/run"), "--out", str(out)]
    old = sys.argv
    sys.argv = sys_argv
    try:
        assert ab.main() == 0
    finally:
        sys.argv = old
    res = json.loads(out.read_text())
    assert res["bitwise"] and res["b_resumed_from_update"] == 1 and res["updates_compared_after_resume"] == 19
    proto = json.loads((ROOT / "docs/posttrain/sft_pilot_protocol_v3.json").read_text())
    view = {"difference": 0.5, "bootstrap_ci95": [0.4, 0.6], "candidate": 0.7, "n": 500}
    comparison = {
        "test": {
            "versus_base_chat": {"instruction_following_in_distribution": view},
            "versus_base_fewshot": {"instruction_following_in_distribution": view},
            "protocol_valid": {"candidate": 0.95},
            "significant_regressions_vs_fewshot": [],
            "abstention": {"uncertainty_strict": 0.8, "false_abstention_on_facts": 0.1},
        }
    }
    ret = {"bpb": 0.93, "mode": "full", "precision": "fp32", "model": {"checkpoint": "ck"}}
    comparison["candidate_model"] = {"checkpoint": "ck"}
    ck = {"exact": True, "verify": [True, "ok"], "checkpoint": {"checkpoint": "ck"}}
    hist = {
        "checkpoint": "ck",
        "updates": proto["pilot"]["updates"],
        "dev_history": [{"update": u, "dev_loss": 1.0 / u} for u in (1, 2, 3, 4)],
    }
    good = decide.decide(proto, comparison, ret, res, ck, hist)
    assert good["decision"] == "PILOT_CANDIDATE_QUALIFIED"
    bad = decide.decide(proto, comparison, {**ret, "bpb": 0.95}, res, ck, hist)
    assert bad["decision"] == "NOT_PROMOTED" and not bad["gates"]["G-RET"]["pass"]
    rising = {**hist, "dev_history": [{"update": u, "dev_loss": float(u)} for u in (1, 2, 3, 4)]}
    assert not decide.decide(proto, comparison, ret, res, ck, rising)["gates"]["EARLY-STOP"]["pass"]
    vacuous = {**res, "bitwise": False, "numerical_only": False}
    assert not decide.decide(proto, comparison, ret, vacuous, ck, hist)["gates"]["G-REC"]["pass"]
    assert not decide.decide(proto, comparison, ret, res, {**ck, "verify": [False, "x"]}, hist)["gates"]["G-REC"][
        "pass"
    ]
    comparison["test"]["significant_regressions_vs_fewshot"] = ["math"]
    assert decide.decide(proto, comparison, ret, res, ck, hist)["decision"] == "NOT_PROMOTED"
    with pytest.raises(SystemExit, match="different candidates"):
        decide.decide(proto, comparison, {**ret, "model": {"checkpoint": "other"}}, res, ck, hist)


def test_resume_replays_a_dev_eval_saved_before_it_ran(env):
    c = contract(env["ds_sha"], env["tok"], env["digest"], checkpoint={"every_updates": 7}, dev_every=7)
    ref = trainer(env, c, out="dref")
    assert ref.run() == 0
    a = trainer(env, c, out="dsplit")
    assert a.run(stop_after=9) == S.EXIT_STOPPED  # checkpoint at 7 was published before update 7's dev eval
    b = trainer(env, c, out="dsplit")
    assert b.run() == 0
    assert b.dev_history == ref.dev_history


def _init_spec(ckpt):
    m = json.loads((ckpt / "manifest.json").read_text())
    return {"kind": "sft_checkpoint", "checkpoint": m["name"], "manifest_sha256": m["manifest_sha256"],
            "model_state_sha256": m["state_sha256"]["model"]}  # fmt: skip


def test_init_from_a_verified_sft_checkpoint_starts_a_new_lineage(env):
    a = trainer(env, contract(env["ds_sha"], env["tok"], env["digest"], run_id="first"), out="a")
    assert a.run() == 0
    ckpt = max(_ckpts(a.run_dir))[1]
    c = contract(env["ds_sha"], env["tok"], env["digest"], run_id="second", init=_init_spec(ckpt))
    b = trainer(env, c, out="b", init_checkpoint=ckpt)
    assert tensor_tree_sha(b.model.state_dict()) == _init_spec(ckpt)["model_state_sha256"]
    assert b.update == 0 and not b.opt.state  # fresh optimizer and schedule
    assert b.run() == 0
    final = json.loads((max(_ckpts(b.run_dir))[1] / "manifest.json").read_text())
    assert final["lineage"]["init"]["checkpoint"] == ckpt.name and final["lineage"]["root"] == "juniper1-base-v1"


def test_init_mismatches_are_refused(env):
    a = trainer(env, contract(env["ds_sha"], env["tok"], env["digest"], run_id="first"), out="a")
    a.run()
    ckpt = max(_ckpts(a.run_dir))[1]
    bad = {**_init_spec(ckpt), "model_state_sha256": "0" * 64}
    with pytest.raises(S.ContractError, match="init model state"):
        trainer(
            env, contract(env["ds_sha"], env["tok"], env["digest"], run_id="x", init=bad), out="x", init_checkpoint=ckpt
        )
    with pytest.raises(S.ContractError, match="declares none"):
        trainer(env, contract(env["ds_sha"], env["tok"], env["digest"], run_id="y"), out="y", init_checkpoint=ckpt)
    with pytest.raises(S.ContractError, match="pass --init-checkpoint"):
        trainer(env, contract(env["ds_sha"], env["tok"], env["digest"], run_id="z", init=_init_spec(ckpt)), out="z")


def test_cli_refuses_a_silent_cpu_fallback(env, monkeypatch):
    monkeypatch.setattr(S.torch.cuda, "is_available", lambda: False)
    cpath = env["tmp"] / "c.json"
    cpath.write_text(json.dumps(contract(env["ds_sha"], env["tok"], env["digest"])))
    args = ["--contract", str(cpath), "--freeze-record", str(env["record"]), "--base-checkpoint", str(env["base"]),
            "--tokenizer", str(env["tok"]), "--dataset", str(env["ds"]), "--out-root", str(env["tmp"] / "o"),
            "--staging", str(env["tmp"] / "s")]  # fmt: skip
    with pytest.raises(SystemExit, match="no usable CUDA device"):
        S.main(args)
