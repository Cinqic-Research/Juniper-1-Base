"""The frozen-base loader returns weights only for the exact bytes its freeze record binds."""

import json

import pytest
import torch

from j1bench import ckpt as C
from j1bench.frozen import DIGEST_KEY, FREEZE_SCHEMA, FreezeError, load_frozen_base, record_digest
from j1bench.model import ModelConfig, Transformer
from j1bench.train import tensor_tree_sha

CFG = {
    "d_ff": 32, "d_model": 16, "head_dim": 8, "init_std": 0.02, "max_seq_len": 16, "n_head": 2, "n_kv_head": 1,
    "n_layer": 2, "norm_eps": 1e-5, "qk_norm": "rmsnorm_per_head", "rope_theta": 100000.0, "tie_embeddings": True,
    "vocab_size": 64,
}  # fmt: skip


def _fixture(tmp_path):
    torch.manual_seed(0)
    model = Transformer(ModelConfig(**CFG))
    sd = model.state_dict()
    payload = {"model.pt": sd, "optim.pt": {"s": torch.ones(3)}, "trainer.pt": {"step": 7}}
    meta = {"run_id": "r", "run_contract_sha256": "abc", "run_contract": {"model": CFG}}
    path = C.save_checkpoint(
        tmp_path / "ck", tmp_path / "st", step=7, payload=payload, meta=meta, torch_save=torch.save
    ).path
    man = json.loads((path / "manifest.json").read_text())
    tok = tmp_path / "tokenizer.json"
    tok.write_text("{}")
    rec = {
        "schema": FREEZE_SCHEMA,
        "freeze_id": "juniper1-base-v1",
        "decision": "BASE_FREEZE_APPROVED",
        "source_checkpoint": {
            "name": path.name, "run_id": "r", "step": 7, "manifest_sha256": man["manifest_sha256"],
            "files": {"model.pt": {"sha256": man["files"]["model.pt"]["sha256"]}},
        },
        "model": {"config": CFG, "state_sha256": tensor_tree_sha(sd), "parameters": model.num_params()},
        "tokenizer": {"sha256": C.sha256_file(tok)},
    }  # fmt: skip
    return path, rec, tok, model


def _write(tmp_path, rec, rehash=True):
    if rehash:
        rec = {**rec, DIGEST_KEY: record_digest(rec)}
    p = tmp_path / "freeze.json"
    p.write_text(json.dumps(rec))
    return p, rec.get(DIGEST_KEY, "")


def _load(tmp_path, rec, path, tok, rehash=True):
    p, digest = _write(tmp_path, rec, rehash)
    return load_frozen_base(p, path, tokenizer_path=tok, expected_record_sha256=digest)


def test_loads_exact_frozen_weights(tmp_path):
    path, rec, tok, ref = _fixture(tmp_path)
    model, got = _load(tmp_path, rec, path, tok)
    assert got["freeze_id"] == "juniper1-base-v1" and not model.training
    x = torch.randint(0, 64, (1, 8))
    assert torch.equal(model(x), ref.eval()(x))


@pytest.mark.parametrize("fid", ["latest", "best", "juniper1-base-latest", "juniper1-base-v0", ""])
def test_ambiguous_or_unversioned_ids_refused(tmp_path, fid):
    path, rec, tok, _ = _fixture(tmp_path)
    with pytest.raises(FreezeError, match="versioned"):
        _load(tmp_path, {**rec, "freeze_id": fid}, path, tok)


def test_unapproved_tampered_or_unpinned_record_refused(tmp_path):
    path, rec, tok, _ = _fixture(tmp_path)
    with pytest.raises(FreezeError, match="not BASE_FREEZE_APPROVED"):
        _load(tmp_path, {**rec, "decision": "BASE_FREEZE_BLOCKED"}, path, tok)
    good = {**rec, DIGEST_KEY: record_digest(rec)}
    with pytest.raises(FreezeError, match="self-hash"):
        _load(tmp_path, {**good, "freeze_id": "juniper1-base-v2"}, path, tok, rehash=False)
    # A consistently rehashed but different record is refused by the caller's pin.
    p, _ = _write(tmp_path, {**rec, "freeze_id": "juniper1-base-v2"})
    with pytest.raises(FreezeError, match="pinned"):
        load_frozen_base(p, path, tokenizer_path=tok, expected_record_sha256=good[DIGEST_KEY])


def test_record_identity_mismatches_refused(tmp_path):
    path, rec, tok, _ = _fixture(tmp_path)
    src = rec["source_checkpoint"]
    cases = [
        ({**rec, "source_checkpoint": {**src, "name": "ckpt-000000007-deadbeef"}}, "is not the frozen"),
        ({**rec, "source_checkpoint": {**src, "step": 8}}, "manifest identity"),
        ({**rec, "source_checkpoint": {**src, "manifest_sha256": "2" * 64}}, "manifest identity"),
        ({**rec, "source_checkpoint": {**src, "files": {"model.pt": {"sha256": "3" * 64}}}}, "model.pt bytes"),
        ({**rec, "tokenizer": {"sha256": "1" * 64}}, "tokenizer"),
        ({**rec, "model": {**rec["model"], "config": {**CFG, "rope_theta": 10000.0}}}, "configuration"),
        ({**rec, "model": {**rec["model"], "state_sha256": "0" * 64}}, "tensor-tree"),
        ({**rec, "model": {**rec["model"], "parameters": rec["model"]["parameters"] + 1}}, "parameter count"),
    ]
    for bad, reason in cases:
        with pytest.raises(FreezeError, match=reason):
            _load(tmp_path, bad, path, tok)


def test_changed_checkpoint_bytes_refused(tmp_path):
    path, rec, tok, _ = _fixture(tmp_path)
    raw = bytearray((path / "model.pt").read_bytes())
    raw[-1] ^= 1
    (path / "model.pt").write_bytes(bytes(raw))
    with pytest.raises(FreezeError, match="verification failed"):
        _load(tmp_path, rec, path, tok)
