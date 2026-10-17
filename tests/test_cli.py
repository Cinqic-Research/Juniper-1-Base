"""CLI smoke tests: argument parsing and the complete/verify/tokenize paths, with the model loader stubbed."""

import json

import torch

from juniper1 import cli
from juniper1.model import FROZEN_CONFIG, ModelConfig, Transformer
from juniper1.tokenizer import Juniper1Tokenizer

TINY = {
    **FROZEN_CONFIG,
    "n_layer": 2,
    "d_model": 32,
    "n_head": 4,
    "n_kv_head": 2,
    "head_dim": 8,
    "d_ff": 64,
    "max_seq_len": 64,
}


def _stub_loader(monkeypatch):
    torch.manual_seed(0)
    model = Transformer(ModelConfig(**TINY)).eval()
    tok = Juniper1Tokenizer()
    calls = {}

    def fake_load_model(source, *, revision=None, device="cpu", dtype=None, verify_digest=True):
        calls.update(source=source, revision=revision, device=device, dtype=dtype)
        return model, tok, {"files": {}, "config": "ok"}

    monkeypatch.setattr("juniper1.load.load_model", fake_load_model)
    return calls


def test_complete_runs_end_to_end(monkeypatch, capsys):
    calls = _stub_loader(monkeypatch)
    assert cli.main(["complete", "Hello", "--model", "stub/dir", "--max-new-tokens", "4", "--json"]) == 0
    rec = json.loads(capsys.readouterr().out)
    assert rec["prompt_tokens"] >= 1 and len(rec["generated_ids"]) == 4 and rec["stop"] in ("length", "eos", "context")
    assert calls["source"] == "stub/dir" and calls["dtype"] == torch.float32


def test_complete_plain_text_and_sampling(monkeypatch, capsys):
    _stub_loader(monkeypatch)
    assert (
        cli.main(
            [
                "complete",
                "Hi",
                "--model",
                "x",
                "--max-new-tokens",
                "3",
                "--temperature",
                "0.9",
                "--top-k",
                "5",
                "--seed",
                "3",
            ]
        )
        == 0
    )
    assert capsys.readouterr().out.startswith("Hi")


def test_verify_path(monkeypatch, capsys):
    _stub_loader(monkeypatch)
    assert cli.main(["verify", "--model", "x"]) == 0
    assert json.loads(capsys.readouterr().out)["tensor_digest"] == "ok"


def test_tokenize_path(capsys):
    assert cli.main(["tokenize", "text with <|user|> inside"]) == 0
    rec = json.loads(capsys.readouterr().out)
    assert rec["roundtrip_ok"] and all(i >= 64 for i in rec["ids"])
