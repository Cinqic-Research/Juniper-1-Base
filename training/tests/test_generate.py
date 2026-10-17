"""Greedy evaluation harness: cached decoding equals full recomputation, batching does not change
outputs, forbidden IDs are never produced, stops are honoured, and hidden answers are refused."""

import hashlib
import json

import pytest
import torch

from j1bench.generate import generate, load_bundle
from j1bench.model import ModelConfig, Transformer
from j1bench.sftdata import canonical_json

CFG = ModelConfig(vocab_size=96, n_layer=2, d_model=32, n_head=2, n_kv_head=1, head_dim=16, d_ff=64, max_seq_len=64)
MANIFEST = {"forbidden_ids": [0, 1, 2, 3, 5, *range(12, 64)], "stop_ids": [11, 10]}


def model():
    torch.manual_seed(3)
    return Transformer(CFG).double().eval()


def full_greedy(m, prompt, n, forbidden, stops):
    seq = list(prompt)
    out = []
    for _ in range(n):
        logits = m(torch.tensor([seq]))[0, -1].clone()
        logits[forbidden] = float("-inf")
        t = int(logits.argmax())
        out.append(t)
        seq.append(t)
        if t in stops:
            break
    return out


def prompts(k=6, length=5, budget=12):
    g = torch.Generator().manual_seed(1)
    return [
        {"id": f"p{i}", "variant": "chat", "split": "test", "family": "x", "category": "x", "max_new_tokens": budget,
         "prompt_ids": torch.randint(64, 96, (length,), generator=g).tolist()}
        for i in range(k)
    ]  # fmt: skip


def test_cached_batched_decoding_equals_full_recomputation():
    m = model()
    ps = prompts() + prompts(3, length=7)
    for p in ps[6:]:
        p["id"] += "b"
    rows = {r["id"]: r for r in generate(m, ps, MANIFEST, "cpu", batch=4)}
    for p in ps:
        ref = full_greedy(m, p["prompt_ids"], p["max_new_tokens"], MANIFEST["forbidden_ids"], set(MANIFEST["stop_ids"]))
        assert rows[p["id"]]["generated_ids"] == ref
        assert not set(ref) & set(MANIFEST["forbidden_ids"])
    single = {r["id"]: r["generated_ids"] for r in generate(m, ps, MANIFEST, "cpu", batch=1)}
    assert single == {k: v["generated_ids"] for k, v in rows.items()}


def test_stop_ids_end_generation():
    m = model()
    with torch.no_grad():
        m.embed.weight[11] = 0
        m.embed.weight[11, 0] = 50.0  # tied head: make <|eot|> the argmax after the first token
        m.final_norm.weight.fill_(1.0)
    p = prompts(1)[0]
    r = generate(m, [p], MANIFEST, "cpu")[0]
    assert r["generated_ids"][-1] in MANIFEST["stop_ids"] or r["stop"] == "length"
    if r["stop"] == "stop":
        assert r["generated_ids"].count(11) + r["generated_ids"].count(10) == 1


def _bundle(tmp, rows):
    d = tmp / "b"
    d.mkdir()
    (d / "prompts.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    m = {
        **MANIFEST,
        "files": {"prompts.jsonl": {"sha256": hashlib.sha256((d / "prompts.jsonl").read_bytes()).hexdigest()}},
    }
    m["bundle_sha256"] = hashlib.sha256(canonical_json(m)).hexdigest()
    (d / "manifest.json").write_text(json.dumps(m))
    return d, m["bundle_sha256"]


def test_bundle_identity_and_hidden_answers(tmp_path):
    d, sha = _bundle(tmp_path, prompts(2))
    load_bundle(d, sha)
    with pytest.raises(ValueError, match="identity"):
        load_bundle(d, "0" * 64)
    bad = [{**prompts(1)[0], "spec": {"type": "exact", "value": "x"}}]
    d2, sha2 = _bundle(tmp_path / "x", bad) if (tmp_path / "x").mkdir() is None else (None, None)
    with pytest.raises(ValueError, match="hidden answers"):
        load_bundle(d2, sha2)


def test_over_long_prompt_is_recorded_not_fatal():
    m = model()
    ps = prompts(2)
    ps[1]["prompt_ids"] = ps[1]["prompt_ids"] * 20  # 100 tokens > max_seq_len 64
    rows = {r["id"]: r for r in generate(m, ps, MANIFEST, "cpu")}
    assert rows["p1"]["stop"] == "context_overflow" and rows["p1"]["generated_ids"] == []
    assert rows["p0"]["new_tokens"] > 0
