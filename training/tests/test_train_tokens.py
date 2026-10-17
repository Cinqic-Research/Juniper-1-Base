"""End-to-end trainer run on j1.token-shards.v1 data: eval by source, bits per byte, exact resume."""

import hashlib
import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from tokenizers import Tokenizer, models

from j1bench.train import Trainer, build_parser

VOCAB = 512


def _tokenizer(path):
    vocab = {f"<|c{i}|>": i for i in range(64)}
    vocab.update({f"<0x{b:02X}>": 64 + b for b in range(256)})
    letters = [chr(ord("a") + i) for i in range(26)]
    for i in range(VOCAB - 320):
        vocab[letters[i % 26] * (1 + i // 26)] = 320 + i
    Tokenizer(models.BPE(vocab=vocab, merges=[], byte_fallback=True)).save(str(path))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _shards(tmp_path):
    out = tmp_path / "shards"
    out.mkdir()
    tj = tmp_path / "tokenizer.json"
    sha = _tokenizer(tj)
    rng = np.random.default_rng(0)
    rec = {"build": "b", "tokenizer": {"tokenizer_json_sha256": sha, "total_vocab": VOCAB}, "splits": {}}
    for split in ("train", "validation"):
        stream, offs, src = [], [], []
        for i in range(300):
            d = rng.integers(64, VOCAB, size=rng.integers(8, 120))
            offs.append(len(stream))
            src.append(f"s{i % 2}")
            stream += [*d.tolist(), 0]
        np.array(stream, dtype="<u2").tofile(out / f"{split}-tokens-00000.bin")
        pq.write_table(
            pa.table({"source_id": src, "offset": offs, "n_tokens": [0] * 300}), out / f"{split}-docs.parquet"
        )
        rec["splits"][split] = {
            "token_files": [{"file": f"{split}-tokens-00000.bin", "tokens": len(stream)}],
            "doc_index": {"file": f"{split}-docs.parquet"},
        }
    (out / "shards.json").write_text(json.dumps(rec))
    return out, tj


def _args(tmp_path, out, tj, extra):
    return build_parser().parse_args(
        [
            "--preset",
            "tiny",
            "--data",
            f"tokens:{out}",
            "--tokenizer",
            str(tj),
            "--seq-len",
            "64",
            "--micro-bs",
            "4",
            "--accum",
            "2",
            "--steps",
            "12",
            "--warmup",
            "2",
            "--lr",
            "1e-3",
            "--device",
            "cpu",
            "--precision",
            "fp32",
            "--eval-every",
            "4",
            "--eval-windows",
            "8",
            "--ckpt-root",
            str(tmp_path / "ck"),
            "--ckpt-every-steps",
            "6",
            "--stage-dir",
            str(tmp_path / "stage"),
            "--run-id",
            "t",
            "--metrics",
            str(tmp_path / extra[0]),
            *extra[1:],
        ]
    )


def test_tokens_training_eval_and_exact_resume(tmp_path):
    out, tj = _shards(tmp_path)
    full = Trainer(_args(tmp_path, out, tj, ["full.jsonl"])).run()
    assert full["model"]["vocab_size"] == VOCAB
    det = full["last_eval_detail"]
    assert set(det["by_source"]) == {"s0", "s1"} and det["bpb"] > 0
    assert all(v["bytes"] > 0 for v in det["by_source"].values())
    ck = sorted(p for p in (tmp_path / "ck" / "t").iterdir() if p.name.startswith("ckpt-000000004"))[0]
    resumed = Trainer(
        _args(tmp_path, out, tj, ["resumed.jsonl", "--resume", str(ck), "--ckpt-root", str(tmp_path / "ck2")])
    ).run()
    a = [json.loads(x) for x in (tmp_path / "full.jsonl").read_text().splitlines()]
    b = [json.loads(x) for x in (tmp_path / "resumed.jsonl").read_text().splitlines()]
    steps_a = {r["step"]: r for r in a if "batch_fp" in r and r["step"] > 4}
    steps_b = {r["step"]: r for r in b if "batch_fp" in r}
    assert sorted(steps_b) == list(range(5, 13))
    for s in steps_b:
        assert steps_a[s]["batch_fp"] == steps_b[s]["batch_fp"]
        assert steps_a[s]["loss_hex"] == steps_b[s]["loss_hex"]
    assert resumed["final_param_sha256"] == full["final_param_sha256"]

    from j1bench import timeaudit

    rep = timeaudit.audit(tmp_path / "timelogs" / "t")
    assert rep["ok"], rep["problems"]
    roll = timeaudit.rollup(tmp_path / "timelogs" / "t")
    cats = roll["per_resource"]["gpu:cpu"]["categories_s"]
    assert {"checkpoint", "evaluation", "gpu_compute", "restoration"} <= set(cats)
    ends = [e for e in timeaudit.load(tmp_path / "timelogs" / "t")[0] if e["event_type"] == "run_end"]
    assert [e["outcome"] for e in ends] == ["completed", "completed"]


def test_resumed_second_microbatch_is_covered_by_batch_fp(tmp_path):
    """Review N-06: with --accum 2, perturbing the second microbatch of the resumed update changes ``batch_fp``."""
    out, tj = _shards(tmp_path)
    Trainer(_args(tmp_path, out, tj, ["full.jsonl"])).run()
    ck = sorted(p for p in (tmp_path / "ck" / "t").iterdir() if p.name.startswith("ckpt-000000004"))[0]
    tr = Trainer(_args(tmp_path, out, tj, ["resumed.jsonl", "--resume", str(ck), "--ckpt-root", str(tmp_path / "ck2")]))
    inner, calls = tr.data.next_batch, {"n": 0}

    def perturbed():
        x, y, u = inner()
        calls["n"] += 1
        if calls["n"] == 2:  # second microbatch of the first resumed update (step 5)
            x = x.clone()
            x[0, 0] = (int(x[0, 0]) + 1) % VOCAB
        return x, y, u

    tr.data.next_batch = perturbed
    tr.run()
    a, b = (
        {r["step"]: r for r in map(json.loads, (tmp_path / f).read_text().splitlines()) if "batch_fp" in r}
        for f in ("full.jsonl", "resumed.jsonl")
    )
    assert a[5]["batch_fp"] != b[5]["batch_fp"]
    assert all(a[s]["batch_fp"] == b[s]["batch_fp"] for s in range(6, 13))  # the data order itself is unchanged


def test_tokenizer_mismatch_refused(tmp_path):
    out, tj = _shards(tmp_path)
    tj.write_text(tj.read_text() + " ")
    t = Trainer(_args(tmp_path, out, tj, ["m.jsonl", "--no-init-ckpt"]))
    try:
        t.evaluate()
    except SystemExit as e:
        assert "does not match" in str(e)
    else:
        raise AssertionError("mismatched tokenizer accepted")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_fp16_grad_scaler_state_resumes_exactly_on_gpu(tmp_path):
    out, tj = _shards(tmp_path)
    gpu = ["--device", "cuda", "--precision", "fp16", "--deterministic"]

    full = Trainer(_args(tmp_path, out, tj, ["f16.jsonl", *gpu])).run()
    ck = sorted(p for p in (tmp_path / "ck" / "t").iterdir() if p.name.startswith("ckpt-000000004"))[0]
    res = Trainer(
        _args(tmp_path, out, tj, ["f16r.jsonl", *gpu, "--resume", str(ck), "--ckpt-root", str(tmp_path / "ck2")])
    ).run()
    assert res["final_param_sha256"] == full["final_param_sha256"]
    assert res["final_optim_sha256"] == full["final_optim_sha256"]
