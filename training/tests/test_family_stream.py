import hashlib
import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from j1bench.data import TokenStream
from j1bench.model import FAMILY, ModelConfig, Transformer, analytic_params
from j1bench.train import build_parser


@pytest.mark.parametrize("name", sorted(FAMILY))
@pytest.mark.parametrize("vocab", [24576, 28672, 32768])
def test_family_param_count_matches_closed_form(name, vocab):
    cfg = ModelConfig(vocab_size=vocab, **FAMILY[name])
    with torch.device("meta"):
        m = Transformer(cfg)
    assert m.num_params() == analytic_params(cfg)


def test_core_count_is_the_qualified_number():
    assert analytic_params(ModelConfig(vocab_size=32768, **FAMILY["j1-core-28x640"])) == 148_315_264


def test_param_groups_exclude_norms_and_embedding():
    m = Transformer(ModelConfig(vocab_size=128, n_layer=2, d_model=64, n_head=4, n_kv_head=2, head_dim=16, d_ff=128))
    decay, no_decay = m.param_groups(0.1)
    ids = {id(p) for p in no_decay["params"]}
    names = {n for n, p in m.named_parameters() if id(p) in ids}
    assert "embed.weight" in names and all(n == "embed.weight" or n.endswith("norm.weight") for n in names)
    assert all(p.dim() == 2 for p in decay["params"]) and no_decay["weight_decay"] == 0.0
    assert len(decay["params"]) + len(no_decay["params"]) == len(list(m.parameters()))
    assert len(m.param_groups(0.1, "all")) == 1


def test_kv_cache_decode_matches_full_forward():
    torch.manual_seed(0)
    cfg = ModelConfig(
        vocab_size=256, n_layer=3, d_model=64, n_head=4, n_kv_head=2, head_dim=16, d_ff=128, max_seq_len=64
    )
    m = Transformer(cfg).double().eval()
    seq = torch.randint(0, 256, (1, 20))
    full = m(seq)
    caches = [[] for _ in m.blocks]
    out = [m.step_logits(seq[:, :8], caches, 0)]
    for t in range(8, 20):
        out.append(m.step_logits(seq[:, t : t + 1], caches, t))
    torch.testing.assert_close(torch.stack(out[1:], 1), full[:, 8:20], rtol=1e-9, atol=1e-9)
    torch.testing.assert_close(out[0], full[:, 7], rtol=1e-9, atol=1e-9)


def _shards(tmp_path, n_docs=50):
    rng = np.random.default_rng(0)
    out = tmp_path / "shards"
    out.mkdir()
    rec = {"build": "b", "tokenizer": {"tokenizer_json_sha256": "t" * 64, "total_vocab": 512}, "splits": {}}
    for split in ("train", "validation"):
        docs = [rng.integers(320, 512, size=rng.integers(5, 60)) for _ in range(n_docs)]
        stream, offs = [], []
        for d in docs:
            offs.append(len(stream))
            stream += [*list(d), 0]
        arr = np.array(stream, dtype="<u2")
        half = len(arr) // 2
        files = []
        for k, part in enumerate((arr[:half], arr[half:])):
            name = f"{split}-tokens-{k:05d}.bin"
            part.tofile(out / name)
            files.append({"file": name, "tokens": len(part)})
        pq.write_table(
            pa.table(
                {"source_id": [f"s{i % 3}" for i in range(n_docs)], "offset": offs, "n_tokens": [len(d) for d in docs]}
            ),
            out / f"{split}-docs.parquet",
        )
        rec["splits"][split] = {"token_files": files, "doc_index": {"file": f"{split}-docs.parquet"}}
    (out / "shards.json").write_text(json.dumps(rec))
    return out, arr


def test_stream_crosses_files_and_resumes_exactly(tmp_path):
    out, _ = _shards(tmp_path)
    a = TokenStream(str(out), "train", 16, 4, seed=3)
    full = np.concatenate([np.fromfile(out / f"train-tokens-{k:05d}.bin", dtype="<u2") for k in range(2)])
    assert np.array_equal(a.tokens(0, len(full)), full.astype(np.int64))
    seen = [a.next_batch() for _ in range(3)]
    st = a.state_dict()
    nxt = [a.next_batch()[0] for _ in range(40)]  # crosses into the next epoch
    b = TokenStream(str(out), "train", 16, 4, seed=3)
    b.load_state_dict(st)
    assert all(torch.equal(x, b.next_batch()[0]) for x in nxt)
    assert a.epoch >= 1 and not torch.equal(seen[0][0], nxt[0])
    x, y, n = seen[0]
    assert torch.equal(x[:, 1:], y[:, :-1]) and n == y.numel()


def test_epoch_permutation_covers_every_window_once(tmp_path):
    out, _ = _shards(tmp_path)
    s = TokenStream(str(out), "train", 16, 1, seed=1)
    firsts = {tuple(s.next_batch()[0][0, :4].tolist()) for _ in range(s.n_windows)}
    assert len(firsts) == s.n_windows and s.epoch == 0


def test_limit_tokens_and_identity(tmp_path):
    out, _ = _shards(tmp_path)
    s = TokenStream(str(out), "train", 16, 2, limit_tokens=200)
    assert s.n_windows == (200 - 1) // 16
    ident = s.identity()
    assert ident["shards_json_sha256"] == hashlib.sha256((out / "shards.json").read_bytes()).hexdigest()


def test_eval_windows_attribute_targets_to_sources(tmp_path):
    out, _ = _shards(tmp_path)
    s = TokenStream(str(out), "validation", 16, 1, shuffle=False)
    wins = s.eval_windows(5)
    idx = pq.read_table(out / "validation-docs.parquet").to_pydict()
    for w, (_toks, src) in enumerate(wins):
        for j in range(16):
            pos = w * 16 + 1 + j
            doc = max(i for i, o in enumerate(idx["offset"]) if o <= pos)
            assert s.source_names[src[j]] == idx["source_id"][doc]


def test_parser_accepts_family_presets():
    a = build_parser().parse_args(["--preset", "j1-small-12x384", "--data", "tokens:/x", "--wd-policy", "matrices"])
    assert a.preset == "j1-small-12x384"


def test_truncated_or_corrupt_token_files_refused(tmp_path):
    out, _ = _shards(tmp_path)
    meta = json.loads((out / "shards.json").read_text())
    f0 = out / meta["splits"]["train"]["token_files"][0]["file"]
    meta["splits"]["train"]["token_files"][0]["sha256"] = hashlib.sha256(f0.read_bytes()).hexdigest()
    (out / "shards.json").write_text(json.dumps(meta))
    TokenStream(str(out), "train", 16, 2, verify=True)
    data = bytearray(f0.read_bytes())
    data[3] ^= 0xFF
    f0.write_bytes(bytes(data))
    with pytest.raises(ValueError, match="sha256"):
        TokenStream(str(out), "train", 16, 2, verify=True)
    f0.write_bytes(bytes(data[:-2]))
    with pytest.raises(ValueError, match="missing or not"):
        TokenStream(str(out), "train", 16, 2)


def test_preload_yields_identical_batches(tmp_path):
    out, _ = _shards(tmp_path)
    a = TokenStream(str(out), "train", 16, 3, seed=5)
    b = TokenStream(str(out), "train", 16, 3, seed=5, preload=True)
    assert b.identity()["preload"] is True
    for _ in range(25):
        assert torch.equal(a.next_batch()[0], b.next_batch()[0])
