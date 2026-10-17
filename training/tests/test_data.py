import numpy as np
import pytest

from j1bench.data import IGNORE, MemmapPacked, SyntheticPacked, batch_fingerprint, write_token_file


def test_packed_stream_covers_tokens_exactly_once_and_resumes_exactly():
    a = SyntheticPacked(101, 32, 3, seed=5, min_len=5, max_len=40)
    seq = [a.next_batch() for _ in range(4)]
    state = a.state_dict()
    nxt = a.next_batch()
    b = SyntheticPacked(101, 32, 3, seed=5, min_len=5, max_len=40)
    b.load_state_dict(state)
    again = b.next_batch()
    assert batch_fingerprint(nxt[0], nxt[1]) == batch_fingerprint(again[0], again[1])
    # labels are inputs shifted by one within a row
    x, y, u = seq[0]
    assert (x[0, 1:] == y[0, :-1]).all()
    flat_y = np.concatenate([s[1].numpy().ravel() for s in seq])
    flat_x = np.concatenate([s[0].numpy().ravel() for s in seq])
    assert (flat_y[:-1] == flat_x[1:]).all()  # every token predicted exactly once, no gaps
    assert u == 3 * 32


def test_padded_mode_counts_only_useful_tokens():
    d = SyntheticPacked(101, 64, 4, seed=1, min_len=5, max_len=30, pad_mode=True)
    _, y, u = d.next_batch()
    assert u == int((y != IGNORE).sum()) and u < 4 * 64


def test_memmap_resume(tmp_path):
    p = tmp_path / "tok.bin"
    sha = write_token_file(str(p), 10_000, 500)
    a = MemmapPacked(str(p), 500, 64, 2, sha)
    for _ in range(3):
        a.next_batch()
    s = a.state_dict()
    n1 = a.next_batch()
    b = MemmapPacked(str(p), 500, 64, 2, sha)
    b.load_state_dict(s)
    n2 = b.next_batch()
    assert (n1[0] == n2[0]).all() and (n1[1] == n2[1]).all()


def test_memmap_contract_binds_content_and_rejects_same_size_replacement(tmp_path):
    p = tmp_path / "tok.bin"
    sha = write_token_file(str(p), 1000, 500)
    source = MemmapPacked(str(p), 500, 64, 2)
    assert source.identity()["file_sha256"] == sha
    with p.open("r+b") as f:
        f.seek(10)
        original = f.read(1)
        f.seek(10)
        f.write(bytes([original[0] ^ 1]))
    with pytest.raises(ValueError, match="sha256 mismatch"):
        MemmapPacked(str(p), 500, 64, 2, sha)
    assert MemmapPacked(str(p), 500, 64, 2).identity()["file_sha256"] != sha


def test_batches_fingerprint_covers_every_microbatch_in_order():
    """Review N-06: with accumulation, every microbatch of an update is fingerprinted, in order."""
    from j1bench.data import batches_fingerprint

    d = SyntheticPacked(101, 32, 3, seed=5, min_len=5, max_len=40)
    b1, b2 = d.next_batch(), d.next_batch()
    fp = batches_fingerprint([b1, b2])
    assert batches_fingerprint([b1]) == batch_fingerprint(b1[0], b1[1])  # one batch: unchanged semantics
    assert fp != batches_fingerprint([b1]) and fp != batches_fingerprint([b2, b1])
    x2 = b2[0].clone()
    x2[0, 0] = (int(x2[0, 0]) + 1) % 101
    assert batches_fingerprint([b1, (x2, b2[1], b2[2])]) != fp  # a perturbed second microbatch changes it
    assert batches_fingerprint([b1, b2]) == fp  # deterministic


def _shards_with_eos(tmp_path, eos):
    import json

    import pyarrow as pa
    import pyarrow.parquet as pq

    out = tmp_path / "shards"
    out.mkdir()
    rng = np.random.default_rng(0)
    stream, offs, lens, src = [], [], [], []
    for i in range(12):
        d = [int(v) for v in rng.integers(0, 50, size=int(rng.integers(3, 20))) if v != eos]
        offs.append(len(stream))
        lens.append(len(d))
        src.append(f"s{i % 2}")
        stream += [*d, eos]
    np.array(stream, dtype="<u2").tofile(out / "validation-tokens-00000.bin")
    pq.write_table(pa.table({"source_id": src, "offset": offs, "n_tokens": lens}), out / "validation-docs.parquet")
    rec = {
        "build": "b",
        "eos_id": eos,
        "tokenizer": {"tokenizer_json_sha256": "0" * 64, "total_vocab": 64},
        "splits": {
            "validation": {
                "token_files": [{"file": "validation-tokens-00000.bin", "tokens": len(stream)}],
                "doc_index": {"file": "validation-docs.parquet"},
            }
        },
    }
    (out / "shards.json").write_text(json.dumps(rec))
    return out, np.array(stream, dtype=np.int64)


def test_full_split_eval_uses_the_split_eos_id(tmp_path):
    """Review N-07: the leading context-free EOS and ``eos_targets`` follow shards.json ``eos_id``, not 0."""
    from j1bench.data import IGNORE, TokenStream, full_split_eval

    out, stream = _shards_with_eos(tmp_path, eos=7)
    s = TokenStream(str(out), "validation", 16, 1, shuffle=False)
    assert s.eos == 7
    e = full_split_eval(s)
    assert e["x"][0, 0] == 7  # leading EOS is the split's separator
    assert e["identity"]["eos_targets"] == int((stream == 7).sum()) == 12
    flat_y = e["y"].ravel()
    assert (flat_y[flat_y != IGNORE] == stream).all()  # every stream token scored exactly once
    assert int((stream == 0).sum()) != 12  # the literal 0 would have given a different count
