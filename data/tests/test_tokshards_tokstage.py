import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from j1data import jtok, tokshards, tokstage
from j1data.hashing import unit_interval

TEXTS = {
    "train": [f"def f{i}(x):\n    return x * {i}  # doc {i}\n" for i in range(300)]
    + [f"Paragraph {i}: the color of the sky is blue." for i in range(300)],
    "validation": ["Validation text about behavior.", "    print('valid')\n"],
    "test": ["Test document."],
}


def _fixture(tmp_path: Path) -> Path:
    root = tmp_path / "data"
    acc = root / "accepted" / "b1"
    shards = []
    for split, texts in TEXTS.items():
        for part in range(2):
            rows = texts[part::2]
            if not rows:
                continue
            rel = f"{split}/src{part}/s.parquet"
            (acc / rel).parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(
                pa.table(
                    {
                        "doc_id": [f"{split}-{part}-{k}" for k in range(len(rows))],
                        "text": rows,
                        "source_id": [f"src{part}"] * len(rows),
                        "domain": ["computing"] * len(rows),
                    }
                ),
                acc / rel,
            )
            shards.append({"file": rel, "split": split, "source_id": f"src{part}"})
    (acc / "manifest.json").write_text(json.dumps({"shards": shards}))
    return root


def test_stage_sample_train_freeze_and_shards(tmp_path, monkeypatch):
    root = _fixture(tmp_path)
    summary = tokstage.run(root, "b1", rate=1.0, sizes=[384], out_dir=tmp_path / "tok", stem_alphabet=False)
    cand = summary["candidates"]["j1tok-0k"]
    assert cand["roundtrip_failures"] == 0 and cand["control_ids_from_text"] == 0
    assert summary["sample"]["documents"] == 600
    assert set(summary["eval_set"]) == {"src0", "src1"}  # validation only
    frozen = tmp_path / "frozen"
    rec = tokstage.freeze(tmp_path / "tok" / "j1tok-0k", frozen, "test", {})
    assert rec["total_vocab"] == 384 and rec["control_tokens"]["<|eot|>"] == 11
    with pytest.raises(SystemExit):
        tokstage.freeze(tmp_path / "tok" / "j1tok-0k", frozen, "again", {})

    monkeypatch.setattr(tokshards, "TOKENS_PER_FILE", 1000)  # force several stream files
    out = tmp_path / "shards"
    rec = tokshards.build(root, "b1", frozen, out)
    assert tokshards.verify(out) == []
    tr = rec["splits"]["train"]
    assert tr["documents"] == 600 and len(tr["token_files"]) > 1
    stream = np.concatenate([np.fromfile(out / f["file"], dtype="<u2") for f in tr["token_files"]])
    idx = pq.read_table(out / tr["doc_index"]["file"]).to_pydict()
    keys = [unit_interval(tokshards.ORDER_NAMESPACE, d) for d in idx["doc_id"]]
    assert keys == sorted(keys)  # global keyed order
    tok = jtok.load((frozen / "tokenizer.json").read_text())
    by_id = {f"train-{p}-{k}": t for p in range(2) for k, t in enumerate(TEXTS["train"][p::2])}
    for did, off, n in zip(idx["doc_id"], idx["offset"], idx["n_tokens"], strict=True):
        assert stream[off + n] == tokshards.EOS
        assert tok.decode(stream[off : off + n].tolist()) == by_id[did]
    assert int((stream == tokshards.EOS).sum()) == 600

    # Corruption is detected.
    victim = out / tr["token_files"][0]["file"]
    data = bytearray(victim.read_bytes())
    data[0] ^= 1
    victim.write_bytes(bytes(data))
    assert any("bad token file" in p for p in tokshards.verify(out))


def test_build_refuses_mismatched_tokenizer(tmp_path):
    root = _fixture(tmp_path)
    tokstage.run(root, "b1", rate=1.0, sizes=[384], out_dir=tmp_path / "tok", stem_alphabet=False)
    frozen = tmp_path / "frozen"
    tokstage.freeze(tmp_path / "tok" / "j1tok-0k", frozen, "test", {})
    (frozen / "tokenizer.json").write_text((frozen / "tokenizer.json").read_text() + " ")
    with pytest.raises(SystemExit, match="does not match"):
        tokshards.build(root, "b1", frozen, tmp_path / "s")


def test_prefix_export_equal_bytes(tmp_path):
    root = _fixture(tmp_path)
    tokstage.run(root, "b1", rate=1.0, sizes=[384], out_dir=tmp_path / "tok", stem_alphabet=False)
    frozen = tmp_path / "frozen"
    tokstage.freeze(tmp_path / "tok" / "j1tok-0k", frozen, "test", {})
    full = tmp_path / "full"
    tokshards.build(root, "b1", frozen, full)
    acc = root / "accepted" / "b1"
    doc_bytes = {}
    for f in acc.rglob("*.parquet"):
        t = pq.read_table(f).to_pydict()
        doc_bytes.update({d: len(x.encode()) for d, x in zip(t["doc_id"], t["text"], strict=True)})
    rec = tokshards.export_prefix(full, tmp_path / "pre", 5000, doc_bytes)
    assert tokshards.verify(tmp_path / "pre") == []
    k = rec["prefix_of"]["train_documents"]
    assert 0 < k < 600 and rec["prefix_of"]["train_text_bytes"] <= 5000
    whole = np.concatenate(
        [
            np.fromfile(full / f["file"], dtype="<u2")
            for f in json.loads((full / "shards.json").read_text())["splits"]["train"]["token_files"]
        ]
    )
    pre = np.fromfile(tmp_path / "pre" / rec["splits"]["train"]["token_files"][0]["file"], dtype="<u2")
    assert np.array_equal(pre, whole[: pre.size]) and pre[-1] == tokshards.EOS
    assert int((pre == tokshards.EOS).sum()) == k


def test_sample_exclusion_is_applied_and_recorded(tmp_path):
    root = _fixture(tmp_path)
    excluded = frozenset({"train-0-0", "train-1-5", "not-in-corpus"})
    texts, info = tokstage.train_sample(root, "b1", 1.0, excluded)
    assert info["documents"] == 598 and len(texts) == 598
    assert info["excluded_doc_ids"] == ["train-0-0", "train-1-5"]
    assert "exclusion_list_sha256" in info
    full, info_full = tokstage.train_sample(root, "b1", 1.0)
    assert len(full) == 600 and "excluded_doc_ids" not in info_full


def test_prefix_excluding_removes_documents_and_keeps_byte_rule(tmp_path, monkeypatch):
    root = _fixture(tmp_path)
    tokstage.run(root, "b1", rate=1.0, sizes=[384], out_dir=tmp_path / "tok", stem_alphabet=False)
    frozen = tmp_path / "frozen"
    tokstage.freeze(tmp_path / "tok" / "j1tok-0k", frozen, "test", {})
    monkeypatch.setattr(tokshards, "TOKENS_PER_FILE", 1000)
    out = tmp_path / "shards"
    tokshards.build(root, "b1", frozen, out)
    idx = pq.read_table(out / "train-docs.parquet").to_pydict()
    doc_bytes = {
        f"train-{p}-{k}": len(t.encode()) for p in range(2) for k, t in enumerate(TEXTS["train"][p::2])
    }
    budget = sum(doc_bytes[d] for d in idx["doc_id"][:50])
    excl = frozenset({idx["doc_id"][3], idx["doc_id"][10]})
    rec = tokshards.export_prefix_excluding(out, tmp_path / "pre", budget, doc_bytes, excl)
    p = tmp_path / "pre"
    assert tokshards.verify(p) == []
    sub = pq.read_table(p / "train-docs.parquet").to_pydict()
    assert not excl & set(sub["doc_id"]) and rec["prefix_of"]["excluded_doc_ids"] == sorted(excl)
    total = rec["prefix_of"]["train_text_bytes"]
    assert total <= budget
    rest = [d for d in idx["doc_id"] if d not in excl and d not in set(sub["doc_id"])]
    assert total + doc_bytes[rest[0]] > budget  # same stopping rule as export_prefix
    assert sub["doc_id"] == [d for d in idx["doc_id"] if d not in excl][: len(sub["doc_id"])]  # keyed order
    tfiles = rec["splits"]["train"]["token_files"]
    assert len(tfiles) > 1  # TOKENS_PER_FILE is patched: the stream spans several files
    stream = np.concatenate([np.fromfile(p / f["file"], dtype="<u2") for f in tfiles])
    assert (
        rec["splits"]["validation"] == json.loads((out / "shards.json").read_text())["splits"]["validation"]
    )
    tok = jtok.load((frozen / "tokenizer.json").read_text())
    by_id = {f"train-{q}-{k}": t for q in range(2) for k, t in enumerate(TEXTS["train"][q::2])}
    for did, off, n in zip(sub["doc_id"], sub["offset"], sub["n_tokens"], strict=True):
        assert stream[off + n] == tokshards.EOS and tok.decode(stream[off : off + n].tolist()) == by_id[did]
