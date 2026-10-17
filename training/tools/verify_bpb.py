"""Independent check of the bits-per-byte denominator on real token shards.

For each candidate shard directory, every document of the chosen split is
  1. matched by ``doc_id`` to its accepted-corpus text,
  2. re-encoded from that text with the candidate tokenizer (no shard-writer code involved)
     and compared to the stored token IDs,
  3. decoded back to text and compared byte for byte,
  4. scored with ``j1bench.data.token_byte_lengths`` and compared to the source UTF-8 length.
It also checks that the shard split holds exactly the accepted split's documents, so the
full-split evaluation scores identical source bytes for every candidate.

Usage: verify_bpb.py ACCEPTED_SPLIT_DIR TOKENIZER_JSON SHARD_DIR [--split validation] [--out X.json]
Exit status is nonzero on any mismatch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from j1bench.data import token_byte_lengths


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("accepted_split_dir", type=Path)
    ap.add_argument("tokenizer", type=Path)
    ap.add_argument("shard_dir", type=Path)
    ap.add_argument("--split", default="validation")
    ap.add_argument("--max-docs", type=int, default=0, help="0 = all documents of the split")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()

    texts: dict[str, str] = {}
    for f in sorted(a.accepted_split_dir.rglob("*.parquet")):
        t = pq.read_table(f, columns=["doc_id", "text", "n_bytes"])
        for d, x, nb in zip(
            t.column("doc_id").to_pylist(), t.column("text").to_pylist(), t.column("n_bytes").to_pylist(), strict=True
        ):
            if len(x.encode("utf-8")) != nb:
                raise SystemExit(f"accepted n_bytes disagrees with text for {d}")
            texts[d] = x
    meta = json.loads((a.shard_dir / "shards.json").read_text())
    tj_sha = hashlib.sha256(a.tokenizer.read_bytes()).hexdigest()
    if tj_sha != meta["tokenizer"]["tokenizer_json_sha256"]:
        raise SystemExit("tokenizer does not match the shards")
    s = meta["splits"][a.split]
    idx = pq.read_table(a.shard_dir / s["doc_index"]["file"])
    stream = np.concatenate([np.fromfile(a.shard_dir / f["file"], dtype="<u2") for f in s["token_files"]]).astype(
        np.int64
    )
    tok = Tokenizer.from_file(str(a.tokenizer))
    tb = token_byte_lengths(a.tokenizer, meta["tokenizer"]["total_vocab"])
    ids = idx.column("doc_id").to_pylist()
    offs = idx.column("offset").to_numpy()
    lens = idx.column("n_tokens").to_numpy()
    n = len(ids) if not a.max_docs else min(a.max_docs, len(ids))
    bad = {"missing_text": 0, "reencode_mismatch": 0, "decode_mismatch": 0, "byte_table_mismatch": 0, "eos_missing": 0}
    examples: list = []
    total_bytes = 0
    for i in range(n):
        d = ids[i]
        if d not in texts:
            bad["missing_text"] += 1
            continue
        stored = stream[offs[i] : offs[i] + lens[i]]
        if stream[offs[i] + lens[i]] != 0:
            bad["eos_missing"] += 1
        text = texts[d]
        nb = len(text.encode("utf-8"))
        total_bytes += nb
        enc = np.array(tok.encode(text, add_special_tokens=False).ids, dtype=np.int64)
        if len(enc) != len(stored) or not np.array_equal(enc, stored):
            bad["reencode_mismatch"] += 1
            if len(examples) < 5:
                examples.append({"doc_id": d, "kind": "reencode", "stored": len(stored), "encoded": len(enc)})
        if tok.decode(stored.tolist(), skip_special_tokens=False) != text:
            bad["decode_mismatch"] += 1
            if len(examples) < 5:
                examples.append({"doc_id": d, "kind": "decode"})
        if int(tb[stored].sum()) != nb:
            bad["byte_table_mismatch"] += 1
            if len(examples) < 5:
                examples.append({"doc_id": d, "kind": "bytes", "table": int(tb[stored].sum()), "utf8": nb})
    coverage = {}
    if not a.max_docs:
        coverage = {
            "accepted_docs": len(texts),
            "shard_docs": len(ids),
            "set_equal": set(ids) == set(texts),
            "accepted_bytes": sum(len(x.encode("utf-8")) for x in texts.values()),
        }
    rec = {
        "tool": "verify_bpb.v1",
        "split": a.split,
        "tokenizer_json_sha256": tj_sha,
        "shards_json_sha256": hashlib.sha256((a.shard_dir / "shards.json").read_bytes()).hexdigest(),
        "documents_checked": n,
        "source_bytes_checked": total_bytes,
        "failures": bad,
        "examples": examples,
        "coverage": coverage,
        "ok": not any(bad.values()) and (not coverage or coverage["set_equal"]),
    }
    text = json.dumps(rec, indent=1)
    if a.out:
        a.out.write_text(text)
    print(text)
    return 0 if rec["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
