"""Independent check of the j1-base-v0.2 provisional token-count tables.

Written separately from ``j1data.tokshards`` (no shared code beyond pyarrow/numpy/tokenizers and the
frozen-tokenizer loader ``j1data.jtok.load``, which sets ``encode_special_tokens`` so literal control
spellings in text encode as ordinary bytes, as the shards were produced):

1. **Full table reconciliation** for every candidate: per split, the per-document ``n_tokens`` sum, the
   ``document_tokens``/``stream_tokens``/``separator_tokens`` fields, the ``by_source`` sums and the
   token-file sizes agree; each split holds exactly the accepted manifest's document IDs; every
   document is followed by exactly one EOS (checked by reading the token at ``offset + n_tokens``).
2. **Sampled unbatched re-encoding**: a keyed-hash sample (``sha256("j1.count-check.v1\\0" + doc_id)``),
   stratified so every (split, source) stratum contributes at least ``--min-per-stratum`` documents,
   is re-encoded **one document per call** from the accepted text with each candidate tokenizer and
   compared with the stored IDs and the per-document count.

This is a sampled encoding check plus a full arithmetic/inventory reconciliation; it is NOT a full
independent recount of every document's encoding.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from j1data.jtok import load as load_frozen_tokenizer

NS = b"j1.count-check.v1\0"


def _sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        while b := fh.read(1 << 24):
            h.update(b)
    return h.hexdigest()


def u(doc_id: str) -> float:
    return int.from_bytes(hashlib.sha256(NS + doc_id.encode()).digest()[:8], "big") / 2**64


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("accepted", type=Path, help="accepted/j1-base-v0.2")
    ap.add_argument("--cand", action="append", required=True, help="NAME=SHARD_DIR=TOKENIZER_JSON")
    ap.add_argument("--rate", type=float, default=0.002)
    ap.add_argument("--min-per-stratum", type=int, default=20)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--prefix-bytes", type=int, default=0, help="shard dirs are prefix exports of this byte budget"
    )
    ap.add_argument("--exclude", type=Path, help="JSON doc_id list removed from the prefix (adjudication)")
    a = ap.parse_args()
    t0 = time.time()
    man = json.loads((a.accepted / "manifest.json").read_text())
    # Accepted doc IDs and sources per split (metadata columns only).
    acc: dict[str, dict[str, str]] = {}
    for split in ("train", "validation", "test"):
        ids: dict[str, str] = {}
        for f in sorted((a.accepted / split).rglob("*.parquet")):
            t = pq.read_table(f, columns=["doc_id", "source_id"])
            ids.update(zip(t.column("doc_id").to_pylist(), t.column("source_id").to_pylist(), strict=True))
        acc[split] = ids
    expected_prefix: list[str] | None = None
    if a.prefix_bytes:
        # Independent recomputation of the prefix: pack order = ascending (unit_interval("j1.pack-order.v1",
        # doc_id), doc_id); skip excluded IDs; stop at the first document that would exceed the budget.
        nbytes: dict[str, int] = {}
        for f in sorted((a.accepted / "train").rglob("*.parquet")):
            t = pq.read_table(f, columns=["doc_id", "n_bytes"])
            nbytes.update(zip(t.column("doc_id").to_pylist(), t.column("n_bytes").to_pylist(), strict=True))

        def order_key(d: str) -> tuple[float, str]:
            h = hashlib.sha256(f"j1.pack-order.v1\x00{d}".encode()).digest()
            return ((int.from_bytes(h[:8], "big") >> 11) / float(1 << 53), d)

        excl = set(json.loads(a.exclude.read_text())) if a.exclude else set()
        expected_prefix, total = [], 0
        for d in sorted(nbytes, key=order_key):
            if d in excl:
                continue
            if total + nbytes[d] > a.prefix_bytes:
                break
            total += nbytes[d]
            expected_prefix.append(d)
        acc["train"] = {d: acc["train"][d] for d in expected_prefix}
        acc.pop("test", None)  # prefix exports hold the train prefix and the full validation split only
    # Sample selection: keyed rate plus the lowest-u documents of each stratum.
    sample: dict[str, set[str]] = {}
    for split, ids in acc.items():
        strata: dict[str, list] = defaultdict(list)
        for d, s in ids.items():
            strata[s].append((u(d), d))
        chosen = set()
        for xs in strata.values():
            xs.sort()
            chosen.update(d for x, d in xs if x < a.rate)
            chosen.update(d for _, d in xs[: a.min_per_stratum])
        sample[split] = chosen
    texts: dict[str, str] = {}
    for split in sample:
        for f in sorted((a.accepted / split).rglob("*.parquet")):
            t = pq.read_table(f, columns=["doc_id", "text"])
            for d, x in zip(t.column("doc_id").to_pylist(), t.column("text").to_pylist(), strict=True):
                if d in sample[split]:
                    texts[d] = x
    rec: dict = {
        "tool": "independent_count_check.v1",
        "accepted_manifest_sha256": hashlib.sha256((a.accepted / "manifest.json").read_bytes()).hexdigest(),
        "build_identity": man["build_identity"],
        "accepted_docs": {k: len(v) for k, v in acc.items()},
        "sample_namespace": NS.decode().rstrip("\0"),
        "sample_rate": a.rate,
        "min_per_stratum": a.min_per_stratum,
        "sample_docs": {k: len(v) for k, v in sample.items()},
        "prefix_bytes": a.prefix_bytes or None,
        "prefix_documents_expected": len(expected_prefix) if expected_prefix is not None else None,
        "sample_texts_found": len(texts),
        "candidates": {},
    }
    ok = len(texts) == sum(len(v) for v in sample.values())
    for spec in a.cand:
        name, sd, tj = spec.split("=")
        sd, tj = Path(sd), Path(tj)
        meta = json.loads((sd / "shards.json").read_text())
        tok = load_frozen_tokenizer(tj.read_text())  # jtok.load semantics: encode_special_tokens on
        c: dict = {
            "shards_json_sha256": hashlib.sha256((sd / "shards.json").read_bytes()).hexdigest(),
            "tokenizer_json_sha256": hashlib.sha256(tj.read_bytes()).hexdigest(),
            "tokenizer_matches_shards": hashlib.sha256(tj.read_bytes()).hexdigest()
            == meta["tokenizer"]["tokenizer_json_sha256"],
            "splits": {},
        }
        ok &= c["tokenizer_matches_shards"]
        c["accepted_manifest_matches"] = (
            meta.get("accepted_manifest_sha256") == rec["accepted_manifest_sha256"]
        )
        c["splits_present"] = sorted(meta["splits"]) == sorted(acc)
        ok &= c["accepted_manifest_matches"] and c["splits_present"]
        eos_id = int(meta["eos_id"])
        for split in sorted(acc):
            s = meta["splits"].get(split)
            if s is None:
                continue
            idx = pq.read_table(
                sd / s["doc_index"]["file"], columns=["doc_id", "source_id", "offset", "n_tokens"]
            )
            ids = idx.column("doc_id").to_pylist()
            src = idx.column("source_id").to_pylist()
            off = idx.column("offset").to_numpy().astype(np.int64)
            ln = idx.column("n_tokens").to_numpy().astype(np.int64)
            files = [np.memmap(sd / f["file"], dtype="<u2", mode="r") for f in s["token_files"]]
            starts = np.cumsum([0] + [len(f) for f in files])

            def tok_at(pos: np.ndarray, files=files, starts=starts) -> np.ndarray:
                fi = np.searchsorted(starts, pos, side="right") - 1
                return np.array(
                    [files[i][p - starts[i]] for i, p in zip(fi, pos, strict=True)], dtype=np.int64
                )

            by_src: dict[str, int] = defaultdict(int)
            for x, n in zip(src, ln.tolist(), strict=True):
                by_src[x] += n
            contiguous = bool(off[0] == 0 and np.all(off[1:] == off[:-1] + ln[:-1] + 1))
            eos_ok = bool(np.all(tok_at(off + ln) == eos_id))
            files_sha_ok = all(_sha_file(sd / f["file"]) == f.get("sha256") for f in s["token_files"])
            order_ok = expected_prefix is None or split != "train" or ids == expected_prefix
            sp = {
                "documents": len(ids),
                "accepted_documents": len(acc[split]),
                "doc_set_equal_accepted": set(ids) == set(acc[split]),
                "doc_ids_unique": len(set(ids)) == len(ids),
                "sum_n_tokens": int(ln.sum()),
                "table_document_tokens": s["document_tokens"],
                "table_stream_tokens": s["stream_tokens"],
                "table_separator_tokens": s["separator_tokens"],
                "file_tokens": int(starts[-1]),
                "by_source_equal": {k: v["tokens"] for k, v in s["by_source"].items()} == dict(by_src),
                "offsets_contiguous": contiguous,
                "token_files_sha256_ok": files_sha_ok,
                "prefix_order_equal_independent": order_ok,
                "eos_after_every_doc": eos_ok,
            }
            sp["arithmetic_ok"] = (
                sp["sum_n_tokens"] == s["document_tokens"]
                and s["stream_tokens"] == s["document_tokens"] + s["separator_tokens"]
                and s["separator_tokens"] == len(ids)
                and sp["file_tokens"] == s["stream_tokens"]
            )
            # Sampled unbatched re-encoding.
            pos = {d: i for i, d in enumerate(ids)}
            n_chk = mism_len = mism_ids = missing = 0
            examples = []
            for d in sorted(sample[split]):
                if d not in pos or d not in texts:
                    missing += 1
                    continue
                i = pos[d]
                enc = tok.encode(texts[d], add_special_tokens=False).ids
                n_chk += 1
                if len(enc) != ln[i]:
                    mism_len += 1
                    if len(examples) < 5:
                        examples.append({"doc_id": d, "table": int(ln[i]), "reencoded": len(enc)})
                    continue
                stored = tok_at(np.arange(off[i], off[i] + ln[i]))
                if not np.array_equal(stored, np.asarray(enc, dtype=np.int64)):
                    mism_ids += 1
            sp["sample"] = {
                "checked": n_chk,
                "missing": missing,
                "count_mismatches": mism_len,
                "id_mismatches": mism_ids,
                "examples": examples,
            }
            sp["ok"] = (
                sp["arithmetic_ok"]
                and sp["doc_set_equal_accepted"]
                and sp["doc_ids_unique"]
                and sp["by_source_equal"]
                and contiguous
                and eos_ok
                and files_sha_ok
                and order_ok
                and missing == 0
                and mism_len == 0
                and mism_ids == 0
            )
            ok &= sp["ok"]
            c["splits"][split] = sp
        rec["candidates"][name] = c
    rec["ok"] = bool(ok)
    rec["seconds"] = round(time.time() - t0, 1)
    a.out.write_text(json.dumps(rec, indent=1))
    print(json.dumps({"ok": rec["ok"], "sample_docs": rec["sample_docs"], "seconds": rec["seconds"]}))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
