"""Comparison inputs for a successor corpus without rebuilding its full shards.

Writes a ``j1.token-shards.v1`` directory holding (a) the keyed-order train prefix of whole documents of the
source shards with the excluded IDs skipped (same byte-budget rule as ``tokshards.export_prefix_excluding``)
and (b) the validation split with the excluded IDs removed. Kept documents' tokens and EOS are copied
unchanged in their stored order, so the result is byte-identical to the corresponding part of a full rebuild
of the successor (per-document tokenization is context-free and the order is the keyed order);
``--check-against`` verifies that claim later against official successor shards. The test split is not
exported.

Usage: successor_prefix.py SRC_SHARDS ACCEPTED_TRAIN_DIR DST --max-bytes N --exclude IDS.json
       [--check-against OFFICIAL_SUCCESSOR_SHARDS [--record OUT.json]]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from j1data.shards import write_json_atomic
from j1data.tokshards import EOS, _StreamWriter


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        while b := f.read(1 << 24):
            h.update(b)
    return h.hexdigest()


def _take_fn(src: Path, files: list[dict]):
    maps = [np.memmap(src / f["file"], dtype="<u2", mode="r") for f in files]
    starts = np.cumsum([0] + [len(m) for m in maps])

    def take(a: int, n: int) -> np.ndarray:
        out = np.empty(n, dtype="<u2")
        k = 0
        while k < n:
            fi = int(np.searchsorted(starts, a + k, side="right")) - 1
            o = a + k - int(starts[fi])
            t = min(n - k, len(maps[fi]) - o)
            out[k : k + t] = maps[fi][o : o + t]
            k += t
        return out

    return take


def _subset(src: Path, dst: Path, split: str, rec: dict, keep_rows: list[int], idx: pa.Table) -> dict:
    take = _take_fn(src, rec["token_files"])
    offs = idx.column("offset").to_numpy()
    lens = idx.column("n_tokens").to_numpy()
    ids = idx.column("doc_id").to_pylist()
    writer = _StreamWriter(dst, split)
    new_off, pos = [], 0
    for i in keep_rows:
        chunk = take(int(offs[i]), int(lens[i]) + 1)
        if chunk[-1] != EOS:
            raise SystemExit(f"document {ids[i]} is not followed by EOS in {src}")
        writer.write(chunk)
        new_off.append(pos)
        pos += chunk.size
    tfiles = writer.close()
    sub = idx.take(pa.array(keep_rows, pa.int64())).set_column(
        idx.column_names.index("offset"), "offset", pa.array(new_off, pa.int64())
    )
    name = f"{split}-docs.parquet"
    pq.write_table(sub, dst / name, compression="zstd")
    by_source: dict[str, dict[str, int]] = defaultdict(lambda: {"documents": 0, "tokens": 0})
    for s_id, n in zip(sub.column("source_id").to_pylist(), sub.column("n_tokens").to_pylist(), strict=True):
        by_source[s_id]["documents"] += 1
        by_source[s_id]["tokens"] += n
    k = len(keep_rows)
    return {
        "split": split,
        "documents": k,
        "document_tokens": pos - k,
        "separator_tokens": k,
        "stream_tokens": pos,
        "by_source": dict(sorted(by_source.items())),
        "token_files": tfiles,
        "doc_index": {"file": name, "sha256": sha(dst / name), "rows": k},
    }


def export(src: Path, accepted_train: Path, dst: Path, max_bytes: int, exclude: frozenset[str]) -> dict:
    if dst.exists():
        raise SystemExit(f"output exists (immutable): {dst}")
    rec = json.loads((src / "shards.json").read_text())
    doc_bytes: dict[str, int] = {}
    for f in sorted(accepted_train.rglob("*.parquet")):
        t = pq.read_table(f, columns=["doc_id", "n_bytes"])
        doc_bytes.update(zip(t.column("doc_id").to_pylist(), t.column("n_bytes").to_pylist(), strict=True))
    partial = dst.with_name(dst.name + ".partial")
    partial.mkdir(parents=True)
    tr_idx = pq.read_table(src / rec["splits"]["train"]["doc_index"]["file"])
    keep, total, skipped = [], 0, []
    for i, d in enumerate(tr_idx.column("doc_id").to_pylist()):
        if d in exclude:
            skipped.append(d)
            continue
        if total + doc_bytes[d] > max_bytes:
            break
        total += doc_bytes[d]
        keep.append(i)
    train = _subset(src, partial, "train", rec["splits"]["train"], keep, tr_idx)
    va_idx = pq.read_table(src / rec["splits"]["validation"]["doc_index"]["file"])
    va_ids = va_idx.column("doc_id").to_pylist()
    va_keep = [i for i, d in enumerate(va_ids) if d not in exclude]
    validation = _subset(src, partial, "validation", rec["splits"]["validation"], va_keep, va_idx)
    out = {
        **{k: v for k, v in rec.items() if k != "splits"},
        "successor_prefix_of": {
            "shards_json_sha256": sha(src / "shards.json"),
            "train_documents": len(keep),
            "train_text_bytes": total,
            "train_excluded_in_range": sorted(skipped),
            "validation_excluded": sorted(set(va_ids) & exclude),
            "exclusion_list_sha256": hashlib.sha256(json.dumps(sorted(exclude)).encode()).hexdigest(),
            "max_bytes": max_bytes,
        },
        "splits": {"train": train, "validation": validation},
    }
    write_json_atomic(partial / "shards.json", out)
    partial.rename(dst)
    return out


CHECKS = ("validation_token_files_equal", "validation_doc_index_equal", "train_prefix_equals_official_head")


def check_against(prefix: Path, official: Path) -> dict:
    """Validation must be byte-identical; the train prefix must equal the official train stream's head.

    The record names both inputs (directory names and ``shards.json`` SHA-256s) and the compared token
    count, so it can be re-matched to its inputs later.
    """
    p, o = (json.loads((x / "shards.json").read_text()) for x in (prefix, official))
    n = int(p["splits"]["train"]["stream_tokens"])
    res: dict = {
        "schema": "j1.successor-prefix-check.v2",
        "prefix": prefix.resolve().name,
        "prefix_shards_json_sha256": sha(prefix / "shards.json"),
        "official": official.resolve().name,
        "official_shards_json_sha256": sha(official / "shards.json"),
        "compared_train_tokens": n,
        "prefix_train_documents": int(p["splits"]["train"]["documents"]),
        "official_train_stream_tokens": int(o["splits"]["train"]["stream_tokens"]),
        "validation_token_files_equal": [f["sha256"] for f in p["splits"]["validation"]["token_files"]]
        == [f["sha256"] for f in o["splits"]["validation"]["token_files"]],
        "validation_doc_index_equal": p["splits"]["validation"]["doc_index"]["sha256"]
        == o["splits"]["validation"]["doc_index"]["sha256"],
    }
    pt = np.concatenate(
        [np.memmap(prefix / f["file"], dtype="<u2", mode="r") for f in p["splits"]["train"]["token_files"]]
    )
    ot = _take_fn(official, o["splits"]["train"]["token_files"])(
        0, min(n, res["official_train_stream_tokens"])
    )
    res["train_prefix_equals_official_head"] = bool(pt.size == n and ot.size == n and np.array_equal(pt, ot))
    res["ok"] = all(res[k] for k in CHECKS)
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path)
    ap.add_argument("accepted_train", type=Path)
    ap.add_argument("dst", type=Path)
    ap.add_argument("--max-bytes", type=int)
    ap.add_argument("--exclude", type=Path)
    ap.add_argument("--check-against", type=Path)
    ap.add_argument("--record", type=Path, help="with --check-against: also write the record here")
    a = ap.parse_args(argv)
    if a.check_against:
        res = check_against(a.dst, a.check_against)
        if a.record:
            a.record.write_text(json.dumps(res, indent=1) + "\n")
        print(json.dumps(res))
        return 0 if res["ok"] else 1
    out = export(a.src, a.accepted_train, a.dst, a.max_bytes, frozenset(json.loads(a.exclude.read_text())))
    print(
        json.dumps({s: {k: v[k] for k in ("documents", "stream_tokens")} for s, v in out["splits"].items()})
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
