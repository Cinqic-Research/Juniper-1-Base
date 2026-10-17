"""Full independent recount of a model-ready token-shard set against its accepted text.

Written separately from ``j1data.tokshards`` (the producer) and from ``tools/independent_count_check.py``
(the earlier sampled check). It shares no code with either; it reads only the accepted parquet files, the
frozen ``tokenizer.json`` bytes and the shard directory.

Phase A (per accepted parquet file; resumable, one durable result file per input file):

* every document's text is re-hashed (SHA-256 and UTF-8 byte count against the accepted columns);
* every document is encoded **one document per call** (``Tokenizer.encode``; the producer used
  ``encode_batch_fast`` in batches) with ``encode_special_tokens`` enabled, and decoded back
  (``decode(encode(text)) == text`` for every document);
* no control ID (< 64) may come from text; literal control spellings in the text are counted, and for
  those documents the encoding a plain ``Tokenizer.from_str`` load would produce (the flag is not
  persisted in ``tokenizer.json``) is recorded so the hazard is measured, not assumed;
* a keyed-hash sample is also encoded by a separate pure-Python BPE implementation (regex split,
  character symbols, byte fallback, rank-ordered merges) built from the tokenizer JSON.

Phase B (per split): the shard index must hold exactly the accepted document IDs, in ascending
``unit_interval("j1.pack-order.v1", doc_id)`` order (recomputed here), at contiguous offsets with one
EOS (ID 0) after every document; each document's stored IDs must hash to the Phase-A hash; token-file
sizes and SHA-256 must match ``shards.json``; by-source and split totals are recomputed.

Output: ``<out>/recount.json`` (tables and pass/fail) plus per-file Phase-A records under ``<out>/files``.

Totals: ``phase_a.tokens`` is the sum of per-document token counts (**document tokens**, no separators).
The record's ``totals`` block names the three corpus-wide figures explicitly: ``document_tokens_total``
(= ``phase_a.tokens``), ``separator_tokens_total`` (one EOS per document) and ``stream_tokens_total``
(document tokens + separators = the token-file sizes / 2 summed over all splits).

Phase-A unit reuse (``--out`` shared between runs) is keyed by the input file's relative path and accepted
only when the unit's recorded ``file_sha256`` equals the current input file; otherwise the unit is recomputed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import defaultdict
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

N_CONTROL = 64
EOS = 0
ORDER_NS = "j1.pack-order.v1"
SAMPLE_NS = b"j1.full-recount.bpe-sample.v1\0"
VERSION = "j1.full-recount.v2"  # v2: adds the `totals` block (v1 records carry phase_a.tokens only)


def sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        while b := fh.read(1 << 24):
            h.update(b)
    return h.hexdigest()


def order_key(doc_id: str) -> float:
    d = hashlib.sha256(f"{ORDER_NS}\x00{doc_id}".encode()).digest()
    return (int.from_bytes(d[:8], "big") >> 11) / float(1 << 53)


def sampled(doc_id: str, rate: float) -> bool:
    return int.from_bytes(hashlib.sha256(SAMPLE_NS + doc_id.encode()).digest()[:8], "big") / 2**64 < rate


class PyBPE:
    """Independent encoder for the j1.tokenizer.v1 model family (char BPE + byte fallback)."""

    def __init__(self, tok_json: dict):
        import regex

        m = tok_json["model"]
        if (
            m["type"] != "BPE"
            or not m["byte_fallback"]
            or m.get("continuing_subword_prefix")
            or m.get("dropout")
        ):
            raise ValueError("unsupported model configuration")
        if tok_json.get("normalizer") is not None:
            raise ValueError("unexpected normalizer")
        pt = tok_json["pre_tokenizer"]
        if pt["type"] != "Split" or pt["behavior"] != "Isolated" or pt["invert"]:
            raise ValueError("unsupported pre-tokenizer")
        self.rx = regex.compile(pt["pattern"]["Regex"])
        self.vocab: dict[str, int] = m["vocab"]
        self.rank: dict[tuple[str, str], int] = {}
        for i, mg in enumerate(m["merges"]):
            a, b = mg if isinstance(mg, list) else mg.split(" ")
            self.rank.setdefault((a, b), i)
        self.byte_id = [self.vocab[f"<0x{b:02X}>"] for b in range(256)]

    def _pieces(self, text: str) -> list[str]:
        # "Isolated" split: matches become pieces and the gaps between matches become pieces too.
        out, pos = [], 0
        for mt in self.rx.finditer(text):
            if mt.start() > pos:
                out.append(text[pos : mt.start()])
            if mt.end() > mt.start():
                out.append(mt.group())
            pos = mt.end()
        if pos < len(text):
            out.append(text[pos:])
        return out

    def _word(self, piece: str) -> list[int]:
        syms: list[str | None] = []  # None marks a byte-fallback run that cannot merge
        fallback: dict[int, list[int]] = {}
        for ch in piece:
            if ch in self.vocab:
                syms.append(ch)
            else:
                fallback[len(syms)] = [self.byte_id[b] for b in ch.encode()]
                syms.append(None)
        while True:
            best, at = None, -1
            for i in range(len(syms) - 1):
                a, b = syms[i], syms[i + 1]
                if a is None or b is None:
                    continue
                r = self.rank.get((a, b))
                if r is not None and (best is None or r < best):
                    best, at = r, i
            if best is None:
                break
            merged = syms[at] + syms[at + 1]  # type: ignore[operator]
            if merged not in self.vocab:
                raise ValueError(f"merge result missing from vocab: {merged!r}")
            # Merge every occurrence of this pair left to right (same result as one-at-a-time by rank).
            new: list[str | None] = []
            new_fb: dict[int, list[int]] = {}
            i = 0
            while i < len(syms):
                if (
                    i < len(syms) - 1
                    and syms[i] == syms[at]
                    and syms[i + 1] == syms[at + 1]
                    and syms[i] is not None
                ):
                    new.append(merged)
                    i += 2
                else:
                    if syms[i] is None:
                        new_fb[len(new)] = fallback[i]
                    new.append(syms[i])
                    i += 1
            syms, fallback = new, new_fb
        ids: list[int] = []
        for i, s in enumerate(syms):
            ids.extend(fallback[i] if s is None else [self.vocab[s]])
        return ids

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        for p in self._pieces(text):
            ids.extend(self._word(p))
        return ids


_W: dict = {}


def _init(tok_path: str, rate: float) -> None:
    os.environ.setdefault("RAYON_NUM_THREADS", "1")
    from tokenizers import Tokenizer

    raw = Path(tok_path).read_text()
    t = Tokenizer.from_str(raw)
    t.encode_special_tokens = True
    plain = Tokenizer.from_str(raw)  # flag deliberately NOT set: measures the documented hazard
    tj = json.loads(raw)
    controls = [a["content"] for a in tj["added_tokens"] if a["id"] < N_CONTROL]
    _W.update(tok=t, plain=plain, controls=controls, bpe=PyBPE(tj), rate=rate)


def phase_a_file(job: tuple[str, str, str, str]) -> dict:
    src, rel, split, out = job
    dst = Path(out)
    sp_ = dst.with_suffix(".summary.json")
    if dst.exists():
        # Reuse a completed unit only when it was computed from byte-identical input: the unit key is
        # the relative path, which a successor rewrite keeps while the file content changes.
        try:
            cached = json.loads(sp_.read_text())
        except (OSError, json.JSONDecodeError):
            cached = None
        if isinstance(cached, dict) and cached.get("file_sha256") == sha_file(Path(src)):
            return cached
        dst.unlink()
        sp_.unlink(missing_ok=True)
    t0 = time.time()
    tok, plain, controls, bpe, rate = (_W[k] for k in ("tok", "plain", "controls", "bpe", "rate"))
    tab = pq.read_table(src, columns=["doc_id", "source_id", "text", "sha256", "n_bytes", "split"])
    ids_c, src_c, n_c, sha_c = [], [], [], []
    errors: list[str] = []
    lit_docs = lit_plain_ctrl = bpe_checked = bpe_mismatch = 0
    for did, sid, text, sha, nb, sp in zip(
        *(tab.column(c).to_pylist() for c in tab.column_names), strict=True
    ):
        b = text.encode()
        if hashlib.sha256(b).hexdigest() != sha or len(b) != nb:
            errors.append(f"{did}: text hash/bytes mismatch")
        if sp != split:
            errors.append(f"{did}: split column {sp} != directory split {split}")
        enc = tok.encode(text, add_special_tokens=False).ids
        arr = np.asarray(enc, dtype=np.uint16)
        if arr.size and int(arr.min()) < N_CONTROL:
            errors.append(f"{did}: control ID from text")
        if tok.decode(enc, skip_special_tokens=False) != text:
            errors.append(f"{did}: decode(encode(text)) != text")
        if any(c in text for c in controls):
            lit_docs += 1
            p = plain.encode(text, add_special_tokens=False).ids
            if p and min(p) < N_CONTROL:
                lit_plain_ctrl += 1
        if sampled(did, rate):
            bpe_checked += 1
            if bpe.encode(text) != enc:
                bpe_mismatch += 1
                errors.append(f"{did}: pure-Python BPE disagrees with tokenizers")
        ids_c.append(did)
        src_c.append(sid)
        n_c.append(arr.size)
        sha_c.append(hashlib.sha256(arr.astype("<u2").tobytes()).hexdigest())
    rec = pa.table(
        {"doc_id": ids_c, "source_id": src_c, "n_tokens": pa.array(n_c, pa.int64()), "ids_sha256": sha_c}
    )
    tmp = dst.with_suffix(".partial")
    pq.write_table(rec, tmp)
    summary = {
        "file": rel,
        "file_sha256": sha_file(Path(src)),
        "split": split,
        "documents": len(ids_c),
        "tokens": int(sum(n_c)),
        "literal_control_docs": lit_docs,
        "literal_control_docs_plain_load_gives_control_ids": lit_plain_ctrl,
        "bpe_sample_checked": bpe_checked,
        "bpe_sample_mismatch": bpe_mismatch,
        "errors": errors[:50],
        "n_errors": len(errors),
        "seconds": round(time.time() - t0, 2),
    }
    sp_.with_suffix(".tmp").write_text(json.dumps(summary))
    sp_.with_suffix(".tmp").replace(sp_)
    tmp.replace(dst)  # result file last: its presence marks the unit complete
    return summary


def phase_b(split: str, shard_dir: Path, meta: dict, recs: pa.Table) -> dict:
    errors: list[str] = []
    want = {
        d: (n, h, s)
        for d, n, h, s in zip(
            *(recs.column(c).to_pylist() for c in ("doc_id", "n_tokens", "ids_sha256", "source_id")),
            strict=True,
        )
    }
    if len(want) != recs.num_rows:
        errors.append("duplicate doc_id in accepted text")
    sm = meta["splits"][split]
    idx = pq.read_table(shard_dir / sm["doc_index"]["file"])
    if sha_file(shard_dir / sm["doc_index"]["file"]) != sm["doc_index"]["sha256"]:
        errors.append("doc index sha256 differs from shards.json")
    files = sm["token_files"]
    for f in files:
        p = shard_dir / f["file"]
        if p.stat().st_size != 2 * f["tokens"]:
            errors.append(f"{f['file']}: size != 2*tokens")
        if sha_file(p) != f["sha256"]:
            errors.append(f"{f['file']}: sha256 differs from shards.json")
    maps = [np.memmap(shard_dir / f["file"], dtype="<u2", mode="r") for f in files]
    bounds = np.cumsum([0] + [m.size for m in maps])
    stream_len = int(bounds[-1])

    def take(a: int, n: int) -> np.ndarray:
        out, pos = [], a
        while n > 0:
            k = int(np.searchsorted(bounds, pos, side="right") - 1)
            off = pos - int(bounds[k])
            m = min(n, maps[k].size - off)
            out.append(np.asarray(maps[k][off : off + m]))
            pos, n = pos + m, n - m
        return np.concatenate(out) if out else np.zeros(0, "<u2")

    seen = set()
    expect_off, prev_key = 0, (-1.0, "")
    by_src: dict[str, dict[str, int]] = defaultdict(lambda: {"documents": 0, "tokens": 0})
    for did, sid, off, n in zip(
        *(idx.column(c).to_pylist() for c in ("doc_id", "source_id", "offset", "n_tokens")), strict=True
    ):
        if did in seen:
            errors.append(f"{did}: duplicated in shard index")
        seen.add(did)
        k = (order_key(did), did)
        if k <= prev_key:
            errors.append(f"{did}: pack order violated")
        prev_key = k
        if off != expect_off:
            errors.append(f"{did}: offset {off} != expected {expect_off}")
        w = want.get(did)
        if w is None:
            errors.append(f"{did}: in shards but not in accepted text")
        else:
            if n != w[0]:
                errors.append(f"{did}: n_tokens {n} != recount {w[0]}")
            if w[2] != sid:
                errors.append(f"{did}: source {sid} != accepted {w[2]}")
            seg = take(off, n + 1)
            if seg.size != n + 1 or int(seg[-1]) != EOS:
                errors.append(f"{did}: no EOS after document")
            if hashlib.sha256(seg[:n].astype("<u2").tobytes()).hexdigest() != w[1]:
                errors.append(f"{did}: stored IDs differ from recount")
            if n and int(seg[:n].min()) < N_CONTROL:
                errors.append(f"{did}: control ID inside document")
        by_src[sid]["documents"] += 1
        by_src[sid]["tokens"] += n
        expect_off = off + n + 1
        if len(errors) > 200:
            errors.append("... stopped after 200 errors")
            break
    missing = set(want) - seen
    if missing:
        errors.append(f"{len(missing)} accepted documents absent from shards")
    if expect_off != stream_len:
        errors.append(f"stream length {stream_len} != indexed end {expect_off}")
    doc_tokens = sum(v["tokens"] for v in by_src.values())
    for key, val in (
        ("documents", len(seen)),
        ("document_tokens", doc_tokens),
        ("stream_tokens", stream_len),
        ("separator_tokens", len(seen)),
    ):
        if sm.get(key) != val:
            errors.append(f"shards.json {key}={sm.get(key)} != recount {val}")
    if dict(sorted(by_src.items())) != sm.get("by_source"):
        errors.append("shards.json by_source differs from recount")
    return {
        "documents": len(seen),
        "document_tokens": doc_tokens,
        "separator_tokens": len(seen),
        "stream_tokens": stream_len,
        "by_source": dict(sorted(by_src.items())),
        "errors": errors,
        "pass": not errors,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("accepted", type=Path)
    ap.add_argument("shards", type=Path)
    ap.add_argument("tokenizer", type=Path)
    ap.add_argument("--expect-tokenizer-sha256", required=True)
    ap.add_argument("--expect-manifest-sha256", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--bpe-sample-rate", type=float, default=0.0005)
    a = ap.parse_args()
    t0 = time.time()
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "files").mkdir(exist_ok=True)
    pre = []
    if sha_file(a.tokenizer) != a.expect_tokenizer_sha256:
        pre.append("tokenizer sha256 mismatch")
    if sha_file(a.accepted / "manifest.json") != a.expect_manifest_sha256:
        pre.append("accepted manifest sha256 mismatch")
    meta = json.loads((a.shards / "shards.json").read_text())
    if meta.get("accepted_manifest_sha256") != a.expect_manifest_sha256:
        pre.append("shards.json is bound to a different accepted manifest")
    if meta.get("tokenizer", {}).get("tokenizer_json_sha256") != a.expect_tokenizer_sha256:
        pre.append("shards.json is bound to a different tokenizer")
    if pre:
        print(json.dumps({"preconditions": pre}))
        return 2
    man = json.loads((a.accepted / "manifest.json").read_text())
    jobs = []
    for sh in man["shards"]:
        rel = sh["file"]
        key = hashlib.sha256(rel.encode()).hexdigest()[:20]
        jobs.append((str(a.accepted / rel), rel, sh["split"], str(a.out / "files" / f"{key}.parquet")))
    on_disk = {str(p.relative_to(a.accepted)) for p in a.accepted.rglob("*.parquet")}
    listed = {j[1] for j in jobs}
    inv_err = sorted(on_disk ^ listed)[:20]
    jobs.sort(key=lambda j: -Path(j[0]).stat().st_size)
    summaries = []
    log = (a.out / "progress.log").open("a")
    with get_context("spawn").Pool(
        a.workers, initializer=_init, initargs=(str(a.tokenizer), a.bpe_sample_rate)
    ) as pool:
        for i, s in enumerate(pool.imap_unordered(phase_a_file, jobs), 1):
            summaries.append(s)
            log.write(
                json.dumps(
                    {
                        "t": round(time.time() - t0, 1),
                        "done": i,
                        "of": len(jobs),
                        "file": s["file"],
                        "errors": s["n_errors"],
                    }
                )
                + "\n"
            )
            log.flush()
            os.fsync(log.fileno())
    a_err = [e for s in summaries for e in s["errors"]]
    result: dict = {
        "schema": VERSION,
        "inputs": {
            "accepted_manifest_sha256": a.expect_manifest_sha256,
            "tokenizer_json_sha256": a.expect_tokenizer_sha256,
            "shards_json_sha256": sha_file(a.shards / "shards.json"),
            "build_identity": man.get("build_identity"),
        },
        "phase_a": {
            "files": len(summaries),
            "inventory_mismatch": inv_err,
            "documents": sum(s["documents"] for s in summaries),
            "tokens": sum(s["tokens"] for s in summaries),
            "errors": a_err[:100],
            "n_errors": sum(s["n_errors"] for s in summaries),
            "literal_control_docs": sum(s["literal_control_docs"] for s in summaries),
            "literal_control_docs_plain_load_gives_control_ids": sum(
                s["literal_control_docs_plain_load_gives_control_ids"] for s in summaries
            ),
            "bpe_sample_checked": sum(s["bpe_sample_checked"] for s in summaries),
            "bpe_sample_mismatch": sum(s["bpe_sample_mismatch"] for s in summaries),
        },
        "splits": {},
    }
    for split in ("train", "validation", "test"):
        parts = [pq.read_table(j[3]) for j in jobs if j[2] == split]
        recs = (
            pa.concat_tables(parts)
            if parts
            else pa.table({"doc_id": [], "n_tokens": [], "ids_sha256": [], "source_id": []})
        )
        result["splits"][split] = phase_b(split, a.shards, meta, recs)
        if man["accepted_by_split"].get(split) != result["splits"][split]["documents"]:
            result["splits"][split]["errors"].append("document count differs from accepted manifest")
            result["splits"][split]["pass"] = False
    doc_total = sum(s["document_tokens"] for s in result["splits"].values())
    sep_total = sum(s["separator_tokens"] for s in result["splits"].values())
    result["totals"] = {
        "document_tokens_total": doc_total,
        "separator_tokens_total": sep_total,
        "stream_tokens_total": sum(s["stream_tokens"] for s in result["splits"].values()),
        "note": "phase_a.tokens counts document tokens only; stream tokens include one EOS per document",
    }
    result["totals"]["consistent_with_phase_a"] = (
        doc_total == result["phase_a"]["tokens"] and sep_total == result["phase_a"]["documents"]
    )
    result["pass"] = (
        result["totals"]["consistent_with_phase_a"]
        and not inv_err
        and result["phase_a"]["n_errors"] == 0
        and all(s["pass"] for s in result["splits"].values())
    )
    result["seconds"] = round(time.time() - t0, 1)
    out = a.out / "recount.json"
    out.with_suffix(".tmp").write_text(json.dumps(result, indent=1, sort_keys=True))
    out.with_suffix(".tmp").replace(out)
    print(json.dumps({"pass": result["pass"], "seconds": result["seconds"]}))
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
