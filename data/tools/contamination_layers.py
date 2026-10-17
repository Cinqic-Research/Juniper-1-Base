"""Strengthened contamination layers (``j1.contamination-layers.v1``) on top of ``j1.contamination.v3``.

The v3 matcher indexes *informative* word 13-grams, so short, numeric, structured and code items are
largely unmatchable (internal suites 2.1-39.3 % matchable). Two additional layers:

* **Layer A, all-token 10-grams.** Text is NFKC-normalized and case-folded; tokens are runs of
  letters/digits and every single punctuation character (``[^\\W_]+|[^\\w\\s]|_``), so code, JSON,
  numbers and formulas keep their structure while whitespace/indentation/case differences vanish.
  Every protected item with at least 10 tokens is indexed by its distinct 10-gram hashes. A
  document *matches* an item when it contains at least ``--min-coverage`` (pre-registered 0.6) of
  the item's distinct 10-grams. Grams that occur in at least ``--common-df`` documents of the scanned
  set are reported as common and excluded from coverage in the calibrated pass (two-pass rule).
* **Layer C, distinctive identifiers.** Tokens of >= 16 characters from ``[0-9a-z_-]`` containing
  at least one digit and one letter (hashes, UUIDs, run IDs) from protected items; any occurrence
  in a document matches.

Restart safety: ``scan`` writes one result file per accepted shard, named by the shard's SHA-256 and
the index identity, published atomically; an existing result is reused only if its recorded input
and index identities match. Inputs are re-hashed on every run.

Subcommands: ``build`` (items -> index), ``scan`` (documents -> hits), ``controls`` (planted leaks and
transformed copies -> recall; random corpus text -> false-positive rate).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import time
import unicodedata
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from j1data.ngrams import ngram_hashes, word_hashes

VERSION = "j1.contamination-layers.v1"
TOOL_SHA = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
N = 10
TOKEN_RE = re.compile(r"[^\W_]+|[^\w\s]|_")
IDENT_RE = re.compile(r"(?<![0-9a-z_-])(?=[0-9a-z_-]*\d)(?=[0-9a-z_-]*[a-z])[0-9a-z_-]{16,}(?![0-9a-z_-])")


def normalize(text: str) -> str:
    return unicodedata.normalize("NFKC", text).casefold()


MIN_ALNUM = 3  # a 10-gram is indexed only with >= 3 letter/digit tokens (no pure punctuation runs)
COMMON_DF = 10  # pre-registered: grams in >= 10 corpus documents are common (excluded in the calibrated pass)


def grams(text: str) -> np.ndarray:
    toks = TOKEN_RE.findall(normalize(text))
    if len(toks) < N:
        return np.empty(0, dtype=np.uint64)
    h = ngram_hashes(word_hashes(toks), N)
    alnum = np.fromiter((t[0].isalnum() for t in toks), dtype=np.int64, count=len(toks))
    c = np.concatenate([[0], np.cumsum(alnum)])
    keep = (c[N:] - c[:-N]) >= MIN_ALNUM
    return np.unique(h[keep])


def idents(text: str) -> set[str]:
    return set(IDENT_RE.findall(normalize(text)))


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        while b := f.read(1 << 24):
            h.update(b)
    return h.hexdigest()


# --- build ------------------------------------------------------------------------------------


def build(items: dict[str, list[str]], out: Path, source_fingerprint: str) -> dict:
    gram_list, item_list, item_bench, item_ngrams = [], [], [], []
    ident_map: dict[str, set[int]] = {}
    report = {}
    iid = 0
    for bench in sorted(items):
        n_items = indexed = idented = 0
        for text in items[bench]:
            n_items += 1
            g = grams(text)
            ids = idents(text)
            if len(g) or ids:
                if len(g):
                    gram_list.append(g)
                    item_list.append(np.full(len(g), iid, dtype=np.int64))
                    indexed += 1
                for x in ids:
                    ident_map.setdefault(x, set()).add(iid)
                idented += bool(ids)
                item_bench.append(bench)
                item_ngrams.append(len(g))
                iid += 1
        report[bench] = {"items": n_items, "layer_a_items": indexed, "layer_c_items": idented}
    G = np.concatenate(gram_list) if gram_list else np.empty(0, dtype=np.uint64)
    I = np.concatenate(item_list) if item_list else np.empty(0, dtype=np.int64)
    order = np.argsort(G, kind="stable")
    G, I = G[order], I[order]
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "grams.npy", G)
    np.save(out / "gram_items.npy", I)
    np.save(out / "item_ngrams.npy", np.array(item_ngrams, dtype=np.int64))
    (out / "item_bench.json").write_text(json.dumps(item_bench))
    (out / "idents.json").write_text(json.dumps({k: sorted(v) for k, v in sorted(ident_map.items())}))
    meta = {
        "version": VERSION,
        "n": N,
        "min_alnum_tokens_per_gram": MIN_ALNUM,
        "common_df_threshold": COMMON_DF,
        "token_regex": TOKEN_RE.pattern,
        "ident_regex": IDENT_RE.pattern,
        "normalization": "NFKC + casefold",
        "source_fingerprint": source_fingerprint,
        "items": iid,
        "gram_rows": len(G),
        "distinct_grams": len(np.unique(G)),
        "identifiers": len(ident_map),
        "benchmarks": report,
    }
    files = ["grams.npy", "gram_items.npy", "item_ngrams.npy", "item_bench.json", "idents.json"]
    meta["files"] = {f: sha256_file(out / f) for f in files}
    meta["index_identity"] = hashlib.sha256(json.dumps(meta["files"], sort_keys=True).encode()).hexdigest()
    (out / "index.json").write_text(json.dumps(meta, indent=1))
    return meta


class Index:
    def __init__(self, d: Path):
        self.meta = json.loads((d / "index.json").read_text())
        for f, h in self.meta["files"].items():
            if sha256_file(d / f) != h:
                raise SystemExit(f"index file {f} hash mismatch")
        self.G = np.load(d / "grams.npy")
        self.I = np.load(d / "gram_items.npy")
        self.n_item = np.load(d / "item_ngrams.npy")
        self.bench = json.loads((d / "item_bench.json").read_text())
        self.idents = {k: v for k, v in json.loads((d / "idents.json").read_text()).items()}
        self.common: set[int] = set()

    def _common_counts(self, common: np.ndarray) -> np.ndarray:
        """Per item: how many of its distinct grams are common. Cached per common array; the cache holds a
        reference to that array, so its identity cannot be reused by a different array (review N4)."""
        if getattr(self, "_cc_obj", None) is not common:
            mask = np.isin(self.G, common)
            self._cc = np.bincount(self.I[mask], minlength=len(self.n_item))
            self._cc_obj = common
        return self._cc

    def match(self, text: str, min_cov: float, common: np.ndarray | None = None) -> dict | None:
        """Coverage is over each item's NON-common grams (review finding F3). Items whose grams are all
        common are reported separately as ``common_only`` when the document holds >= min_cov of them,
        so a widely copied item can never hide itself."""
        g = grams(text)
        hit_items: dict[int, float] = {}
        common_only: dict[int, float] = {}
        if len(g) and len(self.G):
            lo = np.searchsorted(self.G, g, side="left")
            hi = np.searchsorted(self.G, g, side="right")
            m = hi > lo
            if m.any():
                gg, lo, hi = g[m], lo[m], hi[m]
                use_common = common is not None and len(common) > 0
                is_common = np.isin(gg, common) if use_common else np.zeros(len(gg), dtype=bool)
                items_all = np.concatenate([self.I[a:b] for a, b in zip(lo, hi, strict=True)])
                rep = hi - lo
                cnt_all = np.bincount(items_all, minlength=len(self.n_item))
                cnt_common = np.bincount(items_all[np.repeat(is_common, rep)], minlength=len(self.n_item))
                n_common = self._common_counts(common) if use_common else np.zeros_like(cnt_all)
                n_eff = self.n_item - n_common
                for i in np.nonzero(cnt_all)[0].tolist():
                    if n_eff[i] > 0:
                        c = (cnt_all[i] - cnt_common[i]) / n_eff[i]
                        if c >= min_cov:
                            hit_items[i] = c
                    elif self.n_item[i] > 0:
                        c = cnt_all[i] / self.n_item[i]
                        if c >= min_cov:
                            common_only[i] = c
        ids = idents(text) & self.idents.keys() if self.idents else set()
        if not hit_items and not ids and not common_only:
            return None
        return {
            "layer_a": {str(i): round(c, 4) for i, c in sorted(hit_items.items())},
            "layer_a_benchmarks": sorted({self.bench[i] for i in hit_items}),
            "layer_a_common_only": {str(i): round(c, 4) for i, c in sorted(common_only.items())},
            "layer_c": sorted(ids),
            "layer_c_benchmarks": sorted({self.bench[i] for x in ids for i in self.idents[x]}),
        }

    def doc_grams_in_index(self, text: str) -> np.ndarray:
        g = grams(text)
        if not len(g):
            return g
        lo = np.searchsorted(self.G, g, side="left")
        return g[np.searchsorted(self.G, g, side="right") > lo]


# --- scan -------------------------------------------------------------------------------------

_IDX: Index | None = None


def _init(index_dir: str) -> None:
    global _IDX
    _IDX = Index(Path(index_dir))


def _scan_unit(task: tuple[str, str, float, list[str] | None, str | None, str]) -> dict:
    path, out_dir, min_cov, only_ids, common_path, expected_sha = task
    assert _IDX is not None
    p = Path(path)
    shard_sha = sha256_file(p)
    if shard_sha != expected_sha:
        raise RuntimeError(f"{p}: SHA-256 {shard_sha} != accepted manifest {expected_sha}")
    out = Path(out_dir) / f"{shard_sha[:32]}-{_IDX.meta['index_identity'][:16]}.json"
    common = np.load(common_path) if common_path else None
    if out.is_file():
        r = json.loads(out.read_text())
        if (
            r.get("tool_sha256") == TOOL_SHA
            and r.get("min_coverage") == min_cov
            and r.get("shard_sha256") == shard_sha
            and r.get("index_identity") == _IDX.meta["index_identity"]
            and r.get("common_sha256") == (sha256_file(Path(common_path)) if common_path else None)
            and r.get("only_ids_sha256")
            == (
                hashlib.sha256("\n".join(sorted(only_ids)).encode()).hexdigest()
                if only_ids is not None
                else None
            )
        ):
            r["reused"] = True
            return r
    want = set(only_ids) if only_ids is not None else None
    t0 = time.time()
    hits, docs, df = [], 0, {}
    for batch in pq.ParquetFile(p).iter_batches(columns=["doc_id", "text"], batch_size=512):
        for d, text in zip(batch.column("doc_id").to_pylist(), batch.column("text").to_pylist(), strict=True):
            if want is not None and d not in want:
                continue
            docs += 1
            for x in _IDX.doc_grams_in_index(text).tolist():
                df[x] = df.get(x, 0) + 1
            m = _IDX.match(text, min_cov, common)
            if m:
                hits.append({"doc_id": d, **m})
    r = {
        "version": VERSION,
        "shard": str(p),
        "shard_sha256": shard_sha,
        "tool_sha256": TOOL_SHA,
        "index_identity": _IDX.meta["index_identity"],
        "common_sha256": sha256_file(Path(common_path)) if common_path else None,
        "only_ids_sha256": hashlib.sha256("\n".join(sorted(only_ids)).encode()).hexdigest()
        if only_ids is not None
        else None,
        "min_coverage": min_cov,
        "documents": docs,
        "hits": hits,
        "indexed_gram_df": {str(k): v for k, v in df.items()},
        "seconds": round(time.time() - t0, 2),
    }
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(r))
    os.replace(tmp, out)
    return r


def scan(index_dir: Path, shards: list[tuple[Path, str]], out_dir: Path, workers: int, min_cov: float,
         only_ids: dict[str, list[str]] | None, common: Path | None) -> dict:  # fmt: skip
    out_dir.mkdir(parents=True, exist_ok=True)
    tasks = [
        (str(s), str(out_dir), min_cov, (only_ids or {}).get(str(s)) if only_ids is not None else None,
         str(common) if common else None, sha)
        for s, sha in shards
        if only_ids is None or only_ids.get(str(s))
    ]  # fmt: skip
    results = []
    with ProcessPoolExecutor(workers, initializer=_init, initargs=(str(index_dir),)) as ex:
        for r in ex.map(_scan_unit, tasks, chunksize=1):
            results.append(r)
    df: dict[str, int] = {}
    for r in results:
        for k, v in r["indexed_gram_df"].items():
            df[k] = df.get(k, 0) + v
    hits = [h for r in results for h in r["hits"]]
    return {
        "units": len(results),
        "units_reused": sum(1 for r in results if r.get("reused")),
        "documents": sum(r["documents"] for r in results),
        "hit_documents": len(hits),
        "hits": hits,
        "gram_df": df,
    }


# --- controls ---------------------------------------------------------------------------------


def _transform(text: str, kind: str, rng: random.Random) -> str:
    if kind == "verbatim":
        return text
    if kind == "case_space":
        return re.sub(r"[ \t]+", lambda _: " " * rng.randint(1, 3), text.upper())
    if kind == "reindent":
        return "\n".join(
            ("\t" if rng.random() < 0.5 else "        ") + ln.strip() for ln in text.splitlines()
        )
    if kind == "embedded":
        return (
            "Here is a related note from a forum post.\n\n"
            + text
            + "\n\nThanks for reading, see you next time."
        )
    raise ValueError(kind)


def controls(idx: Index, items: dict[str, list[str]], negatives: list[str], min_cov: float, per_bench: int,
             common: np.ndarray | None = None,
             seed: int = 7) -> dict:  # fmt: skip
    rng = random.Random(seed)
    out: dict = {"positive": {}, "negative": {}}
    for bench in sorted(items):
        pool = [t for t in items[bench] if len(TOKEN_RE.findall(normalize(t))) >= N or idents(t)]
        sample = rng.sample(pool, min(per_bench, len(pool)))
        res = {}
        for kind in ("verbatim", "case_space", "reindent", "embedded"):
            found = sum(1 for t in sample if (m := idx.match(_transform(t, kind, rng), min_cov, common)) and bench in (
                m["layer_a_benchmarks"] + m["layer_c_benchmarks"]))  # fmt: skip
            res[kind] = {"planted": len(sample), "detected": found}
        out["positive"][bench] = res
    flagged = [i for i, t in enumerate(negatives) if idx.match(t, min_cov, common)]
    out["negative"] = {
        "documents": len(negatives),
        "flagged": len(flagged),
        "flagged_rate": len(flagged) / max(1, len(negatives)),
    }
    return out


# --- CLI --------------------------------------------------------------------------------------


def _items(layout_root: Path, clones: Path, cache: Path) -> tuple[dict[str, list[str]], str]:
    if cache.is_file():
        d = json.loads(cache.read_text())
        return d["items"], d["source_fingerprint"]
    from j1data.exclusions import _collect
    from j1data.paths import DataLayout

    got = _collect(DataLayout(layout_root), json.loads(clones.read_text()))
    items = {k: v[0] for k, v in got.items()}
    sources = {k: v[1] for k, v in got.items()}
    fp = hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest()
    cache.write_text(json.dumps({"source_fingerprint": fp, "sources": sources, "items": items}))
    return items, fp


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--data-root", type=Path, required=True)
    b.add_argument("--clones", type=Path, required=True)
    b.add_argument("--items-cache", type=Path, required=True)
    b.add_argument("--out", type=Path, required=True)
    s = sub.add_parser("scan")
    s.add_argument("--index", type=Path, required=True)
    s.add_argument("--accepted", type=Path, required=True, help="accepted build dir (manifest.json + shards)")
    s.add_argument("--only-ids", type=Path, help="JSON list of doc_ids to scan (default: every document)")
    s.add_argument("--splits", default="train,validation,test")
    s.add_argument("--out-dir", type=Path, required=True)
    s.add_argument("--report", type=Path, required=True)
    s.add_argument("--workers", type=int, default=6)
    s.add_argument("--min-coverage", type=float, default=0.6)
    s.add_argument(
        "--common", type=Path, help="npy of common gram hashes excluded from coverage (calibrated pass)"
    )
    c = sub.add_parser("controls")
    c.add_argument("--index", type=Path, required=True)
    c.add_argument("--items-cache", type=Path, required=True)
    c.add_argument(
        "--negatives", type=Path, required=True, help="accepted split dir sampled as negative text"
    )
    c.add_argument("--per-bench", type=int, default=200)
    c.add_argument("--n-negative", type=int, default=3000)
    c.add_argument("--min-coverage", type=float, default=0.6)
    c.add_argument("--out", type=Path, required=True)
    c.add_argument("--common", type=Path, help="calibrated pass: common gram hashes excluded from coverage")
    a = ap.parse_args()
    if a.cmd == "build":
        items, fp = _items(a.data_root, a.clones, a.items_cache)
        meta = build(items, a.out, fp)
        print(json.dumps({k: meta[k] for k in ("index_identity", "items", "distinct_grams", "identifiers")}))
        return 0
    if a.cmd == "scan":
        man = json.loads((a.accepted / "manifest.json").read_text())
        splits = set(a.splits.split(","))
        shards = [(a.accepted / e["file"], e["sha256"]) for e in man["shards"] if e.get("split") in splits]
        only = None
        if a.only_ids:
            want = set(json.loads(a.only_ids.read_text()))
            only = {}
            for sp, _ in shards:
                ids = pq.read_table(sp, columns=["doc_id"]).column("doc_id").to_pylist()
                sel = [d for d in ids if d in want]
                if sel:
                    only[str(sp)] = sel
        t0 = time.time()
        res = scan(a.index, shards, a.out_dir, a.workers, a.min_coverage, only, a.common)
        df = res.pop("gram_df")
        rep = {
            "version": VERSION,
            "index_identity": json.loads((a.index / "index.json").read_text())["index_identity"],
            "accepted_manifest_sha256": sha256_file(a.accepted / "manifest.json"),
            "build_identity": man["build_identity"],
            "splits": sorted(splits),
            "only_ids_sha256": sha256_file(a.only_ids) if a.only_ids else None,
            "common_sha256": sha256_file(a.common) if a.common else None,
            "min_coverage": a.min_coverage,
            **res,
            "gram_df_summary": {
                "distinct_index_grams_seen": len(df),
                "df_ge_10": sum(1 for v in df.values() if v >= 10),
                "df_ge_50": sum(1 for v in df.values() if v >= 50),
            },
            "seconds": round(time.time() - t0, 1),
        }
        a.report.write_text(json.dumps(rep, indent=1))
        np.save(
            a.report.with_suffix(".df_ge_10.npy"),
            np.array(sorted(int(k) for k, v in df.items() if v >= 10), dtype=np.uint64),
        )
        print(
            json.dumps(
                {k: rep[k] for k in ("units", "units_reused", "documents", "hit_documents", "seconds")}
            )
        )
        return 0
    idx = Index(a.index)
    items = json.loads(a.items_cache.read_text())["items"]
    rng = random.Random(11)
    files = sorted(a.negatives.rglob("*.parquet"))
    neg: list[str] = []
    for f in rng.sample(files, min(len(files), 40)):
        neg += pq.read_table(f, columns=["text"]).column("text").to_pylist()[:100]
    neg = rng.sample(neg, min(a.n_negative, len(neg)))
    res = controls(idx, items, neg, a.min_coverage, a.per_bench, np.load(a.common) if a.common else None)
    res["common_sha256"] = sha256_file(a.common) if a.common else None
    res["index_identity"] = idx.meta["index_identity"]
    a.out.write_text(json.dumps(res, indent=1))
    print(json.dumps(res["negative"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
