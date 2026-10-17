"""F9: index and scan the HumanEval+ / MBPP+ test fields and MBPP challenge tests (``j1.contamination-f9.v1``).

Pre-registered in ``docs/contamination_f9_criteria.md``. Matching is the unchanged ``j1.contamination-layers.v1``
logic (``tools/contamination_layers.py``): all-token 10-grams, coverage >= 0.6 over non-common grams, common
grams = df >= 10 over the scanned corpus (two-pass rule), ``common_only`` items, Layer C identifiers.

What is new here:

* **Items.** The registered suite files on disk (``data/exclusions/raw/<id>/<revision>/``) are re-hashed and must
  equal the layers-v1 provenance; each added field value becomes one whole-value item plus 64-token windows.
* **Pass 1** (``scan``) writes one restartable result per accepted shard (keyed by shard SHA-256, index
  identity and tool SHA-256): uncalibrated hits, F9-gram document frequencies, and, for every document holding
  any F9 gram or identifier, that document's in-index gram set and identifiers.
* **Pass 2** is computed offline from those recorded sets with the df >= 10 common set (``match_grams``); a
  document without any F9 gram or identifier cannot match, so this equals a re-scan.

Everything fails closed: missing/mismatched inputs, unknown columns, non-string values, identity mismatches.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import random
import re
import sys
import time
import tomllib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
import contamination_layers as L

VERSION = "j1.contamination-f9.v1"
TOOL_SHA = hashlib.sha256(Path(__file__).read_bytes() + L.TOOL_SHA.encode()).hexdigest()
REPO = Path(__file__).resolve().parents[1]
W = 64  # pre-registered window length in layers-v1 tokens (stride W, last window = final W tokens)
MIN_COV = 0.6  # layers-v1 pre-registered coverage
COMMON_DF = L.COMMON_DF  # 10

# label -> (registry benchmark id, file patterns inside the revision, fields, field must be a non-empty string)
SPECS: dict[str, tuple[str, list[str], list[str], bool]] = {
    "humanevalplus-test": ("humanevalplus", ["data/*.parquet"], ["test"], True),
    "mbppplus-test": ("mbppplus", ["data/*.parquet"], ["test"], True),
    "mbpp-challenge-tests": ("mbpp", ["full/*.parquet"], ["challenge_test_list", "test_setup_code"], False),
}

# Generic identifiers adjudicated NOT_PROHIBITED (reports/qualification-2026-10-10/contamination/
# adjudication_pilot_inputs.json, layer_c_identifiers.values; the four described constants as exact literals).
GENERIC_IDENTS = frozenset(
    {
        "x86_64-linux-gnu", "glucose-6-phosphate", "adaptiveavgpool2d", "25-hydroxyvitamin", "3-dimethylbutane",
        "glucose-1-phosphate", "2-dichlorobenzene", "glucose-6-phosphatase", "3-dichlorobenzene",
        "4-dichlorobenzene", "6-trichloroanisole",
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",  # sha256 of the empty string
        "akiaiosfodnn7example",  # AWS documented example access key ID (case-folded)
        "123456789abcdef0",
        "5f4dcc3b5aa765d61d8327deb882cf99",  # md5('password')
    }
)  # fmt: skip


class F9Error(SystemExit):
    """Fail-closed stop (exit status carries the message)."""


# --- items ------------------------------------------------------------------------------------


def windows(text: str, w: int = W) -> list[str]:
    """Substrings of the normalized text covering ``w`` consecutive layers-v1 tokens, stride ``w``; the last
    window is the final ``w`` tokens. Values shorter than ``w`` tokens give no window."""
    norm = L.normalize(text)
    spans = [m.span() for m in L.TOKEN_RE.finditer(norm)]
    n = len(spans)
    if n < w:
        return []
    starts = list(range(0, n - w + 1, w))
    if starts[-1] != n - w:
        starts.append(n - w)
    return [norm[spans[s][0] : spans[s + w - 1][1]] for s in starts]


def _values(record: dict[str, Any], field: str, required: bool, where: str) -> list[str]:
    if field not in record:
        raise F9Error(f"{where}: column {field!r} missing")
    v = record[field]
    if isinstance(v, str):
        if required and not v.strip():
            raise F9Error(f"{where}: empty {field!r}")
        return [v] if v.strip() else []
    if v is None and not required:
        return []
    if isinstance(v, list) and not required and all(isinstance(x, str) for x in v):
        return ["\n".join(v)] if v else []
    raise F9Error(f"{where}: {field!r} has unexpected type {type(v).__name__}")


def collect(raw_root: Path, provenance: dict[str, Any], registry: dict[str, Any]) -> tuple[dict[str, list[str]], dict]:
    """F9 items per label, from the registered raw files, verified against the layers-v1 provenance."""
    reg = {e["id"]: e for e in registry["benchmark"]}
    items: dict[str, list[str]] = {}
    report: dict[str, Any] = {"version": VERSION, "window_tokens": W, "labels": {}}
    for label, (bench, patterns, fields, required) in SPECS.items():
        if bench not in reg or bench not in provenance:
            raise F9Error(f"{label}: benchmark {bench} absent from registry or provenance")
        src = provenance[bench]
        if src.get("revision") != reg[bench]["revision"] or src.get("repo") != reg[bench]["repo"]:
            raise F9Error(f"{label}: provenance revision/repo differs from configs/exclusions.toml")
        recorded = [f for f in src["file_hashes"] if any(fnmatch.fnmatch(f["path"], p) for p in patterns)]
        if not recorded:
            raise F9Error(f"{label}: no recorded files match {patterns}")
        out: list[str] = []
        seen: set[str] = set()
        stats = {"records": 0, "values": 0, "whole_items": 0, "window_items": 0, "duplicates_dropped": 0}
        files = []
        for f in sorted(recorded, key=lambda e: e["path"]):
            p = raw_root / bench / src["revision"] / f["path"]
            if not p.is_file():
                raise F9Error(f"{label}: missing input {p}")
            digest = L.sha256_file(p)
            if digest != f["sha256"]:
                raise F9Error(f"{label}: {p} SHA-256 {digest} != recorded {f['sha256']}")
            files.append({"path": f["path"], "sha256": digest})
            table = pq.read_table(p)
            for col in fields:
                if col not in table.column_names:
                    raise F9Error(f"{label}: {f['path']} lacks column {col!r}")
            for i, rec in enumerate(table.to_pylist()):
                stats["records"] += 1
                for col in fields:
                    for v in _values(rec, col, required, f"{label}:{f['path']}#{i}"):
                        stats["values"] += 1
                        for kind, t in [("whole_items", v)] + [("window_items", x) for x in windows(v)]:
                            if t in seen:
                                stats["duplicates_dropped"] += 1
                                continue
                            seen.add(t)
                            out.append(t)
                            stats[kind] += 1
        if not out:
            raise F9Error(f"{label}: no items")
        items[label] = out
        report["labels"][label] = {
            "benchmark": bench, "repo": src["repo"], "revision": src["revision"], "fields": fields,
            "files": files, **stats,
        }  # fmt: skip
    return items, report


def build_index(items: dict[str, list[str]], report: dict, exclusions_dir: Path) -> tuple[Path, dict]:
    """Build into a staging dir, then publish as ``layers-v1-f9-<identity16>``; never overwrite another index."""
    fp = hashlib.sha256(json.dumps(report, sort_keys=True).encode()).hexdigest()
    stage = exclusions_dir / f".f9-staging-{os.getpid()}"
    if stage.exists():
        raise F9Error(f"staging dir {stage} exists")
    meta = L.build(items, stage, fp)
    (stage / "f9_sources.json").write_text(json.dumps(report, indent=1, sort_keys=True))
    final = exclusions_dir / f"layers-v1-f9-{meta['index_identity'][:16]}"
    if final.exists():
        old = json.loads((final / "index.json").read_text())
        for f in stage.iterdir():
            f.unlink()
        stage.rmdir()
        if old["index_identity"] != meta["index_identity"]:
            raise F9Error(f"{final} holds a different index")
        return final, old
    os.rename(stage, final)
    return final, meta


# --- matching from recorded gram sets -----------------------------------------------------------


def match_grams(idx: L.Index, g_in: np.ndarray, ids: list[str], min_cov: float,
                common: np.ndarray | None) -> dict | None:  # fmt: skip
    """``L.Index.match`` computed from a document's in-index distinct grams and identifiers (equality is tested)."""
    g = np.unique(np.asarray(g_in, dtype=np.uint64))
    hit_items: dict[int, float] = {}
    common_only: dict[int, float] = {}
    if len(g) and len(idx.G):
        lo = np.searchsorted(idx.G, g, side="left")
        hi = np.searchsorted(idx.G, g, side="right")
        m = hi > lo
        if not m.all():
            raise F9Error("recorded gram not in index (index/result mismatch)")
        use_common = common is not None and len(common) > 0
        is_common = np.isin(g, common) if use_common else np.zeros(len(g), dtype=bool)
        items_all = np.concatenate([idx.I[a:b] for a, b in zip(lo, hi, strict=True)])
        rep = hi - lo
        cnt_all = np.bincount(items_all, minlength=len(idx.n_item))
        cnt_common = np.bincount(items_all[np.repeat(is_common, rep)], minlength=len(idx.n_item))
        n_common = idx._common_counts(common) if use_common else np.zeros_like(cnt_all)
        n_eff = idx.n_item - n_common
        for i in np.nonzero(cnt_all)[0].tolist():
            if n_eff[i] > 0:
                c = (cnt_all[i] - cnt_common[i]) / n_eff[i]
                if c >= min_cov:
                    hit_items[i] = c
            elif idx.n_item[i] > 0:
                c = cnt_all[i] / idx.n_item[i]
                if c >= min_cov:
                    common_only[i] = c
    found = set(ids) & idx.idents.keys() if idx.idents else set()
    if not hit_items and not found and not common_only:
        return None
    return {
        "layer_a": {str(i): round(c, 4) for i, c in sorted(hit_items.items())},
        "layer_a_benchmarks": sorted({idx.bench[i] for i in hit_items}),
        "layer_a_common_only": {str(i): round(c, 4) for i, c in sorted(common_only.items())},
        "common_only_benchmarks": sorted({idx.bench[i] for i in common_only}),
        "layer_c": sorted(found),
        "layer_c_benchmarks": sorted({idx.bench[i] for x in found for i in idx.idents[x]}),
    }


def labels_of(idx: L.Index, m: dict | None) -> set[str]:
    if not m:
        return set()
    return (
        set(m["layer_a_benchmarks"])
        | set(m["layer_c_benchmarks"])
        | {idx.bench[int(i)] for i in m["layer_a_common_only"]}
    )


# --- pass 1 -------------------------------------------------------------------------------------

_IDX: L.Index | None = None


def _init(index_dir: str) -> None:
    global _IDX
    _IDX = L.Index(Path(index_dir))


def scan_unit(task: tuple[str, str, str, str, str]) -> dict:
    path, out_dir, expected_sha, split, source = task
    idx = _IDX
    if idx is None:
        raise F9Error("index not loaded")
    p = Path(path)
    shard_sha = L.sha256_file(p)
    if shard_sha != expected_sha:
        raise F9Error(f"{p}: SHA-256 {shard_sha} != accepted manifest {expected_sha}")
    identity = idx.meta["index_identity"]
    out = Path(out_dir) / f"{shard_sha[:32]}-{identity[:16]}.json"
    if out.is_file():
        r = json.loads(out.read_text())
        if (r.get("tool_sha256"), r.get("shard_sha256"), r.get("index_identity")) == (TOOL_SHA, shard_sha, identity):
            r["reused"] = True
            return r
    t0 = time.time()
    docs = 0
    df: dict[int, int] = {}
    hits, touched = [], []
    for batch in pq.ParquetFile(p).iter_batches(columns=["doc_id", "text"], batch_size=512):
        for d, text in zip(batch.column("doc_id").to_pylist(), batch.column("text").to_pylist(), strict=True):
            docs += 1
            g = L.grams(text)
            if len(g) and len(idx.G):
                lo = np.searchsorted(idx.G, g, side="left")
                g = g[np.searchsorted(idx.G, g, side="right") > lo]
            ids = sorted(L.idents(text) & idx.idents.keys()) if idx.idents else []
            if not len(g) and not ids:
                continue
            gl = g.tolist()
            for x in gl:
                df[x] = df.get(x, 0) + 1
            touched.append({"doc_id": d, "grams": gl, "idents": ids})
            m = idx.match(text, MIN_COV, None)
            if m:
                hits.append({"doc_id": d, **m})
    r = {
        "version": VERSION,
        "shard": str(p),
        "split": split,
        "source_id": source,
        "shard_sha256": shard_sha,
        "tool_sha256": TOOL_SHA,
        "index_identity": identity,
        "min_coverage": MIN_COV,
        "documents": docs,
        "hits_uncalibrated": hits,
        "touched": touched,
        "indexed_gram_df": {str(k): v for k, v in df.items()},
        "seconds": round(time.time() - t0, 2),
    }
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(r))
    os.replace(tmp, out)
    return r


def shards_of(accepted: Path, expected_manifest_sha: str, splits: set[str]) -> tuple[dict, list[dict]]:
    got = L.sha256_file(accepted / "manifest.json")
    if got != expected_manifest_sha:
        raise F9Error(f"accepted manifest SHA-256 {got} != expected {expected_manifest_sha}")
    man = json.loads((accepted / "manifest.json").read_text())
    shards = [e for e in man["shards"] if e.get("split") in splits]
    if not shards:
        raise F9Error("no shards selected")
    return man, shards


def calibrate(idx: L.Index, units: list[dict]) -> tuple[np.ndarray, list[dict]]:
    """Common set (df >= COMMON_DF over all scanned documents) and the calibrated pass-2 hits."""
    df: dict[int, int] = {}
    for r in units:
        for k, v in r["indexed_gram_df"].items():
            df[int(k)] = df.get(int(k), 0) + v
    common = np.array(sorted(k for k, v in df.items() if v >= COMMON_DF), dtype=np.uint64)
    hits = []
    for r in units:
        for t in r["touched"]:
            m = match_grams(idx, np.array(t["grams"], dtype=np.uint64), t["idents"], MIN_COV, common)
            if m:
                hits.append({"doc_id": t["doc_id"], "split": r["split"], "source_id": r["source_id"], **m})
    return common, hits


# --- controls -----------------------------------------------------------------------------------


def perturb(text: str, kind: str, rng: random.Random) -> str:
    if kind in ("verbatim", "case_space", "reindent"):
        return L._transform(text, kind, rng)
    if kind == "comma_spacing":
        return re.sub(r",\s*", lambda _: "," + " " * rng.choice((0, 2)), text)
    if kind == "truncated_80":
        spans = [m.span() for m in L.TOKEN_RE.finditer(text)]
        keep = int(len(spans) * 0.8)
        return text[: spans[keep - 1][1]] if keep else ""
    raise ValueError(kind)


KINDS = ("verbatim", "case_space", "reindent", "comma_spacing", "truncated_80")


def plant(host: str, payload: str, rng: random.Random) -> str:
    lines = host.split("\n")
    k = rng.randint(0, len(lines))
    return "\n".join([*lines[:k], payload, *lines[k:]])


def controls(idx: L.Index, items: dict[str, list[str]], hosts: list[str], negatives: list[str],
             common: np.ndarray | None, per_label: int = 200, seed: int = 9) -> dict:  # fmt: skip
    if not hosts:
        raise F9Error("no host documents for planting")
    rng = random.Random(seed)
    out: dict[str, Any] = {"positive": {}, "negative": {}}
    for label in sorted(items):
        sample = rng.sample(items[label], min(per_label, len(items[label])))
        res = {}
        for kind in KINDS:
            found = 0
            for t in sample:
                doc = plant(rng.choice(hosts), perturb(t, kind, rng), rng)
                found += label in labels_of(idx, idx.match(doc, MIN_COV, common))
            res[kind] = {"planted": len(sample), "detected": found, "rate": round(found / max(1, len(sample)), 4)}
        out["positive"][label] = res
    flagged = sum(1 for t in negatives if idx.match(t, MIN_COV, common))
    out["negative"] = {"documents": len(negatives), "flagged": flagged, "flagged_rate": flagged / max(1, len(negatives))}
    return out


def train_texts(accepted: Path, shards: list[dict], n_files: int, per_file: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    train = [e for e in shards if e["split"] == "train"]
    texts: list[str] = []
    for e in rng.sample(train, min(n_files, len(train))):
        p = accepted / e["file"]
        if L.sha256_file(p) != e["sha256"]:
            raise F9Error(f"{p}: SHA-256 mismatch")
        texts += pq.read_table(p, columns=["text"]).column("text").to_pylist()[:per_file]
    return texts


# --- adjudication -------------------------------------------------------------------------------


def adjudicate(hits: list[dict]) -> list[dict]:
    """Criteria v2 corpus-scale rule for F9 (every F9 item is test content, rule 1c)."""
    out = []
    for h in hits:
        for k in ("doc_id", "split", "source_id", "layer_a", "layer_a_common_only", "layer_c"):
            if k not in h:
                raise F9Error(f"hit without {k!r}: {h.get('doc_id')}")
        if h["split"] not in ("train", "validation", "test"):
            raise F9Error(f"{h['doc_id']}: unknown split {h['split']!r}")
        if h["layer_a"] or h["layer_a_common_only"]:
            labels = sorted(set(h["layer_a_benchmarks"]) | set(h.get("common_only_benchmarks", [])))
            decision = "PROHIBITED"
            reason = f"1c: test content of protected item(s) ({', '.join(labels) or 'common-only'}); Layer A coverage >= 0.6"
        elif h["layer_c"] and set(h["layer_c"]) <= GENERIC_IDENTS:
            decision, reason = "NOT_PROHIBITED", "2: only generic identifiers (adjudication_pilot_inputs.json)"
        elif h["layer_c"]:
            decision, reason = "PROHIBITED", "Layer C non-generic identifier(s) from protected tests (rule 3: uncertain)"
        else:
            raise F9Error(f"{h['doc_id']}: empty hit")
        out.append({"doc_id": h["doc_id"], "split": h["split"], "source_id": h["source_id"],
                    "decision": decision, "reason": reason})  # fmt: skip
    return out


# --- CLI ------------------------------------------------------------------------------------------


def _load_items(args: argparse.Namespace) -> tuple[dict[str, list[str]], dict]:
    prov = json.loads(args.provenance.read_text())["sources"]
    with (REPO / "configs" / "exclusions.toml").open("rb") as f:
        registry = tomllib.load(f)
    return collect(args.raw_root, prov, registry)


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("build", "controls"):
        p = sub.add_parser(name)
        p.add_argument("--raw-root", type=Path, required=True, help="data/exclusions/raw")
        p.add_argument("--provenance", type=Path, required=True, help="layers-v1 items_cache.json (sources)")
    sub.choices["build"].add_argument("--exclusions-dir", type=Path, required=True)
    c = sub.choices["controls"]
    c.add_argument("--index", type=Path, required=True)
    c.add_argument("--accepted", type=Path, required=True)
    c.add_argument("--manifest-sha256", required=True)
    c.add_argument("--common", type=Path)
    c.add_argument("--out", type=Path, required=True)
    s = sub.add_parser("scan")
    s.add_argument("--index", type=Path, required=True)
    s.add_argument("--accepted", type=Path, required=True)
    s.add_argument("--manifest-sha256", required=True)
    s.add_argument("--units-dir", type=Path, required=True)
    s.add_argument("--out-dir", type=Path, required=True)
    s.add_argument("--workers", type=int, default=4)
    s.add_argument("--limit", type=int, help="first N shards only (smoke test)")
    a = sub.add_parser("adjudicate")
    a.add_argument("--hits", type=Path, required=True)
    a.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    if args.cmd == "build":
        items, report = _load_items(args)
        final, meta = build_index(items, report, args.exclusions_dir)
        print(json.dumps({"index": str(final), "index_identity": meta["index_identity"], "items": meta["items"],
                          "benchmarks": meta["benchmarks"]}))  # fmt: skip
        return 0
    if args.cmd == "controls":
        items, _ = _load_items(args)
        idx = L.Index(args.index)
        _, shards = shards_of(args.accepted, args.manifest_sha256, {"train"})
        hosts = train_texts(args.accepted, shards, 20, 50, seed=21)
        neg = train_texts(args.accepted, shards, 40, 100, seed=11)
        neg = random.Random(11).sample(neg, min(3000, len(neg)))
        common = np.load(args.common) if args.common else None
        res = controls(idx, items, hosts, neg, common)
        res |= {"index_identity": idx.meta["index_identity"], "tool_sha256": TOOL_SHA,
                "common_sha256": L.sha256_file(args.common) if args.common else None}  # fmt: skip
        args.out.write_text(json.dumps(res, indent=1))
        print(json.dumps(res))
        return 0
    if args.cmd == "scan":
        if not 1 <= args.workers <= 4:
            raise F9Error("workers must be 1..4")
        man, shards = shards_of(args.accepted, args.manifest_sha256, {"train", "validation", "test"})
        if args.limit:
            shards = shards[: args.limit]
        args.units_dir.mkdir(parents=True, exist_ok=True)
        args.out_dir.mkdir(parents=True, exist_ok=True)
        tasks = [(str(args.accepted / e["file"]), str(args.units_dir), e["sha256"], e["split"], e["source_id"])
                 for e in shards]  # fmt: skip
        t0 = time.time()
        units = []
        with ProcessPoolExecutor(args.workers, initializer=_init, initargs=(str(args.index),)) as ex:
            for i, r in enumerate(ex.map(scan_unit, tasks, chunksize=1)):
                units.append(r)
                if i % 50 == 0:
                    print(f"{i + 1}/{len(tasks)} units, {time.time() - t0:.0f}s", flush=True)
        idx = L.Index(args.index)
        if any(u["index_identity"] != idx.meta["index_identity"] for u in units):
            raise F9Error("unit/index identity mismatch")
        common, hits = calibrate(idx, units)
        np.save(args.out_dir / "common_f9_df_ge_10.npy", common)
        unc = [{"split": u["split"], "source_id": u["source_id"], **h} for u in units for h in u["hits_uncalibrated"]]
        (args.out_dir / "hits_pass1_uncalibrated.json").write_text(json.dumps(unc))
        (args.out_dir / "hits_pass2.json").write_text(json.dumps(hits))
        rep = {
            "version": VERSION,
            "tool_sha256": TOOL_SHA,
            "index_identity": idx.meta["index_identity"],
            "accepted_manifest_sha256": args.manifest_sha256,
            "build_identity": man["build_identity"],
            "units": len(units),
            "units_reused": sum(1 for u in units if u.get("reused")),
            "documents": sum(u["documents"] for u in units),
            "documents_by_split": {sp: sum(u["documents"] for u in units if u["split"] == sp)
                                   for sp in ("train", "validation", "test")},  # fmt: skip
            "touched_documents": sum(len(u["touched"]) for u in units),
            "distinct_index_grams_seen": len({k for u in units for k in u["indexed_gram_df"]}),
            "common_grams_df_ge_10": len(common),
            "common_sha256": L.sha256_file(args.out_dir / "common_f9_df_ge_10.npy"),
            "hit_documents_pass1_uncalibrated": len(unc),
            "hit_documents_pass2": len(hits),
            "seconds": round(time.time() - t0, 1),
        }
        (args.out_dir / "scan_f9.json").write_text(json.dumps(rep, indent=1))
        print(json.dumps(rep))
        return 0
    hits = json.loads(args.hits.read_text())
    dec = adjudicate(hits)
    args.out.write_text(json.dumps(dec))
    print(json.dumps({"hits": len(dec), "prohibited": sum(d["decision"] == "PROHIBITED" for d in dec)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
