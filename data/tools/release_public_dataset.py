#!/usr/bin/env python3
"""Build the public edition of the Juniper 1 Base corpus (j1-base-v0.2.2) from the accepted shards.

Seven sources ship as text payload with their per-document licence and attribution fields; three sources
(stackv2-edu, finemath-4plus, fineweb-edu-dedup) ship as reconstruction manifests without text, as decided in
configs/owner_decisions.toml (2026-10-16) and docs/release/dataset_release_plan_2026-10-16.md. Reads the
accepted parquet in place; writes per-config, per-split parquet plus release_manifest.json with SHA-256 and
row counts. Refuses to write text for a held source.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

TEXT_SOURCES = [
    "libretexts",
    "pressbooks",
    "oercommons",
    "python-peps",
    "wikimedia",
    "arxiv-papers",
    "stackexchange",
]
MANIFEST_SOURCES = ["stackv2-edu", "finemath-4plus", "fineweb-edu-dedup"]
SPLITS = ["train", "validation", "test"]
CC_RE = re.compile(r"https?://creativecommons\.org/(licenses/(by|by-sa)/(\d\.\d)|publicdomain/zero/1\.0)")


def license_fields(lic: str | None) -> tuple[str, str]:
    """(license_url, license_spdx) from the upstream licence string; the verbatim string stays
    authoritative."""
    if not lic:
        return "", ""
    m = CC_RE.search(lic)
    if m:
        url = m.group(0)
        if "publicdomain/zero" in url:
            return url, "CC0-1.0"
        kind, ver = m.group(2), m.group(3)
        return url, f"CC-{kind.upper()}-{ver}"
    low = lic.lower()
    if "public domain" in low:
        return "", "LicenseRef-Public-Domain"
    if "gnu free documentation" in low or "gfdl" in low:
        return "https://www.gnu.org/licenses/fdl-1.3.html", "LicenseRef-GFDL-unversioned"
    if "cc0" in low or "zero" in low:
        return "https://creativecommons.org/publicdomain/zero/1.0/", "CC0-1.0"
    return "", "LicenseRef-Unmapped"


def authors_of(meta: dict) -> list[str]:
    a = meta.get("authors")
    if isinstance(a, str) and a:
        # arXiv: one string, "A, B, and C"
        return [x.strip() for x in re.split(r",\s*(?:and\s+)?|\s+and\s+", a) if x.strip()]
    if isinstance(a, list):
        out = []
        for x in a:
            if isinstance(x, list) and x:
                out.append(str(x[0]))
            elif isinstance(x, dict):
                out.append(str(x.get("name") or x.get("url") or x))
            else:
                out.append(str(x))
        return out
    a = meta.get("author")
    if isinstance(a, str) and a:
        return [s.strip() for s in a.split(", ") if s.strip()]
    if isinstance(a, list):
        return [str(x) for x in a]
    return []


def text_rows(tbl: pa.Table, source: str) -> pa.Table:
    cols = tbl.to_pydict()
    n = tbl.num_rows
    out = {
        k: []
        for k in (
            "doc_id",
            "source_id",
            "split",
            "text",
            "sha256",
            "norm_sha256",
            "n_bytes",
            "license",
            "license_url",
            "license_spdx",
            "url",
            "title",
            "authors",
            "upstream_id",
            "upstream_file",
            "upstream_meta",
            "redactions",
            "kind",
            "domain",
            "formats",
        )
    }
    for i in range(n):
        meta = json.loads(cols["meta"][i] or "{}")
        lic = meta.get("license") or ""
        url, spdx = license_fields(lic)
        out["doc_id"].append(cols["doc_id"][i])
        out["source_id"].append(source)
        out["split"].append(cols["split"][i])
        out["text"].append(cols["text"][i])
        out["sha256"].append(cols["sha256"][i])
        out["norm_sha256"].append(cols["norm_sha256"][i])
        out["n_bytes"].append(cols["n_bytes"][i])
        out["license"].append(lic)
        out["license_url"].append(url)
        out["license_spdx"].append(spdx)
        out["url"].append(meta.get("url") or cols["source_key"][i])
        out["title"].append(meta.get("title"))
        out["authors"].append(authors_of(meta))
        out["upstream_id"].append(str(meta.get("doc_id") or cols["source_key"][i]))
        out["upstream_file"].append(str(meta.get("provenance") or cols["unit_id"][i]))
        out["upstream_meta"].append(json.dumps(meta, ensure_ascii=False, sort_keys=True))
        out["redactions"].append(cols["redactions"][i])
        out["kind"].append(cols["kind"][i])
        out["domain"].append(cols["domain"][i])
        out["formats"].append(cols["formats"][i])
    schema = pa.schema(
        [
            ("doc_id", pa.string()),
            ("source_id", pa.string()),
            ("split", pa.string()),
            ("text", pa.large_string()),
            ("sha256", pa.string()),
            ("norm_sha256", pa.string()),
            ("n_bytes", pa.int64()),
            ("license", pa.string()),
            ("license_url", pa.string()),
            ("license_spdx", pa.string()),
            ("url", pa.string()),
            ("title", pa.string()),
            ("authors", pa.list_(pa.string())),
            ("upstream_id", pa.string()),
            ("upstream_file", pa.string()),
            ("upstream_meta", pa.string()),
            ("redactions", pa.string()),
            ("kind", pa.string()),
            ("domain", pa.string()),
            ("formats", pa.list_(pa.string())),
        ]
    )
    return pa.Table.from_pydict(out, schema=schema)


LOCATOR_KEYS = {
    "stackv2-edu": [
        "blob_id",
        "repo_name",
        "path",
        "revision_id",
        "content_id",
        "snapshot_id",
        "license_type",
        "detected_licenses",
        "license",
        "gha_license_id",
        "language",
        "length_bytes",
        "star_events_count",
        "fork_events_count",
    ],
    "finemath-4plus": [
        "warc_filename",
        "warc_record_offset",
        "crawl",
        "fetch_time",
        "score",
        "int_score",
        "language_score",
    ],
    "fineweb-edu-dedup": ["dump", "file_path", "date", "score", "int_score", "language_score", "token_count"],
}


def manifest_rows(tbl: pa.Table, source: str) -> pa.Table:
    cols = tbl.to_pydict()
    n = tbl.num_rows
    keys = LOCATOR_KEYS[source]
    out = {
        k: []
        for k in (
            "doc_id",
            "source_id",
            "split",
            "sha256",
            "norm_sha256",
            "n_bytes",
            "upstream_id",
            "url",
            "upstream_file",
            *keys,
            "upstream_meta",
        )
    }
    for i in range(n):
        meta = json.loads(cols["meta"][i] or "{}")
        out["doc_id"].append(cols["doc_id"][i])
        out["source_id"].append(source)
        out["split"].append(cols["split"][i])
        out["sha256"].append(cols["sha256"][i])
        out["norm_sha256"].append(cols["norm_sha256"][i])
        out["n_bytes"].append(cols["n_bytes"][i])
        out["upstream_id"].append(str(meta.get("blob_id") or meta.get("url") or cols["source_key"][i]))
        out["url"].append(str(meta.get("url") or cols["source_key"][i]))
        prov = meta.get("provenance")
        out["upstream_file"].append(f"{cols['unit_id'][i]}" + (f":{prov}" if prov else ""))
        for k in keys:
            v = meta.get(k)
            out[k].append(
                json.dumps(v, ensure_ascii=False)
                if isinstance(v, (list, dict))
                else (None if v is None else str(v))
            )
        out["upstream_meta"].append(json.dumps(meta, ensure_ascii=False, sort_keys=True))
    fields = [(k, pa.int64() if k == "n_bytes" else pa.string()) for k in out]
    return pa.Table.from_pydict(out, schema=pa.schema(fields))


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        while b := f.read(16 << 20):
            h.update(b)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--accepted", required=True, help=".../accepted/j1-base-v0.2.2")
    ap.add_argument("--out", required=True)
    ap.add_argument("--rows-per-file", type=int, default=50000)
    a = ap.parse_args()
    acc, out = Path(a.accepted), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    man = {
        "schema": "juniper1.public-dataset-release.v1",
        "build": "j1-base-v0.2.2",
        "accepted_manifest_sha256": sha256_file(acc / "manifest.json"),
        "configs": {},
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    for source in TEXT_SOURCES + MANIFEST_SOURCES:
        kind = "text" if source in TEXT_SOURCES else "manifest"
        cfg = f"{kind}-{source}"
        cdir = out / cfg
        cdir.mkdir(exist_ok=True)
        entry = {"kind": kind, "source_id": source, "splits": {}}
        for split in SPLITS:
            files = sorted((acc / split / source).glob("*.parquet"))
            rows = 0
            nfile = 0
            part = []
            part_rows = 0
            written = []
            lic_counts: dict[str, int] = {}

            def flush(cdir=cdir, split=split, written=written, cfg=cfg):
                nonlocal part, part_rows, nfile
                if not part:
                    return
                t = pa.concat_tables(part)
                p = cdir / f"{split}-{nfile:05d}.parquet"
                pq.write_table(t, p, compression="zstd", compression_level=9)
                written.append(
                    {
                        "path": f"{cfg}/{p.name}",
                        "rows": t.num_rows,
                        "bytes": p.stat().st_size,
                        "sha256": sha256_file(p),
                    }
                )
                nfile += 1
                part = []
                part_rows = 0

            for f in files:
                tbl = pq.read_table(f)
                t = text_rows(tbl, source) if kind == "text" else manifest_rows(tbl, source)
                if kind == "text":
                    for v in t.column("license").to_pylist():
                        lic_counts[v] = lic_counts.get(v, 0) + 1
                    assert all(bool(u) for u in t.column("url").to_pylist()), f"{source} {f.name}: empty url"
                rows += t.num_rows
                part.append(t)
                part_rows += t.num_rows
                if part_rows >= a.rows_per_file:
                    flush()
            flush()
            entry["splits"][split] = {
                "documents": rows,
                "files": written,
                **({"license_counts": lic_counts} if kind == "text" else {}),
            }
            print(f"{cfg} {split}: {rows} rows, {len(written)} files", flush=True)
        man["configs"][cfg] = entry
    man["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    man["documents_total"] = sum(
        s["documents"] for c in man["configs"].values() for s in c["splits"].values()
    )
    (out / "release_manifest.json").write_text(json.dumps(man, indent=1) + "\n")
    print("documents_total", man["documents_total"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
