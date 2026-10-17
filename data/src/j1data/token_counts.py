"""Provisional token counting and tokenizer-stage preparation.

Counts every accepted document under each provisional measurement tokenizer
(see ``tokenization``) so the corpus can be sized against the 3B+ token
objective across the plausible vocabulary range. Counts are labelled with the
measurement tokenizer's hash and are never a canonical Juniper 1 count.

For the later tokenizer stage this module also prepares, without training or
freezing anything: a reproducible training-split document sample (keyed hash
of doc id) and domain probe texts drawn from the validation split.
No token-ID training shards are written; the accepted normalized text is the
canonical corpus and is re-tokenized once a tokenizer is frozen.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from tokenizers import Tokenizer

from j1data import faults
from j1data.hashing import sha256_file, sha256_json, unit_interval
from j1data.shards import ShardWriter, remove_stale_temporaries, write_json_atomic
from j1data.tokenization import encode_lengths

VERSION = "j1.tokenized.v1"
SAMPLE_NAMESPACE = "j1.tokenizer-sample.v1"
PROBE_NAMESPACE = "j1.tokenizer-probe.v1"
_BATCH = 512


def accepted_shards(manifest: dict[str, Any], split: str | None = None) -> list[dict[str, Any]]:
    return [s for s in manifest["shards"] if split is None or s["split"] == split]


def tokenizer_sample(root: Path, manifest: dict[str, Any], rate: float) -> tuple[list[str], dict[str, Any]]:
    """Documents of the training split selected by a keyed hash of their id: a
    representative sample of the accepted mixture that does not depend on order."""
    texts: list[str] = []
    ids: list[str] = []
    by_source: Counter[str] = Counter()
    for shard in accepted_shards(manifest, "train"):
        table = pq.read_table(root / shard["file"], columns=["doc_id", "text", "source_id"])
        for did, text, source in zip(
            *(table.column(c).to_pylist() for c in ("doc_id", "text", "source_id")), strict=True
        ):
            if unit_interval(SAMPLE_NAMESPACE, did) < rate:
                texts.append(text)
                ids.append(did)
                by_source[source] += len(text.encode("utf-8"))
    info = {
        "namespace": SAMPLE_NAMESPACE,
        "rate": rate,
        "documents": len(ids),
        "bytes": sum(by_source.values()),
        "bytes_by_source": dict(sorted(by_source.items())),
        "doc_ids_sha256": sha256_json(sorted(ids)),
    }
    return texts, info


def write_sample(directory: Path, texts: list[str], info: dict[str, Any]) -> dict[str, Any]:
    """Persist the sample for the tokenizer stage (text plus the selection record)."""
    directory.mkdir(parents=True, exist_ok=True)
    writer = ShardWriter(
        directory / "sample.parquet", pa.schema([("text", pa.large_string())]), rows_per_group=1024
    )
    for text in texts:
        writer.append({"text": text})
    written = writer.close()
    record = {**info, "file": written}
    write_json_atomic(directory / "sample.json", record)
    return record


_URL_RE = re.compile(r"https?://[^\s<>\"')\]]{6,200}")
_PATH_RE = re.compile(r"(?<![\w:/])(?:~|\.{1,2})?(?:/[\w.\-]+){2,}/?|[A-Za-z]:\\(?:[\w.\-]+\\)*[\w.\-]+")
_API_RE = re.compile(r"\b[A-Za-z_][\w]*(?:\.[A-Za-z_][\w]*){2,}\(?")
_SCI_UNICODE_RE = re.compile(
    r"^.*[\u0370-\u03ff\u2070-\u209f\u2200-\u22ff\u27c0-\u27ef\u2980-\u2aff\u00b1\u00b2\u00b3\u00b5\u00d7\u00f7].*$",
    re.M,
)


def build_probes(
    root: Path, manifest: dict[str, Any], per_domain: int = 200, max_chars: int = 4000
) -> dict[str, list[str]]:
    """Domain probe texts drawn from the validation split (never the training split)."""
    probes: dict[str, list[str]] = defaultdict(list)
    snippets: dict[str, list[str]] = defaultdict(list)
    rows = []
    for shard in accepted_shards(manifest, "validation"):
        table = pq.read_table(
            root / shard["file"], columns=["doc_id", "text", "kind", "domain", "formats", "meta", "source_id"]
        )
        rows.extend(table.to_pylist())
    rows.sort(key=lambda r: unit_interval(PROBE_NAMESPACE, r["doc_id"]))
    for row in rows:
        text = row["text"][:max_chars]
        meta = json.loads(row["meta"])
        buckets = []
        if row["source_id"] == "stackv2-edu":
            language = meta.get("language") or ""
            buckets.append(
                "python" if language == "Python" else "markdown" if language == "Markdown" else "other_code"
            )
        elif row["kind"] == "prose" and row["domain"] in ("general", "humanities_social", "health_medicine"):
            buckets.append("english_prose")
        if row["kind"] == "math":
            buckets.append("math_latex")
        if row["domain"] in ("physical_science", "life_science"):
            buckets.append("science")
        buckets.extend(
            f"format_{f}" for f in row["formats"] if f in ("json", "yaml", "xml_html", "shell", "markdown")
        )
        for bucket in buckets:
            if len(probes[bucket]) < per_domain:
                probes[bucket].append(text)
        snippets["urls"].extend(_URL_RE.findall(row["text"])[:5])
        snippets["paths"].extend(_PATH_RE.findall(row["text"])[:5])
        snippets["api_names"].extend(_API_RE.findall(row["text"])[:5])
        snippets["scientific_unicode"].extend(_SCI_UNICODE_RE.findall(row["text"])[:3])
    for name, values in snippets.items():
        unique = sorted(set(values), key=lambda v: unit_interval(PROBE_NAMESPACE, v))
        probes[name] = unique[: per_domain * 5]
    return dict(sorted(probes.items()))


COUNT_SCHEMA_FIELDS = [
    ("doc_id", pa.string()),
    ("source_id", pa.string()),
    ("split", pa.string()),
    ("domain", pa.string()),
    ("kind", pa.string()),
]


def count_tokens(
    accepted_root: Path,
    out_root: Path,
    manifest: dict[str, Any],
    tokenizers: dict[str, Tokenizer],
    tokenizer_hashes: dict[str, str],
) -> list[dict[str, Any]]:
    """Per-document token counts for every measurement tokenizer, one table per accepted shard.

    Resumable: a shard is skipped when its marker exists, its count table still
    hashes to the recorded value, and it was counted from the same accepted shard
    with the same tokenizers."""
    schema = pa.schema([*COUNT_SCHEMA_FIELDS, *((f"tokens_{n}", pa.int64()) for n in sorted(tokenizers))])
    markers = []
    remove_stale_temporaries(out_root)
    for position, shard in enumerate(manifest["shards"]):
        stem = shard["file"].removesuffix(".parquet")
        marker_path = out_root / "_units" / f"{stem}.json"
        if marker_path.exists():
            marker = json.loads(marker_path.read_text())
            output = marker["outputs"][0]
            if (
                marker["accepted_sha256"] == shard["sha256"]
                and marker["tokenizers"] == tokenizer_hashes
                and sha256_file(out_root / output["file"]) == output["sha256"]
            ):
                markers.append(marker)
                continue
        marker_path.unlink(missing_ok=True)  # a marker never outlives the output it describes
        writer = ShardWriter(out_root / "counts" / f"{stem}.parquet", schema)
        totals: Counter[str] = Counter()
        parquet = pq.ParquetFile(accepted_root / shard["file"])
        try:
            for batch in parquet.iter_batches(
                batch_size=_BATCH, columns=["doc_id", "source_id", "split", "domain", "kind", "text"]
            ):
                columns = batch.to_pydict()
                texts = columns.pop("text")
                lengths = {name: encode_lengths(tok, texts) for name, tok in sorted(tokenizers.items())}
                for name, values in lengths.items():
                    totals[name] += int(values.sum())
                for index in range(len(texts)):
                    writer.append(
                        {
                            **{k: v[index] for k, v in columns.items()},
                            **{f"tokens_{n}": int(v[index]) for n, v in lengths.items()},
                        }
                    )
                faults.point("count.batch")
            info = writer.close()
        except BaseException:
            writer.abort()
            raise
        faults.point("count.shard.written")
        marker = {
            "schema": "j1.token-count-marker.v1",
            "accepted_shard": shard["file"],
            "accepted_sha256": shard["sha256"],
            "split": shard["split"],
            "source_id": shard["source_id"],
            "documents": shard["rows"],
            "tokenizers": tokenizer_hashes,
            "tokens": dict(totals),
            "outputs": [{**info, "file": f"counts/{stem}.parquet"}],
        }
        write_json_atomic(marker_path, marker)
        faults.point("count.shard.commit")
        markers.append(marker)
        if position % 100 == 0:
            print(f"counted {position + 1}/{len(manifest['shards'])}", flush=True)
    return markers
