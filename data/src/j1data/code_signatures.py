"""License-header-aware MinHash signatures for code deduplication.

Short source files that share a standard license header (MIT, Apache-2.0,
BSD) look like near-duplicates when the header dominates their shingles; the
dedup calibration found clusters of hundreds of unrelated files joined that
way. For near-duplicate detection only, code documents are re-signed after
removing comment blocks that contain license or copyright language and
Markdown "License" sections. Stored text is never changed.

Signatures are recomputed at dedup time from the candidate text and stored
next to the dedup outputs, so processed units keep their identity.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from j1data.dedup import UnitRef
from j1data.minhash import MinHashConfig, MinHasher
from j1data.ngrams import word_hashes
from j1data.normalize import signature_words
from j1data.pools import run_unordered
from j1data.shards import SignatureSink, verify_files, write_json_atomic

VERSION = "j1.code-signatures.v1"

_LICENSE_RE = re.compile(
    r"copyright|licen[cs]e|permission is hereby granted|all rights reserved|spdx-license-identifier|"
    r"warrant(?:y|ies)|redistribution and use|provided \"as is\"|gnu (?:lesser )?general public",
    re.IGNORECASE,
)
_COMMENT_LINE_RE = re.compile(r"^\s*(?:#|//|--|;|\*|/\*|\*/|<!--|-->|rem\b|%|'|\"\"\"|''')", re.IGNORECASE)
_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+(.*)$")


def strip_license_text(text: str) -> str:
    """Drop comment blocks with license language and Markdown license sections."""
    lines = text.split("\n")
    keep = [True] * len(lines)
    start = None
    for index in range(len(lines) + 1):
        is_comment = index < len(lines) and (
            _COMMENT_LINE_RE.match(lines[index]) is not None
            or (start is not None and not lines[index].strip())
        )
        if is_comment and start is None:
            start = index
        elif not is_comment and start is not None:
            block = "\n".join(lines[start:index])
            if _LICENSE_RE.search(block):
                for position in range(start, index):
                    keep[position] = False
            start = None
    in_license_section = False
    for index, line in enumerate(lines):
        heading = _HEADING_RE.match(line)
        if heading:
            in_license_section = bool(re.search(r"licen[cs]e|copyright", heading.group(1), re.IGNORECASE))
        if in_license_section:
            keep[index] = False
    body = "\n".join(line for line, kept in zip(lines, keep, strict=True) if kept)
    return body if body.strip() else text


def resign_unit(ref: UnitRef, out_path: Path, config: MinHashConfig) -> dict[str, Any]:
    hasher = MinHasher(config)
    sink = SignatureSink(out_path, config.num_perm)
    try:
        for batch in pq.ParquetFile(ref.candidates).iter_batches(batch_size=2048, columns=["text"]):
            for text in batch.column("text").to_pylist():
                words = signature_words(strip_license_text(text))
                sink.append(hasher.signature(hasher.shingles(word_hashes(words))))
        info = sink.close()
    except BaseException:
        sink.abort()
        raise
    if info["rows"] != ref.rows:
        raise RuntimeError(f"{ref.unit_id}: re-signed {info['rows']} rows, expected {ref.rows}")
    return info


def _job(args: tuple[UnitRef, Path, MinHashConfig]) -> dict[str, Any]:
    ref, path, config = args
    info = resign_unit(ref, path, config)
    # Marker last: an override without a verifying marker is never reused.
    write_json_atomic(path.with_suffix(".json"), {**info, "unit_id": ref.unit_id, "source_id": ref.source_id})
    return info


def _reusable(ref: UnitRef, path: Path) -> bool:
    marker_path = path.with_suffix(".json")
    if not path.exists() or not marker_path.exists():
        return False
    marker = json.loads(marker_path.read_text())
    return bool(
        marker.get("rows") == ref.rows
        and marker.get("unit_id") == ref.unit_id
        and not verify_files(path.parent, [{**marker, "file": path.name}])
        and np.load(path, mmap_mode="r").shape[1] == ref.rows
    )


def resign(refs: list[UnitRef], out_dir: Path, config: MinHashConfig, sources: set[str], workers: int) -> int:
    """Re-sign every unit of ``sources``; outputs whose marker (rows, SHA-256) verifies are kept."""

    jobs = []
    for ref in refs:
        if ref.source_id not in sources:
            continue
        path = out_dir / ref.source_id / f"{ref.unit_id}.npy"
        if _reusable(ref, path):
            continue
        jobs.append((ref, path, config))
    for _ in run_unordered(_job, jobs, workers):
        pass
    return len(jobs)
