"""Per-unit processing: raw records to candidate documents.

For every record of a unit, in this fixed order:

1. selection   adapter decisions (namespace, site, per-document license, vendored)
2. extraction  empty or undecodable text
3. privacy     credentials / high-risk identifiers reject; incidental PII is redacted
4. contamination  evaluation n-gram or canary overlap rejects
5. quality     domain-aware heuristics reject

A document's *primary* stage is the first that rejects it; all quality reason
codes are still recorded. Survivors become candidates with their exact hash,
signature hash and MinHash signature; corpus-wide deduplication happens later
in ``dedup`` because it needs every unit.

A unit is complete only when its marker JSON exists; outputs are written
atomically first and listed in the marker with SHA-256 and row counts. On
restart, units whose marker verifies are skipped and stray partial outputs of
unfinished units are overwritten, so a crash can neither duplicate nor lose
documents.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa

from j1data import domains, normalize, privacy, quality
from j1data.adapters import RawRecord, UnitSpec, make_adapter
from j1data.contamination import ContaminationIndex
from j1data.hashing import doc_id, sha256_json, sha256_text, unit_interval
from j1data.minhash import MinHashConfig, MinHasher
from j1data.ngrams import word_hashes
from j1data.paths import DataLayout, safe_child
from j1data.registry import Source
from j1data.shards import ShardWriter, SignatureSink, verify_files, write_json_atomic

VERSION = "j1.process.v2"
SLOW_DOCUMENT_SECONDS = 2.0
CHECKPOINT_SECONDS = 300.0

CANDIDATE_SCHEMA = pa.schema(
    [
        ("doc_id", pa.string()),
        ("source_id", pa.string()),
        ("source_key", pa.string()),
        ("group_key", pa.string()),
        ("text", pa.large_string()),
        ("sha256", pa.string()),
        ("norm_sha256", pa.string()),
        ("n_chars", pa.int64()),
        ("n_bytes", pa.int64()),
        ("n_words", pa.int64()),
        ("kind", pa.string()),
        ("domain", pa.string()),
        ("formats", pa.list_(pa.string())),
        ("us_spellings", pa.int32()),
        ("uk_spellings", pa.int32()),
        ("english_fw_frac", pa.float32()),
        ("code_line_frac", pa.float32()),
        ("redactions", pa.string()),
        ("meta", pa.string()),
    ]
)

REJECT_SCHEMA = pa.schema(
    [
        ("doc_id", pa.string()),
        ("source_id", pa.string()),
        ("source_key", pa.string()),
        ("stage", pa.string()),
        ("reasons", pa.list_(pa.string())),
        ("n_chars", pa.int64()),
        ("n_bytes", pa.int64()),
        ("kind", pa.string()),
        ("benchmarks", pa.list_(pa.string())),
        ("audit_text", pa.large_string()),
    ]
)


@dataclass
class Settings:
    layout: DataLayout
    build_id: str
    quality: quality.QualityConfig
    minhash: MinHashConfig
    contamination_dir: Path
    audit_reject_text_rate: float
    identity: str
    verified_sources: set[str] | None = None


_STATE: dict[str, Any] = {}


def init_worker(settings: Settings) -> None:
    _STATE["settings"] = settings
    _STATE["minhasher"] = MinHasher(settings.minhash)
    _STATE["index"] = ContaminationIndex.load(settings.contamination_dir)
    _STATE["exclusion_index_identity"] = _STATE["index"].identity()


def unit_dir(layout: DataLayout, build_id: str, kind: str, source_id: str) -> Path:
    return layout.work(build_id) / kind / source_id


def marker_path(layout: DataLayout, build_id: str, unit: UnitSpec) -> Path:
    return unit_dir(layout, build_id, "units", unit.source_id) / f"{unit.unit_id}.json"


def unit_complete(layout: DataLayout, build_id: str, unit: UnitSpec, identity: str, verify: bool) -> bool:
    path = marker_path(layout, build_id, unit)
    if not path.exists():
        return False
    marker = json.loads(path.read_text())
    if marker["build_identity"] != identity:
        raise RuntimeError(f"{unit.unit_id}: produced under a different build identity; use a new build id")
    if verify:
        work = layout.work(build_id)
        problems = verify_files(work, marker["outputs"])
        if problems:
            path.unlink()
            return False
    return True


def _raw_input_identity(source: Source, unit: UnitSpec, manifest: dict[str, Any]) -> str:
    """Bind a progress journal to the exact acquired input behind this unit."""
    if source.adapter == "cinqic_git":
        return sha256_json(
            {
                "source_id": source.source_id,
                "revision": source.revision,
                "files": [
                    (entry.get("path"), entry.get("git_blob_oid")) for entry in manifest.get("files", [])
                ],
            }
        )
    matches = [entry for entry in manifest.get("files", []) if entry.get("path") == unit.file]
    if len(matches) != 1 or not isinstance(matches[0].get("sha256"), str):
        raise RuntimeError(
            f"{source.source_id}/{unit.unit_id}: missing unique raw-file hash in acquisition manifest"
        )
    return str(matches[0]["sha256"])


def _quarantine_outputs(work: Path, entries: list[dict[str, Any]], reason: str) -> list[dict[str, str]]:
    """Move invalid committed files aside before replaying their uncommitted suffix."""
    moved: list[dict[str, str]] = []
    for entry in entries:
        relative = entry.get("file")
        if not isinstance(relative, str):
            continue
        try:
            source = safe_child(work, relative)
        except ValueError:
            continue
        if source.is_symlink() or not source.is_file():
            continue
        target_dir = work / "quarantine" / "process-progress"
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / f"{sha256_json({'file': relative, 'reason': reason})[:16]}-{source.name}"
        suffix = 1
        while target.exists():
            target = (
                target_dir
                / f"{sha256_json({'file': relative, 'reason': reason})[:16]}-{suffix}-{source.name}"
            )
            suffix += 1
        source.replace(target)
        moved.append({"file": relative, "quarantined_as": target.relative_to(work).as_posix()})
    return moved


def _load_progress(
    path: Path,
    work: Path,
    identity: str,
    raw_input_identity: str,
    source: Source,
    unit: UnitSpec,
    exclusion_identity: str,
    initial_cursor: int,
) -> tuple[list[dict[str, Any]], int, list[dict[str, Any]]]:
    if not path.exists():
        return [], initial_cursor, []
    progress = json.loads(path.read_text())
    if not isinstance(progress, dict):
        raise RuntimeError(f"malformed process progress journal: {path}")
    expected = {
        "schema": "j1.process-progress.v1",
        "build_identity": identity,
        "process_version": VERSION,
        "source_id": source.source_id,
        "source_revision": source.revision,
        "unit_id": unit.unit_id,
        "raw_file": unit.file,
        "raw_input_identity": raw_input_identity,
        "exclusion_index_identity": exclusion_identity,
    }
    for key, value in expected.items():
        if progress.get(key) != value:
            raise RuntimeError(f"{unit.unit_id}: process progress {key} changed; use a new build id")
    segments = progress.get("segments")
    if not isinstance(segments, list):
        raise RuntimeError(f"malformed process progress segments: {path}")
    valid: list[dict[str, Any]] = []
    cursor = initial_cursor
    recovery_events = list(progress.get("recovery_events", []))
    for index, segment in enumerate(segments):
        if (
            not isinstance(segment, dict)
            or segment.get("index") != index
            or segment.get("cursor_start") != cursor
            or not isinstance(segment.get("cursor_end"), int)
            or segment["cursor_end"] < cursor
            or (segment["cursor_end"] == cursor and (index != 0 or segment.get("docs_in") != 0))
            or not isinstance(segment.get("outputs"), list)
        ):
            invalid_from = index
            reason = f"malformed segment {index}"
            break
        problems = verify_files(work, segment["outputs"])
        if problems:
            invalid_from = index
            reason = "; ".join(problems)
            break
        valid.append(segment)
        cursor = segment["cursor_end"]
    else:
        invalid_from = len(segments)
        reason = ""
    if invalid_from < len(segments):
        quarantined: list[dict[str, str]] = []
        for segment in segments[invalid_from:]:
            if isinstance(segment, dict) and isinstance(segment.get("outputs"), list):
                quarantined.extend(_quarantine_outputs(work, segment["outputs"], reason))
        recovery_events.append(
            {
                "invalid_segment_index": invalid_from,
                "reason": reason,
                "quarantined_outputs": quarantined,
            }
        )
        progress["segments"] = valid
        progress["next_cursor"] = cursor
        progress["recovery_events"] = recovery_events
        write_json_atomic(path, progress)
    return valid, cursor, recovery_events


def _reject_row(
    record: RawRecord,
    did: str,
    source_id: str,
    stage: str,
    reasons: list[str],
    kind: str | None,
    **extra: Any,
) -> dict[str, Any]:
    return {
        "doc_id": did,
        "source_id": source_id,
        "source_key": record.key,
        "stage": stage,
        "reasons": reasons,
        "n_chars": len(record.text),
        "n_bytes": len(record.text.encode("utf-8", "surrogatepass")),
        "kind": kind,
        "benchmarks": extra.get("benchmarks", []),
        "audit_text": extra.get("audit_text"),
    }


def process_unit(
    source: Source, unit: UnitSpec, raw_dir: Path, manifest: dict[str, Any], clone: Path | None
) -> dict[str, Any]:
    """Process a logical input unit as immutable, restartable time-bounded segments."""
    settings: Settings = _STATE["settings"]
    if not source.admissible:
        raise PermissionError(
            f"{source.source_id}: processing refused for license state {source.license_state}"
        )
    if settings.verified_sources is None or source.source_id not in settings.verified_sources:
        raise RuntimeError(f"{source.source_id}: acquired-source integrity preflight is required")
    minhasher: MinHasher = _STATE["minhasher"]
    index: ContaminationIndex = _STATE["index"]
    exclusion_identity: str = _STATE["exclusion_index_identity"]
    layout, build = settings.layout, settings.build_id
    work = layout.work(build)
    started = time.monotonic()
    input_identity = _raw_input_identity(source, unit, manifest)
    adapter = make_adapter(source, raw_dir, manifest, clone)
    initial_cursor = unit.row_start
    progress_path = (
        work / "progress" / "process" / source.source_id / f"{sha256_json(unit.unit_id)[:24]}.json"
    )
    segments, cursor, recovery_events = _load_progress(
        progress_path,
        work,
        settings.identity,
        input_identity,
        source,
        unit,
        exclusion_identity,
        initial_cursor,
    )
    progress_base = {
        "schema": "j1.process-progress.v1",
        "process_version": VERSION,
        "build_identity": settings.identity,
        "source_id": source.source_id,
        "source_revision": source.revision,
        "unit_id": unit.unit_id,
        "raw_file": unit.file,
        "raw_input_identity": input_identity,
        "exclusion_index_identity": exclusion_identity,
    }
    write_json_atomic(
        progress_path,
        {
            **progress_base,
            "next_cursor": cursor,
            "segments": segments,
            "recovery_events": recovery_events,
        },
    )

    def new_chunk(part_index: int, cursor_start: int) -> dict[str, Any]:
        unit_key = sha256_json(unit.unit_id)[:24]
        directory = work / "segments" / source.source_id / unit_key
        stem = f"part-{part_index:06d}"
        candidate_path = directory / f"{stem}.candidates.parquet"
        reject_path = directory / f"{stem}.rejects.parquet"
        signature_path = directory / f"{stem}.signatures.npy"
        return {
            "index": part_index,
            "cursor_start": cursor_start,
            "started": time.monotonic(),
            "candidates_path": candidate_path,
            "rejects_path": reject_path,
            "signatures_path": signature_path,
            "candidates": ShardWriter(candidate_path, CANDIDATE_SCHEMA),
            "rejects": ShardWriter(reject_path, REJECT_SCHEMA),
            "signatures": SignatureSink(signature_path, minhasher.config.num_perm),
            "stages": Counter(),
            "reasons": Counter(),
            "redactions": Counter(),
            "kinds": Counter(),
            "benchmarks": Counter(),
            "short_exact_benchmarks": Counter(),
            "bytes_by_stage": Counter(),
            "bytes_in": 0,
            "chars_in": 0,
            "docs_in": 0,
            "slow": [],
            "accepted_ids": hashlib.sha256(),
            "rejected_ids": hashlib.sha256(),
            "accepted_ids_sample": [],
            "rejected_ids_sample": [],
        }

    def output_entry(info: dict[str, Any], path: Path) -> dict[str, Any]:
        return {**info, "file": path.relative_to(work).as_posix()}

    def commit_chunk(chunk: dict[str, Any], cursor_end: int) -> dict[str, Any]:
        candidate_info = chunk["candidates"].close()
        reject_info = chunk["rejects"].close()
        signature_info = chunk["signatures"].close()
        outputs = [
            output_entry(candidate_info, chunk["candidates_path"]),
            output_entry(reject_info, chunk["rejects_path"]),
            output_entry(signature_info, chunk["signatures_path"]),
        ]
        if candidate_info["rows"] != signature_info["rows"]:
            raise RuntimeError(f"{unit.unit_id}: candidate and signature row counts differ")
        if chunk["docs_in"] != sum(chunk["stages"].values()):
            raise RuntimeError(f"{unit.unit_id}: segment accounting does not cover every document")
        segment = {
            "index": chunk["index"],
            "cursor_start": chunk["cursor_start"],
            "cursor_end": cursor_end,
            "docs_in": chunk["docs_in"],
            "bytes_in": chunk["bytes_in"],
            "chars_in": chunk["chars_in"],
            "stages": dict(chunk["stages"]),
            "bytes_by_stage": dict(chunk["bytes_by_stage"]),
            "reasons": dict(chunk["reasons"]),
            "redactions": dict(chunk["redactions"]),
            "kinds": dict(chunk["kinds"]),
            "contamination_benchmarks": dict(chunk["benchmarks"]),
            "contamination_short_exact_benchmarks": dict(chunk["short_exact_benchmarks"]),
            "accepted_ids_sha256": chunk["accepted_ids"].hexdigest(),
            "rejected_ids_sha256": chunk["rejected_ids"].hexdigest(),
            "accepted_ids_sample": chunk["accepted_ids_sample"],
            "rejected_ids_sample": chunk["rejected_ids_sample"],
            "outputs": outputs,
            "seconds": round(time.monotonic() - chunk["started"], 2),
            "slow_documents": chunk["slow"],
        }
        segments.append(segment)
        write_json_atomic(
            progress_path,
            {
                **progress_base,
                "next_cursor": cursor_end,
                "segments": segments,
                "recovery_events": recovery_events,
            },
        )
        return segment

    chunk = new_chunk(len(segments), cursor)
    try:
        for record in adapter.read(unit, start_at=cursor):
            doc_started = time.perf_counter()
            cursor += 1
            chunk["docs_in"] += 1
            did = doc_id(source.source_id, record.key)
            n_bytes = len(record.text.encode("utf-8", "surrogatepass"))
            chunk["bytes_in"] += n_bytes
            chunk["chars_in"] += len(record.text)

            row: dict[str, Any] | None = None
            if record.skip_reason:
                row = _reject_row(
                    record,
                    did,
                    source.source_id,
                    record.skip_reason.split(".")[0],
                    [record.skip_reason],
                    None,
                )
            else:
                stored = normalize.normalize_stored(record.text)
                if not stored.text.strip():
                    row = _reject_row(record, did, source.source_id, "extraction", ["extraction.empty"], None)
            if row is None:
                scrubbed = privacy.scrub(
                    stored.text, redact_phones=(record.kind or source.source_kind) != "code"
                )
                if scrubbed.reject_reasons:
                    row = _reject_row(record, did, source.source_id, "privacy", scrubbed.reject_reasons, None)
            if row is not None:
                chunk["stages"][row["stage"]] += 1
                chunk["bytes_by_stage"][row["stage"]] += n_bytes
                chunk["reasons"].update(row["reasons"])
                chunk["rejects"].append(row)
                chunk["rejected_ids"].update((did + "\n").encode("utf-8"))
                if len(chunk["rejected_ids_sample"]) < 16:
                    chunk["rejected_ids_sample"].append(did)
            else:
                text = scrubbed.text
                sig_words = normalize.signature_words(text)
                sig_hashes = word_hashes(sig_words)
                sig_lengths = np.fromiter((len(w) for w in sig_words), dtype=np.int64, count=len(sig_words))
                result = quality.evaluate(
                    text,
                    sig_hashes,
                    sig_lengths,
                    settings.quality,
                    record.kind or source.source_kind,
                    record.path,
                    curated=bool(source.options.get("curated")),
                )
                hit = index.match(sig_hashes, text)
                if hit.contaminated:
                    chunk["benchmarks"].update(hit.benchmarks + [f"canary:{c[:8]}" for c in hit.canaries])
                    chunk["short_exact_benchmarks"].update(hit.short_exact_benchmarks)
                    codes = [f"contamination.{b}" for b in hit.benchmarks] + (
                        ["contamination.canary"] if hit.canaries else []
                    )
                    row = _reject_row(
                        record,
                        did,
                        source.source_id,
                        "contamination",
                        codes,
                        result.kind,
                        benchmarks=hit.benchmarks,
                    )
                elif result.reasons:
                    keep_text = unit_interval("j1.audit.v1", did) < settings.audit_reject_text_rate
                    row = _reject_row(
                        record,
                        did,
                        source.source_id,
                        "quality",
                        result.reasons,
                        result.kind,
                        audit_text=text if keep_text else None,
                    )
                if row is not None:
                    chunk["stages"][row["stage"]] += 1
                    chunk["bytes_by_stage"][row["stage"]] += n_bytes
                    chunk["reasons"].update(row["reasons"])
                    chunk["rejects"].append(row)
                    chunk["rejected_ids"].update((did + "\n").encode("utf-8"))
                    if len(chunk["rejected_ids_sample"]) < 16:
                        chunk["rejected_ids_sample"].append(did)
                else:
                    chunk["signatures"].append(minhasher.signature(minhasher.shingles(sig_hashes)))
                    chunk["redactions"].update(scrubbed.redactions)
                    chunk["kinds"][result.kind] += 1
                    us, uk = domains.spelling_counts(sig_hashes)
                    chunk["candidates"].append(
                        {
                            "doc_id": did,
                            "source_id": source.source_id,
                            "source_key": record.key,
                            "group_key": record.group,
                            "text": text,
                            "sha256": sha256_text(text),
                            "norm_sha256": sha256_text(" ".join(sig_words)),
                            "n_chars": len(text),
                            "n_bytes": len(text.encode("utf-8")),
                            "n_words": len(sig_words),
                            "kind": result.kind,
                            "domain": domains.domain_label(sig_hashes, result.kind),
                            "formats": domains.formats_present(text),
                            "us_spellings": us,
                            "uk_spellings": uk,
                            "english_fw_frac": result.features["english_function_word_frac"],
                            "code_line_frac": result.features["code_line_frac"],
                            "redactions": json.dumps(dict(scrubbed.redactions), sort_keys=True),
                            "meta": json.dumps(record.meta, sort_keys=True, default=str),
                        }
                    )
                    chunk["stages"]["candidate"] += 1
                    chunk["bytes_by_stage"]["candidate"] += n_bytes
                    chunk["accepted_ids"].update((did + "\n").encode("utf-8"))
                    if len(chunk["accepted_ids_sample"]) < 16:
                        chunk["accepted_ids_sample"].append(did)
            elapsed = time.perf_counter() - doc_started
            if elapsed > SLOW_DOCUMENT_SECONDS and len(chunk["slow"]) < 20:
                chunk["slow"].append({"doc_id": did, "seconds": round(elapsed, 2), "bytes": n_bytes})
            if time.monotonic() - chunk["started"] >= CHECKPOINT_SECONDS:
                commit_chunk(chunk, cursor)
                chunk = new_chunk(len(segments), cursor)
        if chunk["docs_in"] or not segments:
            commit_chunk(chunk, cursor)
        else:
            chunk["candidates"].abort()
            chunk["rejects"].abort()
            chunk["signatures"].abort()
    except BaseException:
        chunk["candidates"].abort()
        chunk["rejects"].abort()
        chunk["signatures"].abort()
        raise

    def total_counter(name: str) -> Counter[str]:
        result: Counter[str] = Counter()
        for segment in segments:
            result.update(segment[name])
        return result

    outputs = [entry for segment in segments for entry in segment["outputs"]]
    stages = total_counter("stages")
    if stages["candidate"] != sum(entry["rows"] for entry in outputs[::3]):
        raise RuntimeError(f"{unit.unit_id}: candidate output rows disagree with processing journal")
    from datetime import UTC, datetime

    marker = {
        "schema": "j1.unit-marker.v2",
        "process_version": VERSION,
        "build_identity": settings.identity,
        "exclusion_index_identity": exclusion_identity,
        "source_id": source.source_id,
        "source_revision": source.revision,
        "unit_id": unit.unit_id,
        "raw_file": unit.file,
        "raw_input_identity": input_identity,
        "row_range": [unit.row_start, unit.row_end],
        "docs_in": sum(segment["docs_in"] for segment in segments),
        "bytes_in": sum(segment["bytes_in"] for segment in segments),
        "chars_in": sum(segment["chars_in"] for segment in segments),
        "candidate_rows": stages["candidate"],
        "stages": dict(stages),
        "bytes_by_stage": dict(total_counter("bytes_by_stage")),
        "reasons": dict(total_counter("reasons")),
        "redactions": dict(total_counter("redactions")),
        "kinds": dict(total_counter("kinds")),
        "contamination_benchmarks": dict(total_counter("contamination_benchmarks")),
        "contamination_short_exact_benchmarks": dict(total_counter("contamination_short_exact_benchmarks")),
        "segments": segments,
        "outputs": outputs,
        "seconds": round(time.monotonic() - started, 2),
        "slow_documents": [item for segment in segments for item in segment["slow_documents"]][:20],
        "recovery_events": recovery_events,
        "completed_at_utc": datetime.now(UTC).isoformat(),
    }
    if marker["docs_in"] != sum(stages.values()):
        raise RuntimeError(f"{unit.unit_id}: stage accounting does not cover every document")
    write_json_atomic(marker_path(layout, build, unit), marker)
    return marker
