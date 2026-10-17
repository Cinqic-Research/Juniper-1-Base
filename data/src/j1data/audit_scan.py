"""Read-only, resumable contamination scan for an accepted Parquet corpus.

This audit is deliberately separate from ``j1data verify``: it reads the
accepted manifest, shards, and exclusion index without preparing or rewriting
the historical build.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import hashlib
import json
import os
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from j1data import faults
from j1data.contamination import ContaminationIndex
from j1data.hashing import sha256_file
from j1data.ngrams import word_hashes
from j1data.normalize import signature_words

SCHEMA = "j1.accepted-contamination-scan.v1"
SCANNER_VERSION = "j1.audit-scan.v1"
DEFAULT_BATCH_SIZE = 16
MAX_WORKERS = 4
_WORKER_INDEX: ContaminationIndex | None = None


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(_canonical_json(value) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        Path(temporary_name).replace(path)
        directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            Path(temporary_name).unlink()
        raise


def _safe_shard_path(accepted_dir: Path, value: object) -> Path:
    if not isinstance(value, str):
        raise ValueError("manifest shard path is not text")
    relative = PurePosixPath(value)
    if (
        relative.is_absolute()
        or not relative.parts
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise ValueError(f"unsafe manifest shard path: {value!r}")
    root = accepted_dir.resolve(strict=True)
    candidate = accepted_dir
    for part in relative.parts:
        candidate = candidate / part
        if candidate.is_symlink():
            raise ValueError(f"manifest shard path contains a symlink: {value!r}")
    path = candidate.resolve(strict=True)
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f"manifest shard does not resolve to a file below accepted root: {value!r}")
    return path


def scan_one_shard(
    accepted_dir: Path,
    entry: dict[str, Any],
    index: ContaminationIndex,
    scan_identity: str,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> dict[str, Any]:
    """Re-hash and scan one immutable shard, retaining no source text."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    relative = entry.get("file")
    path = _safe_shard_path(accepted_dir, relative)
    expected_sha256 = entry.get("sha256")
    if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
        raise ValueError(f"invalid expected SHA-256 for {relative!r}")
    actual_sha256 = sha256_file(path)
    if actual_sha256 != expected_sha256:
        raise ValueError(f"accepted shard hash differs from manifest: {relative!r}")

    rows = 0
    hit_documents = 0
    benchmark_hits: dict[str, int] = {}
    canary_hit_documents = 0
    hit_id_hashes: list[str] = []
    try:
        parquet = pq.ParquetFile(path)
        if not {"doc_id", "text"}.issubset(parquet.schema_arrow.names):
            raise ValueError(f"accepted shard lacks doc_id/text columns: {relative!r}")
        for batch in parquet.iter_batches(columns=["doc_id", "text"], batch_size=batch_size):
            ids = batch.column(0).to_pylist()
            texts = batch.column(1).to_pylist()
            for doc_id, text in zip(ids, texts, strict=True):
                if not isinstance(doc_id, str) or not isinstance(text, str):
                    raise ValueError(f"accepted shard has a non-text id or text value: {relative!r}")
                result = index.match(word_hashes(signature_words(text)), text)
                rows += 1
                if result.contaminated:
                    hit_documents += 1
                    if len(hit_id_hashes) < 16:
                        hit_id_hashes.append(hashlib.sha256(doc_id.encode("utf-8")).hexdigest())
                    for benchmark in result.benchmarks:
                        benchmark_hits[benchmark] = benchmark_hits.get(benchmark, 0) + 1
                    if result.canaries:
                        canary_hit_documents += 1
    except (OSError, pa.ArrowException) as error:
        raise ValueError(f"cannot read accepted shard {relative!r}: {error}") from error

    expected_rows = entry.get("rows")
    if not isinstance(expected_rows, int) or rows != expected_rows:
        raise ValueError(f"accepted shard row count differs from manifest: {relative!r}")
    return {
        "schema": SCHEMA,
        "scan_identity": scan_identity,
        "file": relative,
        "source_id": entry.get("source_id"),
        "split": entry.get("split"),
        "input_sha256": actual_sha256,
        "rows": rows,
        "hit_documents": hit_documents,
        "benchmark_hit_documents": dict(sorted(benchmark_hits.items())),
        "canary_hit_documents": canary_hit_documents,
        "hit_doc_id_sha256_sample": hit_id_hashes[:16],
    }


def _initialize_worker(index_dir: str) -> None:
    global _WORKER_INDEX
    _WORKER_INDEX = ContaminationIndex.load(Path(index_dir))


def _scan_worker(task: tuple[str, dict[str, Any], str]) -> dict[str, Any]:
    accepted_dir_text, entry, scan_identity = task
    if _WORKER_INDEX is None:
        raise RuntimeError("contamination index was not loaded in worker")
    return scan_one_shard(Path(accepted_dir_text), entry, _WORKER_INDEX, scan_identity)


def _valid_partial(
    path: Path,
    entry: dict[str, Any],
    scan_identity: str,
    expected_report_sha256: str | None,
) -> dict[str, Any] | None:
    if expected_report_sha256 is None or not path.is_file():
        return None
    try:
        if sha256_file(path) != expected_report_sha256:
            return None
    except OSError:
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    benchmark_hits = value.get("benchmark_hit_documents") if isinstance(value, dict) else None
    id_sample = value.get("hit_doc_id_sha256_sample") if isinstance(value, dict) else None
    canary_hits = value.get("canary_hit_documents") if isinstance(value, dict) else None
    hit_documents = value.get("hit_documents") if isinstance(value, dict) else None
    if (
        not isinstance(value, dict)
        or value.get("schema") != SCHEMA
        or value.get("scan_identity") != scan_identity
        or value.get("file") != entry.get("file")
        or value.get("source_id") != entry.get("source_id")
        or value.get("split") != entry.get("split")
        or value.get("input_sha256") != entry.get("sha256")
        or value.get("rows") != entry.get("rows")
        or not isinstance(hit_documents, int)
        or not 0 <= hit_documents <= int(entry.get("rows", -1))
        or not isinstance(benchmark_hits, dict)
        or not all(
            isinstance(name, str) and isinstance(count, int) and 0 <= count <= hit_documents
            for name, count in benchmark_hits.items()
        )
        or not isinstance(canary_hits, int)
        or not 0 <= canary_hits <= hit_documents
        or not isinstance(id_sample, list)
        or len(id_sample) > 16
        or not all(
            isinstance(digest, str)
            and len(digest) == 64
            and all(char in "0123456789abcdef" for char in digest)
            for digest in id_sample
        )
    ):
        return None
    return value


def _read_progress(path: Path, scan_identity: str, expected_files: set[str]) -> dict[str, str]:
    """Return digest-bound completed reports; corrupt or stale journals authorize no reuse."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    reports = value.get("completed_reports") if isinstance(value, dict) else None
    if (
        not isinstance(value, dict)
        or value.get("schema") != SCHEMA
        or value.get("scan_identity") != scan_identity
        or not isinstance(reports, dict)
        or not set(reports).issubset(expected_files)
        or not all(
            isinstance(name, str)
            and isinstance(digest, str)
            and len(digest) == 64
            and all(char in "0123456789abcdef" for char in digest)
            for name, digest in reports.items()
        )
    ):
        return {}
    return reports


def _validate_manifest(accepted_dir: Path, manifest: dict[str, Any], build_id: str) -> list[dict[str, Any]]:
    if manifest.get("schema") != "j1.accepted-manifest.v1" or manifest.get("build_id") != build_id:
        raise ValueError("accepted manifest does not identify the requested build")
    raw_entries = manifest.get("shards")
    if (
        not isinstance(raw_entries, list)
        or not raw_entries
        or not all(isinstance(entry, dict) for entry in raw_entries)
    ):
        raise ValueError("accepted manifest has no valid shard inventory")

    entries: list[dict[str, Any]] = []
    seen: set[str] = set()
    rows_by_split: dict[str, int] = {}
    for raw in raw_entries:
        relative = raw.get("file")
        path = _safe_shard_path(accepted_dir, relative)
        if not isinstance(relative, str) or relative in seen:
            raise ValueError(f"accepted manifest has a duplicate shard path: {relative!r}")
        seen.add(relative)
        digest = raw.get("sha256")
        rows = raw.get("rows")
        split = raw.get("split")
        source_id = raw.get("source_id")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(char not in "0123456789abcdef" for char in digest)
            or not isinstance(rows, int)
            or isinstance(rows, bool)
            or rows < 0
            or not isinstance(split, str)
            or split not in {"train", "validation", "test"}
            or not isinstance(source_id, str)
            or not source_id
        ):
            raise ValueError(f"accepted manifest has invalid shard metadata: {relative!r}")
        # Resolve now so inventory validation cannot silently accept an unsafe path.
        if not path.is_file():
            raise ValueError(f"accepted manifest shard is not a regular file: {relative!r}")
        entry = dict(raw)
        entries.append(entry)
        rows_by_split[split] = rows_by_split.get(split, 0) + rows

    accounting = manifest.get("accepted_by_split")
    if not isinstance(accounting, dict) or not all(
        isinstance(split, str)
        and split in {"train", "validation", "test"}
        and isinstance(rows, int)
        and not isinstance(rows, bool)
        and rows >= 0
        for split, rows in accounting.items()
    ):
        raise ValueError("accepted manifest has invalid accepted_by_split accounting")
    recorded = {split: rows for split, rows in accounting.items() if rows}
    actual = {split: rows for split, rows in rows_by_split.items() if rows}
    if recorded != actual:
        raise ValueError("accepted manifest split accounting differs from shard inventory")

    parquet_inventory = {
        path.relative_to(accepted_dir).as_posix()
        for path in accepted_dir.rglob("*.parquet")
        if path.is_file() or path.is_symlink()
    }
    if parquet_inventory != seen:
        raise ValueError("on-disk Parquet inventory differs from accepted manifest")
    return entries


def _validate_index_metadata(
    index: ContaminationIndex, metadata: dict[str, Any]
) -> dict[str, dict[str, int]]:
    if metadata.get("index_identity") != index.identity():
        raise ValueError("exclusion index sidecar identity differs from loaded arrays")
    if metadata.get("source_fingerprint") != index.source_fingerprint:
        raise ValueError("exclusion index sidecar source fingerprint differs from loaded arrays")
    total_ngrams = metadata.get("total_ngrams")
    total_items = metadata.get("total_items")
    if (
        not isinstance(total_ngrams, int)
        or isinstance(total_ngrams, bool)
        or total_ngrams < 0
        or not isinstance(total_items, int)
        or isinstance(total_items, bool)
        or total_items < 0
        or metadata.get("ngram") != index.ngram
        or metadata.get("min_coverage") != index.min_coverage
        or metadata.get("min_informative_words") != index.min_informative_words
        or total_ngrams != int(index.hashes.size)
        # The builder (exclusions.build_index_dir) records total_items as the long n-gram
        # items only; short-exact items are separate entries reconciled per benchmark below.
        or total_items != int(index.item_ngrams.size)
        or metadata.get("canaries") != index.canaries
    ):
        raise ValueError("exclusion index sidecar settings or totals differ from loaded arrays")
    benchmarks = metadata.get("benchmarks")
    if not isinstance(benchmarks, dict) or (
        sum(int(b.get("long_ngram_items", -1)) for b in benchmarks.values()) != int(index.item_ngrams.size)
        or sum(int(b.get("short_exact_items", -1)) for b in benchmarks.values())
        != int(index.short_hashes.size)
    ):
        raise ValueError("exclusion index per-benchmark item counts differ from loaded arrays")
    short_settings = metadata.get("short_exact")
    if not isinstance(short_settings, dict):
        raise ValueError("exclusion index sidecar has invalid short-exact settings")
    short_min_words = short_settings.get("min_words")
    short_min_chars = short_settings.get("min_chars")
    short_min_informative_words = short_settings.get("min_informative_words")
    if (
        not isinstance(short_min_words, int)
        or isinstance(short_min_words, bool)
        or not isinstance(short_min_chars, int)
        or isinstance(short_min_chars, bool)
        or not isinstance(short_min_informative_words, int)
        or isinstance(short_min_informative_words, bool)
        or not 1 <= short_min_words < index.ngram
        or short_min_chars < 1
        or short_min_informative_words < 1
    ):
        raise ValueError("exclusion index sidecar has invalid short-exact settings")
    benchmarks = metadata.get("benchmarks")
    if not isinstance(benchmarks, dict) or set(benchmarks) != set(index.benchmark_names):
        raise ValueError("exclusion index sidecar benchmark names differ from loaded arrays")
    if (
        len(index.item_benchmark) != len(index.item_ngrams)
        or len(index.short_lengths) != len(index.short_hashes)
        or len(index.short_lengths) != len(index.short_benchmarks)
        or (index.item_benchmark.size and int(index.item_benchmark.max()) >= len(index.benchmark_names))
        or (index.short_benchmarks.size and int(index.short_benchmarks.max()) >= len(index.benchmark_names))
    ):
        raise ValueError("exclusion index arrays have inconsistent benchmark coverage dimensions")

    long_counts = dict.fromkeys(index.benchmark_names, 0)
    ngram_counts = dict.fromkeys(index.benchmark_names, 0)
    for benchmark, ngrams in zip(index.item_benchmark.tolist(), index.item_ngrams.tolist(), strict=True):
        name = index.benchmark_names[int(benchmark)]
        long_counts[name] += 1
        ngram_counts[name] += int(ngrams)
    short_counts = dict.fromkeys(index.benchmark_names, 0)
    for benchmark in index.short_benchmarks.tolist():
        short_counts[index.benchmark_names[int(benchmark)]] += 1

    for name in index.benchmark_names:
        stats = benchmarks[name]
        if not isinstance(stats, dict):
            raise ValueError(f"exclusion index sidecar has invalid benchmark stats: {name}")
        total = stats.get("items")
        long = stats.get("long_ngram_items")
        short = stats.get("short_exact_items")
        unmatchable = stats.get("unmatchable_items")
        ngrams = stats.get("ngrams")
        if (
            not all(
                isinstance(value, int) and not isinstance(value, bool) and value >= 0
                for value in (total, long, short, unmatchable, ngrams)
            )
            or long != long_counts[name]
            or short != short_counts[name]
            or ngrams != ngram_counts[name]
            or total != long + short + unmatchable
        ):
            raise ValueError(f"exclusion index sidecar coverage differs from loaded arrays: {name}")
    return {name: dict(value) for name, value in benchmarks.items()}


def _aggregate(results: list[dict[str, Any]], metadata: dict[str, Any]) -> dict[str, Any]:
    benchmark_hits: dict[str, int] = {}
    split_totals: dict[str, dict[str, Any]] = {}
    for result in results:
        split = result["split"]
        totals = split_totals.setdefault(
            split,
            {
                "shards_scanned": 0,
                "rows_scanned": 0,
                "hit_documents": 0,
                "canary_hit_documents": 0,
                "benchmark_hit_documents": {},
            },
        )
        totals["shards_scanned"] += 1
        totals["rows_scanned"] += result["rows"]
        totals["hit_documents"] += result["hit_documents"]
        totals["canary_hit_documents"] += result["canary_hit_documents"]
        for benchmark, count in result["benchmark_hit_documents"].items():
            benchmark_hits[benchmark] = benchmark_hits.get(benchmark, 0) + count
            split_benchmarks = totals["benchmark_hit_documents"]
            split_benchmarks[benchmark] = split_benchmarks.get(benchmark, 0) + count
    coverage: dict[str, dict[str, Any]] = {}
    for benchmark, stats in metadata["index_item_coverage"].items():
        total = stats["items"]
        matchable = stats["long_ngram_items"] + stats["short_exact_items"]
        coverage[benchmark] = {
            "items": total,
            "matchable_items": matchable,
            "unmatchable_items": stats["unmatchable_items"],
            "matchable_fraction": round(matchable / total, 6) if total else 1.0,
        }
    return {
        **metadata,
        "status": "complete",
        "shards_scanned": len(results),
        "rows_scanned": sum(result["rows"] for result in results),
        "hit_documents": sum(result["hit_documents"] for result in results),
        "benchmark_hit_documents": dict(sorted(benchmark_hits.items())),
        "canary_hit_documents": sum(result["canary_hit_documents"] for result in results),
        "splits": {name: split_totals[name] for name in sorted(split_totals)},
        "index_item_coverage": coverage,
        "hit_doc_id_sha256_sample": [
            digest for result in results for digest in result["hit_doc_id_sha256_sample"]
        ][:64],
        "interpretation": (
            "Observed matcher hits are not an evaluation-clean certificate when protected items are "
            "unmatchable or paraphrased content is outside the matcher definition."
        ),
    }


def run_scan(data_root: Path, build_id: str, index_dir: Path, workers: int = 2) -> Path:
    if not 1 <= workers <= MAX_WORKERS:
        raise ValueError(f"workers must be between 1 and {MAX_WORKERS}")
    accepted_dir = data_root / "accepted" / build_id
    manifest_path = accepted_dir / "manifest.json"
    manifest_sha256 = sha256_file(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("accepted manifest is not a JSON object")
    entries = _validate_manifest(accepted_dir, manifest, build_id)
    index = ContaminationIndex.load(index_dir)
    index_identity = index.identity()
    index_metadata_path = index_dir / "manifest.json"
    index_metadata_sha256 = sha256_file(index_metadata_path)
    index_metadata = json.loads(index_metadata_path.read_text(encoding="utf-8"))
    if not isinstance(index_metadata, dict):
        raise ValueError("exclusion index sidecar is not a JSON object")
    item_coverage = _validate_index_metadata(index, index_metadata)
    scanner_sha256 = sha256_file(Path(__file__).resolve())
    material = {
        "schema": SCHEMA,
        "scanner_version": SCANNER_VERSION,
        "scanner_sha256": scanner_sha256,
        "build_id": build_id,
        "build_identity": manifest.get("build_identity"),
        "accepted_manifest_sha256": manifest_sha256,
        "index_identity": index_identity,
        "index_manifest_sha256": index_metadata_sha256,
    }
    scan_identity = hashlib.sha256(_canonical_json(material)).hexdigest()
    output_dir = data_root / "audits" / f"contamination-{build_id}-{scan_identity[:16]}"
    if output_dir.resolve().is_relative_to(accepted_dir.resolve()):
        raise ValueError("audit output must not be written inside the accepted corpus")
    partial_dir = output_dir / "shards"
    partial_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        **material,
        "scan_identity": scan_identity,
        "index_path": str(index_dir.resolve()),
        "index_source_fingerprint": index.source_fingerprint,
        "accepted_dir": str(accepted_dir.resolve()),
        "expected_shards": len(entries),
        "expected_rows": sum(entry["rows"] for entry in entries),
        "workers": workers,
        "index_item_coverage": item_coverage,
    }
    results_by_file: dict[str, dict[str, Any]] = {}
    pending: list[dict[str, Any]] = []
    expected_files = {str(entry["file"]) for entry in entries}
    report_digests = _read_progress(output_dir / "progress.json", scan_identity, expected_files)
    for entry in entries:
        relative = str(entry["file"])
        result_path = partial_dir / f"{hashlib.sha256(relative.encode('utf-8')).hexdigest()}.json"
        cached = _valid_partial(result_path, entry, scan_identity, report_digests.get(relative))
        if cached is not None:
            source_path = _safe_shard_path(accepted_dir, relative)
            if sha256_file(source_path) != entry.get("sha256"):
                raise ValueError(f"accepted shard hash differs from manifest: {relative!r}")
        if cached is None:
            pending.append(entry)
        else:
            results_by_file[relative] = cached

    progress = {
        "schema": SCHEMA,
        "scan_identity": scan_identity,
        "completed_reports": {
            relative: report_digests[relative]
            for relative in sorted(results_by_file)
            if relative in report_digests
        },
    }
    _atomic_json(output_dir / "progress.json", progress)
    if pending:

        def record_result(entry: dict[str, Any], result: dict[str, Any]) -> None:
            relative = str(entry["file"])
            result_path = partial_dir / f"{hashlib.sha256(relative.encode('utf-8')).hexdigest()}.json"
            _atomic_json(result_path, result)
            report_digests[relative] = sha256_file(result_path)
            results_by_file[relative] = result
            progress["completed_reports"] = {name: report_digests[name] for name in sorted(results_by_file)}
            _atomic_json(output_dir / "progress.json", progress)
            faults.point("audit_scan.shard.commit")

        if workers == 1:
            for entry in pending:
                record_result(
                    entry,
                    scan_one_shard(accepted_dir, entry, index, scan_identity),
                )
        else:
            tasks = [(str(accepted_dir), entry, scan_identity) for entry in pending]
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=workers,
                initializer=_initialize_worker,
                initargs=(str(index_dir),),
            ) as pool:
                futures = {pool.submit(_scan_worker, task): task[1] for task in tasks}
                for future in concurrent.futures.as_completed(futures):
                    record_result(futures[future], future.result())

    results = [results_by_file[str(entry["file"])] for entry in entries]
    summary = _aggregate(results, metadata)
    if summary["rows_scanned"] != metadata["expected_rows"]:
        raise ValueError("completed scan row total differs from accepted manifest")
    _atomic_json(output_dir / "result.json", summary)
    return output_dir / "result.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(os.environ["J1_DATA_ROOT"]) if "J1_DATA_ROOT" in os.environ else None,
    )
    parser.add_argument("--build-id", default="j1-base-v0.1")
    parser.add_argument("--index", type=Path, required=True, help="immutable exclusion-index directory")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if args.data_root is None:
        parser.error("--data-root or J1_DATA_ROOT is required")
    result = run_scan(args.data_root, args.build_id, args.index, args.workers)
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
