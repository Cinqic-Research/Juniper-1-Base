import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from j1data.audit_scan import run_scan, scan_one_shard
from j1data.contamination import ContaminationIndex
from j1data.hashing import sha256_file

SYNTHETIC_ITEM = (
    "The brass lantern in a quiet station blinks twice whenever the northbound train "
    "is delayed beyond midnight."
)
SHORT_ITEM = "The brass lantern shines above a quiet train station after midnight."


def _fixture_index(item: str = SYNTHETIC_ITEM) -> ContaminationIndex:
    index, _ = ContaminationIndex.build({"synthetic-fixture": [item]}, [], 13, 0.5, 6)
    return index


def _write_shard(tmp_path, texts: list[str]) -> tuple[object, dict[str, object]]:  # type: ignore[no-untyped-def]
    accepted = tmp_path / "accepted" / "build"
    accepted.mkdir(parents=True)
    path = accepted / "fixture.parquet"
    pq.write_table(
        pa.table(
            {
                "doc_id": [f"doc-{index}" for index in range(len(texts))],
                "text": texts,
            }
        ),
        path,
    )
    entry: dict[str, object] = {
        "file": "fixture.parquet",
        "rows": len(texts),
        "sha256": sha256_file(path),
        "source_id": "fixture",
        "split": "train",
    }
    return accepted, entry


def _write_scan_inputs(tmp_path: Path, texts: list[str], item: str = SYNTHETIC_ITEM) -> tuple[Path, Path]:
    accepted, entry = _write_shard(tmp_path, texts)
    entry["split"] = "train"
    manifest = {
        "schema": "j1.accepted-manifest.v1",
        "build_id": "build",
        "build_identity": "fixture-build-identity",
        "accepted_by_split": {"train": len(texts)},
        "shards": [entry],
    }
    (accepted / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    index, stats = ContaminationIndex.build({"synthetic-fixture": [item]}, [], 13, 0.5, 6)
    index_dir = tmp_path / "exclusions" / "fixture"
    identity = index.save(index_dir)
    index_manifest = {
        "schema": "j1.exclusion-manifest.v1",
        "index_identity": identity,
        "ngram": index.ngram,
        "min_coverage": index.min_coverage,
        "min_informative_words": index.min_informative_words,
        "short_exact": {"min_words": 6, "min_chars": 48, "min_informative_words": 5},
        "canaries": index.canaries,
        "source_fingerprint": index.source_fingerprint,
        "benchmarks": {name: {**stats[name]} for name in stats},
        "total_ngrams": int(index.hashes.size),
        "total_items": int(index.item_ngrams.size),  # as written by exclusions.build_index_dir
    }
    (index_dir / "manifest.json").write_text(json.dumps(index_manifest), encoding="utf-8")
    return tmp_path, index_dir


def test_read_only_scan_rehashes_and_finds_planted_match(tmp_path) -> None:  # type: ignore[no-untyped-def]
    accepted, entry = _write_shard(
        tmp_path,
        [f"Station notes: {SYNTHETIC_ITEM}", "Photosynthesis uses light."],
    )
    result = scan_one_shard(accepted, entry, _fixture_index(), "scan-id")

    assert result["rows"] == 2
    assert result["hit_documents"] == 1
    assert result["benchmark_hit_documents"] == {"synthetic-fixture": 1}
    assert result["hit_doc_id_sha256_sample"] == [hashlib.sha256(b"doc-0").hexdigest()]
    assert "Study guide" not in str(result)


def test_read_only_scan_rejects_manifest_hash_mismatch(tmp_path) -> None:  # type: ignore[no-untyped-def]
    accepted, entry = _write_shard(tmp_path, [SYNTHETIC_ITEM])
    entry["sha256"] = "0" * 64

    with pytest.raises(ValueError, match="hash differs"):
        scan_one_shard(accepted, entry, _fixture_index(), "scan-id")


@pytest.mark.parametrize("relative", ["../outside.parquet", "/tmp/outside.parquet"])
def test_read_only_scan_rejects_unsafe_manifest_paths(tmp_path, relative: str) -> None:  # type: ignore[no-untyped-def]
    accepted, entry = _write_shard(tmp_path, [SYNTHETIC_ITEM])
    entry["file"] = relative

    with pytest.raises(ValueError, match="unsafe manifest shard path"):
        scan_one_shard(accepted, entry, _fixture_index(), "scan-id")


def test_scan_matches_short_exact_items(tmp_path: Path) -> None:
    data_root, index_dir = _write_scan_inputs(tmp_path, [f"Station note: {SHORT_ITEM}"], SHORT_ITEM)

    result_path = run_scan(data_root, "build", index_dir, workers=1)
    report = json.loads(result_path.read_text(encoding="utf-8"))

    assert report["hit_documents"] == 1
    assert report["benchmark_hit_documents"] == {"synthetic-fixture": 1}
    assert report["splits"]["train"]["rows_scanned"] == 1


def test_scan_rejects_duplicate_shard_paths_and_split_accounting(tmp_path: Path) -> None:
    data_root, index_dir = _write_scan_inputs(tmp_path, ["ordinary document"])
    manifest_path = data_root / "accepted" / "build" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["shards"].append(manifest["shards"][0])
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate shard path"):
        run_scan(data_root, "build", index_dir, workers=1)

    manifest["shards"] = manifest["shards"][:1]
    manifest["accepted_by_split"] = {"train": 9}
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="split accounting"):
        run_scan(data_root, "build", index_dir, workers=1)


def test_scan_rejects_exclusion_sidecar_mismatch(tmp_path: Path) -> None:
    data_root, index_dir = _write_scan_inputs(tmp_path, ["ordinary document"])
    sidecar = index_dir / "manifest.json"
    manifest = json.loads(sidecar.read_text(encoding="utf-8"))
    manifest["index_identity"] = "0" * 64
    sidecar.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="sidecar identity"):
        run_scan(data_root, "build", index_dir, workers=1)

    index, stats = ContaminationIndex.build({"synthetic-fixture": [SYNTHETIC_ITEM]}, [], 13, 0.5, 6)
    manifest = {
        "schema": "j1.exclusion-manifest.v1",
        "index_identity": index.identity(),
        "ngram": index.ngram,
        "min_coverage": index.min_coverage,
        "min_informative_words": index.min_informative_words,
        "short_exact": {"min_words": 6, "min_chars": 48, "min_informative_words": 5},
        "canaries": index.canaries,
        "source_fingerprint": index.source_fingerprint,
        "benchmarks": {name: {**stats[name]} for name in stats},
        "total_ngrams": int(index.hashes.size),
        "total_items": int(index.item_ngrams.size),  # as written by exclusions.build_index_dir
    }
    manifest["benchmarks"]["synthetic-fixture"]["unmatchable_items"] += 1
    sidecar.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="coverage differs"):
        run_scan(data_root, "build", index_dir, workers=1)


def test_scan_rejects_unlisted_parquet_files(tmp_path: Path) -> None:
    data_root, index_dir = _write_scan_inputs(tmp_path, ["ordinary document"])
    accepted = data_root / "accepted" / "build"
    pq.write_table(pa.table({"doc_id": ["extra"], "text": ["extra"]}), accepted / "extra.parquet")

    with pytest.raises(ValueError, match="inventory differs"):
        run_scan(data_root, "build", index_dir, workers=1)


def test_scan_recomputes_tampered_partial_or_missing_progress(tmp_path: Path) -> None:
    data_root, index_dir = _write_scan_inputs(tmp_path, [SYNTHETIC_ITEM])
    result_path = run_scan(data_root, "build", index_dir, workers=1)
    report = json.loads(result_path.read_text(encoding="utf-8"))
    assert report["hit_documents"] == 1

    output_dir = result_path.parent
    partial = output_dir / "shards" / f"{hashlib.sha256(b'fixture.parquet').hexdigest()}.json"
    cached = json.loads(partial.read_text(encoding="utf-8"))
    cached["hit_documents"] = 0
    partial.write_text(json.dumps(cached), encoding="utf-8")
    rerun_path = run_scan(data_root, "build", index_dir, workers=1)
    assert json.loads(rerun_path.read_text(encoding="utf-8"))["hit_documents"] == 1

    cached = json.loads(partial.read_text(encoding="utf-8"))
    cached["hit_documents"] = 0
    partial.write_text(json.dumps(cached), encoding="utf-8")
    (output_dir / "progress.json").unlink()
    rerun_path = run_scan(data_root, "build", index_dir, workers=1)
    assert json.loads(rerun_path.read_text(encoding="utf-8"))["hit_documents"] == 1
