"""Finalize: deduplicated candidates to accepted, split shards.

1. Every retained candidate gets its split from its group key (``splits``).
2. A held-out index maps each word 13-gram of the validation and test
   documents to the single held-out document containing it (13-grams shared by
   several held-out documents are boilerplate and dropped).
3. A training document that covers at least ``leak_fraction`` of some held-out
   document's 13-grams, or whose own 13-grams are at least ``leak_fraction``
   held-out, is a cross-split near copy that escaped deduplication and is
   removed from training (reason ``split.leakage``).
4. Accepted documents are written to ``accepted/<build>/<split>/<source>/<unit>.parquet``
   with their split, duplicate-cluster size and provenance columns; a marker per
   unit lists every shard with SHA-256 and row count.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pyarrow as pa
import pyarrow.parquet as pq

from j1data import faults, splits
from j1data.dedup import UnitRef
from j1data.ngrams import ngram_hashes, word_hashes
from j1data.normalize import signature_words
from j1data.paths import DataLayout, safe_child
from j1data.pools import run_unordered
from j1data.process import CANDIDATE_SCHEMA
from j1data.shards import ShardWriter, remove_stale_temporaries, verify_files, write_json_atomic

VERSION = "j1.finalize.v1"
LEAK_NGRAM = 13

ACCEPTED_SCHEMA = pa.schema(
    [*CANDIDATE_SCHEMA, ("split", pa.string()), ("dup_cluster_size", pa.int32()), ("unit_id", pa.string())]
)


@dataclass(frozen=True)
class FinalizeSettings:
    layout: DataLayout
    build_id: str
    split_config: splits.SplitConfig
    leak_fraction: float
    identity: str


def _doc_ngrams(text: str) -> npt.NDArray[np.uint64]:
    return np.unique(ngram_hashes(word_hashes(signature_words(text)), LEAK_NGRAM))


def unit_splits(ref: UnitRef, keep: npt.NDArray[np.bool_], config: splits.SplitConfig) -> list[str | None]:
    groups = pq.read_table(ref.candidates, columns=["group_key"]).column("group_key").to_pylist()
    return [splits.assign(g, config) if k else None for g, k in zip(groups, keep, strict=True)]


def build_heldout_index(
    refs: list[UnitRef], keeps: dict[str, npt.NDArray[np.bool_]], config: splits.SplitConfig, directory: Path
) -> dict[str, Any]:
    grams: list[npt.NDArray[np.uint64]] = []
    owners: list[npt.NDArray[np.int64]] = []
    doc_ids: list[str] = []
    for ref in refs:
        if not ref.rows:
            continue
        assigned = unit_splits(ref, keeps[ref.candidates.as_posix()], config)
        rows = [i for i, s in enumerate(assigned) if s in ("validation", "test")]
        if not rows:
            continue
        table = pq.read_table(ref.candidates, columns=["doc_id", "text"]).take(rows)
        for did, text in zip(
            table.column("doc_id").to_pylist(), table.column("text").to_pylist(), strict=True
        ):
            hashes = _doc_ngrams(text)
            grams.append(hashes)
            owners.append(np.full(hashes.size, len(doc_ids), dtype=np.int64))
            doc_ids.append(did)
    all_grams = np.concatenate(grams) if grams else np.empty(0, dtype=np.uint64)
    all_owners = np.concatenate(owners) if owners else np.empty(0, dtype=np.int64)
    unique, first, counts = np.unique(all_grams, return_index=True, return_counts=True)
    exclusive = counts == 1
    index_grams = unique[exclusive]
    index_owner = all_owners[first[exclusive]].astype(np.int32)
    per_doc = np.bincount(index_owner, minlength=len(doc_ids)).astype(np.int32)
    directory.mkdir(parents=True, exist_ok=True)
    np.save(directory / "grams.npy", index_grams)
    np.save(directory / "owner.npy", index_owner)
    np.save(directory / "per_doc.npy", per_doc)
    (directory / "doc_ids.json").write_text(json.dumps(doc_ids))
    return {
        "heldout_docs": len(doc_ids),
        "exclusive_ngrams": int(index_grams.size),
        "shared_ngrams_dropped": int((~exclusive).sum()),
    }


class LeakChecker:
    def __init__(self, directory: Path, leak_fraction: float) -> None:
        self.grams = np.load(directory / "grams.npy", mmap_mode="r")
        self.owner = np.load(directory / "owner.npy", mmap_mode="r")
        self.per_doc = np.load(directory / "per_doc.npy", mmap_mode="r")
        self.leak_fraction = leak_fraction

    def leaks(self, text: str) -> bool:
        hashes = _doc_ngrams(text)
        if hashes.size == 0 or self.grams.size == 0:
            return False
        positions = np.searchsorted(self.grams, hashes)
        positions[positions == self.grams.size] = 0
        found = self.grams[positions] == hashes
        if not found.any():
            return False
        if found.sum() >= self.leak_fraction * hashes.size:
            return True
        owners, shared = np.unique(self.owner[positions[found]], return_counts=True)
        return bool(np.any(shared >= self.leak_fraction * np.maximum(self.per_doc[owners], 1)))


_STATE: dict[str, Any] = {}


def init_worker(settings: FinalizeSettings, index_dir: Path) -> None:
    _STATE["settings"] = settings
    _STATE["checker"] = LeakChecker(index_dir, settings.leak_fraction)


def _kept_records(ref: UnitRef, keep: npt.NDArray[np.bool_]) -> Iterator[tuple[int, dict[str, Any]]]:
    """Kept rows with their row index; unkept rows are dropped in Arrow, never converted."""
    if not ref.rows:
        return
    offset = 0
    for batch in pq.ParquetFile(ref.candidates).iter_batches(batch_size=256):
        mask = keep[offset : offset + batch.num_rows]
        rows = np.flatnonzero(mask)
        if rows.size:
            for row, record in zip(rows, batch.filter(pa.array(mask)).to_pylist(), strict=True):
                yield offset + int(row), record
        offset += batch.num_rows


def finalize_unit(ref: UnitRef, keep_path: Path, select_path: Path | None) -> dict[str, Any]:
    settings: FinalizeSettings = _STATE["settings"]
    checker: LeakChecker = _STATE["checker"]
    layout = settings.layout
    with np.load(keep_path) as npz:
        retained, cluster_size = npz["keep"], npz["cluster_size"]
    selected = np.load(select_path) if select_path is not None else np.ones_like(retained)
    keep = retained & selected
    root = layout.accepted(settings.build_id)
    writers: dict[str, ShardWriter] = {}
    counts: Counter[str] = Counter()
    leaked = 0
    try:
        for row_index, record in _kept_records(ref, keep):
            split = splits.assign(record["group_key"], settings.split_config)
            if split == "train" and checker.leaks(record["text"]):
                leaked += 1
                continue
            if split not in writers:
                writers[split] = ShardWriter(
                    root / split / ref.source_id / f"{ref.unit_id}.parquet", ACCEPTED_SCHEMA
                )
            writers[split].append(
                {
                    **record,
                    "split": split,
                    "dup_cluster_size": int(cluster_size[row_index]),
                    "unit_id": ref.unit_id,
                }
            )
            counts[split] += 1
        outputs = []
        for split, writer in sorted(writers.items()):
            info = writer.close()
            outputs.append({**info, "file": f"{split}/{ref.source_id}/{info['file']}", "split": split})
    except BaseException:
        for writer in writers.values():
            writer.abort()
        raise
    marker = {
        "schema": "j1.finalize-marker.v1",
        "finalize_version": VERSION,
        "build_identity": settings.identity,
        "source_id": ref.source_id,
        "unit_id": ref.unit_id,
        "candidates": ref.rows,
        "deduplicated_away": int((~retained).sum()),
        "not_selected": int((retained & ~selected).sum()),
        "split_leakage_removed": leaked,
        "accepted_by_split": dict(counts),
        "outputs": outputs,
    }
    faults.point("finalize.unit.written")
    removed = int((~keep).sum())
    if ref.rows != removed + leaked + sum(counts.values()):  # removed covers duplicates and unselected
        raise RuntimeError(f"{ref.unit_id}: finalize accounting mismatch")
    write_json_atomic(root / "_units" / ref.source_id / f"{ref.unit_id}.json", marker)
    return marker


def _finalize_job(job: tuple[UnitRef, Path, Path | None]) -> dict[str, Any]:
    return finalize_unit(*job)


def validate_marker_inventory(
    refs: list[UnitRef], markers: list[dict[str, Any]], build_identity: str
) -> None:
    """Refuse stale, out-of-scope, or differently identified accepted markers."""
    expected = {(ref.source_id, ref.unit_id) for ref in refs}
    actual_list = [(marker.get("source_id"), marker.get("unit_id")) for marker in markers]
    actual = set(actual_list)
    duplicates = sorted(key for key in actual if actual_list.count(key) > 1)
    stale = sorted(actual - expected)
    missing = sorted(expected - actual)
    mismatched = sorted(
        (marker.get("source_id"), marker.get("unit_id"))
        for marker in markers
        if marker.get("build_identity") != build_identity
    )
    duplicate_refs = len(expected) != len(refs)
    if stale or missing or mismatched or duplicates or duplicate_refs:
        raise RuntimeError(
            "accepted marker inventory does not match current inputs; "
            f"stale={stale[:5]}, missing={missing[:5]}, identity_mismatch={mismatched[:5]}, "
            f"duplicate_markers={duplicates[:5]}, duplicate_refs={duplicate_refs}. "
            "Use a new build id for changed scope or inputs."
        )


def _remove_marker_outputs(root: Path, outputs: Any) -> None:
    if not isinstance(outputs, list):
        return
    for entry in outputs:
        if not isinstance(entry, dict) or not isinstance(entry.get("file"), str):
            continue
        try:
            path = safe_child(root, entry["file"])
        except ValueError:
            continue
        if path.is_file() and not path.is_symlink():
            path.unlink()


def _reusable_finalize_markers(refs: list[UnitRef], root: Path, build_identity: str) -> set[tuple[str, str]]:
    """Verify every existing finalization marker and its shards before reuse."""
    expected = {(ref.source_id, ref.unit_id): ref for ref in refs}
    if len(expected) != len(refs):
        raise RuntimeError("finalization input contains duplicate source/unit references")
    markers: dict[tuple[str, str], Path] = {}
    for path in sorted((root / "_units").glob("*/*.json")):
        marker = json.loads(path.read_text())
        key = (marker.get("source_id"), marker.get("unit_id"))
        if key not in expected:
            raise RuntimeError(f"stale accepted marker {path}; use a new build id")
        if key in markers:
            raise RuntimeError(f"duplicate accepted markers for {key}")
        if marker.get("build_identity") != build_identity:
            raise RuntimeError(f"accepted marker {path} has a different build identity")
        if marker.get("candidates") != expected[key].rows:
            raise RuntimeError(f"accepted marker {path} has a stale candidate row count")
        counts = marker.get("accepted_by_split")
        outputs = marker.get("outputs")
        if not isinstance(counts, dict) or not isinstance(outputs, list):
            raise RuntimeError(f"accepted marker {path} is malformed")
        output_counts: Counter[str] = Counter()
        for entry in outputs:
            if not isinstance(entry, dict):
                raise RuntimeError(f"accepted marker {path} has a malformed output entry")
            split, rows = entry.get("split"), entry.get("rows")
            if not isinstance(split, str) or not isinstance(rows, int):
                raise RuntimeError(f"accepted marker {path} has an invalid output split or row count")
            output_counts[split] += rows
        if dict(output_counts) != {key: value for key, value in counts.items() if value}:
            raise RuntimeError(f"accepted marker {path} output counts do not match its split counts")
        accounted = (
            marker.get("deduplicated_away", -1)
            + marker.get("not_selected", -1)
            + marker.get("split_leakage_removed", -1)
            + sum(counts.values())
        )
        if accounted != marker["candidates"]:
            raise RuntimeError(f"accepted marker {path} accounting does not close")
        markers[key] = path

    valid: set[tuple[str, str]] = set()
    for key, path in markers.items():
        marker = json.loads(path.read_text())
        problems = verify_files(root, marker["outputs"])
        if problems:
            _remove_marker_outputs(root, marker["outputs"])
            path.unlink()
            continue
        valid.add(key)
    return valid


def validate_selection(
    refs: list[UnitRef],
    selection_dir: Path,
    identity: str,
    code: dict[str, Any] | None = None,
    expected_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Fail closed unless every unit has a current, in-scope selection mask."""
    from j1data import select

    summary_path = safe_child(selection_dir, "summary.json")
    if summary_path.is_symlink() or not summary_path.is_file():
        raise RuntimeError("selection summary is missing")
    summary = json.loads(summary_path.read_text())
    if not isinstance(summary, dict) or summary.get("schema") != "j1.selection-summary.v1":
        raise RuntimeError("selection summary has an unknown schema")
    if summary.get("selection_version") != select.VERSION:
        raise RuntimeError("selection summary uses an obsolete selection implementation")
    if summary.get("build_identity") != identity:
        raise RuntimeError("selection summary belongs to a different build identity")
    if summary.get("code") != code:
        raise RuntimeError("selection summary code revision differs from the current build command")
    if expected_config is not None and summary.get("config") != expected_config:
        raise RuntimeError("selection summary configuration differs from the current selection config")
    sources = summary.get("sources")
    expected_sources = {ref.source_id for ref in refs}
    if not isinstance(sources, dict) or set(sources) != expected_sources:
        raise RuntimeError("selection summary source inventory does not match deduplicated units")
    expected_files: set[Path] = set()
    for ref in refs:
        relative = f"{ref.source_id}/{ref.unit_id}.npy"
        path = safe_child(selection_dir, relative)
        if (selection_dir / ref.source_id).is_symlink() or path.is_symlink() or not path.is_file():
            raise RuntimeError(f"selection mask is missing for {ref.source_id}/{ref.unit_id}")
        mask = np.load(path, allow_pickle=False)
        if mask.dtype != np.bool_ or mask.ndim != 1 or mask.size != ref.rows:
            raise RuntimeError(
                f"selection mask has an invalid shape or dtype for {ref.source_id}/{ref.unit_id}"
            )
        expected_files.add(path.resolve())
    actual_files = {path.resolve() for path in selection_dir.glob("*/*.npy")}
    if actual_files != expected_files:
        raise RuntimeError("selection masks contain missing, duplicate, or out-of-scope unit files")
    return summary


def run(
    settings: FinalizeSettings,
    refs: list[UnitRef],
    workers: int,
    code: dict[str, Any] | None = None,
    selection_dir: Path | None = None,
    admissible_sources: set[str] | None = None,
    require_selection: bool = False,
    selection_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Finalize retained (and, with ``selection_dir``, selected) candidates into accepted shards."""

    layout = settings.layout
    if admissible_sources is not None:
        inadmissible = sorted({ref.source_id for ref in refs} - admissible_sources)
        if inadmissible:
            raise PermissionError(f"cannot finalize unknown or inadmissible sources: {inadmissible}")
    if require_selection and selection_dir is None:
        raise RuntimeError("release build finalization requires a selection stage")
    selection_summary = (
        validate_selection(refs, selection_dir, settings.identity, code, selection_config)
        if selection_dir is not None
        else None
    )
    dedup_dir = layout.work(settings.build_id) / "dedup"
    keep_paths = {
        ref.candidates.as_posix(): dedup_dir / "keep" / ref.source_id / f"{ref.unit_id}.npz" for ref in refs
    }
    select_paths: dict[str, Path | None] = {
        ref.candidates.as_posix(): (
            selection_dir / ref.source_id / f"{ref.unit_id}.npy" if selection_dir else None
        )
        for ref in refs
    }
    keeps = {}
    for key, path in keep_paths.items():
        with np.load(path) as npz:
            keeps[key] = npz["keep"]
        chosen = select_paths[key]
        if chosen is not None:
            keeps[key] = keeps[key] & np.load(chosen)
    index_dir = layout.work(settings.build_id) / "heldout_index"
    remove_stale_temporaries(index_dir)
    remove_stale_temporaries(layout.accepted(settings.build_id))
    heldout = build_heldout_index(refs, keeps, settings.split_config, index_dir)
    root = layout.accepted(settings.build_id)
    reusable = _reusable_finalize_markers(refs, root, settings.identity)
    pending = [ref for ref in refs if (ref.source_id, ref.unit_id) not in reusable]
    jobs = [
        (ref, keep_paths[ref.candidates.as_posix()], select_paths[ref.candidates.as_posix()])
        for ref in pending
    ]
    results = run_unordered(
        _finalize_job,
        jobs,
        workers,
        initializer=init_worker,
        initargs=(settings, index_dir),
        max_tasks_per_child=16,
    )
    for done, _ in enumerate(results, 1):
        faults.point("finalize.parent.unit_done")
        if done % 50 == 0 or done == len(jobs):
            print(f"finalized {done}/{len(jobs)}", flush=True)
    markers = [json.loads(p.read_text()) for p in sorted((root / "_units").glob("*/*.json"))]
    validate_marker_inventory(refs, markers, settings.identity)
    shards = sorted(
        (dict(o, source_id=m["source_id"]) for m in markers for o in m["outputs"]), key=lambda o: o["file"]
    )
    manifest = {
        "schema": "j1.accepted-manifest.v1",
        "code": code,
        "build_id": settings.build_id,
        "build_identity": settings.identity,
        "finalize_version": VERSION,
        "split_version": splits.VERSION,
        "split_config": {
            "namespace": settings.split_config.namespace,
            "validation_fraction": settings.split_config.validation_fraction,
            "test_fraction": settings.split_config.test_fraction,
            "leak_fraction": settings.leak_fraction,
        },
        "heldout_index": heldout,
        "candidates": sum(m["candidates"] for m in markers),
        "deduplicated_away": sum(m["deduplicated_away"] for m in markers),
        "not_selected": sum(m["not_selected"] for m in markers),
        "selection": selection_summary["sources"] if selection_summary else None,
        "split_leakage_removed": sum(m["split_leakage_removed"] for m in markers),
        "accepted_by_split": dict(sum((Counter(m["accepted_by_split"]) for m in markers), Counter())),
        "shards": shards,
    }
    write_json_atomic(root / "manifest.json", manifest)
    return manifest
