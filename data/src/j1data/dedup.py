"""Corpus-wide exact, normalized-exact and MinHash-LSH near deduplication.

All candidates of a build, from every source, are placed in one global order:
source priority, then source id, unit id and row. Within every duplicate
cluster the first document in that order is retained, so the copy from the
most trusted source survives regardless of which adapter delivered it.

Passes, each run on the survivors of the previous one:

1. identity    repeated doc ids (an upstream key delivered twice)
2. exact       equal SHA-256 of the stored text
3. normalized  equal SHA-256 of the matching signature (case, accents,
               punctuation and whitespace differences)
4. near        connected components of the MinHash-LSH band graph (any shared band)

Hashes are compared on their first 128 bits; at 10^8 documents the chance of
any accidental 128-bit collision is below 10^-22. Memory stays bounded by
processing one key column or one LSH band at a time; signatures are stored
transposed (num_perm x n) so a band is a contiguous slice of every unit file.

Recovery (v0.3 15-minute rule): the pass state (``alive``, ``removed_kind``,
``retained_by`` and the union-find ``parent``) is committed to ``dedup/state`` after each
exact pass and, during the LSH bands, at most every ``CHECKPOINT_SECONDS`` and after the
last band. The journal is written last and binds the state to the unit markers, the
signature overrides, the LSH configuration and the build code fingerprint (the SHA-256 of
every ``build.BUILD_MODULES`` file, not the git commit, so an unrelated commit between a
crash and the restart does not discard a committed journal); arrays are re-hashed before
reuse. The git revision (``code``) is recorded in the summary for information only. The
union-find state after band *b* is fully described by ``parent``, so resuming at band
*b + 1* gives the same components as an uninterrupted run. Keep files are written with
fixed ZIP timestamps and listed with SHA-256 in ``summary.json``, which is written last;
a completed stage whose summary verifies is not recomputed.
"""

from __future__ import annotations

import io
import json
import os
import time
import zipfile
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pyarrow as pa
import pyarrow.parquet as pq

from j1data import faults
from j1data.hashing import sha256_file, sha256_json
from j1data.minhash import LSHConfig, band_keys
from j1data.paths import DataLayout, safe_child
from j1data.shards import ShardWriter, remove_stale_temporaries, verify_files, write_json_atomic

VERSION = "j1.dedup.v1"
CHECKPOINT_SECONDS = 300.0
KINDS = ("identity", "exact", "normalized", "near")

REMOVED_SCHEMA = pa.schema(
    [
        ("doc_id", pa.string()),
        ("source_id", pa.string()),
        ("unit_id", pa.string()),
        ("row", pa.int64()),
        ("kind", pa.string()),
        ("cluster_id", pa.string()),
        ("retained_source_id", pa.string()),
    ]
)


@dataclass(frozen=True)
class UnitRef:
    source_id: str
    unit_id: str
    rows: int
    offset: int
    candidates: Path
    signatures: Path


def collect_units(layout: DataLayout, build_id: str, priorities: dict[str, int]) -> list[UnitRef]:
    work = layout.work(build_id)
    markers = []
    for path in sorted((work / "units").glob("*/*.json")):
        marker = json.loads(path.read_text())
        markers.append(marker)
    markers.sort(key=lambda m: (priorities[m["source_id"]], m["source_id"], m["unit_id"]))
    refs, offset = [], 0
    for marker in markers:
        parts = marker.get("segments")
        if not isinstance(parts, list):
            outputs = marker["outputs"]
            if len(outputs) != 3:
                raise RuntimeError(f"{marker['unit_id']}: malformed processing outputs")
            parts = [{"index": 0, "outputs": outputs}]
            suffix = ""
        else:
            suffix = "~part-"
        for part in parts:
            outputs = part.get("outputs")
            if not isinstance(outputs, list) or len(outputs) != 3:
                raise RuntimeError(f"{marker['unit_id']}: malformed processing segment outputs")
            cand, _, sigs = outputs
            if cand.get("rows") != sigs.get("rows"):
                raise RuntimeError(f"{marker['unit_id']}: candidate and signature row counts differ")
            unit_id = f"{marker['unit_id']}{suffix}{int(part['index']):06d}" if suffix else marker["unit_id"]
            refs.append(
                UnitRef(
                    marker["source_id"],
                    unit_id,
                    cand["rows"],
                    offset,
                    safe_child(work, cand["file"]),
                    safe_child(work, sigs["file"]),
                )
            )
            offset += cand["rows"]
    return refs


def _hex128(column: pa.ChunkedArray) -> npt.NDArray[np.uint64]:
    values = column.to_pylist()
    raw = b"".join(bytes.fromhex(v[:32]) for v in values)
    return np.frombuffer(raw, dtype=">u8").astype(np.uint64).reshape(-1, 2)


def _load_keys(refs: list[UnitRef], column: str) -> npt.NDArray[np.uint64]:
    parts = [
        _hex128(pq.read_table(ref.candidates, columns=[column]).column(column)) for ref in refs if ref.rows
    ]
    return np.concatenate(parts) if parts else np.empty((0, 2), dtype=np.uint64)


def first_of_equal(keys: npt.NDArray[np.uint64], alive: npt.NDArray[np.bool_]) -> npt.NDArray[np.int64]:
    """For each alive row, the smallest alive index with an equal key (itself if unique); -1 if dead."""
    result = np.full(keys.shape[0], -1, dtype=np.int64)
    index = np.flatnonzero(alive)
    if index.size == 0:
        return result
    sub = keys[index]
    order = np.lexsort((index, sub[:, 1], sub[:, 0]))
    sorted_keys = sub[order]
    new_run = np.ones(order.size, dtype=bool)
    new_run[1:] = np.any(sorted_keys[1:] != sorted_keys[:-1], axis=1)
    run_start = np.maximum.accumulate(np.where(new_run, np.arange(order.size), 0))
    result[index[order]] = index[order[run_start]]
    return result


def signature_path(ref: UnitRef, overrides: Path | None) -> Path:
    """The unit's processing signatures, or its dedup-time replacement when one exists."""
    if overrides is not None:
        candidate = overrides / ref.source_id / f"{ref.unit_id}.npy"
        if candidate.exists():
            return candidate
    return ref.signatures


def near_components(
    refs: list[UnitRef],
    lsh: LSHConfig,
    alive: npt.NDArray[np.bool_],
    overrides: Path | None = None,
    parent: npt.NDArray[np.int64] | None = None,
    start_band: int = 0,
    on_band: Callable[[int, npt.NDArray[np.int64]], None] | None = None,
) -> npt.NDArray[np.int64]:
    """Map each alive document to its smallest LSH-component index using O(N) memory.

    Each band is sorted independently. Repeated-key groups are unioned into one
    representative immediately; the function never materializes the full LSH edge
    list or an N-by-N sparse graph. ``parent``/``start_band`` resume from a committed
    union-find state; ``on_band(band, parent)`` is called after every band.
    """
    total = alive.size
    alive_index = np.flatnonzero(alive)
    if alive_index.size == 0:
        return np.where(alive, np.arange(total), -1)
    if parent is None:
        parent = np.arange(total, dtype=np.int64)
    for band in range(start_band, lsh.bands):
        keys = np.empty(total, dtype=np.uint64)
        for ref in refs:
            if ref.rows:
                signatures = np.load(signature_path(ref, overrides), mmap_mode="r")
                keys[ref.offset : ref.offset + ref.rows] = band_keys(signatures, band, lsh.rows)
        active_keys = keys[alive_index]
        order = np.argsort(active_keys, kind="stable")
        sorted_keys = active_keys[order]
        last_head = -1
        block_size = 1_000_000
        for start in range(0, order.size, block_size):
            end = min(start + block_size, order.size)
            run = np.ones(end - start, dtype=bool)
            if start:
                run[0] = sorted_keys[start] != sorted_keys[start - 1]
            run[1:] = sorted_keys[start + 1 : end] != sorted_keys[start : end - 1]
            current_heads = np.full(run.size, last_head, dtype=np.int64)
            new_heads = np.flatnonzero(run)
            current_heads[new_heads] = start + new_heads
            run_heads = np.maximum.accumulate(current_heads)
            duplicates = np.flatnonzero(~run)
            if duplicates.size:
                nodes = alive_index[order[start + duplicates]]
                heads = alive_index[order[run_heads[duplicates]]]
                node_roots = _find_many(parent, nodes)
                head_roots = _find_many(parent, heads)
                low = np.minimum(node_roots, head_roots)
                high = np.maximum(node_roots, head_roots)
                np.minimum.at(parent, high, low)
            last_head = int(run_heads[-1])
        del keys, active_keys, order, sorted_keys
        if on_band is not None:
            on_band(band, parent)
    roots = _find_many(parent, alive_index)
    result = np.full(total, -1, dtype=np.int64)
    result[alive_index] = roots
    return result


def _find_many(parent: npt.NDArray[np.int64], nodes: npt.NDArray[np.int64]) -> npt.NDArray[np.int64]:
    """Vectorized path-compressed union-find lookup."""
    roots = parent[nodes].copy()
    while True:
        next_roots = parent[roots]
        if np.array_equal(roots, next_roots):
            break
        roots = next_roots
    parent[nodes] = roots
    return roots


def save_npz_deterministic(path: Path, **arrays: npt.NDArray[Any]) -> str:
    """``np.savez`` equivalent with fixed ZIP timestamps (byte-reproducible); atomic; returns SHA-256."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_STORED) as archive:
        for name, array in arrays.items():
            buffer = io.BytesIO()
            np.lib.format.write_array(buffer, np.asanyarray(array), allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            archive.writestr(info, buffer.getvalue())
    with tmp.open("rb") as handle:
        os.fsync(handle.fileno())
    tmp.replace(path)
    return sha256_file(path)


def build_code_binding() -> dict[str, str]:
    """Per-module SHA-256 of the build code; a recovery binding that survives unrelated commits."""
    from j1data.build import build_code_fingerprint

    return build_code_fingerprint()


def input_binding(layout: DataLayout, build_id: str, refs: list[UnitRef], overrides: Path | None) -> str:
    """Identity of every dedup input: unit markers (which hash their outputs) and override signatures."""
    work = layout.work(build_id)
    markers = sorted((work / "units").glob("*/*.json"))
    override_files = []
    for ref in refs:
        path = signature_path(ref, overrides)
        if path != ref.signatures:
            override_files.append([ref.source_id, ref.unit_id, sha256_file(path)])
    return sha256_json(
        {
            "markers": [[p.relative_to(work).as_posix(), sha256_file(p)] for p in markers],
            "overrides": override_files,
        }
    )


class _StateStore:
    """Committed dedup pass state: arrays first (atomic, hashed), journal last."""

    def __init__(self, directory: Path, binding: dict[str, Any]) -> None:
        self.directory = directory
        self.journal = directory / "journal.json"
        self.binding = binding
        self.commits = 0
        self.generation = 0
        self.max_interval = 0.0
        self.events: list[str] = []

    def load(self) -> tuple[str, dict[str, npt.NDArray[Any]]] | None:
        if not self.journal.exists():
            return None
        try:
            journal = json.loads(self.journal.read_text())
        except json.JSONDecodeError:
            self.events.append("unreadable journal discarded")
            return None
        if journal.get("binding") != self.binding:
            self.events.append("journal for different inputs discarded")
            return None
        if verify_files(self.directory, journal["arrays"]):
            self.events.append("state arrays failed verification; restarting the stage")
            return None
        arrays = {e["name"]: np.load(self.directory / e["file"]) for e in journal["arrays"]}
        self.generation = int(journal.get("generation", 0))
        self.events.append(f"resumed at {journal['next']}")
        return str(journal["next"]), arrays

    def commit(self, next_step: str, **arrays: npt.NDArray[Any]) -> None:
        # Each commit writes a new generation of files, so a crash before the journal is
        # replaced leaves the previous commit intact; superseded files are removed afterwards.
        generation = self.generation + 1
        entries = []
        for name, array in arrays.items():
            path = self.directory / f"{name}.{generation:06d}.npy"
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_name(path.name + ".tmp")
            with tmp.open("wb") as handle:
                np.lib.format.write_array(handle, array, allow_pickle=False)
                handle.flush()
                os.fsync(handle.fileno())
            tmp.replace(path)
            entries.append({"name": name, "file": path.name, "sha256": sha256_file(path)})
        faults.point("dedup.state.arrays")
        write_json_atomic(
            self.journal,
            {"binding": self.binding, "next": next_step, "generation": generation, "arrays": entries},
        )
        self.generation = generation
        self.commits += 1
        current = {entry["file"] for entry in entries} | {self.journal.name}
        for path in self.directory.iterdir():
            if path.name not in current and not path.name.endswith(".tmp"):
                path.unlink()
        faults.point("dedup.state.commit")


def _verified_summary(out: Path, binding: dict[str, Any]) -> dict[str, Any] | None:
    path = out / "summary.json"
    if not path.exists():
        return None
    summary: dict[str, Any] = json.loads(path.read_text())
    if summary.get("recovery_binding") != binding:
        return None
    entries = [summary.get("removed_file"), *summary.get("keep_files", [])]
    if not summary.get("keep_files") or verify_files(out, entries):
        return None
    return summary


def run(
    layout: DataLayout,
    build_id: str,
    priorities: dict[str, int],
    lsh: LSHConfig,
    code: dict[str, Any] | None = None,
    overrides: Path | None = None,
    build_identity: str | None = None,
) -> dict[str, Any]:
    refs = collect_units(layout, build_id, priorities)
    total = sum(ref.rows for ref in refs)
    out = layout.work(build_id) / "dedup"
    remove_stale_temporaries(out)
    binding = {
        "dedup_version": VERSION,
        "build_identity": build_identity,
        "inputs": input_binding(layout, build_id, refs, overrides),
        "units": sha256_json([[r.source_id, r.unit_id, r.rows, r.offset] for r in refs]),
        "lsh": [lsh.bands, lsh.rows],
        "build_code": build_code_binding(),
    }
    done = _verified_summary(out, binding)
    if done is not None:
        for path in (out / "state").glob("*"):
            path.unlink()
        return done
    (out / "summary.json").unlink(missing_ok=True)  # invalidate before any output is rewritten
    store = _StateStore(out / "state", binding)
    started = time.monotonic()
    loaded = store.load()
    step, arrays = loaded if loaded is not None else ("exact:0", {})
    alive = arrays.get("alive", np.ones(total, dtype=bool))
    removed_kind = arrays.get("removed_kind", np.full(total, -1, dtype=np.int8))
    retained_by = arrays.get("retained_by", np.arange(total, dtype=np.int64))
    if alive.size != total:
        raise RuntimeError("dedup state does not match the unit inventory")

    columns = ("doc_id", "sha256", "norm_sha256")
    if step.startswith("exact:"):
        for kind_index in range(int(step.split(":")[1]), len(columns)):
            first = first_of_equal(_load_keys(refs, columns[kind_index]), alive)
            duplicate = alive & (first != np.arange(total))
            removed_kind[duplicate] = kind_index
            retained_by[duplicate] = first[duplicate]
            alive &= ~duplicate
            next_step = f"exact:{kind_index + 1}" if kind_index + 1 < len(columns) else "near:0"
            store.commit(next_step, alive=alive, removed_kind=removed_kind, retained_by=retained_by)
        step = "near:0"
    if step.startswith("near:"):
        last_commit = time.monotonic()

        def on_band(band: int, parent: npt.NDArray[np.int64]) -> None:
            nonlocal last_commit
            faults.point("dedup.near.band")
            if band + 1 == lsh.bands or time.monotonic() - last_commit >= CHECKPOINT_SECONDS:
                store.commit(
                    f"near:{band + 1}",
                    alive=alive,
                    removed_kind=removed_kind,
                    retained_by=retained_by,
                    parent=parent,
                )
                store.max_interval = max(store.max_interval, time.monotonic() - last_commit)
                last_commit = time.monotonic()

        representative = near_components(
            refs,
            lsh,
            alive,
            overrides,
            parent=arrays.get("parent"),
            start_band=int(step.split(":")[1]),
            on_band=on_band,
        )
        duplicate = alive & (representative != np.arange(total))
        removed_kind[duplicate] = KINDS.index("near")
        retained_by[duplicate] = representative[duplicate]
        alive &= ~duplicate
        # Chains through removed documents always end at a retained one.
        for _ in range(4):
            retained_by = retained_by[retained_by]
        store.commit("output", alive=alive, removed_kind=removed_kind, retained_by=retained_by)
    elif step != "output":
        raise RuntimeError(f"unknown dedup journal step {step!r}")

    owner = np.empty(total, dtype=np.int32)
    for position, ref in enumerate(refs):
        owner[ref.offset : ref.offset + ref.rows] = position
    removed = np.flatnonzero(~alive)
    needed = np.unique(np.concatenate([removed, retained_by[removed]]))
    ids: dict[int, str] = {}
    for position in np.unique(owner[needed]):
        ref = refs[position]
        column = pq.read_table(ref.candidates, columns=["doc_id"]).column("doc_id")
        lo, hi = np.searchsorted(needed, [ref.offset, ref.offset + ref.rows])
        for doc in needed[lo:hi]:
            ids[int(doc)] = column[int(doc - ref.offset)].as_py()

    cluster_sizes = np.bincount(retained_by, minlength=total)
    writer = ShardWriter(out / "removed.parquet", REMOVED_SCHEMA)
    by_source: dict[str, Counter[str]] = {}
    cross_source: Counter[str] = Counter()
    try:
        for doc in removed:
            ref = refs[owner[doc]]
            keeper = refs[owner[retained_by[doc]]]
            kind = KINDS[removed_kind[doc]]
            by_source.setdefault(ref.source_id, Counter())[kind] += 1
            if keeper.source_id != ref.source_id:
                cross_source[f"{ref.source_id} -> {keeper.source_id}"] += 1
            writer.append(
                {
                    "doc_id": ids[int(doc)],
                    "source_id": ref.source_id,
                    "unit_id": ref.unit_id,
                    "row": int(doc - ref.offset),
                    "kind": kind,
                    "cluster_id": ids[int(retained_by[doc])],
                    "retained_source_id": keeper.source_id,
                }
            )
        removed_info = writer.close()
    except BaseException:
        writer.abort()
        raise

    keep_files = []
    for ref in refs:
        relative = f"keep/{ref.source_id}/{ref.unit_id}.npz"
        digest = save_npz_deterministic(
            out / relative,
            keep=alive[ref.offset : ref.offset + ref.rows],
            cluster_size=cluster_sizes[ref.offset : ref.offset + ref.rows].astype(np.int32),
        )
        keep_files.append({"file": relative, "sha256": digest})
        faults.point("dedup.output.keep")

    sizes = cluster_sizes[alive]
    summary = {
        "schema": "j1.dedup-summary.v1",
        "dedup_version": VERSION,
        "build_identity": build_identity,
        "candidates": int(total),
        "retained": int(alive.sum()),
        "removed": {kind: int((removed_kind == i).sum()) for i, kind in enumerate(KINDS)},
        "removed_by_source": {s: dict(c) for s, c in sorted(by_source.items())},
        "cross_source_removals": dict(cross_source.most_common()),
        "cluster_size_histogram": {
            str(k): int(v)
            for k, v in zip(*np.unique(np.minimum(sizes, 100), return_counts=True), strict=True)
        },
        "removed_file": removed_info,
        "keep_files": keep_files,
        "recovery_binding": binding,
        "retention_rule": "lowest (source priority, source id, unit id, row) in each cluster",
        "source_priorities": dict(sorted(priorities.items(), key=lambda kv: kv[1])),
        "signature_overrides": sorted(
            {ref.source_id for ref in refs if signature_path(ref, overrides) != ref.signatures}
        ),
        "code": code,
        "lsh": {"bands": lsh.bands, "rows": lsh.rows, "threshold_at_half_detection": round(lsh.threshold, 4)},
    }
    write_json_atomic(out / "summary.json", summary)
    faults.point("dedup.summary.written")
    for path in (out / "state").glob("*"):
        path.unlink()
    write_json_atomic(
        out / "recovery_log.json",
        {
            "seconds": round(time.monotonic() - started, 1),
            "state_commits": store.commits,
            "max_band_commit_interval_seconds": round(store.max_interval, 1),
            "events": store.events,
        },
    )
    return summary
