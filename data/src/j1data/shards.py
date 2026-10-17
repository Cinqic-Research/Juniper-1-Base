"""Crash-safe Parquet shard writing and verification.

A shard is written to ``<name>.tmp``, fsynced and renamed, so a reader never
sees a partial file. Every shard is described by its SHA-256 and row count in
the manifest of the unit that produced it; ``verify_files`` re-hashes shards
before they are reused, so a corrupted or truncated file is detected instead of
silently trusted.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from j1data import faults
from j1data.hashing import sha256_file
from j1data.paths import safe_child

COMPRESSION = "zstd"
COMPRESSION_LEVEL = 3


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class ShardWriter:
    def __init__(
        self, path: Path, schema: pa.Schema, rows_per_group: int = 4096, max_buffer_bytes: int = 32 << 20
    ) -> None:
        self.path = path
        self.tmp = path.with_name(path.name + ".tmp")
        self.schema = schema
        self.rows_per_group = rows_per_group
        # Row groups are cut by size as well as count, so a unit of very long
        # documents (papers, books) cannot hold hundreds of megabytes in memory.
        self.max_buffer_bytes = max_buffer_bytes
        self._buffer: dict[str, list[Any]] = {name: [] for name in schema.names}
        self._buffer_bytes = 0
        self._pending = 0
        self.rows = 0
        self._closed = False
        path.parent.mkdir(parents=True, exist_ok=True)
        self._writer = pq.ParquetWriter(
            self.tmp,
            schema,
            compression=COMPRESSION,
            compression_level=COMPRESSION_LEVEL,
            write_statistics=False,
        )

    def append(self, row: dict[str, Any]) -> None:
        for name in self.schema.names:
            value = row[name]
            self._buffer[name].append(value)
            if isinstance(value, str):
                self._buffer_bytes += len(value)
        self._pending += 1
        if self._pending >= self.rows_per_group or self._buffer_bytes >= self.max_buffer_bytes:
            self._flush()

    def _flush(self) -> None:
        if not self._pending:
            return
        table = pa.Table.from_pydict(self._buffer, schema=self.schema)
        self._writer.write_table(table)
        self.rows += self._pending
        self._buffer = {name: [] for name in self.schema.names}
        self._buffer_bytes = 0
        self._pending = 0

    def close(self) -> dict[str, Any]:
        if self._closed:
            raise RuntimeError(f"shard writer already closed: {self.path}")
        self._flush()
        self._writer.close()
        self._closed = True
        with self.tmp.open("rb") as handle:
            os.fsync(handle.fileno())
        self.tmp.replace(self.path)
        _fsync_directory(self.path.parent)
        return {"file": self.path.name, "rows": self.rows, "sha256": sha256_file(self.path)}

    def abort(self) -> None:
        try:
            if not self._closed:
                self._writer.close()
                self._closed = True
        finally:
            self.tmp.unlink(missing_ok=True)


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    with tmp.open("w") as handle:
        handle.write(json.dumps(value, indent=1, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    faults.point("json.atomic.tmp")
    tmp.replace(path)
    _fsync_directory(path.parent)


def remove_stale_temporaries(directory: Path) -> list[str]:
    """Delete temporaries a killed writer left under ``directory`` (``*.tmp``, ``*.<pid>.tmp``, ``.tmp.npy``).

    Called only at the start of a stage, before any of its writers run, so no live temporary is removed.
    This assumes a single writer per directory: no lock is taken, so a second process started in the
    same stage directory would delete the first one's live temporaries (docs/stage_recovery.md, D-15).
    """
    removed = []
    if directory.is_dir():
        for path in sorted(directory.rglob("*")):
            if path.is_file() and (path.name.endswith(".tmp") or path.name.endswith(".tmp.npy")):
                path.unlink()
                removed.append(path.relative_to(directory).as_posix())
    return removed


def verify_files(directory: Path, entries: list[Any], check_rows: bool = True) -> list[str]:
    """Return a list of problems (empty when every listed file matches)."""
    problems = []
    for entry in entries:
        if not isinstance(entry, dict):
            problems.append("malformed output entry")
            continue
        try:
            relative = entry["file"]
            expected_sha256 = entry["sha256"]
            path = safe_child(directory, relative)
        except (KeyError, TypeError, ValueError):
            problems.append(f"unsafe output path {entry.get('file')!r}")
            continue
        if not path.exists():
            problems.append(f"missing {relative}")
            continue
        if path.is_symlink() or not path.is_file():
            problems.append(f"not a regular file {relative}")
            continue
        if sha256_file(path) != expected_sha256:
            problems.append(f"sha256 mismatch {relative}")
            continue
        if check_rows and path.suffix == ".parquet":
            rows = pq.ParquetFile(path).metadata.num_rows
            expected_rows = entry.get("rows")
            if not isinstance(expected_rows, int) or rows != expected_rows:
                problems.append(f"row count mismatch {relative}: {rows} != {expected_rows}")
    return problems


class SignatureSink:
    """Stream MinHash signatures to disk and store them transposed (num_perm x n).

    Signatures are appended row by row to a raw temporary file in blocks, then
    transposed block-wise into a memory-mapped ``.npy``, so memory stays bounded
    by one block whatever the number of documents in a unit."""

    BLOCK = 8192

    def __init__(self, path: Path, num_perm: int) -> None:
        self.path = path
        self.num_perm = num_perm
        self.rows_path = path.with_name(path.name + ".rows.tmp")
        self.npy_tmp = path.with_name(path.name + ".tmp.npy")
        path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.rows_path.open("wb")
        self._block: list[np.ndarray] = []
        self.rows = 0

    def append(self, signature: np.ndarray) -> None:
        self._block.append(signature)
        if len(self._block) >= self.BLOCK:
            self._flush()

    def _flush(self) -> None:
        if self._block:
            self._handle.write(np.stack(self._block).astype(np.uint32).tobytes())
            self.rows += len(self._block)
            self._block = []

    def close(self) -> dict[str, Any]:
        self._flush()
        self._handle.close()
        out = np.lib.format.open_memmap(
            self.npy_tmp, mode="w+", dtype=np.uint32, shape=(self.num_perm, self.rows)
        )
        if self.rows:
            rows = np.memmap(self.rows_path, dtype=np.uint32, mode="r", shape=(self.rows, self.num_perm))
            for start in range(0, self.rows, self.BLOCK):
                out[:, start : start + self.BLOCK] = rows[start : start + self.BLOCK].T
            del rows
        out.flush()
        del out
        self.rows_path.unlink()
        self.npy_tmp.replace(self.path)
        with self.path.open("rb") as handle:
            os.fsync(handle.fileno())
        _fsync_directory(self.path.parent)
        return {"file": self.path.name, "rows": self.rows, "sha256": sha256_file(self.path)}

    def abort(self) -> None:
        self._handle.close()
        self.rows_path.unlink(missing_ok=True)
        self.npy_tmp.unlink(missing_ok=True)
