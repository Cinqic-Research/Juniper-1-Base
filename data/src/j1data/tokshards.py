"""Model-ready token-ID shards (j1.token-shards.v1) from an accepted build and a frozen tokenizer.

Per split, the output is a single logical token stream stored as fixed-capacity
``tokens-NNNNN.bin`` files (little-endian uint16, since every candidate vocabulary is
below 65,536). Each document's tokens are followed by ``<|endoftext|>`` (ID 0).
``docs-NNNNN.parquet`` indexes every document: doc_id, source_id, domain, global token
offset, and token count.

**Order.** Documents are sorted by ``unit_interval(ORDER_NAMESPACE, doc_id)``, a keyed
hash, so the stream is a deterministic global shuffle of sources that does not depend
on file layout. The sort uses sequential I/O only. Pass 1 tokenizes the accepted shards
in their stored order and appends each document to one of ``N_BUCKETS`` spill files,
chosen by the key's leading bits. Pass 2 loads one bucket at a time, sorts it by key,
and appends it to the stream. Because the buckets partition the key range in order, the
result is exactly the global key order.

**Recovery (v0.3 15-minute rule).** The build is a sequence of committed units inside
``<out>.partial``; ``<out>`` appears only by a final rename after ``shards.json``.

* Pass 1 appends each document as a self-describing record (key, token count, doc id,
  source, domain, tokens) to its bucket's append-only spill log; no per-document state is
  held in memory. At least every ``CHECKPOINT_SECONDS`` (and at the end of the split) every
  log is fsynced and the journal ``.progress-<split>.json`` records each log's committed
  length and the SHA-256 of the bytes appended since the previous commit. A restart
  truncates each log to its committed length and continues with the first uncommitted
  accepted shard, so at most one commit interval plus one shard is recomputed.
* Pass 2 handles one bucket per unit: the bucket's committed chunks are re-hashed, its
  records sorted, and its tokens written with ``pwrite`` into their final positions in
  pre-sized token files (bucket base offsets follow from the pass-1 totals); the bucket's
  index part and the SHA-256 of its token region are then committed. Committed buckets are
  re-verified on resume; a failing one is rewritten.
* Finishing renames and hashes token files, writes the document index in pyarrow's default
  row groups, writes ``shards.json`` and renames the directory. Wall-clock timings go to
  ``build_log.json``, so every other output file is byte-reproducible.

Memory is bounded by one bucket (about 2 bytes x split tokens / ``N_BUCKETS``) plus the
tokenizer, not by the number of documents. ``.identity.json`` binds the partial directory
to the accepted manifest, tokenizer and layout constants; a mismatch refuses to resume.

Training-time packing, windowing and resume state live in the training repository. This
stream only fixes token identities and document order.
"""

from __future__ import annotations

import hashlib
import json
import os
import resource
import shutil
import struct
import time
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tokenizers import Tokenizer

from j1data import faults, jtok
from j1data.hashing import sha256_bytes, sha256_file, sha256_json, unit_interval
from j1data.shards import _fsync_directory, remove_stale_temporaries, write_json_atomic

VERSION = "j1.token-shards.v1"
ORDER_NAMESPACE = "j1.pack-order.v1"
N_BUCKETS = 64
TOKENS_PER_FILE = 1 << 29  # 1 GiB of uint16
EOS = 0
_HDR = struct.Struct("<dIHHH")  # key, n_tokens, UTF-8 lengths of doc_id, source_id, domain
CHECKPOINT_SECONDS = 120.0


def _shard_docs(acc: Path, shard: dict[str, Any]) -> Iterator[tuple[str, str, str, str]]:
    t = pq.read_table(acc / shard["file"], columns=["doc_id", "text", "source_id", "domain"])
    cols = [t.column(c).to_pylist() for c in ("doc_id", "text", "source_id", "domain")]
    yield from zip(*cols, strict=True)


class _SpillCorrupt(RuntimeError):
    pass


def _new_split_state(split: str, n_shards: int) -> dict[str, Any]:
    return {
        "schema": "j1.token-shards-progress.v1",
        "split": split,
        "phase": "spill",
        "shards": n_shards,
        "next_shard": 0,
        "docs_in": 0,
        "bucket_bytes": [0] * N_BUCKETS,
        "bucket_docs": [0] * N_BUCKETS,
        "bucket_tokens": [0] * N_BUCKETS,
        "chunks": [[] for _ in range(N_BUCKETS)],
        "placed": {},
        "files_done": [],
        "commits": 0,
        "recovery_events": [],
        "result": None,
    }


def _spill(
    acc: Path,
    shards: list[dict[str, Any]],
    tok: Tokenizer,
    spill: Path,
    state: dict[str, Any],
    journal: Path,
    batch: int,
    log: dict[str, Any],
) -> None:
    """Pass 1: tokenize accepted shards in stored order into per-bucket append-only spill logs.

    A commit fsyncs every log, records each log's committed length and the SHA-256 of the
    bytes appended since the previous commit, then writes the journal. Restart truncates every
    log to its committed length and continues with the first uncommitted shard."""
    spill.mkdir(parents=True, exist_ok=True)
    files: list[BinaryIO] = []
    for b in range(N_BUCKETS):
        path = spill / f"b{b:02d}.bin"
        committed = state["bucket_bytes"][b]
        if committed and (not path.is_file() or path.stat().st_size < committed):
            raise _SpillCorrupt(f"spill log {path.name} is shorter than its committed length")
        log_file = path.open("r+b" if path.exists() else "w+b")
        log_file.truncate(committed)
        log_file.seek(committed)
        files.append(log_file)
    buffers = [bytearray() for _ in range(N_BUCKETS)]
    digests = [hashlib.sha256() for _ in range(N_BUCKETS)]
    starts = list(state["bucket_bytes"])
    docs = list(state["bucket_docs"])
    tokens = list(state["bucket_tokens"])
    docs_in = state["docs_in"]
    last_commit = time.monotonic()

    def drain() -> None:
        for b, buffer in enumerate(buffers):
            if buffer:
                files[b].write(buffer)
                digests[b].update(buffer)
                buffer.clear()

    def commit(next_shard: int) -> None:
        nonlocal last_commit
        drain()
        for b, handle in enumerate(files):
            handle.flush()
            os.fsync(handle.fileno())
            end = handle.tell()
            if end > starts[b]:
                state["chunks"][b].append([starts[b], end, digests[b].hexdigest()])
                starts[b] = end
                digests[b] = hashlib.sha256()
        state.update(
            next_shard=next_shard,
            docs_in=docs_in,
            bucket_bytes=list(starts),
            bucket_docs=list(docs),
            bucket_tokens=list(tokens),
            commits=state["commits"] + 1,
        )
        write_json_atomic(journal, state)
        now = time.monotonic()
        log["max_commit_interval_seconds"] = max(
            log.get("max_commit_interval_seconds", 0.0), now - last_commit
        )
        last_commit = now
        faults.point("tokshards.spill.commit")

    pending: list[tuple[str, str, str, str]] = []

    def flush() -> None:
        nonlocal docs_in
        encs = tok.encode_batch_fast([p[1] for p in pending], add_special_tokens=False)
        for (did, _, src, dom), enc in zip(pending, encs, strict=True):
            ids = np.asarray(enc.ids, dtype=np.uint16)
            if ids.size and int(ids.min()) < jtok.FIRST_BYTE_ID:
                raise ValueError(f"control ID produced from text in {did}")
            key = unit_interval(ORDER_NAMESPACE, did)
            b = min(int(key * N_BUCKETS), N_BUCKETS - 1)
            names = [s.encode("utf-8") for s in (did, src, dom)]
            buffers[b] += _HDR.pack(key, ids.size, *(len(n) for n in names))
            for name in names:
                buffers[b] += name
            buffers[b] += ids.astype("<u2", copy=False).tobytes()
            docs[b] += 1
            tokens[b] += int(ids.size)
            docs_in += 1
        pending.clear()
        faults.point("tokshards.spill.batch")

    try:
        for position in range(state["next_shard"], len(shards)):
            for doc in _shard_docs(acc, shards[position]):
                pending.append(doc)
                if len(pending) >= batch:
                    flush()
            if pending:
                flush()
            drain()
            if position + 1 == len(shards) or time.monotonic() - last_commit >= CHECKPOINT_SECONDS:
                commit(position + 1)
        if not shards:
            commit(0)
    finally:
        for handle in files:
            handle.close()


def _parse_bucket(raw: bytes) -> list[tuple[float, str, int, int, str, str]]:
    records = []
    pos = 0
    while pos < len(raw):
        key, n, a, s, d = _HDR.unpack_from(raw, pos)
        pos += _HDR.size
        did = raw[pos : pos + a].decode("utf-8")
        src = raw[pos + a : pos + a + s].decode("utf-8")
        dom = raw[pos + a + s : pos + a + s + d].decode("utf-8")
        pos += a + s + d
        records.append((key, did, pos, n, src, dom))
        pos += 2 * n
    if pos != len(raw):
        raise _SpillCorrupt("spill log ends inside a record")
    return records


def _token_file_layout(total: int) -> list[tuple[int, int]]:
    """(first_offset, tokens) of each fixed-capacity token file of a stream of ``total`` tokens."""
    return [(start, min(TOKENS_PER_FILE, total - start)) for start in range(0, total, TOKENS_PER_FILE)]


def _region_io(
    part: Path, split: str, layout: list[tuple[int, int]], offset: int, data: bytes | None, size: int
) -> bytes:
    """Write ``data`` (or read ``size`` tokens) at global token ``offset`` across token files."""
    out = bytearray()
    remaining = size if data is None else len(data) // 2
    pos = 0
    while remaining:
        index = offset // TOKENS_PER_FILE
        first, capacity = layout[index]
        local = offset - first
        take = min(remaining, capacity - local)
        path = part / f"{split}-tokens-{index:05d}.bin.partial"
        if data is None and not path.exists():
            path = part / f"{split}-tokens-{index:05d}.bin"
        fd = os.open(path, os.O_RDWR)
        try:
            if data is None:
                out += os.pread(fd, 2 * take, 2 * local)
            else:
                os.pwrite(fd, data[2 * pos : 2 * (pos + take)], 2 * local)
                os.fsync(fd)
        finally:
            os.close(fd)
        offset += take
        pos += take
        remaining -= take
    return bytes(out)


INDEX_SCHEMA = pa.schema(
    [
        ("doc_id", pa.string()),
        ("source_id", pa.string()),
        ("domain", pa.string()),
        ("offset", pa.int64()),
        ("n_tokens", pa.int32()),
    ]
)
INDEX_ROW_GROUP = 1024 * 1024  # pyarrow's default row-group length: same bytes as one write_table call


def _place(part: Path, split: str, spill: Path, state: dict[str, Any], journal: Path) -> None:
    """Pass 2: per bucket, sort by key and write the tokens into their final stream positions."""
    total = sum(state["bucket_tokens"]) + sum(state["bucket_docs"])
    layout = _token_file_layout(total)
    for index, (_, tokens) in enumerate(layout):
        path = part / f"{split}-tokens-{index:05d}.bin.partial"
        if not path.exists():
            with path.open("wb") as handle:
                handle.truncate(2 * tokens)
        elif path.stat().st_size != 2 * tokens:
            raise RuntimeError(f"{path.name}: unexpected size for an unfinished token file")
    bases = np.concatenate(
        ([0], np.cumsum(np.add(state["bucket_tokens"], state["bucket_docs"], dtype=np.int64)))
    )
    index_dir = spill / "index"
    index_dir.mkdir(parents=True, exist_ok=True)
    for b in range(N_BUCKETS):
        size = int(bases[b + 1] - bases[b])
        done = state["placed"].get(str(b))
        if done is not None:
            region = _region_io(part, split, layout, int(bases[b]), None, size)
            index_path = index_dir / done["index"]["file"]
            if (
                sha256_bytes(region) == done["tokens_sha256"]
                and index_path.is_file()
                and sha256_file(index_path) == done["index"]["sha256"]
            ):
                continue
            state["recovery_events"].append({"bucket": b, "event": "placed bucket failed verification"})
        path = spill / f"b{b:02d}.bin"
        length = state["bucket_bytes"][b]
        with path.open("rb") as handle:
            raw = handle.read(length)
        if len(raw) != length:
            raise _SpillCorrupt(f"{path.name} is shorter than its committed length")
        for start, end, digest in state["chunks"][b]:
            if sha256_bytes(raw[start:end]) != digest:
                raise _SpillCorrupt(f"{path.name} bytes {start}-{end} do not match their commit hash")
        records = _parse_bucket(raw)
        if len(records) != state["bucket_docs"][b]:
            raise _SpillCorrupt(
                f"{path.name} holds {len(records)} records, journal says {state['bucket_docs'][b]}"
            )
        records.sort(key=lambda r: (r[0], r[1]))  # doc_id breaks (astronomically unlikely) key ties
        stream = np.empty(size, dtype="<u2")
        columns: dict[str, list[Any]] = {name: [] for name in INDEX_SCHEMA.names}
        by_source: dict[str, dict[str, int]] = defaultdict(lambda: {"documents": 0, "tokens": 0})
        pos = 0
        for _, did, p, n, src, dom in records:
            stream[pos : pos + n] = np.frombuffer(raw, dtype="<u2", count=n, offset=p)
            stream[pos + n] = EOS
            columns["doc_id"].append(did)
            columns["source_id"].append(src)
            columns["domain"].append(dom)
            columns["offset"].append(int(bases[b]) + pos)
            columns["n_tokens"].append(n)
            by_source[src]["documents"] += 1
            by_source[src]["tokens"] += n
            pos += n + 1
        del raw
        data = stream.tobytes()
        _region_io(part, split, layout, int(bases[b]), data, size)
        faults.point("tokshards.place.written")
        index_path = index_dir / f"b{b:02d}.parquet"
        tmp = index_path.with_name(index_path.name + ".tmp")
        pq.write_table(pa.table(columns, schema=INDEX_SCHEMA), tmp, compression="zstd")
        with tmp.open("rb") as handle:
            os.fsync(handle.fileno())
        tmp.replace(index_path)
        state["placed"][str(b)] = {
            "tokens_sha256": sha256_bytes(data),
            "index": {"file": index_path.name, "sha256": sha256_file(index_path), "rows": len(records)},
            "by_source": dict(by_source),
        }
        state["commits"] += 1
        write_json_atomic(journal, state)
        faults.point("tokshards.place.commit")


def _finish(part: Path, split: str, spill: Path, state: dict[str, Any]) -> dict[str, Any]:
    """Re-check the committed buckets, hash and publish token files, then assemble the document index."""
    total = sum(state["bucket_tokens"]) + sum(state["bucket_docs"])
    layout = _token_file_layout(total)
    bases = np.concatenate(
        ([0], np.cumsum(np.add(state["bucket_tokens"], state["bucket_docs"], dtype=np.int64)))
    )
    for b in range(N_BUCKETS):
        entry = state["placed"][str(b)]
        region = _region_io(part, split, layout, int(bases[b]), None, int(bases[b + 1] - bases[b]))
        if sha256_bytes(region) != entry["tokens_sha256"]:
            raise RuntimeError(f"{split}: bucket {b} tokens no longer match their commit hash")
        index_path = spill / "index" / entry["index"]["file"]
        if not index_path.is_file() or sha256_file(index_path) != entry["index"]["sha256"]:
            raise RuntimeError(f"{split}: bucket {b} index part no longer matches its commit hash")
    del region
    token_files = []
    for index, (first, tokens) in enumerate(layout):
        name = f"{split}-tokens-{index:05d}.bin"
        final, partial = part / name, part / (name + ".partial")
        if partial.exists():
            with partial.open("rb") as handle:
                os.fsync(handle.fileno())
            partial.replace(final)
        token_files.append(
            {"file": name, "tokens": tokens, "first_offset": first, "sha256": sha256_file(final)}
        )
        faults.point("tokshards.finish.file")
    idx_path = part / f"{split}-docs.parquet"
    tmp = idx_path.with_suffix(".partial")
    rows = 0
    writer = pq.ParquetWriter(tmp, INDEX_SCHEMA, compression="zstd")
    try:
        pending: list[pa.Table] = []
        buffered = 0
        for b in range(N_BUCKETS):
            entry = state["placed"][str(b)]
            table = pq.read_table(spill / "index" / entry["index"]["file"], schema=INDEX_SCHEMA)
            pending.append(table)
            buffered += table.num_rows
            while buffered >= INDEX_ROW_GROUP:
                joined = pa.concat_tables(pending).combine_chunks()  # one chunk per column, as before
                writer.write_table(joined.slice(0, INDEX_ROW_GROUP), row_group_size=INDEX_ROW_GROUP)
                rest = joined.slice(INDEX_ROW_GROUP)
                pending, buffered = [rest], rest.num_rows
                rows += INDEX_ROW_GROUP
        if buffered or rows == 0:
            writer.write_table(
                pa.concat_tables(pending).combine_chunks() if pending else INDEX_SCHEMA.empty_table()
            )
            rows += buffered
    finally:
        writer.close()
    with tmp.open("rb") as handle:
        os.fsync(handle.fileno())
    tmp.replace(idx_path)
    if rows != state["docs_in"]:
        raise RuntimeError("document count changed between passes")
    by_source: dict[str, dict[str, int]] = defaultdict(lambda: {"documents": 0, "tokens": 0})
    for entry in state["placed"].values():
        for src, value in entry["by_source"].items():
            by_source[src]["documents"] += value["documents"]
            by_source[src]["tokens"] += value["tokens"]
    return {
        "split": split,
        "documents": rows,
        "document_tokens": sum(state["bucket_tokens"]),
        "separator_tokens": rows,
        "stream_tokens": total,
        "by_source": dict(sorted(by_source.items())),
        "token_files": token_files,
        "doc_index": {"file": idx_path.name, "sha256": sha256_file(idx_path), "rows": rows},
    }


def build_split(
    acc: Path,
    manifest: dict[str, Any],
    tok: Tokenizer,
    split: str,
    out: Path,
    batch: int = 1024,
    log: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one split as a sequence of committed units (see the module docstring); resumable."""
    log = log if log is not None else {}
    t0 = time.time()
    journal = out / f".progress-{split}.json"
    spill = out / f".spill-{split}"
    shards = [s for s in manifest["shards"] if s["split"] == split]
    state = json.loads(journal.read_text()) if journal.exists() else _new_split_state(split, len(shards))
    if state.get("schema") != "j1.token-shards-progress.v1" or state.get("shards") != len(shards):
        raise RuntimeError(f"{journal.name}: progress journal does not match this build")
    log["resumed_from"] = {
        "phase": state["phase"],
        "next_shard": state["next_shard"],
        "placed": len(state["placed"]),
    }
    for attempt in range(2):
        try:
            if state["phase"] == "spill":
                _spill(acc, shards, tok, spill, state, journal, batch, log)
                state["phase"] = "place"
                write_json_atomic(journal, state)
                log["tokenize_and_spill"] = round(time.time() - t0, 1)
            if state["phase"] == "place":
                _place(out, split, spill, state, journal)
                state["phase"] = "finish"
                write_json_atomic(journal, state)
            break
        except _SpillCorrupt as error:
            if attempt:
                raise
            events = [*state["recovery_events"], {"event": "spill discarded", "reason": str(error)}]
            for path in out.glob(f"{split}-tokens-*.bin.partial"):
                path.unlink()
            shutil.rmtree(spill, ignore_errors=True)
            state = _new_split_state(split, len(shards))
            state["recovery_events"] = events
            write_json_atomic(journal, state)
    if state["phase"] == "finish":
        state["result"] = _finish(out, split, spill, state)
        state["phase"] = "done"
        write_json_atomic(journal, state)
        faults.point("tokshards.split.done")
    shutil.rmtree(spill, ignore_errors=True)
    log["recovery_events"] = state["recovery_events"]
    log["commits"] = state["commits"]
    log["total"] = round(time.time() - t0, 1)
    result: dict[str, Any] = state["result"]
    return result


class _StreamWriter:
    def __init__(self, out: Path, split: str):
        self.out, self.split = out, split
        self.offset = 0
        self.files: list[dict[str, Any]] = []
        self._f: BinaryIO | None = None
        self._n = 0
        self._path = out
        self._start = 0

    def _open(self) -> None:
        name = f"{self.split}-tokens-{len(self.files):05d}.bin"
        self._path = self.out / (name + ".partial")
        self._f = self._path.open("wb")
        self._n = 0
        self._start = self.offset
        self.files.append({"file": name})

    def write(self, arr: np.ndarray) -> None:
        pos = 0
        while pos < arr.size:
            if self._f is None or self._n == TOKENS_PER_FILE:
                self._finish()
                self._open()
            assert self._f is not None
            take = min(arr.size - pos, TOKENS_PER_FILE - self._n)
            self._f.write(arr[pos : pos + take].astype("<u2", copy=False).tobytes())
            self._n += take
            self.offset += take
            pos += take

    def _finish(self) -> None:
        if self._f is None:
            return
        self._f.flush()
        os.fsync(self._f.fileno())
        self._f.close()
        final = self.out / self.files[-1]["file"]
        self._path.replace(final)
        self.files[-1].update({"tokens": self._n, "first_offset": self._start, "sha256": sha256_file(final)})
        self._f = None

    def close(self) -> list[dict[str, Any]]:
        self._finish()
        return self.files


def build(root: Path, build_id: str, tokenizer_dir: Path, out: Path) -> dict[str, Any]:
    """Build (or resume) the token shards of ``build_id``; ``out`` appears only when complete."""
    if out.exists():
        raise SystemExit(f"output exists (immutable): {out}")
    acc = root / "accepted" / build_id
    manifest = json.loads((acc / "manifest.json").read_text())
    freeze = json.loads((tokenizer_dir / "freeze.json").read_text())
    tj = tokenizer_dir / "tokenizer.json"
    if sha256_file(tj) != freeze["tokenizer_json_sha256"]:
        raise SystemExit("tokenizer.json does not match its freeze record")
    tok = jtok.load(tj.read_text())
    if tok.get_vocab_size() > 65536:
        raise SystemExit("uint16 shards need a vocabulary below 65,536")
    partial = out.with_name(out.name + ".partial")
    identity = {
        "schema": "j1.token-shards-build-identity.v1",
        "version": VERSION,
        "build": build_id,
        "accepted_manifest_sha256": sha256_file(acc / "manifest.json"),
        "tokenizer_json_sha256": freeze["tokenizer_json_sha256"],
        "tokens_per_file": TOKENS_PER_FILE,
        "n_buckets": N_BUCKETS,
        "order_namespace": ORDER_NAMESPACE,
        "code_sha256": {"tokshards": sha256_file(Path(__file__)), "jtok": sha256_file(Path(jtok.__file__))},
    }
    identity_path = partial / ".identity.json"
    if not identity_path.exists() and _finished_record_matches(partial, identity):
        write_json_atomic(
            identity_path, identity
        )  # killed between dropping the identity and the final rename
    if identity_path.exists():
        if json.loads(identity_path.read_text()) != identity:
            raise SystemExit(f"{partial} belongs to different inputs; remove it to rebuild")
    elif partial.exists() and any(partial.iterdir()):
        raise SystemExit(f"{partial} has no build identity (older or foreign partial output); remove it")
    else:
        write_json_atomic(identity_path, identity)
    remove_stale_temporaries(partial)  # only inside a partial accepted as ours
    t0 = time.time()
    log: dict[str, Any] = {"schema": "j1.token-shards-build-log.v1", "splits": {}}
    record_path = partial / "shards.json"
    if not (record_path.exists() and not verify(partial)):
        splits = {}
        for s in ("train", "validation", "test"):
            log["splits"][s] = {}
            splits[s] = build_split(acc, manifest, tok, s, partial, log=log["splits"][s])
        record = {
            "version": VERSION,
            "build": build_id,
            "accepted_manifest_sha256": identity["accepted_manifest_sha256"],
            "tokenizer": {
                "name": freeze["name"],
                "tokenizer_json_sha256": freeze["tokenizer_json_sha256"],
                "total_vocab": freeze["total_vocab"],
            },
            "dtype": "uint16-le",
            "eos_id": EOS,
            "order": {
                "namespace": ORDER_NAMESPACE,
                "rule": "ascending unit_interval(namespace, doc_id), then doc_id",
            },
            "tokens_per_file": TOKENS_PER_FILE,
            "splits": splits,
        }
        write_json_atomic(record_path, record)
        faults.point("tokshards.record.written")
    problems = verify(partial)
    if problems:  # corruption after commit, not a crash: fail closed rather than publish
        raise SystemExit(f"{partial} fails verification after completion: {problems[:5]}")
    record = json.loads(record_path.read_text())
    log["seconds"] = round(time.time() - t0, 1)
    log["peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    write_json_atomic(partial / "build_log.json", log)
    expected = {"shards.json", "build_log.json", identity_path.name}
    for s in record["splits"].values():
        expected |= {f["file"] for f in s["token_files"]} | {s["doc_index"]["file"]}
    for leftover in sorted(partial.glob(".progress-*.json")):
        leftover.unlink()
    unexpected = sorted(p.name for p in partial.iterdir() if p.name not in expected)
    if unexpected:
        raise SystemExit(f"{partial} holds unexpected files; refusing to publish: {unexpected[:5]}")
    identity_path.unlink()
    faults.point("tokshards.publish.identity_dropped")
    partial.rename(out)
    _fsync_directory(out.parent)
    return dict(record)


def _finished_record_matches(partial: Path, identity: dict[str, Any]) -> bool:
    """True when ``partial`` holds a complete, verifying record of exactly these inputs (identity dropped)."""
    record_path = partial / "shards.json"
    if not record_path.is_file():
        return False
    record = json.loads(record_path.read_text())
    return (
        record.get("build") == identity["build"]
        and record.get("version") == identity["version"]
        and record.get("accepted_manifest_sha256") == identity["accepted_manifest_sha256"]
        and record.get("tokenizer", {}).get("tokenizer_json_sha256") == identity["tokenizer_json_sha256"]
        and record.get("tokens_per_file") == identity["tokens_per_file"]
        and record.get("order", {}).get("namespace") == identity["order_namespace"]
        and not verify(partial)
    )


def verify(out: Path) -> list[str]:
    """Re-hashes every file and checks index/stream consistency; returns problems."""
    rec = json.loads((out / "shards.json").read_text())
    problems = []
    for split, s in rec["splits"].items():
        total = 0
        for f in s["token_files"]:
            p = out / f["file"]
            if not p.is_file() or p.stat().st_size != 2 * f["tokens"] or sha256_file(p) != f["sha256"]:
                problems.append(f"{split}: bad token file {f['file']}")
            total += f["tokens"]
        if total != s["stream_tokens"] or total != s["document_tokens"] + s["separator_tokens"]:
            problems.append(f"{split}: token totals inconsistent")
        idx = out / s["doc_index"]["file"]
        if sha256_file(idx) != s["doc_index"]["sha256"]:
            problems.append(f"{split}: doc index hash mismatch")
    return problems


def export_prefix_excluding(
    src: Path, dst: Path, max_bytes: int, doc_bytes: dict[str, int], exclude: frozenset[str]
) -> dict[str, Any]:
    """``export_prefix`` with documents removed by a contamination adjudication.

    Documents are taken in the shared keyed order, skipping excluded IDs, while the UTF-8 total stays at
    most ``max_bytes`` (same stopping rule). Each kept document's tokens and EOS are copied from ``src``.
    Excluded validation documents are not supported (the caller must rebuild the validation split).
    """
    if dst.exists():
        raise SystemExit(f"output exists (immutable): {dst}")
    rec = json.loads((src / "shards.json").read_text())
    va = rec["splits"]["validation"]
    va_ids = set(
        pq.read_table(src / va["doc_index"]["file"], columns=["doc_id"]).column("doc_id").to_pylist()
    )
    if va_ids & exclude:
        raise SystemExit("excluded documents in the validation split: rebuild validation instead")
    partial = dst.with_name(dst.name + ".partial")
    partial.mkdir(parents=True)
    tr = rec["splits"]["train"]
    idx = pq.read_table(src / tr["doc_index"]["file"])
    ids = idx.column("doc_id").to_pylist()
    offs = idx.column("offset").to_numpy()
    lens = idx.column("n_tokens").to_numpy()
    files = [np.memmap(src / f["file"], dtype="<u2", mode="r") for f in tr["token_files"]]
    starts = np.cumsum([0] + [len(m) for m in files])

    def take(a: int, n: int) -> np.ndarray:
        out = np.empty(n, dtype="<u2")
        k = 0
        while k < n:
            fi = int(np.searchsorted(starts, a + k, side="right")) - 1
            o = a + k - int(starts[fi])
            t = min(n - k, len(files[fi]) - o)
            out[k : k + t] = files[fi][o : o + t]
            k += t
        return out

    total, keep, skipped = 0, [], []
    for i, d in enumerate(ids):
        if d in exclude:
            skipped.append(d)
            continue
        if total + doc_bytes[d] > max_bytes:
            break
        total += doc_bytes[d]
        keep.append(i)
    writer = _StreamWriter(partial, "train")
    new_off, pos = [], 0
    for i in keep:
        chunk = take(int(offs[i]), int(lens[i]) + 1)
        if chunk[-1] != EOS:
            raise SystemExit(f"document {ids[i]} is not followed by EOS in {src}")
        writer.write(chunk)
        new_off.append(pos)
        pos += chunk.size
    tfiles = writer.close()
    sub = idx.take(pa.array(keep)).set_column(
        idx.column_names.index("offset"), "offset", pa.array(new_off, pa.int64())
    )
    pq.write_table(sub, partial / "train-docs.parquet", compression="zstd")
    by_source: dict[str, dict[str, int]] = defaultdict(lambda: {"documents": 0, "tokens": 0})
    for s_id, n in zip(sub.column("source_id").to_pylist(), sub.column("n_tokens").to_pylist(), strict=True):
        by_source[s_id]["documents"] += 1
        by_source[s_id]["tokens"] += n
    for f in [*va["token_files"], va["doc_index"]]:
        (partial / f["file"]).write_bytes((src / f["file"]).read_bytes())
    k = len(keep)
    out = {
        **rec,
        "prefix_of": {
            "shards_json_sha256": sha256_file(src / "shards.json"),
            "train_documents": k,
            "train_text_bytes": total,
            "excluded_doc_ids": sorted(skipped),
            "exclusion_list_sha256": sha256_json(sorted(exclude)),
        },
        "splits": {
            "train": {
                "split": "train",
                "documents": k,
                "document_tokens": pos - k,
                "separator_tokens": k,
                "stream_tokens": pos,
                "by_source": dict(sorted(by_source.items())),
                "token_files": tfiles,
                "doc_index": {
                    "file": "train-docs.parquet",
                    "sha256": sha256_file(partial / "train-docs.parquet"),
                    "rows": k,
                },
            },
            "validation": va,
        },
    }
    write_json_atomic(partial / "shards.json", out)
    partial.rename(dst)
    return out


def export_prefix(src: Path, dst: Path, max_bytes: int, doc_bytes: dict[str, int]) -> dict[str, Any]:
    """A smaller j1.token-shards.v1 directory holding the train-stream prefix of whole documents
    whose UTF-8 text totals at most ``max_bytes``, plus the full validation split.

    Every candidate tokenizer shares the document order, so equal ``max_bytes`` gives every
    candidate the same documents. That is the equal-bytes condition for tokenizer pilots.
    ``doc_bytes`` maps doc_id to UTF-8 byte length (from the accepted build).
    """
    if dst.exists():
        raise SystemExit(f"output exists (immutable): {dst}")
    rec = json.loads((src / "shards.json").read_text())
    partial = dst.with_name(dst.name + ".partial")
    partial.mkdir(parents=True)
    tr = rec["splits"]["train"]
    idx = pq.read_table(src / tr["doc_index"]["file"])
    ids = idx.column("doc_id").to_pylist()
    total, k = 0, 0
    while k < len(ids) and total + doc_bytes[ids[k]] <= max_bytes:
        total += doc_bytes[ids[k]]
        k += 1
    n_tok = int(idx.column("offset")[k - 1].as_py() + idx.column("n_tokens")[k - 1].as_py() + 1)
    writer = _StreamWriter(partial, "train")
    pos = 0
    for f in tr["token_files"]:
        if pos >= n_tok:
            break
        arr = np.fromfile(src / f["file"], dtype="<u2", count=min(f["tokens"], n_tok - pos))
        writer.write(arr)
        pos += arr.size
    files = writer.close()
    sub = idx.slice(0, k)
    pq.write_table(sub, partial / "train-docs.parquet", compression="zstd")
    by_source: dict[str, dict[str, int]] = defaultdict(lambda: {"documents": 0, "tokens": 0})
    for s_id, n in zip(sub.column("source_id").to_pylist(), sub.column("n_tokens").to_pylist(), strict=True):
        by_source[s_id]["documents"] += 1
        by_source[s_id]["tokens"] += n
    va = rec["splits"]["validation"]
    for f in [*va["token_files"], va["doc_index"]]:
        (partial / f["file"]).write_bytes((src / f["file"]).read_bytes())
    out = {
        **rec,
        "prefix_of": {
            "shards_json_sha256": sha256_file(src / "shards.json"),
            "train_documents": k,
            "train_text_bytes": total,
        },
        "splits": {
            "train": {
                "split": "train",
                "documents": k,
                "document_tokens": n_tok - k,
                "separator_tokens": k,
                "stream_tokens": n_tok,
                "by_source": dict(sorted(by_source.items())),
                "token_files": files,
                "doc_index": {
                    "file": "train-docs.parquet",
                    "sha256": sha256_file(partial / "train-docs.parquet"),
                    "rows": k,
                },
            },
            "validation": va,
        },
    }
    write_json_atomic(partial / "shards.json", out)
    partial.rename(dst)
    return out
