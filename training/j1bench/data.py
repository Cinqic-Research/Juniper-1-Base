"""Deterministic, exactly-resumable token sources for qualification.

Two sources share one contract: ``next_batch()`` returns (inputs, labels,
useful_tokens) and ``state_dict()``/``load_state_dict()`` capture the exact
consumption position, so a resumed run yields the identical next batch.

* SyntheticPacked: documents are generated on demand from (seed, doc_index), so
  no data is stored. Each document follows a simple learnable rule so that
  loss decreases and a broken training step is visible.
* MemmapPacked: reads a pre-tokenized uint16/uint32 token file sequentially
  (streamed from disk through the page cache), as a stand-in for the future
  pre-tokenized corpus.

No real corpus data is used; nothing here is a canonical data pipeline.
"""

from __future__ import annotations

import hashlib
import os
import time
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

IGNORE = -100
# Bounded chunk (tokens) for streaming passes over a token stream; 4M uint16 tokens = 8 MiB of input and at most
# 32 MiB of int64 temporaries inside np.bincount. Accounting over the 3.4B-token stream never holds more than this.
ACCOUNTING_CHUNK_TOKENS = 1 << 22
# Readers that hash and load token files in parallel (hashlib and np.fromfile release the GIL).
LOAD_WORKERS = 4


def _copy_doc(seed: int, index: int, vocab: int, eos: int, min_len: int, max_len: int) -> np.ndarray:
    # Random segment followed by its exact repeat: the first half is unpredictable
    # (~ln V nats/token), the second half is predictable by in-context copying,
    # so a learning model approaches ~ln(V)/2 average loss.
    rng = np.random.Generator(np.random.PCG64([seed, index]))
    half = max(1, int(rng.integers(min_len, max_len + 1)) // 2)
    seg = rng.integers(0, vocab - 1, size=half)
    return np.concatenate([seg, seg, [eos]]).astype(np.int64)


_PERMS: dict = {}


def _perm_doc(seed: int, index: int, vocab: int, eos: int, min_len: int, max_len: int) -> np.ndarray:
    # Fixed corpus-wide permutation chain with 10% noise: t[i+1] = P[t[i]] (or random).
    # Learnable without in-context ability; entropy floor ~= 0.9*0.105 + 0.1*ln(V-1) nats/token.
    v = vocab - 1
    if (seed, v) not in _PERMS:
        _PERMS[(seed, v)] = np.random.Generator(np.random.PCG64([seed, 2**31])).permutation(v)
    perm = _PERMS[(seed, v)]
    rng = np.random.Generator(np.random.PCG64([seed, index]))
    n = int(rng.integers(min_len, max_len + 1))
    noise = rng.random(n) < 0.1
    rnd = rng.integers(0, v, size=n)
    t = np.empty(n + 1, dtype=np.int64)
    t[0] = rnd[0]
    for i in range(1, n):
        t[i] = rnd[i] if noise[i] else perm[t[i - 1]]
    t[n] = eos
    return t


def _doc(seed: int, index: int, vocab: int, eos: int, min_len: int, max_len: int) -> np.ndarray:
    rng = np.random.Generator(np.random.PCG64([seed, index]))
    n = int(rng.integers(min_len, max_len + 1))
    # Per-document arithmetic progression mod V' with 10% token noise:
    # t[i] = (c + b*i) mod V'. Learnable in context (next = prev + b), and
    # vectorised so data generation never bottlenecks the GPU.
    v = vocab - 1  # eos = vocab-1 is reserved
    b = int(rng.integers(1, v))
    c = int(rng.integers(0, v))
    t = (c + b * np.arange(n, dtype=np.int64)) % v
    noise = rng.random(n) < 0.1
    t[noise] = rng.integers(0, v, size=int(noise.sum()))
    out = np.empty(n + 1, dtype=np.int64)
    out[:n] = t
    out[n] = eos
    return out


class SyntheticPacked:
    """Packs synthetic documents into rows of seq_len+1 tokens without padding.

    ``pad_mode=True`` instead places one document per row and pads with
    IGNORE labels, to exercise padding accounting.
    """

    def __init__(
        self,
        vocab: int,
        seq_len: int,
        batch_size: int,
        seed: int = 1234,
        min_len: int = 32,
        max_len: int = 3000,
        pad_mode: bool = False,
        kind: str = "affine",
    ):
        self.vocab, self.seq_len, self.batch_size, self.seed = vocab, seq_len, batch_size, seed
        self.min_len, self.max_len, self.pad_mode = min_len, max_len, pad_mode
        self.kind = kind
        self._gen = {"affine": _doc, "copy": _copy_doc, "perm": _perm_doc}[kind]
        self.eos = vocab - 1
        self.doc_index = 0
        self.offset = 0
        self.batches = 0
        self._cache: tuple[int, np.ndarray] | None = None

    def identity(self) -> dict:
        return {
            "kind": "synthetic_packed_v1",
            "doc_kind": self.kind,
            "vocab": self.vocab,
            "seq_len": self.seq_len,
            "batch_size": self.batch_size,
            "seed": self.seed,
            "min_len": self.min_len,
            "max_len": self.max_len,
            "pad_mode": self.pad_mode,
        }

    def _get(self, i: int) -> np.ndarray:
        if self._cache is None or self._cache[0] != i:
            self._cache = (i, self._gen(self.seed, i, self.vocab, self.eos, self.min_len, self.max_len))
        return self._cache[1]

    def _row_packed(self) -> np.ndarray:
        need = self.seq_len + 1
        parts, have = [], 0
        while have < need:
            d = self._get(self.doc_index)
            take = min(len(d) - self.offset, need - have)
            parts.append(d[self.offset : self.offset + take])
            have += take
            self.offset += take
            if self.offset == len(d):
                self.doc_index += 1
                self.offset = 0
        # Rows overlap by one token so every token is predicted exactly once.
        self.offset -= 1
        if self.offset < 0:
            self.doc_index -= 1
            self.offset = len(self._get(self.doc_index)) - 1
        return np.concatenate(parts)

    def _row_padded(self) -> tuple[np.ndarray, np.ndarray]:
        d = self._get(self.doc_index)[: self.seq_len + 1]
        self.doc_index += 1
        row = np.full(self.seq_len + 1, self.eos, dtype=np.int64)
        row[: len(d)] = d
        lab = row[1:].copy()
        lab[len(d) - 1 :] = IGNORE
        return row, lab

    def next_batch(self):
        rows, labs = [], []
        for _ in range(self.batch_size):
            if self.pad_mode:
                r, lab = self._row_padded()
            else:
                r = self._row_packed()
                lab = r[1:]
            rows.append(r[:-1])
            labs.append(lab)
        self.batches += 1
        x = torch.from_numpy(np.stack(rows))
        y = torch.from_numpy(np.stack(labs))
        return x, y, int((y != IGNORE).sum())

    def state_dict(self) -> dict:
        return {"doc_index": self.doc_index, "offset": self.offset, "batches": self.batches}

    def load_state_dict(self, s: dict) -> None:
        self.doc_index, self.offset, self.batches = s["doc_index"], s["offset"], s["batches"]
        self._cache = None


def write_token_file(path: str, n_tokens: int, vocab: int, seed: int = 99) -> str:
    """Writes a synthetic pre-tokenized file built from SyntheticPacked docs; returns its sha256."""
    dtype = np.uint16 if vocab <= 65536 else np.uint32
    src = SyntheticPacked(vocab, 4095, 1, seed=seed)
    h = hashlib.sha256()
    written = 0
    with Path(path).open("wb") as f:
        while written < n_tokens:
            x, _, _ = src.next_batch()
            chunk = x.numpy().astype(dtype).ravel()[: n_tokens - written]
            b = chunk.tobytes()
            f.write(b)
            h.update(b)
            written += len(chunk)
    return h.hexdigest()


class MemmapPacked:
    """Sequential contiguous windows over a token file; wraps around at the end."""

    def __init__(self, path: str, vocab: int, seq_len: int, batch_size: int, file_sha256: str | None = None):
        self.path, self.vocab, self.seq_len, self.batch_size = path, vocab, seq_len, batch_size
        self.dtype = np.uint16 if vocab <= 65536 else np.uint32
        h = hashlib.sha256()
        with Path(path).open("rb") as f:
            while chunk := f.read(16 << 20):
                h.update(chunk)
        actual_sha256 = h.hexdigest()
        if file_sha256 is not None and actual_sha256 != file_sha256:
            raise ValueError(f"token file sha256 mismatch: {actual_sha256} != {file_sha256}")
        self.tokens = np.memmap(path, dtype=self.dtype, mode="r")
        if len(self.tokens) < seq_len + 1:
            raise ValueError("token file is too short for a training window")
        self.file_sha256 = actual_sha256
        self.pos = 0
        self.batches = 0
        self.epoch = 0

    def identity(self) -> dict:
        return {
            "kind": "memmap_packed_v1",
            "path": self.path,
            "n_tokens": len(self.tokens),
            "file_sha256": self.file_sha256,
            "vocab": self.vocab,
            "seq_len": self.seq_len,
            "batch_size": self.batch_size,
        }

    def next_batch(self):
        need = self.seq_len + 1
        rows = []
        for _ in range(self.batch_size):
            if self.pos + need > len(self.tokens):
                self.pos = 0
                self.epoch += 1
            rows.append(np.asarray(self.tokens[self.pos : self.pos + need], dtype=np.int64))
            self.pos += self.seq_len
        self.batches += 1
        t = torch.from_numpy(np.stack(rows))
        x, y = t[:, :-1], t[:, 1:]
        return x, y, int(y.numel())

    def state_dict(self) -> dict:
        return {"pos": self.pos, "batches": self.batches, "epoch": self.epoch}

    def load_state_dict(self, s: dict) -> None:
        self.pos, self.batches, self.epoch = s["pos"], s["batches"], s["epoch"]


def batches_fingerprint(batches: Sequence[tuple]) -> str:
    """sha256 over the concatenated x and y bytes of every microbatch, in order (review N-06).

    ``batches`` holds ``(x, y, ...)`` tuples as returned by ``next_batch``; with gradient accumulation every
    microbatch of an update is covered, so a resume or equivalence check on ``batch_fp`` detects a divergence in
    any of them. For one batch this equals ``batch_fingerprint(x, y)``.
    """
    h = hashlib.sha256()
    for b in batches:
        h.update(b[0].cpu().numpy().tobytes())
        h.update(b[1].cpu().numpy().tobytes())
    return h.hexdigest()[:16]


def batch_fingerprint(x: torch.Tensor, y: torch.Tensor) -> str:
    return batches_fingerprint([(x, y)])


class TokenStream:
    """Training windows over a ``j1.token-shards.v1`` split (Juniper-1-Data ``tokshards``).

    The split's token files form one logical stream of documents separated by
    ``<|endoftext|>``. It is cut into windows of ``seq_len + 1`` tokens that overlap by one,
    so every token after the first is a target exactly once per epoch. Documents are
    already in keyed-hash order. Each epoch additionally visits the windows in a
    permutation seeded by ``(seed, epoch)``. ``limit_tokens`` restricts training to the
    stream prefix, so pilots of different lengths share data. State is ``(epoch, cursor)``;
    restoring it reproduces the next batch exactly.

    Cross-document attention inside a window is allowed (EOS-separated packing), and
    every position is trained. This policy is recorded in the run contract.
    """

    def __init__(
        self,
        shards_dir: str,
        split: str,
        seq_len: int,
        batch_size: int,
        seed: int = 0,
        limit_tokens: int | None = None,
        shuffle: bool = True,
        verify: bool = False,
        preload: bool = False,
        progress: Callable[..., None] | None = None,
    ):
        import json

        self.dir = Path(shards_dir)
        meta_bytes = (self.dir / "shards.json").read_bytes()
        self.meta = json.loads(meta_bytes)
        self.meta_sha256 = hashlib.sha256(meta_bytes).hexdigest()
        self.split, self.seq_len, self.batch_size, self.seed, self.shuffle = split, seq_len, batch_size, seed, shuffle
        s = self.meta["splits"][split]
        self.eos = int(self.meta.get("eos_id", 0))
        # Truncated or swapped files are refused before training (size always, SHA-256 when verify).
        for f in s["token_files"]:
            p = self.dir / f["file"]
            if not p.is_file() or p.stat().st_size != 2 * f["tokens"]:
                raise ValueError(f"token file {f['file']} is missing or not {f['tokens']} tokens")
        # One read per file (B17): with preload the bytes are read into RAM once and hashed from that buffer;
        # without preload the memmap is hashed in bounded chunks. Shuffled windows over a memmap turn into random
        # disk reads, which a spinning disk cannot serve at A100 consumption rates, hence preload on the launch host.
        # Files are processed by a bounded pool; the result order is the manifest order.
        t0 = time.perf_counter()
        self.bytes_read = 0
        self.bytes_hashed = 0

        def load_one(i: int) -> np.ndarray:
            f = s["token_files"][i]
            p = self.dir / f["file"]
            arr: np.ndarray
            if preload:
                arr = np.fromfile(p, dtype="<u2")
                if verify and f.get("sha256") and hashlib.sha256(memoryview(arr).cast("B")).hexdigest() != f["sha256"]:
                    raise ValueError(f"token file {f['file']} sha256 mismatch")
            else:
                arr = np.memmap(p, dtype="<u2", mode="r")
                if verify and f.get("sha256"):
                    h = hashlib.sha256()
                    for k in range(0, len(arr), 1 << 23):
                        h.update(memoryview(np.ascontiguousarray(arr[k : k + (1 << 23)])).cast("B"))
                    if h.hexdigest() != f["sha256"]:
                        raise ValueError(f"token file {f['file']} sha256 mismatch")
            if progress is not None:
                progress(
                    "token-file",
                    file=f["file"],
                    tokens=len(arr),
                    bytes=2 * len(arr),
                    hashed=bool(verify and f.get("sha256")),
                    loaded=preload,
                )
            return arr

        n_files = len(s["token_files"])
        with ThreadPoolExecutor(max_workers=max(1, min(LOAD_WORKERS, n_files))) as pool:
            self.files = list(pool.map(load_one, range(n_files)))
        hashed = [a for a, f in zip(self.files, s["token_files"], strict=True) if verify and f.get("sha256")]
        self.bytes_hashed = sum(2 * len(a) for a in hashed)
        self.bytes_read = sum(2 * len(a) for a in self.files) if (preload or verify) else 0
        self.load_seconds = time.perf_counter() - t0
        self.preload = preload
        self.starts = np.cumsum([0] + [len(f) for f in self.files])
        n = int(self.starts[-1]) if limit_tokens is None else min(int(limit_tokens), int(self.starts[-1]))
        self.n_tokens = n
        self.n_windows = (n - 1) // seq_len
        if self.n_windows < batch_size:
            raise ValueError("split too small for one batch")
        self.vocab = self.meta["tokenizer"]["total_vocab"]
        self.epoch, self.cursor, self.batches = 0, 0, 0
        self._order: tuple[int, np.ndarray] | None = None

    def identity(self) -> dict:
        return {
            "kind": "token_stream_v1",
            "shards_json_sha256": self.meta_sha256,
            "tokenizer_json_sha256": self.meta["tokenizer"]["tokenizer_json_sha256"],
            "build": self.meta["build"],
            "split": self.split,
            "limit_tokens": self.n_tokens,
            "seq_len": self.seq_len,
            "batch_size": self.batch_size,
            "seed": self.seed,
            "shuffle": self.shuffle,
            "packing": "EOS-separated documents; cross-document attention allowed; all positions trained",
            "preload": self.preload,
        }

    def tokens(self, start: int, length: int) -> np.ndarray:
        out = np.empty(length, dtype=np.int64)
        pos, k = start, 0
        while k < length:
            f = int(np.searchsorted(self.starts, pos, side="right")) - 1
            off = pos - int(self.starts[f])
            take = min(length - k, len(self.files[f]) - off)
            out[k : k + take] = self.files[f][off : off + take]
            k += take
            pos += take
        return out

    def iter_chunks(self, start: int, length: int, chunk: int = ACCOUNTING_CHUNK_TOKENS) -> Iterator[np.ndarray]:
        """Stream positions ``[start, start+length)`` as uint16 views of at most ``chunk`` tokens (no int64 copy).

        Chunks never cross a token-file boundary, so the yielded arrays are views of the loaded files.
        """
        if chunk <= 0:
            raise ValueError("chunk must be positive")
        if start < 0 or length < 0 or start + length > int(self.starts[-1]):
            raise ValueError(f"range [{start}, {start + length}) outside the stream of {int(self.starts[-1])} tokens")
        pos, end = start, start + length
        while pos < end:
            f = int(np.searchsorted(self.starts, pos, side="right")) - 1
            off = pos - int(self.starts[f])
            take = min(end - pos, len(self.files[f]) - off, chunk)
            yield self.files[f][off : off + take]
            pos += take

    def truncate(self, limit_tokens: int) -> None:
        """Restrict training to the stream prefix without rebuilding (reloading, rehashing) the stream.

        Equivalent to constructing the stream with ``limit_tokens``: it only moves the logical bounds
        (``n_tokens``, ``n_windows``) and the window permutation; the loaded files are untouched. Refused once a
        batch has been consumed, so a resume cursor can never refer to a different window set.
        """
        if self.batches or self.cursor or self.epoch:
            raise ValueError("truncate() is only valid before the first batch")
        full = int(self.starts[-1])
        n = min(int(limit_tokens), full)
        n_windows = (n - 1) // self.seq_len
        if n_windows < self.batch_size:
            raise ValueError("split too small for one batch")
        self.n_tokens, self.n_windows, self._order = n, n_windows, None

    def _window_order(self) -> np.ndarray:
        if self._order is None or self._order[0] != self.epoch:
            idx = np.arange(self.n_windows, dtype=np.int64)
            if self.shuffle:
                np.random.Generator(np.random.PCG64([self.seed, self.epoch, 0x6A31])).shuffle(idx)
            self._order = (self.epoch, idx)
        return self._order[1]

    def next_batch(self):
        rows = []
        for _ in range(self.batch_size):
            if self.cursor == self.n_windows:
                self.epoch += 1
                self.cursor = 0
            w = int(self._window_order()[self.cursor])
            self.cursor += 1
            rows.append(self.tokens(w * self.seq_len, self.seq_len + 1))
        self.batches += 1
        t = torch.from_numpy(np.stack(rows))
        x, y = t[:, :-1], t[:, 1:]
        return x, y, int(y.numel())

    def state_dict(self) -> dict:
        return {"epoch": self.epoch, "cursor": self.cursor, "batches": self.batches}

    def load_state_dict(self, s: dict) -> None:
        self.epoch, self.cursor, self.batches = s["epoch"], s["cursor"], s["batches"]
        self._order = None

    def eval_windows(self, n_windows: int) -> list[tuple[np.ndarray, np.ndarray]]:
        """The first ``n_windows`` disjoint windows (inputs+targets, source labels per target)."""
        import pyarrow.parquet as pq

        idx = pq.read_table(
            self.dir / self.meta["splits"][self.split]["doc_index"]["file"], columns=["source_id", "offset", "n_tokens"]
        )
        offsets = idx.column("offset").to_numpy()
        sources = idx.column("source_id").to_pylist()
        names = sorted(set(sources))
        code = np.array([names.index(s) for s in sources], dtype=np.int64)
        self.source_names = names
        out = []
        for w in range(min(n_windows, self.n_windows)):
            toks = self.tokens(w * self.seq_len, self.seq_len + 1)
            pos = np.arange(w * self.seq_len + 1, w * self.seq_len + self.seq_len + 1)
            out.append((toks, code[np.searchsorted(offsets, pos, side="right") - 1]))
        return out


def token_byte_lengths(tokenizer_json: str | Path, vocab_size: int) -> np.ndarray:
    """UTF-8 source bytes represented by each token ID of a ``j1.tokenizer`` byte-fallback BPE.

    Control IDs (``added_tokens`` marked special) carry no source text and count 0 bytes;
    byte-fallback tokens ``<0xNN>`` count 1 byte; learned tokens are literal UTF-8 strings
    (the tokenizer has no normalizer and no byte-to-unicode remapping, which is checked here).
    ``tools/verify_bpb.py`` verifies the table against decoded documents and source bytes.
    """
    import json
    import re

    t = json.loads(Path(tokenizer_json).read_text())
    if t.get("normalizer") is not None or t["model"].get("type") != "BPE" or not t["model"].get("byte_fallback"):
        raise ValueError("token byte table supports only unnormalized byte-fallback BPE tokenizers")
    dec = json.dumps(t.get("decoder"))
    if "ByteLevel" in dec or "Metaspace" in dec or "Replace" in dec:
        raise ValueError(f"decoder rewrites token text; byte table would be wrong: {dec}")
    special = {a["id"] for a in t.get("added_tokens", []) if a.get("special")}
    out = np.full(vocab_size, -1, dtype=np.int64)
    byte_re = re.compile(r"^<0x[0-9A-F]{2}>$")
    for tok, i in t["model"]["vocab"].items():
        out[i] = 0 if i in special else 1 if byte_re.match(tok) else len(tok.encode("utf-8"))
    for a in t.get("added_tokens", []):
        out[a["id"]] = 0
    if (out < 0).any():
        raise ValueError(f"{int((out < 0).sum())} IDs below vocab_size have no tokenizer entry")
    return out


# Parallel histogram workers (forked processes; np.bincount holds the GIL, so threads do not scale). Each worker
# counts one file-aligned segment of at most HISTOGRAM_SEGMENT_TOKENS tokens in bounded chunks; the partial
# histograms are summed in the parent. Integer counts, so the result is identical to the serial pass.
HISTOGRAM_WORKERS = max(1, min(8, (os.cpu_count() or 2) // 2))
HISTOGRAM_SEGMENT_TOKENS = 1 << 27
_FORK_STREAM: TokenStream | None = None


def _histogram_serial(stream: TokenStream, start: int, length: int, chunk: int, progress) -> np.ndarray:
    counts = np.zeros(stream.vocab, dtype=np.int64)
    done = 0
    for view in stream.iter_chunks(start, length, chunk):
        h = np.bincount(view, minlength=stream.vocab)
        if len(h) > stream.vocab:
            bad = int(np.flatnonzero(h[stream.vocab :])[0]) + stream.vocab
            raise ValueError(f"token id {bad} at or above the vocabulary size {stream.vocab} (corrupt token file)")
        counts += h
        done += len(view)
        if progress is not None:
            progress("histogram", tokens=done, of=length)
    if done != length:
        raise AssertionError(f"histogram covered {done} of {length} tokens")
    return counts


def _histogram_segment(args: tuple[int, int, int]) -> np.ndarray:
    start, length, chunk = args
    assert _FORK_STREAM is not None
    return _histogram_serial(_FORK_STREAM, start, length, chunk, None)


def _segments(stream: TokenStream, start: int, length: int, max_len: int) -> list[tuple[int, int]]:
    """File-aligned sub-ranges of at most ``max_len`` tokens covering ``[start, start+length)``."""
    out = []
    pos, end = start, start + length
    while pos < end:
        f = int(np.searchsorted(stream.starts, pos, side="right")) - 1
        take = min(end - pos, int(stream.starts[f + 1]) - pos, max_len)
        out.append((pos, take))
        pos += take
    return out


def token_histogram(
    stream: TokenStream,
    start: int,
    length: int,
    chunk: int = ACCOUNTING_CHUNK_TOKENS,
    progress: Callable[..., None] | None = None,
    workers: int | None = None,
) -> np.ndarray:
    """Exact per-ID counts of stream positions ``[start, start+length)`` in bounded chunks (B17).

    Returns an int64 array of ``stream.vocab`` counts. A token ID at or above the vocabulary size is corruption
    (the shards are uint16 and the tokenizer has ``total_vocab`` IDs) and is refused. Ranges longer than one
    segment are counted by ``workers`` forked processes (``J1_HIST_WORKERS`` overrides the default), each over a
    bounded segment; the partial counts are summed, so the result equals the serial pass exactly.
    """
    global _FORK_STREAM
    if workers is None:
        workers = int(os.environ.get("J1_HIST_WORKERS", HISTOGRAM_WORKERS))
    segs = _segments(stream, start, length, HISTOGRAM_SEGMENT_TOKENS)
    if workers <= 1 or len(segs) <= 1 or not hasattr(os, "fork"):
        return _histogram_serial(stream, start, length, chunk, progress)
    import multiprocessing as mp
    import warnings

    counts = np.zeros(stream.vocab, dtype=np.int64)
    done = 0
    _FORK_STREAM = stream
    try:
        with warnings.catch_warnings():
            # Python warns about fork() in a multi-threaded process (the time-log heartbeat thread is alive). The
            # children only run np.bincount over inherited read-only arrays and take no Python lock, so this is safe.
            warnings.simplefilter("ignore", DeprecationWarning)
            pool = mp.get_context("fork").Pool(min(workers, len(segs)))
        with pool:
            for (_, seg_len), h in zip(
                segs, pool.imap(_histogram_segment, [(s, n, chunk) for s, n in segs]), strict=True
            ):
                counts += h
                done += seg_len
                if progress is not None:
                    progress("histogram", tokens=done, of=length)
    finally:
        _FORK_STREAM = None
    if done != length:
        raise AssertionError(f"histogram covered {done} of {length} tokens")
    return counts


def _exact_dot(counts: np.ndarray, token_bytes: np.ndarray) -> int:
    """sum(counts * token_bytes) as an exact Python integer (no int64 overflow is possible)."""
    if len(token_bytes) != len(counts):
        raise ValueError(f"byte table has {len(token_bytes)} entries for a vocabulary of {len(counts)}")
    if (token_bytes < 0).any():
        raise ValueError("byte table has negative entries")
    nz = np.flatnonzero(counts)
    return sum(int(counts[i]) * int(token_bytes[i]) for i in nz)


def consumed_prefix_record(
    stream: TokenStream,
    token_bytes: np.ndarray | None,
    chunk: int = ACCOUNTING_CHUNK_TOKENS,
    progress: Callable[..., None] | None = None,
) -> dict:
    """Exact accounting of what one full epoch over ``stream`` trains on.

    With ``limit_tokens`` chosen so the window count is a whole number of updates, one epoch
    visits every window once; the targets are stream positions ``1 .. n_tokens-1``. The record
    gives the consumed targets and their source bytes, the dropped tail of the declared prefix,
    and the documents covered, so equal-byte comparisons can be checked rather than assumed.

    Counts are accumulated from bounded chunks into vocabulary-sized int64 histograms (B17): the
    full stream is never materialised as int64 and no per-target byte lookup array is built. The
    byte sums are exact Python integers. The record's keys and values are identical to the
    per-target formulation.
    """
    import pyarrow.parquet as pq

    full = int(stream.starts[-1])
    n = stream.n_tokens
    n_targets = stream.n_windows * stream.seq_len
    rec: dict = {
        "policy": "one epoch over every window of the truncated prefix; all positions trained; EOS targets included",
        "prefix_stream_tokens": full,
        "limit_tokens": n,
        "windows": stream.n_windows,
        "target_tokens": n_targets,
        "dropped_tail_tokens": full - n,
    }
    # Targets are positions [1, n_targets]; the dropped tail is [n, full); position 0 is the first input token.
    # n == n_targets + 1 whenever the prefix was truncated to whole windows; the ranges are kept distinct so the
    # record is exact for any limit_tokens.
    targets = token_histogram(stream, 1, n_targets, chunk, progress)
    rec["eos_targets"] = int(targets[stream.eos])
    if token_bytes is not None:
        token_bytes = np.asarray(token_bytes)
        rec["target_bytes"] = _exact_dot(targets, token_bytes)
        if full > n:
            tail = token_histogram(stream, n, full - n, chunk, progress)
            rec["dropped_tail_bytes"] = _exact_dot(tail, token_bytes)
        else:
            rec["dropped_tail_bytes"] = 0
        first = token_histogram(stream, 0, 1, chunk)
        rec["prefix_bytes"] = rec["target_bytes"] + rec["dropped_tail_bytes"] + _exact_dot(first, token_bytes)
    di = stream.meta["splits"][stream.split]["doc_index"]
    idx = pq.read_table(stream.dir / di["file"], columns=["offset", "n_tokens"])
    off = idx.column("offset").to_numpy()
    ln = idx.column("n_tokens").to_numpy()
    last = n - 1  # last consumed stream position (a target)
    rec["documents_in_prefix"] = len(off)
    rec["documents_fully_consumed"] = int(((off + ln) <= last).sum())  # doc tokens end before the cutoff
    rec["documents_partially_consumed"] = int(((off <= last) & ((off + ln) > last)).sum())
    return rec


def full_split_eval(stream: TokenStream) -> dict:
    """Every token of every document in ``stream``'s split as a scored target, in fixed windows.

    The split stream is ``doc0 EOS doc1 EOS ...``. One EOS is prepended as the context-free start,
    so the first token of the first document is also scored. Windows hold ``seq_len`` targets and
    overlap by one input token (the same packing as training: cross-document attention allowed).
    The final partial window is padded with IGNORE targets. Returns inputs, targets, per-target
    source codes, document ordinals, and the identity of the scored set. Because each candidate
    tokenizer's shards hold the same documents, the scored source bytes are identical across
    tokenizers by construction; ``scored_bytes`` lets callers check it.
    """
    import pyarrow.parquet as pq

    s = stream.meta["splits"][stream.split]
    idx = pq.read_table(stream.dir / s["doc_index"]["file"])
    off = idx.column("offset").to_numpy().astype(np.int64)
    sources = idx.column("source_id").to_pylist()
    names = sorted(set(sources))
    code = np.array([names.index(x) for x in sources], dtype=np.int64)
    full = int(stream.starts[-1])
    # Leading EOS (the split's separator id, ``eos_id`` in shards.json; review N-07): shifts stream positions by +1.
    seq = np.concatenate([[stream.eos], stream.tokens(0, full)])
    targets_pos = np.arange(1, len(seq))  # positions in seq; stream position = p - 1
    doc_of = np.searchsorted(off, targets_pos - 1, side="right") - 1
    T = stream.seq_len
    n_t = len(seq) - 1
    n_w = -(-n_t // T)
    x = np.zeros((n_w, T), dtype=np.int64)
    y = np.full((n_w, T), IGNORE, dtype=np.int64)
    src = np.full((n_w, T), -1, dtype=np.int64)
    doc = np.full((n_w, T), -1, dtype=np.int64)
    for w in range(n_w):
        a, b = w * T, min(w * T + T, n_t)
        x[w, : b - a] = seq[a:b]
        y[w, : b - a] = seq[a + 1 : b + 1]
        src[w, : b - a] = code[doc_of[a:b]]
        doc[w, : b - a] = doc_of[a:b]
    ident = {
        "kind": "full_split_eval_v1",
        "split": stream.split,
        "documents": len(off),
        "doc_index_sha256": s["doc_index"].get("sha256"),
        "token_files_sha256": [f.get("sha256") for f in s["token_files"]],
        "scored_targets": int(n_t),
        "eos_targets": int((seq[1:] == stream.eos).sum()),
        "windows": int(n_w),
        "context_policy": "fixed windows of seq_len targets over the EOS-separated split stream, "
        "leading EOS, cross-document attention allowed, no overlap beyond one input token",
    }
    return {"x": x, "y": y, "src": src, "doc": doc, "source_names": names, "identity": ident}
