"""Evaluation-contamination index and matcher.

Each protected evaluation *item* (one field group of one benchmark record) is
indexed as informative 13-grams when it is long enough. Short items that meet
strict length and information thresholds are protected by exact normalized
word-sequence matching. A training document is contaminated when

* it covers at least ``min_coverage`` of some item's 13-grams, or
* it contains a registered canary string.

Exact matching is used only for complete items of 6-12 words and at least 48
normalized characters, with at least five informative words. This catches
copied short prompts and code fragments while avoiding partial matches on
common short phrases.

Two refinements of plain "any shared 13-gram" matching (GPT-3, App. C) came
from auditing matches on real documents:

* N-grams with fewer than ``min_informative_words`` alphabetic words of two or
  more characters (digit runs, alphabet runs, zero padding) are not indexed;
  they matched unrelated tables and listings.
* Requiring per-item coverage, rather than one shared n-gram, stops famous
  quotations that MMLU history passages embed from flagging every document
  that quotes the same primary source, while a copied question, solution or
  prompt still covers most of its item.

Items with no informative 13-gram (shorter than 13 words, or purely numeric)
cannot be matched this way; their counts are reported in the index manifest.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from j1data.hashing import sha256_bytes
from j1data.ngrams import U64, ngram_hashes, word_hashes
from j1data.normalize import signature_words

VERSION = "j1.contamination.v3"


@dataclass
class ContaminationHit:
    best_coverage: float
    benchmarks: list[str]
    canaries: list[str]
    short_exact_benchmarks: list[str]

    @property
    def contaminated(self) -> bool:
        return bool(self.benchmarks or self.canaries)


def informative_ngrams(words: list[str], n: int, min_informative_words: int) -> U64:
    if len(words) < n:
        return np.empty(0, dtype=np.uint64)
    contentful = np.fromiter(
        (len(w) >= 2 and not w.isdigit() for w in words), dtype=np.int32, count=len(words)
    )
    window = np.convolve(contentful, np.ones(n, dtype=np.int32), mode="valid")
    grams = ngram_hashes(word_hashes(words), n)
    return np.unique(grams[window >= min_informative_words])


class ContaminationIndex:
    def __init__(
        self,
        ngram: int,
        min_coverage: float,
        min_informative_words: int,
        hashes: U64,
        item_ids: npt.NDArray[np.int32],
        item_benchmark: npt.NDArray[np.int16],
        item_ngrams: npt.NDArray[np.int32],
        benchmark_names: list[str],
        canaries: list[str],
        short_lengths: npt.NDArray[np.uint8],
        short_hashes: U64,
        short_benchmarks: npt.NDArray[np.int16],
        source_fingerprint: str,
    ) -> None:
        self.ngram = ngram
        self.min_coverage = min_coverage
        self.min_informative_words = min_informative_words
        self.hashes = hashes
        self.item_ids = item_ids
        self.item_benchmark = item_benchmark
        self.item_ngrams = item_ngrams
        self.benchmark_names = benchmark_names
        self.canaries = [c.lower() for c in canaries]
        self.short_lengths = short_lengths
        self.short_hashes = short_hashes
        self.short_benchmarks = short_benchmarks
        self.source_fingerprint = source_fingerprint

    @staticmethod
    def build(
        items: dict[str, list[str]],
        canaries: list[str],
        ngram: int,
        min_coverage: float,
        min_informative_words: int,
        short_min_words: int = 6,
        short_min_chars: int = 48,
        short_min_informative_words: int = 5,
        source_fingerprint: str = "",
    ) -> tuple[ContaminationIndex, dict[str, dict[str, int]]]:
        names = sorted(items)
        hash_parts: list[U64] = []
        id_parts: list[npt.NDArray[np.int32]] = []
        item_benchmark: list[int] = []
        item_ngrams: list[int] = []
        short_entries: list[tuple[int, int, int]] = []
        stats: dict[str, dict[str, int]] = {}
        for bench_index, name in enumerate(names):
            unmatchable = 0
            short_exact = 0
            long_matchable = 0
            total = 0
            for text in items[name]:
                words = signature_words(text)
                grams = informative_ngrams(words, ngram, min_informative_words)
                if grams.size == 0:
                    if (
                        short_min_words <= len(words) < ngram
                        and len(" ".join(words)) >= short_min_chars
                        and sum(len(word) >= 2 and not word.isdigit() for word in words)
                        >= short_min_informative_words
                    ):
                        length = len(words)
                        item_hash = int(ngram_hashes(word_hashes(words), length)[0])
                        short_entries.append((length, item_hash, bench_index))
                        short_exact += 1
                    else:
                        unmatchable += 1
                    continue
                item_id = len(item_ngrams)
                hash_parts.append(grams)
                id_parts.append(np.full(grams.size, item_id, dtype=np.int32))
                item_benchmark.append(bench_index)
                item_ngrams.append(int(grams.size))
                total += int(grams.size)
                long_matchable += 1
            stats[name] = {
                "items": len(items[name]),
                "unmatchable_items": unmatchable,
                "short_exact_items": short_exact,
                "long_ngram_items": long_matchable,
                "ngrams": total,
            }
        hashes = np.concatenate(hash_parts) if hash_parts else np.empty(0, dtype=np.uint64)
        ids = np.concatenate(id_parts) if id_parts else np.empty(0, dtype=np.int32)
        order = np.lexsort((ids, hashes))
        short = np.asarray(short_entries, dtype=[("length", "u1"), ("hash", "u8"), ("benchmark", "i2")])
        short.sort(order=("length", "hash", "benchmark"))
        index = ContaminationIndex(
            ngram,
            min_coverage,
            min_informative_words,
            hashes[order],
            ids[order],
            np.asarray(item_benchmark, dtype=np.int16),
            np.asarray(item_ngrams, dtype=np.int32),
            names,
            canaries,
            short["length"],
            short["hash"],
            short["benchmark"],
            source_fingerprint,
        )
        return index, stats

    def item_matches(self, doc_grams: U64) -> tuple[npt.NDArray[np.int32], npt.NDArray[np.int64]]:
        """Items sharing at least one n-gram with ``doc_grams`` (unique) and the shared counts."""
        left = np.searchsorted(self.hashes, doc_grams, side="left")
        right = np.searchsorted(self.hashes, doc_grams, side="right")
        lengths = right - left
        found = lengths > 0
        if not found.any():
            return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int64)
        starts, lengths = left[found], lengths[found]
        offsets = np.repeat(starts - np.cumsum(lengths) + lengths, lengths) + np.arange(int(lengths.sum()))
        items, counts = np.unique(self.item_ids[offsets], return_counts=True)
        return items, counts

    def match(self, signature_word_hashes: U64, raw_text: str) -> ContaminationHit:
        lowered = raw_text.lower()
        canaries = [c for c in self.canaries if c in lowered]
        benchmarks: list[str] = []
        short_exact_benchmarks: list[str] = []
        best = 0.0
        if signature_word_hashes.size >= self.ngram and self.hashes.size:
            grams = np.unique(ngram_hashes(signature_word_hashes, self.ngram))
            items, counts = self.item_matches(grams)
            if items.size:
                coverage = counts / self.item_ngrams[items]
                best = float(coverage.max())
                hit_items = items[coverage >= self.min_coverage]
                benchmarks = sorted({self.benchmark_names[b] for b in self.item_benchmark[hit_items]})
        if signature_word_hashes.size and self.short_lengths.size:
            short_matches: set[int] = set()
            for length in np.unique(self.short_lengths):
                length_value = int(length)
                if signature_word_hashes.size < length_value:
                    continue
                candidates = np.unique(ngram_hashes(signature_word_hashes, length_value))
                start = int(np.searchsorted(self.short_lengths, length_value, side="left"))
                end = int(np.searchsorted(self.short_lengths, length_value, side="right"))
                protected = self.short_hashes[start:end]
                hits = np.searchsorted(protected, candidates)
                valid = hits < protected.size
                if valid.any():
                    matched = candidates[valid][protected[hits[valid]] == candidates[valid]]
                    if matched.size:
                        short_matches.update(
                            self.short_benchmarks[start:end][np.isin(protected, matched)].tolist()
                        )
            short_exact_benchmarks = sorted(self.benchmark_names[i] for i in short_matches)
        return ContaminationHit(
            best,
            sorted(set(benchmarks) | set(short_exact_benchmarks)),
            canaries,
            short_exact_benchmarks,
        )

    def identity(self) -> str:
        meta = json.dumps(
            [
                VERSION,
                self.ngram,
                self.min_coverage,
                self.min_informative_words,
                self.benchmark_names,
                self.canaries,
                self.source_fingerprint,
            ],
            separators=(",", ":"),
        ).encode()
        arrays = (
            self.hashes,
            self.item_ids,
            self.item_benchmark,
            self.item_ngrams,
            self.short_lengths,
            self.short_hashes,
            self.short_benchmarks,
        )
        return sha256_bytes(meta + b"".join(np.ascontiguousarray(a).tobytes() for a in arrays))

    def save(self, directory: Path) -> str:
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "ngrams.npy", self.hashes)
        np.save(directory / "item_ids.npy", self.item_ids)
        np.save(directory / "item_benchmark.npy", self.item_benchmark)
        np.save(directory / "item_ngrams.npy", self.item_ngrams)
        np.save(directory / "short_lengths.npy", self.short_lengths)
        np.save(directory / "short_hashes.npy", self.short_hashes)
        np.save(directory / "short_benchmarks.npy", self.short_benchmarks)
        meta = {
            "version": VERSION,
            "ngram": self.ngram,
            "min_coverage": self.min_coverage,
            "min_informative_words": self.min_informative_words,
            "benchmarks": self.benchmark_names,
            "canaries": self.canaries,
            "source_fingerprint": self.source_fingerprint,
        }
        (directory / "index.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
        return self.identity()

    @staticmethod
    def load(directory: Path) -> ContaminationIndex:
        meta = json.loads((directory / "index.json").read_text())
        if meta["version"] != VERSION:
            raise ValueError(f"index {directory} has version {meta['version']}, expected {VERSION}")
        return ContaminationIndex(
            meta["ngram"],
            meta["min_coverage"],
            meta["min_informative_words"],
            np.load(directory / "ngrams.npy", mmap_mode="r"),
            np.load(directory / "item_ids.npy", mmap_mode="r"),
            np.load(directory / "item_benchmark.npy"),
            np.load(directory / "item_ngrams.npy"),
            meta["benchmarks"],
            meta["canaries"],
            np.load(directory / "short_lengths.npy", mmap_mode="r"),
            np.load(directory / "short_hashes.npy", mmap_mode="r"),
            np.load(directory / "short_benchmarks.npy", mmap_mode="r"),
            meta["source_fingerprint"],
        )


def explain(index: ContaminationIndex, text: str, limit: int = 5) -> list[tuple[str, float, str]]:
    """Return matched benchmark evidence for long-window and short exact hits."""
    words = signature_words(text)
    evidence: list[tuple[str, float, str]] = []
    grams = (
        np.unique(ngram_hashes(word_hashes(words), index.ngram))
        if len(words) >= index.ngram and index.hashes.size
        else np.empty(0, dtype=np.uint64)
    )
    items, counts = index.item_matches(grams)
    out = []
    for item, count in sorted(zip(items.tolist(), counts.tolist(), strict=True), key=lambda x: -x[1])[:limit]:
        item_hashes = np.asarray(index.hashes)[np.asarray(index.item_ids) == item]
        start = int(np.flatnonzero(np.isin(grams, item_hashes))[0])
        name = index.benchmark_names[int(index.item_benchmark[item])]
        out.append((name, count / int(index.item_ngrams[item]), " ".join(words[start : start + index.ngram])))
    if index.short_lengths.size:
        short_benchmarks: set[int] = set()
        word_hashes_ = word_hashes(words)
        for length in np.unique(index.short_lengths):
            n = int(length)
            if len(words) < n:
                continue
            short_grams = np.unique(ngram_hashes(word_hashes_, n))
            start = int(np.searchsorted(index.short_lengths, n, side="left"))
            end = int(np.searchsorted(index.short_lengths, n, side="right"))
            protected = index.short_hashes[start:end]
            positions = np.searchsorted(protected, short_grams)
            valid = positions < protected.size
            if valid.any():
                matched = short_grams[valid][protected[positions[valid]] == short_grams[valid]]
                short_benchmarks.update(
                    index.short_benchmarks[start:end][np.isin(protected, matched)].tolist()
                )
        evidence.extend(
            (index.benchmark_names[bench], 1.0, "normalized exact short item")
            for bench in sorted(short_benchmarks)
        )
    return (out + evidence)[:limit]
