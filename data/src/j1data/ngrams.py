"""Deterministic 64-bit word and word-n-gram hashing.

Word hashes come from xxh64 with a fixed seed, so they are identical across
processes, machines and Python versions (unlike the built-in ``hash``).
N-gram hashes are a polynomial combination of word hashes followed by a
SplitMix64 finalizer, computed with wrapping uint64 arithmetic in NumPy.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import xxhash

U64 = npt.NDArray[np.uint64]

WORD_SEED = 0x4A31_5744  # "J1WD"
_MULTIPLIER = np.uint64(0x9E37_79B9_7F4A_7C15)


_CACHE: dict[str, int] = {}
_CACHE_LIMIT = 300_000


def _hash_and_cache(word: str) -> int:
    value = xxhash.xxh64_intdigest(word.encode(), WORD_SEED)
    if len(_CACHE) < _CACHE_LIMIT:
        _CACHE[word] = value
    return value


def word_hashes(words: list[str]) -> U64:
    # Word frequencies are Zipfian, so a bounded per-process cache answers most lookups.
    get = _CACHE.get
    return np.fromiter((get(w) or _hash_and_cache(w) for w in words), dtype=np.uint64, count=len(words))


def _splitmix64(x: U64) -> U64:
    with np.errstate(over="ignore"):
        x = x + np.uint64(0x9E37_79B9_7F4A_7C15)
        x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58_476D_1CE4_E5B9)
        x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D0_49BB_1331_11EB)
        return x ^ (x >> np.uint64(31))


def ngram_hashes(words: U64, n: int) -> U64:
    """Hashes of all contiguous n-grams; a text shorter than n yields one hash of all its words."""
    count = len(words)
    if count == 0:
        return np.empty(0, dtype=np.uint64)
    if count < n:
        n = count
    span = count - n + 1
    acc = np.zeros(span, dtype=np.uint64)
    with np.errstate(over="ignore"):
        for offset in range(n):
            acc = acc * _MULTIPLIER + words[offset : offset + span]
    return _splitmix64(acc ^ np.uint64(n))
