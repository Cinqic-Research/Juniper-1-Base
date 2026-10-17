"""MinHash signatures and LSH band keys for near-duplicate detection.

Shingles are word 5-grams of the matching signature (see ``normalize``). Each
shingle is reduced to 32 bits ``x`` and hashed by multiply-add-shift,
``((a * x + b) mod 2**64) >> 32`` with random 64-bit ``a`` (odd) and ``b``, a
strongly universal family (Dietzfelbinger, 1996) that needs no modulo.

Full 32-bit signatures are stored per document, so the LSH banding (and with
it the similarity threshold) is chosen at deduplication time and can be
recalibrated without reprocessing text. With 14 bands of 8 rows the S-curve
crosses 0.5 at Jaccard (1/14) ** (1/8) ~= 0.72, the FineWeb / datatrove
operating point; the calibration is documented in docs/dedup.md.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from j1data.ngrams import U64, ngram_hashes

VERSION = "j1.minhash.v1"
_CHUNK = 4096


@dataclass(frozen=True)
class MinHashConfig:
    num_perm: int
    shingle_words: int
    seed: int


@dataclass(frozen=True)
class LSHConfig:
    bands: int
    rows: int

    def check(self, num_perm: int) -> None:
        if self.bands * self.rows > num_perm:
            raise ValueError("bands * rows exceeds the stored signature length")

    @property
    def threshold(self) -> float:
        """Jaccard similarity at which a pair is detected with probability ~0.5."""
        return float((1.0 / self.bands) ** (1.0 / self.rows))


class MinHasher:
    def __init__(self, config: MinHashConfig) -> None:
        self.config = config
        rng = np.random.default_rng(config.seed)
        self._a = rng.integers(
            0, np.iinfo(np.uint64).max, size=config.num_perm, dtype=np.uint64, endpoint=True
        )
        self._a |= np.uint64(1)
        self._b = rng.integers(
            0, np.iinfo(np.uint64).max, size=config.num_perm, dtype=np.uint64, endpoint=True
        )

    def shingles(self, word_hash_array: U64) -> U64:
        return ngram_hashes(word_hash_array, self.config.shingle_words)

    def signature(self, shingles: U64) -> npt.NDArray[np.uint32]:
        values = np.unique(shingles >> np.uint64(32))
        if values.size == 0:
            values = np.zeros(1, dtype=np.uint64)
        signature = np.full(self.config.num_perm, np.iinfo(np.uint64).max, dtype=np.uint64)
        with np.errstate(over="ignore"):
            for start in range(0, values.size, _CHUNK):
                block = values[start : start + _CHUNK, None]
                permuted = (block * self._a + self._b) >> np.uint64(32)
                np.minimum(signature, permuted.min(axis=0), out=signature)
        return signature.astype(np.uint32)


def band_keys(signatures: npt.NDArray[np.uint32], band: int, rows: int) -> U64:
    """LSH key of ``band`` for every column of a (num_perm, n) signature matrix."""
    block = signatures[band * rows : (band + 1) * rows].astype(np.uint64)
    acc = np.zeros(block.shape[1], dtype=np.uint64)
    with np.errstate(over="ignore"):
        for row in block:
            acc = acc * np.uint64(0x100_0000_01B3) + row
        acc ^= np.uint64(band) << np.uint64(56)
        acc = (acc ^ (acc >> np.uint64(33))) * np.uint64(0xFF51_AFD7_ED55_8CCD)
        acc = (acc ^ (acc >> np.uint64(33))) * np.uint64(0xC4CE_B9FE_1A85_EC53)
    result: U64 = acc ^ (acc >> np.uint64(33))
    return result


def jaccard(a: npt.NDArray[np.uint64], b: npt.NDArray[np.uint64]) -> float:
    left, right = np.unique(a), np.unique(b)
    union = np.union1d(left, right).size
    return float(np.intersect1d(left, right, assume_unique=True).size / union) if union else 1.0
