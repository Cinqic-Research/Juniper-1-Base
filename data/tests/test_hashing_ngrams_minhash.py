from pathlib import Path

import numpy as np
import pytest

from j1data.dedup import UnitRef, near_components
from j1data.hashing import doc_id, sha256_text, unit_interval
from j1data.minhash import LSHConfig, MinHashConfig, MinHasher, band_keys, jaccard
from j1data.ngrams import ngram_hashes, word_hashes
from j1data.normalize import signature_words

CONFIG = MinHashConfig(num_perm=112, shingle_words=5, seed=1)
LSH = LSHConfig(bands=14, rows=8)


def keys(signature: np.ndarray) -> np.ndarray:
    matrix = signature[:, None]
    return np.array([band_keys(matrix, band, LSH.rows)[0] for band in range(LSH.bands)], dtype=np.uint64)


def test_doc_id_is_stable_and_source_scoped() -> None:
    assert doc_id("wikimedia", "https://x/wiki/A") == doc_id("wikimedia", "https://x/wiki/A")
    assert doc_id("wikimedia", "https://x/wiki/A") != doc_id("stackexchange", "https://x/wiki/A")
    assert len(doc_id("s", "k")) == 32


def test_golden_hashes_do_not_drift() -> None:
    # Pinned values: a change here re-identifies every document or invalidates
    # every dedup/contamination index, so it must be a deliberate version bump.
    assert doc_id("fineweb-edu-dedup", "<urn:uuid:0>") == "0cbe07de5de83a75478553f10bac2bde"
    assert int(word_hashes(["juniper"])[0]) == 10840210393651579652
    assert int(ngram_hashes(word_hashes(["a", "b", "c", "d", "e"]), 5)[0]) == 16136010663202085169
    hasher = MinHasher(CONFIG)
    text = "the quick brown fox jumps over the lazy dog again and again"
    assert int(keys(_signature(hasher, text))[0]) == 15621475613939940089
    assert sha256_text("") == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


def test_unit_interval_is_deterministic_and_uniformish() -> None:
    values = np.array([unit_interval("ns", str(i)) for i in range(20000)])
    assert values.min() >= 0 and values.max() < 1
    assert abs(values.mean() - 0.5) < 0.01
    assert unit_interval("ns", "a") == unit_interval("ns", "a") != unit_interval("other", "a")


def test_ngram_hashes_short_text_and_order_sensitivity() -> None:
    words = word_hashes(["a", "b", "c"])
    assert ngram_hashes(words, 5).size == 1
    forward = ngram_hashes(word_hashes(["one", "two", "three", "four"]), 2)
    backward = ngram_hashes(word_hashes(["four", "three", "two", "one"]), 2)
    assert not set(forward.tolist()) & set(backward.tolist())


def _signature(hasher: MinHasher, text: str) -> np.ndarray:
    return hasher.signature(hasher.shingles(word_hashes(signature_words(text))))


def _text(seed: int, n: int = 400) -> str:
    rng = np.random.default_rng(seed)
    return " ".join(f"w{v}" for v in rng.integers(0, 5000, size=n))


def test_minhash_estimates_jaccard() -> None:
    hasher = MinHasher(CONFIG)
    base = _text(1).split()
    for keep in (0.95, 0.8, 0.5):
        edited = list(base)
        rng = np.random.default_rng(int(keep * 100))
        for i in rng.choice(len(edited), size=int(len(edited) * (1 - keep) / 5), replace=False):
            edited[i] = "zz" + str(i)
        a, b = " ".join(base), " ".join(edited)
        truth = jaccard(
            hasher.shingles(word_hashes(signature_words(a))), hasher.shingles(word_hashes(signature_words(b)))
        )
        estimate = float(np.mean(_signature(hasher, a) == _signature(hasher, b)))
        assert abs(estimate - truth) < 0.12, (keep, truth, estimate)


def test_identical_documents_share_every_band_and_unrelated_share_none() -> None:
    hasher = MinHasher(CONFIG)
    a, b = _text(7), _text(8)
    assert np.array_equal(keys(_signature(hasher, a)), keys(_signature(hasher, a + " ")))
    assert not np.intersect1d(keys(_signature(hasher, a)), keys(_signature(hasher, b))).size


def test_case_and_punctuation_variants_collide() -> None:
    hasher = MinHasher(CONFIG)
    text = _text(3)
    shouted = text.upper().replace(" ", ", ")
    assert np.array_equal(_signature(hasher, text), _signature(hasher, shouted))


def test_lsh_threshold_and_capacity_check() -> None:
    assert abs(LSH.threshold - 0.7192) < 1e-3
    LSH.check(112)
    with pytest.raises(ValueError):
        LSHConfig(15, 8).check(112)


def test_near_components_unions_transitive_bands_and_ignores_dead_rows(tmp_path: Path) -> None:
    signatures = np.random.default_rng(17).integers(0, 2**32, size=(8, 5), dtype=np.uint32)
    signatures[0:4, 1] = signatures[0:4, 0]
    signatures[4:8, 2] = signatures[4:8, 1]
    signatures[0:4, 4] = signatures[0:4, 3]
    signature_path = tmp_path / "signatures.npy"
    np.save(signature_path, signatures)
    refs = [UnitRef("source", "unit", 5, 0, tmp_path / "candidates.parquet", signature_path)]

    result = near_components(refs, LSHConfig(2, 4), np.array([1, 1, 1, 1, 0], dtype=bool))

    assert result.tolist() == [0, 0, 0, 3, -1]
