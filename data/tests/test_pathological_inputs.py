"""Every per-document function must stay linear on adversarial shapes.

A quadratic regex turns one malformed web page into hours of CPU; these inputs
reproduce the shapes that caused it (a long letter-free token, long blank runs,
a single enormous line, long base64-like runs).
"""

import time

import numpy as np
import pytest

from j1data import domains, privacy, quality
from j1data.build import load_pipeline_config
from j1data.ngrams import word_hashes
from j1data.normalize import normalize_stored, signature_words

CONFIG = quality.QualityConfig.from_dict(
    {k: v for k, v in load_pipeline_config()["quality"].items() if k != "version"}
)
SIZE = 200_000

SHAPES = {
    "letter_free_token": "12345.678" * (SIZE // 9),
    "blank_lines": "\n" * SIZE + "x",
    "spaces_and_newlines": " \n" * (SIZE // 2),
    "single_long_line": "word " * (SIZE // 5),
    "base64_like": "QUJD" * (SIZE // 4),
    "punctuation": "-*" * (SIZE // 2),
    "pipes": "|" * SIZE,
    "at_signs": "a@" * (SIZE // 2),
    "digits_dashes": "1-2 " * (SIZE // 4),
}


@pytest.mark.parametrize("shape", sorted(SHAPES))
def test_per_document_work_is_fast(shape: str) -> None:
    text = SHAPES[shape]
    started = time.perf_counter()
    stored = normalize_stored(text).text
    scrubbed = privacy.scrub(stored).text
    words = signature_words(scrubbed)
    hashes = word_hashes(words)
    lengths = np.fromiter((len(w) for w in words), dtype=np.int64, count=len(words))
    result = quality.evaluate(scrubbed, hashes, lengths, CONFIG, None, None)
    quality.evaluate(scrubbed, hashes, lengths, CONFIG, "code", "x.py")
    domains.formats_present(scrubbed)
    domains.domain_label(hashes, result.kind)
    assert time.perf_counter() - started < 3.0
