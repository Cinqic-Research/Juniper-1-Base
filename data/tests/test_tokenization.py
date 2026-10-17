"""Provisional measurement tokenizers: exact round trip, byte fallback, no special tokens."""

import json

from tokenizers import Tokenizer

from j1data import tokenization

TEXTS = [
    "def area(r: float) -> float:\n    return 3.14159 * r ** 2\n",
    "The quick brown fox jumps over the lazy dog. " * 20,
    "E = mc^2, \\frac{a}{b}, x_{i+1} ≤ ∞",
    '{"name": "juniper", "values": [1, 2, 3]}\nkey: value\n  - item\n',
    "$ ls -la /usr/local/bin && curl https://example.org/api/v2?q=1\n",
]


def tokenizer() -> Tokenizer:
    return tokenization.train(tokenization.MeasurementSpec("t", 600, 120, 1), TEXTS * 5)


def test_round_trip_is_exact_including_unseen_characters() -> None:
    tok = tokenizer()
    for text in [*TEXTS, "東京 \U0001f600 Ωμεγα \r\n tab\there"]:
        ids = tok.encode(text, add_special_tokens=False).ids
        assert tok.decode(ids) == text


def test_unseen_characters_fall_back_to_bytes() -> None:
    tok = tokenizer()
    ids = tok.encode("\U0001f600", add_special_tokens=False).ids
    assert len(ids) == 4 and all(i < tokenization.FIRST_LEARNED_ID for i in ids)


def test_measurement_tokenizer_has_no_special_or_control_tokens() -> None:
    tok = tokenizer()
    serialized = json.loads(tok.to_str())
    assert serialized["added_tokens"] == []
    assert not any(t.startswith("<|") for t in serialized["model"]["vocab"])
    ids = tok.encode("<|endoftext|>", add_special_tokens=False).ids
    assert len(ids) > 1


def test_identity_is_deterministic() -> None:
    assert tokenization.identity(tokenizer()) == tokenization.identity(tokenizer())
