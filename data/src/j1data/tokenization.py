"""Provisional measurement tokenizers for corpus sizing.

These are NOT Juniper 1 tokenizers. The Juniper 1 tokenizer is designed,
evaluated, selected and frozen in a separate stage after the corpus exists.
Here a byte-fallback BPE is trained at a few vocabulary sizes spanning the
range the design specification considers (about 24K-32K), only so that the
accepted corpus can be sized in tokens and the 3B+ objective checked
under every plausible vocabulary. Nothing here fixes vocabulary size, merges,
special or control tokens, a chat template or token IDs, and no training
shards are written with these tokenizers.

Layout: the 256 byte-fallback tokens ``<0x00>``-``<0xFF>`` followed by the
learned alphabet and merges; no special tokens. There is no normalizer and
spaces stay inside tokens, so decoding reproduces the input exactly.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np
from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers, trainers

from j1data.hashing import sha256_text

VERSION = "j1.measure-tokenizer.v1"
BYTE_TOKENS = [f"<0x{b:02X}>" for b in range(256)]
FIRST_LEARNED_ID = len(BYTE_TOKENS)
PRETOKENIZE_PATTERN = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)"
    r"|[^\r\n\p{L}\p{N}]?\p{L}+"
    r"|\p{N}"
    r"| ?[^\s\p{L}\p{N}]+[\r\n]*"
    r"|\s*[\r\n]+"
    r"|\s+(?!\S)"
    r"|\s+"
)


@dataclass(frozen=True)
class MeasurementSpec:
    name: str
    total_vocab: int
    limit_alphabet: int
    min_frequency: int


def _pre_tokenizer() -> pre_tokenizers.PreTokenizer:
    return pre_tokenizers.Split(Regex(PRETOKENIZE_PATTERN), behavior="isolated", invert=False)


def train(spec: MeasurementSpec, texts: Iterable[str]) -> Tokenizer:
    draft = Tokenizer(models.BPE())
    draft.pre_tokenizer = _pre_tokenizer()
    trainer = trainers.BpeTrainer(  # type: ignore[no-untyped-call]
        vocab_size=spec.total_vocab - FIRST_LEARNED_ID,
        min_frequency=spec.min_frequency,
        limit_alphabet=spec.limit_alphabet,
        show_progress=False,
        special_tokens=[],
    )
    draft.train_from_iterator(texts, trainer)
    learned = json.loads(draft.to_str())["model"]
    vocab = {token: index for index, token in enumerate(BYTE_TOKENS)}
    for token, _ in sorted(learned["vocab"].items(), key=lambda item: item[1]):
        if token in vocab:
            raise ValueError(f"learned token collides with a byte token: {token!r}")
        vocab[token] = len(vocab)
    return assemble(vocab, learned["merges"])


def assemble(vocab: dict[str, int], merges: list[Any]) -> Tokenizer:
    pairs = [tuple(m) if isinstance(m, list) else tuple(m.split(" ")) for m in merges]
    tokenizer = Tokenizer(
        models.BPE(vocab=vocab, merges=pairs, byte_fallback=True, fuse_unk=False, unk_token=None)
    )
    tokenizer.pre_tokenizer = _pre_tokenizer()
    tokenizer.decoder = decoders.Sequence([decoders.ByteFallback(), decoders.Fuse()])
    return tokenizer


def identity(tokenizer: Tokenizer) -> str:
    """Hash of the serialized tokenizer; identifies which measurement produced a count."""
    return sha256_text(tokenizer.to_str())


def encode_lengths(tokenizer: Tokenizer, texts: list[str]) -> np.ndarray:
    encodings = tokenizer.encode_batch_fast(texts, add_special_tokens=False)
    return np.fromiter((len(e.ids) for e in encodings), dtype=np.int64, count=len(texts))
