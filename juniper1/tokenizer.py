"""Juniper 1 tokenizer (``cand5-j1tok-24k``, layout ``j1.tokenizer.v1``).

ID layout (24,576 IDs, all counts inclusive):

* 0-63     control tokens. Named: ``<|endoftext|>`` (0, document separator / end of text, also the
           pad ID), then the ``juniper-chat-v1`` protocol tokens 1-11; 12-63 are ``<|reserved_NN|>``.
           **The base model was trained with ID 0 only.** IDs 1-63 never appear in the pretraining
           corpus; they are reserved for later stages and carry no trained meaning.
* 64-319   the 256 byte-fallback tokens ``<0x00>``-``<0xFF>``.
* 320-...  learned BPE vocabulary (24,256 tokens).

Text is never normalized and spaces stay inside tokens, so ``decode(encode(text)) == text`` for any
Unicode string. ``encode_special_tokens`` is enabled on load: a literal ``<|user|>`` in text is
encoded as ordinary bytes and never becomes a control ID. ``tokenizer.json`` does not persist that
flag, so always load through this module (or set the flag yourself); a bare
``Tokenizer.from_file`` maps literal control spellings to privileged IDs.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from tokenizers import Tokenizer

TOKENIZER_JSON_SHA256 = "176a89e2197c24d3f024fbb60ce6f346635e7cba912b3579d93b37d486c9adf1"
VOCAB_SIZE = 24576
N_CONTROL = 64
N_BYTE = 256
FIRST_BYTE_ID = N_CONTROL
FIRST_LEARNED_ID = N_CONTROL + N_BYTE

NAMED_CONTROL = [
    "<|endoftext|>",  # 0
    "<|system|>",  # 1
    "<|developer|>",  # 2
    "<|user|>",  # 3
    "<|assistant|>",  # 4
    "<|tool|>",  # 5
    "<|analysis|>",  # 6
    "<|final|>",  # 7
    "<|tool_call|>",  # 8
    "<|end|>",  # 9
    "<|call|>",  # 10
    "<|eot|>",  # 11
]
CONTROL_TOKENS = NAMED_CONTROL + [f"<|reserved_{i:02d}|>" for i in range(len(NAMED_CONTROL), N_CONTROL)]
BYTE_TOKENS = [f"<0x{i:02X}>" for i in range(N_BYTE)]
EOS_ID = 0
PAD_ID = 0

DEFAULT_PATH = Path(__file__).resolve().parent.parent / "tokenizer" / "tokenizer.json"


class TokenizerError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while b := f.read(1 << 20):
            h.update(b)
    return h.hexdigest()


class Juniper1Tokenizer:
    """The only supported way to load the Juniper 1 tokenizer."""

    def __init__(self, path: str | Path = DEFAULT_PATH, *, verify: bool = True):
        path = Path(path)
        if verify:
            got = sha256_file(path)
            if got != TOKENIZER_JSON_SHA256:
                raise TokenizerError(f"tokenizer.json SHA-256 {got} is not the frozen {TOKENIZER_JSON_SHA256}")
        tok = Tokenizer.from_file(str(path))
        tok.encode_special_tokens = True
        vocab = tok.get_vocab()
        if tok.get_vocab_size() != VOCAB_SIZE:
            raise TokenizerError(f"vocabulary size {tok.get_vocab_size()} != {VOCAB_SIZE}")
        if [vocab.get(t) for t in CONTROL_TOKENS] != list(range(N_CONTROL)):
            raise TokenizerError("control token IDs do not match j1.tokenizer.v1")
        if [vocab.get(t) for t in BYTE_TOKENS] != list(range(FIRST_BYTE_ID, FIRST_LEARNED_ID)):
            raise TokenizerError("byte-fallback token IDs do not match j1.tokenizer.v1")
        self._tok = tok
        self.path = path

    @property
    def vocab_size(self) -> int:
        return VOCAB_SIZE

    @property
    def eos_id(self) -> int:
        return EOS_ID

    @staticmethod
    def control_id(name: str) -> int:
        """ID of a named control token. Control IDs enter a sequence only through this API, never from text."""
        return CONTROL_TOKENS.index(name)

    def encode(self, text: str, *, add_eos: bool = False) -> list[int]:
        """Encode ordinary text. Never yields IDs 0-63; raises if it would (defensive check)."""
        ids = self._tok.encode(text, add_special_tokens=False).ids
        if any(i < N_CONTROL for i in ids):
            raise TokenizerError("text encoding produced a control ID; encode_special_tokens was not applied")
        if add_eos:
            ids.append(EOS_ID)
        return ids

    def decode(self, ids: list[int]) -> str:
        """Decode IDs to text. Control IDs decode to their spelling (``skip_special_tokens=False``)."""
        return self._tok.decode(list(ids), skip_special_tokens=False)

    def decode_text(self, ids: list[int]) -> str:
        """Decode, dropping control IDs (0-63)."""
        return self._tok.decode([i for i in ids if i >= N_CONTROL], skip_special_tokens=False)

    def token_strings(self, ids: list[int]) -> list[str]:
        return [self._tok.id_to_token(i) for i in ids]
