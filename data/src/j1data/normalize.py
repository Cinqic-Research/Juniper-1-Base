"""Text normalization.

Two representations are deliberately kept apart:

* the stored training text, which only receives repairs that cannot change
  meaning (line endings, a leading byte-order mark, C0 control characters and
  unpaired surrogates, outer blank lines);
* the matching signature, which is folded aggressively (NFKC, case folding,
  accent stripping, punctuation removal) and is used only for hashing,
  deduplication and decontamination. It is never written to training shards.

NFKC and case folding would corrupt mathematical and programming text (for
example ``x²`` becomes ``x2`` and ``ﬁ`` ligatures or full-width operators are
rewritten), so they stay on the signature side.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

import regex

VERSION = "j1.normalize.v1"

# C0 controls other than TAB and LF, plus DEL. CR is handled by line-ending
# normalization before this runs.
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")
_SURROGATE_RE = re.compile(r"[\ud800-\udfff]")
_WORD_RE = re.compile(r"\w+", re.UNICODE)
_MARKS_RE = regex.compile(r"\p{Mn}+")


@dataclass(frozen=True)
class StoredText:
    text: str
    removed_controls: int
    replaced_surrogates: int


def normalize_stored(raw: str) -> StoredText:
    text = raw.replace("\r\n", "\n").replace("\r", "\n")
    if text.startswith("﻿"):
        text = text[1:]
    text, removed_controls = _CONTROL_RE.subn("", text)
    text, replaced_surrogates = _SURROGATE_RE.subn("�", text)
    text = _strip_outer_blank_lines(text)
    return StoredText(text, removed_controls, replaced_surrogates)


def _strip_outer_blank_lines(text: str) -> str:
    # Leading indentation of the first non-blank line is significant for code,
    # so only whole blank lines are removed at the start.
    lines = text.split("\n")
    start = 0
    while start < len(lines) and not lines[start].strip():
        start += 1
    end = len(lines)
    while end > start and not lines[end - 1].strip():
        end -= 1
    body = lines[start:end]
    if body:
        body[-1] = body[-1].rstrip()
    return "\n".join(body)


def signature_text(text: str) -> str:
    folded = unicodedata.normalize("NFKC", text).casefold()
    if not folded.isascii():
        folded = _MARKS_RE.sub("", unicodedata.normalize("NFKD", folded))
    return " ".join(_WORD_RE.findall(folded))


def signature_words(text: str) -> list[str]:
    return signature_text(text).split()
