"""Fail-closed admission and tokenization for text-only instruction SFT fixtures.

This is preparation tooling, not a training entry point. Tool traces, attachments,
retrieval and analysis need separate rendering and provenance contracts before they
can be admitted. Source-use approval is supplied by the caller, never inferred from
the presence of an episode or from a repository license.
"""

from __future__ import annotations

from collections.abc import Collection
from typing import Any

from tokenizers import Tokenizer

from j1data import episodes
from j1data.protocol import Encoded, Message, serialize

# Oracle-backed labels (episodes.validate binds each to a matching passing check) and NOT_APPLICABLE
# for purely stylistic/text tasks. SCHEMA_VALID_ONLY, MODEL_JUDGED_ONLY and NOT_VERIFIED never qualify.
ADMISSIBLE_FACTUALITY = frozenset(
    {"VERIFIED_EXECUTION", "VERIFIED_KNOWN_ANSWER", "VERIFIED_SOURCE", "VERIFIED_HUMAN", "NOT_APPLICABLE"}
)


# A same-model self-check never counts as the qualifying evidence on its own.
INDEPENDENT_ORACLES = frozenset(
    {
        "execution",
        "known_answer",
        "source_evidence",
        "schema",
        "deterministic_rule",
        "independent_model",
        "human_review",
    }
)


def _open_ended_human(episode: dict[str, Any]) -> bool:
    """The one admission path for NOT_VERIFIED: human-written open-ended conversation (e.g. the OASST1
    subset, rights R07) has no per-item factual oracle. It needs human origin, the source's own human
    review (OASST volunteer ranking/moderation) and passed source filters; claim-heavy conversations are
    rejected before this point and the dataset's independent rubric review gates the build."""
    checks = episode["validation"]["checks"]
    return (
        episode["validation"]["factuality"] == "NOT_VERIFIED"
        and episode["provenance"].get("origin") == "human"
        and any(c["name"] == "human_source_filters" and c["passed"] for c in checks)
        and any(c["oracle"] == "human_review" and c["passed"] for c in checks)
    )


class SFTAdmissionError(ValueError):
    """The episode is unsuitable for this explicitly limited SFT serializer."""


def encode_text_episode(
    tokenizer: Tokenizer,
    episode: dict[str, Any],
    *,
    approved_source_licenses: Collection[str],
    expected_decontamination_index: str,
) -> Encoded:
    """Encode one accepted text-only episode with loss on selected assistant finals.

    The caller must pass an explicit, nonempty set of approved source-license IDs.
    The output uses the native ``juniper-chat-v1`` token grammar and causal labels.
    A later trainer must still verify tokenizer identity, split isolation, full
    dataset admission and the approved training contract.
    """
    problems = episodes.validate(episode)
    if problems:
        raise SFTAdmissionError("invalid episode: " + "; ".join(problems))
    if episode["stage"] != "agent-instruction-sft":
        raise SFTAdmissionError("only agent-instruction-sft episodes are supported")
    if not episode["validation"]["accepted"]:
        raise SFTAdmissionError("episode was not accepted")
    checks = episode["validation"]["checks"]
    if episode["validation"]["factuality"] not in ADMISSIBLE_FACTUALITY and not _open_ended_human(episode):
        raise SFTAdmissionError("episode has no qualifying factuality evidence")
    if not checks or not all(c["passed"] for c in checks):
        raise SFTAdmissionError("episode needs at least one check and every check passing")
    if not any(c["oracle"] in INDEPENDENT_ORACLES for c in checks):
        raise SFTAdmissionError(
            "episode needs a passing check from an oracle independent of the generating model"
        )
    if (
        not expected_decontamination_index
        or episode["decontamination"]["index_identity"] != expected_decontamination_index
    ):
        raise SFTAdmissionError("decontamination index does not match the approved run")
    if isinstance(approved_source_licenses, (str, bytes)) or not all(
        isinstance(x, str) for x in approved_source_licenses
    ):
        raise SFTAdmissionError("approved_source_licenses must be a collection of license IDs, not a string")
    source_license = episode["provenance"].get("source_license")
    if not source_license or source_license not in approved_source_licenses:
        raise SFTAdmissionError("source license lacks explicit approval for this run")
    if episode.get("tools"):
        raise SFTAdmissionError("tool catalog requires a separately qualified serializer")

    messages: list[Message] = []
    train_turns: set[int] = set()
    turn = 0
    waiting_for_final = False
    for index, item in enumerate(episode["messages"]):
        role = item["role"]
        parts = item["content"]
        if len(parts) != 1 or parts[0]["type"] != "text":
            raise SFTAdmissionError(f"message {index}: only one plain text part is supported")
        content = parts[0]["text"]
        if role in ("system", "developer"):
            if turn:
                raise SFTAdmissionError(f"message {index}: host configuration after a user turn")
        elif role == "user":
            if waiting_for_final:
                raise SFTAdmissionError(f"message {index}: previous user turn lacks a final")
            turn += 1
            waiting_for_final = True
        elif role == "assistant":
            if item.get("channel") != "final":
                raise SFTAdmissionError(f"message {index}: only assistant final is supported")
            if not waiting_for_final:
                raise SFTAdmissionError(f"message {index}: final has no preceding user")
            waiting_for_final = False
            if item["train"]:
                train_turns.add(turn)
        else:
            raise SFTAdmissionError(f"message {index}: {role} requires a separately qualified serializer")
        messages.append(Message(role, content, item.get("channel"), turn))

    if waiting_for_final or not train_turns:
        raise SFTAdmissionError("episode lacks a complete trainable user/final pair")
    encoded = serialize(tokenizer, messages, train_turns=train_turns, train_analysis=False)
    if len(encoded.ids) != len(encoded.labels) or not any(label != -100 for label in encoded.labels):
        raise SFTAdmissionError("invalid or empty training labels")
    return encoded


IGNORE = -100


def causal_rows(
    examples: list[Encoded], seq_len: int, pad_id: int, *, vocab_size: int, pack: bool = False
) -> tuple[list[list[int]], list[list[int]]]:
    """Turn encoded episodes into next-token ``(inputs, targets)`` rows of exactly ``seq_len``.

    ``Encoded.labels`` are aligned with ``Encoded.ids`` (the label at ``t`` is the token at
    ``t``). The Juniper-1-Train model's ``forward(idx, labels)`` scores ``logits[t]`` against
    ``labels[t]`` with no internal shift, so its targets must already be the next token. This
    function performs that shift inside each episode: inputs are ``ids[:-1]`` and targets are
    ``labels[1:]``. A target therefore never comes from a different episode, and an episode's
    final stop token is a target but never an input.

    By default each episode gets its own row. ``pack=True`` places whole episodes in order and
    starts a new row when the next one does not fit; packed episodes then share one causal
    attention window (the model has no document mask), so a later episode can attend to an
    earlier episode's system/developer text. Use it only under a stage contract that accepts
    that. An episode longer than ``seq_len + 1`` tokens is rejected rather than truncated, so a
    clipped supervised unit never reaches a batch. Padding uses ``pad_id`` with ``-100`` targets.
    """
    if seq_len < 1:
        raise SFTAdmissionError("seq_len must be positive")
    if not 0 <= pad_id < vocab_size:
        raise SFTAdmissionError("pad_id outside the vocabulary")
    rows_x: list[list[int]] = []
    rows_y: list[list[int]] = []
    cur_x: list[int] = []
    cur_y: list[int] = []

    def flush() -> None:
        rows_x.append(cur_x + [pad_id] * (seq_len - len(cur_x)))
        rows_y.append(cur_y + [IGNORE] * (seq_len - len(cur_y)))

    for n, e in enumerate(examples):
        if len(e.ids) != len(e.labels) or len(e.ids) < 2:
            raise SFTAdmissionError(
                f"example {n}: ids and labels must be equal-length with at least two tokens"
            )
        if any(not 0 <= tid < vocab_size for tid in e.ids):
            raise SFTAdmissionError(f"example {n}: token id outside the vocabulary")
        if any(label not in (IGNORE, tid) for label, tid in zip(e.labels, e.ids, strict=True)):
            raise SFTAdmissionError(f"example {n}: a label differs from its aligned token")
        if e.labels[0] != IGNORE:
            raise SFTAdmissionError(f"example {n}: the first token has no context and cannot be a target")
        x, y = e.ids[:-1], e.labels[1:]
        if all(label == IGNORE for label in y):
            raise SFTAdmissionError(f"example {n}: no trainable target after the shift")
        if len(x) > seq_len:
            raise SFTAdmissionError(f"example {n}: {len(e.ids)} tokens exceed seq_len + 1 = {seq_len + 1}")
        if cur_x and (not pack or len(cur_x) + len(x) > seq_len):
            flush()
            cur_x, cur_y = [], []
        cur_x += x
        cur_y += y
    if cur_x:
        flush()
    return rows_x, rows_y
