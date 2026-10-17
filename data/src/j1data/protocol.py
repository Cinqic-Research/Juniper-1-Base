"""``juniper-chat-v1``: Juniper 1's native conversation and tool protocol.

Grammar (control tokens in angle brackets are single IDs; TEXT is ordinary encoded text):

    conversation  := message*
    message       := <|system|>    TEXT <|end|>
                   | <|developer|> TEXT <|end|>
                   | <|user|>      TEXT <|end|>
                   | <|tool|>      RESULT_JSON <|end|>            host-authored only
                   | <|assistant|> <|analysis|>  TEXT <|end|>
                   | <|assistant|> <|tool_call|> CALL_JSON <|call|>
                   | <|assistant|> <|final|>     TEXT <|eot|>

* Authority is system > developer > user > tool/data. The host assigns every role.
  The model writes only ``<|assistant|>`` messages, and a ``<|tool|>`` message can only
  come from the host.
* Generation starts after a host-supplied ``<|assistant|>`` and stops at ``<|call|>``
  (the host validates and executes the request) or ``<|eot|>`` (the turn is complete).
  ``<|end|>`` after analysis does not stop generation. ``<|call|>`` never completes a turn.
* CALL_JSON is ``{"name": str, "arguments": object}``. The host supplies call IDs, the
  protocol version, permissions and timestamps. RESULT_JSON is Juniper App's
  ``HostToolResult`` (``juniper-tool-protocol-v1``). This keeps model output separate
  from host results.
* There is no text escaping. TEXT is encoded with ``encode_special_tokens``, so a literal
  ``<|system|>`` in user or tool data stays plain bytes. A forged delimiter therefore
  cannot become a control ID.
* Analysis messages of completed turns are dropped when a conversation is re-serialized.
  Within the current turn they are kept across tool round trips.
* Loss mask (SFT): only model-produced tokens are trained. These are the tokens after each
  host-supplied ``<|assistant|>``, through the stop token, plus the ``<|assistant|>`` headers
  the model emits itself between messages of one turn. System, developer, user and tool
  messages are never trained. Analysis tokens are trained only when ``train_analysis``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from tokenizers import Tokenizer

from j1data.jtok import control_id

PROTOCOL = "juniper-chat-v1"
ROLES = ("system", "developer", "user", "tool", "assistant")
CHANNELS = ("analysis", "tool_call", "final")
TERMINATOR = {"analysis": "<|end|>", "tool_call": "<|call|>", "final": "<|eot|>"}
# IDs the model must never emit; a runtime should mask them while sampling.
FORBIDDEN_IN_GENERATION = ("<|system|>", "<|developer|>", "<|user|>", "<|tool|>", "<|endoftext|>")
HOST_STOP = ("<|call|>", "<|eot|>")


class ProtocolError(ValueError):
    pass


@dataclass
class Message:
    role: str
    content: str
    channel: str | None = None  # assistant only
    turn: int = 0  # index of the user turn this message belongs to

    def __post_init__(self) -> None:
        if self.role not in ROLES:
            raise ProtocolError(f"unknown role {self.role!r}")
        if (self.role == "assistant") != (self.channel is not None):
            raise ProtocolError("assistant messages need a channel; other roles must not have one")
        if self.channel is not None and self.channel not in CHANNELS:
            raise ProtocolError(f"unknown channel {self.channel!r}")
        if self.role == "assistant" and self.channel == "tool_call":
            validate_call(self.content)
        if self.role == "tool":
            validate_result(self.content)


def validate_call(text: str) -> dict[str, Any]:
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as e:
        raise ProtocolError(f"tool call is not JSON: {e}") from None
    if not isinstance(obj, dict) or set(obj) != {"name", "arguments"}:
        raise ProtocolError("tool call must be exactly {name, arguments}")
    if not isinstance(obj["name"], str) or not obj["name"] or not isinstance(obj["arguments"], dict):
        raise ProtocolError("tool call name must be a non-empty string and arguments an object")
    return obj


def validate_result(text: str) -> dict[str, Any]:
    obj = json.loads(text)
    if not isinstance(obj, dict) or obj.get("protocolVersion") != "juniper-tool-protocol-v1":
        raise ProtocolError("tool message must be a host-authored juniper-tool-protocol-v1 result")
    return obj


@dataclass
class Encoded:
    ids: list[int] = field(default_factory=list)
    labels: list[int] = field(default_factory=list)  # -100 where not trained

    def add(self, ids: list[int], trained: bool) -> None:
        self.ids.extend(ids)
        self.labels.extend(ids if trained else [-100] * len(ids))


def _text(tok: Tokenizer, text: str) -> list[int]:
    if not tok.encode_special_tokens:
        raise ProtocolError("tokenizer must have encode_special_tokens enabled")
    return tok.encode(text, add_special_tokens=False).ids


def serialize(
    tok: Tokenizer,
    messages: list[Message],
    train_turns: set[int] | None = None,
    train_analysis: bool = True,
    add_generation_prompt: bool = False,
) -> Encoded:
    """Token IDs and SFT labels. ``train_turns`` selects whose assistant messages are trained."""
    out = Encoded()
    last_turn = max((m.turn for m in messages), default=0)
    prev_assistant_same_turn = False
    for m in messages:
        if m.role == "assistant" and m.channel == "analysis" and m.turn < last_turn:
            continue  # completed turns keep no reasoning
        trained = m.role == "assistant" and (train_turns is None or m.turn in train_turns)
        if m.role == "assistant" and m.channel == "analysis":
            trained = trained and train_analysis
        header = [control_id(f"<|{m.role}|>")]
        # The first assistant header of a stretch is host-supplied; later ones are model-emitted.
        out.add(header, trained and prev_assistant_same_turn)
        body: list[int] = []
        if m.channel is not None:
            body.append(control_id(f"<|{m.channel}|>"))
        body += _text(tok, m.content)
        body.append(control_id(TERMINATOR[m.channel] if m.channel else "<|end|>"))
        out.add(body, trained)
        prev_assistant_same_turn = m.role == "assistant" and m.channel == "analysis"
    if add_generation_prompt:
        out.add([control_id("<|assistant|>")], False)
    return out


@dataclass
class ParsedTurn:
    messages: list[Message]
    stop: str  # "call", "eot", or "malformed"
    error: str | None = None


def parse_generation(tok: Tokenizer, ids: list[int], turn: int = 0) -> ParsedTurn:
    """Parses model output that followed a host ``<|assistant|>``. Never raises: malformed
    output becomes ``stop="malformed"`` with whatever complete messages preceded it."""
    forbidden = {control_id(t) for t in FORBIDDEN_IN_GENERATION}
    reserved = {control_id(f"<|reserved_{i:02d}|>") for i in range(12, 64)}
    channel_ids = {control_id(f"<|{c}|>"): c for c in CHANNELS}
    a_id = control_id("<|assistant|>")
    msgs: list[Message] = []
    i = 0
    expect_header = False
    while i < len(ids):
        if expect_header:
            if ids[i] != a_id:
                return ParsedTurn(msgs, "malformed", "expected <|assistant|> after analysis")
            i += 1
        if i >= len(ids) or ids[i] not in channel_ids:
            return ParsedTurn(msgs, "malformed", "missing channel token")
        channel = channel_ids[ids[i]]
        term = control_id(TERMINATOR[channel])
        j = i + 1
        while j < len(ids) and ids[j] != term:
            if ids[j] in forbidden or ids[j] in reserved or ids[j] in channel_ids or ids[j] == a_id:
                return ParsedTurn(msgs, "malformed", f"control token {ids[j]} inside {channel}")
            if ids[j] in {control_id(t) for t in ("<|end|>", "<|call|>", "<|eot|>")}:
                return ParsedTurn(msgs, "malformed", f"wrong terminator for {channel}")
            j += 1
        if j == len(ids):
            return ParsedTurn(msgs, "malformed", f"unterminated {channel}")
        text = tok.decode(ids[i + 1 : j], skip_special_tokens=False)
        try:
            msgs.append(Message("assistant", text, channel, turn))
        except ProtocolError as e:
            return ParsedTurn(msgs, "malformed", str(e))
        if channel == "tool_call":
            return (
                ParsedTurn(msgs, "call")
                if j + 1 == len(ids)
                else ParsedTurn(msgs, "malformed", "text after call")
            )
        if channel == "final":
            return (
                ParsedTurn(msgs, "eot")
                if j + 1 == len(ids)
                else ParsedTurn(msgs, "malformed", "text after eot")
            )
        i = j + 1
        expect_header = True
    return ParsedTurn(msgs, "malformed", "generation ended without <|call|> or <|eot|>")


def fit_to_budget(
    tok: Tokenizer, messages: list[Message], budget: int, reserve_generation: int, max_tool_tokens: int
) -> list[Message]:
    """Context budgeting. In order:

    1. Trim each tool result to ``max_tool_tokens``. The host marks this inside the
       result JSON, and the result stays a host message.
    2. Drop analysis from completed turns, which ``serialize`` does anyway.
    3. Drop whole oldest messages of completed turns, never system or developer and never
       the current turn, until the window fits.

    Raises if system, developer and the current turn alone exceed the window. Roles are
    never merged, so no content is promoted to a higher authority.
    """
    out: list[Message] = []
    for m in messages:
        if m.role == "tool" and len(_text(tok, m.content)) > max_tool_tokens:
            obj = validate_result(m.content)
            raw = json.dumps(obj.get("result"))
            keep = max(0, len(raw) * max_tool_tokens // max(1, len(_text(tok, raw))) - 64)
            obj["result"] = {"truncated": True, "original_chars": len(raw), "prefix": raw[:keep]}
            m = Message("tool", json.dumps(obj), turn=m.turn)
        out.append(m)
    window = budget - reserve_generation
    current = max((m.turn for m in out), default=0)

    def size(ms: list[Message]) -> int:
        return len(serialize(tok, ms, add_generation_prompt=True).ids)

    while size(out) > window:
        victim = next(
            (k for k, m in enumerate(out) if m.role not in ("system", "developer") and m.turn < current), None
        )
        if victim is None:
            raise ProtocolError("system, developer and current turn exceed the context window")
        turn = out[victim].turn
        # Remove a whole completed turn so no tool result is left without its request.
        out = [m for m in out if m.turn != turn or m.role in ("system", "developer")]
    return out
