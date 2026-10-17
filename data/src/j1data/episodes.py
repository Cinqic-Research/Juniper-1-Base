"""Validation of post-training episodes (schemas/episode.schema.json).

Beyond JSON Schema, an episode must respect the rules the Juniper 1 design
places on agent data:

* Order and authority: system, then developer, then the conversation. System
  and developer content is host configuration and must be trusted; attachments,
  retrieved text, quoted text and tool output are data and must be marked
  untrusted, whatever they claim to be.
* A tool call is a request. Every tool message answers an earlier assistant
  call by id and name, and only results with host-execution provenance may
  appear in an accepted episode.
* Loss masks train assistant turns only, so the model never learns to
  impersonate system, user or host tool messages.
* An assistant final answer may not claim that something ran unless a host
  tool result exists in the episode.
* Factuality labels must be backed by a passing check from a matching oracle;
  a same-model judgement never yields a VERIFIED label.
* Reasoning (analysis channel) stays concise.
"""

from __future__ import annotations

import json
import re
from functools import cache
from typing import Any

import jsonschema

from j1data.registry import REPO_ROOT

VERSION = "j1.episodes.v1"
MAX_ANALYSIS_WORDS = 400

EXECUTION_CLAIM_RE = re.compile(
    r"\b(?:i (?:ran|executed|tested)|when i ran|checked by running"
    r"|running (?:it|this|the code) (?:gives|gave|returns|returned|prints)"
    r"|the (?:script|code|program) (?:printed|returned|output)|execution (?:shows|showed|returned))\b",
    re.IGNORECASE,
)
_FACTUALITY_ORACLES = {
    "VERIFIED_EXECUTION": {"execution"},
    "VERIFIED_KNOWN_ANSWER": {"known_answer"},
    "VERIFIED_SOURCE": {"source_evidence"},
    "VERIFIED_HUMAN": {"human_review"},
    "SCHEMA_VALID_ONLY": {"schema"},
}


@cache
def _schema() -> dict[str, Any]:
    return dict(json.loads((REPO_ROOT / "schemas" / "episode.schema.json").read_text()))


@cache
def _validator() -> Any:
    """One checked validator for the schema; ``jsonschema.validate`` would re-check the schema per call."""
    cls = jsonschema.validators.validator_for(_schema())
    cls.check_schema(_schema())
    return cls(_schema())


def validate(episode: dict[str, Any]) -> list[str]:
    """Return every rule violation (empty when the episode is valid)."""
    error = jsonschema.exceptions.best_match(_validator().iter_errors(episode))
    if error is not None:
        return [f"schema: {error.message}"]
    problems: list[str] = []
    messages = episode["messages"]
    roles = [m["role"] for m in messages]

    seen_conversation = False
    for index, message in enumerate(messages):
        role = message["role"]
        if role in ("system", "developer"):
            if seen_conversation:
                problems.append(f"message {index}: {role} message after the conversation started")
            if role == "system" and "developer" in roles[:index]:
                problems.append(f"message {index}: system message after developer message")
            if any(part["trust"] != "trusted" for part in message["content"]):
                problems.append(f"message {index}: {role} content must be trusted host configuration")
        else:
            seen_conversation = True
        for part in message["content"]:
            is_data = role == "tool" or part["type"] != "text"
            if is_data and part["trust"] != "untrusted":
                problems.append(f"message {index}: {part['type']} data in a {role} message must be untrusted")
        if message["train"] and role != "assistant":
            problems.append(f"message {index}: only assistant messages may carry training loss")
        if role == "assistant" and "channel" not in message:
            problems.append(f"message {index}: assistant message without a channel")
        if role != "assistant" and "channel" in message:
            problems.append(f"message {index}: only assistant messages have channels")

    calls: dict[str, str] = {}
    answered: set[str] = set()
    for index, message in enumerate(messages):
        if message["role"] == "assistant" and message.get("channel") == "tool_call":
            call = message.get("tool_call")
            if call is None:
                problems.append(f"message {index}: tool_call channel without a tool_call")
                continue
            if call["call_id"] in calls:
                problems.append(f"message {index}: duplicate call id {call['call_id']}")
            calls[call["call_id"]] = call["name"]
            offered = {t["name"] for t in episode.get("tools", [])}
            if call["name"] not in offered:
                problems.append(f"message {index}: call to a tool that was not offered: {call['name']}")
        elif "tool_call" in message:
            problems.append(f"message {index}: tool_call outside an assistant tool_call message")
        if message["role"] == "tool":
            result = message.get("tool_result")
            if result is None:
                problems.append(f"message {index}: tool message without a tool_result")
                continue
            if calls.get(result["call_id"]) != result["name"]:
                problems.append(f"message {index}: tool result does not answer an earlier call")
            if result["call_id"] in answered:
                problems.append(f"message {index}: call answered twice")
            answered.add(result["call_id"])
            if not result["provenance"]["host_executed"]:
                problems.append(f"message {index}: tool result without host-execution provenance")
        elif "tool_result" in message:
            problems.append(f"message {index}: tool_result outside a tool message")
    if set(calls) - answered and episode["validation"]["accepted"]:
        last = messages[-1]
        if not (last["role"] == "assistant" and last.get("channel") == "tool_call"):
            problems.append("unanswered tool call followed by further turns")

    final = [m for m in messages if m["role"] == "assistant" and m.get("channel") == "final"]
    if not answered:
        for message in final:
            text = " ".join(p["text"] for p in message["content"])
            if EXECUTION_CLAIM_RE.search(text):
                problems.append("final answer claims execution but no host tool result exists")
    for index, message in enumerate(messages):
        if message.get("channel") == "analysis":
            words = sum(len(p["text"].split()) for p in message["content"])
            if words > MAX_ANALYSIS_WORDS:
                problems.append(f"message {index}: reasoning of {words} words exceeds {MAX_ANALYSIS_WORDS}")

    validation = episode["validation"]
    required = _FACTUALITY_ORACLES.get(validation["factuality"])
    if required is not None:
        passing = {c["oracle"] for c in validation["checks"] if c["passed"]}
        if not passing & required:
            problems.append(
                f"factuality {validation['factuality']} without a passing {'/'.join(sorted(required))} check"
            )
    if validation["accepted"] and any(not c["passed"] for c in validation["checks"]):
        problems.append("accepted episode has a failing check")
    if validation["accepted"] and not episode["decontamination"]["passed"]:
        problems.append("accepted episode failed decontamination")
    return problems
