"""Host-side token-ID client for running Juniper 1 in llama.cpp's ``llama-server`` (finding A-02).

llama.cpp's own text tokenizer for the exported GGUF does not reproduce j1tok, and with special
parsing it turns literal ``<|system|>`` text into control IDs. This client never sends text to the
runtime. The host owns every step that touches text or authority:

* the host serializes role messages with ``protocol.serialize`` and the canonical tokenizer
  (``jtok.load``, ``encode_special_tokens`` on), so user or tool text can never become a control ID;
* the runtime receives a JSON array of integers only (``POST /completion``; llama-server uses an
  all-integer prompt verbatim) and returns sampled IDs (``return_tokens``, non-streaming: streaming
  drops IDs of incomplete UTF-8 byte tokens);
* host-only roles, the document separator and reserved IDs are banned by integer ``logit_bias``;
* llama.cpp ends generation at every end-of-generation ID (0, 9, 10, 11). ``<|end|>`` (9) after an
  analysis message is not a turn end in ``juniper-chat-v1``, so the client appends and resubmits;
* the host decodes and validates the turn with ``protocol.parse_generation``; nothing is returned
  to a caller unless the turn parses.

Only loopback ``http://`` URLs are accepted, redirects are refused (a bearer key never leaves the
loopback server), and the text endpoints (``/tokenize``, ``/detokenize``, ``/apply-template``,
``/v1/chat/completions``) are never called.
"""

from __future__ import annotations

import ipaddress
import json
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any

from tokenizers import Tokenizer

from j1data.jtok import N_CONTROL, control_id
from j1data.protocol import FORBIDDEN_IN_GENERATION, Message, ParsedTurn, parse_generation, serialize

VERSION = "j1.runtime-client.v1"
END, CALL, EOT = control_id("<|end|>"), control_id("<|call|>"), control_id("<|eot|>")
ANALYSIS = control_id("<|analysis|>")
BANNED = sorted({control_id(t) for t in FORBIDDEN_IN_GENERATION} | set(range(12, N_CONTROL)))
TEXT_ENDPOINTS = ("/tokenize", "/detokenize", "/apply-template", "/v1/chat/completions", "/v1/completions")


class RuntimeError_(RuntimeError):
    pass


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[no-untyped-def]  # noqa: ARG002
        raise RuntimeError_(f"runtime redirected to {newurl!r}; refused")


def loopback_base(url: str) -> str:
    u = urllib.parse.urlsplit(url)
    if u.scheme != "http" or u.username or u.password or u.query or u.fragment or u.path not in ("", "/"):
        raise RuntimeError_(f"runtime URL must be plain http://<loopback>:<port>, got {url!r}")
    host = u.hostname or ""
    try:
        if not ipaddress.ip_address(host).is_loopback:
            raise RuntimeError_(f"runtime host {host!r} is not loopback")
    except ValueError:
        raise RuntimeError_(f"runtime host {host!r} must be a literal loopback address") from None
    if u.port is None:
        raise RuntimeError_("runtime URL needs an explicit port")
    return f"http://{u.hostname if ':' not in host else '[' + host + ']'}:{u.port}"


@dataclass
class Sampling:
    temperature: float = 0.0  # 0 -> greedy
    top_k: int = 1
    top_p: float = 1.0
    seed: int = 0

    def body(self) -> dict[str, Any]:
        # Set every sampler explicitly; llama-server defaults (top-k 40, min-p, penalties, DRY, XTC) differ.
        return {
            "samplers": ["top_k", "top_p", "temperature"],
            "temperature": self.temperature,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "min_p": 0.0,
            "typical_p": 1.0,
            "repeat_penalty": 1.0,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
            "dry_multiplier": 0.0,
            "xtc_probability": 0.0,
            "seed": self.seed,
        }


@dataclass
class Completion:
    ids: list[int]
    stop_type: str
    raw: dict[str, Any] = field(repr=False)


class TokenIdClient:
    def __init__(
        self, base_url: str, tokenizer: Tokenizer, *, api_key: str | None = None, timeout: float = 120.0
    ):
        if not tokenizer.encode_special_tokens:
            raise RuntimeError_("tokenizer must be loaded with j1data.jtok.load (encode_special_tokens on)")
        self.base = loopback_base(base_url)
        self.tok = tokenizer
        self.vocab = tokenizer.get_vocab_size()
        self.api_key = api_key
        self.timeout = timeout
        # No proxies (http_proxy would otherwise receive the bearer key and prompt) and no redirects.
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect)
        self.paths_called: list[str] = []

    def _post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        if path in TEXT_ENDPOINTS:
            raise RuntimeError_(f"{path} is a text endpoint; Juniper 1 is token-ID only")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(self.base + path, json.dumps(body).encode(), headers, method="POST")
        self.paths_called.append(path)
        try:
            with self._opener.open(req, timeout=self.timeout) as r:
                return dict(json.load(r))
        except urllib.error.HTTPError as e:
            raise RuntimeError_(f"runtime HTTP {e.code} on {path}") from None

    def complete(
        self,
        prompt_ids: list[int],
        n_predict: int,
        sampling: Sampling,
        extra_bias: list[list[Any]] | None = None,
    ) -> Completion:
        if not prompt_ids or not all(type(t) is int and 0 <= t < self.vocab for t in prompt_ids):
            raise RuntimeError_("prompt must be a non-empty list of in-vocabulary integer token IDs")
        if n_predict < 1:
            raise RuntimeError_("n_predict must be positive")
        # Bans come last so a forcing bias can never re-enable a banned ID.
        bias: list[list[Any]] = list(extra_bias or []) + [[t, False] for t in BANNED]
        if not all(type(b[0]) is int for b in bias):
            raise RuntimeError_(
                "logit_bias keys must be integer token IDs (string keys are re-tokenized by the runtime)"
            )
        body = {
            "prompt": list(prompt_ids),
            "n_predict": n_predict,
            "stream": False,
            "return_tokens": True,
            "cache_prompt": True,
            "n_probs": 0,
            "logit_bias": bias,
            **sampling.body(),
        }
        res = self._post("/completion", body)
        ids = res.get("tokens")
        if not isinstance(ids, list) or not all(type(t) is int and 0 <= t < self.vocab for t in ids):
            raise RuntimeError_("runtime did not return generated token IDs")
        if isinstance(res.get("tokens_evaluated"), int) and res["tokens_evaluated"] > len(prompt_ids):
            raise RuntimeError_(
                "runtime evaluated more prompt tokens than were sent (implicit BOS/specials?)"
            )
        return Completion(ids, str(res.get("stop_type", "")), res)

    def turn(
        self, messages: list[Message], *, budget: int, sampling: Sampling, max_segments: int = 8
    ) -> ParsedTurn:
        """One assistant turn (resubmitted after ``<|end|>`` closes an analysis message), parsed."""
        prompt = serialize(self.tok, messages, add_generation_prompt=True).ids
        generated: list[int] = []
        for _ in range(max_segments):
            remaining = budget - len(generated)
            if remaining < 1:
                break
            c = self.complete(prompt + generated, remaining, sampling)
            generated += c.ids
            if not c.ids or c.ids[-1] in (CALL, EOT) or c.stop_type != "eos":
                break
            if c.ids[-1] == END and _open_channel(generated) == "analysis":
                generated.append(control_id("<|assistant|>"))  # model-emitted header of the next message
                continue
            break
        return parse_generation(self.tok, generated)

    def self_test(self) -> dict[str, str]:
        """Forces each relevant ID: 9/10/11 must end generation (EOG set), 1 must stay banned."""
        prompt = [
            control_id("<|user|>"),
            *self.tok.encode("hi", add_special_tokens=False).ids,
            END,
            control_id("<|assistant|>"),
        ]
        out = {}
        for tid in (END, CALL, EOT, control_id("<|system|>")):
            c = self.complete(prompt, 2, Sampling(), extra_bias=[[tid, 100.0]])
            out[str(tid)] = f"{c.stop_type}:{c.ids}"
            if tid == control_id("<|system|>") and tid in c.ids:
                raise RuntimeError_("banned control ID was sampled")
            if tid != control_id("<|system|>") and (c.stop_type != "eos" or c.ids[:1] != [tid]):
                raise RuntimeError_(
                    f"ID {tid} did not end generation as an end-of-generation token: {out[str(tid)]}"
                )
        return out


def _open_channel(ids: list[int]) -> str | None:
    channels = {
        control_id("<|analysis|>"): "analysis",
        control_id("<|final|>"): "final",
        control_id("<|tool_call|>"): "tool_call",
    }
    for t in reversed(ids):
        if t in channels:
            return channels[t]
    return None
