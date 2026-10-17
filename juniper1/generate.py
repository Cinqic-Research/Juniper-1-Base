"""Base-model text continuation with a KV cache: greedy by default, optional seeded sampling.

Juniper 1 Base is a next-token model. There is no chat template, no instruction tuning and no
system prompt: give it text and it continues the text. Control IDs 1-63 are masked out of the
output distribution (they were never trained); ``<|endoftext|>`` (ID 0) may be generated and ends
the continuation when ``stop_at_eos`` is set. Prompts longer than the 2,048-token context are
rejected rather than truncated silently.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .model import Transformer
from .tokenizer import EOS_ID, N_CONTROL, Juniper1Tokenizer


@dataclass(frozen=True)
class GenerationSettings:
    max_new_tokens: int = 64
    temperature: float = 0.0  # 0 = greedy argmax
    top_k: int = 0  # 0 = no truncation
    seed: int = 0
    stop_at_eos: bool = True


@torch.no_grad()
def continue_ids(model: Transformer, prompt_ids: list[int], settings: GenerationSettings) -> tuple[list[int], str]:
    """Return ``(generated_ids, stop_reason)`` for one prompt. Deterministic for fixed inputs and seed."""
    if not prompt_ids:
        raise ValueError("prompt must contain at least one token")
    ctx = model.cfg.max_seq_len
    if len(prompt_ids) >= ctx:
        raise ValueError(f"prompt has {len(prompt_ids)} tokens; the qualified context is {ctx} tokens")
    device = next(model.parameters()).device
    gen = torch.Generator(device="cpu").manual_seed(settings.seed)
    mask = torch.zeros(model.cfg.vocab_size, dtype=torch.bool, device=device)
    mask[1:N_CONTROL] = True
    idx = torch.tensor([prompt_ids], device=device)
    caches: list[list] = [[] for _ in model.blocks]
    logits = model.step_logits(idx, caches, 0)
    out: list[int] = []
    pos = len(prompt_ids)
    budget = min(settings.max_new_tokens, ctx - pos)
    reason = "length"
    for _ in range(budget):
        lg = logits.float().masked_fill(mask, float("-inf"))[0]
        if settings.temperature <= 0:
            nxt = int(lg.argmax())
        else:
            lg = lg / settings.temperature
            if settings.top_k > 0:
                kth = torch.topk(lg, min(settings.top_k, lg.numel())).values[-1]
                lg = lg.masked_fill(lg < kth, float("-inf"))
            probs = torch.softmax(lg, -1).cpu()
            nxt = int(torch.multinomial(probs, 1, generator=gen))
        out.append(nxt)
        if settings.stop_at_eos and nxt == EOS_ID:
            reason = "eos"
            break
        if pos + 1 >= ctx:
            reason = "context"
            break
        logits = model.step_logits(torch.tensor([[nxt]], device=device), caches, pos)
        pos += 1
    if len(out) >= settings.max_new_tokens and reason == "length":
        reason = "length"
    return out, reason


def complete(model: Transformer, tok: Juniper1Tokenizer, prompt: str, settings: GenerationSettings | None = None) -> dict:
    settings = settings or GenerationSettings()
    prompt_ids = tok.encode(prompt)
    ids, reason = continue_ids(model, prompt_ids, settings)
    return {
        "prompt_tokens": len(prompt_ids),
        "generated_ids": ids,
        "completion": tok.decode_text(ids),
        "stop": reason,
        "settings": settings.__dict__,
    }
