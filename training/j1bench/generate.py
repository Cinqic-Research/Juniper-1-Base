"""Deterministic greedy generation over a sealed SFT evaluation bundle (token IDs in, token IDs out).

The harness reads only ``prompts.jsonl`` and the bundle manifest; hidden answers (``answers.jsonl``)
are read by the Juniper-1-Data scorer, never here. Text never enters this process: prompts arrive
as ``juniper-chat-v1`` token IDs built by the canonical serializer, and outputs leave as token IDs.

Decoding: greedy argmax with a KV cache, at most ``max_new_tokens`` per prompt, stopping at a stop
ID (``<|eot|>`` / ``<|call|>`` for the chat variant; ``<|endoftext|>`` for plain-text few-shot).
Forbidden IDs (host-only roles, the document separator and reserved IDs; for few-shot every
control ID) are masked to ``-inf`` before the argmax, as a runtime must do. Prompts of equal
length are decoded together in a fixed order (no padding, so batching does not change the
computation's inputs); the grouping is deterministic, so a rerun reproduces the outputs.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path

import torch

from . import ckpt as C
from .model import ModelConfig, Transformer
from .train import tensor_tree_sha

N_CONTROL = 64


def load_bundle(directory: Path, expected_bundle_sha256: str) -> tuple[dict, list[dict]]:
    d = Path(directory)
    m = json.loads((d / "manifest.json").read_text())
    body = {k: v for k, v in m.items() if k != "bundle_sha256"}
    from .sftdata import canonical_json

    if (
        hashlib.sha256(canonical_json(body)).hexdigest() != m["bundle_sha256"]
        or m["bundle_sha256"] != expected_bundle_sha256
    ):
        raise ValueError("evaluation bundle identity mismatch")
    if C.sha256_file(d / "prompts.jsonl") != m["files"]["prompts.jsonl"]["sha256"]:
        raise ValueError("prompts.jsonl bytes differ from the bundle manifest")
    prompts = [json.loads(line) for line in (d / "prompts.jsonl").read_bytes().split(b"\n") if line]
    if any("spec" in p or "reference" in p for p in prompts):
        raise ValueError("prompts carry hidden answers")
    return m, prompts


def load_checkpoint_model(path: Path) -> tuple[Transformer, dict]:
    """Any verified full-state checkpoint (frozen base or SFT stage) -> model + identity."""
    ok, why = C.verify_checkpoint(path)
    if not ok:
        raise ValueError(f"checkpoint {path} failed verification: {why}")
    m = C.read_manifest(path)
    assert m is not None
    raw_cfg = m.get("model_config") or m.get("run_contract", {}).get("model")
    cfg = ModelConfig(**raw_cfg) if raw_cfg else None
    if cfg is None:
        raise ValueError("checkpoint manifest has no model configuration")
    sd = torch.load(path / "model.pt", map_location="cpu", weights_only=True)
    if tensor_tree_sha(sd) != m["state_sha256"]["model"]:
        raise ValueError("model state digest differs from the manifest")
    model = Transformer(cfg)
    model.load_state_dict(sd, strict=True)
    ident = {
        "checkpoint": m["name"],
        "manifest_sha256": m["manifest_sha256"],
        "model_state_sha256": m["state_sha256"]["model"],
        "run_id": m["run_id"],
        "stage": m.get("stage"),
        "updates": m.get("updates", m.get("step")),
    }
    return model.eval(), ident


@torch.no_grad()
def generate(
    model: Transformer,
    prompts: list[dict],
    manifest: dict,
    device: str,
    *,
    batch: int = 32,
    progress=None,
) -> list[dict]:
    model.to(device).eval()
    if device.startswith("cuda"):
        model.set_gqa_kernel("expand")
    V = model.cfg.vocab_size
    masks = {}
    for variant in ("chat", "base_fewshot"):
        mask = torch.zeros(V, dtype=torch.bool)
        forbidden = manifest["forbidden_ids"] if variant == "chat" else list(range(1, N_CONTROL))
        mask[forbidden] = True
        masks[variant] = mask.to(device)
    stops = {"chat": set(manifest["stop_ids"]), "base_fewshot": {0}}
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for p in sorted(prompts, key=lambda p: (p["variant"], len(p["prompt_ids"]), p["id"])):
        groups[(p["variant"], len(p["prompt_ids"]))].append(p)
    out: list[dict] = []
    done = 0
    for (variant, plen), members in sorted(groups.items()):
        for s in range(0, len(members), batch):
            chunk = members[s : s + batch]
            if plen >= model.cfg.max_seq_len:  # cannot even prefill: recorded as an empty output
                for p in chunk:
                    out.append({"id": p["id"], "variant": variant, "generated_ids": [], "stop": "context_overflow",
                                "new_tokens": 0, "batch_seconds": 0.0})  # fmt: skip
                done += len(chunk)
                continue
            budget = max(p["max_new_tokens"] for p in chunk)
            budget = min(budget, model.cfg.max_seq_len - plen)
            t0 = time.perf_counter()
            idx = torch.tensor([p["prompt_ids"] for p in chunk], device=device)
            caches: list[list] = [[] for _ in model.blocks]
            logits = model.step_logits(idx, caches, 0)
            gen: list[list[int]] = [[] for _ in chunk]
            stopped: list[str | None] = [None] * len(chunk)
            pos = plen
            for _ in range(budget):
                logits = logits.float().masked_fill(masks[variant], float("-inf"))
                nxt = logits.argmax(-1)
                ids = nxt.tolist()
                for i, t in enumerate(ids):
                    if stopped[i] is None and len(gen[i]) < chunk[i]["max_new_tokens"]:
                        gen[i].append(t)
                        if t in stops[variant]:
                            stopped[i] = "stop"
                if all(
                    st is not None or len(g) >= p["max_new_tokens"]
                    for st, g, p in zip(stopped, gen, chunk, strict=True)
                ):
                    break
                if pos + 1 > model.cfg.max_seq_len:
                    break
                logits = model.step_logits(nxt[:, None], caches, pos)
                pos += 1
            dt = time.perf_counter() - t0
            for p, g, st in zip(chunk, gen, stopped, strict=True):
                out.append(
                    {
                        "id": p["id"],
                        "variant": variant,
                        "generated_ids": g,
                        "stop": st or ("length" if len(g) >= p["max_new_tokens"] else "context"),
                        "new_tokens": len(g),
                        "batch_seconds": round(dt, 4),
                    }
                )
            done += len(chunk)
            if progress:
                progress(done, len(prompts))
    return sorted(out, key=lambda r: r["id"])
