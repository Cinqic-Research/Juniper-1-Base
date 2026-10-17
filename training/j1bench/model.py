"""Benchmark implementation of the j1-core-28x640-v0.2 leading candidate.

EXPERIMENTAL / NON-CANONICAL. This module exists to measure training-workload
behaviour on the A100. It does not freeze the architecture: vocabulary size,
RoPE theta, QK-Norm form and initialisation are constructor arguments, and the
defaults below are the provisional values from the design specification.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 32768
    n_layer: int = 28
    d_model: int = 640
    n_head: int = 10
    n_kv_head: int = 5
    head_dim: int = 64
    d_ff: int = 1728
    max_seq_len: int = 2048
    rope_theta: float = 100000.0
    norm_eps: float = 1e-5
    # "rmsnorm_per_head": learnable RMSNorm over head_dim applied to q and k
    # before RoPE. One of several QK-Norm variants; unresolved in the spec.
    qk_norm: str = "rmsnorm_per_head"
    tie_embeddings: bool = True
    init_std: float = 0.02

    def to_dict(self) -> dict:
        return asdict(self)


def _atleast_fp32(t: torch.Tensor) -> torch.Tensor:
    return t if t.dtype in (torch.float32, torch.float64) else t.float()


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Normalise in at least fp32 (stability under bf16 autocast); fp64 stays fp64 for verification.
        xf = _atleast_fp32(x)
        y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return (y * _atleast_fp32(self.weight)).type_as(x)


# Experimental family for scaling pilots (head_dim 64, 2:1 GQA, FFN a multiple of 64,
# about 2.7x d_model). Final-size alternatives follow spec v0.5 section 2.
# The vocabulary is set from the tokenizer at run time.
FAMILY: dict[str, dict] = {
    "j1-micro-8x256": {"n_layer": 8, "d_model": 256, "n_head": 4, "n_kv_head": 2, "d_ff": 704},
    "j1-small-12x384": {"n_layer": 12, "d_model": 384, "n_head": 6, "n_kv_head": 3, "d_ff": 1024},
    "j1-mid-20x512": {"n_layer": 20, "d_model": 512, "n_head": 8, "n_kv_head": 4, "d_ff": 1408},
    "j1-core-28x640": {"n_layer": 28, "d_model": 640, "n_head": 10, "n_kv_head": 5, "d_ff": 1728},
    "j1-deep-30x576": {"n_layer": 30, "d_model": 576, "n_head": 9, "n_kv_head": 3, "d_ff": 2048},
    "j1-wide-20x768": {"n_layer": 20, "d_model": 768, "n_head": 12, "n_kv_head": 6, "d_ff": 1920},
}


def analytic_params(cfg: ModelConfig) -> int:
    """Closed-form parameter count (tied or untied embeddings, RMSNorm weights, per-head QK-Norm)."""
    d, hd = cfg.d_model, cfg.head_dim
    attn = d * cfg.n_head * hd * 2 + d * cfg.n_kv_head * hd * 2
    mlp = 3 * d * cfg.d_ff
    norms = 2 * d + (2 * hd if cfg.qk_norm == "rmsnorm_per_head" else 0)
    emb = cfg.vocab_size * d * (1 if cfg.tie_embeddings else 2)
    return cfg.n_layer * (attn + mlp + norms) + emb + d


def rope_cache(seq_len: int, head_dim: int, theta: float, device=None) -> tuple[torch.Tensor, torch.Tensor]:
    inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv)
    return freqs.cos(), freqs.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: (B, H, T, D); rotate-half convention over the two halves of D.
    d = x.shape[-1] // 2
    x1, x2 = _atleast_fp32(x[..., :d]), _atleast_fp32(x[..., d:])
    c, s = cos[None, None], sin[None, None]
    return torch.cat((x1 * c - x2 * s, x1 * s + x2 * c), dim=-1).type_as(x)


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.q = nn.Linear(cfg.d_model, cfg.n_head * cfg.head_dim, bias=False)
        self.k = nn.Linear(cfg.d_model, cfg.n_kv_head * cfg.head_dim, bias=False)
        self.v = nn.Linear(cfg.d_model, cfg.n_kv_head * cfg.head_dim, bias=False)
        self.o = nn.Linear(cfg.n_head * cfg.head_dim, cfg.d_model, bias=False)
        self.expand_kv = False
        self.q_norm: nn.Module
        self.k_norm: nn.Module
        if cfg.qk_norm == "rmsnorm_per_head":
            self.q_norm = RMSNorm(cfg.head_dim, cfg.norm_eps)
            self.k_norm = RMSNorm(cfg.head_dim, cfg.norm_eps)
        elif cfg.qk_norm == "none":
            self.q_norm = self.k_norm = nn.Identity()
        else:
            raise ValueError(f"unknown qk_norm {cfg.qk_norm!r}")

    def forward(self, x, cos, sin, cache: list | None = None):
        B, T, _ = x.shape
        c = self.cfg
        q = self.q(x).view(B, T, c.n_head, c.head_dim)
        k = self.k(x).view(B, T, c.n_kv_head, c.head_dim)
        v = self.v(x).view(B, T, c.n_kv_head, c.head_dim)
        q = apply_rope(self.q_norm(q).transpose(1, 2), cos, sin)
        k = apply_rope(self.k_norm(k).transpose(1, 2), cos, sin)
        v = v.transpose(1, 2)
        causal = True
        if cache is not None:
            if cache:  # decode: new positions attend to everything cached plus themselves
                k = torch.cat((cache[0], k), dim=2)
                v = torch.cat((cache[1], v), dim=2)
                causal = T > 1
                if causal:
                    raise NotImplementedError("multi-token extension of a non-empty cache")
            cache[:] = [k, v]
        if self.expand_kv:
            # Execution option for GPUs whose fused SDPA kernels lack GQA support (e.g. Turing):
            # repeating K/V heads is mathematically identical to native GQA.
            rep = c.n_head // c.n_kv_head
            k, v = k.repeat_interleave(rep, dim=1), v.repeat_interleave(rep, dim=1)
            y = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
        else:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=causal, enable_gqa=True)
        return self.o(y.transpose(1, 2).reshape(B, T, c.n_head * c.head_dim))


class SwiGLU(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.gate = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.up = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.down = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = Attention(cfg)
        self.mlp_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.mlp = SwiGLU(cfg)

    def forward(self, x, cos, sin, cache: list | None = None):
        x = x + self.attn(self.attn_norm(x), cos, sin, cache)
        return x + self.mlp(self.mlp_norm(x))


class Transformer(nn.Module):
    rope_cos: torch.Tensor
    rope_sin: torch.Tensor

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.final_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.lm_head = None if cfg.tie_embeddings else nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        cos, sin = rope_cache(cfg.max_seq_len, cfg.head_dim, cfg.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.reset_parameters()

    def reset_parameters(self):
        std = self.cfg.init_std
        # Residual output projections scaled by depth (GPT-2 style); provisional.
        out_std = std / math.sqrt(2 * self.cfg.n_layer)
        for name, p in self.named_parameters():
            if p.dim() == 1:
                continue
            nn.init.normal_(p, 0.0, out_std if name.endswith(("o.weight", "down.weight")) else std)

    def set_gqa_kernel(self, kernel: str) -> None:
        """``native``: SDPA with enable_gqa (A100). ``expand``: repeat K/V heads first (no new parameters)."""
        if kernel not in ("native", "expand"):
            raise ValueError(kernel)
        for blk in self.blocks:
            assert isinstance(blk, Block)
            blk.attn.expand_kv = kernel == "expand"

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def param_groups(self, weight_decay: float, policy: str = "matrices") -> list[dict]:
        """``matrices``: decay only 2-D projection weights; norms and the (tied) embedding
        are excluded. ``all``: decay everything (qualification-harness behaviour)."""
        if policy == "all":
            return [{"params": list(self.parameters()), "weight_decay": weight_decay}]
        if policy != "matrices":
            raise ValueError(f"unknown weight-decay policy {policy!r}")
        decay: list[nn.Parameter] = []
        no_decay: list[nn.Parameter] = []
        for name, p in self.named_parameters():
            (no_decay if p.dim() < 2 or name.startswith("embed.") else decay).append(p)
        return [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]

    @torch.no_grad()
    def step_logits(self, idx: torch.Tensor, caches: list[list], pos: int) -> torch.Tensor:
        """Inference: logits for the last position of ``idx`` at absolute offset ``pos``,
        extending per-layer KV caches (empty lists before the prefill)."""
        T = idx.shape[1]
        if pos + T > self.cfg.max_seq_len:
            raise ValueError("context exceeds max_seq_len")
        cos, sin = self.rope_cos[pos : pos + T], self.rope_sin[pos : pos + T]
        x = self.embed(idx)
        for blk, cache in zip(self.blocks, caches, strict=True):
            x = blk(x, cos, sin, cache)
        x = self.final_norm(x[:, -1:])
        w = self.embed.weight if self.lm_head is None else self.lm_head.weight
        return F.linear(x, w)[:, -1]

    def forward(self, idx: torch.Tensor, labels: torch.Tensor | None = None):
        T = idx.shape[1]
        if self.cfg.max_seq_len < T:
            raise ValueError(f"sequence length {T} exceeds max_seq_len {self.cfg.max_seq_len}")
        cos, sin = self.rope_cos[:T], self.rope_sin[:T]
        x = self.embed(idx)
        for blk in self.blocks:
            x = blk(x, cos, sin)
        x = self.final_norm(x)
        w = self.embed.weight if self.lm_head is None else self.lm_head.weight
        logits = F.linear(x, w)
        if labels is None:
            return logits
        # Sum over non-ignored positions; the caller divides by the global
        # useful-token count so gradient accumulation is exact.
        loss_sum = F.cross_entropy(
            _atleast_fp32(logits).reshape(-1, logits.size(-1)), labels.reshape(-1), ignore_index=-100, reduction="sum"
        )
        return logits, loss_sum
