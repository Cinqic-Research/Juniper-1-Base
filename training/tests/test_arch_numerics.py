"""Architecture correctness beyond shapes: gradients, QK-Norm placement, optimizer semantics, GPU numerics."""

import math

import pytest
import torch

from j1bench.model import ModelConfig, Transformer

TINY = {
    "vocab_size": 97,
    "n_layer": 2,
    "d_model": 32,
    "n_head": 4,
    "n_kv_head": 2,
    "head_dim": 8,
    "d_ff": 64,
    "max_seq_len": 16,
}


def test_autograd_matches_finite_differences():
    torch.manual_seed(0)
    m = Transformer(ModelConfig(**TINY)).double()
    x = torch.randint(0, 97, (2, 12))
    y = torch.randint(0, 97, (2, 12))
    params = [m.blocks[0].attn.q.weight, m.blocks[1].mlp.down.weight, m.embed.weight, m.blocks[0].attn.v.weight]

    def loss_fn():
        # The training loss upcasts logits to float32; compute it in float64 here so differences resolve.
        return torch.nn.functional.cross_entropy(m(x).reshape(-1, 97), y.reshape(-1), reduction="sum")

    grads = torch.autograd.grad(loss_fn(), params)
    g = torch.Generator().manual_seed(1)
    for p, gr in zip(params, grads, strict=True):
        for _ in range(3):
            idx = tuple(torch.randint(0, n, (1,), generator=g).item() for n in p.shape)
            eps = 1e-6
            with torch.no_grad():
                p[idx] += eps
                lp = loss_fn().item()
                p[idx] -= 2 * eps
                lm = loss_fn().item()
                p[idx] += eps
            num = (lp - lm) / (2 * eps)
            assert math.isclose(gr[idx].item(), num, rel_tol=1e-5, abs_tol=1e-8)


def test_qk_norm_makes_attention_invariant_to_qk_projection_scale():
    """Per-head RMSNorm on q and k before RoPE: scaling the q/k projections must not change outputs."""
    torch.manual_seed(0)
    m = Transformer(ModelConfig(**{**TINY, "norm_eps": 1e-30})).double().eval()
    x = torch.randint(0, 97, (1, 16))
    ref = m(x)
    with torch.no_grad():
        for b in m.blocks:
            b.attn.q.weight.mul_(7.0)
            b.attn.k.weight.mul_(0.25)
    torch.testing.assert_close(m(x), ref, rtol=1e-9, atol=1e-9)
    m2 = Transformer(ModelConfig(**{**TINY, "qk_norm": "none", "norm_eps": 1e-30})).double().eval()
    m2.load_state_dict(
        {k: v for k, v in m.state_dict().items() if "q_norm" not in k and "k_norm" not in k}, strict=False
    )
    assert not torch.allclose(m2(x), ref)


def test_adamw_decays_matrices_only():
    torch.manual_seed(0)
    m = Transformer(ModelConfig(**TINY))
    opt = torch.optim.AdamW(m.param_groups(0.1), lr=0.01, foreach=True)
    before = {n: p.detach().clone() for n, p in m.named_parameters()}
    for p in m.parameters():
        p.grad = torch.zeros_like(p)  # zero gradient: only decoupled weight decay can move a parameter
    opt.step()
    for n, p in m.named_parameters():
        if p.dim() >= 2 and not n.startswith("embed."):
            torch.testing.assert_close(p, before[n] * (1 - 0.01 * 0.1))
        else:
            torch.testing.assert_close(p, before[n])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_gpu_fp16_matches_cpu_fp32_and_cache_decode():
    torch.manual_seed(0)
    cfg = ModelConfig(**{**TINY, "max_seq_len": 64})
    m = Transformer(cfg).eval()
    x = torch.randint(0, 97, (1, 40))
    ref = m(x)
    g = m.to("cuda", torch.float16)
    out = g(x.cuda()).float().cpu()
    assert (out - ref).abs().max().item() < 2e-2
    caches = [[] for _ in g.blocks]
    first = g.step_logits(x[:, :32].cuda(), caches, 0)
    steps = [g.step_logits(x[:, t : t + 1].cuda(), caches, t) for t in range(32, 40)]
    full = g(x.cuda())
    assert (first - full[:, 31]).abs().max().item() < 2e-2
    assert (torch.stack(steps, 1) - full[:, 32:40]).abs().max().item() < 2e-2


def test_expanded_kv_attention_equals_native_gqa():
    torch.manual_seed(0)
    m = Transformer(ModelConfig(**TINY)).double().eval()
    x = torch.randint(0, 97, (2, 16))
    ref = m(x)
    m.set_gqa_kernel("expand")
    torch.testing.assert_close(m(x), ref, rtol=1e-12, atol=1e-12)
    assert m.num_params() == sum(p.numel() for p in m.parameters())
