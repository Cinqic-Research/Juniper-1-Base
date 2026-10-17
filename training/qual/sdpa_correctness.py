"""Fused attention kernels versus an FP64 reference, forward and backward, for the
j1-28x640 attention geometry (10 query heads, 5 KV heads, head_dim 64, causal).

Also checks that each kernel is causal in the backward pass: perturbing tokens
after position p must not change outputs or gradients at positions <= p.
Reports kernels that are unavailable instead of silently falling back.
"""

import json
import os

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

if os.environ.get("J1_DET") == "1":
    # Separates kernel nondeterminism (atomic accumulation order) from causal leakage.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
torch.manual_seed(0)
dev = "cuda"
B, T, D = 2, 2048, 64
# Geometry: the core candidate (10 query / 5 KV heads) by default; J1_SDPA_GEOMETRY="12,6" checks the selected wide
# shape (review N-09: the launch geometry needs its own fp64-reference evidence on the launch host).
HQ, HKV = (int(x) for x in os.environ.get("J1_SDPA_GEOMETRY", "10,5").split(","))
out = {"geometry": {"query_heads": HQ, "kv_heads": HKV, "head_dim": D, "seq_len": T, "batch": B}}


def ref_attn(q, k, v):
    k = k.repeat_interleave(HQ // HKV, dim=1)
    v = v.repeat_interleave(HQ // HKV, dim=1)
    s = (q @ k.transpose(-1, -2)) / D**0.5
    s = s.masked_fill(torch.triu(torch.ones(T, T, dtype=torch.bool, device=q.device), 1), float("-inf"))
    return s.softmax(-1) @ v


base = [torch.randn(B, h, T, D, device=dev, dtype=torch.float64) for h in (HQ, HKV, HKV)]
go = torch.randn(B, HQ, T, D, device=dev, dtype=torch.float64)
qr, kr, vr = [t.clone().requires_grad_() for t in base]
o_ref = ref_attn(qr, kr, vr)
o_ref.backward(go)
g_ref = [qr.grad, kr.grad, vr.grad]

for name, be in (
    ("flash", SDPBackend.FLASH_ATTENTION),
    ("cudnn", SDPBackend.CUDNN_ATTENTION),
    ("efficient", SDPBackend.EFFICIENT_ATTENTION),
    ("math", SDPBackend.MATH),
):
    rec = {}
    try:
        q, k, v = [t.to(torch.bfloat16).clone().requires_grad_() for t in base]
        with sdpa_kernel(be):
            o = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
            o.backward(go.to(torch.bfloat16))

        def rel(a, b):
            return ((a.double() - b).norm() / b.norm()).item()

        rec["fwd_relerr_vs_fp64"] = rel(o, o_ref)
        rec["grad_relerr_vs_fp64"] = {
            n: rel(g, r) for n, g, r in zip("qkv", (q.grad, k.grad, v.grad), g_ref, strict=True)
        }
        # causal backward check: perturb q/k/v at positions > p
        p = 1000
        q2, k2, v2 = [t.detach().clone() for t in (q, k, v)]
        for t in (q2, k2, v2):
            t[:, :, p + 1 :] = torch.randn_like(t[:, :, p + 1 :])
        q2, k2, v2 = [t.requires_grad_() for t in (q2, k2, v2)]
        g2 = go.to(torch.bfloat16).clone()
        g2[:, :, p + 1 :] = 0
        q1, k1, v1 = [t.detach().clone().requires_grad_() for t in (q, k, v)]
        with sdpa_kernel(be):
            o1 = F.scaled_dot_product_attention(q1, k1, v1, is_causal=True, enable_gqa=True)
            o1.backward(g2)
            o2 = F.scaled_dot_product_attention(q2, k2, v2, is_causal=True, enable_gqa=True)
            o2.backward(g2)
        rec["causal_fwd_prefix_maxdiff"] = (o1[:, :, : p + 1] - o2[:, :, : p + 1]).abs().max().item()
        rec["causal_bwd_prefix_grad_maxdiff"] = max(
            (a.grad[:, :, : p + 1] - b.grad[:, :, : p + 1]).abs().max().item()
            for a, b in ((q1, q2), (k1, k2), (v1, v2))
        )
        rec["status"] = "ok"
        rec["pass"] = (
            rec["fwd_relerr_vs_fp64"] < 2e-2
            and max(rec["grad_relerr_vs_fp64"].values()) < 5e-2
            and rec["causal_fwd_prefix_maxdiff"] == 0
            and rec["causal_bwd_prefix_grad_maxdiff"] == 0
        )
    except RuntimeError as e:
        rec = {"status": "unavailable", "error": str(e).splitlines()[0][:200]}
    out[name] = rec
print(json.dumps(out, indent=1))
