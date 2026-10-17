"""CUDA stack qualification: discovery, BF16 matmul accuracy, autograd, optimizer,
sync, error propagation, H2D/D2H bandwidth."""

import json
import subprocess
import sys
import time

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

out = {"torch": torch.__version__, "cuda_runtime": torch.version.cuda, "cudnn": torch.backends.cudnn.version()}
assert torch.cuda.is_available(), "CUDA not available"
d = torch.device("cuda")
p = torch.cuda.get_device_properties(0)
out["device"] = {
    "name": p.name,
    "total_mem_gib": round(p.total_memory / 2**30, 2),
    "cc": f"{p.major}.{p.minor}",
    "sms": p.multi_processor_count,
}
out["bf16_supported"] = torch.cuda.is_bf16_supported()

# BF16 matmul vs FP64 reference
g = torch.Generator(device=d).manual_seed(0)
a = torch.randn(4096, 4096, device=d, generator=g)
b = torch.randn(4096, 4096, device=d, generator=g)
ref = a.double() @ b.double()
for dt in (torch.bfloat16, torch.float32):
    torch.backends.cuda.matmul.allow_tf32 = False
    c = (a.to(dt) @ b.to(dt)).double()
    rel = ((c - ref).norm() / ref.norm()).item()
    out[f"matmul_relerr_{str(dt).split('.')[-1]}"] = rel
assert out["matmul_relerr_bfloat16"] < 1e-2


def tflops(dt, n=8192, it=20):
    x = torch.randn(n, n, device=d, dtype=dt)
    y = torch.randn(n, n, device=d, dtype=dt)
    for _ in range(3):
        x @ y
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(it):
        x @ y
    torch.cuda.synchronize()
    return 2 * n**3 * it / (time.perf_counter() - t) / 1e12


out["tflops_bf16_8192"] = round(tflops(torch.bfloat16), 1)
torch.backends.cuda.matmul.allow_tf32 = True
out["tflops_tf32_8192"] = round(tflops(torch.float32), 1)
torch.backends.cuda.matmul.allow_tf32 = False
out["tflops_fp32_8192"] = round(tflops(torch.float32, it=5), 1)

# autograd + AdamW (fused and foreach) step changes params and matches each other
torch.manual_seed(0)
m1 = torch.nn.Linear(256, 256).to(d)
m2 = torch.nn.Linear(256, 256).to(d)
m2.load_state_dict(m1.state_dict())
o1 = torch.optim.AdamW(m1.parameters(), lr=1e-3, fused=True)
o2 = torch.optim.AdamW(m2.parameters(), lr=1e-3, foreach=True)
x = torch.randn(64, 256, device=d)
w0 = m1.weight.detach().clone()
for m, o in ((m1, o1), (m2, o2)):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = m(x).float().pow(2).mean()
    loss.backward()
    assert m.weight.grad is not None and torch.isfinite(m.weight.grad).all()
    o.step()
out["adamw_param_changed"] = bool((m1.weight != w0).any())
out["adamw_fused_vs_foreach_maxdiff"] = (m1.weight - m2.weight).abs().max().item()

# SDPA backends available for bf16 causal GQA-shaped inputs
q = torch.randn(4, 10, 2048, 64, device=d, dtype=torch.bfloat16)
k = torch.randn(4, 10, 2048, 64, device=d, dtype=torch.bfloat16)
res = {}
for be in (SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.CUDNN_ATTENTION, SDPBackend.MATH):
    try:
        with sdpa_kernel(be):
            torch.nn.functional.scaled_dot_product_attention(q, k, k, is_causal=True)
        torch.cuda.synchronize()
        res[be.name] = "ok"
    except Exception as e:
        res[be.name] = f"unsupported: {type(e).__name__}: {str(e)[:120]}"
out["sdpa_backends"] = res

# pinned host <-> device bandwidth
buf = torch.empty(1 << 30, dtype=torch.uint8).pin_memory()
dbuf = torch.empty_like(buf, device=d)
torch.cuda.synchronize()
t = time.perf_counter()
dbuf.copy_(buf, non_blocking=True)
torch.cuda.synchronize()
out["h2d_pinned_GBps"] = round(1 / (time.perf_counter() - t), 2)
t = time.perf_counter()
buf.copy_(dbuf)
torch.cuda.synchronize()
out["d2h_pinned_GBps"] = round(1 / (time.perf_counter() - t), 2)
pg = torch.empty(1 << 30, dtype=torch.uint8)
t = time.perf_counter()
dbuf.copy_(pg)
torch.cuda.synchronize()
out["h2d_pageable_GBps"] = round(1 / (time.perf_counter() - t), 2)

# a device-side assert must surface as a nonzero process exit
r = subprocess.run(
    [
        sys.executable,
        "-c",
        "import torch; t=torch.zeros(4,device='cuda'); i=torch.tensor([10],device='cuda'); t[i]=1; "
        "torch.cuda.synchronize(); print('NOT REACHED')",
    ],
    capture_output=True,
    text=True,
)
out["device_assert_exit_code"] = r.returncode
out["device_assert_stderr_tail"] = r.stderr.strip().splitlines()[-1][:200] if r.stderr else ""
assert r.returncode != 0 and "NOT REACHED" not in r.stdout
r = subprocess.run(
    [sys.executable, "-c", "import torch; torch.empty(200*2**30, dtype=torch.uint8, device='cuda')"],
    capture_output=True,
    text=True,
)
out["oom_exit_code"] = r.returncode
out["oom_is_OutOfMemoryError"] = "OutOfMemoryError" in r.stderr
assert r.returncode != 0
print(json.dumps(out, indent=1))
