"""torch.profiler kernel breakdown for a configuration, after warm-up/compilation.
Usage: python qual/profile_step.py OUT.json [train args...]"""

import json
import sys
import time
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

from j1bench.train import Trainer, build_parser

out = sys.argv[1]
t = Trainer(build_parser().parse_args(sys.argv[2:]))
for _ in range(8):
    t.train_step()
torch.cuda.synchronize()
t0 = time.perf_counter()
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    for _ in range(5):
        t.train_step()
    torch.cuda.synchronize()
wall_ms = (time.perf_counter() - t0) * 1e3
# Kernel events only: aten:: CPU ops also report the device time of the kernels
# they launch, which would double-count.
ev = [e for e in prof.key_averages() if e.device_type == torch.autograd.DeviceType.CUDA]
cuda_total = sum(e.self_device_time_total for e in ev)
rows = sorted(
    (
        {"name": e.key[:120], "self_cuda_us": e.self_device_time_total, "calls": e.count}
        for e in ev
        if e.self_device_time_total > 0
    ),
    key=lambda r: -r["self_cuda_us"],
)
wall = {}
groups = {}
for r in rows:
    n = r["name"].lower()
    g = (
        "gemm"
        if any(x in n for x in ("gemm", "sm80_xmma", "cutlass", "ampere_bf16", "cublas", "s16816"))
        else "attention"
        if any(x in n for x in ("flash", "fmha", "attention", "sdpa"))
        else "triton_fused"
        if "triton" in n
        else "optimizer"
        if "adam" in n or "multi_tensor" in n
        else "copy/memset"
        if any(x in n for x in ("memcpy", "memset", "copy"))
        else "other"
    )
    groups[g] = groups.get(g, 0) + r["self_cuda_us"]
res = {
    "steps_profiled": 5,
    "cuda_time_total_ms": cuda_total / 1e3,
    "wall_ms_under_profiler": wall_ms,
    "gpu_busy_fraction": cuda_total / 1e3 / wall_ms,
    "groups_pct": {k: round(100 * v / cuda_total, 1) for k, v in sorted(groups.items(), key=lambda x: -x[1])},
    "top_kernels": rows[:20],
}
Path(out).write_text(json.dumps(res, indent=1))
print(
    json.dumps({k: res[k] for k in ("cuda_time_total_ms", "wall_ms_under_profiler", "gpu_busy_fraction", "groups_pct")})
)
