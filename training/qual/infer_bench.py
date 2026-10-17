"""Inference latency benchmark for Juniper 1 shapes (objectives L1-L4 in docs/prep/01).

Decoding cost does not depend on weight values, so untrained weights give valid
latency and memory figures. Quality is measured elsewhere. One run measures the full
generate loop through the PyTorch eager runtime with a per-layer KV cache:

* TTFT: model already loaded and warm; prefill of the prompt plus the first sampled token.
* Decode throughput: the next ``--gen`` tokens, one at a time, greedy (argmax).
* Cold start: process-local model construction and weight load, reported separately.
* Memory: CUDA peak allocated/reserved, process peak RSS.

The median and p95 are reported over ``--repeats`` runs, with CUDA synchronized at every
timing point.
"""

from __future__ import annotations

import argparse
import json
import platform
import resource
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from j1bench.model import FAMILY, ModelConfig, Transformer


def sync(dev: torch.device) -> None:
    if dev.type == "cuda":
        torch.cuda.synchronize()


def one_run(model: Transformer, prompt: torch.Tensor, gen: int, dev: torch.device) -> tuple[float, float]:
    caches: list[list] = [[] for _ in model.blocks]
    sync(dev)
    t0 = time.perf_counter()
    logits = model.step_logits(prompt, caches, 0)
    nxt = logits.argmax(-1, keepdim=True)
    sync(dev)
    ttft = time.perf_counter() - t0
    pos = prompt.shape[1]
    t1 = time.perf_counter()
    for _ in range(gen):
        logits = model.step_logits(nxt, caches, pos)
        nxt = logits.argmax(-1, keepdim=True)
        pos += 1
    sync(dev)
    return ttft, gen / (time.perf_counter() - t1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="j1-core-28x640", choices=sorted(FAMILY))
    ap.add_argument("--vocab", type=int, default=32768)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="fp16", choices=["fp16", "bf16", "fp32"])
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--prompt", type=int, default=256)
    ap.add_argument("--gen", type=int, default=256)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    dev = torch.device(a.device)
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[a.dtype]
    t0 = time.perf_counter()
    cfg = ModelConfig(vocab_size=a.vocab, max_seq_len=a.ctx, **FAMILY[a.preset])
    torch.manual_seed(0)
    model = Transformer(cfg).to(device=dev, dtype=dtype).eval()
    sync(dev)
    cold = time.perf_counter() - t0
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    prompt = torch.randint(320, a.vocab, (1, a.prompt), device=dev)
    one_run(model, prompt, min(16, a.gen), dev)  # warm-up (allocator, kernels)
    ttfts, rates = [], []
    for _ in range(a.repeats):
        ttft, rate = one_run(model, prompt, a.gen, dev)
        ttfts.append(ttft)
        rates.append(rate)

    def p95(xs: list[float]) -> float:
        return sorted(xs)[min(len(xs) - 1, round(0.95 * (len(xs) - 1)))]

    res = {
        "preset": a.preset,
        "n_params": model.num_params(),
        "device": torch.cuda.get_device_name(dev) if dev.type == "cuda" else platform.processor() or platform.machine(),
        "dtype": a.dtype,
        "threads": a.threads,
        "prompt_tokens": a.prompt,
        "generated_tokens": a.gen,
        "repeats": a.repeats,
        "runtime": f"pytorch {torch.__version__} eager, KV cache, greedy, batch 1",
        "cold_construct_s": round(cold, 3),
        "ttft_s_median": round(statistics.median(ttfts), 4),
        "ttft_s_p95": round(p95(ttfts), 4),
        "decode_tok_s_median": round(statistics.median(rates), 2),
        "decode_tok_s_p5": round(sorted(rates)[max(0, round(0.05 * (len(rates) - 1)))], 2),
        "peak_rss_mib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
    }
    if dev.type == "cuda":
        res["cuda_peak_allocated_mib"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
        res["cuda_peak_reserved_mib"] = round(torch.cuda.max_memory_reserved() / 2**20, 1)
    line = json.dumps(res)
    print(line)
    if a.out:
        with Path(a.out).open("a") as f:
            f.write(line + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
