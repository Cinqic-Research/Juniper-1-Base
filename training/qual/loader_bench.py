"""Data-loader throughput for real token shards (directive §4.3: benchmark the loader against
the A100's expected consumption rate).

Measures ``TokenStream.next_batch`` (shuffled windows, micro-batch x seq_len), including the
host→tensor copy, over a fixed number of batches. Two passes: the first is cold (as far as the OS
page cache allows), the second warm over the same windows. The requirement is
``tokens_per_s >= target`` (default 115,249 tok/s, the measured compiled A100 rate for the 148M
core) with margin. Data time per step is also reported against the 284 ms A100 step.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from j1bench.data import TokenStream


def run(stream: TokenStream, batches: int) -> dict:
    t0 = time.perf_counter()
    per = []
    for _ in range(batches):
        b0 = time.perf_counter()
        stream.next_batch()
        per.append(time.perf_counter() - b0)
    dt = time.perf_counter() - t0
    tokens = batches * stream.batch_size * stream.seq_len
    per.sort()
    return {
        "batches": batches,
        "seconds": round(dt, 3),
        "tokens_per_s": round(tokens / dt),
        "batch_ms_median": round(per[len(per) // 2] * 1e3, 3),
        "batch_ms_p99": round(per[int(0.99 * (len(per) - 1))] * 1e3, 3),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("shards", type=Path)
    ap.add_argument("--micro-bs", type=int, default=16)
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--batches", type=int, default=400)
    ap.add_argument("--target", type=float, default=115249.0)
    ap.add_argument("--preload", action="store_true")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    t0 = time.perf_counter()
    s = TokenStream(str(a.shards), "train", a.seq_len, a.micro_bs, seed=1, verify=False, preload=a.preload)
    open_s = time.perf_counter() - t0
    cold = run(s, a.batches)
    s.load_state_dict({"epoch": 0, "cursor": 0, "batches": 0})
    warm = run(s, a.batches)
    res = {
        "shards": str(a.shards),
        "preload": a.preload,
        "stream_tokens": s.n_tokens,
        "windows": s.n_windows,
        "open_s": round(open_s, 3),
        "cold": cold,
        "warm": warm,
        "target_tokens_per_s": a.target,
        "a100_step_ms_core_mb16": 284,
        "pass": min(cold["tokens_per_s"], warm["tokens_per_s"]) >= 2 * a.target,
    }
    text = json.dumps(res, indent=1)
    if a.out:
        a.out.write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
