"""Compute, wall-clock and cost projections for Juniper 1 training (docs/prep/10).

FLOP convention (PaLM appendix B): training FLOPs per token = 6 * N_matmul + 12 * L * H * d_head * T,
where N_matmul counts every weight used in a matmul. That includes the tied output
projection (V * d) but not the input lookup. Causal masking is ignored, so the attention
term is an upper bound.

Inputs are measured throughputs (valid training tokens/s at seq 2048, BF16, compiled) per
model shape, from ``--measured`` (a JSON {preset: {"tokens_per_s": x, "source": "..."}}).
Shapes without a measurement are projected from the closest measured shape by an
equal-MFU assumption and flagged ``projected``.

Scenarios:
* optimistic   = measured throughput, overhead 3.3 % (measured checkpoint cost), no interruptions
* expected     = 0.95 x throughput, overheads 3.3 % checkpoint + 3 % eval + 2 % startup/compile + 5 % interruption
* conservative = 0.85 x throughput, overheads 5 % checkpoint + 5 % eval + 3 % startup + 15 % interruption
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from j1bench.model import FAMILY, ModelConfig, analytic_params

SCENARIOS = {
    "optimistic": {"tput": 1.0, "overhead": 0.033},
    "expected": {"tput": 0.95, "overhead": 0.033 + 0.03 + 0.02 + 0.05},
    "conservative": {"tput": 0.85, "overhead": 0.05 + 0.05 + 0.03 + 0.15},
}


def flops_per_token(cfg: ModelConfig, seq: int) -> float:
    n_matmul = analytic_params(cfg) - cfg.d_model * (2 * cfg.n_layer + 1)  # minus norm weights
    n_matmul -= cfg.n_layer * 2 * cfg.head_dim if cfg.qk_norm == "rmsnorm_per_head" else 0
    return 6 * n_matmul + 12 * cfg.n_layer * cfg.n_head * cfg.head_dim * seq


def project(measured: dict, vocab: int, seq: int, rate: float, exposures: list[float]) -> dict:
    out = {}
    ref = None
    for name, m in measured.items():
        cfg = ModelConfig(vocab_size=vocab, **FAMILY[name])
        ref = ref or (m["tokens_per_s"] * flops_per_token(cfg, seq), name)
    for name, shape in FAMILY.items():
        cfg = ModelConfig(vocab_size=vocab, **shape)
        fpt = flops_per_token(cfg, seq)
        if name in measured:
            tps, basis = measured[name]["tokens_per_s"], f"measured: {measured[name]['source']}"
        elif ref:
            tps, basis = ref[0] / fpt, f"projected at equal MFU from {ref[1]}"
        else:
            continue
        rows = {}
        for exp in exposures:
            rows[f"{exp / 1e9:g}B"] = {
                sc: {
                    "hours": round(exp / (tps * p["tput"]) / 3600 * (1 + p["overhead"]), 2),
                    "usd": round(exp / (tps * p["tput"]) / 3600 * (1 + p["overhead"]) * rate, 2),
                }
                for sc, p in SCENARIOS.items()
            }
        out[name] = {
            "params": analytic_params(cfg),
            "train_flops_per_token": fpt,
            "tokens_per_s": round(tps),
            "basis": basis,
            "mfu_at_312tflops": round(tps * fpt / 312e12, 3),
            "exposures": rows,
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--measured", type=Path, required=True)
    ap.add_argument("--vocab", type=int, default=32768)
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--rate", type=float, default=1.93, help="USD per billed pod hour (GPU + container disk)")
    ap.add_argument("--exposures", default="3e9,4.5e9,6e9,15e9")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    res = {
        "convention": "6*N_matmul + 12*L*H*d_head*T per training token (PaLM); causal saving ignored",
        "vocab": a.vocab,
        "seq": a.seq,
        "usd_per_hour": a.rate,
        "scenarios": SCENARIOS,
        "models": project(
            json.loads(a.measured.read_text()), a.vocab, a.seq, a.rate, [float(x) for x in a.exposures.split(",")]
        ),
    }
    text = json.dumps(res, indent=1)
    if a.out:
        a.out.write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
