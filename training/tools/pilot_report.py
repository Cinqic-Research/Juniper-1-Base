# ruff: noqa: E501, RUF001  (Markdown table formatting: long literal lines and the multiplication sign are intended)
"""Tabulates the 2026-10-10 pilot evidence (results/pilots-2026-10-10) into Markdown tables for docs/prep/17."""

from __future__ import annotations

import json
import sys
from pathlib import Path

R = Path(sys.argv[1])
G = R / "4090-hh2ru2zuuetw4x"
A = R / "a100-qz3l9wpr1kgx3r"
S = R / "selections"


def load(p: Path) -> dict:
    return json.loads(p.read_text())


out = ["## Tokenizer pilots (RTX 4090, j1-small-12x384, exact epoch over the v5 700 MB prefix)\n"]
out.append(
    "| Candidate | Vocab | Params | Seed | Val bpb | Val loss/token | Consumed source bytes | Train tok/s (4090) |"
)
out.append("|---|---:|---:|---:|---:|---:|---:|---:|")
domains: dict[str, dict] = {}
for c in ("24k", "28k", "32k"):
    for s in (1, 2):
        d = load(G / f"tok-j1tok-{c}-s{s}.json")
        det = d["last_eval_detail"]
        out.append(
            f"| j1tok-{c} | {d['model']['vocab_size']} | {d['n_params']:,} | {s} | {det['bpb']:.5f} | {d['last_eval']:.4f} | "
            f"{d['consumed']['target_bytes']:,} | {d['tokens_per_s_median']:,.0f} |"
        )
        for src, v in det["by_source"].items():
            domains.setdefault(src, {}).setdefault(c, []).append(v["bpb"])
out.append("\n### Validation bpb by source (mean of seeds 1 and 2)\n")
out.append("| Source | 24K | 28K | 32K |")
out.append("|---|---:|---:|---:|")
for src in sorted(domains):
    out.append(
        f"| {src} | "
        + " | ".join(f"{sum(domains[src][c]) / len(domains[src][c]):.4f}" for c in ("24k", "28k", "32k"))
        + " |"
    )
sel = load(S / "select_tokenizer.json")
out.append(
    f"\n**Rule {sel['rule']}: {sel['status']} → {sel['decision']}**; best {sel['best_bpb']:.5f}; tied {sel['tied']}; "
    f"pooled relative seed SD {sel['pooled_rel_seed_sd']:.4%} ({sel['pooled_dof']} dof).\n"
)
out.append(
    "| Candidate | Mean bpb | 4090 train tokens/s (mean) | A100 training bytes/s | Overall fallback | Hard req. |"
)
out.append("|---|---:|---:|---:|---:|---|")
for r in sel["rows"]:
    out.append(
        f"| {r['candidate']} | {r['val_bpb']:.5f} | {r['train_tokens_per_s']:,.0f} | {r['train_bytes_per_s']:,.0f} | {r['byte_fallback_rate']:.4%} | {r['hard_requirements']} |"
    )
for c in ("24k", "28k", "32k"):
    t = load(A / f"tp-j1tok-{c}.json")
    out.append(
        f"\nA100 tp-j1tok-{c}: {t['tokens_per_s_median']:,.0f} tokens/s median over {t['steady_steps']} steady updates, peak {t['peak_vram_gib']:.1f} GiB."
    )
out.append("\n## Architecture LR check (RTX 4090, 15 % prefix, micro-batch 8 × 4)\n")
out.append("| Shape | ×0.5 | ×1 | ×2 | Selected LR |")
out.append("|---|---:|---:|---:|---:|")
for p in ("j1-core-28x640", "j1-deep-30x576", "j1-wide-20x768"):
    r = load(S / f"select_lr_{p}.json")
    v = {x["run"].rsplit("-x", 1)[1]: x["val_loss"] for x in r["rows"]}
    out.append(f"| {p} | {v['0.5']:.4f} | {v['1']:.4f} | {v['2']:.4f} | {r['decision']:.4e} |")
sweep = load(S / "select_lr_small.json")
out.append(
    "\nSmall-model LR sweep (60M tokens, 32K candidate): "
    + ", ".join(f"{x['lr']}: {x['val_loss']:.4f}" for x in sweep["rows"])
    + f" → {sweep['decision']}."
)
arch = S / "select_architecture.json"
if arch.exists():
    a = load(arch)
    out.append(
        f"\n## Architecture comparison (A100, j1tok-24k, full prefix)\n\n**Rule {a['rule']}: {a['status']} → {a['decision']}** "
        f"(noise band {a.get('noise_band_rel', float('nan')):.3%}; equal-compute T* = {a.get('equal_compute_seconds', float('nan')):.0f} s).\n"
    )
    out.append(
        "| Shape | Params | LR | Equal-token bpb | Equal-compute bpb | Δ tok | Δ compute | tok/s | Train s | Peak GiB |"
    )
    out.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in a["rows"]:
        out.append(
            f"| {r['preset']} | {r['n_params']:,} | {r['lr']:.3e} | {r['equal_token_bpb']:.5f} | {r['equal_compute_bpb']:.5f} | "
            f"{r['equal_token_improvement_rel']:+.3%} | {r['equal_compute_improvement_rel']:+.3%} | {r['tokens_per_s_median']:,.0f} | "
            f"{r['train_seconds_excl_first_update']:.0f} | {r['peak_vram_gib']:.1f} |"
        )
print("\n".join(out))
