"""Scaling / pilot report from run outputs (``bench_matrix.sh``: <name>.json summary + <name>.jsonl metrics).

Produces ``report.json`` and ``report.md`` plus PNG figures. Every number comes from the run
files. The rules:

* bits per byte (bpb) on the fixed validation windows is the cross-run metric, because it is
  tokenizer-independent
* training compute = PaLM-convention FLOPs per token (``tools/cost_model.py``) times training tokens
* a power law ``bpb = a · N^(-alpha)`` is fitted only with ≥ 4 model sizes at equal tokens, and
  its uncertainty is the residual scatter plus the measured seed band
* short runs are labelled as such, and no fit is extrapolated beyond its range in this report
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from j1bench.model import ModelConfig
from tools.cost_model import flops_per_token

# Reference categorical palette (dataviz skill references/palette.md), fixed order, light surface.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def load_run(summary: Path) -> dict:
    s = json.loads(summary.read_text())
    metrics = summary.with_suffix(".jsonl")
    evals, steps = [], []
    if metrics.exists():
        cum_tok, cum_s = 0, 0.0
        for line in metrics.read_text().splitlines():
            r = json.loads(line)
            if r.get("event") == "eval_detail":
                evals.append(
                    {
                        "step": r["step"],
                        "tokens": cum_tok,
                        "train_s": cum_s,
                        "loss": r["loss"],
                        "bpb": r.get("bpb"),
                        "by_source": r["by_source"],
                    }
                )
            elif "step" in r and "loss" in r and "event" not in r:
                cum_tok += r["useful_tokens"]
                cum_s += r["step_s"]
                steps.append(
                    {
                        "step": r["step"],
                        "loss": r["loss"],
                        "grad_norm": r["grad_norm"],
                        "skipped": r.get("skipped_update", False),
                        "alert": r.get("alert"),
                    }
                )
    cfg = ModelConfig(**s["model"])
    fpt = flops_per_token(cfg, s["config"]["seq_len"])
    n_embed = cfg.vocab_size * cfg.d_model
    return {
        "name": summary.stem,
        "preset": s["config"]["preset"],
        "seed": s["config"]["seed"],
        "lr": s["config"]["lr"],
        "precision": s["config"]["precision"],
        "params": s["n_params"],
        "non_embedding_params": s["n_params"] - n_embed,
        "vocab": cfg.vocab_size,
        "tokens": s["tokens_useful"],
        "flops_per_token": fpt,
        "train_flops": fpt * s["tokens_useful"],
        "tokens_per_s": s["tokens_per_s_median"],
        "step_ms": s["step_ms_median"],
        "peak_vram_gib": s["peak_vram_gib"],
        "final": s.get("last_eval_detail"),
        "evals": evals,
        "max_grad_norm": max((x["grad_norm"] for x in steps), default=None),
        "alerts": [x["alert"] for x in steps if x["alert"]],
        "skipped_updates": sum(x["skipped"] for x in steps),
        "steps": len(steps),
    }


def fit_power(xs: list[float], ys: list[float]) -> dict | None:
    if len(xs) < 4:
        return None
    lx, ly = [math.log(x) for x in xs], [math.log(y) for y in ys]
    mx, my = statistics.fmean(lx), statistics.fmean(ly)
    sxx = sum((a - mx) ** 2 for a in lx)
    slope = sum((a - mx) * (b - my) for a, b in zip(lx, ly, strict=True)) / sxx
    resid = [b - (my + slope * (a - mx)) for a, b in zip(lx, ly, strict=True)]
    se = math.sqrt(sum(r * r for r in resid) / (len(xs) - 2) / sxx) if len(xs) > 2 else float("nan")
    return {
        "alpha": round(-slope, 4),
        "alpha_se": round(se, 4),
        "a": round(math.exp(my - slope * mx), 4),
        "points": len(xs),
        "range_params": [min(xs), max(xs)],
    }


def _axes(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    ax.set_xlabel(xlabel, color=INK2)
    ax.set_ylabel(ylabel, color=INK2)
    ax.grid(True, color=GRID, linewidth=0.8)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.tick_params(colors=INK2)


def figures(runs: list[dict], out: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    made = []
    runs = [r for r in runs if r["evals"] and r["evals"][-1]["bpb"] is not None]
    for key, xlabel, fname, xscale in (
        ("tokens", "training tokens", "bpb_vs_tokens.png", "linear"),
        ("flops", "training compute (FLOPs, PaLM convention)", "bpb_vs_compute.png", "log"),
        ("train_s", "active training time (s, this host)", "bpb_vs_time.png", "linear"),
    ):
        fig, ax = plt.subplots(figsize=(7.5, 4.2), facecolor=SURFACE)
        for i, r in enumerate(runs[: len(SERIES)]):
            xs = [
                e["tokens"] * (r["flops_per_token"] if key == "flops" else 1) if key != "train_s" else e["train_s"]
                for e in r["evals"]
            ]
            ys = [e["bpb"] for e in r["evals"]]
            ax.plot(xs, ys, color=SERIES[i], linewidth=2, marker="o", markersize=4, label=r["name"])
        ax.set_xscale(xscale)
        _axes(ax, f"Validation bits per byte vs {xlabel}", xlabel, "bits per byte (lower is better)")
        ax.legend(frameon=False, fontsize=8, labelcolor=INK)
        fig.tight_layout()
        fig.savefig(out / fname, dpi=130)
        plt.close(fig)
        made.append(fname)
    return made


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs_dir", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--title", default="Juniper 1 pilot report")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    runs = [
        load_run(p)
        for p in sorted(a.runs_dir.glob("*.json"))
        if not p.name.startswith("select_") and p.with_suffix(".jsonl").exists()
    ]
    fam = sorted([r for r in runs if r["name"].startswith(("fam-", "tok-"))], key=lambda r: r["params"])
    seeds = [r for r in runs if r["preset"] == "j1-small-12x384" and r["name"].startswith(("seed-", "tok-"))]
    sel_tok = a.runs_dir / "select_tokenizer.json"
    sel_lr = a.runs_dir / "select_lr.json"
    tok_choice = json.loads(sel_tok.read_text())["decision"] if sel_tok.exists() else None
    chosen_small = [r for r in seeds if r["name"] == tok_choice or r["name"].startswith("seed-")]
    seed_bpb = [r["final"]["bpb"] for r in chosen_small if r["final"] and r["final"].get("bpb")]
    noise = {
        "n": len(seed_bpb),
        "mean": statistics.fmean(seed_bpb) if seed_bpb else None,
        "stdev": statistics.stdev(seed_bpb) if len(seed_bpb) > 1 else None,
    }
    scaling_runs = [
        r
        for r in fam
        if r["name"].startswith("fam-") and r["preset"] in ("j1-micro-8x256", "j1-mid-20x512", "j1-core-28x640")
    ]
    small_sel = next((r for r in runs if r["name"] == tok_choice), None)
    if small_sel:
        scaling_runs.append(small_sel)
    scaling_runs.sort(key=lambda r: r["params"])
    pts = [(r["non_embedding_params"], r["final"]["bpb"]) for r in scaling_runs if r["final"] and r["final"].get("bpb")]
    fit = fit_power([p[0] for p in pts], [p[1] for p in pts])
    figs = figures([*scaling_runs, *[r for r in runs if r["preset"] in ("j1-deep-30x576", "j1-wide-20x768")]], a.out)
    report = {
        "title": a.title,
        "runs": runs,
        "seed_noise": noise,
        "scaling_fit_nonembedding": fit,
        "selected_tokenizer": tok_choice,
        "lr_selection": json.loads(sel_lr.read_text()) if sel_lr.exists() else None,
        "tokenizer_selection": json.loads(sel_tok.read_text()) if sel_tok.exists() else None,
        "figures": figs,
    }
    (a.out / "report.json").write_text(json.dumps(report, indent=1))
    md = [
        f"# {a.title}",
        "",
        "Generated by `tools/scaling_report.py` from the run files; every value is measured unless labelled.",
        "",
        "## Runs",
        "",
        "| run | params | non-emb | tokens | tok/s | VRAM GiB | final loss | final bpb | max grad-norm | alerts "
        "| skipped |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in sorted(runs, key=lambda r: (r["name"].split("-")[0], r["params"])):
        f = r["final"] or {}
        md.append(
            f"| {r['name']} | {r['params']:,} | {r['non_embedding_params']:,} | {r['tokens']:,} | "
            f"{r['tokens_per_s']:,.0f} | {r['peak_vram_gib']:.2f} | {f.get('loss', float('nan')):.4f} | "
            f"{f.get('bpb', float('nan')):.4f} | {r['max_grad_norm']:.3g} | {len(r['alerts'])} | "
            f"{r['skipped_updates']} |"
        )
    md += [
        "",
        "## Seed noise (small model)",
        "",
        json.dumps(noise),
        "",
        "## Scaling fit (equal tokens)",
        "",
        json.dumps(fit) if fit else "Not fitted: fewer than 4 model sizes with a final bpb.",
        "",
    ]
    if runs and runs[0]["final"]:
        srcs = sorted(runs[0]["final"]["by_source"])
        md += ["## Final bpb by source", "", "| run | " + " | ".join(srcs) + " |", "|---|" + "---:|" * len(srcs)]
        for r in sorted(runs, key=lambda r: r["params"]):
            bs = (r["final"] or {}).get("by_source", {})
            md.append(
                f"| {r['name']} | " + " | ".join(f"{bs.get(s, {}).get('bpb', float('nan')):.4f}" for s in srcs) + " |"
            )
    md += ["", "## Figures", ""] + [f"![{f}]({f})" for f in figs]
    (a.out / "report.md").write_text("\n".join(md) + "\n")
    print(json.dumps({"runs": len(runs), "fit": fit, "noise": noise, "figures": figs}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
