"""Summarizes bench_matrix.sh outputs into results CSV/JSON.

GPU utilization, memory and power are averaged over the steady-state window
(timestamps of the first and last steady steps from the metrics JSONL), so
compilation and warm-up are excluded. Failed runs are kept with their exit code.

Usage: python tools/summarize_bench.py OUT_PREFIX DIR [DIR ...]
"""

import csv
import json
import statistics
import sys
from datetime import UTC, datetime
from pathlib import Path

FLOPS_PEAK_BF16 = 312e12  # A100 dense BF16 tensor-core peak


def steady_window(jsonl: Path, excluded: int):
    steps = [json.loads(x) for x in jsonl.read_text().splitlines() if '"loss_hex"' in x]
    steady = steps[excluded:]
    if not steady:
        return None
    return steady[0]["t"] - steady[0]["step_s"], steady[-1]["t"]


def gpu_stats(csv_path: Path, window):
    rows = []
    for line in csv_path.read_text().splitlines():
        p = [x.strip() for x in line.split(",")]
        try:
            # The pod clock and nvidia-smi timestamps are UTC; parse them as such.
            ts = datetime.strptime(p[0], "%Y/%m/%d %H:%M:%S.%f").replace(tzinfo=UTC).timestamp()
        except ValueError:
            continue
        rows.append((ts, float(p[1]), float(p[3]), float(p[4]), float(p[5])))
    if not rows:
        return {}
    # nvidia-smi timestamps are local pod time (UTC in the container).
    sel = [r for r in rows if window and window[0] <= r[0] <= window[1]]
    if not sel:
        return {"gpu_samples": 0, "gpu_window": "no samples inside steady-state window"}
    return {
        "gpu_window": "steady_state",
        "gpu_util_mean": statistics.mean(r[1] for r in sel),
        "gpu_util_min": min(r[1] for r in sel),
        "gpu_mem_used_mib_max": max(r[2] for r in sel),
        "power_w_mean": statistics.mean(r[3] for r in sel),
        "sm_clock_mhz_mean": statistics.mean(r[4] for r in sel),
        "gpu_samples": len(sel),
        "sample_interval_ms": 500,
    }


def main():
    out, dirs = sys.argv[1], sys.argv[2:]
    rows = []
    for d in map(Path, dirs):
        for st in sorted(d.glob("*.status")):
            name = st.stem
            status = json.loads(st.read_text())
            row = {
                "group": d.name,
                "name": name,
                "args": status["args"],
                "exit_code": status["exit_code"],
                "start_utc": status["start_utc"],
                "end_utc": status["end_utc"],
            }
            sp = d / f"{name}.json"
            if sp.exists():
                s = json.loads(sp.read_text())
                n = s["n_params"]
                cfg = s["model"]
                seq = s["config"]["seq_len"]
                # 6N per token plus causal attention matmuls (QK^T and AV, fwd+bwd, halved by causality).
                attn = 6 * cfg["n_layer"] * cfg["n_head"] * cfg["head_dim"] * seq
                tps = s["tokens_per_s_median"]
                row.update(
                    {
                        k: s[k]
                        for k in (
                            "n_params",
                            "steps_run",
                            "steady_steps",
                            "first_step_s",
                            "step_ms_median",
                            "step_ms_p90",
                            "step_ms_p99",
                            "step_ms_max",
                            "tokens_per_step",
                            "tokens_per_s_median",
                            "tokens_per_s_mean",
                            "tokens_useful",
                            "tokens_total",
                            "peak_vram_gib",
                            "peak_vram_reserved_gib",
                            "peak_host_rss_gib",
                            "final_param_sha256",
                        )
                    }
                )
                row["mfu_estimate"] = (6 * n + attn) * tps / FLOPS_PEAK_BF16
                row["useful_fraction"] = s["tokens_useful"] / s["tokens_total"]
                win = steady_window(d / f"{name}.jsonl", s["config"]["warmup_steps_excluded"])
                g = d / f"{name}.gpu.csv"
                if g.exists():
                    row.update(gpu_stats(g, win))
                dts = [
                    json.loads(x)["data_s"] for x in (d / f"{name}.jsonl").read_text().splitlines() if '"data_s"' in x
                ]
                row["data_s_median_ms"] = statistics.median(dts) * 1000 if dts else None
            else:
                log = (d / f"{name}.log").read_text().strip().splitlines()
                row["failure"] = next((x.strip() for x in reversed(log) if "Error" in x), log[-1] if log else "")[:300]
            rows.append(row)
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with Path(f"{out}.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    Path(f"{out}.json").write_text(json.dumps(rows, indent=1))
    for r in rows:
        print(
            f"{r['group']:12} {r['name']:18} rc={r['exit_code']} "
            + (
                f"{r['tokens_per_s_median']:>8.0f} tok/s p50 {r['step_ms_median']:.0f} p99 {r['step_ms_p99']:.0f}ms "
                f"vram {r['peak_vram_gib']:.1f} mfu {r['mfu_estimate']:.2f} util {r.get('gpu_util_mean', 0):.0f}% "
                f"{r.get('power_w_mean', 0):.0f}W data {r['data_s_median_ms']:.1f}ms rss {r['peak_host_rss_gib']:.1f}G"
                if "tokens_per_s_median" in r
                else r.get("failure", "")
            )
        )


if __name__ == "__main__":
    main()
