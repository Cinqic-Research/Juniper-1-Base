# SFT pilot seed replication v1: results (2026-10-13)

Protocol: [`sft_replication_protocol_v1.json`](../../sft_replication_protocol_v1.json) (registered before any
replication result); stage authorization [`stage_auth/sft_replication_v1.json`](../../stage_auth/sft_replication_v1.json).
Seeds 2 and 3 repeat the protocol-v3 pilot recipe exactly (seed 1 = the original pilot, unchanged). Hardware
FLOWBOX RTX 2060, torch 2.8.0+cu128, fp32, TF32 off, deterministic — identical to seed 1. Trainer package
`f13afbdf…` (same as seed 1). RunPod cost $0.

## Runs

| seed | run | contract | final checkpoint | manifest | model state |
|---|---|---|---|---|---|
| 1 | `j1-sft-v1-pilot` (original) | `561fdeba…` | `ckpt-000001226-55d13f9a` | `dc0bbbb0…` | `a4616f90…` |
| 2 | `j1-sft-v1-rep-s2` | `5a8af04d…` | `ckpt-000001226-0ccc062e` | `ace56e13174cd34d…3e4f` | `eaf742c1…a502` |
| 3 | `j1-sft-v1-rep-s3` | `f9b44d8f…` | `ckpt-000001226-4389d0e3` | `3c0163f7d6a65d76…8a2d` | `4d60c4cf…0e36` |

All 1,226 updates completed; no early stop. Every logged batch fingerprint of seeds 2 and 3 differs from
seed 1 (the seed changes the data order as intended). Each final checkpoint verifies and reproduces its
recorded dev loss exactly. Held-out-JSON generation re-runs: 200/200 identical (seeds 2, 3). Decision
under the unchanged v3 rule: **`PILOT_CANDIDATE_QUALIFIED` for all three seeds** (non-production; a
replication promotes nothing).

## Per seed (test split, strict, original scoring: Data `f0399ed`, scorer v3)

| metric | n | s1 | s2 | s3 | mean | sd |
|---|---|---|---|---|---|---|
| instruction following, in distribution (P1) | 500 | 0.648 | 0.662 | 0.662 | 0.657 | 0.008 |
| held-out instruction families | 272 | 0.011 | 0.011 | 0.011 | 0.011 | 0.000 |
| JSON / held-out JSON | 500 / 200 | 0.704 / 0.260 | 0.700 / 0.250 | 0.698 / 0.245 | 0.701 / 0.252 | |
| math (arithmetic 600 / GSM8K 500) | 1,100 | 0.370 (0.668 / 0.012) | 0.376 (0.680 / 0.012) | 0.375 (0.668 / 0.024) | 0.374 | 0.003 |
| facts | 176 | 0.301 | 0.284 | 0.307 | 0.297 | 0.012 |
| — capitals (country + US state) | 51 | 2 | 0 | 1 | 1.0 | |
| — continents / currencies | 42 / 35 | 0.881 / 0.114 | same | same | | |
| — element symbol / number | 24 / 24 | 0.417 / 0.000 | 0.375 / 0.000 | 0.458 / 0.042 | | |
| Python (program output 200 / MBPP+HumanEval 664) | 864 | 0.057 (0.160 / 0.026) | 0.058 | 0.060 | 0.058 | 0.002 |
| uncertainty / false abstention on facts | 300 / 176 | 0.993 / 0.074 | 0.997 / 0.068 | 0.990 / 0.063 | | |
| hierarchy / injection | 221 / 300 | 0.425 / 0.960 | 0.416 / 0.950 | 0.421 / 0.947 | | |
| protocol-valid turns | 4,233 | 0.957 | 0.958 | 0.948 | 0.954 | 0.005 |
| full-split bpb (limit 0.943872) | — | 0.940726 | 0.940555 | 0.940843 | 0.940708 | 0.00014 |
| final dev loss | 52,919 targets | 0.60907 | 0.61018 | 0.61083 | 0.61003 | 0.00089 |

Paired between seeds (same items, exact McNemar, Holm over the three pairs within each slice): no slice
differs significantly (smallest adjusted p 0.54, GSM8K); P1 smallest adjusted p 0.79. Registered
consistency check (`seed_report_original.json`): same decision, no P1 difference, all pass G-RET →
**consistent**; seed 1's scores are usable as a baseline for SFT v2 thresholds.

## Corrected rescoring view (labelled separately)

Scorer v4 (finding S-09) at Data `67aef244` rescored the preserved raw generations of all three seeds and
of the base: **0 items changed** in any of them (the candidates answer uncertainty items only with the
trained "I don't know."; the base never produced a protocol-valid turn). Recomputed comparisons and
decisions: identical (`seed_report_corrected_v4.json`). This view is not a replication of the unchanged
evaluation and is not merged with the original numbers.

## What the three seeds do and do not show

- The recipe is stable at this scale: seed spread is small relative to item-level uncertainty (P1 Wilson
  95% per seed ≈ ±0.04; seed range 0.014). Three seeds support "no large seed sensitivity", not a precise
  seed-variance estimate.
- The main weaknesses replicate exactly: held-out instruction families 3/272 in every seed; capitals 0–2
  of 51 (base few-shot 11/51); GSM8K 1–2%; retention uses ~88% of the bpb allowance in every seed.
- Coding results are weaker than the upstream tests suggest: of seed 1's 17 MBPP/HumanEval passes, 3 are
  wrong code (independent cross-execution check, Data `docs/sft/codecheck_v1.md`).
- Protocol validity of seed 3 (0.948) would fail a 0.95 gate; it passes v3's 0.90.

Evidence: per-seed `decision.json`, `comparison.json`, retention, checkpoint check, dev history and score
summaries in `seed2/`, `seed3/` (seed 1: `../sft_pilot_v1/`). Raw generations and per-item scores: FLOWBOX
`~/j1post/eval/replication/`, HDD archive `juniper1-archive/assignment-2026-10-13/`.
