# FLOWBOX pilot plan — SUPERSEDED (kept as a fallback)

**Status (2026-10-08 ~23:05Z):** the owner will restore A100 access after a FLOWBOX restart and
prefers the A100 for all heavy GPU work. This plan is therefore **not executed**. It stays
valid as the fallback if A100 access is not restored. The FP16 and expanded-KV code paths
remain for FLOWBOX smoke tests and inference work. See `13_a100_runbook.md`.

## Original plan (pre-registered; written when RunPod was suspended)

**Why:** the owner suspended all RunPod use on 2026-10-08 at about 22:00Z. The $10 stays
reserved but unauthorized. The A100 plan (`09_pilot_plan.md`) is unchanged and runs, rehearsed,
once authorization returns. Until then the same pilots run on FLOWBOX's RTX 2060 at zero
incremental cost. The selection rules in `08_evaluation_framework.md` §3 apply unchanged.

## Hardware-driven differences from the A100 contract

| Item | A100 plan | FLOWBOX | Consequence |
|---|---|---|---|
| Precision | BF16 autocast | **FP16 autocast + dynamic loss scaling** (Turing has no BF16 tensor cores) | Learning dynamics may differ slightly from BF16. Optimal LRs are indicative and not transferred blindly |
| GQA attention | SDPA `enable_gqa` | **expanded K/V heads** (`--gqa-kernel expand`; identical math, test `test_expanded_kv_attention_equals_native_gqa`), because Turing's fused kernels lack GQA and the math fallback needs O(T²) memory | none for learning; slower |
| Micro-batch × accumulation (65,536 tokens/update everywhere) | 16 × 2 | micro 4×8, small 4×8, mid 2×16, core/deep/wide 1×32 | equal tokens per update; per-step noise identical in expectation |
| Token budget per run | ≈175M | **≈75M** (equal-bytes prefix of ≈300 MB text) | ≈0.5 tokens/parameter at 148M: early-training evidence only |
| Throughput (measured 2026-10-08, compiled, while the CPU was busy with the corpus rebuild) | — | micro 103K, small 50K, mid 21K, core 9.5K tok/s | real-data and idle-CPU numbers replace these |

## Order (stops early if time runs out; nothing is cut silently)

1. LR sweep: small, 3 LRs (6e-4, 1.2e-3, 2.4e-3), 24M tokens each. ≈ 25 min.
2. Tokenizer pilots: small on each candidate's equal-bytes prefix. ≈ 75 min. Selection rule §3.1.
3. Family on the selected tokenizer: micro (≈12 min), mid (≈60 min), core (≈2.2 h).
4. Seed repeats: small × 2. ≈ 50 min.
5. Shape alternatives: deep 30×576, wide 20×768. ≈ 2.2 h each.
6. Resume drill on the core run.

Every run writes §8 time logs. CPU-heavy work (corpus finalization, tokenization, sharding)
does not overlap pilot runs except where noted. The memory rule on FLOWBOX still applies: no
GPT-OSS server runs alongside.

## What remains A100-only (blocked until re-authorization)

- BF16 learning-equivalence and throughput on real data with the frozen tokenizer
  (spec Appendix B).
- Real-data checkpoint cadence and save cost on the A100 stack, replacement-pod recovery
  of real-data checkpoints, and the logger qualification on the A100.
- The cost model's measured A100 throughput per shape (only the 148M core has a measured
  synthetic number; the other shapes are projected).
