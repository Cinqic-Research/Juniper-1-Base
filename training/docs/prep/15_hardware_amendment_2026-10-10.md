# Amendment 15: hardware assignment for the v2 pilots (owner acceleration addendum, $40)

- **Written:** 2026-10-10 ≈ 06:45Z, before any v2 pilot result existed. At this time the A100 pod
  `ixxkq50q949bhl` had produced no training result; no 5090/4090 had been provisioned.
- **Authority:** owner "acceleration and budget amendment" (2026-10-10): $40 total task ceiling
  (ordinary billable work stops at $35; reserve ≥ $5); RTX 5090 primary / RTX 4090 backup, at most eight
  of them concurrently; parallel independent runs, one per GPU; A100 remains the reference for final
  execution qualification and hardware-dependent selection criteria (tokenizer throughput tie-break,
  architecture wall-time comparison); no mixed-hardware timings in a selection; record before results.
- **Unchanged:** every selection rule, threshold and seed requirement of `14_pilot_protocol_v2.md` and
  `08 §3`. This amendment changes only *where* each run executes and how the tie-break throughput is
  sourced.

## 1. Assignment

| Stage (14 §4) | Hardware | Notes |
|---|---|---|
| 0 qualification | A100 and, separately, each new GPU type before scale-out | 5090 path: GPU test suite, SDPA/GQA correctness, and a short real-data run with save + fresh-process resume comparison before any pilot run |
| 1 LR sweep (3 runs) | one homogeneous RTX 5090 pod, one run per GPU | LR is selected on validation loss (not timing) |
| 2 tokenizer pilots (3 candidates × seeds 1, 2; seed 3 if flagged) | the same 5090 pod, one run per GPU, identical settings | validation bpb on identical source bytes; all runs on one device type (rule refuses mixed devices) |
| tokenizer tie-break throughput | A100: `tp-<cand>`, 300 updates each, same model/batch/precision/compile, no eval/checkpoints | training bytes/s = A100 median tokens/s × consumed source bytes per target token of that candidate's prefix (a data property). Used only if candidates are tied |
| 4 architecture LR check (9 runs) | the 5090 pod, one run per GPU, micro-batch 8 × accumulation 4 for all three shapes | loss-based per-shape LR selection; same 65,536-token update |
| 5 architecture full runs (3) | A100, sequential, micro-batch 16 × accumulation 2 | equal-token and equal-compute (A100 wall time) both from these runs only |
| 6 architecture seed repeats | A100 | same device as stage 5 |
| 7 recovery suite | A100 | run while 5090 stages proceed; uses the 28K candidate prefix because checkpoint/resume exactness does not depend on the tokenizer choice (14 §4 said "selected tokenizer"; changed here before any result) |

Noise band (architecture-v1) uses the pooled seed SD of the small-model tokenizer runs, all on the
5090 pod (one device type). Loss values are not timings; the band is a relative bpb variability.

## 2. Budget and controllers

- One task ledger; each paid pod runs its own on-pod controller with a sub-allocation so the sum of hard
  limits stays ≤ $35: A100 soft $14 / hard $15.50; first 5090 pod soft $15 / hard $16.50. Any later
  pod receives an allocation only from unspent headroom, recomputed from the provider billing reads.
- The operator polls provider state and billing for every task resource; a scheduled stop task is a
  further independent layer.

## 3. What this cannot show

Hardware differences between 5090 and A100 numerics (both BF16 autocast, compiled) could shift absolute
losses slightly; within each selection the comparison set is device-homogeneous, which is what the rules
require. No 5090 timing enters any selection.

## 4. Review record (fresh reviewer, 506d2af; before any result)

| ID | Finding | Disposition |
|---|---|---|
| H1 | `--tp` optional: the tie-break could silently use 5090 timings | Fixed: when pilots ran off the A100, the rule requires `--tp`, and the throughput runs must be A100 |
| H2 | A100 timing runs could overlap other A100 work | Fixed by procedure: the A100 runs one GPU process at a time (bootstrap tests, recovery suite, `tp-*`, stage 5/6 strictly sequential). Before each timing run the operator records `nvidia-smi --query-compute-apps` showing no other process; `gpu.csv` samples allow audit |
| M1 | `multi_run.sh` did not archive the checkpoint directory of a failed attempt | Fixed |
| M2 | docstring claimed a nonexistent test | Fixed: `tests/test_pilot_runs.py` added |
| L1 | LR rule lacked a device check | Fixed |
| L2, L3 | micro-batch 8×4 on 5090 for the LR check; noise band from 5090 bpb | Accepted as recorded |

## 5. Amendment 15.1 (2026-10-10 ≈ 11:53Z, before any pilot result)

The first 5090 pod was stopped by its controller while the operator was unavailable; two replacement 5090
pods landed on a host whose network stalled every bulk transfer (terminated, ledgered). Per the owner's
fallback, every stage assigned to "the 5090 pod" in §1 runs instead on **one homogeneous RTX 4090 pod**
(`hh2ru2zuuetw4x`, 6 GPUs), with identical settings (micro-batch 16 × 2 for the small model, 8 × 4 for the
architecture LR check). The rules' device-homogeneity checks apply to the 4090 set. The A100 assignments are
unchanged; the A100 is a new host (`qz3l9wpr1kgx3r`), so its host qualification (environment, CUDA/SDPA
checks, test suite) and a reduced real-data recovery re-check run before its timing runs.
