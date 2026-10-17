# Owner continuation package — SUPERSEDED: the epoch completed on 2026-10-12 13:49Z (see launch_r2_report.md §7b)

Remaining owner decisions: post-training (T02) and release evaluation; retention of the volume copies; whether to terminate the two stopped pods (pncbyxaf2dt2jl cannot restart on its host; dxnsn6g7n5s4fw holds nothing not on the volume).

---

# Owner continuation package — j1-base-r2 after the limited tranche (2026-10-11)

Disposition target: READY_FOR_OWNER_CONTINUATION_DECISION. The order (§11) stops the executor before the
remaining epoch; the owner decides whether, when and under what funding the run continues.

## What exists (LIMITED_TRANCHE_COMPLETE, 12:56:33Z)
- 2,000 of 52,034 updates (131,072,000 valid targets, 3.84 % of the epoch); full-split evaluation 1.6487 bpb (500),
  1.3616 (1000), 1.2731 (1500), **1.2228 bpb (2000)**; 65.5k tok/s eager on one A100 SXM 80GB; no skipped or
  non-finite updates; guard never raised a reason. Details and hashes in `launch_r2_report.md`.
- Checkpoints `ckpt-000000000-ac36adac`, `-000000050-536f263d`, `-000000500-c4be7e7f`, `-000001000-4d4593eb`,
  `-000001500-da8a5cb5`, `-000002000-ca0a2f88` on the network volume `72al92fys2` (/vol/ckpt/j1-base-r2), on the
  FLOWBOX HDD archive (all verified) and off-site on Drive + the private HF repo (all six fully preserved at 14:19Z,
  Drive checked and HF read back; `preservation_status.json`).
- Resume point for a continuation: `ckpt-000002000-ca0a2f88` (params sha256 `4e363a7e…98f9f`, optimizer
  `7d66fd89…3031`), trainer step 2,000, same contract sha `eb4c9fb1…cbfcd`.
- Money: launch job closed at 1.46 (cap 3.40); ledger committed 6.64 of 13.70; provider-billed since 2026-10-08:
  52.91. Both pods are EXITED (A100 `pncbyxaf2dt2jl`, CPU `v247r25kcigu04`); the volume (~$0.034/day) and the
  owner's unrelated resources are untouched.
- Resume contract: the run resumes only from a verified checkpoint of `j1-base-r2` with the same code identity
  (`f0b4dea` j1bench tree), data identity, tokenizer, schedule (horizon 52,034, peak LR 0.0016971, warmup 500,
  cosine to 0.1) and seed; `launch_tranche.sh` refuses re-training from the canary once tranche checkpoints exist.

## Remaining epoch (projection, eager profile)
- 50,034 updates at ~1.00 s/update ≈ 13.9 h of A100 time (+ evaluations and checkpoints ≈ 0.3 h) at the
  observed $1.79/h (+ container disk) ≈ $26 of compute, plus the CPU watchdog pod (~$0.06/h) and off-site
  transfer time (1.63 GiB per checkpoint at 1.7–2.2 MB/s; ~17 min each on Drive).
- Compiled profile would roughly halve the time (131k vs 65k tok/s) but FAILED the registered equivalence
  tolerance (final bpb 0.411 % > 0.3 %); re-qualification would need a new registered comparison and the owner's
  tolerance decision.
- Funding: the order's $150 cumulative ceiling and the funding-contract envelope (v6: 13.70 admissible exposure)
  do not cover the remaining epoch; a continuation needs a new deposit decision by the owner (the executor
  cannot deposit) and a fresh versioned funding contract (v7) with the launch gate bound to a new readiness
  matrix.

## Decision items for the owner
1. Continue the epoch (full or in tranches of N updates), pause, or stop at 2,000.
2. Funding: amount and schedule; whether the remaining epoch is one session (~14 h, needs a stable pod) or
   several tranches with resume (each tranche re-runs readiness, authorization and the canary rule from the
   resumed checkpoint).
3. Preservation policy for the remaining checkpoints (every 500 updates = 104 checkpoints × 1.63 GiB ≈ 170 GB
   on Drive/HF; HDD has ~26 GB free): keep every k-th checkpoint off-site, or only evaluation-tagged ones.
4. Whether to re-attempt the compiled profile qualification before the long run.
