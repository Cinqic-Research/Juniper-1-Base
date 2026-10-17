# SFT v2.1 protocol v1: controlled mixture repair

The machine-readable protocol is `sft_v21_protocol_v1.json`. This page summarizes it, and that file governs.
It was registered at the commit that adds it, before any v2.1 training step (smoke included).

## Hypothesis

SFT v2 kept its large generalization gains: held-out paraphrases 0.020 → 0.385 and compositions
0.004 → 0.348. But it regressed on arithmetic (−0.15), injection (−0.15), hierarchy (−0.13) and JSON
(−0.08). Under the unchanged 1,226 × 64 budget, v1.3's 61,806 episodes were each drawn about 1.27 times.
v1.2's 39,234 episodes were each drawn exactly twice. The regressed skills therefore got only 44–59% of
v1.2's draws.

v2.1 gives those skills exactly v1.2's realized draws at the same budget. The test is whether that
restores them to non-inferiority and keeps most of the generalization gain. GSM8K did not regress at
51% exposure, so exposure may not be the whole story. The other open explanations are interference from
the new material and the reduced passes over new material.

## Design: one variable changed against SFT v2

| | SFT v2 | SFT v2.1 |
|---|---|---|
| init | frozen base, fresh optimizer | same |
| recipe | 1,226 updates × 64 episodes, LR 3e-5, cosine to 10%, AdamW, fp32 | same |
| data | j1-sft-v1.3, 1.27 epochs | **j1-sft-v1.3-mix1** (`db2ad50c`), one materialized epoch |
| hardware | RTX 4090 pods | FLOWBOX RTX 2060 (as the v1.2 baselines), $0 |

Realized draws in the mixture:

| Group | Draws | Basis |
|---|---|---|
| arithmetic | 13,228 | v1.2 |
| JSON | 7,994 | v1.2 |
| injection | 5,000 | v1.2 |
| hierarchy | 3,712 | v1.2 |
| Python | 10,894 | v1.2 |
| facts | 1,104 | v1.2 |
| GSM8K | 2,539 | v2 (passed) |
| uncertainty | 1,581 | v2 (passed) |
| new material | 32,412 | 0.76 passes, 60% of v2 |

New material is v1.3's instruction-following, formatting, grounded-QA and multi-turn data. Every row is
byte-identical to a v1.3 row, so nothing new is admitted.

## Sequence

1. **Smoke and recovery** on the mixture, before the diagnostic: bitwise A/B across SIGTERM and resume,
   with corrupted and torn latest checkpoints rejected and rolled back. This is `tools/sft_v21_smoke.sh`.
2. **Diagnostic seed 1.** `tools/sft_v21_pipeline.sh 1`, then `tools/sft_v21_decide.py --mode diagnostic`.
   It runs on the historical v1.3 test bundle, which is a predeclared diagnostic set and not untouched.
   The outcome is `ADVANCE`, `STOP_FAILED` or `STOP_INSUFFICIENT`. Any stop ends the protocol, with no
   tuning and no re-seeding.
3. **Only if ADVANCE: seeds 2 and 3**, then the final decision (`--mode final`). It covers the
   historical bundle across three seeds, plus a single evaluation of the sealed confirmation bundle
   `j1-sft-confirm-v1` (`054109e7`). The outcome is `QUALIFIED_FOR_REVIEW`, `FAILED` or
   `INSUFFICIENT_EVIDENCE`.

## Acceptance (fixed before any v2.1 output)

- **Generalization vs v1.2** (paraphrase, held-out composition):
  - gain ≥ 0.20, CI lower bound > 0.10, Holm p < 0.05, and every seed improves;
  - clean paraphrases and unseen-function compositions: CI lower bound > 0.
- **New capability vs v2:** CI lower bound of (v2.1 − v2) > −0.15 on both endpoints.
- **Protected skills vs v1.2.** Groups: in-distribution IF, JSON, arithmetic, GSM8K, Python, hierarchy,
  injection, uncertainty. Affirmative non-inferiority means PASS only if the CI lower bound > −margin.
  The margin is max(0.03, 2 × baseline seed SD). Underpowered results are INSUFFICIENT, never a pass,
  and each group is gated on its own.
- **Knowledge per relation:**
  - chat answers: the same affirmative rule (margin floor 0.05);
  - the candidate's own few-shot answers must not be significantly worse than the frozen base's in any seed.
- **Gates:**
  - protocol validity ≥ 0.95 and uncertainty ≥ 0.5;
  - false abstention ≤ 0.25;
  - full-split bpb ≤ 0.943872 (never raised);
  - checkpoint verified, smoke pass, and no dev early stop.
- **Scorer:** `j1.sfteval.v3` at Data `2f42da95`. Baseline and v2 generations are rescored into new files.

## Limits stated in advance

- The diagnostic is one seed. Item bootstrap ignores task, entity and phrasing clustering.
- The confirmation bundle has no facts, MBPP or HumanEval items and only 6 hierarchy items. Those groups
  are gated only on the historical bundle, which informed SFT v2.
- Conversational and App-instruction usefulness are not part of this stage. Directive section 10 covers them.
- A `QUALIFIED_FOR_REVIEW` result is a research candidate. It is not promoted.

## Amendment 1 (before any v2.1 output)

This amendment follows a separate Claude subagent review. It changes no gate and no threshold.

**Decision tool**
- Boundaries are compared exactly: a difference equal to −margin is INSUFFICIENT, as registered.
- The full checkpoint identity is bound across all evidence files.
- The SFT v2 comparators are pinned to their final checkpoints.
- The smoke schema is checked.

**Hypothesis and limits.** Two explanations are added.
- *Baseline exposure.* 467 historical test items share task keys with v1.2 training data. On those items
  v2's "regression" is much larger, so part of it is baseline memorization that no mixture can repair.
  This bias works against v2.1. Fresh-item rows are reported, not gated.
- *Repetition.* v2.1 makes more passes per unique episode than v1.2 did.

**Confirmation bundle.** It will be rebuilt as `j1-sft-confirm-v2`, excluding v1.2 train/dev task keys,
before any use. Gated confirmation groups need n ≥ 300.

## Amendment 2 (2026-10-14 2:50 PM EDT, before any v2.1 score or generation is inspected)

This amendment pins the confirmation bundle **`j1-sft-confirm-v3`** (`12350723…`; separate-agent review: APPROVE). It
excludes v1.2 train, dev and test tasks, and any prompt identical to a v1.2 or v1.3 evaluation prompt. Confirm-v1 and confirm-v2
were superseded before any use. The confirmation rule text now matches the tool: a gated group below `min_n` cannot pass. No
gate or threshold changes.
