# SFT v2.1 diagnostic (seed 1): `STOP_FAILED`

Decided 2026-10-14 2:51 PM EDT with `tools/sft_v21_decide.py --mode diagnostic` on protocol v1 plus amendments 1–2.
`decision.json` SHA-256: `22f8573b128e28ce7423ff7539687c8f620b32b8b32c6c5477d0c7ffe96e4ada`.

**Candidate.** `j1-sft-v21-s1/ckpt-000001226-4bf03e97`, trained on the FLOWBOX RTX 2060. The checkpoint
check is exact and dev loss fell to 0.6196 with no early stop. The run is preserved on FLOWBOX at
`~/j1post/runs-v21/v21/`, with per-item scores under `~/j1post/eval/v21/seed1/` (private).

As the protocol requires, the replication seeds (2 and 3) do not run, the confirmation bundle is not used, and
nothing is tuned or re-run under this protocol. No checkpoint is promoted.

## Result against v1.2 (same-seed baseline)

| group | v2 (historical, 3 seeds) | **v2.1 seed 1** | rule | status |
|---|---|---|---|---|
| paraphrase (primary) | +0.365 | **+0.310** [+0.263, +0.357] | ≥ 0.20, CI > 0.10 | pass |
| held-out composition (primary) | +0.343 | **+0.250** [+0.203, +0.300] | ≥ 0.20, CI > 0.10 | pass |
| clean paraphrase / unseen function | — | +0.264 / +0.270 | CI > 0 | pass |
| vs v2 on primary endpoints | — | −0.048 / −0.103 | CI > −0.15 | PASS |
| arithmetic | −0.151 | **+0.000** [−0.028, +0.030] | NI 0.03 | PASS |
| JSON | −0.083 | **−0.014** [−0.028, −0.002] | NI 0.03 | PASS |
| Python | −0.029 | −0.008 | NI 0.03 | PASS |
| GSM8K / uncertainty / in-dist IF | pass | +0.002 / −0.003 / +0.207 | NI 0.03 | PASS |
| injection | −0.149 | **−0.030** [−0.060, −0.003] | NI 0.03 | INSUFFICIENT |
| **hierarchy** | −0.126 | **−0.085** [−0.133, −0.038] | NI 0.03 | **FAIL** |
| **facts.element_symbol (chat, n=24)** | fail | **−0.125** [−0.292, +0.042] | margin 0.083 | **FAIL** (few-shot retention passes) |
| facts.country_continent (chat, n=42) | fail | −0.048 | margin 0.05 | INSUFFICIENT |

Gates: bpb 0.94016 (ceiling 0.943872), protocol-valid 0.961, uncertainty 0.997, false abstention 0.080. All pass.

Reported, not gated: fresh-item hierarchy −0.049 [−0.092, −0.006] (items the v1.2 baseline never trained on).
Formatting +0.97, grounded QA +0.77, multi-turn +0.37.

## Diagnosis

- **Supported.** Exposure explains most of v2's regressions. Restoring v1.2's realized draws at the same
  update budget recovered arithmetic (−0.15 → 0.00), JSON (−0.08 → −0.01) and most of injection (−0.15 → −0.03).
  Generalization stayed far above v1.2. This rests on one seed, so it is diagnostic evidence, not replication.
- **Not supported for hierarchy.** It still regresses at matched draws (2.14 passes over 1,733 v1.3 episodes,
  versus v1.2's 2 passes over 1,856), including on fresh items (−0.049). So hierarchy is not only an exposure
  problem. Candidate explanations, none tested:
  - v1.3's instruction-following, formatting and system-layer material may compete with system-priority
    behavior; formatting gained +0.97;
  - v1.3's hierarchy episodes may be less effective than v1.2's;
  - the extra repetition may play a part.
- **Element symbol (n=24).** The chat answers got worse, but the model's own few-shot answers are not
  significantly worse than the base's. That fits the elicitation pattern seen in v2. A single seed and 24
  items cannot separate signal from noise; the registered rule still counts it as a FAIL.
- **Injection.** INSUFFICIENT at one seed. The point estimate equals −margin.

## What would be next (not part of this protocol)

A successor stage, SFT v2.2, would need its own registered protocol before any run. It would be justified
by this evidence:
- keep the v2.1 mixture;
- add eligible, contamination-checked v1.2 hierarchy and injection replay episodes, distinct content
  that tests the content hypothesis;
- keep the unchanged budget, so one variable changes against v2.1;
- for knowledge relations, evaluate elicitation separately.

None of this has been run.
