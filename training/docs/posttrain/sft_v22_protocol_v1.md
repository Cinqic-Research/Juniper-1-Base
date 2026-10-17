# SFT v2.2 protocol v1: hierarchy and injection content (successor of v2.1)

The machine-readable protocol is `sft_v22_protocol_v1.json`; this page summarizes it, and that file governs. It was
registered at the commit that adds it, before any v2.2 training step. Everything not listed here is unchanged from
`sft_v21_protocol_v1` (amendments 1–2 included): gates, statistics, margins, smoke/recovery, the sealed
confirmation bundle `j1-sft-confirm-v3` (still unused) and the decision tool.

## Why

The v2.1 diagnostic (`results/sft_v21_v1/diagnostic/`) was **STOP_FAILED**:
- **Recovered:** restoring v1.2's realized draws brought back arithmetic, JSON and Python, and most of
  injection (−0.030, INSUFFICIENT).
- **Kept:** generalization stayed well above v1.2.
- **Not recovered:** hierarchy still regressed (−0.085; −0.049 on items the v1.2 baseline never saw), at
  matched draws over only 1,733 v1.3 hierarchy episodes (2.14 passes each).

## Framing (corrected by review before registration)

v2.1 already trained on almost as much unique hierarchy content as the passing v1.2 baseline: 1,733 episodes
at 2.14 passes, against 1,856 at 2.0. So a shortage of unique content was **not** shown to cause the regression;
interference from v1.3's new material is at least as likely. v2.2 tests a **compensation** hypothesis: does more
hierarchy and injection content, at the same draws, offset the deficit?

**Limits.**
- 1,376 of the 1,452 admitted hierarchy episodes embed a task that v1.3 already trains on, and the templates
  are unchanged. The novelty is new combinations, not new content.
- v2.2 trains on all of the baseline's injection set and 78% of its hierarchy set.

**Reported, never gating.** v2.2 minus v2.1 per group, on all items and on fresh items, next to the v1.2
baseline's seed-to-seed SD.

## One variable against v2.1

| | v2.1 (`mix1`) | v2.2 (`mix2`, `0ba1625f`) |
|---|---|---|
| hierarchy draws | 3,712 over 1,733 v1.3 episodes (2.14 passes) | 3,712 over 1,733 v1.3 + 1,452 admitted v1.2 episodes (1.17 passes) |
| injection draws | 5,000 over 2,000 v1.3 episodes (2.5 passes) | 5,000 over 2,000 v1.3 + 2,500 admitted v1.2 episodes (1.11 passes) |
| everything else | — | identical draws, recipe, seed, budget (1,226 × 64) |

**Replay admission.** v1.2 replay episodes come from `j1data.sftreplay`. A v1.2 episode is rejected if any of
these hold:
- it has any v1.3 dev/test, v1.2 test or confirmation task key;
- its user text appears in an evaluation or dev prompt;
- it contains any of v1.3's evaluation-only phrasings (35 patterns). Positive control: the patterns fire on
  400/400 paraphrase and held-out-operation test prompts. Negative control: 0 of 3,000 v1.3 training prompts.
- its content is identical to a v1.3 training episode;
- it is the same primary task as a v1.3 training episode.

## Sequence and limits

Smoke/recovery comes first, then diagnostic seed 1 with the same rule as v2.1. Only an `ADVANCE` allows seeds
2–3 and the single confirmation evaluation.

**Limits.**
- One seed.
- v1.2 hierarchy content may resemble the v1.2-baseline-exposed test items. The fresh-item rows report this.
- A second successive diagnostic on the same historical bundle increases reuse of that bundle. The
  confirmation bundle stays sealed for the final gate.

## Amendment 1 (before any v2.2 score exists)

From a separate-agent re-review. Reported prior rows now compare only the seeds that have a v2.1 run; final mode
would otherwise have crashed. The prior candidate is pinned by checkpoint identity. No gate changes.
