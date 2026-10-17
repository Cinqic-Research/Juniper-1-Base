# Distillation v2 stage results (`distill_v2_stage.json`, Data protocol `gptoss_teacher_protocol_v2`)

**Outcome: `NO_IMPROVEMENT_DETECTED` for every teacher arm** (A1 answer-only, A2 concise verified rationale,
A3 verified equations), decision tool `tools/sft_distill_decide.py`, 5:43 AM EDT 2026-10-14
([`decision.json`](decision.json)). No arm qualifies for the replication seed. This is not evidence that
GPT-OSS-20B distillation cannot help; it shows that this stage design does not, for the reasons below.

All times US Eastern. One seed per arm (seed 1), all arms from the same init (the v1.2 pilot candidate,
seed 1: SFT v2 had FAILED), concurrent processes on one rented RTX 4090 (`rxveduih7p5lws`), scoring on
FLOWBOX with scorer `j1.sftscore.v4` on the v1.3 bundle `935e12ed…`.

## Teacher batch and arms

- GPT-OSS-20B (pinned GGUF `9d7364f0…`, llama.cpp b11270, Harmony template, low reasoning effort,
  temperature 0.6), 1,200 GSM8K train problems on the RTX 2060: 1,121 accepted, 79 rejected (records
  `abd405b1…`). First start failed (wrong expert-offload flag for the 20B model, no output); fixed.
- Verifier amended to v1.1 before any arm trained (layout normalization only; Data `bbce713`).
- **318 matched problems** (a verified target in every arm), replay 3 per arm episode (954 identical v1.3
  episodes in every arm); double build byte-identical; manifests pinned in the stage file before training.

## Results (v1.3 test split, strict, one seed per arm)

| | C0 (human solutions) | A1 | A2 | A3 |
|---|---|---|---|---|
| GSM8K test (n = 500) | 0.032 | 0.018 | 0.022 | 0.016 |
| gain vs C0 (Holm p) | — | −0.014 (0.46) | −0.010 (0.46) | −0.016 (0.35) |
| generated arithmetic | 0.512 | 0.452 (p 0.003) | 0.505 | 0.497 |
| full-split bpb (ceiling 0.943872) | 0.9699 | 0.9595 | 0.9678 | 0.9613 |
| Holm-significant category regression vs C0 | — | math | none | none |
| correct GSM8K answers whose working has a false equation | — | 0 / 9 | 5 / 11 | 4 / 8 |

Category strict accuracy against the initialization ([`category_strict.json`](category_strict.json)):

| category | init (pilot s1) | C0 | A1 | A2 | A3 |
|---|---|---|---|---|---|
| math | 0.374 | 0.294 | 0.255 | 0.285 | 0.278 |
| injection | 0.963 | 0.800 | 0.827 | 0.780 | 0.817 |
| hierarchy | 0.412 | 0.341 | 0.327 | 0.327 | 0.336 |
| json | 0.686 | 0.624 | 0.618 | 0.612 | 0.658 |
| grounded_qa | 0.233 | 0.873 | 0.913 | 0.863 | 0.883 |
| formatting | 0.000 | 0.490 | 0.560 | 0.513 | 0.537 |
| if_paraphrase | 0.020 | 0.102 | 0.095 | 0.083 | 0.098 |

## Why nothing improved (engineer's reading, not part of the decision)

1. **Retention fails in every arm, the control included.** 200 updates × 64 episodes over a 1,272-episode
   arm dataset is about 10 passes; dev loss rises at every evaluation (C0: 0.669 → 0.734 → 0.849 → 0.984
   at updates 50–200), bpb drifts from 0.9407 to 0.96–0.97. The registered update count assumed a much
   larger matched set; with 318 problems the stage overfits. The protocol has no early-stop rule; one
   would have stopped every arm at update 50.
2. **The student cannot use GSM8K-style supervision yet**: GSM8K stays at 0.02–0.03 in every arm (init
   0.014), and generated arithmetic falls in every arm (0.67 at init). The 143M-parameter base has not
   learned multi-step word problems; 318 examples cannot change that.
3. **Rationales make false equations more visible, not fewer**: A2/A3 answers that happen to be correct
   often carry a false equation in their working (A1 trains no working).
4. Replay of v1.3 episodes transfers the v1.3 behaviours (grounded QA, formatting) into every arm equally,
   which is why those rows rise for C0 as well.

A future distillation stage should (a) fix the update budget from the matched-set size (≤ 2 passes) or
register early stopping on dev loss, (b) target skills the student can express (short arithmetic, format
discipline) rather than GSM8K, and (c) keep the exact-McNemar matched design, which worked as intended.

## Evidence

Per arm: score summaries (item lists stay on FLOWBOX: they embed test answers), full-split retention,
checkpoint check (dev loss reproduced exactly), generation identity. Checkpoints (updates 100 and 200 per
arm): seven on private HF `CinqicX/juniper1-checkpoints/checkpoints/j1-distill-v2-*` with fresh-download
verification; the eighth (C0 final `ckpt-000000200-44558b82`) was refused by HF ("private storage limit
reached" for the org) and is preserved on the archive HDD and FLOWBOX, both verified (receipts on the HDD). Pod environment: identical software to SFT v2.
