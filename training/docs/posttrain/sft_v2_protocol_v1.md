# SFT v2 stage protocol v1 (registered before any v2 result)

Machine-readable twin: [`sft_v2_protocol_v1.json`](sft_v2_protocol_v1.json) (the decision tool reads only
the JSON). Owner authority: directive of 2026-10-13, section 9. Stage authorization:
[`stage_auth/sft_v2_v1.json`](stage_auth/sft_v2_v1.json). All times US Eastern.

## 1. Question

Does instruction SFT on `j1-sft-v1.3` (instruction diversity, compositions, multi-turn, grounded QA,
filtered human conversation, protected knowledge) generalize better than the v1.2 pilot recipe while
retaining base capability, under the same optimization budget?

## 2. Design (fixed before any v2 run)

- **Initialization**: `juniper1-base-v1` (freeze record `ef79b324…`), fresh AdamW. No SFT candidate is
  continued.
- **Data**: `j1-sft-v1.3` (manifest and bundle digests in the JSON), scorer `j1.sftscore.v4` at the
  registered Data revision.
- **Optimization**: identical to the v1.2 pilot (protocol v3): 1,226 updates, 64 episodes per update,
  1,024-token micro-batch budget, LR 3e-5 (the v3 stage-1 selection; not re-searched), 40 warmup updates,
  cosine to 10%, AdamW (0.9, 0.95), eps 1e-8, weight decay 0.1 on matrices, clip 1.0, fp32, TF32 off,
  deterministic, single process, dev every 100 updates, checkpoints at ≤ 500 updates or ≤ 900 s. Equal
  update count and LR schedule keep the base-drift budget comparable to the pilot, so the comparison
  isolates the dataset (v1.3 covers 1.25 passes in 1,226 updates; v1.2 covered 2).
- **Hardware**: each seed trains and generates on its own rented RTX 4090 (amended before any run: the RTX 3090 went out of stock) (same image, torch 2.8.0+cu128, fp32,
  TF32 off, deterministic); scoring runs on FLOWBOX. The baseline candidates were trained on the RTX 2060;
  a hardware control regenerates the seed-1 candidate's test outputs on the RTX 2060 and reports agreement.
- **Seeds**: 1, 2, 3 (runs `j1-sft-v2-s{1,2,3}`); the **candidate is the seed-1 final checkpoint**;
  seeds 2 and 3 enter the acceptance rule. No other checkpoint or seed is selected after scoring.
- **Baseline**: the three replicated v1.2-recipe candidates (seeds 1–3, protocol v3 / replication v1),
  evaluated on the v1.3 bundle with the same scorer before this protocol was frozen; pairing is by seed
  number. The frozen base supplies the few-shot knowledge reference.

## 3. Qualification before the stage

1. Data admission: `j1-sft-v1.3` go/no-go record `APPROVED` (Data `docs/sft/reviews/`).
2. Smoke and recovery on v1.3 with the stage contract: a 30-update run A and a run B interrupted by
   SIGTERM after update 15 and resumed must agree bitwise (model, optimizer, batch fingerprints, losses);
   a torn and a corrupted checkpoint must be rejected and the run must roll back to the previous verified
   checkpoint.
3. Each seed's final checkpoint verifies and reproduces its recorded dev loss exactly.

## 4. Acceptance (registered 3:46 PM EDT, before the v1.3-bundle baselines and before any v2 run)

Thresholds that depend on the baseline are fixed **formulas** evaluated mechanically on the baseline results:
non-inferiority margin = max(0.03, 2 × the baseline seeds' sample SD on the group); chat knowledge margin =
max(0.05, 2 × that SD on the relation). The primary margin is a fixed practical goal: +0.10 absolute.

- **Primary (generalization)**, Holm over the endpoints: held-out paraphrases (`if_paraphrase`) and
  held-out compositions (`if_composition_heldout`). Pass: seed-mean improvement ≥ +0.10, paired item
  bootstrap 95% CI lower bound > 0, every seed better than its baseline seed, Holm-adjusted largest per-seed
  exact McNemar p < 0.05.
- **Non-inferiority** against the baseline seed mean (paired bootstrap CI lower bound > −margin):
  in-distribution instruction following (v1.2-style items), JSON, math (arithmetic, GSM8K), Python,
  hierarchy, injection, uncertainty. A CI entirely below −margin is a regression (FAILED); a CI that
  straddles it is INSUFFICIENT_EVIDENCE.
- **Knowledge per relation**: country and US-state capitals, continents, currencies, element symbols and
  numbers, each against the baseline (non-inferiority) and against the base's few-shot answers (Holm over
  relations: failing if every seed is significantly worse). A gain on one relation never offsets another.
- **Gates (every seed)**: protocol-valid ≥ 0.95; full-split fp32 bpb ≤ 0.943872 (unchanged historical
  ceiling); uncertainty strict ≥ 0.5 and false abstention on facts ≤ 0.25; recovery and checkpoint
  integrity; registered update count reached without the early-stop condition (dev loss rising at three
  consecutive dev evaluations).
- **Reported, not gated** (stated before measurement): the rebuild review's sensitivity classes
  (`sft_v2_sensitivity_v1.json`: paraphrase near-copies N1, functionally trained composition pairs N2);
  chat abstention on answerable facts; unseen operations (`heldout.ops.*`, v1.2
  `heldout.if`), multi-turn, formatting, grounded QA; open-ended conversation quality by a blinded
  independent rubric review (`j1.open-ended-rubric.v1`) of seed-1 versus baseline seed-1 outputs on the
  21 test and 10 dev OASST1 prompts (n is too small to gate).

Outcome: `QUALIFIED_FOR_REVIEW` (all gates, primary and non-inferiority pass), `FAILED` (a gate fails,
a regression is established or a knowledge gate fails), otherwise `INSUFFICIENT_EVIDENCE`. Failing to
detect a regression is not proof of retention. The engineer does not promote any checkpoint.

## 5. Rollback and preservation

`juniper1-base-v1` stays the canonical base and rollback target; every v2 run is a new lineage. All
runs, stops and rejected checkpoints are preserved with their evidence.
