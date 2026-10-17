# Juniper 1 initial SFT pilot: prospective protocol v2

> **Superseded before any measurement by [v3](sft_pilot_protocol_v3.md)** (final pre-measurement review, 2026-10-12).

Supersedes [v1](sft_pilot_protocol_v1.md) **before any measurement**: no base or candidate score on the
evaluation bundle had been computed, and no SFT update had been run, when v2 was registered. Reason:
the independent dataset/evaluation review of 2026-10-12 (Juniper-1-Data
`docs/sft/reviews/data_review_2026-10-12.md`) found evaluation-integrity defects in dataset/bundle
v1.0 (cheatable Python scorer, hedged abstentions, train/test entity and MBPP overlap, a degenerate
base comparison). Machine-readable twin: [`sft_pilot_protocol_v2.json`](sft_pilot_protocol_v2.json).

## Changes from v1

| item | v1 | v2 |
|---|---|---|
| dataset | `j1-sft-v1.0` `0e51f1b7…` | `j1-sft-v1.1` `7081dbde86c1328112a16f7ed37efda3f26028d807cbd5f36d0e3f1e25334265` (Data `689204f`) |
| evaluation bundle | `ad3bb1fe…` | `5119c5b409a3df6bd07899805936eef51cc8895a3e46c97918a362f289a0b325` |
| pilot updates | 1,278 | 1,226 (two passes over 39,265 episodes at 64 per update) |
| primary endpoint | +0.20 over base chat (base ≈ 0 by construction) | absolute level **and** better than the base few-shot on the same items |
| regression gate | McNemar vs base chat, no correction | McNemar vs base few-shot, Holm-adjusted across categories |

Everything else (identities of base, tokenizer and protocol; hardware and numerics; the learning-rate
probe rule; the pilot recipe and candidate definition; early stops; the recovery qualification; the
retention gate; what the protocol cannot show) is unchanged from v1.

## Evaluation (test split, strict chat scoring, paired by item)

Two views are reported for every category: the base on the identical chat prompts (mostly format
learning, since the base never saw the protocol) and the base on plain-text 3-shot prompts of the
same items (the base's capability without the protocol; available for instruction following,
arithmetic/GSM8K, facts and JSON).

**Primary endpoint (P1):** in-distribution instruction following (500 items): candidate strict score
≥ 0.50 with Wilson 95% lower bound ≥ 0.40, **and** candidate − base few-shot (paired, same 500 items)
with bootstrap 95% CI lower bound > 0.

**Gates (all must pass):**

| gate | rule |
|---|---|
| G-PROTO | candidate protocol-valid rate on all test chat prompts ≥ 0.90 |
| G-REG | no category where the candidate is below the base few-shot with Holm-adjusted McNemar p < 0.05 |
| G-RET | candidate full-split validation bpb (fp32, registered layout) ≤ 0.943872 (1.03 × 0.916381) |
| G-UNC | candidate strict abstention on `uncertainty` ≥ 0.50 and false abstention on `factuality` ≤ 0.25 |
| G-REC | stage 3 recovery qualification passes |

**Decision:** `PILOT_CANDIDATE_QUALIFIED` if P1 and every gate pass, else `NOT_PROMOTED`. Not a
production model; `juniper1-base-v1` remains the canonical base and rollback target.

Reported, not gated: both views for every category and family, held-out instruction (272) and JSON
(200) families, Python (HumanEval 164, MBPP test 500, program output 200), GSM8K (500), lenient
scores, protocol validity per category, dev metrics, and generation determinism on a repeated subset.
