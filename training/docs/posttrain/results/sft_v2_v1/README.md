# SFT v2 stage results (protocol `sft_v2_protocol_v1`)

**Outcome: `FAILED`** (decision tool `tools/sft_v2_decide.py` on the frozen protocol, run 2:42 AM EDT 2026-10-14;
[`decision.json`](decision.json)). All gates pass and both primary generalization endpoints pass by a wide
margin, but non-inferiority **regressions are established** on arithmetic, prompt injection, instruction
hierarchy and JSON, and chat answers on two knowledge relations are worse than the baseline. The candidate
(seed-1 final, `j1-sft-v2-s1/ckpt-000001226-48e927b5`) is not qualified; nothing is promoted.
`juniper1-base-v1` and the v1.2-recipe candidates are unchanged.

All times US Eastern. Baseline = the three replicated v1.2-recipe candidates (protocol v3 / replication v1)
scored on the same v1.3 bundle `935e12ed…` with scorer `j1.sftscore.v4` (Data `5162bf7`), paired by seed.

## Runs

| seed | run | final checkpoint | hardware | trained (EDT) | full-split bpb | dev loss reproduced |
|---|---|---|---|---|---|---|
| 1 | `j1-sft-v2-s1` | `ckpt-000001226-48e927b5` | RTX 4090 pod `lvmojshkyirv5z` | 10-14 1:45–1:57 AM | 0.94072 | exact |
| 2 | `j1-sft-v2-s2` | `ckpt-000001226-7b85e185` | same pod, concurrent process | 10-14 1:45–1:58 AM | 0.94055 | exact |
| 3 | `j1-sft-v2-s3` | `ckpt-000001226-33333871` | RTX 4090 pod `omzwhqkxidvjgm` | 10-13 4:12–4:23 PM | 0.94066 | exact |

1,226 updates each, no early stop, every checkpoint verifies. Software identical on both pods
([`pod_environment/`](pod_environment/): torch 2.8.0+cu128, numpy 2.1.2, tokenizers 0.23.2, cuDNN 91002).
Seeds 1 and 2 originally ran after seed 3 on pod `omzwhqkxidvjgm`; that pod was stopped by its watchdog
while the controller was down and could not be restarted (no free GPU on its host), so its seed 1/2
outputs were never read and the pod was terminated. Seeds 1 and 2 were re-run from scratch with the same
contracts on a new pod. No result of the first attempt was seen.

**Smoke and recovery** ([`smoke/`](smoke/)): 30-update run A vs run B (SIGTERM after update 16, restore,
resume): model and optimizer digests identical, 14 replayed batch fingerprints and losses identical;
the corrupted and the torn latest checkpoint were rejected at resume (`resume_scan`: "sha256 mismatch
model.pt", "size mismatch optim.pt") and the rollback runs end with the same digests as A. The smoke ran
on the seed-3 pod after seed 3 trained (deviation logged before any decision). Its result file was not
pulled before that pod became unreachable; `recovery_ab.json` and `rollback_*.json` were regenerated from
the preserved manifests and logs with `tools/sft_recovery_ab.py --manifest-only` (recorded digests, no
re-hash of weights, which were not preserved).

## Decision rows (seed means; paired item bootstrap 95% CI of v2 − baseline)

| endpoint | baseline | v2 | difference | CI | rule | result |
|---|---|---|---|---|---|---|
| **held-out paraphrases** (primary) | 0.020 | 0.385 | +0.365 | [0.321, 0.410] | ≥ +0.10, every seed, Holm p | **PASS** (Holm p 3e-39) |
| **held-out compositions** (primary) | 0.004 | 0.348 | +0.343 | [0.293, 0.393] | same | **PASS** (Holm p 6e-28) |
| in-distribution instruction following | 0.301 | 0.558 | +0.257 | [0.223, 0.291] | NI −0.03 | PASS |
| GSM8K | 0.021 | 0.021 | +0.001 | [−0.011, 0.013] | NI −0.03 | PASS |
| uncertainty | 1.000 | 0.991 | −0.009 | [−0.019, −0.001] | NI −0.03 | PASS |
| Python | 0.074 | 0.045 | −0.029 | [−0.040, −0.018] | NI −0.03 | INSUFFICIENT |
| JSON | 0.683 | 0.601 | −0.083 | [−0.107, −0.059] | NI −0.03 | **REGRESSION** |
| instruction hierarchy | 0.422 | 0.295 | −0.126 | [−0.172, −0.082] | NI −0.03 | **REGRESSION** |
| prompt injection | 0.957 | 0.808 | −0.149 | [−0.188, −0.111] | NI −0.03 | **REGRESSION** |
| generated arithmetic | 0.672 | 0.521 | −0.151 | [−0.182, −0.121] | NI −0.03 | **REGRESSION** |

All non-inferiority margins evaluated to the 0.03 floor (the baseline seeds agree closely).

Knowledge (chat answers vs baseline, margin max(0.05, 2 × SD); retention vs the base's few-shot answers):

| relation | baseline chat | v2 chat | CI | chat NI | v2 few-shot vs base few-shot | result |
|---|---|---|---|---|---|---|
| capitals | 0.020 | 0.046 | [−0.013, 0.078] | pass | 0.23/0.23/0.22 vs 0.18 | pass |
| continents | 0.889 | 0.825 | [−0.119, −0.016] | **fail** | 0.55 vs 0.50 | **fail** |
| currencies | 0.124 | 0.124 | [−0.029, 0.029] | pass | — | pass |
| element numbers | 0.014 | 0.000 | [−0.042, 0.000] | pass | — | pass |
| element symbols | 0.417 | 0.306 | [−0.194, −0.042] | **fail** | — | **fail** |

Few-shot (knowledge-retention) answers are never significantly worse than the base's: the knowledge is
retained; the chat-format answers on two relations got worse.

Gates (every seed): protocol-valid 0.967 / 0.967 / 0.961 (≥ 0.95); bpb ≤ 0.943872; uncertainty strict
0.987–0.993 (≥ 0.5); false abstention on facts 0.085 / 0.165 / 0.153 (≤ 0.25); recovery and integrity pass.

## Reported (not gated)

- Sensitivity classes (rebuild review N1/N2): the paraphrase gain holds on **clean** reserved phrasings
  (+0.307) and on phrasings one word from a sibling operation (+0.267), and is smaller where the same
  operation's training phrasing is one word away (+0.180); compositions that compute an **unseen
  function** improve +0.368 (functionally trained pairs +0.215). The gain is therefore not explained by
  near-copies.
- Unseen operations stay near zero: v1.2 held-out generator 0.011 → 0.028, v1.2 held-out operations
  0.026 → 0.066, new held-out operations 0.036 → 0.043. The v2 data teaches new phrasings and new
  combinations of trained operations, not new operations.
- Formatting 0.00 → 0.99, grounded QA 0.21 → 1.00, multi-turn 0.06 → 0.50, factuality −0.025.
- Open-ended conversation ([`open_ended/`](open_ended/); blinded independent review, `j1.open-ended-rubric.v1`,
  21 OASST1 test prompts; the protocol's 10 dev prompts do not exist because OASST1 was excluded from dev):
  candidate passes 2/21, baseline 0/21; mean total 9.81 vs 10.10 of 14; preference 7 candidate, 8 baseline,
  6 ties. Both models fail almost every open-ended request (bare "I don't know.", echoes, repetition, confident
  errors); v1.3 has no open-ended training targets, so no change was expected.
- Hardware control ([`hardware_control.json`](hardware_control.json)): the seed-1 candidate's 6,634 test chat
  outputs regenerated on the RTX 2060 (the baselines' GPU) have identical token IDs on 6,633 items and identical
  strict scores on all 6,634 (strict accuracy 0.4279 on both GPUs). The 4090/2060 difference does not affect
  the comparison.

## Interpretation (engineer, not part of the decision)

The v1.3 data does what it was built for: transfer to unseen phrasings, frames and compositions rises
from almost nothing to about 0.35–0.40, consistently across seeds, and grounded QA, formatting and
multi-turn behaviour are learned. The cost is a measurable loss on skills whose share of the training
mixture fell (generated arithmetic was reduced, injection/hierarchy/JSON were diluted by the new families)
under the same 1,226-update budget. A follow-up stage should restore those mixture shares (or add replay
of the v1.2 episodes for those categories) and keep everything else fixed; that needs a new registered
protocol.

## Preservation

All nine checkpoints (500/1000/1226 per seed): private HF `CinqicX/juniper1-checkpoints/checkpoints/
j1-sft-v2-s{1,2,3}/` with fresh-download verification (receipts on the archive HDD). Raw generations and
item-level scores stay on FLOWBOX and the archive HDD (they embed private test answers).
