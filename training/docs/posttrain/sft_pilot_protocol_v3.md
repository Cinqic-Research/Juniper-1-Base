# Juniper 1 initial SFT pilot: prospective protocol v3

Supersedes [v2](sft_pilot_protocol_v2.md) **before any measurement** (no base or candidate score on any
bundle computed, no SFT update run). A final independent review of the v1.1 dataset/bundle and of the
pilot tooling found scorer and MBPP-isolation defects (fixed in dataset/bundle `j1-sft-v1.2`) and three
pipeline/decision defects (run B of the recovery test was never actually signalled; the recovery gate
could pass without evidence; the registered early-stop rule was not enforced). Machine-readable twin:
[`sft_pilot_protocol_v3.json`](sft_pilot_protocol_v3.json).

## Changes from v2

| item | v2 | v3 |
|---|---|---|
| dataset | `j1-sft-v1.1` `7081dbde…` | `j1-sft-v1.2` `d3738c507580122c24cc948f9c40b79f7dd84b9a3cd31120d1fa283554dc6526` (built by Data `dcbe102`; scored with Data `f0399ed`) |
| evaluation bundle | `5119c5b4…` | `f96320150025d0250b5fbe7f5bc44b1f39b56ea836c8dfe842dbfc48c508eea1` |
| pilot updates | 1,226 | 1,226 (39,234 episodes, 613 per pass, two passes) |
| G-REC | stage 3 | stage 3 with full evidence (see below) **and** the pilot's final checkpoint verifies and reproduces its recorded dev loss exactly |
| early stop | stated | enforced in the decision: the pilot must complete its registered updates and dev loss must never rise at three consecutive dev evaluations |
| Holm family | categories plus overlapping subsets | the categories that have a base few-shot view (factuality, math, instruction following, JSON); the others have no regression gate |

Stage-3 recovery passes when run B (SIGTERM delivered to the trainer process around update 15, verified
checkpoint, resume) ends with model and optimizer digests equal to run A's and identical batch
fingerprints and losses for every replayed update. If kernels prevent bitwise equality, numerical
continuation is accepted only with a recorded stop, a restore, identical batch fingerprints for every
replayed update and equal update counts; the maximum parameter difference is reported.

Everything else — base, tokenizer, protocol identities, hardware and numerics, the learning-rate probe
rule, the pilot recipe and candidate definition, P1, G-PROTO, G-REG, G-RET, G-UNC, the decision values
and what the protocol cannot show — is unchanged from v2.

## Known limitations registered before scoring (re-review of the v1.2 deltas, 2026-10-12)

- Abstention specs for fictional element-symbol questions treat capitalised English words that are
  element symbols ("No", "As", "In", "At", …) as a named symbol, so some correct free-form
  abstentions fail. This can only lower `uncertainty_strict` (conservative for G-UNC).
- About 12 MBPP test tasks still have a functionally equivalent training twin (different code and
  wording). This affects only the reported Python category, which has no gate.
Both are to be fixed in the next dataset version, not in this pilot.
