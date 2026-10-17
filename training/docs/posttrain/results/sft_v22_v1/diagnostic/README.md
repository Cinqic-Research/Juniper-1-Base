# SFT v2.2 diagnostic (seed 1): `STOP_FAILED`

Decided 2026-10-14 4:43 PM EDT with `tools/sft_v21_decide.py --protocol docs/posttrain/sft_v22_protocol_v1.json
--mode diagnostic` (protocol v1 plus amendment 1). `decision.json` SHA-256:
`fbee53be23632c10cd6ac6ccd0c3a16c32eca1524fb27bec3d7cf3589767f763`. The test-access ledger entry was written
before the tool read any score.

**Candidate.**
- Checkpoint `j1-sft-v22-s1/ckpt-000001226-e01db0ab`, trained on the FLOWBOX RTX 2060 on `j1-sft-v1.3-mix2`
  (manifest `0ba1625f…`).
- The checkpoint check is exact. Dev loss reached 0.6197 (v2.1: 0.6196), with no early stop.
- Storage:
  - Final checkpoint and log: HDD `juniper1-archive/sft-v22-2026-10-14/`, verified against
    `SHA256SUMS-j1-sft-v22-s1.txt` and read-only.
  - Intermediate checkpoints: FLOWBOX `~/j1post/runs-v21/v22/`.
  - Per-item scores: `~/j1post/eval/v22/seed1/` (private).

As the protocol requires:
- the replication seeds do not run;
- the confirmation bundle is not used;
- nothing is tuned or re-run under this protocol;
- no checkpoint is promoted.

## Result against v1.2 (same-seed baseline), with v2.1 for reference

| group | v2.1 seed 1 | **v2.2 seed 1** | rule | status |
|---|---|---|---|---|
| paraphrase (primary) | +0.310 | **+0.302** [+0.255, +0.350] | ≥ 0.20, CI > 0.10 | pass |
| held-out composition (primary) | +0.250 | **+0.263** [+0.213, +0.313] | ≥ 0.20, CI > 0.10 | pass |
| vs v2 on primary endpoints | −0.048 / −0.103 | −0.055 / −0.090 | CI > −0.15 | PASS |
| **hierarchy** | −0.085 | **−0.052** [−0.100, −0.005] | NI 0.03 | **FAIL** |
| injection | −0.030 | −0.007 [−0.033, +0.020] | NI 0.03 | INSUFFICIENT |
| **JSON** | −0.014 | **−0.056** [−0.076, −0.036] | NI 0.03 | **FAIL** |
| arithmetic | +0.000 | −0.015 [−0.045, +0.015] | NI 0.03 | INSUFFICIENT |
| Python / GSM8K / uncertainty | −0.008 / +0.002 / −0.003 | −0.012 / +0.008 / −0.007 | NI 0.03 | PASS |
| in-distribution IF | +0.207 | +0.219 | NI 0.03 | PASS |
| facts.element_symbol (chat, n=24) | −0.125 FAIL | −0.083 [−0.250, +0.083] | margin 0.083 | INSUFFICIENT |
| facts.country_continent (chat, n=42) | −0.048 INSUFFICIENT | +0.000 | margin 0.05 | PASS |

**Gates.** All pass: G-PROTO, G-UNC, G-RET (bpb 0.93996, ceiling 0.943872), G-CKPT, G-REC and EARLY-STOP.

**Reported, not gated:**
- Fresh items, which the v1.2 baseline never trained on: hierarchy −0.012 [−0.055, +0.031]; JSON −0.056 [−0.078, −0.036].
- Against the v2.1 seed-1 prior run, at the same seed and budget:
  - hierarchy +0.033 [+0.000, +0.071];
  - injection +0.023 [−0.003, +0.050];
  - **JSON −0.042 [−0.060, −0.026]**;
  - arithmetic −0.015 [−0.032, +0.002].
- Yardstick, the v1.2 seed SD: JSON 0.003 and hierarchy 0.009. The JSON drop is about 14 times the v1.2
  seed-to-seed spread, so it is not seed noise.
- Formatting +0.98, grounded QA +0.77, multi-turn +0.35.

## Diagnosis

- **Replay helped hierarchy and injection, but not enough.**
  - Against v2.1, hierarchy improved by +0.033, and on fresh items it is now within noise of v1.2 (−0.012).
  - On the full historical set it still fails. The baseline-exposed items (v1.2 trained on 48 of the 211)
    explain part of the remaining gap.
  - Injection is now within noise (−0.007) but still INSUFFICIENT at one seed.
- **JSON regressed, at the same JSON draw count.**
  - The mixture changes only hierarchy and injection. JSON keeps the same 7,994 draws as v2.1, though the
    episode order differs.
  - JSON still fell −0.042 against v2.1, far beyond seed noise.
  - Most likely explanation, not tested: the new replay episodes come from the v1.2 generators and teach
    system-priority behavior that competes with strict output-only JSON formatting.
  - So adding replay content for one skill can cost another. The compensation hypothesis, which says
    hierarchy and injection can be restored without collateral loss, is **not supported**.
- **Knowledge.** element_symbol improved from FAIL to INSUFFICIENT, and country_continent now passes. Both
  rest on one seed with small n.

## What would be next (not part of this protocol)

Two successive mixture edits each fixed one skill and moved another. Further single-variable mixture tuning
on the historical bundle risks overfitting to it. A successor stage would need its own registered protocol
before any run. Options the evidence supports:
- accept v2.1's profile, with its hierarchy deficit, and qualify it against a narrower, documented claim;
- treat hierarchy and JSON as a joint objective, using a mixture that keeps v1.2's hierarchy and JSON
  episodes together;
- stop SFT mixture work until the conversational corpus (Data #13) exists, since the target product
  behavior is conversational.

None of these is selected here.
