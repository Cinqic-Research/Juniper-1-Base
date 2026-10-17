# Juniper 1 initial SFT pilot: prospective protocol v1

> **Superseded before any measurement by [v2](sft_pilot_protocol_v2.md)** (independent evaluation review, 2026-10-12).

Registered before any base measurement on the evaluation bundle and before any SFT update of
`juniper1-base-v1`. Machine-readable twin: [`sft_pilot_protocol_v1.json`](sft_pilot_protocol_v1.json).
Changing anything below after a candidate result is seen requires a new protocol version.

## Identities

| item | identity |
|---|---|
| base | `juniper1-base-v1`, freeze record `ef79b324…92ff55` (`j1bench.frozen.load_frozen_base`) |
| dataset | `j1-sft-v1.0`, manifest `0e51f1b7100f8cc1327397b2a1b80e52938b30291077573b5c08e80ce46266d5` (Juniper-1-Data `eaabc58`) |
| evaluation bundle | `ad3bb1fe22e196eb05e8659b4161e13672df4657183332e63a6a695adff28e20` (prompts token IDs; answers read only by `j1data.sfteval`) |
| tokenizer / protocol | j1tok-24k `176a89e2…`, `juniper-chat-v1` |
| scorer | `j1data.sfteval` / `j1data.sftscore` at Juniper-1-Data `eaabc58` |
| decoding | greedy, KV cache, bundle forbidden IDs masked, stop at `<|eot|>`/`<|call|>`, per-item token budget from the bundle |
| hardware | FLOWBOX RTX 2060 (6 GB), fp32, TF32 off, deterministic algorithms; no paid compute |

## Stage 1: learning-rate probes (diagnostic runs, never candidates)

Three runs from the frozen base, identical except the peak LR: 1e-5, 3e-5, 1e-4. Each: 150 updates,
64 episodes per update, 1,024-token micro-batches, AdamW (0.9, 0.95, eps 1e-8, weight decay 0.1 on
matrices only), clip 1.0, 15 warmup updates then constant LR, seed 1. Measured at update 150: dev loss
(all 1,656 dev episodes) and fixed-32-window validation bpb (`tools/sft_retention.py --mode fixed32`).

**Rule:** choose the LR with the lowest dev loss among probes whose fixed-32 bpb is at most 1.03 × the
base's fixed-32 bpb (0.867697 → 0.893728). If none qualifies, choose the smallest LR. Ties (dev loss
within 0.5%) go to the smaller LR.

## Stage 2: pilot run (the single candidate)

From the frozen base with the chosen LR: same optimizer and batch, 1,278 updates (two passes over the
40,902 train episodes at 639 updates per pass), 40 warmup updates, cosine decay to 10% of peak,
seed 1, dev loss every 100 updates, full-state checkpoints at ≤ 500 updates / ≤ 900 s and at the end.
**The candidate is the final checkpoint.** No intermediate checkpoint is selected on any test metric.

Stop early (candidate = NOT_PROMOTED, preserved as negative evidence) if: a non-finite loss or gradient
occurs; dev loss rises at three consecutive dev evaluations; or a checkpoint cannot be published.

## Stage 3: recovery qualification on the real model (separately named runs)

Run A: 30 updates uninterrupted. Run B: identical contract, SIGTERM after update 15 (checkpoint), resume
to 30. Pass if B's final model and optimizer tensor-tree digests equal A's (bitwise) and the batch
fingerprints for updates 16–30 match. If kernels prevent bitwise equality, report the maximum absolute
parameter difference and call it numerical continuation, not exact recovery. The pilot's own final
checkpoint must pass `verify_checkpoint`, restore with matching state digests, and reproduce its
recorded dev loss when re-evaluated.

## Evaluation (test split, chat variant, strict scoring)

Strict = the generation is one well-formed `<|final|> … <|eot|>` turn and its text passes the hidden
spec. Comparisons are paired by item against the frozen base on identical prompts and decoding.
Statistics: Wilson 95% intervals per rate; paired bootstrap 95% CI of the difference (10,000
resamples, seed 20261012); exact two-sided McNemar test.

**Primary endpoint (P1):** instruction following, in-distribution families (`if.*`; 500 items; the 272 held-out `heldout.if.*` items are reported separately):
candidate − base ≥ +0.20 absolute and bootstrap CI lower bound > 0.

**Gates (all must pass):**

| gate | rule |
|---|---|
| G-PROTO | candidate protocol-valid rate on all test chat prompts ≥ 0.90 |
| G-REG | no category with candidate < base and McNemar p < 0.05 |
| G-RET | candidate full-split validation bpb (fp32, registered layout) ≤ 1.03 × base 0.916381 = 0.943872 |
| G-UNC | abstention on `uncertainty` test ≥ 0.50 and false abstention on `factuality` test ≤ 0.25 |
| G-REC | stage 3 passes (bitwise, or the pilot's own checkpoint restore/re-evaluation if A/B is only numerical) |

**Decision:** `PILOT_CANDIDATE_QUALIFIED` if P1 and every gate pass; otherwise `NOT_PROMOTED`. A
qualified pilot is not a production model and does not replace `juniper1-base-v1`; it is eligible as
the starting point of a separately registered next stage.

**Reported, not gated** (with intervals): every category and family; held-out instruction and JSON
families (transfer); Python (HumanEval, MBPP test, program output) where near-zero baselines make
absolute differences the only meaningful measure; GSM8K; lenient scores; protocol-valid rate per
category; base few-shot plain-text baselines (to separate format learning from capability); dev
metrics; run-to-run determinism of generation (a repeated subset must match exactly).

## What this protocol cannot show

The pilot uses one seed and one recipe; seed-to-seed variance is not measured. Test families share
templates with training families (except the held-out ones), so in-distribution gains mostly measure
format and task learning, not general instruction following. Runtime security (host permissions,
tool authorization) is measured separately on the host adapter, not by model scores.
