# Juniper 1 base checkpoint selection report (independent review, 2026-10-12)

**Decision: `BASE_FREEZE_APPROVED` — `juniper1-base-v1` = `j1-base-r2/ckpt-000052034-336e5967` (update 52,034,
the original final checkpoint, retained unmodified).** Freeze record: [`docs/freeze/juniper1-base-v1.json`](../freeze/juniper1-base-v1.json).

Reviewers: GPT-6 Sol (registered the protocol, began scoring; stopped after ~8 minutes at its usage limit) and
Claude Opus 5.5 (completed scoring, numerics, behavior, restore drills and the decision). Every number below was
measured in this review from bytes on FLOWBOX unless marked *logged* (the training run's own record).

## Candidate inventory

| Checkpoint | Update | Weights recoverable | Replicas (verified) | Logged val bpb | Disposition |
|---|---:|---|---|---:|---|
| `ckpt-000052034-336e5967` (final) | 52,034 | yes | HDD; Drive + HF (training receipts **and** independent restore drills, this review) | 0.91646 | **selected** |
| `ckpt-000052000-6b13028e` | 52,000 | yes | HDD; Drive + HF added by this review (`rclone check` 0 differences; HF fresh-download SHA-256) | 0.91628 | eligible, not selected (tie) |
| `ckpt-000050000-e4775a69` | 50,000 | yes | HDD; Drive + HF (training receipts) | 0.91791 | eligible, worse by 0.16% |
| 50,500 / 51,000 / 51,500 | — | **no** (pruned under the launch retention policy) | none | 0.91757 / 0.91705 / 0.91694 | not candidates: log-only evaluations |
| 5,000 … 45,000 keepers, tranche 0–2,000 | ≤45,000 | yes | Drive + HF (training receipts; HDD copies released) | ≥0.92760 | inventoried; not re-scored (≥1.2% worse) |

All three scored candidates pass `verify_checkpoint` (self-hashed manifest, `COMPLETE` marker, payload SHA-256),
load with `torch.load(weights_only=True)` + `strict=True`, reproduce their manifest tensor-tree digests, carry the
same run contract, code identity, dataset identity and tokenizer SHA-256, and instantiate 142,771,456 parameters
(independently recomputed: 24,576×768 embedding + 20 × 6,194,816 per block + 768 final norm).

## Protocol

[`base_evaluation_protocol_v1.md`](base_evaluation_protocol_v1.md), registered by Sol in commit `305ea96` before any
candidate inference. Disclosure: it was registered *after* the training run's logged validation values were known
(52,000 logged slightly lower than 52,034). Its 0.1% practical-tie threshold and later-checkpoint tie-break were
therefore not blind. The decision below would be the same under the alternative tie-break the brief prescribes
(stronger continuity, completeness and preservation evidence), so the disclosure does not change the outcome.

## Held-out language modeling (full validation split, 1,572 windows, 3,218,204 targets, 12,311,466 bytes)

| Candidate | bpb fp16-autocast | bpb fp32 | CE (nats, fp16) | PPL (fp16) | logged (A100 bf16) |
|---|---:|---:|---:|---:|---:|
| 50,000 | 0.917829 | 0.917827 | 2.43379 | 11.4020 | 0.917911 |
| 52,000 | **0.916199** | **0.916198** | 2.42947 | 11.3529 | 0.916280 |
| 52,034 | 0.916384 | 0.916381 | 2.42996 | 11.3584 | 0.916455 |

RTX 2060 (FLOWBOX), torch 2.8.0+cu128, `expand` GQA kernel, TF32 off; inputs verified by SHA-256 before scoring.
fp16 and fp32 agree to ≤2×10⁻⁶ bpb and both reproduce the logged A100 values to ≤1×10⁻⁴, so the ordering is not a
precision artifact. 52,000 is lower than 52,034 by 0.000185 bpb = **0.020% relative**, five times smaller than the
registered 0.1% practical-significance threshold → **practical tie**. Per-source differences (52,034 vs 52,000)
are mixed in sign and within ±0.15%: arxiv +0.06, finemath +0.12, fineweb-edu +0.01, libretexts −0.01,
oercommons +0.04, pressbooks +0.04, python-peps −0.15 (1,984 tokens), stackexchange +0.03, stackv2-edu −0.07,
wikimedia −0.02 (%). No source regresses materially. The validation split was seen as an evaluation set during
training (not as training data), so these numbers check checkpoint choice; they are not unbiased capability
estimates. The test split was not touched.

## Numerical robustness (CPU fp32, all three candidates)

KV-cache incremental decoding vs uncached forward at every position for lengths 1/16/128: max |Δlogit| ≤ 2.1×10⁻⁴
(logit scale ≈ 15–22), prefill-last ≤ 3×10⁻⁵; batch-of-4 vs single rows: exactly 0; finite logits at 2,048 tokens;
2,049-token forward and decode past 2,048 both rejected. Equality of incremental (past-only) and full-forward
logits also shows the full forward is causal. Tokenizer: exact round-trip on Unicode/emoji/ZWJ/whitespace fixtures;
a literal `<|user|>` encodes to plain IDs only when `encode_special_tokens` is set (`j1data.jtok.load`); loading
`tokenizer.json` directly maps it to control ID 3 (known hazard, handled by the Data loader and the SFT adapter).
CPU load 1.7–3.4 s; CPU greedy decode with the PyTorch reference ≈ 25 tok/s (not an optimized runtime).

## Base completion behavior and objective diagnostics (greedy, descriptive)

| Probe (n) | 50,000 | 52,000 | 52,034 |
|---|---:|---:|---:|
| Likelihood multiple choice, simple facts (20) | 16 | 17 | 17 (identical picks to 52,000) |
| Executable Python with hidden asserts (8) | 3 | 1 | 3 |
| Python bodies that parse (8) | 6 | 6 | 6 |
| Few-shot two-digit addition, exact match (20) | 1 | 1 | 1 |
| JSON prefix completes to valid JSON (6) | 1 | 1 | 1 |
| distinct-4-gram, 256-token greedy prose / code | 0.26 / 0.41 | 0.06 / 0.21 | 0.07 / 0.25 |

The diagnostics were added after registration and are labeled as such; they are tiny and are not selection
criteria beyond "no objective behavior failure". All three are typical small base LMs: fluent, topical English and
idiomatic Python, frequent factual confabulation (e.g. "the Dutchman Johannes Gutenberg"), near-zero arithmetic,
and strong greedy repetition on long continuations (expected for greedy decoding of a base model). Full prompts,
token IDs and texts: [`results/behavior_cpu.json`](results/behavior_cpu.json). None of this is instruction-following
evidence.

## Decision

1. Identity and provenance: all three valid; 52,034 is the registered run's terminal state (end of the LR
   schedule, exactly one epoch: 52,034 × 65,536 = 3,410,100,224 targets).
2. Recoverability: 52,034 has the strongest evidence — training-time Drive and HF receipts plus two independent
   restore drills in this review (Drive and HF downloads into clean directories → `verify_checkpoint` OK, tensor-tree
   digest `1fd07c0d…076e4` reproduced). 52,000 was HDD-only until this review.
3. Numerics: equal.
4. Held-out quality: 52,000 better by 0.020% → practical tie under the registered rule.
5. Objective behavior: no regression of 52,034 against 52,000 (equal or better on every probe).
6. No late-stage instability: logged validation decreases monotonically from 50,000 to 52,000 (0.91791 →
   0.91628) and moves +0.02% over the final 34 updates; measured values agree.

Tie-break (registered rule and brief §9): the later authentic checkpoint with the stronger continuity and
preservation evidence → **52,034**. No weights were averaged, merged, modified or re-saved. 52,000 and 50,000 remain
preserved, unmodified, as evidence.
