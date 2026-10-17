# Juniper 1 proposed training contract (for owner freeze) — 2026-10-10

`training_contract_v1.json` (same directory) is the machine-readable contract. This page explains it.
**Status: PROPOSED.** It is prepared for the owner's freeze decision. It does not authorize the canonical
pretraining run; launch authorization is a separate owner decision. Every item marked OPEN or
OWNER_DECISION below blocks launch until closed with evidence.

## 1. Authority and identities

- Owner-issued *Juniper 1 Design Specification v0.7 — Four Destination Backup* (DOCX SHA-256 `121fcb1d…92ee3`),
  owner work order 2026-10-10 and the owner $40 acceleration addendum. Repository commits and trees, the corpus,
  index, tokenizer and environment identities are listed in the JSON (`identities`), and the cross-repository
  lock is `docs/review_2026-10-10/baseline_lock.json`.

## 2. Data and tokens

- Accepted corpus for training: `j1-base-v0.2.1`, the contamination successor of `j1-base-v0.2` (build `aa9c4ed9…`,
  manifest `1a99ef75…`): 555 train documents prohibited by the strengthened layers removed under a rule stated
  before the full-corpus result. Successor build identity `e7909e40…` (derived from its lineage), manifest
  `8d000906…`; train 2,961,286 / validation 2,583 / test 2,610 documents. Successor checks re-run (document sets,
  splits, independent count check, packing). Details: Juniper-1-Data `docs/qualification_2026-10-10.md`. The
  successor still needs its own gate records and owner approval.
- Counts are reported separately: unique accepted corpus tokens (frozen tokenizer), exposure tokens (a schedule
  over those tokens), and post-training supervised target tokens (none in base pretraining).
- 3B-initial / 15B-ceiling: the train split holds ≈ 3.44B document tokens under the proposed tokenizer
  (`j1tok-24k`, v5). That is the initial corpus; no collection toward 15B is proposed. **Proposed exposure:** one
  epoch of the successor train split (≈ 3.4B tokens), no repetition. Any second epoch or expansion needs the
  evidence and owner decision required by v0.7 §6.
- Stage allocation: base pretraining uses the full initial corpus; technical continuation, reasoning distillation,
  SFT and refinement are later stages with separately counted, explicitly allocated data (not part of this run).

## 3. Tokenizer and protocol

- Proposed: `j1tok-24k` (24,576 total IDs: 64 control, 256 byte fallback, learned), clean-sample candidate v5.
  Selected by the pre-registered rule tokenizer-v1.2 (all three candidates tied within 0.5 % bpb; tie broken by
  A100 training bytes/s). **Provisional:** short single-scale pilots; differences are within ≈ 1 seed SD.
- `juniper-chat-v1` grammar and loss masks are implemented in Juniper-1-Data but the App-side template/parser,
  parallel-tool-call policy, truncation semantics, schema validation and streaming parser are OPEN (review A1–A9).
  The protocol must be frozen before SFT, not before base pretraining.

## 4. Architecture and optimizer

- See `docs/prep/17_pilot_results_2026-10-10.md` for the controlled comparison (architecture-v1). The proposed
  shape and its exact instantiated parameter count are in the JSON.
- QK-Norm: learnable RMSNorm over the head dimension, one weight vector per layer shared by all query heads
  (and one for key heads), applied after projection and before RoPE; RoPE θ = 100,000; tied embeddings; no biases;
  pre-RMSNorm; SwiGLU; dropout 0; weight decay 0.1 on 2-D matrices only; AdamW β (0.9, 0.95), ε 1e-8; clip 1.0;
  BF16 autocast; context 2,048; EOS-separated packing with cross-document attention (recorded policy).
- Peak LR, warm-up and global batch for the canonical run are proposed from the pilots (JSON) and remain
  provisional until a longer run confirms them (08 §3.4).

## 5. Execution profile and cost

- Hardware: 1× A100 SXM 80GB (RunPod Secure, $1.79/h listed). Qualified on two hosts on 2026-10-10 (host and GPU records
  in `results/pilots-2026-10-10/*/env` and `requal/`): CUDA stack, SDPA/GQA forward/backward checks, real-data recovery suite (8/8 on host 1, T9 after the cb2a74d fix; exact continuation and
  SIGKILL bitwise on host 2).
- Reference profile: eager BF16 (56.2K tokens/s, core shape, 28K-vocab real-data prefix). Optimized candidate: `torch.compile`
  default (114.4K tokens/s, 2.04×; core shape on the 28K-vocab prefix; loss agreement ≤ 0.9 % relative over 50 updates, final 0.36 % at 160 updates).
  No promotion threshold is owner-approved, so the optimized profile stays **experimental**; the contract proposes
  it with the reference as the fallback.
- **Proposed canonical launch budget (separate from the $40 preparation authorization):** computed in the JSON
  from the measured A100 throughput of the proposed shape, the one-epoch exposure, and explicit overheads. The
  measured rate is from compiled runs, so the JSON also gives the eager-reference fallback budget (scaled by the
  measured eager/compiled ratio of the core shape on real data).

## 6. Checkpoints, preservation, logging

- Cadence: complete checkpoint at init, every 500 updates or 15 active minutes, stage ends and planned stops;
  retention per v0.7 §6 (latest three, hourly for 24 active hours, daily, stage, best, release lineage).
- Destinations: HDD canonical archive, Drive independent backup (Drive-only drill restored repository history,
  representative data plus the complete inventory, tokenizers, a full A100 checkpoint and evidence from Drive bytes
  only), GitHub source history (verified bundles of all 10 repositories), Hugging Face private data. OPEN: complete
  Drive coverage of every PRESERVE artifact, key custody, measured lag limits and the checkpoint primary/replica
  pair for the canonical run.
- Time logging (v0.6 §8) is implemented in the trainer and wrappers; its full qualification matrix is OPEN.

## 7. Evaluation

- Held-out validation split (2,583 docs, 12,348,996 bytes) for loss/bpb; test split unopened. Benchmark
  contamination coverage and its blind spots are recorded in Juniper-1-Data. Capability thresholds, CPU/FP16/Q8/Q4
  and 4K acceptance criteria are OWNER_DECISION items (none approved).

## 8. Owner decisions still required

1. Freeze tokenizer `j1tok-24k` (or choose a re-evaluation with more seeds).
2. Freeze the architecture shape (provisional per the comparison record).
3. Approve the successor accepted corpus `j1-base-v0.2.1` (`e7909e40…`), the one-epoch exposure and stage allocation.
4. Approve the optimized profile or the eager reference for the canonical run.
5. Approve the canonical launch budget and stop conditions.
6. Evaluation thresholds, release rights (weights/data redistribution), Drive key custody.
