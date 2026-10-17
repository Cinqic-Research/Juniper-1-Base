# Juniper 1 independent audit, base freeze and post-training preparation — 2026-10-12

Owner instruction: *Independent End-to-End Audit, Base Checkpoint Selection, Freeze, and Post-Training
Preparation* (2026-10-12). Executors: GPT-6 Sol (17:34–17:49Z, stopped at its usage limit) and Claude Opus 5.5
(continuation, 17:51Z onward). Machine-readable findings: [`findings.json`](findings.json). Results:
[`results/`](results/).

## F. Executive summary

| Disposition | Value |
|---|---|
| Base checkpoint | **`BASE_FREEZE_APPROVED`** — `juniper1-base-v1` = `j1-base-r2/ckpt-000052034-336e5967` (original final checkpoint, unmodified) |
| Post-training preparation | **`POSTTRAIN_PREP_PARTIAL`** |
| Repository integration | **`INTEGRATION_COMPLETE`** for every qualified change (blocked PRs deliberately unmerged) |

- No parameter-changing operation of any kind was run. No checkpoint byte was modified; nothing was deleted.
- Assignment-attributable RunPod spend: **$0.00** (no resource created or started; see §E).
- PREP is PARTIAL because the Juniper App text path is blocked (A-02), no SFT dataset is admitted, and the
  post-training instruction/tool/injection evaluation suites are design-only. Integration is complete for the
  qualified work. AAA #34/#36, cinqic.com #22 and Train #2 stay open on purpose (security, unresolved science, or out
  of scope), each with a recorded explanation.

## A. Audit report

### Starting and ending refs

| Repo | Start (`main`) | End |
|---|---|---|
| Juniper-1-Train | `82b2d142` | `ff212c2e` (+ this receipt update) |
| Juniper-1-Data | `47cd1dd7` | `fee7003d` |
| Cinqic-Backup | `1e95afe6` | unchanged |
| AAA | `b8a0aab9` | `1aa0e385` (#37 only) |

Owner specification v0.7 bytes verified: SHA-256 `121fcb1d…92ee3` matches the recorded identity.

### Coverage ledger (what this review actually did)

| Area | Directly verified (executed / recomputed) | Relied on prior records |
|---|---|---|
| Architecture | config in all three top checkpoints vs spec; parameter count recomputed by hand (142,771,456); strict state-dict load; KV-cache vs uncached at every position; batch vs single; context limit; causality via incremental = full logits; Train suite at the final head 424 passed, 1 skipped (opt-in host-signal test), CUDA tests included on the RTX 2060 | line-by-line review of RoPE/RMSNorm/SwiGLU/optimizer groups (Fable/Opus/Sol audits) |
| Training evidence | raw log: updates 51–52,034 present, no gaps; 85 re-executed updates (8,501–8,585) with identical batch fingerprints; 0 skipped, 0 non-finite; 65,536 useful targets each; 52,034 × 65,536 = 3,410,100,224; early-bpb corrections (Train#23) recomputed | protocol 18b, LR grid, compile-equivalence negative result, B13/B17 fixes (not re-executed) |
| Corpus | document/stream token totals from the recount report; 2,959,371 documents = split sum; 7,108 exclusions (lineage report, hash-consistent); validation shard, doc index, shards.json and tokenizer SHA-256 re-hashed before scoring | source permissions, contamination layers (reports; not re-executed) |
| Checkpoints | `verify_checkpoint` + tensor-tree digests for 50,000 / 52,000 / 52,034; full-split re-scoring in fp16 and fp32; independent Drive and HF restore drills of 52,034 | earlier keepers (receipts only) |
| Preservation | 52,000 copied off-site with verification; receipts for 52,034 cross-checked against restores | Backup repository code (not re-audited in this review) |
| Inference | GGUF export of trained weights; token-ID parity F16 CPU+GPU (gated), Q8_0/Q4_K_M (reported); text-tokenizer parity on 200 validation documents; llama-bench and TTFT on FLOWBOX | — |
| Protocol / SFT | production-tokenizer serialization with forged delimiters; Data suite 336 passed at the PR head | App-side host enforcement (not executed) |
| Cross-repo | AAA#36 transport tests (3/3) at exact head; PR states of AAA#34, cinqic.com#22, Train#2 | AAA science (label-conditioned surrogate) not re-examined |

Not covered in this review: an exhaustive line-level re-audit of the full Train/Data/Backup trees (the time was
spent on freeze-critical evidence), Juniper-App runtime execution, Cinqic-Backup restore code paths, retired
repositories beyond identity statements. These are limitations, not passes.

### Findings and repairs

See [`findings.json`](findings.json) (A-01…A-13). The ones that matter for the next phase:

- **A-01 HIGH (fixed in prep tooling):** Data `Encoded.labels` are token-aligned but Train's `forward(idx, labels)`
  expects pre-shifted targets. A naive SFT integration would train copying. `j1data.sft.causal_rows` shifts within
  each episode (Juniper-1-Data#8).
- **A-02 HIGH (open blocker):** the experimental GGUF's llama.cpp text tokenizer reproduces j1tok on 0/200
  validation documents (divergence at token 0), and with `parse_special` it turns literal `<|system|>` / `<|eot|>`
  into control IDs. Only token-ID inference is qualified.
- A-04: the 52,000 checkpoint was HDD-only; now on Drive and HF with verified read-back.
- A-11: three exited RunPod pods carry plaintext Jupyter passwords in their environment (owner decision).
- A-13: AAA#34's bearer-credential leak remains at its head; AAA#36's repair is prospective and tested, but does not
  validate the research.

Independent review of this review's own code: a separate reviewer found no CRITICAL issue, six MEDIUM and two LOW
weaknesses (factuality denylist, string-license substring match, packing as default, optional tokenizer/record pin
in the loader, unchecked receipts in the freeze builder, weak sandbox for generated Python; tokenizer pin and
range checks). All were fixed with tests and re-verified by the same reviewer; its two follow-up LOW notes (a
same-model self-check alone admitting an episode; a missing sandbox silently scoring tasks as failures) were also
fixed; the earlier behavior run's 24 executed snippets were inspected and contained no imports or
system calls, and the published behavior results were regenerated with the hardened tool.

## B. Base checkpoint selection

[`base_checkpoint_selection_report.md`](base_checkpoint_selection_report.md). Full-split bpb (fp32): 50,000
0.917827 · 52,000 0.916198 · 52,034 0.916381. 52,000 leads by 0.020% relative, below the registered 0.1%
practical-significance threshold. No source regresses by more than 0.15%, numerics are identical, behavior is equal
or better at 52,034, and 52,034 has the stronger recovery evidence. **Selected: 52,034** (original final retained).

## C. Immutable base freeze

[`../freeze/juniper1-base-v1.json`](../freeze/juniper1-base-v1.json), generated by `tools/build_freeze_record.py`
from checkpoint bytes and these results. It binds the checkpoint name, manifest SHA-256 `a2f616c7…932e74`, `model.pt`
SHA-256 `71773e46…49c`, tensor-tree SHA-256 `1fd07c0d…076e4`, optimizer state, config, parameter count, tokenizer,
dataset build/manifest/shards identities, run/code identities, evaluation identities and results hashes, and
recovery receipts. It is self-hashed. `j1bench.frozen.load_frozen_base` returns weights only when the caller pins
the record digest and every identity matches. The original full-state checkpoint (model + AdamW + trainer/RNG/data
cursor) is preserved on the FLOWBOX HDD, Google Drive and private HF. `model.pt` is the inference-weights artifact;
no separate export is canonical. No weights are public.

## D. Post-training preparation package

| Item | State | Where |
|---|---|---|
| Stage design, dependency order, dataset contract, eval registration, checkpoint/recovery contract, context boundary, cost boundary | written (preparation only) | [`../posttrain/preparation_v1.md`](../posttrain/preparation_v1.md) |
| Gate ledger | updated with evidence | [`../posttrain/readiness_v1.json`](../posttrain/readiness_v1.json) |
| Frozen-base binding | implemented + tested | `j1bench/frozen.py`, `tests/test_frozen.py` |
| Episode admission (text-only) | implemented + tested, fail-closed | Juniper-1-Data `src/j1data/sft.py` (`encode_text_episode`) |
| Loss masking and next-token rows | implemented + tested on fixtures and the production tokenizer | `causal_rows`; `results/sft_production_tokenizer_check.txt` |
| Native protocol negative tests | present upstream (forged delimiters in user/tool data, malformed generations, budget/authority) plus new SFT-path tests | Data `tests/test_jtok_protocol.py`, `tests/test_sft.py` |
| Base regression suite (for every future stage) | registered and runnable | `base_evaluation_protocol_v1.md`, `tools/audit_base_candidates.py`, `tools/audit_base_behavior.py` |
| Tool-call / analysis-channel SFT, App adapter, injection/tool/instruction suites | **not ready** | blocked on A-02 and App host traces |
| SFT trainer and enforced save cadence (≤500 updates or 15 min) | **not implemented**; contract only | `preparation_v1.md` |

Resource scenario from measured rates (estimates, not forecasts): the pretraining profile ran at about 65,600 useful
tok/s on one A100 SXM at $1.79/h (+$0.028/h disk). A text SFT stage over 100M packed tokens would therefore need
about 25 minutes of compute (≈$0.8) plus evaluation, checkpoint and transfer overhead. Each full-state checkpoint
is 1.63 GiB (~2 min to HDD, ~13–17 min to HF/Drive). Owner decisions still required: SFT dataset sources and
rights, teacher use (GPT-OSS-20B or other) and its licence terms, the analysis-channel policy, the App text-path
remedy, and any spending authority. **No post-training update occurred.**

## E. Integration and preservation

| PR | Head reviewed | Disposition |
|---|---|---|
| Juniper-1-Train#23 (bpb corrections) | `a1f2dbec` | merged `9a3f6d18` (exact-head guard; CI SUCCESS) |
| Juniper-1-Train#22 (README) | `2ca81678` | merged `e39dc5f4` |
| Juniper-1-Train#21 (post-training gates) | `cb19a468` | merged `c48e1329` |
| Juniper-1-Data#8 (SFT preparation) | `ac36a902` | merged `fee7003d` (CI SUCCESS on head; post-merge push run SUCCESS) |
| Juniper-1-Train#24 (selection, freeze, audit records) | `4854f1e6` | merged `ff212c2e` (CI SUCCESS on head; dispatched post-merge run on `main` SUCCESS) |
| AAA#37 (README status, refreshed with the freeze ID) | `28f49047` | merged `1aa0e385` (all required checks SUCCESS on head) |
| AAA#34, AAA#36, cinqic.com#22 | unchanged | not merged (security + unresolved science; website depends on AAA) |
| Juniper-1-Train#2 (dual-A100 experiment) | unchanged | not merged (experimental, out of scope) |

Preservation (independently verified in this review): 52,034 is restored from HF (18:12Z) and Drive (18:14Z) into
clean directories, passes `verify_checkpoint`, and reproduces the digest. 52,000 is copied to Drive (`rclone check`
0 differences) and HF (commit `e1ce74b5`, fresh-download SHA-256 match). Receipts are in
`juniper1-archive/audit-2026-10-12/` and `results/`; the original run receipts are untouched. Remaining risks: the
FLOWBOX HDD is 96% full (22 GB free); the full accepted corpus text is still not independently preserved
(historical limitation, unchanged); Drive's OAuth app is in Testing (7-day refresh-token expiry).

RunPod (live read 2026-10-12T18:2xZ): 6 pods, all EXITED; no endpoints or clusters; since 15:00Z only standard
network storage (`72al92fys2`, ≈$0.0049/h) accrues. That is a pre-existing owner resource and was not changed. This
assignment created and started nothing: **$0.00 incremental** of the $10 cap. The account balance cannot be read
through the connector.
