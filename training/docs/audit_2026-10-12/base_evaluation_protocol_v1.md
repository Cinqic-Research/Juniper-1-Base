# Juniper 1 base checkpoint comparison protocol v1

Registered 2026-10-12 before independent checkpoint inference. This protocol applies to the authentic `j1-base-r2` checkpoints only. It does not authorize training, weight edits, or public release.

## Eligibility and fixed identities

Compare update 50,000 (`ckpt-000050000-e4775a69`), update 52,000 (`ckpt-000052000-6b13028e`), and final update 52,034 (`ckpt-000052034-336e5967`) as the local top-end candidates. Inventory earlier 5,000-update keepers and the initial tranche through their off-site manifests. An eligible candidate needs a matching run contract, architecture, tokenizer, data identity, a valid `COMPLETE` marker, a self-hashed manifest, all payload hashes, a loadable state dictionary, and finite inference. Missing remote read-back is a preservation limitation, not permission to discard authentic weights. Do not treat a log-only evaluation as a checkpoint.

Model: `j1-wide-20x768`, 142,771,456 tied-embedding parameters, 24,576 tokens, 2,048-token context, config and weights from the checkpoint. Tokenizer: SHA-256 `176a89e2197c24d3f024fbb60ce6f346635e7cba912b3579d93b37d486c9adf1`. Validation: `j1-base-v0.2.2` `cand5-j1tok-24k`, `shards.json` SHA-256 `1cf5e2320ad35e930f84ca0edbf5a9d188a2f524702dc4623e492dab043e3aef`; validation token file SHA-256 `03efc04fc4c1bf1b1960f950cf98de229a9494dca786bb9472410bc9d44e13df` and document index SHA-256 `42e057d7771fc1223c9f59895edb5d4d2bb8cbfacdca5dabe6a91ec37482f787`. Verify local bytes before scoring.

## Ordered measurements

1. Verify every local candidate's full payload hashes and load `model.pt` with `torch.load(..., weights_only=True)`; require exact parameter names, shapes, and tensor-tree hash matching the manifest. Verify the tokenizer and validation identities.
2. Run a fixed smoke comparison on 32 validation windows, selected by integer `linspace(0, 1571, 32)` from the full-split window table. Keep 2,048 input positions, score every non-padding target, and use exactly the same windows and numerical mode for all candidates. This sample is a pipeline check and a noisy trend measure, not the primary selection statistic. Report per-source coverage and token counts. Record runtime, device, dtype, and any change from the training BF16 path.
3. If local resources permit, score the entire 1,572-window validation set for all eligible top-end candidates with the full-split context policy, no random sampling. Compute cross-entropy as total negative log likelihood divided by scored targets; perplexity is its exponential. Compute bits per byte as total negative log likelihood divided by `ln(2)` and 12,311,466 scored source bytes. Report per-source totals, not an unweighted average of source scores. Repeat any unexpected discrepancy before drawing a conclusion.
4. For each candidate, run the same fixed short base-continuation prompts, greedy decoding (temperature 0, maximum 64 new tokens, 2,048-token maximum context, stop at EOS). Prompts cover ordinary prose, a Python function prefix, simple arithmetic text, and JSON text continuation. Preserve generated tokens/text and failures. This is descriptive base-LM evidence, not chat or instruction-following evidence.
5. Check one-token KV-cache logits against uncached logits on fixed token sequences at lengths 1, 16, and 128; check a small batch against independent single rows, finite logits, EOS stop, and context limit. Record numerical tolerances appropriate to the actual precision. Do not use generated text to override a failed numerical check.

No held-out test split, private capability benchmark, or post-training data enters this selection. All candidate scores and failures remain in the report.

## Selection and freeze rule

Require valid load and numerical checks plus independent full-split comparison before declaring `BASE_FREEZE_APPROVED`. Use the lowest full-split bpb only when its advantage is at least 0.1% relative to the next candidate and no material source regression or objective behavior failure appears. Differences below 0.1% are a practical tie; within a tie, prefer the later authentic checkpoint if it has equal recoverability, because it completed the registered epoch. A clear material regression, incomplete replica proof, or evaluation inconsistency blocks automatic selection and requires an explicit recorded decision. A fixed 32-window scout, historical logs, or green CPU CI cannot by itself approve the freeze.

The full-split comparison reuses a validation set observed during training, so it checks checkpoint choice but is not an independent final capability estimate. No candidate may be averaged, merged, or modified.
