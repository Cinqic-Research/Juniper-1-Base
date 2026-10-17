# Independent review of the SFT trainer (Juniper-1-Train `d49edef`), 2026-10-12

Reviewer: independent subagent (trainer/numerics/recovery scope), reproduction scripts kept on
FLOWBOX at `~/j1post/review/trainer/r01–r13` (CPU tiny models; r13 a tiny CUDA model).

**Held up under attack:** exact loss normalization under accumulation and right padding (float64);
no gradient at masked/pad positions; one episode per row; `check_alignment` rejects unshifted,
±1-rolled, packed and internally padded rows; the real `j1-sft-v1.0` loads and passes; bitwise CPU
resume across epoch boundaries, dev evaluations, SIGTERM and lost updates; tiny-model CUDA run
identical run-to-run and after resume; contract/foreign-checkpoint/torn-checkpoint refusals;
non-finite abort (no update, nothing saved); every `torch.load` uses `weights_only=True`.

| ID | sev. | finding | disposition (this branch) |
|---|---|---|---|
| F1 | HIGH | `torch.save` write failures raise `RuntimeError`, which bypassed retry/exit 3 | fixed: caught, staged copy removed; test |
| F2 | HIGH | eval tools could not load SFT checkpoints (no model config in manifest) | fixed: `model_config` in the manifest, loader reads it; test |
| F3 | HIGH | a prompt longer than the context aborted the whole generation run (bundle has one 3,731-token MBPP prompt) | fixed: recorded as `context_overflow` with empty output (identical for every model); test |
| F4 | MED | batch/dataset incompatibilities surfaced mid-run | fixed: checked at startup (episode length vs budget and context, episodes per update, `log_every`); tests |
| F5 | MED | fp16 GradScaler skips still advanced schedule/cursor | fp16 and bf16 refused by contract validation until qualified; test |
| F6 | MED | no runtime single-process guarantee | fixed: `WORLD_SIZE` check and exclusive run-directory lock; test |
| F7 | LOW | final dev eval missing from the final checkpoint when a cadence save hit the last update | fixed: last update always goes through the final block; test |
| F8 | LOW | time cadence checked after the dev eval (late save) | fixed: cadence checked first |
| F9 | LOW | AdamW `step` tensors on CUDA after resume | fixed: `optim.pt` loaded to CPU |
| F10 | LOW | failed saves left staged copies | fixed: staged copy removed on failure (the never-valid published directory is kept and refused) |
| F11 | LOW | accounting/log gaps | fixed: per-epoch skipped episodes logged, `batch_cells`/`pad_tokens` exposure, `--stop-after` exits 6; replayed updates after a resume remain visible in the log after a `restored` event (by design) |
| F12 | LOW | loader did not check ID uniqueness, train/dev disjointness, stop-token final target | fixed; tests |
| F13 | LOW | resume did not bind code identity | fixed: package digest must match for an exact resume; library/device differences are logged |

Reviewer verdict: core numerics correct; qualified for a bounded single-GPU fp32 pilot once F1–F4
and F6 are fixed and fp16 refused — all done above.
