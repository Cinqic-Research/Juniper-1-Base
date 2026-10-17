# Pilot protocol v2: prospective amendment before any real-data pilot result

- **Version:** `j1-pilot-protocol-v2`
- **Written:** 2026-10-10, before any real-data tokenizer or architecture pilot ran. No A100 pilot of
  `docs/prep/09_pilot_plan.md` was ever executed (RunPod use was suspended on 2026-10-08), so no
  result existed when these amendments were made.
- **Authority:** owner work order of 2026-10-10 (§9–§11; $20 RunPod cap, one A100 SXM 80GB at a time)
  under the owner-issued Juniper 1 Design Specification v0.7 (DOCX SHA-256
  `121fcb1de0dc927ce0d8d725c10bb68d199e28185f699d786472191418292ee3`).
- **Status:** supersedes `09_pilot_plan.md` for execution. `09_pilot_plan.md` and the v1.0/v1.1 rules
  in `08_evaluation_framework.md` §3 are preserved unchanged as history. The selection *thresholds*
  are unchanged; this amendment repairs measurement defects and fills the missing architecture rule.
- **Anchor:** this file, `tools/select_rule.py`, `tools/prep_pilots_v2.sh` and `tools/pod_controller.py`
  are frozen by the commit that adds them and are pushed to GitHub before the pod starts.

## 1. Defects found in the v1 plan (reproduced in code review, 2026-10-10)

| ID | Defect | Effect | Repair |
|---|---|---|---|
| PP-1 | `TokenStream.eval_windows(128)` scored the **first 128 token windows** of each candidate's validation stream. | Each tokenizer scored different documents and bytes (≈ 1 MB at 28K vs other amounts elsewhere), so bpb was not comparable. | `--eval-set full-split`: every document of the 2,583-document validation split is scored (`full_split_eval_v1`). Scored source bytes are identical by construction (12,348,996 bytes) and verified per candidate by `tools/verify_bpb.py`. |
| PP-2 | Training ran `steps = windows // 32` over a **shuffled** window order. | Different, random windows were dropped per candidate; "equal bytes" was asserted, not measured. | `--exact-epoch`: the prefix is truncated to whole updates *before* shuffling, every window is consumed exactly once (checked at the end of the run), and the summary records consumed targets, source bytes, dropped tail bytes and documents covered. The rule refuses comparisons whose consumed bytes differ by more than 0.1 %. |
| PP-3 | The bpb byte table assumed IDs `< 64` are controls and `< 320` are bytes. | Correct for the current candidates only by coincidence. | `token_byte_lengths()` derives the table from the tokenizer file (special added tokens = 0 bytes, `<0xNN>` = 1 byte, literal UTF-8 otherwise) and refuses normalizing/rewriting decoders. `verify_bpb.py` re-encodes every validation document from the accepted text and checks stored IDs, the decode and the byte table (all three candidates: 2,583/2,583 documents, 0 failures). |
| PP-4 | The tokenizer tie-breaker used eval bytes-per-token × tokens/s. | Not the bytes actually trained on. | Training bytes/s = consumed source bytes per update ÷ median steady-state update time. |
| PP-5 | `select_rule.py` had no architecture rule and accepted incomplete or mismatched runs. | Architecture decision was not reproducible; a crashed run could still be "selected". | `architecture-v1` implemented; all rules fail closed (`status: INVALID`) on missing, non-finite, incomplete, mismatched code/data/eval-set inputs. 16 rule tests, including an independent recomputation of the pooled seed SD. |
| PP-6 | The v1 cut order could drop the wide candidate and the seed repeats, yet still report a three-way comparison and a noise band. | A selection could be claimed without its required evidence. | §5: cuts are symmetric across candidates; required evidence is never dropped; if it cannot be afforded the outcome is INCONCLUSIVE. |
| PP-7 | The resume drill compared nothing; a zero exit counted as success. | No trajectory evidence. | §4 stage 7: real-data recovery suite with bitwise comparison (deterministic profile) and a pre-registered tolerance check on the optimized profile. |
| PP-8 | `tools/pod_guard.py` sends SIGTERM to the trainer and exits; it never stops the pod and estimates cost from first SSH reachability. | No unattended budget enforcement. | `tools/pod_controller.py` (on-pod, detached, provider uptime + monotonic accounting, authenticated provider stop). See §6. |

## 2. Fixed elements (all pilots)

| Setting | Value |
|---|---|
| Data | `j1-base-v0.2` accepted manifest `1a99ef75…7b37`; per-candidate 700 MB keyed-order train prefix (156,588 identical documents, 699,996,069 text bytes) and the full validation split (2,583 documents, 12,348,996 bytes). Test split unopened. |
| Packing / masks | EOS-separated documents, cross-document attention allowed, every position trained (EOS targets included), no padding in training. Same for every run. |
| Sequence / batch | 2,048 tokens; 32 windows per update (65,536 tokens); micro-batch 16 × accumulation 2 unless OOM forces a symmetric change for all candidates of one comparison. |
| Precision / profile | BF16 autocast, `torch.compile` default, fused AdamW: the experimental optimized profile. Not a canonical profile decision. |
| Optimizer | AdamW β 0.9/0.95, ε 1e-8, weight decay 0.1 on matrices only, clip 1.0; 100-update warmup then cosine to 10 % at the run's final update. |
| Evaluation | `full_split_eval_v1` on the validation split; bpb = summed NLL / (ln 2 × scored source bytes). 8 evaluations per tokenizer run, 12 per architecture run, plus the final update. |
| Checkpoints | 500 updates or 15 active minutes, best-eval and final saves, on pod disk. Pilot checkpoints are investigation artifacts (RECOMPUTABLE-TEMPORARY from the recorded contract) except the core final checkpoint, which is retained for the recovery drill. |
| Timing boundaries | `train_seconds` = sum of synchronized optimizer-update times; the equal-compute comparison excludes the first (compile) update and all evaluation/checkpoint time. Paid end-to-end time (pod start to stop) is reported separately and never substituted. |

## 3. Selection rules (thresholds unchanged)

- **lr-v1.1** — v1 rule (lowest final validation loss; exclude non-finite or > 1.5× running-minimum
  spikes after warmup) plus fail-closed validity checks.
- **tokenizer-v1.2** — v1.1 rule unchanged: hard requirements (zero round-trip and control-ID
  failures; zero byte fallback on `python, json_tools, structured, paths_urls, math, science, english,
  identifiers_rare`; overall validation fallback ≤ 0.5 %); lowest validation bpb; candidates within
  0.5 % relative are tied; ties go to the highest training bytes/s, then to the smaller vocabulary.
  v1.2 measurement: bpb is the **mean over seeds 1 and 2** (pre-registered), all from the full split,
  with exact-epoch consumption. If an untied runner-up lies within 2 × pooled relative seed SD of the
  best, `uncertainty_could_change_result` is set and seed 3 is run for every candidate (symmetric) when
  the budget gate allows; the rule is then re-applied with seeds 1–3. Without it the decision stands as
  provisional.
- **architecture-v1** — keep `j1-core-28x640` unless an alternative improves **both** equal-token bpb
  (final validation bpb after the identical exact epoch) **and** equal-compute bpb (validation bpb at
  `T* = min over candidates of training seconds`, linear interpolation between evaluations, no
  extrapolation) by more than the noise band. Improvement is relative: `(core − alt) / core`.
  Noise band = max(0.5 %, 2 × pooled relative seed SD of the six small-model tokenizer-pilot runs,
  3 degrees of freedom). The small-model SD is a proxy, not proof of large-model uncertainty.
  - One seed clears the band on both → seed 2 for core and that alternative (required before any
    switch); the rule is re-applied on seed means.
  - An alternative clears only one comparison → INCONCLUSIVE; core stays provisional.
  - Nothing clears → core retained (provisional: short runs cannot prove final-scale superiority).
- **LR for architecture shapes (bounded, equal allowance)**: transferred LR = LR_small /
  √(d_model/384). Each shape gets the same three-point check {0.5×, 1×, 2×} on the first 15 % of the
  selected tokenizer's prefix (exact epoch, own short schedule), selected per shape by lr-v1.1. The full
  run uses the selected multiplier. The final-size divergence rule of 08 §3.4 still applies.

## 4. Stages, order and estimated cost (single A100 SXM 80GB, declared $1.79/h compute)

Throughput basis: earlier single-A100 synthetic measurement for core ≈ 102K tokens/s (compiled, micro-batch 16);
small-model throughput is unmeasured and assumed ≈ 450K tokens/s. Estimates include ≈ 1 min compile per run.

| # | Stage | Runs | Est. min | Required? |
|---|---|---|---:|---|
| 0 | Pod/environment qualification (GPU identity, MIG, versions, mounts, rate), data fetch from HF with SHA-256 verification, GPU test suite incl. eager-vs-compiled numerics | — | 20 | yes |
| 1 | LR sweep, small, 32K tokenizer, {6e-4, 1.2e-3, 2.4e-3}, 60M-token exact epoch | 3 | 9 | yes |
| 2 | Tokenizer pilots, small, 24K/28K/32K × seeds 1, 2, full 700 MB prefix exact epoch | 6 | 45 | yes |
| 3 | Tokenizer selection (tokenizer-v1.2); optional seed 3 ×3 if flagged | (3) | (23) | conditional |
| 4 | Architecture LR check: core/deep/wide × {0.5, 1, 2} on 15 % prefix | 9 | 45 | yes, symmetric |
| 5 | Architecture full runs: core/deep/wide, seed 1, selected tokenizer, full prefix | 3 | 95 | yes |
| 6 | Architecture selection; seed 2 for core + winner if NEEDS_SEED_REPEATS | (2) | (65) | conditional |
| 7 | Real-data recovery suite on core (`tools/recovery_suite.py`, deterministic profile, bitwise) and 20-update optimized-profile resume comparison | ~14 | 20 | yes |
| | **Total (required)** | | **≈ 235 min ≈ $7.0** | |

## 5. Budget and cut order (symmetric)

- Task cap $20.00; reserve ≥ $3.00; ordinary work stops at an estimated **$17.00**; the controller's hard stop is at $19.00.
- Before each stage `tools/budget_gate.py` requires: current estimate + 1.3 × stage estimate ≤ $17.00.
- If a required stage does not fit: (1) drop optional conditional stages; (2) shorten stage 4 to {1×}
  for all three shapes (symmetric). No reduced-token variant of stage 5 is pre-registered: if stage 5
  does not fit at the full prefix for all three shapes, the architecture outcome is INCONCLUSIVE.
  Never drop a candidate or the stage-2 seed 2. If stage 2 cannot complete, the tokenizer outcome is
  INCONCLUSIVE. Each stage gate also requires ≥ 60 GiB free on the checkpoint filesystem.

## 6. Unattended controller

`tools/pod_controller.py` runs detached on the pod from the first minute. It polls every 30 s:
provider uptime and rate (GraphQL, pod-scoped key), monotonic elapsed time, GPU utilisation and result
heartbeats. Soft stop at $17 (stop file → no new stage; SIGTERM → trainers checkpoint; grace for the
FLOWBOX pull; authenticated `podStop`); hard stop at $19; idle/stall stop after 15 minutes; deadline stop.
It never terminates or deletes. Operator-side checks via the provider API and a scheduled backup stop
task are independent additional layers.

## 7. What the results can and cannot show

Same as `09_pilot_plan.md`: short runs (~1.2 tokens/parameter for the 148M class) rank candidates under
these conditions; they cannot prove converged quality, downstream capability, or final-scale superiority.
All runs, including failures and negative results, are kept.

## 8. Independent review before launch (2026-10-10, fresh reviewer, heads d026e0d → 10f1385)

Verdict on d026e0d: REJECT for launch until must-fix items were closed. All dispositions below were
made before any paid run.

| ID | Finding | Disposition |
|---|---|---|
| T1 | Controller tests used the real /proc scan and SIGTERMed live trainers on the host | Fixed (10f1385): injectable PID source + regression test |
| T2 | (coordinator, same class) legacy `test_pod_guard` ran the guard's TERM command locally and killed FLOWBOX rehearsal runs; also the cause of its intermittent failure | Fixed: test is opt-in (`J1_ALLOW_HOST_SIGNAL_TESTS=1`) |
| C1 | An exception in a controller iteration (ledger fsync, stop-file write, rglob race) killed the controller | Fixed: ledger writes non-fatal; iteration errors caught; three consecutive errors stop the pod |
| C2 | A zero-exit stop call ended enforcement even if the pod kept running | Fixed: post-stop polling re-issues the stop after three non-EXITED polls |
| R1 | A diverged LR run (no summary) voided the whole LR rule | Fixed: a run that ended on the trainer's non-finite abort is excluded; any other missing summary stays INVALID |
| S1 | Globs silently dropped a crashed candidate | Fixed: explicit run paths; architecture rule takes `--required` and refuses an incomplete set |
| S2 | Re-runs appended to old metrics; stale summaries could be accepted | Fixed: completed runs are reused, failed attempts archived under `attempts/`; rules require status exit 0, matching run id, strictly increasing eval steps |
| S3 | Unpruned checkpoints could fill the disk | Fixed: `--ckpt-prune pilot` (latest, newest best, final; invalid kept) and a free-space gate |
| S4 | A failed seed-3 re-selection overwrote the 2-seed decision | Fixed: separate file; 2-seed decision stands as provisional |
| S5 | Cut-order step (3) not implemented | Amended §5 before any result (no reduced-token stage 5) |
| A1 | Equal-compute time wrong for resumed runs | Fixed: the architecture rule refuses resumed runs |
| C3 | Stopped-volume storage outside the controller | Accepted: the reused pod's volume is owner-existing and pre-existing; its storage is reported separately, not created by this task |
| R2–R5 | Uncaught TypeError; dof ≥ 3; LR identity checks; probe control IDs | Fixed |
| D1 | `documents_fully_consumed` off by one | Not changed: targets are stream positions 1…n−1, so a document's EOS at `off+ln` is consumed iff `off+ln ≤ n−1`; metadata only |
| P1 | Eval precision not recorded | Fixed: `logits_precision` in the eval-set identity |
| — | (coordinator) checkpoint manifests labelled real-data runs `qualification-synthetic` with a synthetic eval identity | Fixed: stage id, real eval identity and data description recorded |
| N1 | (re-review) a soft-stopped run (exit 0, final_step < steps) would be "reused" and block the stage | Fixed: reuse requires final_step == steps; otherwise the attempt is archived and re-run |
| N2 | (re-review) archived attempts left their checkpoint directory, confusing pruning | Fixed: the checkpoint run directory moves into `attempts/` too |
| N3 | (re-review) the 60 GiB free-space gate can fail closed on a small volume | Accepted: stage 0 records free space; `MIN_FREE_GB` is set from the measured volume before stage 1 |

Re-review of fe665da: all earlier fixes VERIFIED (D1 dispute accepted); verdict APPROVE_WITH_FIXES (N1).
