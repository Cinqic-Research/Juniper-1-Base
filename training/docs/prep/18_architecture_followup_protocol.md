# Protocol 18: focused core-versus-wide follow-up (prospective successor protocol)

- **Version:** `j1-arch-followup-v1` (rule `architecture-v2`, LR rule `lr-v1.1` unchanged).
- **Written:** 2026-10-10 ≈ 17:00Z, **after** the 2026-10-10 pilot (`17_pilot_results_2026-10-10.md`) was observed and
  **before** any run of this protocol. The pilot is exploratory context for this design (it motivated the grid,
  the seed count and the margins); it is not unseen confirmation evidence and none of its runs counts below.
  Architecture-v1 and its INCONCLUSIVE record stay unchanged.
- **Authority:** owner work order "Juniper 1 — autonomous architecture resolution, qualification, and limited
  pretraining" (2026-10-10) §5, under the owner-issued Juniper 1 Design Specification v0.7 (DOCX SHA-256
  `121fcb1de0dc927ce0d8d725c10bb68d199e28185f699d786472191418292ee3`). Engineer-selected thresholds under that
  order's delegation.
- **Budget:** `docs/autonomous_2026-10-10/budget_contract.json` v2 (comparison allocation $15.97, soft $15.20,
  inside B_task $23.37).
- **Anchor:** this file, `tools/select_rule.py` (`architecture-v2`, `ec-budget`), `tools/followup_v3.sh` and their
  tests are frozen by the commit that adds them, pushed before the pod starts.
- **Immutability (review F1):** from the start of the first screening run, the rule, margins, t values, critical-domain
  list, seeds, budgets, EC semantics, decision order and cut order are **immutable**. The only permitted amendment is
  the mechanical 18.1 record produced by the runner (`amendment_18_1.json`: the two frozen LRs from `lr-v1.1`, the
  `ec-budget` result `r`/`U_fast`/faster shape, boundary flags and SHA-256 of every screening record), committed and
  pushed before any confirmation run. Any other change after the first screening run makes the outcome
  **exploratory**, not a selection under this protocol.

## 1. Question

Which of `j1-core-28x640` (143,072,384 parameters) and `j1-wide-20x768` (142,771,456 parameters) should be the
Juniper 1 base shape, judged separately on **equal-token quality** and on **equal-compute quality / paid
efficiency**, with equal tuning opportunity and at least three fresh matched seeds each. `j1-deep-30x576` is not a
finalist: it cleared neither comparison in the pilot and the order names core and wide.

## 2. Fixed inputs and settings (identical for every run)

| Item | Value |
|---|---|
| Corpus | accepted `j1-base-v0.2.1` (build identity `e7909e40…765a`, manifest `8d000906…7968`); prohibited documents of the successor already removed. No quarantined candidate or v0.2-only prefix is used. |
| Tokenizer | `j1tok-24k` frozen bytes, SHA-256 `176a89e2…adf1`, 24,576 IDs (64 controls); loaded with `encode_special_tokens` on (see `jtok.load`). |
| Model-ready data | `cand5-j1tok-24k` shards of v0.2.1 (`shards.json` `cb2e2b53…6ef3`); **training prefix** = the first ≈ 960 MB of source text of the v0.2.1 train stream in its keyed pack order (`tokshards.export_prefix`, recorded with its own `shards.json`), large enough for every budget below. The training prefix is part of the train split; validation is the full v0.2.1 validation split (2,583 documents); **test is never opened**. |
| Packing / loss | EOS-separated documents, cross-document attention allowed, every position a target (EOS included), no padding, no masked targets in training. Valid training targets = updates × 65,536. |
| Sequence / batch | 2,048 tokens; 32 windows per update (65,536 targets); micro-batch 16 × accumulation 2. |
| Optimizer | AdamW β 0.9/0.95, ε 1e-8, weight decay 0.1 on matrices only (`--wd-policy matrices`), clip 1.0, 100-update linear warmup then cosine to 10 % at the run's **own** final update. |
| Precision / kernels | BF16 autocast, `torch.compile` default mode, fused AdamW, SDPA default backend (flash on A100). Eval: autocast BF16 logits, float32 cross-entropy, float64 sums. |
| Data order | `--exact-epoch` over `--limit-tokens = updates × 65,536 + 1` tokens of the training prefix; window order is a permutation seeded by `(seed, epoch)`. The same seed gives **the same ordered windows** to core and wide (paired). |
| Evaluation | `full_split_eval_v1` on the whole validation split; bpb = summed NLL / (ln 2 × 12,348,996 scored source bytes, verified per run), overall and per source. Screening: 4 evaluations, the last at the final update; confirmation and equal-compute: 12 evaluations, the last at the final update. |
| Hardware | one full A100 SXM 80GB (secure cloud), one GPU process at a time, same host for the whole comparison. GPU identity, HBM, MIG state, driver, CUDA, image digest, torch/triton versions, kernel choices, host allocation, region and rate are recorded at bootstrap. A host change invalidates the timing comparison (§6). |

## 3. Learning-rate search (tuning; seed 11 for every screening run)

- **Budget:** 1,396 updates (91,488,256 targets; half the confirmation budget), own warmup/cosine schedule.
- **Grid:** multipliers {0.25, 0.5, 1, 2} of each shape's previous selected LR, which keeps the previous values as
  anchors, extends two octaves below core's previous lower boundary and explores both sides of wide's best:
  - core: 2.3237e-4, 4.6475e-4, **9.2950e-4**, 1.8590e-3 (extension points 1.1619e-4 / 3.7180e-3)
  - wide: 4.2427e-4, 8.4855e-4, **1.6971e-3**, 3.3942e-3 (extension points 2.1214e-4 / 6.7884e-3)
  (values as generated by the runner: `f"{base*mult:.4e}"`, five significant digits)
- **Selection:** `lr-v1.1` per shape (lowest final **validation** loss; non-finite or > 1.5× running-minimum spikes
  after warmup are excluded and kept as evidence). Training loss is never the criterion.
- **Boundary rule (bounded):** if a shape's selected LR is the lowest or highest grid point, one extra point one
  octave beyond that edge (0.125× or 4×) runs for **that** shape; the rule is re-applied to the five points. At most
  one extension per shape. If the optimum is still at the edge, the LR is used and reported as an **unresolved
  boundary**. No other refinement.
- **Order (interleaved to spread host drift):** core×1, wide×1, core×0.5, wide×0.5, core×2, wide×2, core×0.25,
  wide×0.25, then any extensions (core first).
- The selected LR and the schedule of each shape are **frozen** (recorded in `select_lr_<shape>.json`) before any
  confirmation run starts.

## 4. Confirmation (fresh seeds 21, 22, 23; none of these seeds was used before)

- **Equal tokens (ET):** both shapes, 2,792 updates (182,976,512 targets; the pilot's budget), same ordered windows
  per seed, own warmup/cosine schedule fully decayed at update 2,792.
- **Equal compute (EC):** the matched budget is **T\* = the slower shape's ET training time**. The slower shape's
  ET runs are therefore also its EC runs (schedule fully decayed at T\*). The faster shape gets **dedicated EC runs**
  with the same seeds and its own schedule fully decayed at `U_fast = round(2,792 × r)` updates, where
  `r` = ratio of the median updates per second (training seconds excluding the first update ÷ (updates − 1)) of
  the faster shape's screening runs to the slower shape's (same host and GPU, same session; review F8 replaced the
  median step-rate basis by this time basis, which includes stalls). `r` and `U_fast` are written by
  `select_rule.py ec-budget` into amendment 18.1 and committed before any confirmation run. If `r` is outside
  [0.80, 1.25] the protocol stops (re-design needed). EC runs read a larger window set than ET runs, so EC pairing
  with the slower shape is by seed (not identical windows).
  This avoids comparing a fully decayed faster model with a mid-schedule slower one; no curve interpolation or
  extrapolation enters the EC decision.
- **Time-match check:** the EC comparison is valid only if each EC run's training seconds (synchronized
  optimizer-update time excluding the first update, evaluation and checkpoint time) lies within ±2 % of the
  median ET training seconds of the slower shape (review F8: ±5 % was about half of M at these horizons). Otherwise the EC comparison is reported as unmatched and the
  rule treats EC as unavailable.
- **Order:** C-s21, W-s21, F-s21, F-s22, W-s22, C-s22, C-s23, W-s23, F-s23 (F = faster shape's EC run).
- **Throughput:** each confirmation run reports steady-state median tokens/s (three timed repetitions per shape
  from the three seeds), first-update (compile) seconds, step-time distribution, peak allocated/reserved memory
  and stalls; paid end-to-end efficiency = valid targets ÷ the run's process start-to-end seconds from the bench
  status (data load and verification, compile, training, evaluations and checkpoint saves; pod boot and idle time
  are session overheads reported separately); FLOP estimate = 6 × non-embedding parameters × targets + attention term
  (reported separately from GPU-seconds and dollars).

## 5. Rule `architecture-v2` (registered before any result of this protocol)

Notation: for seed *s*, `q_X(s)` = final full-split validation bpb of candidate X; improvement of X over Y on seed
*s* = `(q_Y(s) − q_X(s)) / q_Y(s)` (positive = X better). Per comparison (ET, EC) the three paired improvements give
mean *m*, sample SD *sd*, and one-sided 90 % bounds `m ± 1.8856 × sd / √3` (t, 2 dof).

Margins (engineering basis: the tokenizer tie band and architecture-v1 band floor were 0.5 %; differences below
0.5 % bpb after ≈ 1.3 tokens/parameter do not reliably predict the ordering at the final horizon; the pilot's
small-model pooled per-run seed SD was 0.29 %; for independent runs the SD of a paired difference is ≈ √2 × 0.29 ≈
0.41 %, so three paired seeds give a one-sided 90 % half-width of ≈ 1.8856 × 0.41 / √3 ≈ 0.45 % (review F10). At that
noise, rule 3 (equivalence) is nearly unreachable and noninferiority needs a mean ET difference ≥ ≈ −0.05 %; the
135M-scale noise may differ, which is exactly what the three seeds measure):

- `NI = 0.5 %` noninferiority/equivalence margin; `M = 0.5 %` material-improvement threshold.
- Critical domains = every validation source with ≥ 700,000 scored bytes: `arxiv-papers` (710,269),
  `finemath-4plus`, `fineweb-edu-dedup`, `pressbooks`, `stackexchange`, `stackv2-edu`, `wikimedia` (byte counts
  from the v0.2 validation split; re-derived from the run records). `DOM(X)`: on every critical domain the mean
  paired ET improvement of X over Y is ≥ −1.5 %. Smaller sources (libretexts 118,548, oercommons 39,444,
  python-peps 8,465 bytes) are reported but do not gate (too few bytes for a stable per-domain estimate).

Predicates for candidate X against the other candidate Y:

- `NI_ET(X)`: lower 90 % bound of X's ET improvement ≥ −NI.
- `SUP_ET(X)`: ET mean improvement ≥ M **and** lower bound > 0.
- `SUP_EC(X)`: EC available, EC mean improvement ≥ M **and** lower bound > 0.

Decision (fail-closed validity first, otherwise `INVALID`: all 6 ET runs and all 3 EC runs complete with metrics
JSONL, finite overall and per-domain bpb, unresumed, final evaluation at the final update, registered evaluation set
(kind, 2,583 documents, 12,348,996 bytes), stage ids `fu3-confirm-et`/`fu3-confirm-ec`, identical code, GPU UUID and
pod (host record beside each run), identical fixed settings (sequence, batch, warmup, schedule floor, decay policy,
clip, optimizer, precision, compile, SDPA, data and tokenizer paths), the registered seed set, LRs equal to the
frozen 18.1 LRs, EC update count and faster shape equal to 18.1, and the critical-domain set equal to the registered
list):

1. If some X has `SUP_ET(X) ∧ DOM(X)` → **SELECTED X** (quality first).
2. Else if exactly one X has `NI_ET(X) ∧ DOM(X) ∧ SUP_EC(X)` → **SELECTED X** (noninferior on equal tokens,
   materially better at equal compute).
3. Else if both `NI_ET(core)` and `NI_ET(wide)` hold (TOST with α = 0.10 on each side, i.e. the two-sided **80 %**
   interval of the ET difference lies inside ±0.5 %; review F4) and both `DOM` hold, and one candidate's paid
   end-to-end valid targets per second exceeds the other's by **≥ 3 %** (above the expected host drift; review F9) →
   **ENGINEERING_CHOICE_UNDER_UNCERTAINTY** selecting that candidate (registered acceptable-risk criterion: practically
   equivalent quality, materially lower cost; the uncertainty is reported with the choice). Below 3 % → INCONCLUSIVE.
4. Otherwise → **INCONCLUSIVE**; nothing is frozen on this evidence. A faster shape never wins by speed alone
   when its quality gates fail. Core has no default entitlement in v2.

More seeds: seeds 24 and 25 (both shapes, ET and EC) are pre-registered as the only extension, allowed once and only
if the budget gate admits all of them; the rule is then re-applied with n = 5 (t 1.5332, 4 dof). No other optional
stopping. The extension is not expected to fit this budget.

Limits stated in advance: three seeds cannot guarantee a decisive interval; short runs (≈ 1.3 targets per parameter)
rank shapes under these conditions only; LR optima at 91M-target screening may differ from the final horizon;
base-LM loss does not measure agent capability.

## 6. Validity and fault handling

- Fail-closed inputs as in protocol 14 (status, run id, complete updates, strictly increasing evaluations, matched
  code/data/eval identities, no resume for timing). A non-finite run is excluded with evidence (LR stage) or makes
  the comparison INVALID (confirmation).
- One bounded retry per run (the failed attempt is archived under `attempts/`); a screening run that ends on the
  trainer's non-finite abort is a result (excluded from its LR grid with evidence), not a failure. A required run
  that fails twice stops the runner (exit 6). Completed runs are reused only when their recorded arguments equal the
  intended arguments (review F11).
- Host change mid-comparison: confirmation timing restarts on the new host after its qualification, with all
  confirmation runs repeated there (the cut order below then applies). Losses from the first host are reported.
- Cut order if the comparison allocation cannot fund the remainder (symmetric): (1) drop the seed-24/25 extension;
  (2) drop the LR boundary extensions for both shapes; (3) drop the end-of-session throughput drift probes. Required
  ET/EC seeds are never dropped; if they cannot all complete, the outcome is INCONCLUSIVE. Enforcement (review F3):
  before any boundary extension the runner gates (extensions + all nine confirmation runs) together and skips the
  extensions for **both** shapes if that does not fit; before confirmation it gates all nine runs together (aggregate
  factor 1.1 on per-run paid minutes measured in the pilot: screening 15, ET 29, EC 30; single runs keep factor 1.3).
- Drift control: a 300-update throughput probe of each shape at session start (bootstrap) and end; > 3 % drift is
  reported and, if > 5 %, the EC time-match check is re-evaluated against probe-adjusted rates (report only; no
  decision change).

## 7. Independent check

Before the pod starts, a reviewer that did not write this protocol checks: fairness of tuning, data pairing,
EC schedule semantics, the rule's predicates and code against this text, failure rejection tests, and that no
pilot-v2 run is counted. Findings and dispositions are recorded in §8 of this file before any run.

## 8. Review record (before any run of this protocol)

Independent reviewer (fresh agent, did not write the protocol) on commit `61ac20a`: **APPROVE_WITH_FIXES**. Confirmed:
budgets/schedules/paired data order produced by the runner flags (`data.py` window count and `(seed, epoch)`
permutation; `train.py` exact-epoch checks and per-run cosine horizon); predicates, decision order, margins and t
values match the text; tuning symmetric; seeds 11 and 21-23 unused before; no pilot run can enter.

| ID | Sev | Finding | Disposition (before any run) |
|---|---|---|---|
| F1 | HIGH | rule changes after results were not excluded | Immutability clause above; 18.1 restricted to mechanical content |
| F2 | HIGH | failed screening runs silently left the LR grid; no retry; runner continued after confirmation failure | Explicit grid paths; divergence vs failure classified; one retry; exit 6 (flow tests) |
| F3 | HIGH | extensions could consume money needed by required runs; asymmetric skip possible | Joint gate (extensions + 9 confirmations), symmetric skip; 9-run gate before confirmation (flow test) |
| F4 | HIGH | "two-sided 90 %" was really TOST α 0.10 = two-sided 80 % | Text and rule basis corrected (α 0.10 kept, labelled honestly) |
| F5 | HIGH | resumed run accepted when its JSONL was missing | JSONL required; `steps_run == final_step` (test) |
| F6 | MED | LR/schedule/seed/U_fast not bound to 18.1 | Rule reads 18.1; checks LRs, U_fast, faster shape, fixed settings, stage ids, registered seed sets (tests) |
| F7 | MED | ET runs started before 18.1 was committed; marker unchecked | All confirmation waits for the marker holding the SHA-256 of `amendment_18_1.json` (test) |
| F8 | MED | ±5 % time tolerance too loose; r from step-rate median | ±2 %; r from training seconds per update (tests) |
| F9 | MED | equivalence picked any efficiency difference; NaN fell to core; wall time excluded data load | ≥ 3 % materiality else INCONCLUSIVE; finite checks; paid time = process start-to-end (tests) |
| F10 | MED | power statement used per-run SD for paired differences | Corrected above |
| F11 | MED | host change undetectable; reuse ignored arguments; faster/slower re-derived | GPU UUID + pod id sidecar per run checked by both rules; argument-checked reuse; faster from 18.1 (tests) |
| F12 | LOW | critical set derived, not checked | Derived set must equal the registered list (test) |
| F13 | LOW | eval kind/size, final-step detail, NaN domain not checked | All checked (tests) |
| F14 | LOW | ec-budget mixed hosts/steps/seeds accepted | Cross-shape checks (tests) |
| F15 | LOW | LR digits; unresolved boundary not recorded | Text uses generated values; `unresolved_boundary` in `select_lr_*.json` and 18.1 |
| F16 | LOW | "4 + final" evaluations | Text corrected |

Re-review of `1fc7e4b` (same reviewer): **APPROVE_WITH_FIXES**: F1-F10, F12, F13, F15, F16 VERIFIED; F11 PARTIAL; F14
verified with LOW gaps; new N1-N4. Dispositions (before any run), each with a regression test:

| ID | Sev | Finding | Disposition |
|---|---|---|---|
| F11 | (partial) | confirmation host never compared with the screening host behind 18.1 | Rule requires `ec_budget.host` (recorded in 18.1) == confirmation GPU UUID/pod |
| F6/F14 | LOW | `gqa_kernel`/`verify_data` not fixed; ec-budget ignored settings and resume | Added to fixed settings; ec-budget checks fixed settings, stage id, JSONL presence and resume |
| N1 | MED | a diverged screening run was re-rolled on restart | A diverged run with matching arguments is a final result, never re-run |
| N2 | MED | a diverged confirmation run was retried and replaced | Divergence in confirmation writes an INVALID record and exits 7 (no retry) |
| N3 | LOW/MED | 9-run gate before the wait, counting completed runs | Gate on remaining (incomplete) runs, before and after the amendment wait |
| N4 | LOW | missing `--amendment` crashed | INVALID record |

Final re-review of `4a11697`: **APPROVE** (all items verified, no stage-3 regression). Its LOW observation (the one-retry
limit was per invocation) is closed: archived attempts count toward the limit across restarts (test).

## 10. Successor registration 18b (supersedes the protocol-18 runs started 2026-10-10T17:56Z)

- **Why:** after two screening runs of this protocol had completed (`scr-j1-core-28x640-x1`, `scr-j1-wide-20x768-x1`),
  the F9 contamination closure (Data `5b58382`, criteria pre-registered at `4c24740`) adjudicated 7,008 documents of
  `j1-base-v0.2.1` PROHIBITED (train 6,995, validation 7, test 6), and the internal-source exclusion (Data `1743277`)
  removed 100 `cinqic-*` documents. The protocol-18 training prefix held 520 of these 7,108 documents (1,761,096 tokens,
  0.70 %) and its validation split held 7. The order forbids prohibited documents in comparisons, so protocol 18 is
  stopped and every one of its runs is **superseded and not counted**; their records stay in `/vol/runs/fu3`.
- **Observed before this registration (disclosed, not used):** `scr-j1-core-28x640-x1` validation loss 3.44820 /
  bpb 1.30115; `scr-j1-wide-20x768-x1` 3.54372 / 1.33720 (v0.2.1 prefix and validation), their four intermediate
  evaluations each, and their screening throughput (117,773 and 129,005 tokens/s, r ≈ 1.095). None of it can enter an
  18b rule: U_fast comes from 18b's own screening runs. No other result exists.
- **Only changes (everything else in sections 1-9 and their review fixes is unchanged):**
  - Training prefix: `cand5-j1tok-24k-successor-prefix-960MB-v022excl` (Data `tools/successor_prefix.py`): the
    keyed-order 960 MB prefix of the successor `j1-base-v0.2.2` (= v0.2.1 minus the 7,108-document union list
    `reports/j1-base-v0.2.2/excluded_doc_ids_union.json`), 215,831 documents, 250,772,987 stream tokens, `shards.json`
    SHA-256 in `docs/autonomous_2026-10-10/inputs_18b.json`. Independently checked: no excluded ID; all validation and a
    3,000-document train sample token-identical to v0.2.1; successor order.
  - Validation: the successor validation split, 2,576 documents, **12,311,466** scored bytes (constants in
    `tools/select_rule.py`); critical domains unchanged (the same seven sources ≥ 700,000 bytes).
  - Run directory `/vol/runs/fu3b`; amendment record **18b.1** (same mechanical content as 18.1; the on-pod file keeps
    the name `amendment_18_1.json` and `"amendment": "18.1"`, which the rule requires) committed under
    `docs/autonomous_2026-10-10/amendment_18b_1/` before any confirmation run by
    `tools/amendment_autocommit.sh j1wd /vol/runs/fu3b docs/autonomous_2026-10-10/amendment_18b_1 18b.1`.
  - Input binding (review M1): the runner refuses to start unless `sha256($PREFIX/shards.json)` equals the registered
    value in `inputs_18b.json`.
- **Immutability** (section header clause) applies from the first 18b screening run. An independent reviewer checked
  that 18b differs from 18 only as listed (APPROVE_WITH_FIXES on `17c2552`; M1 input binding and L1-L3 fixed before
  the first 18b run).
