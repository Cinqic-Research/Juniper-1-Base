# Juniper 1 — independent audit, repair, integration and pretraining preparation (owner order 2026-10-10)

Executor: Claude Fable 5.1 (goal mode). Authority: the owner order of 2026-10-10 and Design Specification v0.7
(DOCX SHA-256 `121fcb1de0dc927ce0d8d725c10bb68d199e28185f699d786472191418292ee3`, recomputed 2026-10-11).
Work state, ledgers and raw evidence: FLOWBOX HDD `juniper1-archive/fable-2026-10-10/` (WORKSTATE.md, ledger/,
findings/, evidence/). This report is updated as the mission progresses; dispositions use the order's vocabulary.

## 1. Scope and coverage map

| Area | Inspected | Executed | Metadata only / unavailable |
| --- | --- | --- | --- |
| Train: `j1bench/` (data, train, ckpt, timelog, timeaudit, model, store, replicate), `tools/` (launch chain, controllers, manifest, readiness, contract, money, compile checker, recovery suite, replicator), `qual/`, tests, CI | all production source and tests read by the lead or a fresh-context reviewer | full CPU suite on FLOWBOX (376 passed at 99273bb); GPU suite, CUDA/SDPA/FA-1 qualification, recovery suite, time-logger qualification and B17 measurements on the A100 | the owner-specific GitHub branch-protection rules (403 on the private repos) |
| Train evidence: protocol 18b records, compile check, launch attempt, budget contract, money rollup, readiness v2 | parsed programmatically; selection and money rollup recomputed | `select_rule.py architecture-v2` rerun reproduces `select_architecture_v2.json` byte-for-byte | on-pod volume state of the 2026-10-10 session (read live on 2026-10-11) |
| Data: `src/j1data`, `tools/`, configs, manifests, reports for v0.2.2 and stage recovery | read by a fresh-context reviewer; identities recomputed from the HDD canonical files | tests (255 / 278 / 291 merged), dry merges | HF/Drive bytes (receipts only), full per-document re-encoding |
| Backup: `src/cinqic_backup`, tests, docs, deployed config (numeric gates only), Drive drill restores | read; drill manifests re-verified with `sha256sum -c` | unit tests (66 / 55) | live Drive/HF objects (no provider access in review) |
| Other repositories (AAA, App, website, retired projects) | entry points and claims only | none | not in the launch path; AAA #34/#36 and website #22 left unmerged (research/security gates unresolved) |

## 2. Findings ledgers

- `findings/review_train_pr6_tooling.md` (T-01..T-22), `findings/review_numerics_selection.md` (N-01..N-17),
  `findings/review_data_pr5_pr4.md` (D-01..D-16), `findings/review_backup_pr1_pr2.md` (K-01..K-16),
  `findings/review_repairs_99273bb.md` (fresh-context review of the repairs).
- Mandatory items from the order: B17 (root cause confirmed and repaired, measured), B13 (funding contract v5,
  A100 time-logger PASS, supervision re-armed per job), stale readiness (fail-closed binding, regressions),
  B16 validity (review record must name the deployed commit), manifest glob (fixed, tested), recovery
  qualification (final-volume suite re-run on the launch stack), data recovery (PARTIAL, honest labels, fixes in
  the Data follow-up), preservation scope (18-file receipt is launch inputs only; corpus text off-site still
  NOT_VERIFIED), backup documents (reconciled in the Backup follow-up without rewriting history), AAA/website
  (kept separate).

## 3. B17 (startup) — VERIFIED on the A100

Root cause at 8534f2a: the trainer rebuilt the token stream for `--exact-epoch` (second hash + preload of 6.8 GB),
materialised 3.4e9 int64 targets plus a per-target byte lookup (~27 GB each) in `consumed_prefix_record`, and emitted
no progress record before update 1. Repair: one stream build and one read per file, `TokenStream.truncate()`,
bounded-chunk exact histogram accounting (forked workers), startup phase records with real counters, a time-log
startup span before the run span, and a `startup_complete` record. Regression tests: exact equality with the
historical per-target formulation (shard boundaries, truncation, tails, chunk sizes, parallel vs serial), one build
and one read per file, bounded requests, out-of-vocabulary and same-size corruption refused, exact big-integer sums,
truncate == constructor limit for batches/fingerprints/resume, startup records, exact-epoch resume, and the real
v0.2.2 validation split (prefix bytes == 12,311,466).

A100 measurements (pod pncbyxaf2dt2jl, GPU-5c42020f…, full v0.2.2 stream, launch arguments, eager profile):
r1 169.8 s (serial accounting, first process after the restart), r3 31.0 s (forked histogram), peak RSS 7.8 GiB,
accounting identical to the FLOWBOX recomputation. The gate-binding record is re-measured on the deployed launch
commit before authorization (cold process after a pod restart).

## 4. B13 (time and money)

- Funding contract v5: balance 18.07 (owner screenshot 2026-10-11T03:09Z, reconciled against the 01:38Z reading
  plus the $10 deposit and intervening charges), S = 48.83 provider-billed, admissible exposure 13.70, frozen
  allocations, per-resource caps, concurrency, deadlines, stop conditions; qualification admitted, base launch
  blocked until a successor version records the launch gate. Admission control: `tools/funding_admission.py`
  (file-locked aggregate ledger).
- A100 time-logger qualification PASS on the network volume and container disk (B13a). Shutdown drills D1–D6
  transcribed to JSON and pinned to the markdown record (B13b). Both supervisors armed per job with matching
  limits; the watchdog runs on a separate CPU pod from the committed code on the volume.

## 5. Integration

Merged with exact-head guards after recorded technical reviews: Train #6 → opus/pretraining-prep (ac448fd),
Data #5 → opus/pretraining-prep (1645d61), Data #4 → opus/pretraining-prep (b1f211a), Backup #2 →
feat/four-destination-backup (6b732c4). Follow-up PRs: Train #7 (this work), Data fixes, Backup fixes; then
preparation → main integration PRs.

## 6. Compiled profile (registered pair, 2026-10-11)

One matched eager/compiled pair of 1,000 updates on the launch basis (full v0.2.2 stream, horizon 52,034, warmup
500, seed 1; identical batch fingerprints at every update). Result under the registered rule: **FAIL** (final bpb
eager 1.36626 vs compiled 1.36065, relative difference 0.411 % > 0.3 %; mean relative loss 501-1000 0.417 % within
0.5 %; max per-step 2.6 % > 2 %; critical domains outside 0.5 %). The registered diagnostic eager replicate
(identical batches) gives the eager-vs-eager noise floor: final bpb 0.153 %, mean loss 0.250 %. The compiled
difference is 2.7x the noise floor and has the same sign in both pairs (compiled lower), so it is not explained by
kernel nondeterminism alone; the registered tolerance stands and the **launch profile is eager** (65.3k tok/s;
compiled 131k tok/s is not promoted). Records: `docs/fable_2026-10-10/compile_equivalence_v2.json`,
`evidence/ce2/`. Cost of the pair plus diagnostic plus drill reference: 2.30 (controller estimate).

## 7. Recovery and preservation

- Recovery suite on the network volume with the final code and the launch shape (eager, deterministic reference
  profile): 8/8 PASS (`evidence/recovery_fable2/results.json`); bounded storage (checkpoints of finished tests
  removed after their verdict).
- Off-site: FLOWBOX upload bandwidth is 1.7 MB/s to Drive and 2.2 MB/s to HF, so each 1.63 GiB checkpoint takes
  about 17 / 13 minutes; launch policy LP-3 freezes a 4 h delivery deadline per checkpoint from HDD verification,
  retention on the volume until both receipts exist, and compute stop allowed once the primary and the HDD
  replica are verified. HDD replica lag limit stays 600 s / 1,000 updates (guard-enforced).
- Drive-only clean-room drill on the current stack: repository bundles (Train/Data/Backup heads), time logs, the
  v0.2.2 launch inputs and a deterministic drill checkpoint restored from Drive bytes alone into an NVMe clean
  room by `tools/drive_only_restore.sh` (result recorded in section 9 when complete), followed by a compute
  resume on the A100 compared bitwise with the uninterrupted reference.

## 8. Status (2026-10-11 13:05Z)

| Item | Disposition |
| --- | --- |
| B17 startup | VERIFIED (A100: 169.8 s → 31.0 s → cold 18.4 s on 8f91e88 → cold 13.0 s on the deployed f0b4dea; launch canary startup 12.8 s incl. restore) |
| B13a time logger | VERIFIED (A100 PASS, volume and container disk) |
| B13b shutdown drills | VERIFIED (transcribed record, pinned) |
| B13c funding | VERIFIED (contract v6 FC-6b; readiness-bound LAUNCH admission; ledger closed: launch 1.46, committed 6.64 / 13.70) |
| Compiled profile | NOT_VERIFIED → launch profile eager (registered pair FAIL: final bpb 0.411 % > 0.3 %) |
| Recovery suite (final volume, final code) | VERIFIED 8/8 |
| Drive-only clean-room drill | VERIFIED (restore ok; compute resume bitwise with the reference, recorded corrective continuation) |
| Integration | Train #6, #7 → prep; #8 prep → main 17a68c2; #9 (fable/launch-r2, code + records) → prep ca6be44; #10 → main 2d9b814; #11/#12 (preservation-complete docs) → prep dd71894 → main c2dd7a5. Data #5, #4, #6 → prep; #7 → main 47cd1dd. Backup #2, #3 → feat; #1 → main 1e95afe. All with review records and exact-head guards (`merge_receipt.md`). |
| Readiness | ready, bound (matrix 90c62469, manifest b9c1257a, gates 1459155b, funding 3834c3d6) |
| Canary (50) | ACCEPTED (C1–C9) |
| Limited tranche | **LIMITED_TRANCHE_COMPLETE** (2,000 updates; 1.2228 bpb; all checkpoints replicated) |
| Continuation | **EPOCH_COMPLETE** 2026-10-12 13:49Z: j1-base-r2 trained to 52,034 updates (final 0.9165 bpb); interrupted once by a FLOWBOX power loss (guard stop at 8,585, resumed from 8,500); all keepers preserved off-site; cost 30.56 of the v7 envelope |
| Off-site preservation | VERIFIED: tranche checkpoints and every keeper of the epoch (0…50000, 52034) on Drive + HF, read back; corpus text off-site still NOT_VERIFIED |

## 9. Known limitations and open items

- X-01: the authorization binds the gate-definition file by hash (uploaded with the authorization), not by its
  presence in the deployed commit; the hash is recorded in LAUNCH_AUTHORIZED and the preconditions log.
- X-02: some stale comments remain in the launch tooling (documented in the fix reviews; no behavioural effect).
- `tests/test_launch_manifest.py::test_t05` needs git history and fails on the pod's tar-deployed tree
  (environmental; passes on FLOWBOX; recorded in the host record).
- K-05: the FLOWBOX HDD archive has ~24 GB free after the six replicas; a continuation needs a retention rule.
- Launch attempts 1 and 2 were refused by design (host identity after a pod restart; watchdog session not
  re-emitted by `rearm_supervision.sh`, which rewrites the config but does not restart the watchdog) — a
  follow-up should make the re-arm script restart the watchdog or make the supervision check accept a config
  newer than the session event.
- The eager-vs-compiled question remains open for the long run (compiled would halve the ~14 h remaining epoch).
