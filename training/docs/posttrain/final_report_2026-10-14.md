# Juniper 1 engineering assignment 2026-10-13 — final report

Owner directive: *Juniper 1 engineering execution directive* (Markus Gillyard, 2026-10-13; $5 RunPod cap, raised
by the owner to $10 at 2:12 AM EDT 10-14; $5.25 spent, §10).
Engineer: Claude Opus 5.5 (goal mode) with independent reviewer subagents (labelled; not the owner's final
review). All times US Eastern. Machine-readable findings: [`findings_2026-10-14.json`](findings_2026-10-14.json).
Nothing was merged, promoted, released or made public; every change is a draft PR for the separate reviewer.

## 1. Status by priority

| priority | status | evidence |
|---|---|---|
| A. Replicate SFT v1.2 (seeds 2, 3) | **DONE** — both `PILOT_CANDIDATE_QUALIFIED`, consistent with seed 1 | Train#29, `results/sft_replication_v1/` |
| B. Dataset v1.3 | **DONE** — `APPROVED_WITH_CONDITIONS` after one rejected build | Data#11, `docs/sft/reviews/` |
| D. SFT v2 | **`FAILED`** (primary generalization passes strongly; four non-inferiority regressions and two chat knowledge relations) | Train#31, `results/sft_v2_v1/` |
| C. Native App token-ID adapter | **IMPLEMENTED + LIVE-VERIFIED on desktop Linux**; Android blocked | App#46 |
| Infrastructure: A-11 | **REMEDIATED (config)**; live unusability NOT VERIFIED | `findings` A-11 |
| Infrastructure: storage | measured; ~21 GB of checkpoints offloaded to private HF with fresh-download verification, then removed locally | `findings` |
| Infrastructure: backup/DR | 5 Drive restore drills **VERIFIED** (bytes) + checkpoint usability; governed encrypted path **BLOCKED** (owner) | Backup#4 |
| E. GPT-OSS distillation | **`NO_IMPROVEMENT_DETECTED`** for all three teacher arms; retention fails in every arm incl. the control (stage design overfits 318 matched problems) | Data#11, Train#31, `results/distill_v2_v1/` |
| F. Tool-use support | **IMPLEMENTED/TESTED**, real host traces; no training (gated) | App#47, Data#10, Train#28 |
| G. Bounded agentic | proposal + prerequisites (gated on F) | Train#30 `later_stages_v1.md` |
| H. Context extension / AAA | requirements recorded; AAA unchanged | Train#30 |
| Docs reconciliation | README/roadmap errata | Train#30 |

## 2. Repository identities and merge order

Start (all = the directive's snapshot): Train `68dc3b8`, Data `b7af963`, App `e3e8656`, Backup `1e95afe`,
AAA `1aa0e38`, cinqic.com `33ded1e`, Calculator `15b765c`, LM-1 `a166627`, Reference-4B `484fc3e`,
LM-1.1 `0074142`. Draft PRs (none merged; CI green at last check):

| order | PR | branch | purpose |
|---|---|---|---|
| 1 | Juniper-1-Data#11 | `sft/dataset-v1.3` | dataset v1.3, scorer v4, codecheck, OASST builder (unused), distillation v2 tooling |
| 2 | Juniper-1-Train#29 | `sft/replication-v1` | seed replication, knowledge diagnostic, v2 decision tool |
| 3 | Juniper-1-Train#31 | `sft/v2-stage` | SFT v2 and distillation v2 protocols, results, this report (builds on #29) |
| 4 | Juniper-1-Train#30 | `docs/status-2026-10-13` | README/roadmap errata, later-stages proposal |
| 5 | Juniper-App#46 | `a02/native-token-id-adapter` | native token-ID backend (independent of 1–4) |
| 6 | Juniper-App#47, Data#10, Train#28 | tool-use support | host trace harness, `j1-tool-v0.1`, tool rows (after #46) |
| 7 | Cinqic-Backup#4 | `dr/restore-drills-2026-10-13` | restore drill receipts |

## 3. Phase A — seed replication (detail: `results/sft_replication_v1/README.md`)

Protocol registered before any result; seeds 2 and 3 repeat the v3 pilot exactly (trainer package
`f13afbdf`, same RTX 2060 environment, original contract hash unchanged). All three seeds:
`PILOT_CANDIDATE_QUALIFIED`. P1 0.648 / 0.662 / 0.662 (sd 0.008); held-out families 0.011 in every seed;
full-split bpb 0.94073 / 0.94055 / 0.94084 (limit 0.94387); no pairwise seed difference (smallest Holm p
0.54). Registered consistency check: consistent. Corrected rescoring (scorer v4) of the preserved raw
generations of the three seeds and the base changes **no** item (separate labelled view). Candidates on HDD
and private HF (fresh-download verified).

## 4. Phase B — dataset `j1-sft-v1.3` (Data `docs/sft/dataset_v1.3.md`)

- **Manifest `8584130b…`, bundle `935e12ed…`** (Data `5162bf7`); 61,806 train / 1,718 dev episodes,
  1,011,381 target tokens; 6,634 test items. Double build byte-identical; independent `verify` and
  reviewer re-render pass. Private HF `CinqicX/juniper1-corpus@24ae3b3a` (readback identical).
- S-09 fixed (scorer v4). Coding: cross-execution twin detection (37 MBPP test tasks had identical-behaviour
  twins in v1.2 train/dev; 144 tasks quarantined), out-of-process checker (rejects 14/14 known exploits),
  stronger graded suites (wrong-code acceptance 11.4% → 2.2%); rights entry R14 (EvalPlus, eval only).
- Generalization: compositional operations with evaluation-only paraphrases (23), frames (2), pairs (64) and
  operations (10); formatting/extraction, multi-turn, grounded QA. Targets computed only.
- OASST1: filtered builder implemented; the independent blinded rubric review rejected the source
  (model-pasted text in admitted rows) → no OASST1 training data.
- Knowledge diagnostic (dev only, `results/knowledge_diag_v1/`): no general knowledge loss after SFT
  (few-shot 21 vs base 18 of 85); capitals loss is chat elicitation (0/24 chat vs 5/24 few-shot); removing
  fact data makes the model abstain on everything → v1.3 keeps facts; v2 gates retention on few-shot answers.
- The first build `bf769ecd` was **rejected** by the independent admission review (held-out pairs leaked via
  chained conversations, reserved paraphrases via other families, NOTICE gaps, 2:1 abstention ratio); fixed,
  rebuilt, re-reviewed: `APPROVE_WITH_CONDITIONS` (conditions tracked in the go/no-go record).

## 5. Phase D — SFT v2 (detail: `results/sft_v2_v1/README.md`)

Protocol frozen at 3:46 PM EDT 10-13 before any v2 run and before the v1.3-bundle baselines: training
identical to the pilot (1,226 updates, LR 3e-5, fp32, deterministic) so the comparison isolates the dataset;
seeds 1–3 on rented RTX 4090s; candidate = seed-1 final; baseline = the three replicated v1.2-recipe
candidates on the same bundle and scorer, paired by seed. Smoke/recovery: A/B bitwise, corrupted and torn
checkpoints rejected with rollback.

**Outcome `FAILED`.** Primary endpoints pass by a wide margin on every seed: held-out paraphrases
0.020 → 0.385 (+0.365, CI [0.321, 0.410]) and held-out compositions 0.004 → 0.348 (+0.343, CI [0.293,
0.393]), holding on clean reserved phrasings (+0.31) and unseen-function compositions (+0.37). In-distribution
instruction following +0.26, grounded QA 0.21 → 1.00, formatting 0 → 0.99, multi-turn 0.06 → 0.50. But
regressions are established (CI entirely below −0.03): generated arithmetic −0.15, prompt injection −0.15,
instruction hierarchy −0.13, JSON −0.08; chat answers on continents (−0.06) and element symbols (−0.11) fail
the knowledge margin (few-shot retention against the base passes on every relation). Every gate passes on
every seed (bpb 0.9406–0.9407 ≤ 0.94387). Unseen operations stay near zero. Open-ended blinded review
(reported): both models fail almost all 21 OASST1 prompts (2/21 vs 0/21). Interpretation: v1.3 teaches
transfer across phrasings and combinations, but its mixture diluted arithmetic, injection, hierarchy and
JSON under the fixed update budget; a follow-up needs a new protocol restoring those shares (or replay).

Deviations, all recorded before the decision: smoke after seed 3 (defective first GPU); seeds 1–2 re-run
on a new pod because the first pod could not be restarted (its outputs were never read); smoke result
files regenerated from preserved manifests and logs. Hardware control: the seed-1 candidate's 6,634 test
outputs regenerated on the RTX 2060 match the RTX 4090 (6,633 identical token sequences, every strict score
identical). Checkpoints: private HF with fresh-download checks.

## 6. Phase C — native App adapter (App#46)

Rust `juniper1` backend: `tokenizers` 0.23.2 with load-time special-token settings; juniper-chat-v1
serialize/budget/parse/analysis continuation ported; integer prompts to `/completion` only, integer bans,
non-streaming IDs, loopback-only, no proxies or redirects, listener ownership; model/tokenizer/protocol/
runtime identity verified before launch; App tool gate reused; text-path guard unchanged; not exposed in
the UI or catalog. Live on llama-server b11270 (CPU) with both real GGUFs: 34/34 checks; greedy ID parity
with the Python client 48/48 (base and candidate); golden parity 5/5 incl. 200/200 regression documents
(private fixtures, digest-pinned). Android: still blocked (regression test + written gaps). Not verified:
Windows/macOS locally (CI passes), start-up timeout/cancel during load. Privacy note: the subagent once sent
the owner's email in a crates.io User-Agent; token IDs of the private tokenizer were briefly on a public
branch (history rewritten).

## 7. Infrastructure

- **A-11**: Jupyter passwords and port 8888 removed from three exited pods (env PATCH; SSH key kept); no
  local copies or consumers; old credentials' live unusability NOT VERIFIED (would need a paid start).
- **Storage**: HDD 15 GB free (97%); NVMe ~20 GB. Seven unique non-candidate checkpoints offloaded to private
  HF with fresh-download verification, then removed locally; reproducible Rust build caches removed.
- **Backup/DR** (Backup#4): Drive-only restores of Git bundles, dataset v1.2, tokenizer, base and pilot
  checkpoints VERIFIED by bytes; pilot checkpoint usability (verify, load, full state restore, dev loss)
  VERIFIED on CPU; FLOWBOX is not a clean machine; fresh Git bundles of Train/Data/App (incl. current main)
  on HDD. BLOCKED (owner): Google OAuth app publishing (Testing mode; refresh likely fails after
  ~Oct 16), key-recovery sheet custody, governed encrypted namespace/key-recovery config.

## 8. Phase E — GPT-OSS-20B distillation (detail: `results/distill_v2_v1/README.md`)

Protocol v2 (Data, registered 10-13) and the Train stage file (registered before any arm; init = the v1.2
pilot seed-1 candidate because SFT v2 failed). Teacher: pinned GPT-OSS-20B GGUF on llama.cpp b11270 with the
qualified 20B evaluation configuration on the RTX 2060, 1,200 GSM8K train problems (1,121 accepted).
Verifier amended to v1.1 before any arm trained (layout normalization only; v1 marked correct equalities
such as "60 + 25 = 85 ." unverifiable). 318 matched problems; arms C0 (human solutions), A1 (answer), A2
(verified concise rationale), A3 (verified equations), each + 954 identical v1.3 replay episodes; double
build byte-identical; four arms trained and generated concurrently on one rented RTX 4090.

**Outcome: `NO_IMPROVEMENT_DETECTED` for A1, A2 and A3.** GSM8K test: C0 0.032, A1 0.018, A2 0.022, A3
0.016 (all differences non-significant, Holm p ≥ 0.35). Retention fails in **every arm including C0**
(full-split bpb 0.960–0.970 > 0.943872; dev loss rises at every evaluation): 200 updates × 64 episodes is
~10 passes over the 1,272-episode arm datasets. A1 is significantly worse than C0 on math. Correct GSM8K
answers in A2/A3 often carry a false equation in their working (5/11, 4/8). Reading: the 143M student does
not yet solve GSM8K (0.014 at init), and the registered update budget overfits the matched set; a future
stage should size updates from the matched set (≤ 2 passes) or register dev-loss early stopping and target
skills the student can express. No arm warranted the replication seed.

## 9. Phases F–H

- **F (tool use)**: App harness compiles the App's real `tools.rs` and grant logic (3,789 real host calls incl.
  denials, invalid arguments, timeouts); `j1-tool-v0.1` (3,279/328/329 episodes, double build identical,
  re-verified against a fresh harness); tool serializer trains only call text and finals; scorer separates
  model behaviour from host enforcement; Train loads tool rows only under an `agent-tool-sft` contract.
  No tool training (needs its own stage authorization after the adapter and SFT stages qualify).
- **G**: bounded agentic workflow proposal with host-enforced bounds and controls (`later_stages_v1.md`).
- **H**: context-extension requirements; AAA#34/#36 and cinqic.com#22 unchanged.

## 10. Cost and resources

Provider billing (RunPod hourly buckets, settled; read 10:30 AM EDT 10-14) for this assignment, from 3 PM
EDT 10-13: **$5.25 total** (pod GPU $5.14 + pod disk $0.11); no pod charge after 6:00 AM EDT 10-14. Excluded: the owner's pre-existing volume storage
(~$0.005/h, $0.07 over the window). Authority: $5, raised by the owner to $10 at 2:12 AM EDT 10-14. No
top-up, auto-recharge or subscription. Owner balance readings: $13.03 (10-12), $10.31 (12:40 AM EDT 10-14), $7.31 at a $0.005/h spend rate (volume
storage only) at 10:25 AM EDT 10-14; the 10-12 → 10-14 drop of $5.72 = $5.25 assignment + ~$0.23 storage + ~$0.24
not attributable from the API (likely late-posted charges of the previous order).

| stage | pods | GPU | spend | stage cap |
|---|---|---|---|---|
| SFT v2 | `ujsfypdqm5rhui` (community 4090, defective GPU, terminated), `omzwhqkxidvjgm` (secure 4090: seed 3 + smoke; watchdog-stopped, unrecoverable, terminated), `lvmojshkyirv5z` (secure 4090: seeds 1 + 2, terminated) | RTX 4090 | $4.31 | $4.90 (amended from $3.00 via $4.00, before the decision) |
| Distillation v2 | `rxveduih7p5lws` (secure 4090, 4:23–5:42 AM EDT, terminated) | RTX 4090 | $0.94 | $1.50 |
| everything else | FLOWBOX (RTX 2060): replication, dataset, baselines, teacher batch, scoring, hardware control | — | $0 | — |

Teardown: no assignment-created pod, volume or endpoint remains (list at 5:45 AM EDT: only the owner's
older EXITED pods `dxnsn6g7n5s4fw`, `v247r25kcigu04`, `pncbyxaf2dt2jl`, `k3mul4cjkmiong`, `dxl7jz3zezlz0m`,
`4i13u1p447t6e7`, untouched). Integrity: `juniper1-base-v1` loads through the pinned loader (record
`ef79b324…`, every checkpoint file re-hashed, 5:47 AM EDT); `docs/freeze/` unchanged since `68dc3b8`. Ledgers: archive HDD `assignment-2026-10-13/ledger/*.jsonl`.

## 11. Owner (human-only) actions

1. Google Cloud console: publish the Cinqic Backup OAuth app (Audience → Publish); reconnect rclone.
2. Print and store the backup key-recovery sheet; record custody.
3. Decide the governed Drive namespace/key-recovery configuration (Backup#4 lists the fields).
4. Optional: author seed conversations for open-ended writing (OASST1 was rejected).
5. Hugging Face: the `CinqicX` org reached its private storage limit at 7:15 AM EDT 10-14 (one distillation
   checkpoint is on the HDD only): decide on a plan upgrade or on pruning superseded checkpoints.
6. Review and merge the draft PRs in the order above (the engineer does not merge).

## 12. Reproduction

Environment as on FLOWBOX: Train venv (torch 2.8.0+cu128, numpy 2.1.2, tokenizers 0.23.2), Data venv,
`J1_DATA_ROOT=/media/cinqic/Cinqic Storage/juniper1-data/data`. Every pipeline is restartable.

```bash
# dataset v1.3 (Data 5162bf7): build twice and compare, then verify
PYTHONPATH=src python -m j1data.sftbuild build --config configs/sft_v13.toml --root "$J1_DATA_ROOT" --out /tmp/v13A
PYTHONPATH=src python -m j1data.sftbuild verify --out /tmp/v13A --tokenizer "$J1_DATA_ROOT/tokenizer/j1-base-v0.2.2/frozen/cand5-j1tok-24k/tokenizer.json"
# seed replication (Train sft/replication-v1)
tools/sft_replicate_pipeline.sh 2   # and 3; env vars in the script header
python tools/sft_seed_report.py --seed s1 EVAL_V12 --seed s2 EVAL_S2 --seed s3 EVAL_S3 --label original_scoring --out report.json
# knowledge diagnostic
tools/sft_knowledge_diag.sh
# SFT v2 (Train sft/v2-stage): smoke, seeds, baselines, decision
tools/sft_v2_smoke.sh; tools/sft_v2_pipeline.sh 1   # 2, 3
tools/sft_baseline_eval.sh rep_s1=CKPT rep_s2=CKPT rep_s3=CKPT base=BASE_CKPT
python tools/sft_v2_decide.py --protocol docs/posttrain/sft_v2_protocol_v1.json --v2 1 E1 --v2 2 E2 --v2 3 E3 \
  --baseline 1 B1 --baseline 2 B2 --baseline 3 B3 --base-scores BASE/cand_scores.json \
  --recovery-ab SMOKE/recovery_ab.json --sensitivity docs/posttrain/sft_v2_sensitivity_v1.json --out decision.json
# distillation v2 (Data distill + Train distill tools)
PYTHONPATH=src python -m j1data.distill pool ...; python -m j1data teacher-run --prompts PROMPTS --run-id gptoss-v2
PYTHONPATH=src python -m j1data.distill arms ...; tools/sft_distill_pipeline.sh C0  # A1 A2 A3
python tools/sft_distill_decide.py --arm C0 E_C0 --arm A1 E_A1 --arm A2 E_A2 --arm A3 E_A3 --data-src DATA/src --out d.json
```

