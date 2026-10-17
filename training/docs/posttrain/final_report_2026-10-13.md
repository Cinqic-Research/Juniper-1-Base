# Juniper 1 post-training assignment — final engineering report (2026-10-12/13)

Owner order: *Independent Engineering Audit, Post-Training Implementation, Qualification, and
Controlled Research Expansion* (2026-10-12, $10 RunPod ceiling). Executor: Claude Opus 5.5 with
independent reviewer subagents. Machine-readable findings: [`findings_2026-10-13.json`](findings_2026-10-13.json).

## 1. Outcome

A first real instruction-SFT dataset (`j1-sft-v1.2`), a qualified SFT trainer, a frozen evaluation
suite with prospective rules, and a token-ID-only runtime path were built, independently reviewed
three times, and used for one bounded pilot run **on FLOWBOX's RTX 2060 at $0 RunPod cost**. Under the
preregistered protocol v3 the pilot candidate is **`PILOT_CANDIDATE_QUALIFIED`** — a qualified,
**non-production** starting point for the next stage. It learned the trained formats and tasks well
(in-distribution instruction following 0.648, JSON 0.704, the trained abstention phrase 0.993,
transformations of text containing injected instructions 0.960, 95.7% well-formed turns) but shows
essentially **no transfer to unseen instruction types** (held-out families 0.011) and is weak on
knowledge and reasoning (GSM8K 0.012, HumanEval 0.012). On country and US-state capitals it is
**worse than the base's few-shot answers** (2/51 vs 11/51; exploratory, not a registered test — the
registered factuality category, dominated by a continent gain, shows no significant change).
Base-language retention passed narrowly (bpb 0.9407 vs limit 0.9439; 89% of the allowance used). `juniper1-base-v1` is
unchanged and remains the canonical base and rollback target.

### Gate ledger

| gate | status | evidence |
|---|---|---|
| AUDIT | PARTIAL | repositories inventoried; Train/Data/App critical interfaces inspected and independently reviewed; Backup code inspected but not re-audited; AAA science not re-examined (§3) |
| BASE | PASS | freeze record digest `ef79b324…` and `model.pt`/`optim.pt`/`trainer.pt` hashes re-verified after all work; never written |
| DATA | PASS | `j1-sft-v1.2` manifest `d3738c50…`, deterministic, independently verified, rights ledger; two known limitations registered (§5) |
| MASK | PASS | loader rejects unshifted/±1 labels; independent re-rendering of all rows/masks; masked/pad gradient tests |
| TRAINER | PASS | `j1bench.sft`; 13 review findings fixed; 460 Train tests |
| RECOVERY | PASS | synthetic fault injection (mid-update kill, torn writes, corruption, rollback, contract changes) bitwise; real-model SIGTERM/resume **bitwise**; pilot checkpoint reproduces its dev loss exactly |
| EVAL | PASS | bundle `f9632015…` frozen before measurement; base baselines; protocol v3 registered before any score (v1/v2 superseded before measurement after reviews) |
| TOKENIZER (A-02) | PASS | text path isolated (App guard merged); host token-ID path qualified live for base and candidate (§8) |
| APP | PARTIAL | guard merged (Juniper-App#45); native token-ID adapter not built; Juniper 1 not offered in the App |
| BUDGET | PASS | $0.00 new RunPod charges (read-only checks only); no resource created or started |
| SFT PILOT | PASS | 1,226 updates completed, no early stop |
| PROMOTION | PASS | `PILOT_CANDIDATE_QUALIFIED` by the mechanical rule; independently re-derived byte-for-byte from raw token IDs (§10) |
| TEACHER | PASS | legality/feasibility protocol; sandbox fix; local 30-prompt feasibility batch, 28/30 verified (§11) |
| INTEGRATION | PASS | App#45, Data#9, Train#26 merged on reviewed heads with green CI (§12) |
| PRESERVATION | PASS | dataset v1.1/v1.2, pilot checkpoint and evaluation records on HDD + Drive + private HF with read-back (§13) |

## 2. Repository identities

| repo | start (`main`) | end |
|---|---|---|
| Juniper-1-Train | `d693d484` | `21bddf55` (PR #26) + this receipts PR |
| Juniper-1-Data | `fee7003d` | `b7af9634` (PR #9) |
| Juniper-App | `11225ba7` | `e3e86568` (PR #45 merged) |
| Cinqic-Backup | `1e95afe6` | unchanged |
| AAA | `1aa0e385` | unchanged (#34 head moved to `507dc415`: README/docs merge from main only; blockers unchanged) |
| cinqic.com | `33ded1e0` | unchanged (#22 still blocked on AAA) |
| Juniper-LM-1.1 / -LM-1 / -Reference-4B / Cinqic-Calculator | `0074142e` / `a166627f` / `484fc3ec` / `15b765cf` | unchanged (retired/reference; identity only) |

Design specification v0.7 bytes re-verified: SHA-256 `121fcb1d…92ee3`.

## 3. Audit coverage

| area | directly inspected / executed | relied on prior evidence | not reviewed |
|---|---|---|---|
| frozen base | record, loader, payload hashes; full-split fp32 bpb recomputed (0.916381 = record) | selection report | — |
| model math | `model.py` read; masking/padding/accumulation invariance tests; KV-cache vs full forward | earlier line-level audits | new line-level RoPE/QK-norm derivation |
| training code | new SFT trainer, loader, harness (three independent reviews) | pretraining trainer | — |
| data | new SFT pipeline end to end; all 42,558 v1.0 targets independently recomputed | pretraining corpus admission | — |
| runtime/App | llama-server source paths and live probes; App routes, guard call sites | — | App frontend |
| backup | Cinqic-Backup README/status (WIP, open gates) | — | its code paths (not relied upon) |
| cross-project | AAA#34/#36, cinqic.com#22, Train#2 heads and diffs | AAA science findings | — |

## 4. Defects found and fixed (all with regression tests; detail in the findings ledger)

- Teacher oracle ran teacher-generated code with the host filesystem visible → bubblewrap harness.
- Evaluation integrity in v1.0 (cheatable Python scorer, hedged abstentions, entity/MBPP leaks,
  degenerate base comparison, non-independent `verify`) → v1.1/v1.2 before any measurement.
- Trainer: `torch.save` failures bypassed retry, SFT checkpoints unloadable by eval tools, over-long
  prompt aborting generation, startup checks, unqualified fp16, single-process lock, cadence/dev order,
  dev-eval replay after resume, exact-resume code binding.
- Pipeline: recovery run B never signalled, vacuous recovery gate, unenforced early stop, stale results.
- Runtime client honoured `http_proxy` (bearer-key leak path) → proxies ignored.
- `episodes.validate` re-checked the JSON schema per call (45k episodes impractical) → one validator.
- A-02 qualification KV-cache check invalid once a model stops early → corrected, both models rerun.
- App CI blocked by new advisories (`source-map-js`, `quinn-proto`) → minimal lockfile updates.

## 5. SFT dataset `j1-sft-v1.2`

Manifest `d3738c507580122c24cc948f9c40b79f7dd84b9a3cd31120d1fa283554dc6526`, built by Data `dcbe102`
(two independent builds byte-identical; `verify` re-renders every row/mask independently). Train
39,234 episodes / 2,259,218 tokens / **769,875 loss-bearing targets**; dev 1,578; test 4,233 evaluation
items; 7,728 recorded rejections. Sources: Cinqic generators (computed targets; the only fixed target
phrase is "I don't know."), GSM8K `main` (MIT; 2,500 train, annotations re-executed), MBPP (CC BY 4.0;
sandbox-verified, bug-fix variants), Wikidata CC0 snapshot (`ad9a2bb5…`); HumanEval eval-only.
Excluded: hh-rlhf, Dolly, Alpaca, hosted-assistant outputs. Decontaminated against all 16 registered
benchmarks; split isolation by task keys including embedded tasks. Registered limitations: element-
symbol words ("No", "As") can wrongly fail an abstention (conservative for G-UNC); ~12 MBPP test tasks
keep functionally equivalent train twins (affects only the ungated Python report). Docs: Data
`docs/sft/`.

## 6. Trainer and recovery

`j1bench.sft`: strict frozen-base load; fresh AdamW (0.9, 0.95, wd 0.1 on matrices, clip 1.0) and
warmup-cosine schedule; exact per-update target normalization; one episode per row; fp32 only
(bf16/fp16 refused until qualified); deterministic; full-state checkpoints at ≤ 500 updates or ≤ 900 s,
at SIGTERM and at the end; bounded retry then exit 3; numerical abort without applying the update;
exclusive run lock; exact resume requires identical contract and trainer code. Pilot cadence observed:
checkpoints at 500, 1,000, 1,226 (661 s, 694 s, 334 s apart). Real-model recovery A/B on the RTX 2060:
SIGTERM landed at update 16, resume replayed updates 17–30 with identical batches and losses; final
model and optimizer digests identical (**bitwise**).

## 7. Evaluation suite and baselines

Bundle `f96320150025d0250b5fbe7f5bc44b1f39b56ea836c8dfe842dbfc48c508eea1`: prompts as token IDs, hidden
specs read only by the scorer; 4,233 test chat items (instruction following 772 incl. 272 held-out,
math 1,100 incl. GSM8K 500, Python 864 incl. HumanEval 164 and MBPP 500, JSON 500 incl. 200 held-out,
facts 176, uncertainty 300, hierarchy 221, injection 300) plus 1,645 base few-shot prompts. Protocol v3
(Train `docs/posttrain/sft_pilot_protocol_v3.md`). Base: chat strict 0.000 in every category (it never
saw the protocol); few-shot capability: facts 0.233, instruction following 0.114, JSON 0.540, math 0.070.
Base full-split bpb 0.916381 (fp32), identical to the freeze record.

## 8. A-02 and the App boundary

Host-side canonical tokenization (Option A). `j1data.runtime_client` sends only integer prompts to
`/completion`, bans host-only/reserved IDs by integer `logit_bias`, reads IDs back non-streaming,
resubmits after analysis `<|end|>`, returns only parsed turns, loopback only, no proxies or redirects.
Live on llama-server b11270 for base and candidate: 200/200 regression documents as exact canonical IDs
(0/200 through llama.cpp's text tokenizer), 0 forged control IDs across all 64 spellings, 0 banned IDs
sampled, byte fallback, KV-cache continuation 3/3, greedy parity 48/48 (base) and 47/48 (candidate, an
F16 near-tie). Candidate latency: 0.13 s/turn CPU, 0.04 s GPU. The App refuses Juniper GGUFs on every
text path (merged); a native Rust adapter is the remaining integration step. Record:
`docs/posttrain/a02_token_id_qualification.md`.

## 9. Cloud resources and billing

No RunPod resource was created, started, modified or deleted. Live reads (2026-10-13): six pods, all
EXITED; no endpoints; one 50 GB network volume (`72al92fys2`, pre-existing) accruing about $0.0049/h;
hourly billing since the assignment began shows only that storage. **New attributable charges: $0.00.**
The account balance is not readable through the connector. A-11 persists: three exited owner pods
(`k3mul4cjkmiong`, `dxl7jz3zezlz0m`, `4i13u1p447t6e7`) still hold plaintext Jupyter passwords in their
environment; terminating or rotating them is an owner decision (values not reproduced here).

## 10. The SFT pilot

Contract from protocol v3: LR probes 1e-5 / 3e-5 / 1e-4 (150 updates each). 1e-4 had the lowest dev
loss (0.702) but exceeded the retention limit (fixed-32 bpb 0.8979 > 0.8937); **3e-5** chosen (dev
0.796, bpb 0.8764). Pilot: 1,226 updates (two passes), 64 episodes/update, 1,024-token micro-batches,
40 warmup, cosine to 10%, fp32, seed 1; ~29 min on the RTX 2060. Dev loss 0.962 → 0.609, monotone.
Candidate `ckpt-000001226-55d13f9a`, manifest `dc0bbbb0cd655cc5114c1ee73ac777634985fca8a713a9f149f7686b08a110ce`,
model state `a4616f9045cfe250f188595fba591dd4c157cb613e331ceaa89aa889f8a79e95`.

| category (test, strict) | candidate | base chat | base few-shot (same items) |
|---|---|---|---|
| instruction following, in-distribution (500) | **0.648** | 0.000 | 0.114 (+0.534, CI +0.488…+0.578) |
| instruction following, held-out families (272) | 0.011 | 0.000 | — |
| JSON (500; held-out shapes 200: 0.260) | 0.704 | 0.000 | 0.540 → 1.000 on few-shot items |
| math (1,100; arithmetic 0.58–0.82, GSM8K 0.012) | 0.370 | 0.000 | 0.070 → 0.154 on few-shot items |
| facts (176; continents 0.88, capitals 0.05) | 0.301 | 0.000 | 0.233 (n.s.) |
| Python (864; program output 0.16, MBPP 0.03, HumanEval 0.012) | 0.057 | 0.000 | — |
| uncertainty (300) / false abstention on facts | 0.993 / 0.074 | 0 / 0 | — |
| hierarchy (221) | 0.425 | 0.000 | — |
| injection (300) | 0.960 | 0.000 | — |
| protocol-valid turns (4,233) | 0.957 | 0.000 | — |
| validation bpb (full split, fp32) | 0.940726 | 0.916381 | limit 0.943872 |

Decision rule (protocol v3): P1, G-PROTO, G-REG, G-RET, G-UNC, G-REC and early stop all pass →
**`PILOT_CANDIDATE_QUALIFIED`**. Generation determinism: a repeated subset matched 200/200 (with
different batch composition).

Independent re-verification (CPU, from the raw token IDs, scorer at `f0399ed`): every per-item score,
the comparison and the decision reproduced byte-for-byte; P1 recomputed (324/500, Wilson LB 0.605,
bootstrap CI +0.488…+0.578); checkpoint lineage, LR rule and recovery evidence confirmed; a random
audit of 60 strict verdicts found 0 scorer errors (95% upper bound ~6%). Retention and dev-loss
recomputation were not repeated (GPU) and rest on the recorded runs.

What this does **not** show, stated plainly:
- general instruction following — held-out instruction families 3/272; held-out JSON 0.260 comes
  entirely from one of three held-out shapes (52/61; the other two 0/139);
- knowledge — country/US-state capitals fell from 11/51 (base few-shot) to 2/51 (exploratory McNemar
  p≈0.012, not registered, would not survive correction); GSM8K 6/500 vs base few-shot 12/500 (n.s.);
- injection resistance in general — the injection items mostly measure transformation accuracy over
  text containing injected instructions (the failures inspected were transformation errors, not
  obedience), and uncertainty items were answered with the single trained phrase;
- coding — 3 of the 17 MBPP/HumanEval passes are wrong code that weak upstream tests accept;
- the "versus base chat" view measures format learning only (the base scores 0 there);
- seed-to-seed variance (one seed) or production readiness.
Evidence: `docs/posttrain/results/sft_pilot_v1/`; per-item outputs on HDD and HF (§13).

## 11. GPT-OSS-20B reasoning distillation

Legality and design: Data `docs/distillation/gptoss_teacher_protocol_v1.md` (local copy of the pinned
Apache-2.0 weights only; hosted outputs excluded; independent verification of answers *and*
intermediate equations; arms from answer-only to concise rationales; prospective criteria; compute
estimates). Teacher artifact hash re-verified (`9d7364f0…3d23d`). Local feasibility batch ($0,
`results/teacher_feasibility_2026-10-13.json`): 30 GSM8K-train prompts, pinned runtime b11270 on the
RTX 2060, Harmony, low reasoning effort, up to 3 candidates — **28/30 verified** against the known answer
(24 on the first candidate), median 6.6 s per candidate, median 25 reasoning / 44 answer words; 24
numeric intermediate equations re-executed with 0 errors (3 algebraic lines skipped). Practical
finding: teacher answers use LaTeX, so the verifier must normalize it (a first, non-normalizing pass
reported spurious errors). Teacher code execution now runs only in the bubblewrap harness. No
distillation training was run; the next step is the arm comparison in the protocol.

## 12. Tests, CI and merges

Train: 460 passed, 1 skipped; ruff/mypy clean. Data: 414 passed locally; CI 403 passed (sandbox
tests run in CI). App: CI all jobs green on `4e07cfa`; merged `e3e86568`, post-merge `main` CI green.
Merge receipts (each merged on its exact reviewed head with green CI; review trail in a PR comment):

| PR | head | merge | post-merge CI on `main` |
|---|---|---|---|
| Juniper-App#45 (text-path guard + advisory fixes) | `4e07cfaf` | `e3e86568` | validation success |
| Juniper-1-Data#9 (dataset v1.2, scorer, runtime client, teacher sandbox) | `f0399ed4` | `b7af9634` | CI success |
| Juniper-1-Train#26 (trainer, harness, protocols, A-02, results, report) | `c097c3a` | `21bddf55` | dispatched CPU CI success (run 38351492832) |

Not merged on purpose: AAA#34/#36, cinqic.com#22 (unchanged blockers), Juniper-1-Train#2 (experimental,
conflicting).

## 13. Preservation, rollback, open blockers

| artifact | HDD (canonical) | Drive | private HF |
|---|---|---|---|
| `j1-sft-v1.1` | `juniper1-data/data/sft/build/j1-sft-v1.1` | 0 differences | `CinqicX/juniper1-corpus` `d17ae8df`, read-back |
| `j1-sft-v1.2` | `…/sft/build/j1-sft-v1.2` | 0 differences | `0bef28c9`, read-back |
| pilot checkpoint | `juniper1-archive/sft-2026-10-12/ckpt/j1-sft-v1-pilot/` (verified) | 0 differences | `CinqicX/juniper1-checkpoints` `10cdd276`, fresh download verifies |
| evaluation records (per-item) | `juniper1-archive/sft-2026-10-12/eval/v1.2` (SHA256SUMS) | — | `2802c620` |

Not complete disaster recovery: the HDD is 96% full (~20 GB free), Drive's OAuth app is in Testing
(7-day refresh-token expiry), and Cinqic-Backup is still WIP. Rollback: `juniper1-base-v1` is untouched;
every SFT artifact has its own identity and lineage; code changes revert by their merge commits.

Open: native App adapter; A-11 owner decision; AAA#34/#36 and cinqic.com#22 (unchanged blockers);
Train#2 (experimental, conflicting); the two registered dataset limitations.

## 14. Next justified actions (by evidence and dependency)

1. Dataset v1.3: fix symbol-word abstention false negatives and MBPP functional twins (cross-execution
   check); add **instruction diversity** — the held-out failure is the main finding — e.g. many more
   paraphrased templates and compositional instructions; protect knowledge (the capitals drop
   suggests the bare-label fact targets or the mix degraded recall: add per-relation regression
   gates and consider retention data); then a new prospective protocol.
2. Native token-ID adapter in Juniper App (tokenizers 0.23.2, protocol port with golden tests,
   748d4225b CPU build, live negative tests) before any App exposure.
3. Seed replication of the pilot recipe (variance) before treating 0.648 as stable.
4. Distillation feasibility follow-up per the teacher protocol (answer-only vs concise-rationale arms).
5. Owner: A-11 pod credentials; Drive OAuth publishing status; HDD capacity.
