# Juniper 1 pretraining preparation — Phase 0 investigation report

Date: 2026-10-08 (times UTC unless noted). Directive: Cinqic engineering directive of
2026-10-08, "Comprehensive Pretraining Preparation". Authority: Juniper 1 Design
Specification v0.5 (A100 SXM 80GB), which carries forward v0.2–v0.4.

## 1. Repository and PR states

| Repository | Default head | Open PRs (head) | Relevance |
|---|---|---|---|
| Juniper-1-Data (private) | `11f096f6a6df` | #1 `data/j1-base-v0.1` @ `445162cf530b`, CHANGES REQUIRED (independent review comment, 2026-10-08) | corpus, governance, tokenizer stage |
| Juniper-1-Train (private) | `5f1ad1f09e8e` | #1 `opus/a100-qualification` @ `a14b3b1e2bb4`, PARTIALLY QUALIFIED, unreviewed | A100 harness, checkpointing, recovery |
| Juniper-App (public) | `11225ba7588e` | none | `juniper-tool-protocol-v1`, OpenAI-compatible runtime boundary |
| AAA (public) | `439f39bc3ae0` | #34 `opus/juniper1-erudition` @ `452856277028` | evidence methodology only; not mixed into the LM core |
| Juniper-LM-1.1 (private) | `6165652fb3b1` | none | GPT-OSS-20B teacher/reference qualification |
| Juniper-LM-1, Juniper-Reference-4B (archived) | `a166627f70ee`, `484fc3ec3289` | none | historical |
| Cinqic-Calculator, cinqic.com (public) | `15b765cf98d9`, `33ded1e0f3fd` | cinqic.com #22 | standards / publication policy |

No repository beyond these nine exists in the organization.

**Uncommitted work found.** The Juniper-1-Data working tree held 34 modified and 4 new
files (+2,347/−433 lines) from an earlier 2026-10-08 remediation session (process v2
journals, layered contamination matching, v0.3 compliance record, owner-decision brief,
read-only audit scanner, build id `j1-base-v0.2`). It was preserved byte-for-byte to
`juniper1-archive/preserved-worktrees/juniper-1-data-uncommitted-20261008T193808Z/`
(`tracked.patch` sha256 `b7a1e6b8…`, `untracked.tar` sha256 `3f56a431…`). Its 187 tests,
ruff, and mypy pass (one file needed `ruff format`). It was then committed as `14afa01` on
the new branch `opus/pretraining-prep`.

## 2. Implementation inventory and reuse

| Component | Location | State | Reuse decision |
|---|---|---|---|
| Acquisition (pinned HF revisions, LFS SHA-256, Range resume) | Data `acquire.py` | reviewed, 13 manifests match | reuse unchanged |
| Process/filters/redaction, process v2 journals | Data `process.py` | v2 uncommitted→committed `14afa01` | reuse; rebuild corpus as `j1-base-v0.2` |
| Dedup (exact + MinHash), selection, splits, finalize | Data | reviewed; memory finding J1-07 open | reuse; measure peak RSS in the rebuild |
| Exclusion index (13-gram + exact-short) | Data `exclusions.py` | index `6bc49448…` | reuse; coverage limit stays documented |
| Accepted-corpus audit scanner | Data `audit_scan.py` | full v0.1 scan complete | reuse for v0.2 verification |
| Measurement tokenizers | Data `measure-*` | provisional only | **not** the Juniper tokenizer |
| Model `j1-core-28x640` | Train `j1bench/model.py` | 148,315,264 params @ 32,768 vocab | reuse; generalize to a model family; fix open items below |
| Data loaders | Train `j1bench/data.py` | synthetic + flat memmap, no shuffle, no doc masks | extend: shard manifest, shuffle, doc boundaries |
| Checkpoint manager | Train `j1bench/ckpt.py` | fault-tested on A100 | reuse unchanged |
| Training loop | Train `j1bench/train.py` | qualified on synthetic data | extend: param groups, real data, eval by domain |
| FLOWBOX pull replicator | Train `j1bench/replicate.py` | tested | reuse; add an HF destination with receipts |
| App tool boundary | App `src-tauri/src/tools.rs` | `juniper-tool-protocol-v1`, host-authored results | integration contract only |

## 3. Outstanding defects and gaps

1. **Corpus identity.** The historical v0.1 corpus (`13acc9a4…` processing identity) was built before
   the review fixes. The corrected pipeline must rebuild under a new build id (spec v0.3: an
   old corpus must never be relabelled as reproduced).
2. **Contamination.** The full read-only scan of v0.1 with the improved index found **97
   accepted documents** that match protected benchmark items (MMLU 74, ARC 16, MATH 5, MBPP 3,
   JuniperBench-Code 2). v0.1 is therefore unsuitable for evaluation-clean claims. The v0.2
   rebuild applies this index during processing. Internal-set matchability remains low
   (2.1%–39.3%), which is a documented coverage limit.
3. **Source-use decisions.** These were PENDING at the start of this session. The owner
   approved all four for training input in this session on 2026-10-08 (recorded in
   `configs/owner_decisions.toml`). Weight-release treatment and dataset redistribution
   remain PENDING.
4. **Data-stage recovery cadence (J1-13).** Dedup, selection, finalization and counting
   lack 15-minute transactional journals.
5. **Harness gaps for real training:**
   - Weight decay applies to norms and embeddings.
   - There are no document-boundary attention masks and no shuffle.
   - Evaluation is a single scalar with no domain breakdown.
   - Initialization constants are hard-coded.
   - The QK-Norm form and RoPE θ are not frozen.
   - No inference or latency path exists.
6. **Storage durability (G10).** The global volume is beta, can't be attached by API, and has
   no independent cloud backup. The FLOWBOX archive has no backup.
7. **FLOWBOX GPU unavailable.** `nvidia-smi` reports "Driver/library version mismatch" (NVML
   595.99), so the loaded kernel module is older than the user-space library. This needs a
   reboot (owner action). Until then, FLOWBOX GPU latency qualification is blocked and CPU
   work is unaffected.

## 4. Conflicting requirements

| Conflict | Resolution path |
|---|---|
| v0.4/v0.5 require two pod-independent durable cloud destinations plus an HDD archive backup; the directive allows only FLOWBOX HDD + GitHub + Hugging Face | Versioned storage revision (spec v0.6 proposal): HF private repos as the independent cloud copy; state the degraded protections and stop conditions explicitly |
| 500-update / 15-minute checkpoint cadence vs. ~30 s synchronous saves on the global volume | Large global batch makes the 15-minute trigger dominate (~3%); pilots below 15 min save at stage end |
| Spec "3B accepted tokens counted with frozen tokenizer" vs. provisional 3.35B measure-32k | Recount after tokenizer freeze; do not trim on provisional counts |
| Directive's "~64 reserved specials" vs. App protocol tokens | Specials include protocol tokens; the total vocabulary count is stated inclusive |

## 5. Storage and compute constraints (measured 2026-10-08)

| Resource | Observed |
|---|---|
| FLOWBOX HDD `/dev/sda1` ext4 at `/media/cinqic/Cinqic Storage` | 458 GB, 218 GB used, **217 GB free**. Juniper data 148 GB (raw 95 GB, work 48 GB, accepted 4.9 GB). The archive holds 5.6 GB |
| FLOWBOX root NVMe | 234 GB, 76 GB free (encrypted home) |
| FLOWBOX RAM | 14 GiB usable, 1 GiB swap; heavy jobs run under `MemoryMax=12G` |
| Hugging Face | user `Cinqic` (org `CinqicX`), free tier, no repositories yet; `huggingface_hub` 1.33.0 with `hf-xet` 1.7.0 |
| GitHub | private repos 781 KB / 97 KB; no LFS in use |
| RunPod | pod `4i13u1p447t6e7` EXITED, A100-SXM4-80GB, $1.79/h GPU (+~$0.139/h container disk while running), image `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`; global volume `653478ff…` holds ~9.5 GB |
| RunPod spend baseline | day bucket 2026-10-08 = **$3.5501** at 19:40Z (all prior qualification). This assignment's cap is **+$10.00** over this baseline |

## 6. Workstream dependencies

```
owner source decisions ──> v0.2 rebuild ──> gates/verify ──> tokenizer sample ──> tokenizer freeze
                                                                   │                    │
                                     protocol v1 (App reconciliation) ──────────────────┤
                                                                                        v
model family + tests ──────────────> recount + token-ID shards ──> HF upload ──> A100 pilots
storage tool + failure tests ──────> HF/HDD receipts ─────────────────────────────┘     │
                                                                                         v
                                   scaling report, cost model, readiness matrix, handoff
```

## 7. Prioritized implementation plan

1. **Data (FLOWBOX CPU, running now):** rebuild `j1-base-v0.2`, then run gates, verify, the full
   contamination audit, and the v0.1→v0.2 difference report.
2. **Definition and governance docs:** the model definition and the storage-design revision
   proposal (v0.6).
3. **Storage tool:** a content-addressed manifest; HF upload and download with SHA-256 receipts;
   failure tests.
4. **Tokenizer:**
   - Draw a deterministic train-split sample from v0.2.
   - Train byte-level BPE candidates at 24K, 28K and 32K, each including 64 reserved specials.
   - Evaluate them, write the `juniper-chat-v1` protocol, and freeze.
5. **Model family:**
   - Use parameterized shapes.
   - Exclude norms, embeddings and other 1-D parameters from weight decay.
   - Add a document-mask option.
   - Write architecture tests and a CPU inference/latency benchmark.
6. **Shards:** recount the corpus, write uint16 token-ID shards with packing and shuffle state,
   and benchmark the loader.
7. **A100 within $10:**
   - Restore from HF only (recovery from external artifacts).
   - Run real-data throughput.
   - Train the scaling family at equal tokens.
   - Run a short final-size probe.
8. **Reports:**
   - Scaling analysis and cost model.
   - Evaluation framework.
   - Machine-readable readiness matrix P01–P20.
   - Updated specification draft and handoff.
   - PRs left open for independent review.
