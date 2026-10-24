> Copy of the closure record in the private Juniper-1-Train repository (PR #35). Relative paths such as `docs/...` refer to that repository. Most of those files are exported here under `training/docs/` and the Juniper-1-Data paths under `data/`, but some, such as the funding contracts, were not exported ([export mapping](EXPORT_MAPPING.md)). For current dataset availability see [DATASET.md](DATASET.md): FineWeb-Edu, FineMath and Stack v2 Edu are all published as reconstruction manifests only.

# Juniper 1 R&D closure record (2026-10-16)

Owner decision: on 2026-10-16 Markus Gillyard decided that continued Juniper 1 research and post-training no
longer justify their time and cost, directed that **Juniper 1 Base** be finished and published on GitHub and
Hugging Face under Apache-2.0 for the Cinqic-owned artifacts, with the dataset and reproducibility materials
published to the extent the source terms permit, and that Juniper 1 R&D be closed accurately. This record is
the single closure statement for the training project; it links the data repository and the public release.
Frozen protocols, contracts and historical results are unchanged by it.

## What was completed

| Item | Identity |
|---|---|
| Released base | `juniper1-base-v1` = `j1-base-r2/ckpt-000052034-336e5967` (freeze record `docs/freeze/juniper1-base-v1.json`, record SHA-256 `ef79b324fff459c8ffce7ac2e1b1542960a254b8383a6e8b19fd20402e92ff55`) |
| Pretraining | one epoch, 52,034 updates × 65,536 targets = 3,410,100,224 targets of `j1-base-v0.2.2`; eager BF16 on one A100; 52,025 s active training; validation 0.916381 bpb fp32 (full split, re-measured 2026-10-12) |
| Model | `j1-wide-20x768`, 142,771,456 parameters, 2,048-token qualified context; tokenizer `cand5-j1tok-24k` (`176a89e2…adf1`) |
| Corpus | `j1-base-v0.2.2`, build identity `89b08d57…bffc5`, accepted manifest `54a3adb9…4185`, shards.json `1cf5e232…3aef` |
| Public release | GitHub `Cinqic-Research/Juniper-1-Base`; Hugging Face model `CinqicX/juniper1-base`, dataset `CinqicX/juniper1-base-corpus-v0.2.2`; release record in the public repository (`docs/RELEASE_RECORD.md`) |

## Post-training outcomes (all closed; nothing promoted)

| Stage | Record | Outcome |
|---|---|---|
| SFT pilot `j1-sft-v1.2`, protocol v3, seed 1 | `docs/posttrain/results/sft_pilot_v1/decision.json` | `PILOT_CANDIDATE_QUALIFIED` (research candidate, non-production) |
| Replication seeds 2, 3 | `docs/posttrain/results/sft_replication_v1/` | `PILOT_CANDIDATE_QUALIFIED`, consistent with seed 1 |
| SFT v2 on dataset v1.3 | `docs/posttrain/final_report_2026-10-14.md`, `results/sft_v2_v1/` | `FAILED` (arithmetic, JSON, injection, hierarchy regressions) |
| SFT v2.1 diagnostic | `docs/posttrain/results/sft_v21_v1/diagnostic/decision.json` | `STOP_FAILED` |
| SFT v2.2 diagnostic | `docs/posttrain/results/sft_v22_v1/diagnostic/decision.json` | `STOP_FAILED` |
| Distillation v2 (arms A1-A3, control C0) | `docs/posttrain/results/distill_v2_v1/decision.json` | `NO_IMPROVEMENT_DETECTED` in every arm |
| Conversation seed corpus `j1-conv-seed-v1` (Data) | Data `docs/sft/conv/` records, contract amendment 1 | admitted (203 episodes) but never used for training; collection stopped |
| Tool-use data/loader support (Data #10, Train #28, App #47) | draft PRs | superseded: no agent-tool-SFT stage will run on Juniper 1; reusable code preserved on its branches |

## Canceled or superseded plans

`docs/posttrain/later_stages_v1.md` (bounded agentic stage, context extension, AAA requirements),
`docs/posttrain/roadmap_tool_use_v1.md`, the SFT v2.x successor line, any 15B-token continuation, and the
Juniper 1.5 / Juniper 2 discussions are **closed**. They are kept as history; none is an active authorization.
A future Juniper model needs a separate owner decision and project.

## Preserved artifacts

| Artifact | Where |
|---|---|
| Frozen base full state (`model.pt`, `optim.pt`, `trainer.pt`, manifest) and checkpoints 50,000 / 52,000 | FLOWBOX HDD `juniper1-archive/fable-2026-10-10/ckpt/j1-base-r2/`; Google Drive `juniper1/checkpoints/j1-base-r2/`; private HF `CinqicX/juniper1-checkpoints` (29.7 GB for the run); public HF `CinqicX/juniper1-base` (`full-state/`) |
| Corpus `j1-base-v0.2.2` accepted text, token shards, tokenizer, reports | HDD `juniper1-data/data/`; private HF `CinqicX/juniper1-corpus@24ae3b3a`; Drive |
| SFT pilot, replication, v2 (s1-s3), v2.1, v2.2, distillation checkpoints | private HF `CinqicX/juniper1-checkpoints` (finals and recorded intermediates); HDD `juniper1-archive/sft-*`, `assignment-2026-10-13`; v2.1/v2.2 step-500/1,000 intermediates and diagnostic/replication intermediates exist only on the FLOWBOX NVMe (`~/j1post/runs*`, listed in `juniper1-archive/storage-receipts/nvme-cleanup-2026-10-16.json`) |
| Raw training log (`tranche.jsonl`, 52,069 update records, 105 evaluations) | HDD `juniper1-archive/fable-2026-10-10/evidence/continuation_session/tranche/`; curve extract in the public repository `evidence/` |
| Protocols, contracts, decisions, reviews | this repository (`docs/`), Juniper-1-Data (`docs/`), GitHub bundles on the HDD (`github-bundles-2026-10-16/`) |

## Costs and resources

RunPod provider-billed account spend attributable to the Juniper 1 effort, 2026-10-08 through 2026-10-16
(daily billing API): **$89.89** (GPU pods $84.84, pod disks $1.54, CPU watchdog pods $2.29, network storage
$1.22). The pretraining continuation itself settled at about $30.56 under funding contract v7
(`docs/fable_2026-10-10/funding_contract_v7.json`), the tranche at $1.46 under v6; the 2026-10-13 assignment
spent $5.25 of its $10 cap; the 2026-10-12 audit and the 2026-10-16 release work spent $0. All SFT, distillation
and evaluation work ran on the FLOWBOX RTX 2060.

Resource disposition on 2026-10-16: all six Juniper 1 pods are `EXITED` (`dxnsn6g7n5s4fw`, `v247r25kcigu04`,
`pncbyxaf2dt2jl`, `k3mul4cjmiong`, `dxl7jz3zezlz0m`, `4i13u1p447t6e7`); stopped pods have billed nothing since
2026-10-15. The 50 GB network volume `72al92fys2` (`juniper1-exec-2026-10-10`, US-MD-1) remains and bills about
$0.12 per day; its contents are superseded by the HDD/Drive/HF copies. Deleting it is an owner decision (this
closure does not delete volumes). No scheduled jobs, systemd units, timers or restart triggers for Juniper 1
training exist on FLOWBOX (checked 2026-10-16).

## Remaining limitations (carried into the public release)

- Validation numbers are a training-time monitoring set; the test split was never scored.
- 4,096-token context is unqualified. GGUF/llama.cpp text tokenization did not match the tokenizer on the
  historical 200-document check; only token-ID paths were qualified. No Transformers integration is provided.
- The tokenizer was selected on the parent v0.2 validation split (2,583 documents), not the v0.2.2 split (2,576).
- The v0.2.2 token shards' `shards.json` is not reproduced byte for byte by the current writer.
- FineWeb-Edu and FineMath text is not redistributed; those sources are published as reconstruction manifests.

Prepared by Claude Fable 5.1 under the owner's 2026-10-16 directive.
