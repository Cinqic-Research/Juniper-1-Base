# Export mapping

This public repository is a clean export from two private Cinqic repositories, not a visibility flip. The
private repositories hold operational records (cloud pod configuration, funding ledgers, supervision logs,
backup receipts, local paths and private review material) that are not part of the release. Every file under
`training/` and `data/` is listed in `EXPORT_MAPPING.json` with its source repository, source commit, source
path and SHA-256, so provenance can be checked file by file.

| Public path | Source | Commit | Notes |
|---|---|---|---|
| `training/j1bench/` | `Cinqic-Research/Juniper-1-Train` `j1bench/` | `52bd74ca` (branch `closure/juniper1-2026-10-16`, 2026-10-16: main `4d39c0e5` plus the closure record and README status; identical code) | `__init__.py`, `ckpt.py`, `data.py`, `model.py`, `replicate.py`, `store.py`, `timeaudit.py`, `timelog.py`, `train.py` are byte-identical to the checkpoint's code identity at `828edd19` (verified by SHA-256); `frozen.py`, `generate.py`, `sft.py`, `sftdata.py` were added later |
| `training/tools/`, `training/tests/`, `training/qual/`, `training/env/` | same | same | evaluation, selection, contract, freeze-record and SFT decision tools and their tests; pod, launch, funding, deployment and watchdog tooling is not exported |
| `training/docs/` | same | same | freeze record, 2026-10-12 audit and selection reports with raw results, training contract, preparation protocols, architecture-selection evidence, launch and audit reports, post-training protocols and decision records; funding contracts, money ledgers, host/pod evidence, supervision ledgers and sealed-test access logs are not exported |
| `training/results/` | same | same | historical A100 qualification evidence |
| `data/` | `Cinqic-Research/Juniper-1-Data` | `2231c7fd` (branch `release/juniper1-base-status-2026-10-16`, 2026-10-17: main `542bec3b` plus the owner release decisions, rights analysis, README status and the public-dataset builder; identical pipeline code) | pipeline source, configs, schemas, tools, tests, manifests for `j1-base-v0.1`/`v0.2`, reports for all base builds, data-pipeline documentation; SFT/distillation datasets, their manifests and teacher-model code are not exported (closed post-training work) |
| `juniper1/tokenizer.json` | FLOWBOX archive `juniper1-data/data/tokenizer/j1-base-v0.2.2/frozen/cand5-j1tok-24k/` | — | SHA-256 `176a89e2…adf1`, bound in the freeze record |
| `juniper1/model.py` | derived from `j1bench/model.py` | `52bd74ca` | defaults changed to the frozen configuration; docstring updated; otherwise identical |

Modifications made during export: one email address in `training/docs/audit_2026-10-12/findings.json` was
replaced with `[owner-email-redacted]` (flagged `modified` in `EXPORT_MAPPING.json`); `training/README.md` and the
export banner at the top of `data/README.md` are written by the export script. No other exported file was altered. Evidence files keep the absolute
workstation paths (`/home/cinqic/...`, `/media/cinqic/Cinqic Storage/...`) and cloud pod identifiers they were
recorded with; several of them are self-hashed records whose bytes must not change.
