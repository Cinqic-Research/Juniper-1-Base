# Training source and evidence (export of Juniper-1-Train)

This directory is an allow-listed export of Cinqic's private `Juniper-1-Train` repository at commit
`52bd74ca` (main `4d39c0e5` plus the 2026-10-16 closure record; see `../EXPORT_MAPPING.md`). The `j1bench` package files that the frozen checkpoint binds by
hash (`__init__.py`, `ckpt.py`, `data.py`, `model.py`, `replicate.py`, `store.py`, `timeaudit.py`,
`timelog.py`, `train.py`) are byte-identical to the code identity recorded in the checkpoint manifest
(`code_identity.git_revision` `828edd19`, package SHA-256 `81a3a277…204e99`).

| Path | Purpose |
|---|---|
| `j1bench/model.py` | the model (identical mathematics to `../juniper1/model.py`; defaults here are the earlier 28×640 candidate, so always pass the frozen configuration) |
| `j1bench/data.py` | `TokenStream`: the frozen-corpus reader (EOS-separated uint16 shards, keyed-hash document order, seeded window permutation) |
| `j1bench/train.py` | training loop (BF16 autocast, AdamW, accumulation, cosine schedule, full-state checkpoints, exact resume) |
| `j1bench/ckpt.py` | manifest-last checkpoint publication and verification |
| `j1bench/frozen.py` | `load_frozen_base`: loads the base only by its freeze record |
| `j1bench/generate.py`, `j1bench/sft.py`, `j1bench/sftdata.py` | post-training harness (closed stage; kept for the record) |
| `tools/verify_bpb.py` | bits-per-byte evaluation under the registered protocol |
| `tools/audit_base_*.py`, `tools/build_freeze_record.py` | the 2026-10-12 audit scorers and the freeze-record builder |
| `docs/freeze/juniper1-base-v1.json` | the self-hashed freeze record |
| `docs/audit_2026-10-12/` | independent audit, selection protocol and report, raw results |
| `docs/contract/training_contract_v2.json` | every hyperparameter and identity of the recorded run |
| `docs/prep/`, `docs/autonomous_2026-10-10/selection_18b/` | design notes, evaluation framework, architecture-selection protocol and evidence |
| `docs/fable_2026-10-10/` | launch report, audit report, continuation package, repair log |
| `docs/posttrain/` | post-training protocols, final reports and decision records (all closed) |
| `results/` | historical A100 qualification evidence |
| `tests/` | CPU tests for the model, data reader, checkpoints, resume, generation and SFT harness (the contract-builder test is not exported: it reads a private funding record) |

Run the tests from this directory: `python -m pytest -q tests` (needs `torch`, `numpy`, `pyarrow`, `tokenizers`;
versions in `env/flowbox-venv.lock.txt`). Operational tooling for the cloud pods (bootstrap, controllers,
watchdogs, launch guards, funding ledgers) is not exported; it is not needed to train or evaluate the model.
