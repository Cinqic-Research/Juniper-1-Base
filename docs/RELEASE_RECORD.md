# Release record: Juniper 1 Base v1.0.0 (`juniper1-base-v1`)

Status: **STAGED, PENDING INDEPENDENT REVIEW** (updated when publication completes; see the end of this file).

| Item | Value |
|---|---|
| Owner authorization | Markus Gillyard, written directive of 2026-10-16 (publish Juniper 1 Base on GitHub and Hugging Face, Apache-2.0 for Cinqic-owned artifacts; close Juniper 1 R&D) |
| Model identity | `juniper1-base-v1` = `j1-base-r2/ckpt-000052034-336e5967`; freeze record SHA-256 `ef79b324fff459c8ffce7ac2e1b1542960a254b8383a6e8b19fd20402e92ff55` |
| Source `model.pt` | SHA-256 `71773e46db2511332af97891b6485b04bf414a448dab9c4b050988e0b532649c`, 571,153,211 B |
| Tensor-tree digest | `1fd07c0d0c6c6b355460ab82ebe02530047b5682a4e6a41b9e4f0146323076e4` |
| `model.safetensors` | SHA-256 `9dcf74b5db154951d1edbeea6801c04193facb8efc4d5b8b2796e6e675301952`, 571,107,560 B, float32, 222 tensors, exact round trip |
| `tokenizer.json` | SHA-256 `176a89e2197c24d3f024fbb60ce6f346635e7cba912b3579d93b37d486c9adf1`, 680,825 B |
| Corpus | `j1-base-v0.2.2`; build identity `89b08d5754c46dfd18ec239fdacb21ae899c1cf9d7606392401b5224c54bffc5`; accepted manifest `54a3adb9650c96bf67b44b721386722383caeb774b48b0bf970ad6f9634d4185`; `shards.json` `1cf5e2320ad35e930f84ca0edbf5a9d188a2f524702dc4623e492dab043e3aef` |
| GitHub | `Cinqic-Research/Juniper-1-Base`, tag `v1.0.0` (commit recorded below) |
| Hugging Face model | `CinqicX/juniper1-base` (revision recorded below) |
| Hugging Face dataset | `CinqicX/juniper1-base-corpus-v0.2.2` (revision recorded below) |
| License scope | Apache-2.0: weights, tokenizer, code, manifests, documentation (Cinqic). Third-party training data: own licenses (`THIRD_PARTY_NOTICES.md`, `docs/RIGHTS.md`). Dataset card: `license: other` with `LICENSES.md`. |
| Private origins | Juniper-1-Train `52bd74ca` (main `4d39c0e5` + closure PR #35), Juniper-1-Data `14ce3ffe` (main `542bec3b` + release PR #16); every file in `EXPORT_MAPPING.json` |

## Checks performed before publication

| Check | Result | Evidence |
|---|---|---|
| Conversion: `model.pt` SHA-256, tensor digest, parameter count, safetensors round trip | pass | `scripts/convert_checkpoint_to_safetensors.py` run of 2026-10-16; `MANIFEST.json` |
| Reference loader vs release loader: logits (CPU fp32) and greedy continuations on 5 prompts | identical (max abs diff 0.0; greedy identical); CUDA fp32 vs CPU ≤ 6.9e-5, greedy identical | `evidence/parity_reference_vs_safetensors.json` |
| Package tests (tokenizer layout, no control IDs from text, round trips, byte fallback, KV-cache equivalence, context limit, deterministic sampling) | 24 passed | `tests/` on FLOWBOX, torch 2.8.0 |
| Exported training code equals the checkpoint's code identity | 9/9 files SHA-256 equal | `EXPORT_MAPPING.md` |
| Secrets and private data scan of the public tree | no credentials; one email redacted; workstation paths and pod identifiers remain in historical evidence files (documented) | `EXPORT_MAPPING.md` |
| Clean-environment install and completion from the staged destinations | recorded below | |
| Independent release review | recorded below | |

## Publication log

(filled in as steps complete; times US Eastern)
