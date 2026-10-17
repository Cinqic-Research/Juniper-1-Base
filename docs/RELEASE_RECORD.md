# Release record: Juniper 1 Base v1.0.0 (`juniper1-base-v1`)

Status: **PUBLISHED 2026-10-17** (US Eastern). Independent release review: APPROVED_WITH_CONDITIONS at GitHub head `85410b9` (report `Juniper-1-Base-release-review.md`, appendix A.6/A.7); all conditions met as logged below.

| Item | Value |
|---|---|
| Owner authorization | Markus Gillyard, written directive of 2026-10-16 (publish Juniper 1 Base on GitHub and Hugging Face, Apache-2.0 for Cinqic-owned artifacts; close Juniper 1 R&D) |
| Model identity | `juniper1-base-v1` = `j1-base-r2/ckpt-000052034-336e5967`; freeze record SHA-256 `ef79b324fff459c8ffce7ac2e1b1542960a254b8383a6e8b19fd20402e92ff55` |
| Source `model.pt` | SHA-256 `71773e46db2511332af97891b6485b04bf414a448dab9c4b050988e0b532649c`, 571,153,211 B |
| Tensor-tree digest | `1fd07c0d0c6c6b355460ab82ebe02530047b5682a4e6a41b9e4f0146323076e4` |
| `model.safetensors` | SHA-256 `9dcf74b5db154951d1edbeea6801c04193facb8efc4d5b8b2796e6e675301952`, 571,107,560 B, float32, 222 tensors, exact round trip |
| `tokenizer.json` | SHA-256 `176a89e2197c24d3f024fbb60ce6f346635e7cba912b3579d93b37d486c9adf1`, 680,825 B |
| Corpus | `j1-base-v0.2.2`; build identity `89b08d5754c46dfd18ec239fdacb21ae899c1cf9d7606392401b5224c54bffc5`; accepted manifest `54a3adb9650c96bf67b44b721386722383caeb774b48b0bf970ad6f9634d4185`; `shards.json` `1cf5e2320ad35e930f84ca0edbf5a9d188a2f524702dc4623e492dab043e3aef` |
| GitHub | `Cinqic-Research/Juniper-1-Base`, public; tag `v1.0.0` = `85410b9bd68548e7d4d798acf1b70a1c735c86a2`; release page https://github.com/Cinqic-Research/Juniper-1-Base/releases/tag/v1.0.0 |
| Hugging Face model | `CinqicX/juniper1-base`, public, revision `cd4dce2cfa7f8005a6bd7f8f100257352abc116c` (model.safetensors, config, MANIFEST, tokenizer, card, `full-state/` incl. `optim.pt` 1,142,330,507 B sha `04cd1a35…f3ce`) |
| Hugging Face dataset | `CinqicX/juniper1-base-corpus-v0.2.2`, public, revision `9cb35fc710771243ab85cc5ccce914260673fec6` (117 files: 79 parquet = `release_manifest.json` by SHA-256 and row count, card, LICENSES, reconstruction records) |
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
| Clean-environment install and completion from the public destinations | pass: fresh venv, `pip install juniper1[hub] @ git+…@v1.0.0`, fresh HF cache, no credentials; `juniper1 verify` ok; Quickstart completion reproduces the recorded token IDs; tokenizer round trip ok (01:49 AM EDT) | `/home/cinqic/j1release/cleanenv/run.log` (FLOWBOX) |
| Independent release review | APPROVED_WITH_CONDITIONS at `85410b9` after two iterations (first: CHANGES_REQUIRED on documentary/packaging items F-01..F-13, all fixed); artifacts: hashes, exact parity on 7 new prompts, 4,000-string tokenizer fuzz, dataset census | `~/Documents/Fable-Review-Reports-2026-10-16/Juniper-1-Base-release-review.md` |

## Publication log (US Eastern, 2026-10-17)

| Time | Step | Result |
|---|---|---|
| 01:41-01:44 AM | GitHub Actions on `85410b9` (run 39006948316) | package (3.10, 3.12) and training-code jobs: success |
| 01:45 AM | GitHub repository made public; tag `v1.0.0` pushed at `85410b9` | public; tag verified |
| 01:46 AM | GitHub Release v1.0.0 created (REST; the CLI lacked the workflow scope) | https://github.com/Cinqic-Research/Juniper-1-Base/releases/tag/v1.0.0 |
| 01:46 AM | Hub model repo made public; `full-state/optim.pt` uploaded (deduplicated) | revision `cd4dce2c`; all 11 files listed with expected LFS SHA-256 |
| 01:49 AM | Unauthenticated read-back (fresh venv, fresh HF cache) | install, verify, completion, tokenize: pass; GitHub, release, Hub model: HTTP 200 |
| 01:51-02:06 AM | Hub dataset repo made public; staged edition uploaded (first attempt rejected: card `license_link` must be an https URL; fixed) | revision `9cb35fc7`; 117 files, 0 missing, 0 hash mismatches; anonymous download of three parquet files matches `release_manifest.json`; card and LICENSES identical to staging |
| 01:55 AM | Preservation: Git bundle of the public repository, dataset manifest/card/licenses, model files and parity evidence copied to Google Drive `juniper1/release-2026-10-16/` (`rclone check`: 0 differences); HDD record folder `juniper1-release-2026-10-16/release-v1.0.0/` | done |
| 01:55 AM | cinqic.com research page updated (PR #24 merged, Pages deployed; live page shows the Juniper 1 Base section with working links) | done |

This record was committed after the tag; `v1.0.0` points at `85410b9`, which the independent review approved. Later commits on `main` are documentation only.
