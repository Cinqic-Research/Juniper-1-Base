# Independent review: `j1-base-v0.1` and PR #1

- **Review date:** 2026-10-08
- **Initial PR head:** `d17280efcbb1033ef6880b4d4c461ba2ae611609`
- **Repository:** `Cinqic-Research/Juniper-1-Data`
- **Disposition:** **CHANGES REQUIRED; BLOCKED for training use and merge.**

This is a technical and scientific review record, not a training, tokenizer,
redistribution, or legal approval. PR #1 remains open. The accepted corpus is not
cleared for tokenizer training or pretraining while the owner decisions below remain
pending. No corpus bytes were edited or rebuilt during this review.

## Scope and evidence boundary

The review inventoried the 180-file PR diff and examined the risk-bearing code,
repository tests and configuration, the 8-repository Cinqic-Research inventory, the
current open AAA PR #34 and Cinqic.com PR #22, and the corpus manifests on the mounted
`Cinqic Storage` volume.
The Oct 8 owner directive in the supplied assignment controls. The full v0.2 design
document was not among the supplied files; the available Oct 7 v0.1 DOCX is
superseded and was not treated as current authority. No GPU work, tokenizer training,
model training, public distribution, or cross-repository change was performed.

The selected GitHub identity is `MarkusGillyard`, which GitHub also identifies as
the author of PR #1. GitHub cannot record a formal independent approval from a PR's
author. This report therefore does not claim a formal GitHub approval.

## Corpus facts independently checked

- The committed and HDD accepted manifests are byte-identical; manifest SHA-256 is
  `6aaf450f94a42f8ad0630a1b937cc44f09671a72e5b7e21c0cdd9b5399d6d044`.
- All **357 accepted shards** were independently re-hashed against the manifest.
  There were no hash or Parquet row-count mismatches. The shard files total
  **5,210,910,500 bytes**; accepted UTF-8 text totals **13,189,434,431 bytes**.
- The independently read split totals are **2,961,724 train**, **2,583 validation**,
  and **2,610 test** documents. Full-corpus recomputation found zero duplicate doc
  IDs, text hashes, or normalized-signature hashes; zero split mismatches; and zero
  group keys spanning splits.
- The manifest's candidate accounting closes at **20,196,733 candidates**:
  154,587 deduplicated, 17,074,984 not selected, 245 split-leakage removals, and
  2,966,917 accepted. Token-count manifest entries sum to **3,442,113,805** for
  measure-24k, **3,393,020,202** for measure-28k, and **3,354,619,004** for
  measure-32k. These are provisional measurement-tokenizer totals, not Juniper 1
  token counts; this review did not independently re-encode the full corpus.
- The reported 28k source shares recompute arithmetically from manifest entries:
  FineWeb-Edu 31.33%, Stack v2 code 17.40%, Wikimedia 13.70%, Stack Exchange
  13.00%, FineMath 12.60%, and arXiv 6.23%. These describe composition only; they do
  not establish model capability or an optimal mixture.
- The registered canary strings and exclusion-index identity were independently
  checked across all accepted text: neither canary was present, and the index
  identity matched. The separate benchmark contamination result is only a sample:
  the committed report's 5,793 scanned documents had zero flags.
- Contamination coverage is weak for several internal evaluations. The exclusion
  manifest marks 5,674/5,761 AAA exam items, 139/142 AAA Erudition items, 3,192/3,536
  GPT-OSS qualification items, 962/1,013 Juniper Reference items, 3,051/3,341 App
  wrapper items, and 3,170/4,472 JuniperBench-Code items as unmatchable. This does
  not prove contamination, but zero flags in the sample cannot establish absence for
  most of those strings. Keep evaluation-clean claims blocked until coverage is
  improved and independently checked.
- A deterministic screening sample of **66 accepted documents** covered all 13
  sources, all four assigned kinds, nine domain labels, document-length extremes,
  redaction-bearing examples, and near-duplicate clusters. It surfaced long
  documents close to the 200,000-word ceiling, FineMath page chrome, generated-looking
  instructional snippets, placeholder and metadata artifacts in PEP/arXiv text, and
  heuristic kind/domain labels. This is a screening sample, not a full content-quality
  approval; no rejected-document boundary sample was independently reproduced. The
  sample selects the lowest SHA-256(doc_id) in each source-kind and domain stratum,
  plus the shortest/longest document and one redacted/near-duplicate example per
  source, deduplicates by document id, and sorts by source, kind, id. The sorted ID
  list SHA-256 is `fefcf59b92ebaf880125cf202ab757930037e2ecbd93e47106a9ca050398971d`.
- All 13 active acquisition manifests match their configured source, revision,
  repository, and mechanism. A read-only scan of accepted license metadata found no
  NC, ND, GPL, AGPL, or LGPL substrings. This does not resolve source-owner decisions
  or upstream license-detection limits.

## Findings and remediation

| ID | Severity | Finding and evidence | Review disposition |
| --- | --- | --- | --- |
| J1-01 | HIGH | Explicit `process --sources` bypassed the source admissibility check. A temporary-root reproduction dispatched a `REJECTED` source successfully. | Fixed: explicit processing now fails closed for unknown or inadmissible sources; regression tests cover both. |
| J1-02 | HIGH | Processing identity omitted release scope, acquisition manifests, selection and split settings, and later-stage code. Finalization skipped any existing marker and aggregated every marker, allowing changed inputs or out-of-scope markers to be reused. Partial processing also could target the configured release build. | Fixed in code: build identity now binds those inputs; finalization rejects stale, missing, or identity-mismatched marker inventories; partial source/unit runs require a distinct exploratory build id. Existing v0.1 artifacts were not regenerated. Raw input file bytes are still not re-hashed before processing, so byte-level source integrity after acquisition remains NOT VERIFIED. |
| J1-03 | HIGH | Freeze gates unconditionally passed license exclusion; test gate ignored the suite return code, and smoke gate trusted a boolean. A temporary-root reproduction passed a `REJECTED` source and suite return code 23. | Fixed in code: gates inspect source states, training-use decisions are a distinct gate, test evidence requires zero suite and end-to-end return codes and is bound to commit, input fingerprint, and build identity. Historical reports remain unchanged. |
| J1-04 | HIGH | Internal benchmark contamination coverage is insufficient for a broad “clean evaluation” claim, with 70.89%–98.49% of several internal benchmark strings unmatchable. | Unresolved. The sample result is not treated as a whole-corpus guarantee. Improve source item matching or limit claims and add a quantified coverage gate before evaluation use. |
| J1-05 | HIGH | FineWeb-Edu/FineMath training-use positions, CC BY-SA treatment, and Stack v2 provenance remain pending in the repository's own data card. | Unresolved owner decisions. `configs/owner_decisions.toml` records the pending decisions and the training-use gate reports NOT VERIFIED. No legal conclusion is made. |
| J1-06 | MEDIUM | Broad `Creative Commons - Attribution - *` and `CC BY*` patterns admitted NC/ND descriptors in direct tests. | Fixed: restricted NC/ND markers are rejected before allowlist matching; positive BY and BY-SA controls remain accepted. The scanned current accepted metadata had no such strings; no corpus content changed. |
| J1-07 | MEDIUM | Deduplication, selection, and held-out-index stages retain corpus-scale arrays in memory; resource behavior was not stress-tested at larger sizes. | Unresolved. No destructive/OOM stress run was attempted. Keep the documented memory cap and measure peak RSS before any expansion. |
| J1-08 | NOT VERIFIED | Repository branch protection and rulesets cannot be inspected with the available GitHub API; GitHub returned 403 because the private repository's plan does not expose those controls. | A lightweight CI workflow now checks code and fixtures on PR pushes. Branch-protection and required-check policy remain NOT VERIFIED; consult the live PR checks for the exact current head. |
| J1-09 | NOT VERIFIED | The complete v0.2 design document was unavailable in the supplied materials. | Current-facing docs now refer to v0.2 and the Oct 8 owner directive; v0.1 is explicitly historical. Reconcile against the complete v0.2 document when available. |
| J1-10 | MEDIUM | HF manifest paths were joined directly to raw-data roots, and Git blob IDs from manifests were passed to `git cat-file` without validation. | Fixed in code: upstream paths resolve under the raw root (including symlink checks), and Git blob IDs must be 40- or 64-character lowercase hex. Regression tests cover traversal, absolute paths and symlink escapes. |

## Code changes in this review

- Added fail-closed source selection, restrictive license marker checks, current-input
  test evidence, source-state and owner-decision gates (including required-record,
  duplicate-record, and source-scope checks), and stale finalization-marker refusal.
- Bound release builds to acquisition-manifest fingerprints and output-affecting
  configuration, and require an isolated build id for partial processing.
- Added path traversal and Git object-id validation for upstream acquisition manifests.
- Added negative and positive regression cases for explicit source selection,
  NC/ND license forms, test return codes, and finalization marker inventories.
- Added lightweight GitHub Actions checks for formatting, lint, strict typing, and
  the fixture test suite. The workflow does not download corpus data.
- Corrected the current spec references and source-use wording. The previous
  `tests.json` and freeze report are preserved as historical evidence rather than
  rewritten to appear current.
- No accepted shard, tokenizer, corpus manifest, or training artifact was modified.

## Checks and remaining gates

The review environment used Python 3.12 in the project venv. The pre-change baseline
reported 146 passing tests. Post-change validation passed Ruff formatting, Ruff lint,
strict mypy, and **169 tests**; the separate end-to-end fixture test also passed.
GitHub Actions results are recorded on the live PR for each pushed head. Branch
protection and required-check policy remain NOT VERIFIED because the repository API
returned 403 for those controls.

Merge and downstream use remain blocked until the unresolved high findings are
addressed, the full v0.2 authority is checked, CI passes on the exact final head, and
the owner records the relevant source-use decisions. Tokenizer design may be planned
separately, but training a tokenizer on this corpus is not authorized by this review.

## Deterministic content-screen sample IDs

The ID list below is sorted by source, assigned kind, and document id. Together with
the selection rule above, it identifies the 66 accepted documents screened in this
review without copying corpus text into the repository.

```text
arxiv-papers: c58bc7ff1a94b239403b00fa01475501, d3edeed48e2348ad109827cc7b9c57eb, 2e8b555affbad1546a30b3783436f1e2, dab3b3530b4b68832c884d7faaa88bcd, db1350f6648e153ef77e5e2d79147fa8
cinqic-aaa-docs: 1a377841997672e3e77e87e54e9f436f, 7a3b2463adbac0a625aead12a663e9b0, e7fce9f2173d2adfc89e06b5cb18bc41
cinqic-calculator-docs: 3fb0b6bf659cfb64510feb8cbc4b37e0, 6eeae2546887179985a50cde14b21750
cinqic-juniper-app-docs: 07c997799515424c8bc3ad8f28f415a4, 4d28400b2ecde86dc29b481d2b11c87c, 7c2f14a27380d3dff3d8f367b2f8090c
finemath-4plus: 3988fd005ca8d8f8e41688f2e3d81c15, 64110f87bafc4905dc5f1c2b5980383d, 8822f8d2b1173927437e54adac20454c, f3b59ee8355c2801bcb27b9743ac6c59, fdf47ec0861f5852b40424702d4f6ebb
fineweb-edu-dedup: 3a9b185155f88552c8bb570ac2ffcbef, 0dd63e0c1d6486a94855645921266aaf, 621d4bbfb62797448269a8200df4aea4, 7ccdc292af74739a3a8fab0bcb5ec52d, 869930a33a8d54f87f83e9f043fb8598, 8761a8c1cdaeecc4162fe13182a283c9, bfe13552a4daf484c062dd921b82420d, f1c25ba6c28d99579c1704838afad226, 12310eb5abb5841d8f58335650d20aeb, 3474939f115edf7eab46c705eb335b90
libretexts: c9823f7d2c510c8f34da9c43c47dcbde, 0474e97d0d66ccdfe27ffed6fafeb4d3, 4804962047c3080262bc7ea39c75d24e, 4888fc77c37b0715e822dd572073cfc7, dc7a63d90d76e062ee1ac862082e314a
oercommons: 154276e192ce3b231d90359156ec8a01, 2f6ceb8e87dc26f576ee58f1de8fd280, 7f3936fdaba7bd8ef3f3e2b1b4e7cccf, b5335313f5f145f0bae0c24a65c8cdcb, c19d959207c0f0cd1a8dba1a6e19e376, 2cd00cbe4e727282f2095623fa454929
pressbooks: afdad62ceee00f674e4dee4b0a6d4bfe, 21a3ec02d80d31eda64736b5ef632b64, e6906bba6aba5f499e2836d1912ba9f3, e9345f57c8b9dc894ef9be664cb08e61, 55424ddcb2ef848fbf1d1fcbca4acde9, bcd091964072153b262320802ed531cc
python-peps: 8033bd54f5192a96411398ce7a6fff53, 9660f0968a858c04535149cbdedb8f35, 9abbc3ded6380dc244de48dd5237740f, f5f51b2767275ed5ce857b123b20bc4c
stackexchange: 56dd0b29fc49f4574b1143dfd027ea2e, bd1b725d7c369d853929e009dc86b95c, 45dd2a33a9b581ab8d29e799700db533, 61ba7fc90fd15f59bcc251226ae85208, d0d675c3df93b2699e840390cbe5bebd
stackv2-edu: 00b8ad3bbc41ae0b54f0f9dfa3a44b2a, 0199ef74828ed050c9fb3ec51bb4a9c0, 1fbd1d312fa2ef63f0e7baea62d1faae, ac6f31389ff4b7c18e43cff5200b532b, 4bccfe610141fc1b7c3f0764f795d9ec, 090b35e00abea3a78185068145b56b75
wikimedia: d86a95674f69f5001e638f1938530693, 715a563a5d938cec4fe0a51b78306e18, 7911e958e2db26cd838391d55913f99a, c88c7fab73e81f13e2d6f1ae39f3fd52, 52afa22d9b69724c0a0d3a7321435de0, b99fbbc5ac9c38be2a333f4117a54b90
```
