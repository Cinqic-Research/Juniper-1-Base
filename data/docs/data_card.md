# Data card: Juniper 1 base-pretraining corpus `j1-base-v0.2.2` (launch), `j1-base-v0.2` (parent) and `j1-base-v0.1` (historical)

## Launch corpus: `j1-base-v0.2.2` (added 2026-10-11, review finding D-04)

Status: **accepted launch input for base pretraining (2026-10-10).** Successor of `j1-base-v0.2.1`
(itself a successor of `j1-base-v0.2`) derived by removing 7,108 documents: 7,008 under the F9
contamination criteria (HumanEval+ / MBPP+ test fields, MBPP challenge tests) and 100 under the
internal-source exclusion (`docs/source_exclusion_internal_2026-10-10.md`). Tokenized with the frozen
Juniper 1 tokenizer `cand5-j1tok-24k`; full independent recount PASS. Release and redistribution
decisions remain pending. Full record: `docs/qualification_2026-10-10.md` section 6.

| Item | Value |
| --- | --- |
| Build identity | `89b08d5754c46dfd18ec239fdacb21ae899c1cf9d7606392401b5224c54bffc5` (lineage-derived; parent v0.2.1 `e7909e40…765a`, manifest `8d000906…7968`; exclusion list `4bd2e8ef…4ca1`; criteria commit `869a208`) |
| Accepted manifest SHA-256 | `54a3adb9650c96bf67b44b721386722383caeb774b48b0bf970ad6f9634d4185` (`reports/j1-base-v0.2.2/manifest.json`) |
| Documents | **2,959,371** accepted: train **2,954,191**; validation **2,576**; test **2,604** |
| Document tokens (`cand5-j1tok-24k`) | **3,413,019,479** (train 3,407,159,863; validation 3,215,628; test 2,643,988) |
| Stream tokens (+ one EOS per document; token-file bytes / 2) | **3,415,978,850** (train 3,410,114,054; validation 3,218,204; test 2,646,592) |
| Validation scored bytes | 12,311,466 |
| Tokenizer | `tokenizer.json` `176a89e2197c24d3f024fbb60ce6f346635e7cba912b3579d93b37d486c9adf1` (24,576 IDs: 64 control, 256 byte, 24,256 learned); `freeze.json` `08a2d898…edf7` |
| Model-ready shards | `token_shards/j1-base-v0.2.2/cand5-j1tok-24k`, `shards.json` `1cf5e2320ad35e930f84ca0edbf5a9d188a2f524702dc4623e492dab043e3aef` |
| Exclusions from the parent | 7,108 = 7,008 F9 (train 6,995 / validation 7 / test 6) + 100 internal-source; removed train 7,095 / validation 7 / test 6 |
| Recount | `reports/j1-base-v0.2.2/full_recount_cand5-j1tok-24k.json`: PASS, 1,342 files, 2,959,371 documents, 1,484.5 s (`docs/full_recount_2026-10-10.md`) |
| Preservation | HDD canonical root, Hugging Face `CinqicX/juniper1-corpus@6f8c9867`, Drive; `MANIFEST.sha256` `8e1e9fc9…9a71` (18 files); `preservation_receipts.json`, `hf_receipt.json`, notes |

### Composition by source (v0.2.2, all splits; tokens = `cand5-j1tok-24k` document tokens from the recount record)

| Source | Documents | Document tokens | Share of tokens |
| --- | ---: | ---: | ---: |
| fineweb-edu-dedup | 970,513 | 1,079,852,836 | 31.64% |
| stackv2-edu | 547,654 | 585,631,167 | 17.16% |
| wikimedia | 489,622 | 472,888,757 | 13.86% |
| stackexchange | 558,829 | 443,046,046 | 12.98% |
| finemath-4plus | 297,874 | 424,296,121 | 12.43% |
| arxiv-papers | 10,380 | 210,458,332 | 6.17% |
| pressbooks | 43,881 | 106,899,715 | 3.13% |
| libretexts | 36,091 | 78,660,138 | 2.30% |
| oercommons | 3,888 | 8,571,178 | 0.25% |
| python-peps | 639 | 2,715,189 | 0.08% |

The three internal Cinqic documentation sources present in v0.2 are absent from v0.2.2. Byte, domain and
language compositions were not recomputed for v0.2.2; the v0.2 figures below describe the parent and differ
from v0.2.2 by the 7,663 documents removed across the two successor steps (555 + 7,108).

### Known limitations (v0.2.2, in addition to the v0.2 list)

- The frozen tokenizer was evaluated and selected on the v0.2 validation split (2,583 documents), not the
  v0.2.2 launch validation split (2,576): `docs/tokenizer.md`.
- The token shards were produced by the previous `tokshards` writer; the current writer does not reproduce
  `shards.json` bit for bit (`docs/stage_recovery.md`, "Reproducibility of the v0.2.2 token shards").
- The recount record's `phase_a.tokens` is the document-token total; stream tokens are 2,959,371 higher
  (`reports/j1-base-v0.2.2/recount_totals_note.json`).

---

## Parent build: `j1-base-v0.2` (superseded as the training input by `j1-base-v0.2.2`; retained as lineage)

Status: **built 2026-10-08/09 under the corrected pipeline; integrity-verified; training use of
all sources approved by the owner (2026-10-08). Not tokenized with the frozen Juniper 1
tokenizer at the time of this section; the canonical token counts are those of `j1-base-v0.2.2`
above.** Release and redistribution decisions remain pending. The figures in this section describe
v0.2 (2,967,034 documents) and are kept unchanged as history.

| Item | Value |
| --- | --- |
| Build identity | `aa9c4ed9522449521881a64b1dd810006df6249f56d8c1dae8b4e7cc1de8d005` (code `e3f243e`; all `BUILD_MODULES` byte-identical at `2cb7654`) |
| Accepted manifest SHA-256 | `1a99ef755084dcfb1af3b207a476cc009c2bed99694e648fbc2eb8c3a54f37b3` (1,345 shards) |
| Exclusion index | `6bc49448dc1d75b9b8e6c78c98aa4d1f4293bba972e2f7969e048779837c35d5` (13-gram plus exact-short layers) |
| Documents | 2,967,034 accepted: train 2,961,841; validation 2,583; test 2,610 |
| UTF-8 bytes | 13,189,029,618 |
| Tokens (training documents; candidate tokenizers, not yet selected) | 24K 3,435,832,033 · 28K 3,384,119,981 · 32K 3,343,943,878 (`docs/tokenizer.md`) |
| Candidates / near-dup removed / not selected / split-leak removed | 20,196,491 / 154,548 / 17,074,664 / 245 (from 37,282,242 raw documents) |

### Verification (2026-10-09)

- **Gate report lineage:** `reports/j1-base-v0.2/freeze_gates.json` is the
  original pre-fix snapshot. It compares 518 segmented finalization markers
  with 129 logical process units and therefore understates downstream gates.
  The later HDD report is preserved byte-for-byte as
  `reports/j1-base-v0.2/freeze_gates_after_segment_fix_2026-10-09.json`:
  source `/media/cinqic/Cinqic Storage/juniper1-data/data/reports/j1-base-v0.2/freeze_gates.json`,
  SHA-256 `e25e6f1a51fd410358254782b2fc9edd65c7077e4dc3b7116ebad4f633705466`;
  518 of 518 expected finalization markers, 19 PASS, 3 NOT VERIFIED and
  1 NOT APPLICABLE. The 129 logical units and their 518 committed segments
  were independently reconciled against the current source manifests and
  marker identities. The remaining NOT VERIFIED gates cover data-stage
  transactional recovery, canonical A100 checkpoint recovery, and independent
  provisional token-count verification. This later report is a recorded gate
  evaluation, not a new full-corpus scan or an approval of canonical training.
- **`verify` (`reports/j1-base-v0.2/verification.json`):** all 1,345 shards re-hashed with
  matching row counts; zero duplicate doc IDs, text hashes and signature hashes; zero split
  mismatches; no group spans splits; document accounting closes from raw to accepted; the
  sampled contamination screen found 0 hits.
- **Full read-only contamination audit:** all 2,967,034 accepted documents scanned against
  index `6bc49448…`. **0 benchmark hits, 0 canary hits** (`audits/contamination-j1-base-v0.2-627735a29f372e52/result.json`).
  v0.1 had 97 hits under the same index. Matcher-coverage limits for internal evaluation sets
  remain (see Known limitations).
- **Difference from v0.1:** 2,966,819 documents shared; 98 dropped (the contamination
  exclusions); 215 added by selection backfill (FineMath 128, FineWeb-Edu 78, other 9); no
  document changed split; total bytes changed by -0.4 MB.
- **Resources (FLOWBOX):**

  | Stage | Wall time | Peak RSS |
  | --- | ---: | ---: |
  | process (10 workers, 129 units) | 4:11:28 | 0.73 GB per process |
  | dedup | 12:50 | 2.9 GB |
  | select | 2:46 | 3.0 GB |
  | finalize | 20:12 | 1.7 GB |

  Every post-process stage either completes within 15 minutes or commits per-work-unit outputs.

### Composition by source

| Source | Documents | Bytes | Share of bytes |
| --- | ---: | ---: | ---: |
| fineweb-edu-dedup | 970,650 | 4,729,843,831 | 35.86% |
| stackv2-edu | 552,426 | 2,039,516,696 | 15.46% |
| wikimedia | 489,645 | 1,805,986,839 | 13.69% |
| stackexchange | 560,124 | 1,637,993,359 | 12.42% |
| finemath-4plus | 298,990 | 1,469,051,584 | 11.14% |
| arxiv-papers | 10,524 | 680,028,270 | 5.16% |
| pressbooks | 43,899 | 455,313,987 | 3.45% |
| libretexts | 36,125 | 321,755,251 | 2.44% |
| oercommons | 3,890 | 37,642,070 | 0.29% |
| python-peps | 652 | 11,243,154 | 0.09% |
| cinqic-aaa-docs | 52 | 504,722 | 0.00% |
| cinqic-juniper-app-docs | 52 | 124,556 | 0.00% |
| cinqic-calculator-docs | 5 | 25,299 | 0.00% |

Domain shares by bytes: general 21.0%, humanities_social 17.7%, code 14.2%, mathematics 13.1%, computing 10.2%, physical_science 6.8%, health_medicine 6.7%, life_science 5.3%, engineering 5.0%.

### Language composition (directive §3.2)

English function-word share per document: median 0.357, p5 0.074, p1
0.022. The low tail is dominated by code and math. Spelling variety by documents:
US 918,642, UK 242,273, neutral 1,806,119. The corpus is English-first, with
noticeable UK spelling (Wikipedia, Stack Exchange and textbooks). Multilingual content is
incidental and not a product promise.

### Known limitations (v0.2)

- **Contamination coverage.** The audit only finds what the matcher can match. Internal sets
  have low matchable fractions (AAA exams 2.8 %, AAA Erudition 2.1 %, GPT-OSS qualification
  15.3 %, Juniper Reference 8.6 %, App wrapper 11.0 %, JuniperBench-Code 39.3 %), and
  paraphrases are out of scope. Zero hits is not an evaluation-clean certificate for those sets.
- **Quality edge cases from the first review are unchanged by this rebuild:** documents near
  the 200,000-word ceiling, FineMath page chrome, generated-looking snippets, PEP/arXiv metadata
  artifacts, and heuristic kind/domain labels. Filter changes would need a new build identity
  and are deferred to evidence from pilot domain losses.
- **Data-stage recovery.** Process has 5-minute transactional journals. Dedup, select and
  finalize rely on stage atomicity (each completes in under 15 minutes, or commits per
  work-unit markers) rather than in-stage journals (J1-13, partial).
- **Token counts.** No canonical count exists until the Juniper 1 tokenizer is frozen. The
  candidate recounts are recorded in `docs/tokenizer.md`.

---

# Historical: `j1-base-v0.1`

Status: **historical v0.1 artifact; not requalified under the corrected pipeline and
not approved for training.** The first review independently checked its accepted-shard
hashes and split isolation, but the corpus was not rebuilt under the corrected input
identity, raw-byte preflight, or layered contamination index. Values below describe
the historical artifact only. This card separates what was
planned, acquired, accepted, rejected, redistributable, training-only, generated,
verified and uncertain. Every number comes from build outputs; full tables are in
`reports/j1-base-v0.1/corpus_summary.md`.

## Identity

| Item | Value |
| --- | --- |
| Corpus identity | SHA-256 of `manifests/j1-base-v0.1/accepted_manifest.json`: `6aaf450f94a42f8ad0630a1b937cc44f09671a72e5b7e21c0cdd9b5399d6d044` (lists all 357 accepted shards with SHA-256 and row counts, the split configuration, the selection summary and the code revision) |
| Processing identity | `13acc9a46f940b393ca950b21fe147911cd3a9307acd7c311f58e7b459ff2e86` (`manifests/j1-base-v0.1/build.json`: processing configuration, source and exclusion registries, exclusion index, processing-module fingerprint) |
| Exclusion index | `63a50e5ab358a26fbd60d3ade3548abfcce90e95666ca82e583ee15c733be0d6` |
| Token measurement | **Provisional.** Byte-fallback BPE measurement tokenizers trained on a 1% sample of the accepted training split: measure-24k `e8d5cc38…`, measure-28k `ac18965d…`, measure-32k `3d85a5c8…` (`manifests/j1-base-v0.1/measurement_tokenizers/`). Not Juniper 1 tokenizers. |

The processing identity above records the original v0.1 build. Review changes bind
future runs to processing scope, stable acquisition content identities, selection,
split, exclusion-source contents, and output-affecting code. The v0.1 corpus was not
rebuilt under those inputs and cannot be treated as current pipeline output.

## Purpose and scope

Tokenizer-independent candidate base-pretraining corpus for Juniper 1. The v0.2
specification is the design editing baseline; supplied v0.3 (8 October 2026) adds
lifecycle checkpoint and recovery requirements. This corpus version is v0.1.
The initial target is about 3B accepted tokens; 15B is a hard ceiling, not a target,
and any expansion requires training evidence and an owner decision
(`docs/scope_change_2026-10-08.md`). The accepted normalized text is the corpus. No
Juniper 1 tokenizer was designed or frozen and no token-ID training shards exist.

Other stages (technical continuation, reasoning distillation, agent/instruction SFT,
refinement, evaluation) are separate and not built here (`docs/stages.md`).

## Historical headline numbers

| Quantity | Value |
| --- | --- |
| Accepted documents | **2,966,917** (train 2,961,724; validation 2,583; test 2,610) |
| Accepted text | 13.19 GB UTF-8 (4.9 GB zstd Parquet) |
| Accepted tokens, all splits (provisional) | **3,442,113,805** (measure-24k), **3,393,020,202** (measure-28k), **3,354,619,004** (measure-32k) |
| Accepted tokens, training split (provisional) | 3,436,225,167 / 3,387,219,280 / **3,348,884,495** |
| Synthetic or model-generated text | none |
| Repeated exposure counted | none (each document once) |

These are historical provisional counts from the v0.1 build and depend on the
measurement tokenizer. They must be recomputed from a requalified corpus and later
under the separately designed Juniper 1 tokenizer. No tokenizer is frozen here.

## Acquired (historical v0.1 manifests; Hugging Face files carried SHA-256, Git inputs were pinned by blob IDs)

| Source | Origin | Revision | Files / GB acquired | Units processed |
| --- | --- | --- | --- | --- |
| FineWeb-Edu (SmolLM-Corpus dedup) | `HuggingFaceTB/smollm-corpus` | `3ba9d605` | 11 / 26.13 | 30 of 99 |
| FineMath 4+ | `HuggingFaceTB/finemath` | `e92b25a6` | 64 / 18.37 | 24 of 128 |
| Wikimedia (English) | `common-pile/wikimedia_filtered` | `0641bb84` | 35 / 21.56 | all |
| Stack Exchange | `common-pile/stackexchange_filtered` | `c0ac7373` | 21 / 16.79 | 11 of 21 |
| Stack-Edu code (Blue Oak permissive) | `common-pile/stackv2_edu_filtered` | `c354dbe8` | 14 / 12.71 | all |
| arXiv (open licenses) | `common-pile/arxiv_papers_filtered` | `033cf7f5` | 8 / 6.04 | all |
| LibreTexts, Pressbooks, OER Commons, PEPs | `common-pile/*_filtered` | see `docs/sources.md` | 4 / 0.33 | all |
| Cinqic documentation (AAA, Juniper App, Calculator) | Cinqic-Research repositories | `439f39bc`, `11225ba7`, `15b765cf` | 136 files | all |

Raw data on disk: 95 GB, kept for evidence-gated expansion and reprocessing. Units
outside the processing scope (`configs/build_scope.toml`) were not processed.
Considered but not admitted, with reasons: `configs/considered_sources.toml`.

## Processing

Selection and per-document license checks -> conservative normalization -> secret
and PII handling -> evaluation decontamination -> domain-aware quality filtering ->
corpus-wide identity / exact / normalized / MinHash-LSH deduplication (license-aware
signatures for code) with priority-ordered retention -> per-source quota selection ->
group-level splits and held-out leakage removal. Decisions: `docs/decisions.md`;
calibration evidence: `docs/calibration.md`, `docs/dedup.md`.

## Accepted corpus by source (provisional tokens)

| Source | Documents | measure-24k | measure-32k | Share (32k) |
| --- | --- | --- | --- | --- |
| FineWeb-Edu | 970,620 | 1,080,728,260 | 1,050,864,178 | 31.3% |
| Stack-Edu code | 552,427 | 601,280,917 | 583,575,584 | 17.4% |
| Wikimedia | 489,643 | 473,865,223 | 459,489,541 | 13.7% |
| Stack Exchange | 560,125 | 445,455,112 | 436,201,115 | 13.0% |
| FineMath 4+ | 298,897 | 429,269,729 | 422,762,906 | 12.6% |
| arXiv | 10,525 | 213,983,533 | 209,144,229 | 6.2% |
| Pressbooks | 43,902 | 107,150,594 | 104,408,255 | 3.1% |
| LibreTexts | 36,127 | 78,791,664 | 76,904,738 | 2.3% |
| OER Commons | 3,890 | 8,611,957 | 8,375,462 | 0.25% |
| Python PEPs | 652 | 2,795,296 | 2,717,054 | 0.08% |
| Cinqic documentation | 109 | 181,520 | 175,942 | 0.005% |

Code: Python 1.12 GB of text, other languages (C, C++, Go, Java, JavaScript,
TypeScript, Rust, Shell) 0.75 GB, Markdown 0.17 GB.

## Mixture (provisional, measure-32k)

| Kind | Share | | Domain (lexicon heuristic) | Share |
| --- | --- | --- | --- | --- |
| prose | 49.5% | | general | 20.5% |
| math | 21.1% | | humanities and social | 16.2% |
| code | 16.0% | | code | 16.0% |
| technical (prose with code or markup) | 13.4% | | mathematics | 15.2% |
| | | | computing | 10.4% |
| | | | physical science | 6.6% |
| | | | health and medicine | 5.8% |
| | | | life science | 4.8% |
| | | | engineering | 4.5% |

Domain labels are a transparent lexicon heuristic (`j1.domains.v1`), not a trained
classifier. Structured formats (documents containing them; overlapping, upper
bounds): Markdown 45.8%, YAML-like `key: value` lines 25.6%, LaTeX 14.9%, tables
5.1%, XML/HTML 3.5%, shell 1.4%, JSON 0.4%. Source concentration: largest source
31.3%, Herfindahl 0.185.

Source shares come from per-source budgets in `configs/selection.toml` (rationale
recorded there), filled from the deduplicated pool: FineWeb-Edu and FineMath in order
of their upstream quality scores (selected FineWeb-Edu score >= 3.02, median 3.33;
FineMath >= 4.03), other sources by a reproducible uniform sample, small curated
sources whole. The budgets are provisional and versioned.

## Rejected and removed material (processed units)

| Stage | Documents | Notes |
| --- | --- | --- |
| Raw documents read | 37,282,242 (158.4 GB) | |
| Selection, license, extraction | 14,724,764 | other wikis and namespaces, meta and non-English sites, Stack-Edu score < 3, generated/vendored files, non-open per-document licenses, empty text |
| Privacy (credentials, SSNs, card numbers) | 12,430 | password assignments 5,357; card numbers 1,952; credentialed URLs 1,732; Google API keys 1,291; private keys 914; JWTs 780; and others |
| Evaluation contamination | 2,068 | MATH 1,163; MMLU 300; MBPP/MBPP+ 201; HumanEval/+ 225; GSM8K 125; HellaSwag 67; ARC 52; BBH 40; JuniperBench-Code 28; GPT-OSS qualification 18; AAA exams 1; canary strings 58 |
| Quality | 2,346,247 (7.6 GB) | top reasons per source in the corpus summary |
| Candidates | 20,196,733 (90.4 GB) | |
| Identity / exact / normalized / near duplicates | 9,001 / 15 / 3,112 / 142,459 | cross-source example: 47,652 FineMath copies of Stack Exchange questions removed in favour of the attributable Stack Exchange copy |
| Not selected (pool beyond budgets) | 17,074,984 | |
| Training documents removed as near copies of held-out documents | 245 | |
| Accepted | 2,966,917 | |

Redactions in admitted documents (values replaced, never logged): e-mail 549,006;
home-directory user names 297,317; phone numbers 66,673; public IPv4 53,849.

Raw and rejected volume in tokens is not measured (rejected text is not kept);
estimates from each source's accepted bytes per token are in
`reports/j1-base-v0.1/rejected_token_estimates.json` and are labelled estimates.

## Licensing and redistribution

| State | Sources | Meaning for Cinqic |
| --- | --- | --- |
| APPROVED | Cinqic documentation, Python PEPs | Cinqic-owned (Apache-2.0) or public domain / CC0 |
| APPROVED_WITH_ATTRIBUTION | Wikimedia, Stack Exchange, arXiv (open licenses), LibreTexts, Pressbooks, OER Commons | Upstream documents include CC BY-SA material. Training use APPROVED by the owner on 2026-10-08; weight-distribution treatment is PENDING; Cinqic does not redistribute corpus text. |
| APPROVED_WITH_ATTRIBUTION | Stack-Edu code | Common Pile filtered release reports permissive file licenses; training use APPROVED by the owner on 2026-10-08 with these provenance, file-level detection and opt-out limits recorded; release treatment PENDING. |
| TRAINING_ONLY_NO_REDISTRIBUTION | FineWeb-Edu, FineMath | Training use APPROVED by the owner on 2026-10-08 under the recorded ODC-By compilation and Common Crawl terms; page copyright remains with authors; release treatment PENDING. |

No clearance for distributing Juniper 1 weights under Apache 2.0 is claimed by this
corpus record. Cinqic does not redistribute corpus text. Required attributions are carried in each
document's `meta` (URL, authors, license) and summarized in `docs/sources.md`.

## Synthetic and generated data

None. Cosmopedia and other synthetic sets were rejected for base pretraining;
teacher (GPT-OSS-20B) data belongs to separately versioned post-training stages, and
no teacher generation was run under this goal.

## Verification (`reports/j1-base-v0.1/verification.json`)

- This review independently re-hashed all 357 accepted shards and read row counts
  from Parquet metadata: no problems.
- No duplicate doc ids, text hashes or signature hashes among accepted documents.
- Every split recomputed from its group key: 0 mismatches, 0 groups spanning splits.
- Accounting closes: raw = rejections + candidates; candidates = duplicates +
  unselected + leakage removals + accepted.
- Provisional token counts: 1,557 documents re-encoded one at a time with freshly
  loaded tokenizers reproduce the batched counts for all three measurement tokenizers.
- The historical report records 5,793 accepted documents re-scanned with a pure-Python
  13-gram implementation: 0 flagged in that sample. This is sample evidence only.
  The rebuilt layered exclusion index has separate item-level coverage results in
  `docs/spec_compliance_v0.3.md`; the complete read-only scan against these preserved
  v0.1 shards is still in progress. Neither result requalifies the corpus under the
  corrected build inputs.
- The committed `tests.json` and `freeze_gates.json` are historical. The original
  gate code could pass a nonzero test exit and unconditionally passed licensing;
  current test evidence is recorded under `test-runs/` and bound to a matching
  commit, input fingerprint and build identity.

## Known limitations

- **Token counts are provisional** until the Juniper 1 tokenizer is frozen.
- Source-use approvals remain pending. The original review did not receive the full
  v0.2 specification. The supplied v0.3 revision adds checkpoint requirements while
  retaining unchanged v0.2 requirements; neither revision requalifies this historical
  corpus. See the successor compliance record.
- Corpus verification does not by itself establish training-use authorization.
- Near-duplicate detection (14x8 MinHash-LSH) misses partial copies at Jaccard
  0.4-0.7 (labelled recall 0.68, precision 0.90; `docs/dedup.md`).
- Decontamination requires covering half of one evaluation item's informative
  13-grams; paraphrased or translated benchmark content is not detected, and items
  shorter than 13 words cannot be matched. When a quoted passage is most of an MMLU
  item, documents quoting it are removed (conservative).
- Quality filters are heuristic. Known false positives: the web-spam lexicon on
  Wikipedia articles about cryptocurrency or gambling, list pages labelled
  "not English", conservative credential and card rules (`docs/quality_audit.md`).
- Residual page chrome: FineMath SlideShare/transcript pages (1.3% of FineMath
  documents); "click here" in 0.85% of FineWeb-Edu documents.
- The Common Pile `<EMAIL_ADDRESS>` redaction marker remains in arXiv (70%) and PEP
  (94%) documents; Cinqic's own redaction uses `email@example.com`.
- American English is selected through sources, not enforced per document: 79% of
  documents with spelling evidence lean US; UK-leaning text (Wikipedia, web) remains.
- Domain, kind and format labels are lexicon and pattern heuristics; format shares
  are upper bounds (the YAML pattern matches any `key: value` line).
- Stack Exchange coverage is limited to the 11 processed files of 21 acquired
  (listed in `configs/build_scope.toml`: math, two Stack Overflow shards, superuser,
  unix and the files processed before the scope change); sites in the other files
  are not represented.
- Licensing positions marked for owner review remain open (below).

## Open decisions for the owner

- Training use of FineWeb-Edu, FineMath, CC BY-SA documents and Stack v2 was
  APPROVED by the owner on 2026-10-08 (`configs/owner_decisions.toml`; evidence in
  [`owner_decision_brief_v0.3.md`](owner_decision_brief_v0.3.md)). This is an owner
  decision, not legal clearance. Weight-release license treatment and any dataset
  redistribution remain `PENDING` as `[[release_decision]]` records.
- Revise the provisional per-source budgets after pilot domain losses.
- Any expansion beyond this corpus (15B hard ceiling) requires measured evidence and
  a new owner decision.
- Design and freeze the Juniper 1 tokenizer (separate stage), then produce token
  shards from the accepted text.
