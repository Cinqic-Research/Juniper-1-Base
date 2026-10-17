# Accepted-corpus qualification record — 2026-10-10

Authority: owner work order of 2026-10-10 under the owner-issued v0.7 specification (DOCX SHA-256
`121fcb1d…92ee3`). This record is evidence for the corpus/tokenizer gates; it does not approve canonical
training, release, or redistribution. Evidence files live under the HDD archive
`juniper1-archive/overnight-2026-10-10/evidence/` (indexed in `EVIDENCE_INDEX.json`) and are referenced by name.

## 1. Identities

| Item | Identity | Status |
|---|---|---|
| `j1-base-v0.1` (historical) | accepted manifest `6aaf450f…`; Data PR #1 head `445162cf` | preserved, unmerged, CHANGES REQUIRED; not a training corpus (97 benchmark-matching docs under index `6bc49448…`) |
| `j1-base-v0.2` build | build identity `aa9c4ed9…d8005`; accepted manifest `1a99ef75…7b37` | 1,345 shards; 2,967,034 docs (train 2,961,841 / validation 2,583 / test 2,610); 20,196,491 candidates → 154,548 deduplicated away, 17,074,664 not selected, 245 split-leakage removals |
| v0.2 gate reports | original pre-segmentation `freeze_gates.json`; corrected `freeze_gates_after_segment_fix_2026-10-09.json` `e25e6f1a…5466` | both preserved; 19 PASS / 3 NOT VERIFIED / 1 N/A (129 logical units vs 518 finalization segments reconciled in the data card) |
| `j1-base-v0.2.1` (successor, proposed for training on 2026-10-10; **superseded** by `j1-base-v0.2.2`, section 6) | build identity `e7909e40df202f0fb407369802d4d1d0becc02f9f86a777ef50a840d700c765a` (lineage-derived); manifest `8d00090632bf6ca07f4dd75d3deea89421de823366c8216c6a85c9f003237968`; `successor_of` v0.2 manifest `1a99ef75…`, 555 documents removed (all train), criteria commit `34ce84b` | train 2,961,286 / validation 2,583 / test 2,610; derived by `tools/successor_manifest.py` (parent shards untouched; affected shards rewritten; fail-closed re-check) |
| Proposed model-ready shards (v0.2.1; **superseded** by the v0.2.2 shards in section 6) | `token_shards/j1-base-v0.2.1/cand5-j1tok-24k`, shards.json `cb2e2b53d5b4ce26210bca2367c20ea751362bc81d92294c2e81b7f2cda76ef3`, tokenizer `176a89e2…adf1` | `tokshards.verify` clean; independent count check PASS (`independent_count_check_v0.2.1_24k.json`, 6,325-document unbatched re-encode, full reconciliation). Train 3,433,653,388 document tokens (3,436,614,674 with EOS); validation 3,227,354; test 2,657,081 (unopened) |

## 2. Independent provisional count check (gate "token count independently verified")

- Tool `tools/independent_count_check.py` (no use of `j1data.tokshards` code; reads the shard files).
- v3 candidate shards (24K/28K/32K): **PASS** — full reconciliation of every split (doc sets equal to the
  accepted manifest, offsets contiguous, EOS after every document, by-source sums, stream = document +
  separator tokens = file tokens) and a keyed, stratified, **unbatched** re-encoding of 6,327 documents with
  0 count and 0 ID mismatches (`independent_count_check_v0.2.json`). Train document tokens: 24K 3,435,832,033;
  28K 3,384,119,981; 32K 3,343,943,878.
- Prefix mode self-test: the independently recomputed 700 MB prefix (pack order, byte budget) equals the
  exported v3 28K prefix exactly (156,588 documents, same order; every per-split check true)
  (`count_check_selftest_v3_prefix.json`). That record's top-level `ok` is **false**: it was produced by the
  pre-fix tool, whose `splits_present` check required a test split that prefix exports never contain. The later
  prefix runs (`independent_count_check_v5_prefixes.json`) use the fixed tool and pass.
- This is a sampled encoding check plus full arithmetic/inventory reconciliation, **not** a full independent
  recount. The v3 candidates are now quarantined (section 3); the v5 prefixes are checked by the same tool
  (`independent_count_check_v5_prefixes.json`). A full recount with the frozen tokenizer remains required after
  selection.

## 3. Tokenizer candidates

| Set | Sample | Status |
|---|---|---|
| v1/v2 (`candidates-v1`, `candidates-v2`) | earlier alphabets | historical |
| v3 (`candidates`, frozen `cand-*`) | 89,016 train docs (rate 0.03) | **quarantined**: the sample contained documents adjudicated PROHIBITED (1 under criteria v1, 18 under v2) |
| v4 (`candidates-v4`, `cand4-*`) | sample minus 1 | **quarantined** (adjudication v2 found 17 more) |
| v5 (`candidates-v5`, `cand5-*`) | sample minus 18 (recorded in `sample.json`) | pilot candidates; all hard requirements pass |

Retraining v3 on its own recorded sample was not byte-reproducible (2–6 vocabulary entries differ; tie order in
the trainer), so identity is carried by the frozen bytes, never by "retrain gives the same file".

## 4. Contamination coverage (strengthened layers)

- Existing matcher `j1.contamination.v3` (informative word 13-grams, index `6bc49448…`): full v0.2 scan 0 hits.
  That result only certifies the registered matcher.
- New layers `j1.contamination-layers.v1` (`tools/contamination_layers.py`, index `eddb2f37…`): all-token
  10-grams (≥ 3 alphanumeric tokens) with coverage over non-common grams, common grams (df ≥ 10 in a
  full-corpus first pass) excluded, items whose grams are all common reported separately, plus distinctive
  identifiers. Pass 1 covered all 2,967,034 documents.
- Matchable coverage (layer A items / items): public suites 0.94–1.00 (MBPP 0.81→0.99, MBPP+ 0.78→1.00);
  internal suites remain low: aaa-erudition 0.06, aaa-exams 0.07, app-wrapper 0.13, reference-eval 0.17,
  gptoss-qualification 0.21, juniperbench-code 0.41 (+ identifier layer: 1,416 aaa-exams, 1,166 juniperbench items).
- Controls (`controls_layers_v1_calibrated_b.json`): planted verbatim / case-and-space / re-indented / embedded
  copies detected 182–200 of 200 per suite; 3,000 train documents flagged 8 (0.27 %, candidate true hits).
- Adjudication criteria v2 (`docs/contamination_adjudication_criteria_v2.md`, committed before the corrected
  results): PROHIBITED takes precedence; uncertain → PROHIBITED.
- Pilot inputs (tokenizer sample + 700 MB prefix + validation, 243,331 docs): 37 PROHIBITED
  (`adjudication_pilot_inputs_v2.json`); validation clean.
- **Blind spots (explicit):** items < 10 tokens without an identifier; partial copies below 0.6 coverage;
  paraphrase, translation, renamed identifiers, reordered choices, number formatting; answer-only leakage;
  canaries rely on the v3 matcher; HumanEval+ / MBPP+ extra test fields unprotected (F9). Removal reaches only
  documents with hits: `cinqic-aaa-docs` overlaps the internal AAA evaluation repositories, where Layer A covers
  about 7 % of items (J1-04).
- Full corpus (2,967,034 docs, `scan_full_pass2.json`): 1,423 documents with hits; the corpus-scale rule
  (stated before the result) prohibits 555 (525 Layer A / common-only — largest: MATH 37, MBPP 29,
  JuniperBench-code 29, MBPP+ 16, MMLU 14, HumanEval 8, AAA exams 7 — and 30 with non-generic identifiers, including
  internal AAA run identifiers). All 555 were in the train split.
- Successor identity (review H1, fixed in `06fd5a2`): the first derivation copied the parent's build identity into
  the v0.2.1 manifest, so parent-bound records could have validated for it. The successor identity is now derived
  from its lineage (`j1data.successor`), `gates.evaluate` refuses an inherited identity, and the tool fails closed
  when an excluded ID is absent from the parent. The first derivation and its shards are kept under
  `data/quarantine/superseded-h1-2026-10-10/` (count check `…v0.2.1_24k.superseded-h1.json`). The re-derivation reproduced every accepted shard and every token/doc
  file byte for byte; only the manifest and `shards.json` changed (`rederive_v0.2.1_h1_comparison.json`). Blocks copied from
  the parent (`selection`, `candidates`, dedup statistics, …) are listed in `inherited_from_parent` and describe
  the parent. Unchanged successor shards are hard links to the parent's files; both trees are immutable.
- Successor checks re-run: document sets and splits (group-keyed assignments unchanged; no prohibited ID survives),
  token counts (independent check), packing (`tokshards.verify`, EOS after every document). Selection was not
  re-run: removal is a strict subset deletion and does not change the selection of the remaining documents.

## 5. Findings reconciliation

| ID | Status on 2026-10-10 |
|---|---|
| J1-01, J1-02, J1-03, J1-06, J1-10 | fixed in code (2026-10-08 review); unchanged |
| J1-04 | partially addressed: strengthened layers raise coverage for public suites; internal suites stay limited (section 4) |
| J1-05 | owner training-use approvals recorded 2026-10-08 (`configs/owner_decisions.toml`: 4 APPROVED); release/redistribution PENDING |
| J1-07 | OPEN (no memory stress run) |
| J1-08 | NOT VERIFIABLE (branch protection not exposed on the private plan) |
| J1-09 | superseded: owner-issued v0.7 located and verified |
| J1-11, J1-12 | fixed prospectively; historical markers preserved |
| J1-13 | OPEN: transactional recovery not qualified for dedup, selection, finalization, audits, tokenizer stages, counting and token-shard generation at the 15-minute cadence |
| J1-14 | OPEN until the frozen-tokenizer recount and stage allocation (the full recount is now recorded for v0.2.1 and v0.2.2, section 6 and `docs/full_recount_2026-10-10.md`) |

## 6. `j1-base-v0.2.2`: the launch corpus (added 2026-10-11; review finding D-04)

Sections 1-5 were written while v0.2.1 was the proposed training input. Later on 2026-10-10 two further
exclusion criteria were applied and the successor **`j1-base-v0.2.2`** became the accepted launch input. The
v0.2.1 text above is kept unchanged as history; where it says "proposed for training" it is superseded by
this section. Every value here comes from the committed records under `reports/j1-base-v0.2.2/`.

### Lineage

| Item | Value |
|---|---|
| Parent | `j1-base-v0.2.1`, build identity `e7909e40df202f0fb407369802d4d1d0becc02f9f86a777ef50a840d700c765a`, manifest `8d00090632bf6ca07f4dd75d3deea89421de823366c8216c6a85c9f003237968` |
| Build identity (lineage-derived, `j1data.successor`) | `89b08d5754c46dfd18ec239fdacb21ae899c1cf9d7606392401b5224c54bffc5` (recomputed from `successor_of`: match, `tests/test_successor.py`) |
| Accepted manifest SHA-256 | `54a3adb9650c96bf67b44b721386722383caeb774b48b0bf970ad6f9634d4185` (`reports/j1-base-v0.2.2/manifest.json` is the byte-identical copy) |
| Exclusion list (union) | `excluded_doc_ids_union.json`, SHA-256 `4bd2e8ef7f5f95c1c2541d1e1c6945a563eece4b3726ed6d80c6a743e0af4ca1`, 7,108 unique IDs |
| Criteria commit recorded in `successor_of` | `869a208` (the commit adding the union list and `tools/successor_prefix.py`); the criteria texts are `4c24740` (F9 pre-registration, `docs/contamination_f9_criteria.md`) and `1743277` (`docs/source_exclusion_internal_2026-10-10.md`). The abbreviation is kept as recorded because it is hashed into the identity; new derivations record full SHAs (review D-08) |
| Derived by | `tools/successor_manifest.py` from the Data worktree on `opus/autonomous-2026-10-10` (archive `stage/derive_v022.sh`); parent shards untouched, affected shards rewritten, fail-closed re-check (`successor_lineage.json`: `exclusions_absent_from_successor` true over 2,959,371 documents) |

### Exclusions (7,108 = 7,008 + 100)

- **F9** (`reports/contamination_f9/prohibited_f9.json`, index `7378a0f9…3e8b`, criteria commit `4c24740`): 7,008
  unique documents (train 6,995 / validation 7 / test 6) whose text matches HumanEval+ / MBPP+ test fields or MBPP
  challenge tests under the pre-registered rule.
- **Internal Cinqic sources** (`reports/j1-base-v0.2.2/excluded_internal_source_doc_ids.json`, criteria commit
  `1743277`): 100 documents from `cinqic-aaa-docs`, `cinqic-juniper-app-docs` and `cinqic-calculator-docs` (source
  exclusion; these sources are absent from v0.2.2).
- Intersection of the two lists: 0; union = 7,108 = `successor_of.excluded_documents`; `removed_by_split`
  train 7,095 / validation 7 / test 6 (every ID present in the parent, none in the successor).

### Counts

| Quantity | Value |
|---|---|
| Documents | **2,959,371**: train **2,954,191** / validation **2,576** / test **2,604** |
| Document tokens (`cand5-j1tok-24k`, no separators) | 3,413,019,479: train 3,407,159,863 / validation 3,215,628 / test 2,643,988 |
| Stream tokens (document tokens + one EOS per document = token-file bytes / 2) | **3,415,978,850**: train 3,410,114,054 / validation 3,218,204 / test 2,646,592 |
| Validation scored bytes (Train rule: control 0, byte-fallback 1, learned len(utf-8)) | **12,311,466** (= UTF-8 bytes of the 2,576 validation texts; 7,371 byte-fallback tokens, 2,576 EOS; recomputed from the HDD validation token file and the accepted texts by the 2026-10-10 review and the 2026-10-11 re-review, not from a committed record; the Train launch manifest and validation_record.json carry the same value) |
| Tokenizer | frozen `cand5-j1tok-24k`, `tokenizer.json` `176a89e2197c24d3f024fbb60ce6f346635e7cba912b3579d93b37d486c9adf1`, `freeze.json` `08a2d898…edf7` (evaluated on the v0.2 validation split, `docs/tokenizer.md`) |
| Model-ready shards | `token_shards/j1-base-v0.2.2/cand5-j1tok-24k`, `shards.json` `1cf5e2320ad35e930f84ca0edbf5a9d188a2f524702dc4623e492dab043e3aef` (7 train token files, 1 validation, 1 test, 3 doc indexes) |

The recount record's `phase_a.tokens` (3,413,019,479) is the **document**-token total; summaries must say which of
the two totals they quote (`reports/j1-base-v0.2.2/recount_totals_note.json`, review D-01).

### Full independent recount: PASS

`reports/j1-base-v0.2.2/full_recount_cand5-j1tok-24k.json` (schema `j1.full-recount.v1`, record SHA-256
`ac29e08120980e677717191fc40cb39706fc796ead65da5bb50c7470ed51dbea`): **PASS**, 0 errors, 1,342 accepted files,
2,959,371 documents re-hashed and re-encoded one per call with the frozen tokenizer bytes, every document
round-trips, no control ID from text (22 documents with literal control spellings encode as ordinary bytes under
`jtok.load`; a plain load would turn all 22 into control IDs), pure-Python BPE on 29,779 sampled documents with 0
mismatches, pack order / offsets / EOS positions / per-document ID hashes verified against the shards, token-file
sizes and SHA-256s equal to `shards.json`; 1,484.5 s on FLOWBOX. Details: `docs/full_recount_2026-10-10.md`.

### Comparison inputs

`reports/j1-base-v0.2.2/prefix_18b_vs_official.json`: the 18B-byte train prefix exported from the v0.2.1 shards
with the union exclusions (`tools/successor_prefix.py`) equals the head of the official v0.2.2 train stream, and
the validation token file and doc index are byte-identical (four booleans, all true). That v1 record does not name
its inputs (review D-11); the tool now writes a self-describing record (`j1.successor-prefix-check.v2`).

### Preservation receipts

`reports/j1-base-v0.2.2/preservation_receipts.json` (`MANIFEST.sha256` SHA-256
`8e1e9fc9aad246842a634830286f4c4f1cb76e70578765593b7885a0b7319a71`, 18 files):

- HDD canonical root `/media/cinqic/Cinqic Storage/juniper1-data/data`: verified; 17 of the 18 MANIFEST files live
  there, `accepted/excluded_doc_ids_union.json` exists only in the archive staging payload as a byte-identical
  copy of `excluded_doc_ids.json` (`preservation_receipts_note.json`, review D-05).
- Hugging Face `CinqicX/juniper1-corpus` commit `6f8c986783c8662485748fa0153bae5b2e39aeaa`: upload, fresh download
  and SHA-256 of every file; the per-file readback receipt is committed as `hf_receipt.json` (19 entries, all
  `readback_ok`; hashes equal `MANIFEST.sha256`, `tests/test_v022_records.py`; review D-12).
- Google Drive `cinqic-drive:juniper1/data/j1-base-v0.2.2`: `rclone check` against server hashes, `MANIFEST.sha256`
  last.
- Source permissions: `source_permissions.json` (every v0.2.2 source training-approved; release decisions PENDING).

### Status

v0.2.2 is the accepted launch input for base pretraining under the owner work order of 2026-10-10. This record
remains evidence for the corpus/tokenizer gates; it does not approve release or redistribution. J1-13 (stage
recovery) is PARTIAL (`docs/stage_recovery.md`); the v0.2.2 token shards were produced by the previous
`tokshards` writer and their `shards.json` is not bit-reproducible by the current one (review D-03).
