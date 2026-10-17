# Dataset release plan: `juniper1-base-corpus-v0.2.2` public edition (proposal, 2026-10-16 08:37 PM EDT)

Proposal by the rights-and-provenance subagent under the owner's 2026-10-16 directive. Nothing here is published; `dataset-redistribution` in `configs/owner_decisions.toml` stays PENDING until the owner records the source-by-source decision (rights table Q-4). Dispositions and evidence: `rights_table_2026-10-16.md`. Notices: `NOTICE_draft.md` section B.

## 1. Layout

One Hugging Face dataset repository (proposed id `CinqicX/juniper1-base-corpus-v0.2.2`; the existing private `CinqicX/juniper1-corpus` stays private and unchanged), ten configs in two families, three splits each (`train`, `validation`, `test`, exactly the v0.2.2 split assignment), plus a `reconstruction/` folder of Cinqic records.

| Config | Family | Source | Docs | Stored text bytes (v0.2.2, measured) | Tokens (cand5-j1tok-24k) | Per-config licence (documented in the card body and LICENSES.md; the Hub schema has only one top-level `license`) | Est. payload on HF (parquet, zstd) |
| --- | --- | --- | ---: | ---: | ---: | --- | ---: |
| `text-libretexts` | text payload | `libretexts` | 36,091 | 321,179,048 (0.321 GB) | 78,660,138 | mixed per row: cc-by-4.0 59.7%, cc-by-sa-4.0 30.5%, public domain 3.1%, cc-by-3.0 2.8%, GFDL (unversioned) 2.0%, cc-by-sa-3.0 1.8%, cc-by-sa-2.5 <0.1% | 0.116 GB |
| `text-pressbooks` | text payload | `pressbooks` | 43,881 | 454,838,985 (0.455 GB) | 106,899,715 | mixed per row: cc-by-4.0 75.5%, cc-by-sa-4.0 14.2%, public domain 6.3%, cc0-1.0 3.9% | 0.164 GB |
| `text-oercommons` | text payload | `oercommons` | 3,888 | 37,466,287 (0.037 GB) | 8,571,178 | mixed per row: cc-by-4.0 86.5%, cc-by-sa-4.0 8.9%, public domain 4.6% | 0.013 GB |
| `text-python-peps` | text payload | `python-peps` | 639 | 10,923,682 (0.011 GB) | 2,715,189 | cc0-1.0 / public domain | 0.004 GB |
| `text-wikimedia` | text payload | `wikimedia` | 489,622 | 1,805,732,717 (1.806 GB) | 472,888,757 | cc-by-sa-4.0 | 0.650 GB |
| `text-arxiv-papers` | text payload | `arxiv-papers` | 10,380 | 667,941,657 (0.668 GB) | 210,458,332 | mixed per row: cc-by-4.0 88.5%, cc-by-sa-4.0 4.9%, cc0-1.0 3.9%, cc-by-3.0 2.0%, public domain 0.7% | 0.240 GB |
| `text-stackexchange` | text payload | `stackexchange` | 558,829 | 1,629,665,160 (1.630 GB) | 443,046,046 | mixed per row: cc-by-sa-3.0 61.1%, cc-by-sa-4.0 37.4%, cc-by-sa-2.5 1.6% | 0.587 GB |
| `manifest-stackv2-edu` | reconstruction manifest | `stackv2-edu` | 547,654 | 1,989,528,445 (1.990 GB) | 585,631,167 | manifest rows: apache-2.0 (Cinqic record); referenced files: per-row `detected_licenses` | 0.230 GB |
| `manifest-finemath-4plus` | reconstruction manifest | `finemath-4plus` | 297,874 | 1,454,527,330 (1.455 GB) | 424,296,121 | odc-by | 0.125 GB |
| `manifest-fineweb-edu-dedup` | reconstruction manifest | `fineweb-edu-dedup` | 970,513 | 4,727,772,527 (4.728 GB) | 1,079,852,836 | odc-by | 0.408 GB |

Totals: payload text 4,927,747,536 bytes uncompressed (4.928 GB), estimated 1.774 GB compressed; manifests about 0.763 GB. Held text (Q-1, Q-2) would add 8,171,828,302 bytes uncompressed (8.172 GB). Compression factors are estimates (0.36 for prose/code parquet-zstd, about 420 B per manifest row); the build step records the real sizes.

Size note: the public edition must be written from the HDD canonical root with about 8 GB free; write per-config, per-split files directly to the HF upload staging on another volume or stream with `huggingface_hub` commits of one config at a time. Do not duplicate the accepted parquet on the HDD.

## 2. Schemas

### 2a. Text payload rows (`text-*` configs)

| Column | Type | Source of value | Note |
| --- | --- | --- | --- |
| `doc_id` | string | accepted parquet `doc_id` | Cinqic 32-hex document id; joins to manifests, exclusion lists and token-shard document indexes |
| `source_id` | string | `source_id` |  |
| `split` | string | `split` | train / validation / test, unchanged |
| `text` | large_string | `text` | Cinqic stored text (modified from upstream; see LICENSES.md Modifications) |
| `sha256` | string | `sha256` | SHA-256 of `text` (UTF-8) |
| `norm_sha256` | string | `norm_sha256` | signature hash used by dedup |
| `n_bytes` | int64 | `n_bytes` |  |
| `license` | string | `meta.license` verbatim | e.g. `Creative Commons - Attribution Share-Alike - https://creativecommons.org/licenses/by-sa/4.0/`, `Public Domain` |
| `license_url` | string | parsed from `meta.license`; empty for Public Domain | CC deed URL |
| `license_spdx` | string | mapped: CC-BY-4.0, CC-BY-3.0, CC-BY-SA-4.0, CC-BY-SA-3.0, CC-BY-SA-2.5, CC0-1.0, LicenseRef-Public-Domain, LicenseRef-GFDL-unversioned (upstream gives no GFDL version) | derived; the verbatim `license` stays authoritative |
| `url` | string | `meta.url` | canonical location; CC attribution URI |
| `title` | string | `meta.title` (absent for PEPs, arXiv, Stack Exchange: left null) |  |
| `authors` | list<string> | `meta.author` (split on `, `) or `meta.authors` (list / list of [name,id] for wikimedia) | wikimedia: names only; stackexchange: profile URLs |
| `upstream_id` | string | `meta.doc_id` | upstream identifier (URL, PEP number, page id, arXiv id, question id) |
| `upstream_file` | string | `meta.provenance` | pinned Common Pile file name and record number |
| `upstream_meta` | string (JSON) | `meta` verbatim | everything else (site, wiki, book_url, subject, institution, submitter, doc_created, all_licenses, removed_boilerplate_lines) |
| `redactions` | string (JSON) | `redactions` | counts of privacy redactions applied |
| `kind / domain / formats` | string / string / list<string> | same columns | heuristic labels, informational |

Not exported: `group_key`, `unit_id`, `dup_cluster_size`, spelling/fw/code-line fractions (internal), `source_key` (identical to `upstream_id` or `url` for these seven sources).

### 2b. Reconstruction manifest rows (`manifest-*` configs; no text)

| Column | stackv2-edu | finemath-4plus | fineweb-edu-dedup |
| --- | --- | --- | --- |
| doc_id / source_id / split / sha256 / norm_sha256 / n_bytes | yes | yes | yes |
| upstream_id | `meta.blob_id` (= `source_key`) | `meta.url` (= `source_key`) | `source_key` (`<urn:uuid:...>`) |
| url | `meta.url` (raw.githubusercontent at `revision_id`) | `meta.url` | `meta.url` |
| locator | `repo_name`, `path`, `revision_id`, `content_id`, `snapshot_id` | `warc_filename`, `warc_record_offset`, `crawl`, `fetch_time` | `dump`, `file_path` (s3://commoncrawl WARC), `date` |
| licence fields | `license_type`, `detected_licenses`, `license`, `gha_license_id` | none exist upstream | none exist upstream |
| upstream_file | `unit_id` prefix (e.g. `stack-edu-0007`) + `meta.provenance` record number | `unit_id` prefix (`train-00001-of-00064`); no record number upstream | `unit_id` prefix (`train-00003-of-00234`); no record number upstream |
| scores | `star_events_count`, `fork_events_count`, `language`, `length_bytes` | `score`, `int_score`, `language_score` | `score`, `int_score`, `language_score`, `token_count` |

`reconstruction/` folder (Cinqic records, Apache-2.0, byte-identical to the preserved copies): `manifest.json` (54a3adb9...), `MANIFEST.sha256` (8e1e9fc9...), `excluded_doc_ids_union.json` (4bd2e8ef...), `successor_lineage.json`, `source_permissions.json`, `shards.json` (1cf5e232...), the thirteen `manifests/j1-base-v0.2/sources/*.acquisition.json` (pinned upstream file inventory: repo, revision, path, git blob oid, size, upstream LFS sha256, URL), `configs/sources.toml`, `configs/selection.toml`, `configs/build_scope.toml`, and the code commit `e3f243e16f877cfb5431a225f398dfa59058f785` reference. The token `.bin` files and `*-docs.parquet` are **not** included (rights table, token shards row).

## 3. Dataset-card YAML metadata (recommendation)

Do not set `license: apache-2.0`: the content is third-party and mixed. Recommended top-level and per-config metadata:

```yaml
license: other
license_name: mixed-per-document-see-licenses-md
license_link: LICENSES.md
pretty_name: Juniper 1 Base corpus v0.2.2 (public edition)
language: [en]
task_categories: [text-generation]
size_categories: [1M<n<10M]
source_datasets:
  - common-pile/libretexts_filtered
  - common-pile/pressbooks_filtered
  - common-pile/oercommons_filtered
  - common-pile/python_enhancement_proposals_filtered
  - common-pile/wikimedia_filtered
  - common-pile/arxiv_papers_filtered
  - common-pile/stackexchange_filtered
  - common-pile/stackv2_edu_filtered
  - HuggingFaceTB/finemath
  - HuggingFaceTB/smollm-corpus
configs:
  - config_name: text-wikimedia
    data_files: [{split: train, path: text-wikimedia/train-*.parquet}, {split: validation, path: text-wikimedia/validation-*.parquet}, {split: test, path: text-wikimedia/test-*.parquet}]
  # ... one entry per config; the per-config licence is stated in the card body table and in LICENSES.md
  # (the Hub schema has a single top-level `license`; per-config licences are documentation, not metadata)
```

Card body must carry: the config/licence table from section 1; the attribution statement (how `authors`/`url`/`license` discharge CC BY-SA 3(a)); the Modifications paragraph; the Common Pile metadata caveat and citation; the ODC-By 4.2 notice for the two manifest configs; a takedown/correction contact; a statement that the Juniper 1 Base model licence (Apache-2.0) does not relicense any row.

## 4. Reconstruction claims

Grounded in `docs/reproduction.md`, `docs/stage_recovery.md` ("Reproducibility of the v0.2.2 token shards"), `reports/j1-base-v0.2.2/*` and `tests/test_end_to_end.py`.

| Claim | Can be made? | Basis |
| --- | --- | --- |
| The exact document set of v0.2.2 (2,959,371 `doc_id`s with split, `sha256`, `norm_sha256`, `n_bytes`) is published for all ten sources | YES | text-* rows + manifest-* rows; `manifest.json` lists every accepted parquet shard with row count and SHA-256 |
| Every upstream input is identified byte-exactly | YES | acquisition manifests: repo, pinned revision, path, git blob oid, size, upstream LFS SHA-256, URL (`manifests/j1-base-v0.2/sources/*.acquisition.json`) |
| A third party can re-derive the stored `text` of a manifest-only row from the pinned upstream record and verify it against `sha256` | YES, with the published pipeline at commit `e3f243e`, per document | `process` is per-document for normalization (`j1.normalize.v1`), boilerplate removal and redaction; `tests/test_end_to_end.py` rebuilds byte-identically on fixtures. Not demonstrated end-to-end on the full v0.2.2 set by an outside party; state it as 'verifiable per row by sha256', not as 'verified' |
| A third party can re-run the whole pipeline and obtain the same accepted set | ONLY WITH ALL INPUTS, INCLUDING THE THREE PRIVATE `cinqic-*` GIT SOURCES | dedup priorities 1-3 are the `cinqic-*` sources and dedup/select are global; v0.2.2 is a successor derived from v0.2 by exclusion lists, so the public path is: accept the published `doc_id` set as the definition, verify by hash |
| The token shards `train-tokens-*.bin` can be regenerated byte-identically from the published text + tokenizer | NOT CLAIMED for the full set | byte identity of token files shown only on a 5% subset of v0.2 train shards plus held-out shards (`docs/stage_recovery.md`) |
| `shards.json` (1cf5e232...) can be regenerated byte-identically | NO | the current writer omits the per-split `seconds` fields; a rerun is equal except for `seconds` and must get a new identity |
| Document-token totals (3,413,019,479) are reproducible from the published text + tokenizer | YES (claimed as a count, not as files) | independent full recount PASS, `reports/j1-base-v0.2.2/full_recount_cand5-j1tok-24k.json`; reproducible by tokenizing every row, including manifest-only rows once their text is re-derived |
| The three `cinqic-*` sources are absent from v0.2.2 | YES | `excluded_internal_source_doc_ids.json`, `successor_lineage.json` (`exclusions_absent_from_successor: true`) |

Wording to use on the card: "The published rows define the corpus. Each row's `sha256` lets you verify text you re-derive from the pinned upstream file with the published pipeline. Token shards and `shards.json` are identified by hash and are not claimed to be bit-reproducible by the current code."

## 5. Build steps (for the implementer, after Q-1..Q-4 are answered)

1. Owner records `dataset-redistribution` (and `weights-release-license-treatment`) in `configs/owner_decisions.toml`; a PR updates `configs/sources.toml` `redistribution` strings for the seven payload sources to name the public edition (registry change, reviewed).
1. Add `tools/release_public_dataset.py` (new, Apache-2.0): reads accepted parquet in place, emits per-config parquet to a staging dir on a volume with free space, writes `release_manifest.json` with per-file SHA-256 and row counts; refuses the held configs unless the owner decision names them.
1. Build `LICENSES.md` from `NOTICE_draft.md` B and the card from this plan; run a schema check that every text row has non-empty `license` and `url` (0 missing expected from the census) and that no `license` value falls outside the allowed set of the registry's `allowed_document_licenses`.
1. Upload config by config with `huggingface_hub`; record the commit sha and a readback hash receipt (as `hf_receipt.json` did for the private repo); add the receipt to `reports/j1-base-v0.2.2/`.
1. Update `docs/data_card.md` with a 'Public edition' subsection pointing at the repo and the receipt (append-only; frozen figures unchanged).

