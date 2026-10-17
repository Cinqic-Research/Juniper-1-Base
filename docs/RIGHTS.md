# Rights table for the Juniper 1 Base release (corpus `j1-base-v0.2.2`)

> **Resolution (2026-10-16, recorded 2026-10-17 01:50 AM EDT).** The owner's 2026-10-16 directive was applied: `configs/owner_decisions.toml`
> now carries `weights-release-license-treatment-2026-10-16` and `dataset-redistribution-2026-10-16` (status `DECIDED`), adopting this
> table's defaults: Cinqic-owned artifacts under Apache-2.0 with the NOTICE disclosures; seven text-payload sources under their own
> licenses; FineWeb-Edu, FineMath and Stack v2 Edu as reconstruction manifests only (Q-1 and Q-2 stay open as owner questions; the
> release is complete without them). Q-3 and Q-4 below are therefore resolved. The original PENDING records are preserved unchanged.
> Sentences below that read as a proposal ("this branch", "remain PENDING", "NOT RE-VERIFIED") describe the state when the table was written.

Generated 2026-10-16 08:37 PM EDT by the rights-and-provenance subagent (Fable 5.1 lead session). Machine-readable twin: `RIGHTS.json` (`rights_table_2026-10-16.json` in the private repository). This document records evidence and a proposal; it changes no registry record. `configs/owner_decisions.toml` release decisions `weights-release-license-treatment` and `dataset-redistribution` remain PENDING.

Vocabulary: **the registry says** = `configs/sources.toml` / `configs/owner_decisions.toml`; **the upstream card says** = text fetched today at the pinned revision (quoted verbatim); **assessment** = my reading, not legal advice and not an owner decision.

## 1. Sources (10 of 10 in v0.2.2; the three `cinqic-*` sources are absent from v0.2.2)

Measured columns come from a full read of the accepted parquet `meta` and `n_bytes` columns today (2,959,371 documents); tokens from the data card recount.

| Source | Upstream (pinned revision) | Registry license state | Upstream card `license` key | Per-document `license` values (full census) | Attribution obligations | Status | Destination | Exact action |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `libretexts` | [common-pile/libretexts_filtered@70388bca52b4](https://huggingface.co/datasets/common-pile/libretexts_filtered/tree/70388bca52b4a93515e14b1d56618fd7944988fd) | APPROVED_WITH_ATTRIBUTION; CC-BY-4.0, CC-BY-SA-4.0, CC-BY-3.0, CC-BY-SA-3.0, CC0-1.0, Public-Domain | none (no `license` key in the YAML front matter) | CC BY 4.0 21,541 (59.7%); CC BY-SA 4.0 11,023 (30.5%); Public Domain 1,128 (3.1%); CC BY 3.0 1,015 (2.8%); GNU Free Documentation License 735 (2.0%); CC BY-SA 3.0 646 (1.8%); CC BY-SA 2.5 3 (0.0%) | Per document: author (meta.author, 82.2%), title, url, license string. CC BY/BY-SA 3(a): creator, license notice, URI, indication of modifications; BY-SA rows ShareAlike. 735 rows are `GNU Free Documentation License` (version not stated): copies must carry the GFDL text and notice (GFDL section 2/4); 1,128 rows Public Domain. NOTE: the registry `identifiers` list for libretexts omits GFDL although `allowed_document_licenses` admits it. | **REDISTRIBUTABLE_WITH_TERMS** | HF dataset payload (config text-libretexts) | Ship text + per-document license/author/title/url. Card states modifications (line-ending/control repair, outer blank lines, exact-line site-chrome removal, privacy redactions). BY-SA rows keep CC BY-SA; GFDL rows ship with LICENSES/GFDL-1.3.txt and the notice; no relicensing. |
| `pressbooks` | [common-pile/pressbooks_filtered@1a1d3b50d77f](https://huggingface.co/datasets/common-pile/pressbooks_filtered/tree/1a1d3b50d77f834370f8eb4c0d174668dd1676bb) | APPROVED_WITH_ATTRIBUTION; CC-BY-4.0, CC-BY-SA-4.0, CC0-1.0, Public-Domain | none | CC BY 4.0 33,152 (75.5%); CC BY-SA 4.0 6,219 (14.2%); Public Domain 2,781 (6.3%); CC0 1.0 1,729 (3.9%) | Per document: author, title, url, book_url, license string. CC BY/BY-SA attribution as above; Public Domain / CC0 rows: none required (source credit kept anyway). | **REDISTRIBUTABLE_WITH_TERMS** | HF dataset payload (config text-pressbooks) | Ship text + per-document license/author/title/url/book_url. Same modification statement. BY-SA rows keep CC BY-SA. |
| `oercommons` | [common-pile/oercommons_filtered@506b6159dadc](https://huggingface.co/datasets/common-pile/oercommons_filtered/tree/506b6159dadcbc0dc67611cea024eedb04232fb2) | APPROVED_WITH_ATTRIBUTION; CC-BY-4.0, CC-BY-SA-4.0, CC0-1.0, Public-Domain | none | CC BY 4.0 3,365 (86.5%); CC BY-SA 4.0 345 (8.9%); Public Domain 178 (4.6%) | Per document: author, title, url, license string. CC BY attribution for CC BY rows; none for Public Domain rows. | **REDISTRIBUTABLE_WITH_TERMS** | HF dataset payload (config text-oercommons) | Ship text + per-document license/author/title/url. |
| `python-peps` | [common-pile/python_enhancement_proposals_filtered@582170907dd3](https://huggingface.co/datasets/common-pile/python_enhancement_proposals_filtered/tree/582170907dd303c207770fceacd38e6abf133edc) | APPROVED; Public-Domain, CC0-1.0 | none | Public Domain 639 (100.0%) | None required. pep_number, authors and url carried as courtesy provenance. | **REDISTRIBUTABLE_WITH_TERMS** | HF dataset payload (config text-python-peps) | Ship text + pep_number/authors/url/license. |
| `wikimedia` | [common-pile/wikimedia_filtered@0641bb84bd9b](https://huggingface.co/datasets/common-pile/wikimedia_filtered/tree/0641bb84bd9b7162bcddf8be7836822161a9a342) | APPROVED_WITH_ATTRIBUTION; CC-BY-SA-4.0, CC-BY-SA-3.0, GFDL-1.3 | none | CC BY-SA 4.0 489,622 (100.0%) | CC BY-SA 4.0: attribution by URL to the article (Wikimedia ToU section 7 accepts hyperlink/URL) plus authors list where present (meta.authors); ShareAlike on the shared copy; indicate modifications. | **REDISTRIBUTABLE_WITH_TERMS** | HF dataset payload (config text-wikimedia) | Ship text + url/title/wiki/authors/license. Config-level license CC BY-SA 4.0. Card carries the ToU attribution statement and the modification statement. |
| `arxiv-papers` | [common-pile/arxiv_papers_filtered@033cf7f53f9b](https://huggingface.co/datasets/common-pile/arxiv_papers_filtered/tree/033cf7f53f9b348deec868c1a5a48484f3ee9e52) | APPROVED_WITH_ATTRIBUTION; CC-BY-4.0, CC-BY-SA-4.0, CC0-1.0, Public-Domain | none | CC BY 4.0 9,182 (88.5%); CC BY-SA 4.0 508 (4.9%); CC0 1.0 410 (3.9%); CC BY 3.0 211 (2.0%); Public Domain 69 (0.7%) | Per paper: authors (100%), arXiv URL, license string (CC BY 4.0 88.5%, CC BY-SA 4.0 4.9%, CC0 3.9%, CC BY 3.0 2.0%, Public Domain 0.7%). CC BY/BY-SA attribution; CC0/PD rows none. Modified form (LaTeX-to-Markdown by Common Pile, then Juniper repairs) must be indicated. | **REDISTRIBUTABLE_WITH_TERMS** | HF dataset payload (config text-arxiv-papers) | Ship text + authors/url/license/submitter. BY-SA rows keep CC BY-SA. |
| `stackexchange` | [common-pile/stackexchange_filtered@c0ac7373830c](https://huggingface.co/datasets/common-pile/stackexchange_filtered/tree/c0ac7373830c688a43fc12d1988c4b19ccd884ab) | APPROVED_WITH_ATTRIBUTION; CC-BY-SA-4.0, CC-BY-SA-3.0, CC-BY-SA-2.5 | none | CC BY-SA 3.0 341,191 (61.1%); CC BY-SA 4.0 208,903 (37.4%); CC BY-SA 2.5 8,735 (1.6%) | CC BY-SA (version per document: 3.0 61.1%, 4.0 37.4%, 2.5 1.6%): link to the question URL (meta.url, 100%), author profile URLs (meta.authors, 86.3%), ShareAlike on the shared copy; indicate modifications. | **REDISTRIBUTABLE_WITH_TERMS** | HF dataset payload (config text-stackexchange) | Ship text + url/site/authors/license/all_licenses. Per-row license version respected (no up-levelling 2.5/3.0 to 4.0). See flagged risk R-2 (official-dump terms) in the questions section. |
| `stackv2-edu` | [common-pile/stackv2_edu_filtered@c354dbe88469](https://huggingface.co/datasets/common-pile/stackv2_edu_filtered/tree/c354dbe88469a1153e97c6a63ac50591849654de) | APPROVED_WITH_ATTRIBUTION; MIT, Apache-2.0, BSD-2/3-Clause, ISC, Unlicense, 0BSD, Zlib, other Blue Oak permissive | none | MIT 374,848 (68.4%); Apache-2.0 98,723 (18.0%); BSD-3-Clause 26,854 (4.9%); BSD-2-Clause 10,297 (1.9%); Unlicense 8,136 (1.5%); CC0-1.0 4,004 (0.7%); MIT,Apache-2.0 3,346 (0.6%); +740 more (see JSON) | Per file: repo_name, path, blob_id, revision_id, detected_licenses, url (raw.githubusercontent). Full census: license_type=permissive on 100%; `license` string present on 99.58% (2,300 files empty); 86 distinct SPDX tokens, MIT 68.4%, Apache-2.0 18.0%, BSD-3-Clause 4.9%, BSD-2-Clause 1.9%, Unlicense 1.5%, CC0 0.7%, plus a long tail that includes BSD-4-Clause, BSD-4-Clause-UC, MIT-advertising, OpenSSL, Artistic-2.0, MS-PL, Python-2.0, PHP-3.0 (Blue Oak membership of these NOT verified today; only 14 identifiers were checked against the list). MIT/BSD/Apache-2.0/ISC/Zlib require the copyright notice and permission/license text to accompany redistributed copies; the per-file payload carries neither the repository LICENSE file nor, in most files, a copyright line; 4-clause BSD and OpenSSL add advertising/acknowledgement clauses. | **RECONSTRUCTION_ONLY** | HF dataset reconstruction manifest (config manifest-stackv2-edu) | Ship doc_id + blob_id + repo_name + path + revision_id + raw URL + detected_licenses + sha256/norm_sha256/n_bytes/split + pinned upstream file inventory. Hold the code text payload for owner question Q-2. |
| `finemath-4plus` | [HuggingFaceTB/finemath@e92b25a61673](https://huggingface.co/datasets/HuggingFaceTB/finemath/tree/e92b25a616738fe95dc186b64dfb19f9c8525594) | TRAINING_ONLY_NO_REDISTRIBUTION; ODC-By-1.0, LicenseRef-CommonCrawl-ToU | odc-by | none (no per-document license field) | ODC-By 4.2 notice on any distributed Database/Derivative Database (license URI, keep notices); ODC-By 4.3 notice on Produced Works (the model). Page-level copyright is not licensed by ODC-By; no per-document license field exists. | **RECONSTRUCTION_ONLY** | HF dataset reconstruction manifest (config manifest-finemath-4plus) | Ship doc_id + url + warc_filename + warc_record_offset + crawl + sha256/norm_sha256/n_bytes/split + pinned upstream file inventory, under ODC-By 1.0 with the 4.2 notice. Hold the text payload for owner question Q-1. |
| `fineweb-edu-dedup` | [HuggingFaceTB/smollm-corpus@3ba9d6057741](https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus/tree/3ba9d605774198c5868892d7a8deda78031a781f) | TRAINING_ONLY_NO_REDISTRIBUTION; ODC-By-1.0, LicenseRef-CommonCrawl-ToU | odc-by | none (no per-document license field) | As finemath-4plus; attribute HuggingFaceTB/smollm-corpus and HuggingFaceFW/fineweb-edu. | **RECONSTRUCTION_ONLY** | HF dataset reconstruction manifest (config manifest-fineweb-edu-dedup) | Ship doc_id + upstream id (source_key, urn:uuid) + url + dump + file_path + sha256/norm_sha256/n_bytes/split + pinned upstream file inventory, under ODC-By 1.0 with the 4.2 notice. Hold the text payload for owner question Q-1. |

### 1a. Primary evidence relied on (URL and exact statement)

**`libretexts`**

- <https://huggingface.co/datasets/common-pile/libretexts_filtered/blob/70388bca52b4a93515e14b1d56618fd7944988fd/README.md> — "check each textbook section for a license statement indicating that it is in the public domain or under a CC BY, CC BY-SA, or the GNU Free Documentation License"
- <https://huggingface.co/datasets/common-pile/libretexts_filtered/blob/70388bca52b4a93515e14b1d56618fd7944988fd/README.md> — "Per-document license information is available in the `license` entry of the `metadata` field of each example."

**`pressbooks`**

- <https://huggingface.co/datasets/common-pile/pressbooks_filtered/blob/1a1d3b50d77f834370f8eb4c0d174668dd1676bb/README.md> — "listed as public domain or under CC BY or CC BY-SA licenses."
- <https://huggingface.co/datasets/common-pile/pressbooks_filtered/blob/1a1d3b50d77f834370f8eb4c0d174668dd1676bb/README.md> — "Per-document license information is available in the `license` entry of the `metadata` field of each example."

**`oercommons`**

- <https://huggingface.co/datasets/common-pile/oercommons_filtered/blob/506b6159dadcbc0dc67611cea024eedb04232fb2/README.md> — "retrieve English-language content released into the public domain or under CC BY or CC BY-SA licenses."
- <https://huggingface.co/datasets/common-pile/oercommons_filtered/blob/506b6159dadcbc0dc67611cea024eedb04232fb2/README.md> — "Per-document license information is available in the `license` entry of the `metadata` field of each example."

**`python-peps`**

- <https://peps.python.org/pep-0001/> — "This document is placed in the public domain or under the CC0-1.0-Universal license, whichever is more permissive."
- <https://peps.python.org/pep-0001/> — "Each new PEP must be placed under a dual license of public domain and CC0-1.0-Universal"
- <https://huggingface.co/datasets/common-pile/python_enhancement_proposals_filtered/blob/582170907dd303c207770fceacd38e6abf133edc/README.md> — "The majority of PEPs are published in the Public Domain, 5 were published under the "Open Publication License" and omitted from this dataset."

**`wikimedia`**

- <https://huggingface.co/datasets/common-pile/wikimedia_filtered/blob/0641bb84bd9b7162bcddf8be7836822161a9a342/README.md> — "Official Wikimedia wikis are released under a CC BY-SA license."
- <https://foundation.wikimedia.org/wiki/Policy:Terms_of_Use> — "Creative Commons Attribution-ShareAlike 4.0 International License"
- <https://foundation.wikimedia.org/wiki/Policy:Terms_of_Use> — "Through hyperlink (where possible) or URL to the article to which you contributed."
- <https://foundation.wikimedia.org/wiki/Policy:Terms_of_Use> — "CC BY-SA 4.0 or later"

**`arxiv-papers`**

- <https://huggingface.co/datasets/common-pile/arxiv_papers_filtered/blob/033cf7f53f9b348deec868c1a5a48484f3ee9e52/README.md> — "This dataset includes text from all papers uploaded under CC BY, CC BY-SA, and CC0 licenses through a three-step pipeline:"
- <https://info.arxiv.org/help/license/index.html> — "This license gives limited rights to arXiv to distribute the article"

**`stackexchange`**

- <https://huggingface.co/datasets/common-pile/stackexchange_filtered/blob/c0ac7373830c688a43fc12d1988c4b19ccd884ab/README.md> — "we extract all questions and answers from community-uploaded dumps from December of 2024"
- <https://huggingface.co/datasets/common-pile/stackexchange_filtered/blob/c0ac7373830c688a43fc12d1988c4b19ccd884ab/README.md> — "Per-document license information is available in the `license` entry of the `metadata` field of each example."
- <https://archive.org/details/stackexchange> — "All user content contributed to the Stack Exchange network is cc-by-sa 4.0 licensed, intended to be shared and remixed."
- <https://stackoverflow.com/help/licensing> — "NOT RE-VERIFIED 2026-10-16: stackoverflow.com and meta.stackexchange.com were unreachable from this session; registry evidence URL kept, per-document license versions taken from the data (2.5/3.0/4.0)."

**`stackv2-edu`**

- <https://huggingface.co/datasets/common-pile/stackv2_edu_filtered/blob/c354dbe88469a1153e97c6a63ac50591849654de/README.md> — "When multiple licenses are detected in a single repository, we ensure that all of the licenses are on the Blue Oak Council certified license list."
- <https://huggingface.co/datasets/common-pile/stackv2_edu_filtered/blob/c354dbe88469a1153e97c6a63ac50591849654de/README.md> — "Per-document license information is available in the `license` entry of the `metadata` field of each example."
- <https://blueoakcouncil.org/list> — "The Council publishes this list to identify permissive licenses, so that everyone can recognize, rely on, and use them without expensive legal help."

**`finemath-4plus`**

- <https://huggingface.co/datasets/HuggingFaceTB/finemath/blob/e92b25a616738fe95dc186b64dfb19f9c8525594/README.md> — "license: odc-by"
- <https://huggingface.co/datasets/HuggingFaceTB/finemath/blob/e92b25a616738fe95dc186b64dfb19f9c8525594/README.md> — "The dataset is released under the **Open Data Commons Attribution License (ODC-By) v1.0**"
- <https://huggingface.co/datasets/HuggingFaceTB/finemath/blob/e92b25a616738fe95dc186b64dfb19f9c8525594/README.md> — "The use of this dataset is also subject to [CommonCrawl's Terms of Use](https://commoncrawl.org/terms-of-use)."
- <https://opendatacommons.org/licenses/by/1-0/> — "this License does not cover any rights (other than Database Rights or in contract)"
- <https://commoncrawl.org/terms-of-use> — "non-assignable, non-transferable, non-sublicensable, non-exclusive, limited license"
- <https://commoncrawl.org/terms-of-use> — "BY USING THE CRAWLED CONTENT, YOU AGREE TO RESPECT THE COPYRIGHTS AND OTHER APPLICABLE RIGHTS OF THIRD PARTIES"
- <https://commoncrawl.org/terms-of-use> — "including the use of Crawled Content for the purpose of developing, training, or deploying AI Systems"

**`fineweb-edu-dedup`**

- <https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus/blob/3ba9d605774198c5868892d7a8deda78031a781f/README.md> — "license: odc-by"
- <https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus/blob/3ba9d605774198c5868892d7a8deda78031a781f/README.md> — "FineWeb-Edu-Dedup is a deduplicated subset of the [FineWeb-Edu]"
- <https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/blob/main/README.md> — "The dataset is released under the **Open Data Commons Attribution License (ODC-By) v1.0**"
- <https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/blob/main/README.md> — "The use of this dataset is also subject to [CommonCrawl's Terms of Use](https://commoncrawl.org/terms-of-use)."
- <https://opendatacommons.org/licenses/by/1-0/> — "this License does not cover any rights (other than Database Rights or in contract)"
- <https://commoncrawl.org/terms-of-use> — "non-assignable, non-transferable, non-sublicensable, non-exclusive, limited license"

Shared evidence for all eight Common Pile sources: the card caveat "license laundering and inaccurate metadata can cause us to erroneously assign the incorrect license to some documents" (each card, section License Issues) and the Common Pile paper <https://arxiv.org/abs/2506.05209> ("The Common Pile v0.1: An 8TB Dataset of Public Domain and Openly Licensed Text", Kandpal et al., CC BY 4.0 abstract page).

ODC-By 1.0 (<https://opendatacommons.org/licenses/by/1-0/>) section 4.3, relied on for the model NOTICE: "Creating and Using a Produced Work does not require the notice in Section 4.2. However, if you Publicly Use a Produced Work, You must include a notice associated with the Produced Work reasonably calculated to make any Person that uses, views, accesses, interacts with, or is otherwise exposed to the Produced Work aware that Content was obtained from the Database, Derivative Database, or the Database as part of a Collective Database, and that it is available under this License." Section 4.2: "Include a copy of this License or its Uniform Resource Identifier (URI) with the Database or Derivative Database". Section 4.4: "You may not sublicense the Database."

CC BY-SA 4.0 legal code (<https://creativecommons.org/licenses/by-sa/4.0/legalcode.en>) 3(a)(1)(A): "identification of the creator(s) of the Licensed Material", "a copyright notice;", "a notice that refers to this Public License;", "a URI or hyperlink to the Licensed Material to the extent reasonably practicable;"; 3(a)(2): "You may satisfy the conditions in Section 3(a)(1) in any reasonable manner based on the medium, means, and context in which You Share the Licensed Material."; 3(b)(1): "The Adapter’s License You apply must be a Creative Commons license with the same License Elements, this version or later, or a BY-SA Compatible License."

Common Crawl Terms of Use (<https://commoncrawl.org/terms-of-use>): section 1 "non-assignable, non-transferable, non-sublicensable, non-exclusive, limited license"; section 3 "BY USING THE CRAWLED CONTENT, YOU AGREE TO RESPECT THE COPYRIGHTS AND OTHER APPLICABLE RIGHTS OF THIRD PARTIES"; section 9 indemnity clause (ii) "use of Crawled Content in connection with artificial intelligence, machine learning, or other similar technologies" ... "including the use of Crawled Content for the purpose of developing, training, or deploying AI Systems". The Terms contain no explicit redistribution clause.

### 1b. Per-document metadata presence (full census of `meta`, fraction of documents)

| Source | docs | `license` | `url` | `title` | `author` | `authors` | `doc_id` | `license_type` | `detected_licenses` | `repo_name` | `path` | `blob_id` | `site` | `pep_number` | `wiki` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `libretexts` | 36,091 | 100.0% | 100.0% | 100.0% | 82.2% |  | 100.0% |  |  |  |  |  |  |  |  |
| `pressbooks` | 43,881 | 100.0% | 100.0% | 100.0% | 99.6% |  | 100.0% |  |  |  |  |  |  |  |  |
| `oercommons` | 3,888 | 100.0% | 100.0% | 100.0% | 90.4% |  | 100.0% |  |  |  |  |  |  |  |  |
| `python-peps` | 639 | 100.0% | 100.0% |  |  | 100.0% | 100.0% |  |  |  |  |  |  | 100.0% |  |
| `wikimedia` | 489,622 | 100.0% | 100.0% | 100.0% |  | 91.6% | 100.0% |  |  |  |  |  |  |  | 100.0% |
| `arxiv-papers` | 10,380 | 100.0% | 100.0% |  |  | 100.0% | 100.0% |  |  |  |  |  |  |  |  |
| `stackexchange` | 558,829 | 100.0% | 100.0% |  |  | 86.3% | 100.0% |  |  |  |  |  | 100.0% |  |  |
| `stackv2-edu` | 547,654 | 99.6% | 100.0% |  |  |  | 100.0% | 100.0% | 99.6% | 100.0% | 100.0% | 100.0% |  |  |  |
| `finemath-4plus` | 297,874 |  | 100.0% |  |  |  |  |  |  |  |  |  |  |  |  |
| `fineweb-edu-dedup` | 970,513 |  | 100.0% |  |  |  |  |  |  |  |  |  |  |  |  |

`doc_id` here is the upstream identifier inside `meta` (LibreTexts/Pressbooks/OER URL, PEP number, wiki page id, arXiv id, Stack Exchange question id, Stack v2 blob id); Cinqic's own `doc_id` column is a separate 32-hex id present on every row. FineMath and FineWeb-Edu carry no license, title or author field at all; their only identifiers are `url` (+ `warc_filename`/`warc_record_offset`, or `dump`/`file_path` with the `urn:uuid` in `source_key`).

## 2. Cinqic-owned artifact classes

| Artifact | Origin | Status | Destination | Notices | Exact action |
| --- | --- | --- | --- | --- | --- |
| weights (juniper1-base-v1, ckpt 52034) | Cinqic-trained from scratch on j1-base-v0.2.2 | **CINQIC_OWNED_APACHE** | Hugging Face model repo + GitHub release pointer | Apache-2.0 LICENSE + NOTICE: ODC-By 4.3 Produced-Work notice (FineWeb-Edu, FineMath); CC BY-SA and Common Pile training disclosure; permissive-code disclosure; statement that the model license does not relicense third-party content. | Publish under Apache-2.0 with NOTICE_draft.md text and model-card data disclosure. Owner must first record `weights-release-license-treatment` in configs/owner_decisions.toml (owner-only edit; see section 3). |
| tokenizer (cand5-j1tok-24k: tokenizer.json 176a89e2..., freeze.json 08a2d898...) | Cinqic-trained on the corpus text | **CINQIC_OWNED_APACHE** | Hugging Face model repo (and dataset repo copy) | Apache-2.0; same NOTICE (a vocabulary is a Produced Work under ODC-By 4.3). | Publish with the weights. |
| pipeline code (Juniper-1-Data src/, configs/, schemas/, tools/, tests/) | Cinqic-authored; repo LICENSE already Apache-2.0 | **CINQIC_OWNED_APACHE** | GitHub (public repo or public mirror of the release branch) | Apache-2.0; third-party Python dependencies under their own licenses (not vendored). | Publish. Before the repository is made public, a git-history audit for private material (local paths, SFT/private partitions, owner statements) is a separate pre-publication step; this table does not assume the history is clean. |
| manifests, hashes, doc-id lists (reports/j1-base-v0.2.2/*.json, MANIFEST.sha256, excluded_doc_ids_union.json, manifests/j1-base-v0.2/sources/*.acquisition.json) | Cinqic-produced records (facts, hashes, upstream URLs) | **CINQIC_OWNED_APACHE** | GitHub + HF dataset repo (reconstruction/ folder) | Apache-2.0 for the records; the upstream URLs/ids they contain are facts. Where a manifest is a Derivative Database of an ODC-By database (FineMath/FineWeb row lists), add the ODC-By 4.2 notice. | Publish verbatim (byte-identical to the preserved copies, hashes already recorded). |
| token shards (token_shards/j1-base-v0.2.2/cand5-j1tok-24k: *.bin, *-docs.parquet, shards.json) | Cinqic-produced; the .bin files are a reversible token encoding of all ten sources' text | **RECONSTRUCTION_ONLY** | shards.json + hashes: GitHub/HF; *.bin and *-docs.parquet: NOT published (remain in the private CinqicX/juniper1-corpus repo) | Token IDs decode to the third-party text, so the .bin files carry every source's terms, including the two ODC-By/Common Crawl sources held under Q-1. | Publish shards.json (5,702 B) and the SHA-256 list only; document that a regenerated set is a new artifact (docs/stage_recovery.md). |

## 3. Unresolved rights questions for the owner (each holds an artifact)

| Id | Artifact held | Question | Evidence | Default if unanswered |
| --- | --- | --- | --- | --- |
| Q-1 | text payload of finemath-4plus and fineweb-edu-dedup (1,268,387 documents, 44.1% of tokens) | May the Common Crawl-derived page text be republished as a Derivative Database under ODC-By 1.0 (as HuggingFaceTB/HuggingFaceFW do), or does Cinqic keep the registry state TRAINING_ONLY_NO_REDISTRIBUTION? ODC-By 2.4 does not license the page contents; the Common Crawl licence is non-sublicensable; the registry state was set by the owner-approved registry and the directive's 'actual source terms' clause does not by itself resolve it. | ODC-By 2.4 and 4.4 quotes above; Common Crawl ToU section 1 (non-sublicensable) and section 3; registry state `TRAINING_ONLY_NO_REDISTRIBUTION` with redistribution text "Cinqic does not redistribute this text."; `docs/decisions.md` line 36. Counter-evidence: HuggingFaceTB republishes the same rows under `license: odc-by`, and no per-page rights are asserted in the data. | manifest only |
| Q-2 | text payload of stackv2-edu (547,654 files, 17.2% of tokens) | Ship permissive-licensed code files as text when the per-file payload cannot carry the repository LICENSE/copyright notice that MIT/BSD/Apache-2.0/ISC/Zlib require to accompany copies, when the full census shows 86 distinct licence identifiers (72 not checked against the Blue Oak list today, including 4-clause BSD and OpenSSL with advertising clauses) and 2,300 files with an empty licence string, and when the June-2025 Common Pile snapshot does not reflect later Software Heritage/Stack v2 opt-outs? | Card: "all of the licenses are on the Blue Oak Council certified license list"; measured (full census): `license` present on 99.58%, 747 distinct value strings / 86 SPDX tokens, MIT 68.4% / Apache-2.0 18.0% / BSD-3-Clause 4.9% of files; registry risks (Software Heritage opt-outs, June 2025 snapshot); owner decision `stackv2-license-and-provenance` scope "training input only", "Record training and distribution treatment separately." | manifest only (blob_id + repo + path + raw URL at revision) |
| Q-3 | weights, tokenizer (the whole model release) | Record `weights-release-license-treatment` in configs/owner_decisions.toml (owner-only edit). The 2026-10-16 directive supplies the decision; the registry record is still PENDING and the release gate text says it 'must be resolved before public weights ship'. | `configs/owner_decisions.toml` [[release_decision]] `weights-release-license-treatment` status PENDING: "Before a public Apache-2.0 weight release, record the treatment of weights trained on FineWeb-Edu and FineMath (ODC-By section 4.3 notice), CC BY-SA documents (ShareAlike and attribution), and Stack v2 code (provenance, opt-outs), with model-card disclosure." | model release held until the record is written |
| Q-4 | every HF dataset config (the dataset repo as a whole) | Record `dataset-redistribution` source-by-source in configs/owner_decisions.toml (owner-only edit), using the dispositions in this table. | `configs/owner_decisions.toml` [[release_decision]] `dataset-redistribution` status PENDING: "The training corpus stays private. Any redistribution of corpus text, shards, or derived token IDs needs a source-by-source decision." | dataset release held until the record is written |

Not blocking, flagged:

- **R-1** Common Pile cards warn that license metadata may be wrong ('license laundering and inaccurate metadata...'); carry the same caveat and a takedown contact on the dataset card.
- **R-2** Stack Exchange: the Common Pile source is community-uploaded dumps; Cinqic never accepted the official-dump terms. CC BY-SA redistribution is a licence right, but stackoverflow.com's licensing page could not be re-fetched today; re-verify before publication.
- **R-3** Wikimedia rows lack an authors list on 8.4% of rows (full census: authors present on 91.6%); attribution by article URL is accepted by the Wikimedia ToU, so this is a completeness note, not a block.
- **R-4** stackv2-edu: 2,300 files (0.42%) have license_type=permissive but an empty license string; 86 distinct SPDX tokens occur, of which 14 (MIT, Apache-2.0, BSD-2/3-Clause, ISC, Unlicense, 0BSD, Zlib, CC0-1.0, BSL-1.0, ECL-2.0, OGL-Canada-2.0, MIT-0, Apache-1.1) were confirmed on the Blue Oak list today; the remaining 72 (e.g. BSD-4-Clause, MIT-advertising, OpenSSL, Artistic-2.0, MS-PL, Python-2.0) were not checked. Feeds Q-2.
- **R-5** libretexts: 735 rows are GFDL (no version given) and 3 rows CC BY-SA 2.5; the registry `identifiers` for libretexts list neither, while `allowed_document_licenses` admits GFDL. Payload must carry the GFDL text; registry identifiers should be corrected in a registry PR (not in this branch).

Assessment on the CC BY-SA sources (wikimedia, stackexchange, and the BY-SA rows of libretexts/pressbooks/oercommons/arxiv-papers): redistribution of the *text* under its own CC BY-SA licence with attribution is squarely within the licence, so no question is raised for the dataset payload. Whether the *weights* are Adapted Material of CC BY-SA text is the question the owner's 2026-10-08 decision deliberately left open ("Do not assume the weights are or are not adaptations."); the 2026-10-16 directive to release the weights under Apache-2.0 is an owner choice on that question, and the NOTICE draft discloses the training sources without claiming the weights are or are not adaptations.

## 4. What the owner's 2026-10-16 directive authorizes and what it does not change

Authorizes (owner authority, recorded in the lead session):

- Publishing Cinqic-owned artifacts (weights, tokenizer, pipeline code, manifests/hashes/doc-id lists, shards.json) under Apache-2.0 on GitHub and Hugging Face.
- Publishing dataset and reproducibility materials "to the fullest extent permitted by the actual source terms", which this table reads as: text payload where the per-document licence permits redistribution with terms that the payload schema can discharge (seven sources), reconstruction manifests elsewhere (three sources) pending Q-1/Q-2.
- Preparing the dataset card, NOTICE and release plan in this branch.

Does not change:

- `configs/owner_decisions.toml`: both `[[release_decision]]` records stay PENDING until the owner writes them (Q-3, Q-4). This branch does not edit that file.
- Any third-party licence: Apache-2.0 applies to Cinqic's contribution only; CC BY-SA, CC BY, CC0/public domain, ODC-By and the permissive code licences continue to govern the respective content, and the dataset card must not label mixed content `apache-2.0`.
- The registry state `TRAINING_ONLY_NO_REDISTRIBUTION` for finemath-4plus and fineweb-edu-dedup (a registry change needs an owner decision plus a `configs/sources.toml` edit with its own review).
- The frozen records (`reports/j1-base-v0.2.2/*`, data card, qualification) and the private status of the corpus bytes and token shards on HDD/HF/Drive.
- The Stack Exchange official-dump terms question (R-2) and the Common Pile metadata-accuracy caveat (R-1): disclosed, not resolved.

