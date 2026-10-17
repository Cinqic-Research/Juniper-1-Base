# NOTICE drafts for the Juniper 1 Base release (DRAFT, 2026-10-16 08:37 PM EDT)

Status: proposal prepared on the owner's 2026-10-16 directive. Not yet adopted; `configs/owner_decisions.toml`
release decisions remain PENDING. Two texts follow: **A** for the model repositories (weights, tokenizer, code)
and **B** for the public dataset repository. Quoted licence language is verbatim from the sources named in
`rights_table_2026-10-16.md`.

---

## A. NOTICE for the model release (`NOTICE` file beside `LICENSE` = Apache-2.0)

```
Juniper 1 Base (juniper1-base-v1)
Copyright 2026 Cinqic

This product includes software and model artifacts developed by Cinqic.

1. Scope of the Apache License, Version 2.0

The Apache License 2.0 in the file LICENSE applies to the artifacts Cinqic
created and owns: the model weights and configuration of Juniper 1 Base, the
tokenizer cand5-j1tok-24k (tokenizer.json, freeze.json), the Juniper-1-Data
pipeline source code, and Cinqic's build manifests, hash lists, document-id
lists and reports. It is a licence to Cinqic's contribution only.

The model licence does not relicense, and grants no rights in, any
third-party text, code or data that the model was trained on. Training
material remains under the terms of its own authors and licensors as listed
below. Nothing in this release should be read as a claim that any third-party
work is now available under the Apache License.

2. Training-data notices

Juniper 1 Base was trained on the Cinqic corpus j1-base-v0.2.2
(2,959,371 documents), assembled from the following sources. Per-source
record: docs/data_card.md and docs/release/rights_table_2026-10-16.md in the
Juniper-1-Data repository.

2.1 Open Data Commons Attribution License (ODC-By) v1.0 - notice under
    section 4.3

Content in the training corpus was obtained from the following databases,
each of which is made available under the Open Data Commons Attribution
License v1.0 (https://opendatacommons.org/licenses/by/1-0/):

  - FineMath (HuggingFaceTB/finemath, configuration finemath-4plus,
    revision e92b25a616738fe95dc186b64dfb19f9c8525594), SmolLM2 authors,
    arXiv:2502.02737.
  - FineWeb-Edu as deduplicated in SmolLM-Corpus
    (HuggingFaceTB/smollm-corpus, configuration fineweb-edu-dedup,
    revision 3ba9d605774198c5868892d7a8deda78031a781f), derived from
    HuggingFaceFW/fineweb-edu (Penedo et al.).

This model is a Produced Work of those databases within the meaning of the
ODC-By licence. Section 4.3 of that licence requires this notice: Content was
obtained from the Databases named above, and those Databases are available
under the Open Data Commons Attribution License v1.0. Use of those databases
is also subject to the Common Crawl Terms of Use
(https://commoncrawl.org/terms-of-use). The individual web pages in those
databases remain the copyright of their respective authors; the ODC-By
licence covers the databases, not the page contents, and this model release
grants no rights in those pages.

2.2 Creative Commons Attribution-ShareAlike content

The training corpus includes text licensed under Creative Commons
Attribution-ShareAlike licences (versions 2.5, 3.0 and 4.0):

  - English Wikipedia, Wikibooks, Wikiversity, Wikinews and Wikivoyage
    articles (Wikimedia Foundation projects; contributors credited per
    article), CC BY-SA 4.0, obtained via common-pile/wikimedia_filtered.
  - Stack Exchange questions, answers and comments (authors credited per
    post by profile URL), CC BY-SA 2.5 / 3.0 / 4.0 according to
    contribution date, obtained via common-pile/stackexchange_filtered.
  - Individual CC BY-SA 3.0 and 4.0 documents within the LibreTexts,
    Pressbooks, OER Commons and arXiv sources listed in 2.3.

Attribution for every such document (author list or profile URLs where the
upstream data carries them, title, canonical URL and licence URL) is
preserved in the per-document metadata of the published Juniper 1 Base
dataset and in the Juniper-1-Data manifests. Where text from these sources is
redistributed by Cinqic (the dataset release), it is redistributed under the
same CC BY-SA licence and version as the original, with attribution and an
indication of the modifications made (text repairs, boilerplate-line removal,
privacy redactions). Whether a trained language model constitutes Adapted
Material of CC BY-SA text is a question Cinqic does not resolve here; this
notice discloses the sources and preserves their attribution regardless.

2.3 Creative Commons Attribution, CC0 and public-domain content

  - LibreTexts textbook pages (per-page CC BY 4.0, CC BY 3.0, CC BY-SA
    2.5/3.0/4.0, GNU Free Documentation License, public domain), via
    common-pile/libretexts_filtered.
  - Pressbooks open textbooks (per-chapter CC BY 4.0, CC BY-SA 4.0, CC0 1.0,
    public domain), via common-pile/pressbooks_filtered.
  - OER Commons resources (per-document CC BY 4.0, CC BY-SA 4.0, public
    domain), via common-pile/oercommons_filtered.
  - arXiv papers whose authors chose CC BY 4.0, CC BY 3.0, CC BY-SA 4.0,
    CC0 1.0 or the public domain, via common-pile/arxiv_papers_filtered;
    authors credited per paper.
  - Python Enhancement Proposals (public domain / CC0 1.0), via
    common-pile/python_enhancement_proposals_filtered.

Per-document author, title, URL and licence are preserved as in 2.2.

2.4 Permissively licensed source code

The training corpus includes source files from public repositories whose
detected licences are all on the Blue Oak Council permissive licence list
(https://blueoakcouncil.org/list), predominantly MIT, Apache-2.0,
BSD-2-Clause, BSD-3-Clause, ISC, Unlicense, 0BSD, Zlib and CC0-1.0, obtained
via common-pile/stackv2_edu_filtered (revision
c354dbe88469a1153e97c6a63ac50591849654de), itself derived from The Stack v2
(BigCode / Software Heritage; Lozhkov et al., "StarCoder 2 and The Stack v2:
The Next Generation"). Those licences require that copyright and permission
notices accompany copies of the code. Repository name, file path, blob id,
revision and detected licences for every file are preserved in the
Juniper-1-Data manifests; copyright holders are the respective repository
authors. This model release does not redistribute those files.

2.5 Common Pile attribution

Eight of the ten sources were obtained from The Common Pile v0.1:
Nikhil Kandpal, Brian Lester, Colin Raffel, Sebastian Majstorovic, Stella
Biderman, Baber Abbasi, Luca Soldaini, Enrico Shippole, A. Feder Cooper,
Aviya Skowron, John Kirchenbauer, Shayne Longpre, Lintang Sutawika, Alon
Albalak, Zhenlin Xu, Guilherme Penedo, Loubna Ben Allal, Elie Bakouch, John
David Pressman, Honglu Fan, Dashiell Stander, Guangyu Song, Aaron Gokaslan,
Tom Goldstein, Brian R. Bartoldson, Bhavya Kailkhura, Tyler Murray.
"The Common Pile v0.1: An 8TB Dataset of Public Domain and Openly Licensed
Text", arXiv:2506.05209 (2025). The Common Pile cards note that "license
laundering and inaccurate metadata can cause us to erroneously assign the
incorrect license to some documents"; Cinqic carries that caveat forward and
will act on substantiated reports sent to the contact in the model card.

3. Not included

Cinqic's internal documentation sources, synthetic text, and any source
whose terms restrict training or redistribution (see
docs/sources.md, "Considered but not admitted") are not part of the
training corpus.
```

---

## B. THIRD_PARTY_NOTICES / licence section for the dataset release

Use as `LICENSES.md` in the dataset repository and summarise in the dataset card's `license_name`/`license_link`.

```
Juniper 1 Base corpus release (j1-base-v0.2.2, public edition)
Copyright 2026 Cinqic for the compilation, schema, manifests and hashes
(Apache License 2.0). The compilation licence does not apply to the
documents themselves. Each document is licensed by its own author or
licensor under the licence named in that row's `license` and `license_url`
fields, and may be used only under that licence. Nothing in this repository
is offered under the Apache License except Cinqic's own files.

Text-payload configurations and their per-document licences:

  text-libretexts        CC BY 4.0; CC BY-SA 4.0; public domain; CC BY 3.0;
                         GNU Free Documentation License; CC BY-SA 3.0;
                         CC BY-SA 2.5
  text-pressbooks        CC BY 4.0; public domain; CC BY-SA 4.0; CC0 1.0
  text-oercommons        CC BY 4.0; CC BY-SA 4.0; public domain
  text-python-peps       public domain / CC0 1.0
  text-wikimedia         CC BY-SA 4.0 (Wikimedia Terms of Use, section 7)
  text-arxiv-papers      CC BY 4.0; CC BY-SA 4.0; CC0 1.0; CC BY 3.0;
                         public domain
  text-stackexchange     CC BY-SA 3.0; CC BY-SA 4.0; CC BY-SA 2.5

Attribution. Every row carries: `license` (as published upstream),
`license_url`, `url` (canonical location of the original), `title` where the
source has one, `author` / `authors` where the upstream data carries them,
`upstream_id` and `upstream_file` (the pinned Common Pile file and record).
For CC BY and CC BY-SA documents these fields are the attribution required by
section 3(a) of the licence; reusers must keep them with the text.

GNU Free Documentation License. 735 LibreTexts rows are licensed under the
GFDL (version not stated upstream). The licence text is included as
LICENSES/GFDL-1.3.txt; copies of those rows must keep this notice and the
licence text with them.

ShareAlike. Documents licensed CC BY-SA remain under CC BY-SA of the same
version; this release does not change their licence. Anyone who adapts them
must license the adaptation under "a Creative Commons license with the same
License Elements, this version or later, or a BY-SA Compatible License"
(CC BY-SA 4.0, section 3(b)(1)).

Modifications. The text in this release is not byte-identical to the
upstream record. Cinqic applied: CRLF/CR to LF, removal of a leading
byte-order mark, removal of C0 control characters other than TAB and LF,
replacement of unpaired surrogates, removal of leading and trailing blank
lines (j1.normalize.v1); removal of exact site-chrome lines in LibreTexts
pages (count in `removed_boilerplate_lines`); redaction of e-mail and IP
addresses and credential-shaped strings (`redactions`); and, upstream,
Common Pile's own conversions (HTML or wikitext or LaTeX to plain text or
Markdown). The `sha256` field hashes Cinqic's stored text.

Reconstruction-manifest configurations (no document text):

  manifest-stackv2-edu          files from repositories whose detected
                                licences Common Pile reports as Blue Oak
                                permissive (MIT 68%, Apache-2.0 18%, BSD,
                                ISC, Unlicense, CC0 and a long tail); rows give
                                repo_name, path, blob_id, revision_id,
                                detected_licenses and the raw URL. Copyright
                                and permission notices of each repository
                                apply to any copy you reconstruct.
  manifest-finemath-4plus       Open Data Commons Attribution License v1.0
  manifest-fineweb-edu-dedup    Open Data Commons Attribution License v1.0

ODC-By notice (section 4.2) for the two manifest configurations: these row
lists are Derivative Databases of HuggingFaceTB/finemath and
HuggingFaceTB/smollm-corpus (fineweb-edu-dedup), each "released under the
Open Data Commons Attribution License (ODC-By) v1.0"
(https://opendatacommons.org/licenses/by/1-0/); "The use of this dataset is
also subject to CommonCrawl's Terms of Use"
(https://commoncrawl.org/terms-of-use). They are distributed under the same
ODC-By 1.0 terms, with this notice, and with no additional restrictions. The
page contents those rows point to are not included and are not licensed by
this release.

Common Pile. Eight configurations derive from The Common Pile v0.1
(Kandpal et al., arXiv:2506.05209). Cite that paper when using them. The
Common Pile authors warn that "license laundering and inaccurate metadata
can cause us to erroneously assign the incorrect license to some documents";
report suspected errors to the contact in the dataset card and to
https://github.com/r-three/common-pile/discussions/new.

Juniper 1 Base model. The model trained on this corpus is released
separately under the Apache License 2.0 with its own NOTICE; that licence
does not relicense the documents in this repository.
```
