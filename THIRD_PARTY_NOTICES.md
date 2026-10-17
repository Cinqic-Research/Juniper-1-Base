# Third-party notices

The Cinqic-owned parts of this release are under Apache-2.0 (see `LICENSE` and `NOTICE`). The following
third-party materials were used as training input and, where stated, are redistributed in the dataset
release under their own terms. Pinned revisions, per-source evidence links and the measured composition are in
`data/configs/sources.toml`, `data/docs/sources.md` and `data/docs/data_card.md`. The release-rights table with
quoted license statements is in `docs/RIGHTS.md`.

| Source (`source_id`) | Upstream | Pinned revision | License(s) | Share of training tokens | In the dataset release |
|---|---|---|---|---:|---|
| `fineweb-edu-dedup` | `HuggingFaceTB/smollm-corpus` (FineWeb-Edu dedup) | `3ba9d605` | ODC-By 1.0; Common Crawl Terms of Use | 31.64 % | reconstruction manifest only (document IDs, hashes, pinned file inventory); no text |
| `stackv2-edu` | `common-pile/stackv2_edu_filtered` | `c354dbe8` | per-file permissive licenses (MIT, Apache-2.0, BSD-2/3, ISC, Unlicense, 0BSD, Zlib, Blue Oak permissive) | 17.16 % | see `docs/DATASET.md` |
| `wikimedia` | `common-pile/wikimedia_filtered` | `0641bb84` | CC BY-SA 3.0 / 4.0, GFDL 1.3 | 13.86 % | see `docs/DATASET.md` |
| `stackexchange` | `common-pile/stackexchange_filtered` | `c0ac7373` | CC BY-SA 2.5 / 3.0 / 4.0 | 12.98 % | see `docs/DATASET.md` |
| `finemath-4plus` | `HuggingFaceTB/finemath` (FineMath 4+) | `e92b25a6` | ODC-By 1.0; Common Crawl Terms of Use | 12.43 % | reconstruction manifest only; no text |
| `arxiv-papers` | `common-pile/arxiv_papers_filtered` | `033cf7f5` | per-paper CC BY 4.0 / CC BY-SA 4.0 / CC0 / public domain | 6.17 % | see `docs/DATASET.md` |
| `pressbooks` | `common-pile/pressbooks_filtered` | `1a1d3b50` | per-book CC BY 4.0 / CC BY-SA 4.0 / CC0 / public domain | 3.13 % | see `docs/DATASET.md` |
| `libretexts` | `common-pile/libretexts_filtered` | `70388bca` | per-page CC BY 3.0/4.0, CC BY-SA 3.0/4.0, CC0, public domain, GFDL | 2.30 % | see `docs/DATASET.md` |
| `oercommons` | `common-pile/oercommons_filtered` | `506b6159` | per-resource CC BY 4.0 / CC BY-SA 4.0 / CC0 / public domain | 0.25 % | see `docs/DATASET.md` |
| `python-peps` | `common-pile/python_enhancement_proposals_filtered` | `58217090` | public domain / CC0 | 0.08 % | see `docs/DATASET.md` |

Common Pile: Nikhil Kandpal et al., "The Common Pile v0.1: An 8TB Dataset of Public Domain and Openly
Licensed Text", arXiv:2506.05209 (2025). FineWeb-Edu: Penedo et al., "The FineWeb Datasets", 2024; SmolLM-Corpus
and FineMath: Hugging Face TB research team, 2024. Common Crawl: <https://commoncrawl.org/terms-of-use>.

ShareAlike notice. Wikimedia and Stack Exchange text, and the CC BY-SA documents in the textbook and arXiv
sources, are licensed under Creative Commons Attribution-ShareAlike. Any redistribution of that text in the
dataset release stays under CC BY-SA with the per-document attribution supplied by the upstream collection.
The model weights are released by Cinqic under Apache-2.0; Cinqic does not assert that weights are or are not
adaptations of ShareAlike text, and discloses the sources so that users can form their own view.

Software dependencies of the release package (PyTorch, NumPy, Hugging Face `tokenizers` and `safetensors`,
`huggingface_hub`) are not redistributed here and keep their own licenses.
