# Dataset availability: `j1-base-v0.2.2`

The training corpus is published as the Hugging Face dataset
[`CinqicX/juniper1-base-corpus-v0.2.2`](https://huggingface.co/datasets/CinqicX/juniper1-base-corpus-v0.2.2).
Its card metadata is `license: other` with a `LICENSES.md`: the content is third-party and mixed, and the
Apache-2.0 license of this repository covers only Cinqic's compilation, schema, manifests and hashes.

| Config | Source | Documents | Document tokens | Share | What ships | Per-document license(s) |
|---|---|---:|---:|---:|---|---|
| `text-wikimedia` | `wikimedia` | 489,622 | 472,888,757 | 13.86 % | text + attribution | CC BY-SA 4.0 |
| `text-stackexchange` | `stackexchange` | 558,829 | 443,046,046 | 12.98 % | text + attribution | CC BY-SA 3.0 (61 %), 4.0 (37 %), 2.5 (2 %) |
| `text-arxiv-papers` | `arxiv-papers` | 10,380 | 210,458,332 | 6.17 % | text + attribution | CC BY 4.0 (88 %), CC BY-SA 4.0, CC0, CC BY 3.0, public domain |
| `text-pressbooks` | `pressbooks` | 43,881 | 106,899,715 | 3.13 % | text + attribution | CC BY 4.0 (76 %), CC BY-SA 4.0, public domain, CC0 |
| `text-libretexts` | `libretexts` | 36,091 | 78,660,138 | 2.30 % | text + attribution | CC BY 4.0 (60 %), CC BY-SA 4.0 (31 %), public domain, CC BY 3.0, GFDL (735 rows), CC BY-SA 3.0, CC BY-SA 2.5 |
| `text-oercommons` | `oercommons` | 3,888 | 8,571,178 | 0.25 % | text + attribution | CC BY 4.0 (87 %), CC BY-SA 4.0, public domain |
| `text-python-peps` | `python-peps` | 639 | 2,715,189 | 0.08 % | text | public domain / CC0 |
| `manifest-fineweb-edu-dedup` | `fineweb-edu-dedup` | 970,513 | 1,079,852,836 | 31.64 % | manifest only (IDs, hashes, WARC locators) | ODC-By 1.0 (database); page contents not licensed |
| `manifest-stackv2-edu` | `stackv2-edu` | 547,654 | 585,631,167 | 17.16 % | manifest only (blob id, repo, path, revision, detected licenses, raw URL) | per-file permissive licenses |
| `manifest-finemath-4plus` | `finemath-4plus` | 297,874 | 424,296,121 | 12.43 % | manifest only (IDs, hashes, WARC locators) | ODC-By 1.0 (database); page contents not licensed |

Totals: 2,959,371 documents (train 2,954,191 / validation 2,576 / test 2,604), 3,413,019,479 document tokens.
Text payloads cover 38.8 % of training tokens; the three manifest-only sources cover 61.2 %. Every row of every
config carries `doc_id`, `split`, `sha256` and `norm_sha256` of Cinqic's stored text, so a document re-derived
from the pinned upstream file with the published pipeline can be verified byte-exactly. The `reconstruction/`
folder of the dataset holds the accepted manifest (`54a3adb9…4185`), `MANIFEST.sha256`, the exclusion lists,
the successor lineage, `shards.json` (`1cf5e232…3aef`) and the pinned upstream file inventory (repository,
revision, path, blob id, size, upstream SHA-256) for all ten sources.

Why three sources are manifest-only (details and quoted evidence in [RIGHTS.md](RIGHTS.md)):

- **FineWeb-Edu and FineMath** are ODC-By 1.0 databases over Common Crawl content. ODC-By licenses the
  database, not the page contents, and the Common Crawl terms are non-sublicensable. The upstream publisher
  redistributes them; Cinqic's registry recorded them as training-only and the owner has not decided otherwise
  (open question Q-1 in RIGHTS.md). Reconstruction: both upstream datasets are public at the pinned revisions.
- **Stack v2 Edu** code carries permissive licenses that require copyright and permission notices to travel with
  copies; the stored text cannot carry each repository's notices, 86 distinct license identifiers appear (72 not
  checked against the Blue Oak list), and 2,300 files have an empty license string (Q-2). Reconstruction: the
  manifest gives blob id, repository, path and revision for every file.
- **Token shards** (`*.bin`, `*-docs.parquet`) are a reversible encoding of all ten sources' text and are not
  published; `shards.json` and the hashes are.

Not claimed: byte-identical regeneration of the token files for the full set (shown on a 5 % subset plus all
held-out shards), byte-identical `shards.json` (the current writer omits a `seconds` field), and a full
end-to-end pipeline rerun by an outside party (the dedup and selection stages also saw three internal Cinqic
documentation sources that were removed from v0.2.2 and are not published; the published document set is the
definition of the corpus, verified by hash).
