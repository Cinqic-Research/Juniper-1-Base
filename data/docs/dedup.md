# Near-duplicate threshold calibration (j1-base-v0.1)

## Setup

- Signatures: 112 MinHash values over word 5-gram shingles of the matching
  signature (`j1.minhash.v1`), stored per document so banding is chosen at dedup time.
- Pool: 20,196,733 candidates from 129 processed units (all sources).
- Candidate pairs: a permissive 28x4 LSH probe; 200,000 pairs sampled by keyed hash;
  1,500 pairs stratified by signature-estimated similarity; true word 5-gram Jaccard
  computed from text (`reports/j1-base-v0.1/dedup_calibration_summary.json`).
- Labels: 42 pairs stratified by true Jaccard 0.3-1.0, each read in full excerpt and
  labelled duplicate / not duplicate (`reports/j1-base-v0.1/dedup_labels.jsonl`,
  document ids only; no corpus text is committed).

## Detection against true Jaccard (1,500 sampled pairs)

| Banding | 0.4-0.5 | 0.5-0.6 | 0.6-0.7 | 0.7-0.8 | 0.8-0.9 | 0.9-1.0 |
| --- | --- | --- | --- | --- | --- | --- |
| 14x8 (J at 50%: 0.72) | 6/136 | 17/143 | 63/159 | 107/140 | 168/169 | 119/119 |
| 16x7 (0.67) | 17/136 | 44/143 | 90/159 | 127/140 | 169/169 | 119/119 |
| 20x5 (0.55) | 60/136 | 101/143 | 150/159 | 139/140 | 169/169 | 119/119 |
| 9x12 (0.83) | 0/136 | 1/143 | 13/159 | 37/140 | 138/169 | 119/119 |

The empirical curves follow the theoretical S-curves.

## Labelled sample (42 pairs)

| Banding | TP | FP | FN | Precision | Recall |
| --- | --- | --- | --- | --- | --- |
| 14x8 | 19 | 2 | 9 | 0.90 | 0.68 |
| 16x7 | 22 | 4 | 6 | 0.85 | 0.79 |
| 20x5 | 27 | 8 | 1 | 0.77 | 0.96 |
| 9x12 | 13 | 0 | 15 | 1.00 | 0.46 |

The sample over-represents mid-similarity pairs. All 14x8 misses are partial copies
(different extractions of one Q&A, textbook editions, forks, a web copy of part of a
wiki article) at Jaccard 0.33-0.77. False links: two similar SQL schema files, and a
wiki revision-diff page against its article description.

## Cluster structure on the full pool

| Banding | Removed (near, incl. exact) | Largest clusters | Clusters >= 100 |
| --- | --- | --- | --- |
| 14x8, raw code signatures | 152,509 | 630, 609, 434, 396, 328 | 26 |
| 16x7, raw code signatures | 191,971 | 1479, 1237, 1132, 428, 425 | 50 |
| 20x5, raw code signatures | 369,791 | 5580, 4896, 4813, 4097, 1861 | 144 |
| 14x8, license-aware code signatures | 151,763 | 434, 396, 328, 322, 321 | 27 |
| 16x7, license-aware code signatures | 189,811 | 1237, 485, 428, 425, 399 | 49 |
| 20x5, license-aware code signatures | 360,285 | 4896, 4097, 1861, 1614, 1319 | 143 |

Finding: with raw signatures, the largest 14x8 clusters (630, 609) and the largest
20x5 clusters joined hundreds to thousands of **unrelated code files sharing an MIT or
Apache license header**. Fix (`j1.code-signatures.v1`): for code documents, near-dup
signatures are recomputed at dedup time without comment blocks containing license or
copyright language and without Markdown "License" sections; stored text is unchanged.
After the fix every one of the largest 14x8 clusters is a template family (Wikipedia
geography stubs for Benin, Poland and Puerto Rico; "N is an odd composite number"
pages), which is the intended behaviour. Looser banding still chains code through
residual boilerplate (a 485-file cluster at 16x7) and merges larger template families.

## Decision

14x8 with license-aware code signatures. It keeps the FineWeb operating point, has
the best precision of the settings that still detect every pair above Jaccard 0.8, and
produces clean clusters. The accepted cost is lower recall on partial copies
(Jaccard 0.4-0.7); near-duplicates are 0.7% of the pool, and selection draws a
fraction of the pool, so their residual weight in the corpus is small. Revisit with
passage-level deduplication if pilot training shows memorized repeated passages.
