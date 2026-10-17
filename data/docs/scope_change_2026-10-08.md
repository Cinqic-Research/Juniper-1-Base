# Mid-work scope change: 20B+ to 3B+ (owner directive, 2026-10-08)

The original goal asked for at least 20B accepted unique tokens. On 2026-10-08 the
owner replaced it: build a clean corpus of about **3B+ accepted tokens** first;
**15B is a hard ceiling, not a target**; expand only when Juniper 1 training and
evaluation show that more data would help, and only by a new owner decision. The
design specification's 15-30B / 3-10B / 1-3B exposure ranges are superseded with
respect to dataset scale. The tokenizer remains out of scope (owner correction of
2026-10-07).

When the directive arrived, acquisition was complete (about 102 GB raw, all
SHA-256-verified) and the release processing build had completed 55 of 312 units
(all Wikimedia, arXiv, open-textbook, PEP and Cinqic units, and five Stack Exchange
units) before a machine restart.

## Classification of existing work

| Item | Decision | Notes |
| --- | --- | --- |
| Source research, registry, considered-source records | KEEP | Unchanged; the same established datasets are the raw material |
| Acquired raw data (102 GB) | KEEP | Not deleted: verified, legally acceptable, and the basis for evidence-gated expansion; most of it is simply not processed now |
| Processing pipeline, filters, calibration, contamination, privacy, provenance, tests | KEEP | Quality standards unchanged; the build identity did not change, so the 55 completed units remain valid |
| Processing all 312 units | STOP | Replaced by `configs/build_scope.toml`: 30 of 99 FineWeb-Edu units, 24 of 128 FineMath units, 11 of 21 Stack Exchange units (adds math, Stack Overflow, superuser/unix), all 14 code files, all of the small sources |
| Accepted corpus = every deduplicated candidate | MODIFY | New `select` stage (`configs/selection.toml`) fills per-source provisional-token budgets (about 3.4B total) from the deduplicated pool, preferring upstream quality scores where they exist |
| Global exact and near deduplication | KEEP | Runs over the whole processed pool before selection, so no selected document duplicates another |
| 20B storage and time projections; "measure under every vocabulary that 20B is reached" | SUPERSEDED | The corpus is sized against 3B+; provisional counts still use the 24K/28K/32K measurement tokenizers |
| Provisional measurement tokenizers, tokenizer-stage sample and probes | KEEP | Sizing only; no tokenizer is selected or frozen |
| Post-training schemas, teacher pipeline, GPT-OSS pilot prompts | KEEP (dormant) | No teacher generation is run under this goal |
| Dedup calibration, audits, verification, freeze gates | KEEP | Run on the 3B+ corpus |

## Why process more than 3B

The processed pool (about 20B candidate tokens once the scoped units finish) is
larger than the accepted corpus on purpose: per-source budgets are filled with
the highest-scored FineWeb-Edu and FineMath documents and a uniform sample
elsewhere, and global deduplication needs the overlap between sources to be
visible. Units outside the scope are not processed at all.

## Reproducibility follow-up

The statement above that the build identity did not change describes the original
scope-change operation and is retained as historical provenance. The subsequent
identity review found that scope must be bound to the build: the current pipeline
hashes `configs/build_scope.toml`, the selection configuration, stable acquisition
content identities, the exclusion-index source fingerprint, and output-affecting
code. Existing `j1-base-v0.1` outputs were not rebuilt under that corrected identity
and are not evidence of reproducibility under the current pipeline.
