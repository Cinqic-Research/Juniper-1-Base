> **Export note.** This directory is an allow-listed export of Cinqic's private `Juniper-1-Data` repository
> at commit `2231c7fd` (see `../EXPORT_MAPPING.md`): pipeline source, configs, schemas, tools, tests, manifests and
> reports for the base-corpus builds, and the data documentation. Post-training (SFT/distillation) datasets and
> their manifests are not exported. Corpus bytes are published separately as the Hugging Face dataset
> `CinqicX/juniper1-base-corpus-v0.2.2` (text payloads for seven sources, reconstruction manifests for three).

# Juniper 1 Data

Auditable training-data pipeline for **Juniper 1**. The design baseline is the
Juniper 1 Design Specification v0.2, with v0.3 (8 October 2026) adding lifecycle
checkpoint and recovery requirements, alongside the applicable owner directive;
`j1-base-v0.1` is the first corpus version, not the specification version.
The initial dataset target is about 3B accepted tokens; 15B is a hard ceiling, not a
collection target. Provisional measurement-token counts do not certify that target.
**Status (2026-10-16):** the launch corpus `j1-base-v0.2.2` was the approved training input for the
`j1-base-r2` pretraining epoch (owner training approvals of 2026-10-08; qualification
`docs/qualification_2026-10-10.md`). The resulting model is published as **Juniper 1 Base**
(GitHub `Cinqic-Research/Juniper-1-Base`, Hugging Face `CinqicX/juniper1-base`), and the corpus's public
edition as `CinqicX/juniper1-base-corpus-v0.2.2` under the source-by-source decisions recorded in
`configs/owner_decisions.toml` and `docs/release/`. Juniper 1 R&D is closed; this repository holds the
pipeline and its records. Historical gate text below (`docs/spec_compliance_v0.3.md`) describes the
state before launch and is kept unchanged.

This repository holds the machinery, configuration, schemas, manifests and reports.
Corpus bytes live outside Git under a data root (`J1_DATA_ROOT`) on large storage.

## Review roles

From 2026-10-16:

- **Implementer: Claude Opus 5.5.** Builds, tests, and documents changes, and prepares
  the evidence. It may fix defects, but it never certifies its own work as
  independently reviewed.
- **Independent technical reviewer: Claude Fable 5.1,** in a separate session with a
  fresh context. It reviews the exact head and the primary evidence itself, and may
  reject work, require changes, or withhold approval. If it authored, repaired, or
  directed the exact change, it says so and the change stays pending review.
- **Owner: Markus Gillyard.** Decides direction, funding, releases, irreversible
  operations, and policy exceptions. A technical approval is not owner authorization.

Both models are Anthropic's, so this is separate-session review, not
vendor-independent assurance. Reviews by earlier reviewers (GPT-6 Luna, GPT-6 and
GPT-5.6 Sol, Codex) remain historical records. This section does not change any review
gate.

A dataset is not approved because another model reviewed its construction. Sealed
evaluation material and frozen dataset identities are unchanged.
## Layout

| Path | Contents |
| --- | --- |
| `configs/sources.toml` | Source registry: origin, pinned revision, license state and evidence, selection rules |
| `configs/exclusions.toml` | Evaluation-exclusion registry (benchmarks, Cinqic evaluation sets, canaries) |
| `configs/pipeline.toml` | Build id and every filtering, dedup, contamination and split parameter |
| `configs/measurement_tokenizers.toml` | Provisional measurement tokenizers used only to size the corpus in tokens |
| `schemas/` | JSON Schemas for registry entries and manifests |
| `src/j1data/` | Pipeline stages (see below) |
| `tests/` | Unit, integration and end-to-end miniature-build tests |
| `manifests/` | Committed copies of acquisition, exclusion, dedup, accepted and tokenizer manifests |
| `reports/` | Measured corpus statistics and audits |
| `docs/` | Data card, design decisions, reproduction guide |

## Stages

```
acquire     pinned downloads (HTTP Range resume, LFS SHA-256 verification) -> data/raw
exclusions  layered evaluation index from pinned benchmark revisions        -> data/exclusions
process     selection, normalization, privacy, contamination, quality      -> work/<build>/{candidates,rejects,bands,units}
dedup       global identity / exact / normalized / MinHash-LSH dedup       -> work/<build>/dedup
select      per-source budgets (about 3.4B provisional tokens) from the pool  -> work/<build>/selection
finalize    group-level splits, held-out leakage check, accepted shards    -> accepted/<build>
measure-tokenizers / count-tokens   provisional sizing tokenizers, per-document counts,
            tokenizer-stage sample and probes   -> measurement_tokenizers/, token_counts/, tokenizer_stage/
report / verify              statistics, data card tables, independent recounts
```

Processing writes immutable sub-unit shards and a transactional progress journal at
five-minute intervals. Each segment records its source cursor, accounting, accepted
and rejected ID samples and hashes, and output hashes. Resume validates committed
segments and replays only the uncommitted transformations. A compressed Dolma source
must be decompressed through its saved line cursor to resume; committed records are
not filtered again. Release-build resumes re-hash outputs automatically; exploratory
builds can request that check with `process --verify`. Before processing, selected
source files are re-hashed against their acquisition manifests; Git sources are
checked against the pinned tree. The build identity binds pipeline configuration,
release scope, stable source acquisition identities, exclusion source files and
index, registries, and code through finalization. Use a new build id when any of
those inputs change.

The v0.3 recovery contract also covers tokenizer, architecture, training, evaluation,
and export stages. Those external training-stack save, restore, replica-durability,
and failure-injection gates have not been implemented or qualified by this dataset
repository. (Historical note: this paragraph predates the 2026-10-11 launch; the training stack's
own recovery evidence is in Juniper-1-Train.) Current coverage and remaining data
stage gaps are tracked in `docs/spec_compliance_v0.3.md`.

The accepted normalized text is the canonical corpus. This repository does not
design, select or freeze the Juniper 1 tokenizer and writes no token-ID training
shards; token counts come from clearly labelled provisional measurement tokenizers
and exist only to size the corpus. The tokenizer is a separate follow-up stage.

See `docs/reproduction.md` for exact commands. `docs/data_card.md` describes the
historical v0.1 corpus; `docs/spec_compliance_v0.3.md` records current qualification
and remediation status.
