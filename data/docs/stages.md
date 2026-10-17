# Dataset stages

The v0.3 design specification requires separate, independently versioned collections
and tested recovery at each stage boundary.
Each stage has its own identity and output location; nothing moves between stages
implicitly, and exposure (how often a stage's tokens are seen) is a training-run
decision recorded separately from the unique corpus.

| Stage | Identity | State | Location |
| --- | --- | --- | --- |
| Base pretraining | `j1-base-v0.1` (historical identity in `work/j1-base-v0.1/build.json`) | Historical artifact exists; it has not been requalified under the current pipeline identity or layered exclusion index, and owner training-use decisions remain pending. | `accepted/j1-base-v0.1/` |
| Technical continuation | not assigned | **Not built.** A declared selection over `j1-base` can increase repeated exposure without adding new unique corpus tokens. Newly acquired and accepted sources are unique corpus data and count once within the initial 3B-token allocation, even if assigned to continuation; their build and exposure identities must remain separate. | — |
| Reasoning distillation | teacher runs `synthetic/runs/<run id>` (`j1.teacher.v1`) | **Not built.** Pipeline, schemas, oracle checks and tests exist. Distillation may be allocated within the initial 3B-token budget, but this remediation work reserves teacher generation for explicit owner authorization; no generation was initiated. | `synthetic/` |
| Agent and instruction SFT | `j1.episode.v1` schema and validator; `j1data.sft.encode_text_episode` for text-only preparation fixtures | **Preparation only.** The adapter rejects unapproved source licenses, unaccepted records, tool traces, attachments, retrieval and analysis; no SFT dataset or parameter updates exist. Executed-action examples still require genuine Juniper App host traces and a separately qualified serializer. | — |
| Refinement | — | **Not built.** Failure-directed sets follow pilot evaluations. | — |
| Evaluation | exclusion registry index (`exclusions/<identity>`) | No Juniper 1 evaluation suite exists yet. Known Cinqic and public evaluation sets are registered for exclusion; a new suite is added to `configs/exclusions.toml` before use and accepted shards are re-checked. | `exclusions/` |

## Rules that hold across stages

* Registered evaluation material is filtered from the base corpus by long-item
  coverage, exact matching for eligible short items, and canaries. The evidence is not
  a universal certificate; each later dataset or generated-prompt stage needs its own
  exclusion identity and verification.
* Teacher output never enters base pretraining.
* Generated examples are accepted only through an oracle independent of the
  teacher, and their factuality label names that oracle.
* A tool result in an episode must carry host-execution provenance; the model's
  own claim of an action is never a positive example of that action.
* Assistant turns alone carry training loss.

## Recovery readiness

CPU data-stage recovery (gate J1-13) is recorded in `docs/stage_recovery.md`: the
per-stage inventory, the pre-registered test plan and pass criteria, and the results.
Processing, token-shard generation, deduplication, selection, finalization, counting and
the contamination audit scan commit bounded units with hashed outputs and markers written
last, and have kill/replay/corruption tests; tokenizer training (library without
resumable state) is a bounded per-candidate restart. Deduplication still keeps O(N)
global arrays, so its memory scales with the corpus (see that record). The training,
evaluation, and export stack is outside this repository and has no demonstrated
save/restore, durable-replica, or failure-injection gate. Freeze-gate output keeps the
aggregate CPU/data recovery gate at NOT VERIFIED until an independent review accepts this
evidence.
