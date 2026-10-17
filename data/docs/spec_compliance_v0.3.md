# Juniper 1 v0.3 specification compliance and remediation status

**Authority:** Juniper 1 Design Specification v0.2 is the design editing baseline;
the supplied v0.3 revision adds mandatory lifecycle checkpointing and recovery
requirements. This record applies both versions, retaining v0.2 requirements except
where the owner explicitly changed them or v0.3 adds a requirement. Neither document
certifies this repository, approves corpus use, or authorizes training.

**Repository:** `Cinqic-Research/Juniper-1-Data`, PR #1. The live PR head was
`445162cf530ba4e14c54730009502c5e327fea0d` when this remediation began. Corrections
described below are local and unpushed until their full verification is complete.

**Current disposition:** **CHANGES REQUIRED.** The historical `j1-base-v0.1` corpus
has not been rebuilt or requalified under the corrected pipeline and v0.3 recovery
rules. Source-use decisions remain pending. No tokenizer training, model training,
cloud provisioning, publication, or merge has occurred.

## Specification-to-repository matrix

| v0.3 requirement | Repository evidence | Status |
| --- | --- | --- |
| Juniper 1 is a new, text-only ~150M model trained from scratch; GPT-OSS-20B is a separate teacher/reference/evaluation model. | Dataset documentation keeps teacher outputs out of the base corpus; no model weights or initialization code are in this repository. | ALIGNED; no model run performed |
| Architecture values and tokenizer sizes are candidates until measured and approved. | Measurement tokenizers are named and labelled provisional; no canonical tokenizer is represented as frozen. | ALIGNED |
| Initial dataset target is about 3B accepted tokens; 15B is a hard ceiling, not a collection target. | Existing `selection.toml` budgets yield roughly 3.4B under a provisional measurement tokenizer; those counts do not establish the final target under a Juniper tokenizer. The historical corpus has not been arbitrarily trimmed. | OPEN; target measurement and selection decision remain gated |
| Base pretraining, continuation, distillation, SFT, refinement, and evaluation remain distinct. | `docs/stages.md` and separate schemas preserve stages; no teacher-generation run was made. | ALIGNED for current data scope |
| CPU/data work publishes immutable shards and transactional progress at least every 15 minutes and at each boundary; release-bound resume re-hashes outputs. | Process v2 now commits immutable candidate/reject/signature segments every five active minutes, journals cursor and input/output identities, records stage counts and accepted/rejected ID samples/hashes, and validates/replays only the uncommitted suffix. Configured release resumes re-hash outputs by default. Tests simulate a mid-unit resume and check content identities. | PARTIAL: process stage implemented; dedup, selection, finalization, and provisional measurement/counting still need 15-minute recovery qualification |
| Restore must preserve data-consumption position, assignments, manifests, and reproducible inputs. | Processing progress records input unit cursor and output identities. Finalized Parquet records document IDs and split labels, but the other data-stage recovery journals are incomplete. | PARTIAL |
| Canonical training requires complete model/optimizer/scheduler/scaler/RNG/sampler state, durable independent replicas, restore comparison, failure injection, and rollback. | This repository has no canonical trainer or qualified A100 storage backend; no such evidence is present. | NOT IMPLEMENTED HERE; blocks canonical training |
| Keep failures and historic research immutable; changed inputs get a new build identity. | v0.1 corpus and the first review are retained; build identity binds output-affecting inputs and the current release uses a separate build id. | ALIGNED in code; corpus requalification remains open |
| Release/export is distinct from resumable training state and requires separate authorization. | No release artifact is created here. Source-use and weight-release decisions remain separate and pending. | ALIGNED; release not authorized |

## Previous review findings ledger

The first review report is historical and unchanged. This successor records the
current disposition without revising its contemporaneous evidence.

| Finding | Current disposition | Evidence / remaining limit |
| --- | --- | --- |
| J1-01 source-admissibility bypass | FIXED IN CODE | Explicit processing validates every requested source; previous regression cases remain. |
| J1-02 build identity, stale markers, and reproducibility | PARTIAL | Identity binding, raw-source preflight, and strict marker inventory checks are implemented. The v0.1 corpus was not rebuilt; full content equivalence/reproduction is NOT VERIFIED. |
| J1-03 invalid acceptance gates | FIXED IN CODE, CURRENT EVIDENCE REQUIRED | Test return codes, freshness, source state, marker inventories, and shard identities fail closed. No new freeze evidence has been generated for the corrected corpus. |
| J1-04 benchmark contamination coverage | PARTIAL | Layered short-item matching and source-file fingerprints are implemented. The rebuilt exclusion index improves item matchability, but important internal sets still have large unmatchable fractions. A read-only scan of the historical accepted corpus against that index is in progress; it cannot by itself requalify the corpus under corrected build inputs. |
| J1-05 source-use and licensing decisions | OWNER DECISIONS PENDING | FineWeb-Edu, FineMath, CC BY-SA treatment, and Stack v2 provenance remain `PENDING`; this work does not change them. |
| J1-06 license-filter false positives | FIXED IN CODE | NC/ND restrictions are tested before broad allowlist patterns, with positive controls. |
| J1-07 corpus-scale memory behavior | PARTIAL | Dedup no longer constructs the sparse all-pairs graph. Representative peak-memory qualification of corrected stages remains NOT VERIFIED. |
| J1-08 GitHub protections and CI | NOT VERIFIED | CI passed on the historical PR head at task start. Branch protection/ruleset API access remains unavailable; final-head CI must be queried after push. |
| J1-09 design authority missing | RESOLVED FOR DOCUMENT ACCESS; IMPLEMENTATION RECONCILIATION IN PROGRESS | v0.3 was supplied and reviewed. The source document does not certify implementation. |
| J1-10 unsafe paths and object IDs | FIXED IN CODE | Path containment, symlink rejection, and Git object-ID validation have regression coverage. |

## v0.3 data-stage checkpoint contract

Processing now writes each segment's candidate, reject, and signature artifacts
before atomically advancing its progress journal. The journal binds the current
build identity, source revision, raw-file SHA-256 (or pinned Git blob set), exclusion
index identity, cursor, segment accounting, deterministic accepted/rejected ID
samples and hashes, and output hashes. The process acceptance gate independently
recomputes each segment's accepted/rejected ID digest and sample from its Parquet
rows. Resume validates committed segment files;
invalid committed files are moved under `work/<build>/quarantine/process-progress/`
before that segment and its suffix are replayed. A complete marker is published
after all segments. Release-build processing re-hashes completed outputs even when
`--verify` is omitted.

For Dolma gzip streams, reaching a saved line cursor requires decompressing the
prefix again; the pipeline does not repeat filtering or output writes for committed
records. This is a remaining recovery-time cost, not evidence of byte-offset seeking.

The other CPU/data stages do not yet have complete transactional journals with a
demonstrated maximum interval below 15 minutes. Existing per-unit markers and
atomic output files do not by themselves meet v0.3 when a unit can run longer than
that interval. The historical v0.1 processing records include units up to 3,379
seconds, so that corpus cannot be retroactively described as meeting the new rule.
Freeze-gate output now includes separate `NOT VERIFIED` results for full CPU/data
stage recovery and canonical A100 checkpoint qualification; the process-stage
`PASS` cannot stand in for either aggregate result.

## Newly recorded engineering findings

| ID | Finding | Status |
| --- | --- | --- |
| J1-11 | Historical process units exceeded the new 15-minute data-stage checkpoint interval (maximum observed: 3,379.08 seconds). | Fixed prospectively in process v2; historical markers preserved and not relabelled. |
| J1-12 | Release processing previously skipped completed markers without output re-hashing unless `--verify` was supplied. | Fixed prospectively: configured release-build resumes verify hashes by default. |
| J1-13 | Dedup, selection, finalization, and measurement/counting do not yet demonstrate v0.3 recovery at the required cadence. | PARTIAL on branch `opus/stage-recovery-2026-10-10` (see `docs/stage_recovery.md`): committed units and automated fault tests for every CPU stage; real-data kill drills only for deduplication, selection and token shards. Open: real-data drills for finalization, counting, audit scan and tokenizer candidates; a full-scale dedup drill; deduplication memory at the 15B ceiling (O(N), about 11-13 GiB projected); a code re-signing fault test; the held-out index marker (plan item 4, a recorded deviation). Independent review APPROVE_WITH_FIXES; fixes applied. |
| J1-14 | Approximately 3.35B historical measure-32k tokens are not a measurement under a frozen Juniper tokenizer and do not settle the initial 3B allocation. | OPEN; do not trim or freeze based on provisional counts alone. |

## Exclusion-index coverage update

The immutable index currently selected by `data/exclusions/CURRENT` is
`6bc49448dc1d75b9b8e6c78c98aa4d1f4293bba972e2f7969e048779837c35d5`. It indexes
105,390 protected items and 6,374,932 informative n-grams. The added exact-short
layer improves item matchability, but the following internal-set coverage remains
insufficient for a blanket evaluation-clean claim:

| Protected set | Matchable items | Unmatchable items | Matchable fraction |
| --- | ---: | ---: | ---: |
| AAA exams | 161 / 5,761 | 5,600 | 2.79% |
| AAA Erudition | 3 / 142 | 139 | 2.11% |
| GPT-OSS qualification | 540 / 3,536 | 2,996 | 15.27% |
| Juniper Reference evaluation | 87 / 1,013 | 926 | 8.59% |
| Juniper App wrapper evaluation | 368 / 3,341 | 2,973 | 11.01% |
| JuniperBench-Code | 1,757 / 4,472 | 2,715 | 39.29% |

These are item-level matcher-coverage counts from the index manifest, not counts of
contaminated training documents. The exact layer only matches complete normalized
short sequences; it does not screen paraphrases. A new read-only full-corpus scan is
being run against the preserved v0.1 accepted shards. Even a zero-hit result would
not eliminate the residual coverage limitation or establish that a corrected build
is reproducible.

## Requalification boundary

The accepted shards, raw acquisition records, and v0.1 reports have not been
rewritten. The corrected pipeline has not yet regenerated a new accepted corpus,
recomputed current corpus statistics, or produced fresh freeze-gate evidence. The
current historical statistics must therefore remain labelled v0.1 and provisional.

Training readiness is separate from dataset engineering. This repository has not
implemented or tested the specification's complete A100 checkpoint state, replica
storage, interruption/failure injection, fresh-process restore equivalence, rollback,
or release-export evidence. No result in this document authorizes canonical
tokenizer training, pretraining, post-training, or public distribution.
