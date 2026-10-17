# Findings ledger — overnight review 2026-10-10 (living document; final copy committed with the review report)

Status keys: FIXED (with commit), OPEN, ACCEPTED (recorded limitation), DISPUTED (with reasoning).

## Training preparation (Juniper-1-Train, branch opus/pilot-contract-v2)

| ID | Sev | Finding | Status / evidence |
|---|---|---|---|
| PP-1 | HIGH | `eval_windows(128)` scored the first token windows per candidate: different documents/bytes per tokenizer | FIXED d026e0d: `full_split_eval`; verify_bpb: 2,583 docs / 12,348,996 bytes identical for all candidates |
| PP-2 | HIGH | shuffled windows truncated to whole updates dropped random windows per candidate | FIXED d026e0d: `--exact-epoch`, consumption record, end-of-run check |
| PP-3 | MED | bpb byte table assumed fixed ID ranges | FIXED d026e0d: `token_byte_lengths` from tokenizer file; real-data re-encode/decode check 0 failures |
| PP-4 | MED | tie-break bytes/s from eval bytes-per-token | FIXED d026e0d (consumed bytes); amendment 15: A100 throughput runs |
| PP-5 | HIGH | no architecture rule; rules accepted incomplete runs | FIXED d026e0d/fe665da: architecture-v1, fail-closed rules |
| PP-6 | HIGH | cut order could drop wide/seed repeats yet report a comparison | FIXED protocol 14 §5 |
| PP-7 | MED | resume drill compared nothing | FIXED: real-data recovery suite on A100 (bitwise) |
| RD-1 | HIGH | readiness generator could report ready with no passing gate (PR #3) | FIXED dfcd233 (independent review APPROVE) |
| SCH-1 | HIGH | corrective-continuation scheduler did not re-base on real data (recovery T9) | FIXED cb2a74d; T9 re-run PASS |
| DRV-1 | MED | Drive drill v1: root exclude of the completion marker also excluded the checkpoint's `COMPLETE` | FIXED: anchored excludes, manifest v2; Drive-only restore and A100 resume PASS (bitwise) |
| TRF-1 | MED | final-state review: "recovery suite 9/9" overstated (8 tests; T9 passed after cb2a74d) | FIXED in contract builder, README, P13 |
| TRF-2–7 | LOW | pending-billing figure; unfilled placeholder; hand-typed 114.3K (114.4K, 28K basis); unsupported 8 % overhead; drill pod-side code identity; ledger duplicates / truncated v3 hash | FIXED (billing reconciled final $25.91) |
| ARCH-1 | INFO | architecture-v1 INCONCLUSIVE (wide clears the band on equal compute only); core provisional; LR grid edges for core/deep | ACCEPTED: owner may order seed repeats and a wider LR grid before freezing |
| PP-8 | HIGH | `pod_guard.py` never stopped the pod; cost from first SSH | FIXED: `pod_controller.py`; live-proven unattended stop of pod z3niewydi1gizw (07:10:49Z) |
| T1/T2 | BLOCKER | controller tests and legacy pod_guard test SIGTERMed live trainers on the host | FIXED 10f1385, fe665da |
| C1/C2 | HIGH/MED | controller loop could die; stop not re-verified | FIXED fe665da |
| R1, S1–S5, A1, N1–N3, H1–H2, M1–M2, L1 | HIGH–LOW | review findings on rules/script/amendment | FIXED (docs/prep/14 §8, 15 §4) |
| D1 | LOW | documents_fully_consumed off by one | DISPUTED and accepted by reviewer (targets are positions 1…n−1) |
| CTL-1 | HIGH | pod-scoped key: Bearer 403, runpodctl unauthorized; Python-urllib UA 403 | FIXED 618a8e4/8fdbd89/1ab04c7 (api_key query parameter, PID-1 env, User-Agent) |
| CTL-2 | MED | controller ledger on container disk lost at stop when no FLOWBOX pull loop ran | FIXED by procedure: pull loop per pod |
| SYN-1 | MED | checkpoint manifests labelled real-data runs `qualification-synthetic` | FIXED fe665da |
| SPEC-1 | MED | implementer proposal 06 acted as v0.7 authority with an interim topology lacking Drive | FIXED e942abe: reconciliation record 16; 06 marked dated history |
| PG-1 | LOW | legacy pod_guard test flaky under load (host-process count) | ACCEPTED: opt-in only |
| FA-1 | INFO | A100 sdpa_correctness: flash `pass: false` from causal backward prefix grad diff 3.05e-5 | OPEN: to be explained in the qualification record |

## Data (Juniper-1-Data, branch opus/qualification-2026-10-10)

| ID | Sev | Finding | Status / evidence |
|---|---|---|---|
| CNT-1 | GATE | independent provisional count check NOT VERIFIED | PASS on v3 (quarantined), v5 prefixes and the proposed v0.2.1 24K shards (sampled unbatched re-encode + full reconciliation; not a full recount) |
| CON-1 | HIGH | v3 matcher blind to short/structured/code items (internal suites 2.1–39.3 %) | layers-v1 (all-token 10-grams, identifiers), calibrated; blind spots recorded |
| CON-2 | HIGH | coverage denominator counted common grams (145 items undetectable) | FIXED afaa6bc (review F3) |
| CON-3 | MED | adjudication criteria v1 written after results, no precedence | FIXED 6930a7f (criteria v2 before corrected results) |
| CON-4 | HIGH | tokenizer sample contained prohibited documents (1 under v1; 18 under v2) | v3 and v4 candidates quarantined; v5 rebuilt from the clean sample |
| CON-5 | MED | pilot prefix contained 21 prohibited documents under criteria v2 | v5 prefixes built with `--exclude-doc-ids` (recorded) |
| CON-6 | MED | controls weak (tautological transforms, 230 negatives) | PARTIAL: 3,000 train negatives run (0.27 % flagged); paraphrase/partial blind spots recorded (ACCEPTED limitation) |
| TOK-1 | INFO | tokenizer retrain on the recorded sample not byte-reproducible (2–6 vocab entries) | ACCEPTED: recorded; identity carried by the frozen bytes |
| SUC-H1 | HIGH | successor manifest copied the parent's build identity: parent-bound gate/dedup/selection records would validate for v0.2.1 | FIXED 06fd5a2 (lineage-derived identity; gates refuse inherited identity); v0.2.1 re-derived, first derivation quarantined |
| SUC-M1 | MED | successor tool did not fail closed on excluded IDs absent from the parent | FIXED 06fd5a2 |
| SUC-M2 | MED | doc claimed the prefix self-test matched exactly while its record says `ok: false` | FIXED (doc states the pre-fix `splits_present` cause; per-split checks all true) |
| SUC-N5/N6 | LOW | failed successor run leaves `<out>.partial`; gate checks lineage self-consistency but not the on-disk parent manifest SHA (verified by hand: matches) | OPEN (low; final-state review APPROVE at a8a8cc1) |
| SUC-L1–L4 | LOW | stale inherited blocks; hash separator convention; hard links; internal-suite removal reach | FIXED 06fd5a2 / documented |
| F7/F8/F9 | LOW | count-check coverage gaps; scan reuse identity; HE+/MBPP+ test fields unprotected | F7/F8 FIXED; F9 OPEN |

## Cross-repository (independent §13 review)

| ID | Sev | Finding | Status |
|---|---|---|---|
| A1 | HIGH | no App-side juniper-chat-v1 template/parser; "no App changes required" claim wrong | OPEN (contract open item) |
| A2–A9 | MED–LOW | parallel tool calls, truncation marker, schema validation, streaming parser, forged results in SFT, U+2060 train/serve mismatch, tests | OPEN (contract open items) |
| B1 | HIGH | AAA #34 v0 transport defects still live; #36 not wired in | OPEN; #34 stays CHANGES REQUIRED |
| B2 | HIGH | AAA surrogate sees hidden labels | OPEN |
| B3 | LOW | AAA b9fd607 breaks protected fingerprint | merge #36 only at 8b36f19 |
| C1–C5 | MED–LOW | website wording; LM-1.1 archive flag/description; App ADR note | OPEN (owner/website) |
