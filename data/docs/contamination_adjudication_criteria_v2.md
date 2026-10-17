# Contamination adjudication criteria v2 (layers-v1 hits)

- **Written:** 2026-10-10 ≈ 07:30Z, after independent review finding F5 (criteria v1 were written after the
  first hits were seen and stated no precedence). v2 is stated before the corrected (F3) pass-2 results.
- **Applies to:** every Layer A / Layer C hit of `tools/contamination_layers.py` (index `eddb2f37…` or successors).

1. **PROHIBITED (takes precedence over every exception below).** The document reproduces protected item
   content in any of these forms, regardless of length:
   a. a benchmark question/task statement together with its correct answer, solution, or test;
   b. a question with its answer choices;
   c. a solution or test body (code or worked solution) of a protected item;
   d. any protected item text of at least 10 tokens copied with coverage ≥ 0.6 of its non-common grams
      unless rule 2 applies **and** none of 1a–1c applies.
2. **NOT PROHIBITED** only if none of 1a–1c applies and the match consists solely of: generic short phrasing
   (≤ 12 tokens) without the item's answer; file paths; public API or library names; trivial language idioms
   (≤ 3 lines, no task-specific logic); systematic chemical names; platform triples; public constants
   (digest of the empty string, documented example keys).
3. Uncertain cases are PROHIBITED.
4. A PROHIBITED document is removed from the successor accepted identity and from any dependent artifact's
   inputs; dependent artifacts are rebuilt from clean inputs (work order §6).

## Corpus-scale successor rule (stated 2026-10-10 ≈ 12:55Z, before the full-corpus pass-2 result)

Per-document review of every hit across 2,967,034 documents is not feasible, so criteria v2 rule 3
("uncertain → PROHIBITED") is applied in bulk to the calibrated full-corpus scan (`scan_full_pass2.json`,
tool `afaa6bc`+, index `eddb2f37…`, common grams df ≥ 10):

1. Every document with any Layer A hit (coverage ≥ 0.6 of non-common grams) or any `common_only` hit is
   PROHIBITED, except the six documents individually adjudicated NOT_PROHIBITED in
   `adjudication_pilot_inputs_v2.json`.
2. A document whose only hits are Layer C identifiers is NOT_PROHIBITED only if every identifier is one of the
   15 generic identifiers listed in that adjudication; any other identifier makes it PROHIBITED.
3. PROHIBITED documents in any split are removed; the successor accepted identity is derived from v0.2 by
   manifest-level exclusion with the exclusion list hashed into the identity; split assignments of remaining
   documents are unchanged (group-keyed), and the selection/split/count/packing checks are re-run on the result.
