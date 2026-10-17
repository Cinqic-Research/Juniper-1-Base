# Full independent recount with `cand5-j1tok-24k` (2026-10-10): `j1-base-v0.2.1` and `j1-base-v0.2.2`

Closes the open "sampled, not full" limitation of `independent_count_check.py` (CNT-1) for the accepted successors.
Two runs are recorded: the v0.2.1 run (first section, kept as history; v0.2.1 is **superseded** as the training
input) and the v0.2.2 launch-corpus run (second section).

## `j1-base-v0.2.1` (superseded by v0.2.2 below)

- **Tool:** `tools/full_recount.py` (separately implemented: no code shared with `j1data.tokshards` or the earlier
  sampled check). Phase A re-hashes and re-encodes **every** accepted document one per call with the frozen tokenizer
  bytes (`encode_special_tokens` on, as `jtok.load` requires), decodes it back, refuses control IDs from text, and
  re-encodes a keyed 1 % sample with an independent pure-Python BPE. Phase B recomputes the pack order, offsets, EOS
  positions and per-document ID hashes against the model-ready shards and `shards.json`. Per-file checkpoints make
  it resumable (bounded memory: one parquet file per worker).
- **Inputs:** accepted manifest `8d000906…7968` (build identity `e7909e40…765a`), tokenizer `176a89e2…adf1`, shards
  `shards.json` `cb2e2b53…6ef3`.
- **Result:** `reports/j1-base-v0.2.1/full_recount_cand5-j1tok-24k.json`: **PASS**, 0 errors, 1,345 files, 2,966,479
  documents, 3,439,537,823 document tokens; train 2,961,286 documents / 3,433,653,388 document tokens / 3,436,614,674
  stream tokens (2,961,286 EOS separators); validation 2,583 / 3,227,354 / 3,229,937; test 2,610 / 2,657,081 /
  2,659,691 (test counted mechanically only; no content inspected). Every document round-trips. Pure-Python BPE on
  29,848 sampled documents: 0 mismatches. 1,278 s on FLOWBOX (6 workers, 9 GB cap).
- **Literal control spellings:** 22 accepted documents contain a literal control-token spelling (files in
  `full_recount_literal_control_files.json`). With the frozen loader they encode as ordinary bytes (no control ID in
  any shard). A plain `Tokenizer.from_str` load without `encode_special_tokens` turns all 22 into control IDs, so every
  consumer (trainer, evaluators, runtimes, Juniper App) must use `jtok.load` semantics; this is a launch/interface
  requirement, not a data defect.
- **Reproduce:** `PYTHONPATH=src python tools/full_recount.py <accepted> <shards> <tokenizer.json>
  --expect-tokenizer-sha256 … --expect-manifest-sha256 … --out <dir> --workers 6 --bpe-sample-rate 0.01`.
- **Not covered:** trained-target exposure (a training-side quantity: windows, dropped tails, repeats) is computed
  by the training contract from these stream counts.

## `j1-base-v0.2.2` (launch corpus; added 2026-10-11, review finding D-04)

- **Record:** `reports/j1-base-v0.2.2/full_recount_cand5-j1tok-24k.json` (schema `j1.full-recount.v1`, SHA-256
  `ac29e08120980e677717191fc40cb39706fc796ead65da5bb50c7470ed51dbea`), same tool and procedure as above, fresh
  `--out` (no unit reuse).
- **Inputs:** accepted manifest `54a3adb9…4185` (build identity `89b08d57…bffc5`), tokenizer `176a89e2…adf1`,
  `shards.json` `1cf5e232…3aef`.
- **Result: PASS**, 0 errors, **1,342** accepted files (inventory on disk equals the manifest), **2,959,371**
  documents, **1,484.5 s** on FLOWBOX (6 workers). Document tokens (`phase_a.tokens`) **3,413,019,479**; by split:
  train 2,954,191 documents / 3,407,159,863 document tokens / 3,410,114,054 stream tokens (2,954,191 EOS);
  validation 2,576 / 3,215,628 / 3,218,204; test 2,604 / 2,643,988 / 2,646,592 (test counted mechanically only).
  Stream total **3,415,978,850** = document tokens + 2,959,371 separators = token-file bytes / 2
  (`recount_totals_note.json`). Every document round-trips; no control ID from text; pure-Python BPE on 29,779
  sampled documents: 0 mismatches; by-source and split totals equal `shards.json`; every token file's size and
  SHA-256 equals `shards.json`.
- **Literal control spellings:** 22 documents (counted in the record; the per-file list exists for v0.2.1 only).
  All 22 encode as ordinary bytes under `jtok.load`; a plain load turns all 22 into control IDs
  (`literal_control_docs_plain_load_gives_control_ids` = 22). The loader rule stands for every consumer.
- **Tool changes after this run** (not re-run on v0.2.2; the record above binds the v1 tool's output): schema
  `j1.full-recount.v2` adds a `totals` block (`document_tokens_total`, `separator_tokens_total`,
  `stream_tokens_total`, review D-01), and Phase-A unit reuse under a shared `--out` now verifies the recorded
  `file_sha256` against the current input file (review D-07).
