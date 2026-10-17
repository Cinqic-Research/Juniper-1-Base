# Contamination finding F9: HumanEval+ / MBPP+ test fields (pre-registration)

- **Written:** 2026-10-10, before any F9 index was built or any F9 scan was run. Commits after this one
  may change only the tool and record results; they do not change the rules below.
- **Finding F9** (`docs/qualification_2026-10-10.md` §4): the EvalPlus test fields are not in the
  contamination reference index, so a training document containing those test cases would not be detected.
- **Tool:** `tools/contamination_f9.py` (built on `tools/contamination_layers.py`, `j1.contamination-layers.v1`).

## Inputs (no download)

The suite files are the registered ones (`configs/exclusions.toml`), already on disk under
`data/exclusions/raw/<id>/<revision>/`. Before use, each file's SHA-256 must equal the hash in the layers-v1 items
cache provenance (source fingerprint `820db74f…`). A missing file, a hash mismatch, a missing column, a
non-string value, or an empty result stops the tool (fail closed).

| Benchmark (revision) | File | SHA-256 |
|---|---|---|
| `evalplus/humanevalplus` (`d32357cf…`) | `data/test-00000-of-00001-5973903632b82d40.parquet` | `4436f5c0…` |
| `evalplus/mbppplus` (`b2d74c91…`) | `data/test-00000-of-00001-d5781c9c51e02795.parquet` | `dc20030b…` |
| `google-research-datasets/mbpp` (`4bb6404f…`) | `full/{prompt,test,train,validation}-00000-of-00001.parquet` | as recorded |

In these revisions EvalPlus has no separate `base_input` / `plus_input` columns. The base and plus inputs and
the expected outputs sit inside one `test` string, together with a shared assertion harness:
`inputs = [...]`, `results = [...]`, and a loop that calls `assertion(candidate(*inp), exp, atol)`.

## Fields

| Field | Indexed before F9 | F9 action | F9 label |
|---|---|---|---|
| humanevalplus `prompt`, `canonical_solution` | yes | unchanged | — |
| humanevalplus `test` (base + plus inputs, expected results, harness) | **no** | **added** | `humanevalplus-test` |
| humanevalplus `task_id`, `entry_point` | no | not added (name/identifier only) | — |
| mbppplus `prompt`, `code`, `test_list` | yes | unchanged | — |
| mbppplus `test` (base + plus inputs, expected results, harness) | **no** | **added** | `mbppplus-test` |
| mbppplus `test_imports`, `source_file`, `task_id` | no | not added (import lines, notebook name, number) | — |
| mbpp `text`/`prompt`, `code`, `test_list` | yes | unchanged | — |
| mbpp `full/*` `challenge_test_list` | **no** | **added** (asserts joined by `\n`, as `test_list` is) | `mbpp-challenge-tests` |
| mbpp `full/*` `test_setup_code` (non-empty values) | **no** | **added** | `mbpp-challenge-tests` |
| humaneval `test` | yes (base tests) | unchanged | — |

## Reference items

Each added field value becomes:

1. **one whole-value item** (the value as stored), and
2. **window items**: the value is NFKC-normalized and case-folded, then tokenized with the layers-v1 token regex.
   Windows of **W = 64** consecutive tokens are cut with stride 64. The last window is the final 64 tokens, so it
   may overlap the one before. Each window item is the substring of the normalized text from its first token's
   start to its last token's end. A value shorter than 64 tokens produces only the whole-value item.

Windows exist because the EvalPlus values are long: a median of 18,097 tokens (HumanEval+) and 2,900 (MBPP+),
with a maximum of 412,413. With a single whole-value item, a partial copy (for example only the `inputs` list)
would stay below 0.6 coverage. Any verbatim copy of at least 2·W = 128 contiguous tokens contains a whole window.
Identical item texts under the same label are deduplicated. Items that come from the shared harness end up as
common grams, or else as hits. Both outcomes are reported below.

## Matching (layers-v1, thresholds unchanged)

- All-token 10-grams, at least 3 alphanumeric tokens per gram, NFKC + casefold. A Layer A hit is coverage ≥ **0.6**
  of the item's non-common distinct grams. Items whose grams are all common are reported as `common_only` at ≥ 0.6
  of all their grams. Layer C covers identifiers (≥ 16 chars of `[0-9a-z_-]`, at least one digit and one letter).
- **Common grams:** pass 1 scans every document of `j1-base-v0.2.1` (train, validation, test; manifest `8d000906…`)
  and records each F9 index gram's document frequency. A gram is common when df ≥ **10** (the same pre-registered
  rule as layers-v1). Pass 2 applies the common set.
- Pass 2 is computed from the in-index gram set and identifiers that pass 1 records for every document holding at
  least one F9 gram or identifier. That equals a re-scan, because a document with no F9 gram cannot reach coverage
  > 0, and a test checks the equality. The test split is read only mechanically: only counts and IDs are reported.

## Controls

- Per label, 200 sampled F9 items (fixed seed, every item eligible) are planted at a line boundary inside a
  random **train-split** document. Variants: `verbatim`, `case_space` (upper case, varied spaces), `reindent`,
  `comma_spacing` (spaces after commas removed or doubled), `truncated_80` (first 80 % of the item's tokens).
  A plant is detected when the match attributes it to that label (Layer A, common_only, or Layer C).
- Negatives: 3,000 unmodified train-split documents. The flagged rate is reported; flagged ones are candidate true hits.
- Controls run uncalibrated before pass 1 and calibrated (with the F9 common set) after it.

## Adjudication: criteria v2, unchanged

`docs/contamination_adjudication_criteria_v2.md` applies as written, including its corpus-scale rule:

- Every F9 item is test content of a protected item, so any Layer A or `common_only` hit reproduces a test body
  (rule 1c) and is **PROHIBITED**. The rule-2 exceptions require that none of 1a–1c applies, so none is available.
  The six individual layers-v1 NOT_PROHIBITED adjudications concern other items and do not carry over.
- A document whose only F9 hits are Layer C identifiers is NOT_PROHIBITED only if every identifier exactly equals one
  of the generic identifiers listed literally in
  `reports/qualification-2026-10-10/contamination/adjudication_pilot_inputs.json` (`layer_c_identifiers.values`).
  The four entries recorded there only by description (empty-string digest, AWS example key, `123456789abcdef0`,
  `md5('password')`) count only when the identifier is that exact public constant. Otherwise the document is
  PROHIBITED.
- PROHIBITED takes precedence; uncertain cases are PROHIBITED; this applies in any split. Validation and test hits are
  reported as counts only.

## Outputs

- `reports/contamination_f9/` (committed): `prohibited_f9.json` (doc IDs by split and source, with reasons) and
  `summary_f9.json` (index identity, field coverage, controls, counts).
- `/media/cinqic/Cinqic Storage/juniper1-archive/autonomous-2026-10-10/results/contamination_f9/` (not committed):
  the index, per-shard pass-1 results (restartable; keyed by shard SHA-256, index identity and tool SHA-256), and
  the full hit list. The F9 index is a new identity under `data/exclusions/layers-v1-f9-<identity>/`. The
  `eddb2f37…` index is not modified.
