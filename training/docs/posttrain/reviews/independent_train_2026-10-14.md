# Independent Train review and additive errata — October 14, 2026

Reviewed historical heads: Train #29 `ae8af074fe05060d30216bf4a2c148551e3fa510`
against `68dc3b8b85a352558c0b35064a024cd2779b6862`; Train #31
`50043927333cd71d0d9510ebc3dac703bd00fc02` against #29. This is a separate-agent
review under the owner's October 14 directive. The decision ingestion repair
requires distinct review; the historical evidence and protocols remain unchanged.

## Reproduced historical decisions

The three-seed SFT v2 decision was recomputed from retained private per-item score
files. Every historical decision field, including bootstrap intervals and
per-seed statistics, matches `results/sft_v2_v1/decision.json`. Outcome: **FAILED**.
Generalization endpoints pass; arithmetic, injection, hierarchy and JSON fail
their registered margins; continent and element-symbol chat relations also fail.
The preserved historical rule classifies Python **INSUFFICIENT**, which is not a
pass. Nothing in this correction promotes a model.

An independent binomial recurrence reproduced exact McNemar probabilities from
the paired discordant counts. Distillation GSM8K test comparisons, n = 500:

| Arm | C0-only | Arm-only | Gain | Exact p | Holm p | Full-split bpb |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A1 | 16 | 9 | -0.014 | 0.2295229435 | 0.4590458870 | 0.9595118522 |
| A2 | 14 | 9 | -0.010 | 0.4048728943 | 0.4590458870 | 0.9678049102 |
| A3 | 14 | 6 | -0.016 | 0.1153182983 | 0.3459548950 | 0.9613001700 |

C0 bpb is 0.9698878031. All arms fail the unchanged 0.943872 ceiling.
Retention bpb also agrees algebraically with retained aggregate loss, target
count and byte count. This is retained-evidence recomputation, not a fresh model
generation or full-split model evaluation.

The original and corrected #29 seed reports reproduce exactly apart from
normalizing directory strings. All five knowledge-diagnostic aggregates
reproduce from retained per-item booleans. The following narrative corrections
are additive; original reports are historical evidence:

1. `results/sft_replication_v1/README.md` says the smallest slice-adjusted p is
   approximately 0.54 (GSM8K). Across the complete retained report it is
   **0.064453125**, `family:arith.word`, seeds 1–3, n = 200, discordances 9 vs 1.
   GSM8K's 0.5386962891 and P1's 0.7947926894 are correct within their groups.
   No tested slice has Holm p below 0.05; the registered consistency decision
   remains unchanged.
2. The historical knowledge gate fails only for significant worsening in all
   seeds. Absence of that evidence does not establish affirmative retention or
   non-inferiority. Chat knowledge uses a point-estimate margin, unlike the CI
   criterion for other non-inferiority groups. These historical semantics remain
   intact. A successor needs prospective affirmative per-relation criteria and
   an insufficient-evidence outcome for underpowered comparisons.
3. The small dev diagnostic supports an elicitation hypothesis in the observed
   capitals sample. Statements that knowledge is intact or capitals failure is
   exclusively elicitation extend beyond that evidence. The helper generates
   and scores dev items, but reads the sealed answers file containing all
   splits; its docstring's literal “test split is never read” is too broad.
4. Distillation retention and matched-score failures are verified. Explanations
   that the student cannot use GSM8K, or that 318 examples cannot change this,
   are hypotheses. The experiment did not isolate exposure, task complexity,
   replay dilution, target form and optimization sufficiently to prove a unique
   cause. Ten approximate passes and rising retained dev losses support
   investigating exposure/overfitting. No universal two-pass threshold follows.

## Decision ingestion repair

The original loader silently overwrote duplicate IDs and accepted wrong bundle,
scorer and model headers. The decision additionally trusted independently
supplied retention, checkpoint and development files without binding their
candidate identities. Paired ID equality alone did not validate metadata or
coverage against the sealed bundle. These defects can manufacture an apparently
qualified result from malformed or mismatched files, even though the inspected
historical inputs are sound.

The corrected CLI requires `--evidence-bindings` and its externally pinned
`--evidence-bindings-sha256`. The additive binding schema is
`j1.sft-v2-audited-bindings.v1`, with:

- `protocol_sha256`: SHA-256 of the protocol file bytes.
- `eval_bundle_sha256`, `data_code_revision`, `score_engine`, `scorer` agreeing
  with that protocol; `audit_note` stating how missing historical provenance
  was established and the limits of verification.
- `files`: role-indexed records (`v2:SEED:scores`, `v2:SEED:fewshot`,
  `v2:SEED:retention`, `v2:SEED:check`, `v2:SEED:dev`,
  `baseline:SEED:scores`, `base:scores`, `recovery_ab`, optionally `sensitivity`).
  Every record pins file bytes by `sha256`. Score records also pin the model
  identity and `item_metadata_sha256`: canonical SHA-256 of sorted tuples
  `(id, split, variant, family, category)` for the expected sealed-bundle
  selection. Derive this from authoritative prompts, not the observed scores.

The binding is an explicit audit assertion, not proof that the original scorer
recorded provenance that its headers omit. Original headers name
`j1.sfteval.v2`, but do not record the inner `j1.sftscore.v4` revision.
The audit must reconcile retained generation hashes, source revisions,
checkpoint manifests and bundle selection before creating a binding. Missing
provenance must remain explicit. A self-hash alone cannot supply it.

Strict ingestion rejects duplicate JSON keys and item IDs, non-finite numeric
evidence, non-boolean scores, missing items, changed files, paired metadata
disagreement, wrong candidates, missing knowledge pairs, invalid retention and
development evidence, duplicate seed arguments, and unused binding records.
It validates the report self-hash and pins the unchanged historical rule.

With the independently audited historical binding, the corrected loader returns
the exact original FAILED decision plus binding identity/qualification metadata.
The repair changes neither the trainer package nor the frozen scientific
thresholds. Regression checks cover malformed records and binding failures.

## Reproduction and limits

Private review artifacts are retained in
`/home/cinqic/Documents/Juniper-1-Independent-Review-2026-10-14/`:
`train_recompute.py`, `train-recomputed.json`,
`train-replication-recomputed.json`, `train-v2-audited-bindings.json`,
its pinned `.sha256`, and `train-v2-corrected-loader-decision.json`.
Raw prompts, outputs and test answers are not duplicated into this document.

The audited historical score files pass self-hash, generation-hash, model,
bundle, ID uniqueness, expected prompt metadata and boolean-score checks.
Historical coding passes inherit the separately identified scorer-sandbox
limitations; they are not adversarial coding qualification. Item bootstrap
follows the original contract and does not make related task/entity clusters
independent. Fresh final confirmation is required for successor claims.

Original SFT v2 smoke weights were not preserved. Its manifest/log comparisons
are useful historical evidence, with explicitly limited byte-level
reproducibility. Smoke happened after seed 3. Neither a manifest-only comparison
nor its `bitwise` boolean supplies a fresh recovered full-state qualification.
The registered successor must run smoke/recovery before diagnostic training.

The RTX 4090/2060 control is a narrowly observed inference comparison:
6,633/6,634 token-ID matches and identical strict scores. It does not establish
training equivalence across hardware. SFT v2 used trainer package
`f13afbdf0a67d5d8fc8e1d01f2aaa68331155c53f1372e339a06a859e5c2df89`;
the later optional distillation initialization support changes the package to
`63833a59878535c3345b5b9a286716e9145cc3227cc5a90b12fd654951f0c9ab`.
Original exact resumes correctly refuse a different package.

## SFT v2.1 prerequisites

Initialize a uniquely identified experiment from the untouched frozen base and
fresh optimizer. Preserve the numerical recipe unless a separately registered
defect repair requires a change. Recompute admitted v1.2/v1.3 per-family episode
and loss-token exposure under actual shuffling, remainder dropping and the
1226 × 64 update budget. A mixture share is not its realized token exposure.

The historical pilot has 78,464 episode presentations, approximately two passes;
v2 has 78,464 presentations, approximately 1.27 passes. Final target counts are
about 1.54 million versus 1.28 million. Restoring protected-skill exposure while
preserving v2's transferable phrasings/functions is a testable hypothesis,
not an established exclusive cause. Eligible replay needs current admission
and contamination checks. Preserve all excluded coding/test material.

Before launching, bind the successor dataset/code/environment/checkpoints,
realized exposure, fixed diagnostic seed, stopping rule, replication seeds,
independent confirmation, candidate selection and storage/budget/preservation
limits. Old tests already informed development. Split fresh confirmation by
task/entity/conversation/phrasing cluster; separate task uncertainty from seed
spread. Register practical numerical criteria from verified baselines, with
affirmative per-skill/per-relation non-inferiority and a clean preservation test
for v2 gains. Do not choose arbitrary thresholds or launch from this erratum.

## Reproduction command (corrected loader)

The binding is committed at `docs/posttrain/results/sft_v2_v1/audited_bindings_2026-10-14.json`
(SHA-256 `a8e76549799ab9a4e171b0abc652592e0262e5ef05b20c617c0a2c2abc5b5b05`). The command in
`final_report_2026-10-14.md` predates the two required binding arguments:

```bash
python tools/sft_v2_decide.py --protocol docs/posttrain/sft_v2_protocol_v1.json \
  --v2 1 E1 --v2 2 E2 --v2 3 E3 --baseline 1 B1 --baseline 2 B2 --baseline 3 B3 \
  --base-scores BASE/cand_scores.json --recovery-ab docs/posttrain/results/sft_v2_v1/smoke/recovery_ab.json \
  --sensitivity docs/posttrain/sft_v2_sensitivity_v1.json \
  --evidence-bindings docs/posttrain/results/sft_v2_v1/audited_bindings_2026-10-14.json \
  --evidence-bindings-sha256 a8e76549799ab9a4e171b0abc652592e0262e5ef05b20c617c0a2c2abc5b5b05 --out decision.json
```

Lead re-run (Claude Opus 5.5, 2026-10-14 ~12:38 PM EDT; commit 735f6ca at 12:39 PM) on the retained private per-item scores under
`/home/cinqic/j1post/eval/`: outcome `FAILED`, and every field equals the committed `decision.json`
except the two added binding fields. Review roles: GPT-6.1 Sol's lanes implemented this repair; Claude
re-ran it as lead. Same-vendor separation is limited (see the owner decision of 2026-10-14).

## Chronology finding: protocol freeze and baselines

`sft_v2_protocol_v1.json` says the baselines were evaluated "before the protocol was frozen". Commit
`22bc8c5`, which froze it at 2026-10-13 3:46 PM EDT, says it was frozen "before the v1.3-bundle baselines".
Retained score-file modification times on FLOWBOX (`~/j1post/eval/v13_baseline`) fit neither statement.
rep_s1 is 3:28 PM EDT, before the freeze. rep_s2 (3:55 PM), rep_s3 (4:22 PM) and base (2026-10-14
1:40 AM) are after it. Modification times are not independent proof.

Effect: the auto-margin rule (`max(0.03, 2 × SD of the three baseline seeds)`) was registered before
rep_s2/rep_s3 results existed, so it is predeclared. Only seed 1's baseline could have been seen
before the freeze. That seed comes from the earlier v1.2 pilot, whose v1.3-bundle scores were not
inputs to any gate threshold other than this margin. This does not invalidate the FAILED decision,
which comes from regressions far larger than any margin. Neither original record is edited. The Data
repository records the same discrepancy additively in `dataset_v1.3_go_no_go_closure_2026-10-14.json`.
