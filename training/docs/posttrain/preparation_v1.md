# Juniper 1 post-training preparation contract v1

**Status (2026-10-13): superseded by the implemented stage — see [`final_report_2026-10-13.md`](final_report_2026-10-13.md); one bounded SFT pilot ran under [`sft_pilot_protocol_v3.md`](sft_pilot_protocol_v3.md). Original status: preparation only; no parameter-changing run was authorized.** This
contract describes the work needed after an independently approved base freeze.
It does not select a base checkpoint, approve a dataset, qualify a 4,096-token
context, or certify Juniper App integration. The base identifier and its hashes
must be filled from the independent freeze record before any stage contract can
be approved.

**Frozen base (bound 2026-10-12):** `juniper1-base-v1` →
`j1-base-r2/ckpt-000052034-336e5967`, record
[`docs/freeze/juniper1-base-v1.json`](../freeze/juniper1-base-v1.json) with
`freeze_record_sha256` `ef79b324fff459c8ffce7ac2e1b1542960a254b8383a6e8b19fd20402e92ff55`.
A stage loads it only with `j1bench.frozen.load_frozen_base(record, checkpoint_dir,
tokenizer_path=..., expected_record_sha256=<that digest>)`.

## Dependency order and stage boundaries

1. **Base gate:** verify an original full-state checkpoint, its model/config and
   tokenizer identities, independent validation, and an independent restore.
   Record the immutable freeze ID and the exact source checkpoint SHA-256. A
   training-stage loader must reject `latest`, `best`, and an unbound base ID.
2. **Data gate:** admit only versioned episodes that pass schema, provenance,
   rights, decontamination, role/order, tool-result provenance, and task-specific
   validation. Seal input hashes and the admitted/rejected ledger. A teacher
   answer remains teacher output until an independent oracle verifies it.
3. **Protocol gate:** use `juniper-chat-v1` with the frozen `j1tok-24k` tokenizer
   through `j1data.jtok.load`. Test literal delimiter injection in user,
   retrieval, and tool text, loss masks at every message boundary, and malformed
   tool calls. Juniper App must author roles, execute permissions, validate
   calls, and author tool results. Its native adapter remains an integration
   gate; token-ID inference alone does not clear it.
4. **Evaluation gate:** seal the development/evaluation suite before training.
   Keep the corpus test split closed during checkpoint and recipe selection.
   Separate model-only instruction/tool scores from host-enforced App behavior.
5. **Recovery and budget gate:** demonstrate atomic full-state save and exact
   restoration on synthetic fixtures; bind an actual compute/storage budget,
   shutdown mechanism, and preservation destinations before a paid run.
6. **Owner decision:** only a later explicit authorization may start SFT,
   distillation, refinement, continued pretraining, or context extension.

The possible later stages are an initial instruction SFT, a separately
qualified tool-use SFT, optional reasoning distillation, and refinement. Each
stage gets a distinct data identity, objective, optimizer/schedule proposal,
evaluation protocol, acceptance rule, rollback target, and lineage to the
immutable base. No pretraining optimizer or learning-rate setting carries over
by default. None of these proposals has a validated hyperparameter setting.

## Dataset contract

The source format is the protocol-neutral `j1.episode.v1` schema in
Juniper-1-Data. Admission must bind each episode's accepted status, training
flags, source/license and rights decision, decontamination identity, factuality
checks, and host-execution provenance to the exact token IDs and loss labels.
JSON validity alone is insufficient. Publish immutable input and reject
manifests; never silently skip malformed records.

The curriculum should cover American-English direct answers, Python and
general coding, mathematics and STEM, concise verifiable reasoning, JSON and
schema adherence, tool selection and no-tool decisions, Python execution
requests, tool failures and correction, multi-turn context, instruction
hierarchy, uncertainty, and abstention. Include negative controls for forged
role delimiters, fabricated tool confirmations, poisoned retrieval, invalid
arguments, and unsupported factual claims. The mix and target-token counts
remain proposals until the admitted data and a prospective stage protocol are
measured. Do not produce a bulk teacher corpus to satisfy a count target.

For each admitted example, record the source, license/rights decision, teacher
identity if applicable, independent verification method, dedup group, split,
decontamination result, and explicit trainable assistant messages. Host system,
developer, user, and tool-result tokens have label `-100`. Analysis tokens are
masked unless the approved stage explicitly enables them. Packing may not move
labels across episode boundaries. Truncation must either preserve a complete
supervised unit or reject the example; a clipped call/result pair is invalid.
The selected 2,048-token base context is the only current training window.

## Evaluation registration

Before observing any post-training result, version and hash the prompts,
reference answers, scoring code, decoding settings, split IDs, and selection
rule. Report every scored example, including failures and abstentions. Use
held-out base language loss and raw completion probes to detect regression.
Instruction tests assess hierarchy, directness, uncertainty, and structured
outputs. Code uses sandboxed hidden tests; mathematics uses exact answers or
verified equivalence; tool tests use schema validation, permission denial,
no-tool restraint, and correct handling of host failures. Injection tests
distinguish model compliance from App enforcement. The App layer requires a
real native runtime and host integration test; a mock server does not qualify
it. Record contamination coverage rather than calling a partial scan clean.

Do not choose recipes on the final corpus test split or revise acceptance
thresholds after seeing results without a new prospective protocol. An LLM
judge may supply diagnostics but cannot be the sole selection oracle.

## Checkpoint and recovery contract

A future parameter-changing stage must save by the earlier of 500 completed
optimizer updates or 15 minutes, and at stage/evaluation boundaries and before
any contract change. The manifest binds the frozen source checkpoint; model
parameters and buffers; optimizer, scheduler and scaler if used; completed
updates and supervised-target exposure; exact data cursor/packing state or a
proven replay method; Python/NumPy/Torch/CUDA RNG; objective and loss-mask
revision; code/dependency/config identities; evaluation IDs; parent checkpoint;
file sizes and SHA-256. Publish atomically only after all payloads hash and a
clean restore reproduces the expected next batch and output. A changed
objective, tokenizer, data, precision or hardware contract is an explicit new
lineage, never an undocumented exact resume.

Preserve the original base full-state checkpoint. A stage-specific inference
export cannot replace optimizer and recovery state. Test failure injection on
small synthetic fixtures without executing an optimizer update on Juniper 1.
Keep FLOWBOX HDD, Google Drive, GitHub, and private Hugging Face roles distinct;
verify independent recovery receipts before claiming a stage complete.

## Context and runtime boundary

4,096 tokens is a future experiment. Merely raising the configuration limit
does not establish long-context competence. A later approved protocol needs
genuine long-sequence exposure, position/KV-cache tests, long-context tasks,
and 2,048-token regression gates. Until then, serve and evaluate at 2,048.
The current Qwen3-compatible GGUF text path requires tokenizer and control
token parity and App host-adapter qualification before native integration.

## Resource and cost scenarios

Zero-GPU preparation on FLOWBOX covers schema admission, token/mask fixtures,
synthetic checkpoint restore, static interface checks, and scoring-code tests.
It incurs no new RunPod charge. A later single-A100 stage is a separate owner
decision and needs measured useful target tokens per second, checkpoint and
evaluation overhead, storage size, and a complete run/shutdown budget before a
price estimate can be trusted. The connected RunPod account listed an existing
exited single-A100 pod at a configured $1.79/hour on 2026-10-12; that rate is
only a point-in-time input, not a total-stage forecast or an available credit
balance. A two-A100 stage has no canonical approval and is not a default
scenario. The $10 incremental authority for this audit cannot be reused as
future post-training spending authority.

## Unresolved gates at authoring

- Independent base selection, immutable freeze ID, and replica restores.
- Admitted post-training dataset identities, rights, deduplication, and mix.
- Episode-to-token admission and mask verification on production tokenizer.
- Native Juniper App serializer/parser and end-to-end permission tests.
- Post-training scoring fixtures and prospective acceptance thresholds.
- A verified available RunPod balance and enforceable budget for any later run.

No SFT, preference optimization, distillation, continued pretraining, or other
parameter-changing operation was performed to prepare this contract.
