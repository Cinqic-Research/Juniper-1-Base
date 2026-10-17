# Juniper 1 later stages: bounded agentic workflows, context extension, AAA boundary (2026-10-13)

> **Update 2026-10-14:**
> - SFT v2 `FAILED` its registered non-inferiority gates (`final_report_2026-10-14.md`), so it is not
>   the qualified instruction-SFT base that section 1 assumed. That prerequisite now falls to a
>   successor stage, SFT v2.1 (`sft_v21_protocol_v1.md`), or later.
> - Juniper-1-Data#11, Juniper-1-Train#29 and #31 are merged. Juniper-App#46, #47, Juniper-1-Data#10 and
>   Juniper-1-Train#28 remain open.
> - The proposals below are otherwise unchanged.

Owner directive 2026-10-13, sections 12–13. None of these stages is authorized to change parameters
yet; each needs its prerequisites, its own registered protocol and a stage authorization record. This
document holds the concrete proposals and records what is already prepared. Times are US Eastern.

## 1. Bounded agentic workflows (Phase G)

**Prerequisites (not yet met):** the native token-ID adapter qualified and merged (Juniper-App#46, draft);
tool-use SFT qualified under `tool_use_stage_protocol_v1.md` (Juniper-1-Train#28; data `j1-tool-v0.1`,
Juniper-1-Data#10; host trace harness, Juniper-App#47); a qualified instruction-SFT base for it
(SFT v2, this assignment). Until then: no agentic training or claim.

**Already prepared:** real host-executed traces through the App's own `tools::execute_call` and permission
logic (3,789 calls incl. denials, invalid arguments, timeouts, loop limits); a tool serializer that trains
only the model's call text and final answers; a scorer that separates model behaviour from host enforcement.

**Workflow design.** Small tasks with an observable completion criterion and two to four dependent calls,
built only from approved host tools (calculator, unit conversion, current time; memory/file tools under an
explicit policy fixture), e.g. "convert three measurements and report their sum in metres", "compute the
weekday N days from now", with recoverable errors injected by the host (invalid argument → corrected
retry; denial → honest stop). Fixed bounds enforced by the host, not the model: ≤ 4 rounds, ≤ 3 calls per
round, ≤ 2 retries per call, a token budget per turn, a wall-time limit, cancellation at any round, no
persistent memory writes, no new tools. Every execution keeps its host call ID; duplicate or stale call IDs
are refused; the App audit trail records each request, decision and result.

**Evaluation (fresh workflows, separate task identities):** end-to-end success; intermediate tool errors;
recovery rate; false completion (claims success without the host result); unnecessary actions; permission
violations (host-blocked attempts counted separately from model compliance); denied/unavailable tools;
malicious tool or retrieval text; against two controls: the same model without tools and a single-call
policy. A good single call is never reported as agency.

**Boundary:** the model never grants itself authority, creates trusted results, runs in the background or
modifies itself. The owner's broader Agent concept is a possible Juniper 2 direction, not a new project here.

## 2. Context extension beyond 2,048 tokens (Phase H)

**Not started; only after the immediate SFT sequence.** A real extension needs: a distinct configuration
and checkpoint lineage (base and validated checkpoints untouched); long-context training exposure on
eligible data (documents > 2,048 tokens from the accepted corpus, which exist in `j1-base-v0.2.2`); RoPE
treatment chosen by a controlled comparison (e.g. position interpolation vs base-frequency change) rather
than an inference-time setting; positional/numerical checks (attention entropy, logits on long inputs);
task-relevant long-range evaluation (retrieval at depth, multi-document QA built from held-out validation
documents); memory and latency measurements on the RTX 2060 and one rented GPU; and full 2,048-token
regression (validation bpb ≤ 1.03× base, the SFT test suite). Estimated cost: a few hundred continuation
updates at 4,096 tokens (single GPU hours, not days); a stage authorization with its own budget first.
Architecture or training changes are considered only for a measured limitation with a controlled
comparison; no new architecture search, pretraining restart or large data collection.

## 3. AAA boundary (unchanged)

State checked 2026-10-13: AAA#34 open (head `507dc415`; unresolved bearer-credential URL/redirect handling,
a hidden-label training surrogate and a selection-rule violation), AAA#36 draft (prospective loopback-only
transport; does not qualify the science), cinqic.com#22 open (depends on the AAA research disposition),
Juniper-1-Train#2 draft (experimental, conflicting). This assignment does not integrate AAA, edit its
frozen evidence, approve its PRs or make product claims from it.
