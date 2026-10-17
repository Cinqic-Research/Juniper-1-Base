# Juniper 1 post-training roadmap after the initial SFT pilot (v1, 2026-10-12)

Each stage below is a separate, evidence-gated step with its own admitted dataset version, its own
prospective protocol (baseline, thresholds, decision rule), its own checkpoint lineage rooted in
`juniper1-base-v1`, and its own rollback target. None is authorized by the SFT pilot. Stages are not
combined in one run.

| order | stage | prerequisite | data source (rights) | key evaluation |
|---|---|---|---|---|
| 1 | instruction SFT pilot (`j1-sft-v1.2`; erratum 2026-10-13: this row said `j1-sft-v1.1` / protocol v2, both superseded before measurement) | done this assignment | ledger R00–R04 | protocol v3 |
| 2 | instruction SFT v2 (broader writing, multi-turn; 2026-10-14: v2 `FAILED`, successor SFT v2.1 registered) | pilot qualified; a human-reviewed or teacher-verified source for open-ended answers | OASST1 with filters (R07) or owner-written seeds; local GPT-OSS only for verifiable tasks (R05) | held-out families, human review sample, base retention |
| 3 | reasoning distillation research | stage 1 qualified; new owner authorization for any paid GPU | local GPT-OSS-20B (R05) | Juniper-1-Data `docs/distillation/gptoss_teacher_protocol_v1.md` |
| 4 | tool-call format and no-tool decisions | App token-ID adapter (A-02) implemented with live tests | host traces from Juniper App executing real tools (`host_trace` origin) | schema-valid calls, correct tool choice, no-tool restraint, no fabricated results |
| 5 | tool failures, corrections, untrusted retrieval | stage 4 qualified | host traces with injected failures/denials and poisoned retrieval | recovery behavior, injection resistance with host enforcement measured separately |
| 6 | bounded multi-step agent workflows | stage 5 qualified; App permission model reviewed | host traces | task success under permission limits; zero self-granted permissions |

## Boundaries that every tool stage keeps

- The model writes only `<|assistant|>` messages: `<|tool_call|>` requests in `{name, arguments}` form
  ending with `<|call|>`. The host validates, asks permission, executes, and authors the
  `<|tool|>` result (`juniper-tool-protocol-v1` `HostToolResult`). A model-written tool result or
  claimed execution is never an execution event; `episodes.validate` already rejects episodes whose
  tool results lack host-execution provenance or whose final answers claim execution without one.
- The `juniper-chat-v1` protocol identity does not change silently; a protocol change is a new
  identity and a new tokenizer-level review.
- Tool, injection and permission behavior is scored in two layers: model-only (does it request the
  right thing) and host-enforced (does the App deny what it must). The second layer needs the real
  native runtime; mock servers do not qualify it.
- Analysis-channel training stays off until a stage protocol enables it; llama.cpp ends generation at
  `<|end|>`, so the host adapter must resubmit after analysis (already implemented in
  `j1data.runtime_client`).

## Not planned for Juniper 1

No Decision Model; context extension to 4,096 tokens is a separate future experiment; AAA autonomy
results are not imported into Juniper 1 benchmarks or claims.
