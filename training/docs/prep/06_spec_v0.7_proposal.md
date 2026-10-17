# Juniper 1 Design Specification — proposed revision v0.7 (delta from owner v0.6)

> **Not a current authority (2026-10-10).** The owner issued the actual v0.7 (Four Destination Backup,
> DOCX SHA-256 `121fcb1d…92ee3`) after this proposal was written. Its §7 interim topology (no Google Drive)
> conflicts with v0.7 and is superseded. See [`16_v0.7_reconciliation.md`](16_v0.7_reconciliation.md).
> The text below is preserved unchanged as dated history.

- **Status:** PROPOSED by the implementer for owner review. It is not an approved specification.
- **Base:** owner-issued v0.6 (`Juniper_1_Design_Specification_v0.6_A100_SXM80GB.docx`,
  8 Oct 2026, received during this assignment). v0.6 is v0.5 plus §8 "Strict time logging and
  temporal traceability". Section numbers below follow v0.5/v0.6 §1–7 and v0.6 §9 (evaluation,
  formerly §8). Every v0.2–v0.6 requirement remains in force except where a row proposes an
  explicit, versioned change.
- **History:** this delta does not rewrite earlier claims. v0.5 statements stay as written,
  and the evidence produced since then is cited here.

| § | v0.5 text (summary) | Proposed v0.6 | Evidence / reason | Status |
|---|---|---|---|---|
| 1 | Purpose: ~150M text-only model for reasoning and agentic workflows | Adopt the frozen model definition `j1-definition-v1.0`: purpose, supported and unsupported uses, language policy, hardware targets, provisional latency objectives L1–L4, licensing table | `01_model_definition.md`; owner directive of 2026-10-08 §3 | proposed |
| 1 | Licensing: Apache-2.0 intended, subject to review | Training-use approvals recorded for FineWeb-Edu, FineMath, CC BY-SA docs, Stack v2 (owner, 2026-10-08). Weight-release treatment and dataset redistribution are separate `PENDING` release decisions | Juniper-1-Data `configs/owner_decisions.toml` | training use decided; release pending |
| 3 | Tokenizer: byte-fallback BPE, 24/28/32K, ~64 reserved control IDs | `j1.tokenizer.v1` layout: IDs 0–63 control (12 named, 52 reserved, no meaning), 64–319 byte fallback, then learned. Totals are inclusive and multiples of 64. `encode_special_tokens` is mandatory at load because `tokenizer.json` does not persist it | `jtok.py`; `test_flag_is_not_persisted_so_load_must_set_it` | implemented; vocabulary size pending pilots |
| 3 | Protocol tokens proposed; termination semantics open | Adopt `juniper-chat-v1`: `<\|end\|>` for non-terminal messages, `<\|call\|>` for tool requests (never a completed turn), `<\|eot\|>` for turn completion. The host-only `<\|tool\|>` carries App `HostToolResult`. Grammar, malformed-output rules, loss masks and context budgeting as specified | `04_protocol_juniper_chat_v1.md`; Data tests (protocol and App contract) | implemented |
| 3 | Context budget: App budgets the window | Default 2,048 budget table (system+developer 512, generation 384, analysis 256, per tool result 384) to be frozen with the post-training data | protocol doc §7 | proposed default |
| 2 | Architecture candidates; QK-Norm "placement and sharing to be specified" | QK-Norm = learnable per-head RMSNorm over head_dim, applied to q and k after projection and before RoPE, not shared across layers. Weight decay on 2-D projection matrices only; norms and tied embedding excluded. Initialization N(0, 0.02), residual output projections scaled by 1/√(2L). These are now explicit config fields, not implicit defaults | `j1bench/model.py`; `test_param_groups_exclude_norms_and_embedding` | implemented; shape choice pending pilots |
| 5 | 3B accepted tokens counted with frozen tokenizer | Corpus `j1-base-v0.2` replaces v0.1 for every training use. v0.1 had 97 accepted documents matching protected benchmark items under index `6bc49448…`. v0.1 stays preserved as historical evidence | Juniper-1-Data audit scan; rebuild records | rebuild in progress |
| 5 | Packing policy to be recorded | Pretraining packs EOS-separated documents in keyed-hash order. Cross-document attention is allowed within a 2,048 window and every position is trained. Document masking is not used in base pretraining; it is re-evaluated before any 4K extension | `tokshards.py`, `TokenStream` | proposed |
| 6 | Optimizer/schedule chosen by pilots | AdamW β 0.9/0.95, ε 1e-8, wd 0.1 (matrices), clip 1.0, warmup then cosine to 10 %. Peak LR from the pre-registered sweep | `08_evaluation_framework.md` §3.4 | pending pilots |
| 6 | Checkpoint cadence 500 updates / 15 min | Unchanged. Clarification: best-evaluation saves also reset the cadence counter, so regular saves are never more than 500 updates or 15 active minutes apart | `test_train_tokens.py` | clarified |
| 6 | Checkpoint state restore | `trainer.pt` is loaded with a restricted unpickler (tensors plus numpy RNG types). Arbitrary pickle code is refused | commit `a06491a` | implemented |
| 7 | Three failure domains + independent backup of the archive | **Interim topology** while no storage purchase exists: FLOWBOX HDD (canonical) + private Hugging Face repos (independent cloud copy) + GitHub (code and manifests). The degraded guarantees, residual risks, stop conditions and migration path are stated explicitly | `05_storage_design_revision.md` | proposed; owner approval required |
| 7 | Receipts and audit log | Proof levels `COMMITTED` / `REMOTE_HASH` / `BYTES_VERIFIED`. Only `BYTES_VERIFIED` proves recoverability | `j1bench/store.py`, `tests/test_store.py` | implemented |
| 9 | Evaluation thresholds not approved | Pre-registered selection rules for preparation pilots. The four evaluation layers are kept separate. Capability thresholds stay unapproved | `08_evaluation_framework.md` | proposed |
| App B | Optimized profile vs reference | Unchanged. The `torch.compile` default profile remains *experimental* until held-out quality equivalence is shown on real data | qualification report; pilots | pending pilots |
