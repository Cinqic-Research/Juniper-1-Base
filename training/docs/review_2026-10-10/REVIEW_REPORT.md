# Juniper 1 independent review, qualification and contract preparation — 2026-10-10

**Authority:** owner work order 2026-10-10 (with the $40 acceleration addendum) under the owner-issued
*Juniper 1 Design Specification v0.7 — Four Destination Backup* (DOCX SHA-256
`121fcb1de0dc927ce0d8d725c10bb68d199e28185f699d786472191418292ee3`, verified). This report is the central index;
details live in the referenced records. Canonical pretraining is **not** authorized by any of this work.

**Overall disposition: PARTIAL_QUALIFICATION**

The proposed training contract (`../contract/training_contract_v1.json`) is complete and every measurement in it comes
from recorded evidence. Two things stop it from being ready for the owner's freeze. First, the tokenizer and
architecture selections are provisional: the tokenizer candidates are tied, and the architecture rule came out
INCONCLUSIVE. Second, critical gates remain NOT_VERIFIED (readiness matrix: 3 PASS, 15 NOT_VERIFIED,
2 OWNER_DECISION_REQUIRED). Canonical pretraining is not authorized.


## 1. Where everything is

| Deliverable | Location |
|---|---|
| Findings ledger (all IDs, dispositions) | `FINDINGS_LEDGER.md` (this directory) |
| Baseline lock + readiness matrix | `baseline_lock.json`, `../prep/readiness_matrix.json` (generated from `../prep/readiness_gates.json`) |
| Corpus qualification | Juniper-1-Data `docs/qualification_2026-10-10.md`; criteria `docs/contamination_adjudication_criteria_v2.md` |
| Pilot protocol and amendments | `../prep/14_pilot_protocol_v2.md` (+§8 review), `../prep/15_hardware_amendment_2026-10-10.md` (+§4, §5) |
| Pilot results and selections | `../prep/17_pilot_results_2026-10-10.md`; raw evidence `../../results/pilots-2026-10-10/` |
| Proposed training contract | `../contract/training_contract_v1.json`, `../contract/README.md` |
| Preservation receipts, resource and billing ledger | `PRESERVATION_AND_BILLING.md` (this directory); HDD `juniper1-archive/overnight-2026-10-10/` (`EVIDENCE_INDEX.json`) |
| v0.7 reconciliation | `../prep/16_v0.7_reconciliation.md` |

## 2. Repositories reviewed (live snapshot 2026-10-10 ≈ 06:00Z; PR heads unchanged from the work order)

All ten Cinqic-Research repositories were inventoried (default branch, head commit and tree, branches,
tags, open PRs with heads, reviews and checks; `inventory/gh_inventory_2026-10-10.json` on the HDD). No new
repository appeared. Private repositories do not expose branch protection on the current plan (recorded,
not inferred as approval). The authenticated GitHub identity is the owner account, so no formal approving review
is claimed for any PR; technical review evidence comes from separately tasked reviewers.

## 3. What was found and fixed

See `FINDINGS_LEDGER.md`. Highlights:
- Pilot measurement defects (PP-1…PP-8), including tokenizer comparisons on different validation bytes and
  unenforced equal training bytes; fixed and pre-registered before any result.
- No unattended provider-level budget enforcement (PP-8); replaced by an on-pod controller that stopped pods
  without operator action three times.
- Tests that SIGTERMed live trainers on the host (T1, T2); a readiness generator that could report ready with no
  passing gate (PR #3); a corrective-continuation scheduler defect found on real data (fixed, verified).
- Contamination: the registered matcher left short/structured/code items blind; strengthened layers found
  prohibited documents in the tokenizer sample and pilot prefix; tokenizer candidates were rebuilt twice from
  clean inputs (v3 and v4 quarantined, v5 used).

## 4. Decisions (all provisional; owner freeze required)

| Item | Rule outcome | Proposed | Evidence |
|---|---|---|---|
| Tokenizer | tokenizer-v1.2 DECIDED inside the 0.5 % tie band; tie broken by A100 training bytes/s | `j1tok-24k` (clean-sample v5), **provisional** | `../prep/17…`, `selections/select_tokenizer.json` |
| Architecture | architecture-v1 **INCONCLUSIVE**: wide 20×768 is +0.917 % on equal compute and +0.268 % on equal tokens against a 0.570 % band | core `j1-core-28x640` (143,072,384 params), **provisional**; seed repeats not triggered | `selections/select_architecture.json` |
| Learning rate | lr-v1.1 | core 9.295e-4 (grid edge 0.5×) | `selections/select_lr_*.json` |
| Corpus | corpus-scale successor rule (stated before the result) | `j1-base-v0.2.1` (identity `e7909e40…`, manifest `8d000906…`), 555 train docs removed | Data `docs/qualification_2026-10-10.md` |
| Model-ready shards | independent count check PASS (sampled; not a full recount) | `cand5-j1tok-24k`: 3,433,653,388 train document tokens; one epoch | `shards.json` `cb2e2b53…` |
| Execution profile | no owner-approved promotion threshold | compiled (114.4K vs 56.2K tokens/s eager on real data) with eager as fallback | `a100-ixxkq50q949bhl/profile` |
| Launch budget | computed from measured rates | compiled: 9.32 paid h, $16.69, cap $26; eager fallback: 18.98 h, $33.98, cap $48 | contract JSON |

## 5. Spend and resources

- Provider-billed task spend: **$25.91** for six task pods (`list-pod-billing`, hourly buckets 06:00–15:00Z, final),
  inside the $35 stop of the $40 owner cap (reserve $14.09 unspent). Per-pod amounts are in `PRESERVATION_AND_BILLING.md`.
- All six task pods are terminated (two at creation for a host defect, four at 14:3xZ after a logged eligibility check:
  no volumes, evidence preserved). The owner pods `4i13u1p447t6e7`, `k3mul4cjkmiong` and `dxl7jz3zezlz0m` were not
  started, stopped or deleted. No task resources are billing.

## 6. PR dispositions

| Repository | PR | Disposition |
|---|---|---|
| Juniper-1-Train | #3 `dfcd233` readiness completeness | independent review APPROVE; merged into #4 (`b3ba5c0`) and integrated with it |
| Juniper-1-Train | #4 `opus/pilot-contract-v2` | integration candidate into `opus/pretraining-prep` (exact-head CI and independent review required); merge SHA recorded in the final report and HDD archive |
| Juniper-1-Train | #1, #2 | unchanged (historical qualification evidence) |
| Juniper-1-Data | #2 `722a01d` corrected v0.2 gate report | included in #3 |
| Juniper-1-Data | #3 `opus/qualification-2026-10-10` | final-state independent review APPROVE at `a8a8cc1`; CI green; integration candidate into `opus/pretraining-prep` |
| Juniper-1-Data | #1 v0.1 | kept open and unmerged (CHANGES REQUIRED; historical) |
| AAA | #34, #36 | not merged: #34 CHANGES REQUIRED (B1/B2); #36 is acceptable only at `8b36f19` |
| Cinqic-Backup | #1, #2 | not merged here (outside the integration scope of this order) |
| cinqic.com | #22 | not merged: wording findings C1–C5 are an owner/website decision |

## 7. Coverage and exclusions

- Reviewed in depth: Juniper-1-Train (pilot, trainer, checkpoint, controller, readiness), Juniper-1-Data
  (counting, contamination, tokenizer stage, token shards), Cinqic-Backup (Drive path), AAA #34/#35/#36 transport and
  claims, cinqic.com #22 claims, Juniper-App protocol interfaces (via independent review).
- Not re-reviewed beyond the 2026-10-09 consolidated review: Cinqic-Calculator, Juniper-LM-1, Juniper-Reference-4B
  (archived, historical), Juniper-LM-1.1 internals.
- Not done: Data transactional recovery across all CPU stages (J1-13), complete Drive coverage of every PRESERVE
  artifact, key custody, App-side native protocol adapter, capability evaluation thresholds.
