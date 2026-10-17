# Repository inconsistencies found during the audit (2026-10-07)

This is a dated cross-repository snapshot based on specification v0.1. The Oct 8,
2026 owner directive and specifications v0.2/v0.3 supersede it as current authority; the
architecture descriptions below are historical context, not current design claims.

The Juniper 1 design specification v0.1 (7 October 2026) defines Juniper 1 as a new
~150M-parameter model trained from scratch, with GPT-OSS-20B as teacher, reference
and evaluation baseline only. The following repository text predates that decision
and still describes the older architecture. It was **not edited** here: these
repositories are outside this task's write scope, and historical evidence must stay
historical. The owner should decide which statements to update and which to keep as
dated history.

## AAA (`Cinqic-Research/AAA`)

Branch `origin/opus/juniper1-erudition` (state 2026-10-06) states that Juniper 1's
language model is Juniper LM 1.1 / OpenAI's gpt-oss-20b, or that Juniper 1 consists of
Erudition, World and Language models:

- `README.md:10-14`, `:34`, `:40`
- `CONTRIBUTING.md:5-6`, `:217`, `:223`
- `CHANGELOG.md:5-8`, `:52-55`
- `docs/research_direction.md:27-33`, `:148`, `:284-288`
- `docs/juniper1/README.md:12-18`, `:25`; `docs/juniper1/architecture.md:8-16`, `:40`;
  `docs/juniper1/protocol.md:13`, `:66`, `:115`; `docs/juniper1/data.md:12`;
  `docs/juniper1/decisions.md:38`; `docs/juniper1/development_report.md:5`
- `research/aaa_erudition/__init__.py:4-5`, `characterize.py:1`, `lifecycle.py:3`
  (fingerprinted under the spent aaa.erudition.v0 freeze; changing them moves that
  fingerprint)

`origin/main` carries similar claims: `README.md:9`, `:140`; `CHANGELOG.md:5-6`;
`docs/research_direction.md:28-30`, `:166`.

## cinqic.com (`Cinqic-Research/cinqic.com`)

Branch `origin/research/aaa-juniper1-direction`: `research/index.html:32` ("Juniper 1
has three model components: a language model (OpenAI's gpt-oss-20b ...)") and
`assets/js/company.js:33`. `main` has no such claim.

## Juniper App (`Cinqic-Research/Juniper-App`)

`main` never says "Juniper 1"; `docs/adr/0023-gpt-oss-backend-and-resident-runtime.md:8-9`
calls GPT-OSS "Juniper's first large local language model", which remains accurate
for the App runtime but should not be read as the Juniper 1 model.

## Effect on this corpus

The Cinqic documentation sources exclude any document mentioning `juniper 1`,
`juniper lm`, `gpt-oss`, `juniper reference` or `erudition model` (option
`deny_terms`), so none of the stale architecture text enters training data.

## Other findings

- `Cinqic-History-Dataset` v1.0.0 marks AAA golden answers, frozen exams and
  confirmation code as `training_eligible=True`; v1.1.0 is explicitly not approved for
  training distribution. It is not used here.
- `CC-Bench` and `Cinqic Coder` directories on the storage volume are empty.
