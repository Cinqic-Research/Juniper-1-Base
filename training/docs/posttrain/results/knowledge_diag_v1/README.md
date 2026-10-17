# Knowledge diagnostic v1: results (2026-10-13, 2:26 PM EDT)

Protocol [`sft_knowledge_diag_v1.json`](../../sft_knowledge_diag_v1.json), registered before any arm ran.
Dev split only (85 bundle dev facts items, 24 capitals); scorer v3 (Data `f0399ed`). Arms: 300 updates from
the frozen base, seed 1, LR 3e-5, differing only in the dataset (contracts `f719fdc8…` full, `d17b3171…`
without the 2,887 fact-bearing train episodes). Diagnostic only: no arm is a candidate. $0 (RTX 2060).

| model | chat all | few-shot all | country capital chat / few-shot | US state | continent | currency | element symbol | element number |
|---|---|---|---|---|---|---|---|---|
| base | 0/85 (no protocol) | 18/85 | 0/19 / 3/19 | 0/5 / 1/5 | 0/20 / 10/20 | 0/17 / 1/17 | 0/12 / 3/12 | 0/12 / 0/12 |
| pilot s1 @500 | 16/85 | 19/85 | 0/19 / 3/19 | 0/5 / 1/5 | 11/20 / 11/20 | 1/17 / 1/17 | 4/12 / 3/12 | 0 / 0 |
| pilot s1 final | 19/85 | 21/85 | 0/19 / 4/19 | 0/5 / 1/5 | 13/20 / 11/20 | 2/17 / 1/17 | 4/12 / 4/12 | 0 / 0 |
| arm full | 18/85 | 19/85 | 2/19 / 5/19 | 0/5 / 1/5 | 12/20 / 9/20 | 2/17 / 1/17 | 2/12 / 3/12 | 0 / 0 |
| arm without facts | 0/85 (abstains 83) | 17/85 | 0/19 / 4/19 | 0/5 / 1/5 | 0/20 / 9/20 | 0/17 / 1/17 | 0/12 / 2/12 | 0 / 0 |

Training-label capture in chat (wrong answers equal to a training-split label of the same relation):
arm full 16/67 (0.24), arm without facts 0/85 (it abstains instead), pilot final 27/66 (0.41).

## Predeclared interpretation

- **H1 (label capture) — not supported by its rule.** Capture is real (0.24 vs 0.00, ≥ 0.15), but the second
  condition fails: without fact-bearing data the model answers **no** factual question in chat (83/85
  "I don't know."), so its capitals chat accuracy (0/24) is below the full arm's (2/24).
- **H2 (general drift) — not supported.** Few-shot accuracy after SFT is not below the base's by ≥ 3 items
  (pilot final 21/85, arm without facts 17/85, base 18/85): the knowledge the base elicits few-shot is still
  there after SFT.
- **H3 (elicitation) — not met overall** (the pilot's chat 19/85 is not ≥ 3 below its few-shot level). On the
  capitals subset (exploratory, n = 24) the same pilot model answers 0/24 in chat and 5/24 few-shot: the
  capitals loss is a chat-format elicitation problem, not lost weights.

n is small and the hypotheses are not exclusive; no significance claim.

## Consequences (decided here, before any v1.3 training)

1. v1.3 keeps the fact-bearing families: removing them makes the model abstain on real questions, which is
   worse than label capture.
2. SFT v2 measures knowledge retention separately from chat elicitation: the candidate also answers the
   test facts items under the base's few-shot prompts, gated per relation against the base few-shot (a
   retention gate), while chat capitals stay a separate, reported/non-inferiority item.
3. Improving chat elicitation of base knowledge (rather than teaching facts) is an open problem for a later
   dataset: more relations/entities would shrink the label prior, but test subjects are disjoint by design.
