# A100 pilot plan and budget (pre-registered, written before the session)

> **Superseded for execution on 2026-10-10 by [`14_pilot_protocol_v2.md`](14_pilot_protocol_v2.md)**
> before any pilot ran. That amendment repairs measurement defects PP-1 to PP-8 found in review.
> This plan is preserved unchanged below as the original pre-registration.

**Authority:** directive §9. Incremental RunPod spend for this assignment is capped at
**$10.00** over the baseline day-bucket total of **$3.5501** (`list-billing`,
2026-10-08 19:40Z). No other GPU or paid service is used.

## Budget

| Item | Basis | Est. minutes | Est. $ at $1.93/h |
|---|---|---:|---:|
| Pod start, environment, code deploy, pyarrow install | earlier qualification: ~2–4 min | 6 | 0.19 |
| Data upload to the pod (≈1.1 GB prefixes + tokenizers) | SSH to the pod measured 2.4–2.5 MB/s; overlaps the LR sweep where possible | 10 | 0.32 |
| LR sweep: small model, 3 LRs × 60M tokens | small ≈ 0.35–0.5M tok/s (projected) | 9 | 0.29 |
| Tokenizer pilots: small × 3 candidates × ≈175M tokens (equal bytes) | as above | 24 | 0.77 |
| Family: micro, mid, core at ≈175M tokens | micro ≈ 0.6–1M, mid ≈ 0.2M, core 0.115M tok/s | 45 | 1.45 |
| Shape alternatives: deep 30×576, wide 20×768 at ≈175M tokens | ≈ core | 52 | 1.67 |
| Seed repeats: small × 2 | as small | 16 | 0.51 |
| Compile warm-ups (≈ 1 min × 11 runs), evals, container-disk saves | measured compile 56–66 s cold | 15 | 0.48 |
| Resume drill + checkpoint pull to FLOWBOX + verified stop | — | 10 | 0.32 |
| **Total** | | **≈ 187** | **≈ 6.0** |

**Spending controls:**

- The session stops at the **$8.00** incremental mark: estimated billed hours × $1.93 since
  start, checked before each stage. This leaves $2 for delayed billing and residual
  container-disk charges.
- Planned cut order if throughput falls short of the projections: (1) skip the seed
  repeats, (2) halve the shape-alternative token budget, (3) skip the wide shape. The
  leading-candidate core run and the tokenizer pilots are never cut.
- The pod is stopped immediately after the last needed artifact is verified on FLOWBOX.
  The stop is verified by system log and SSH, because the API reports EXITED ~20 s early.
- Billing is re-read after the session and again later to capture lag. The billed figure,
  not the estimate, is what is reported.

## Fixed run contract (all pilots)

| Setting | Value |
|---|---|
| Sequence length | 2,048 |
| Batch | micro-batch 16, accumulation 2 → 65,536 tokens per update |
| Precision | BF16 autocast, `torch.compile` default (the qualified optimized profile; experimental) |
| Optimizer | AdamW β 0.9/0.95, ε 1e-8, weight decay 0.1 on matrices only, clip 1.0 |
| Schedule | 100-update linear warmup, cosine to 10 % at the run's final update |
| Data | `j1-base-v0.2` keyed-order stream, equal-bytes prefix (identical documents for every candidate), one epoch, no repetition |
| Evaluation | 128 windows (262K tokens) from the start of the validation split, loss and bits per byte per source, 8 evaluations per run |
| Checkpoints | 500 updates or 15 min (whichever first) plus best-eval and final saves, to container disk. Pilots are investigation artifacts, not canonical training |
| LR | small-model optimum from the sweep; larger models use LR_small / √(d_model/384) |
| Seeds | 1 for everything; small model also seeds 2 and 3 for the noise band |

Selection rules are fixed in `08_evaluation_framework.md` §3. `tools/select_rule.py` applies
them automatically during the session, and its JSON output is the decision record.

## What these pilots can and cannot show

- **They can:**
  - verify that the full real-data path learns end to end
  - measure real-data throughput, memory and checkpoint costs per shape
  - rank tokenizer vocabularies on bits per byte under equal data
  - compare shapes at equal tokens and equal wall-clock time
  - estimate seed noise
  - check the exact data cursor on resume
- **They cannot:**
  - predict converged quality: ≈175M tokens is ≈1.2 tokens per parameter for the 148M
    models and ≈12 for the micro model
  - establish downstream capability
  - prove that one shape is superior at the full 3B-token budget

Fitted trends are reported only with their validity range.
