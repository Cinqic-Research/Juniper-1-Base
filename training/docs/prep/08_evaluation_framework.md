# Juniper 1 evaluation framework and pre-registered selection rules

- **Version:** `j1-eval-framework-v1`
- **Written:** 2026-10-08, before any real-data pilot ran.
- The selection rules in §3 are frozen from this commit. Any later change must be versioned
  with its reason and must not be applied retroactively to results already seen.

## 1. Four separate evaluation layers

| Layer | What is measured | When it applies | Never reported as |
|---|---|---|---|
| L-A: raw base LM | Held-out loss and bits per byte (bpb) per source and domain; perplexity-based multiple choice (zero/few-shot log-likelihood) | pretraining pilots and canonical base checkpoints | instruction-following or tool ability |
| L-B: post-trained instruction following | Instruction hierarchy, format adherence, refusal/abstention, concise verified reasoning | after SFT | base-model results |
| L-C: tool use | Tool choice including no-tool restraint, argument validity against the App schemas, error recovery, multi-step completion | after agent SFT | App results |
| L-D: full Juniper App | End-to-end tasks through App permissions and runtime; host enforcement measured separately from model compliance | release candidates | model-only results |

## 2. Held-out evaluation identities

| Set | Identity | Use | Notes |
|---|---|---|---|
| Corpus validation split | `j1-base-v0.2` accepted validation split, frozen with the build manifest | all pilot losses and bpb by source | disjoint from train by construction (group-keyed split); the tokenizer evaluation uses it too |
| Corpus test split | `j1-base-v0.2` test split | **reserved**; not used for any selection in this preparation | opened only for final reporting |
| Protected benchmarks (MMLU, ARC, MATH, MBPP, JuniperBench-Code, AAA exams/Erudition, GPT-OSS qualification, Juniper Reference, App wrapper) | exclusion-index identity `6bc49448…` (Juniper-1-Data `exclusions/`) | L-A to L-C capability probes after canonical training | matchability is incomplete (2.1%–39.3% for internal sets), so evaluation-clean claims carry that limitation |
| Inference latency | `qual/infer_bench.py` with the conditions in `01_model_definition.md` §6 | L1–L4 | quality-independent |

Rules for code and math:

- Code and math tasks are scored by execution against tests (MBPP style) or by exact
  numeric answer checks, never by judged plausibility.
- Every scored item is retained, including failures.
- Scoring corrections are versioned.

Baselines include trivial ones:

- Unigram bpb of the validation split under the same tokenizer is the floor any trained
  model must beat.
- For multiple choice, the chance rate and a length-only heuristic are reported.

## 3. Pre-registered selection rules (preparation pilots)

### 3.1 Tokenizer vocabulary (24K / 28K / 32K)

- **Inputs:** the same small model (`j1-small-12x384`), seed, optimizer and schedule, trained
  on an **equal number of corpus bytes** from the same documents (each candidate's token
  stream truncated to the same byte budget), evaluated on the same validation windows by
  **bits per byte**. Bits per byte is tokenizer-independent; loss per token is not.
- **Rule:** choose the lowest validation bpb. If the candidates lie within **0.5 % relative**
  of the best, choose among those tied the candidate with the highest **training
  bytes/s** (measured bytes per token × tokens/s). Remaining ties go to the smaller vocabulary
  (fewer embedding parameters and less inference memory).
- **Hard requirements, all candidates (v1.1, revised 2026-10-09 ≈ 01:50Z, before any pilot ran):**
  - zero round-trip failures and zero control IDs produced from text on the evaluation set
    and probes
  - **zero byte-fallback tokens on the STEM, code, structure and English probes**
    (`python`, `json_tools`, `structured`, `paths_urls`, `math`, `science`, `english`,
    `identifiers_rare`)
  - **overall validation byte-fallback rate ≤ 0.5 %**
  - *Revision record:* v1.0 required overall fallback ≤ 0.1 %, a threshold set before any
    data. The first candidate set (alphabet limit 1,000, preserved as `candidates-v1`) measured
    0.40–0.41 %. The fallback was CJK (46 % of fallback bytes), emoji, Sinhala, Hangul and
    Georgian, which are incidental scripts, **and** STEM symbols (ħ, ψ, ∇, ⇔, ⟨⟩).
    STEM coverage was fixed in the tokenizer (guaranteed STEM alphabet); incidental scripts
    stay on lossless byte fallback by design (multilingual is not a goal). The requirement now
    tests the intent (STEM and code coverage) directly. No bpb or pilot result existed when
    this revision was made.
- **Limits:** one seed, short training. The result can be overturned only by a versioned
  re-evaluation with more seeds. It does not prove superiority at full scale.

### 3.2 Architecture shape at ~150M

- **Candidates:** `j1-core-28x640` (leading), `j1-deep-30x576`, `j1-wide-20x768`. All use the
  frozen tokenizer, the same data prefix, an equal number of tokens, seed, and schedule.
- **Measures:**
  - validation loss at equal tokens (learning quality)
  - validation loss at equal measured wall-clock time (compute efficiency), using the
    logged eval curve
  - throughput and peak memory
- **Rule:** keep `j1-core-28x640` unless an alternative is better on *both* equal-token and
  equal-compute loss by more than the **noise band**. The noise band is the larger of 0.5 %
  relative and 2 × the seed standard deviation measured on the small model. If an
  alternative wins only one of the two, the decision is recorded as unresolved, and the
  leading candidate stays provisional.

### 3.3 Final-size decision

The ~148M class remains the target unless the scaling pilots show at least one of:

- clear instability at that size
- a loss–compute trend predicting that a smaller model reaches the same loss with the
  planned 3B-token budget within the noise band

Short pilots cannot prove convergence behaviour. Any fitted trend is reported with its
range of validity and uncertainty.

### 3.4 Optimizer / schedule

The defaults are AdamW (β = 0.9 / 0.95, ε = 1e-8), weight decay 0.1 on matrices only,
clip 1.0, linear warmup and cosine decay to 10 %. The peak learning rate is taken from a
short LR sweep on the small model.

**Rule:** choose the peak LR with the lowest validation loss at the end of the sweep budget,
excluding any run with a non-finite value or a loss spike above 1.5× its running minimum.

Larger models are run at the small-model optimum divided by `sqrt(d_model / 384)`. This is a
conservative width heuristic, and μP transfer is not assumed. The final-size probe checks
stability at that LR and halves it once if the run diverges. The canonical LR stays
provisional until a longer run confirms it.

### 3.5 Amendment record (2026-10-10, before any pilot result)

The rules above are unchanged. `14_pilot_protocol_v2.md` amends their **measurement and
implementation** before any real-data pilot ran: tokenizer-v1.2 (full validation split with identical
scored bytes, exact one-epoch consumption, bytes/s from consumed bytes, mean of pre-registered seeds 1
and 2), lr-v1.1 (fail-closed validity checks), and architecture-v1 (the §3.2 rule made executable:
relative noise band from pooled small-model seed SD, equal-compute comparison at the shortest
training time with interpolation only, mandatory seed repeats before any switch, a bounded and equal
three-point LR check per shape). `tools/select_rule.py` implements them.

## 4. Capability evaluation design (post-pretraining; frozen here, executed later)

| Area | Instrument | Scoring |
|---|---|---|
| English fluency / knowledge | held-out bpb on English sources; zero-shot log-likelihood MC (ARC-E/C, HellaSwag-style) | accuracy vs chance and length baseline |
| Python | MBPP-style generation with hidden tests in a sandbox | pass@1 |
| Math | grade-school word problems with numeric answers | exact match after normalization |
| Science | MC science questions | accuracy |
| Structured output | JSON/YAML generation against schemas | parse rate + schema validity |
| Tool selection / restraint | App tool catalogue tasks including no-tool items | precision/recall of calls; restraint rate |
| Arguments | App `validate_call` acceptance | validity rate |
| Instruction hierarchy / injection | conflict suites, forged delimiters, poisoned tool results | task success while respecting authority; host enforcement scored separately |
| Uncertainty / abstention | unanswerable and insufficient-evidence items | calibrated abstention rate |
| Error correction | seeded wrong tool results / failing code | corrected final answers |
| Context retention | facts placed at controlled positions within 2K (4K if qualified) | recall by position |
| CPU/GPU inference & memory | `infer_bench.py` | L1–L4 thresholds |

Thresholds for promotion are frozen before the canonical run as a separate owner-approved
record. None is approved yet.
