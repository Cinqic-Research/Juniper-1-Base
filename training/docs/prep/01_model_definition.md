# Juniper 1 — Official Model Definition

- **Version:** `j1-definition-v1.0`
- **Date:** 2026-10-08
- **Status:** FROZEN for preparation work by the 2026-10-08 owner directive, Sections 3.1–3.7.
  Values marked *provisional* must be measured and confirmed before release.
- **Authority:** owner directive 2026-10-08 and Juniper 1 Design Specification v0.5.
  The proposed spec revision v0.6 (`06_spec_v0.6_proposal.md`) records the changes this
  preparation work recommends.

## 1. Identity and purpose

Juniper 1 is Cinqic's first original, from-scratch, open-weight, text-only small language
model. It is a dense decoder-only Transformer in the ~150M-parameter class, trained from
newly initialized weights. GPT-OSS-20B is a teacher, reference, and evaluation baseline only.
Juniper 1 does not use its architecture or initialization.

**Primary use case:** an efficient general-purpose assistant language model specialized
toward reliable reasoning, coding, structured interaction, and tool-assisted workflows on
consumer hardware.

### Principal strengths sought

These are capability targets. None is a measured claim.

| Area | Target behaviour |
|---|---|
| Communication | contemporary American English; concise, direct answers |
| Knowledge and instructions | general knowledge, instruction following, instruction hierarchy |
| Software | Python first; general software-engineering assistance |
| STEM | mathematical and scientific reasoning, kept short and verifiable |
| Structure | correct JSON / YAML / XML / Markdown / shell / URLs |
| Tools | function-call generation, argument selection, web/retrieval choice, Python execution requests, interpretation of host-authored results |
| Reliability | error recognition and correction; uncertainty expressed from the evidence |

The model must be useful without tools and more capable with them. It must not call tools
unnecessarily: no-tool restraint is a scored behaviour.

### Supported and unsupported uses

| Supported | Not supported or not promised |
|---|---|
| Local assistant inside Juniper App, with App-enforced permissions | Autonomous action without a host permission layer |
| Short reasoning, coding help, and structured-output generation | Frontier-level reasoning; this is not claimed at ~150M parameters without independent evidence |
| Tool-call proposals that the host validates and executes | Self-reported execution results (only host-authored results count) |
| CPU-only and 6–8 GB GPU inference | Multilingual proficiency, image/audio/video, or long-document (>4K) work |
| Research and evaluation of small-model agent control | High-stakes medical, legal, or financial advice; factual authority without retrieval |

## 2. Language policy (frozen)

- **Primary natural language:** American English.
- **Programming:** Python is the highest priority. Ordinary programming and structured-data
  syntax (JSON, YAML, XML, Markdown, shell, URLs, paths, technical notation) must be
  represented losslessly by the tokenizer.
- **Mathematical and scientific Unicode** must remain representable: byte-level coverage is a
  tokenizer acceptance test.
- **Other natural languages** may appear incidentally. Multilingual proficiency is not a
  product promise, and no compute is spent optimizing for it. The language composition of
  the corpus is measured and reported (data card), and incidental multilingual behaviour is
  documented after training.

## 3. Context policy

| Stage | Length | Status |
|---|---|---|
| Base pretraining | 2,048 tokens | frozen |
| Conditional extension | 4,096 tokens | released only after genuine long-sequence continued pretraining plus long-context and regression gates; 2,048 stays the validated fallback |

**Context budgeting** is a host responsibility, specified in `04_protocol_juniper_chat_v1.md`:

- The window holds system, developer, tool schemas, conversation, tool results, reasoning,
  and the generation reserve.
- Truncation drops whole messages, oldest untrusted content first. It never splits a role
  header from its body.
- Truncation never removes the system/developer block, and it never promotes data into a
  higher role.

## 4. Deployment hardware

| Target | Specification | Role |
|---|---|---|
| Primary consumer | ~16 GB system RAM, 8 GB GPU, ordinary desktop/laptop | comfortable whole-system target, not a minimum |
| Secondary | FLOWBOX: Ryzen 7 5700G, 16 GB DDR4, RTX 2060 6 GB, Linux Mint | compatibility and measurement reference |
| CPU-only | FLOWBOX CPU (8C/16T Zen 3), no GPU | must be qualified |
| Training | 1× A100 SXM 80 GB, RunPod Secure Cloud | training only; never required for inference |

**Analytic inference memory** at 148.3M parameters is a bound, not a measurement:

- BF16/FP16 weights take 297 MB.
- The KV cache, FP16 at batch 1, holds 28 layers × 5 KV heads × 64 dims × 2 tensors × 2 bytes
  = 35,840 B per token. That is 73 MB at 2,048 tokens and 147 MB at 4,096.

On these figures the model should fit far below 6 GB of VRAM. Actual peak memory, including
runtime overhead, is a measurement gate (§6).

## 5. Parameter budget

- **Leading design:** ~148M dense parameters (`j1-core-28x640` family). It is adapted to the
  final tokenizer.
- **Final-model investigation range:** 100M–180M. This is not permission to grow the model.
- **Smaller experimental models** (10–90M) exist to measure scaling behaviour only.
- A final design materially outside the 150M class requires explicit justification and an
  owner-approved specification change.

## 6. Latency and memory objectives (provisional acceptance candidates)

These are engineering goals, not guarantees. They are frozen into the release gate only
after the benchmark in `07_inference_benchmark.md` has measured them on the reference devices.

| ID | Metric | Provisional threshold | Conditions |
|---|---|---|---|
| L1 | GPU decode throughput | ≥ 30 generated tok/s | RTX 2060 6 GB (FLOWBOX) as the 8 GB-class stand-in, batch 1, FP16, 256-token prompt, 256 generated tokens, warm, median of 5 runs, p95 reported |
| L2 | GPU time to first token | median ≤ 2.0 s | same device, 256-token prompt, warm; cold load time reported separately |
| L3 | CPU decode throughput | ≥ 5 generated tok/s | Ryzen 7 5700G, 8 threads, FP32 and Q8 reported, batch 1, same prompt |
| L4 | Stable memory at qualified context | no OOM; peak RSS/VRAM reported | 2,048-token prompt + 256 generated; 4,096 only if that context is qualified |

Measurements record the runtime (PyTorch eager, and llama.cpp GGUF when export support is
qualified), precision or quantization, thread count, and warm or cold state.

## 7. Training limitations (declared up front)

- Training uses about 3B unique accepted tokens (15B ceiling, not a target). A 150M model
  trained on this budget sits near the compute-optimal regime of ~20 tokens per parameter.
  Factual recall will be limited and must be supplemented by retrieval in the App.
- There is one A100 and no multi-GPU or distributed setup.
- Teacher outputs (GPT-OSS-20B) are used only for post-training data, are validated, and are
  logged. They are not part of the base corpus.

## 8. Licensing and release policy

| Item | Policy | Status |
|---|---|---|
| Cinqic-authored code | Apache-2.0 | decided |
| Model weights | Apache-2.0 intended, conditional on provenance and source terms | release treatment PENDING owner decision (`weights-release-license-treatment`) |
| Training corpus | private; not redistributed; open-weight ≠ open-dataset | decided; any redistribution PENDING per source |
| Source terms | training use APPROVED by the owner 2026-10-08 for FineWeb-Edu, FineMath, CC BY-SA documents, Stack v2; others pre-approved in the source registry | recorded in Juniper-1-Data `configs/owner_decisions.toml` |
| Attribution | per-document URL, authors and license kept in corpus `meta`; aggregate attribution in the model card | required |
| Teacher outputs | GPT-OSS-20B (Apache-2.0 weights; OpenAI usage policy) for validated post-training examples; generation logged | review before collection |
| Third-party notices | NOTICE file listing dataset sources and licenses in each weight release | required at release |
| Privacy | secret and PII redaction in the corpus; private Cinqic records excluded unless Cinqic-owned and intended | enforced by the pipeline |
| Model card | purpose, limits, data lineage by source and share, license treatment, evaluation identities, checksums | required at release |

The Apache-2.0 declaration licenses only what Cinqic owns. It does not imply ownership of
third-party content or legal clearance.

## 9. Evaluation and success criteria

These are defined in `08_evaluation_framework.md`. The four separate evaluation layers are
raw base LM, post-trained instruction following, tool use, and full App behaviour.
Instruction-tuned metrics are never reported for the base model. Promotion thresholds are
frozen before any candidate is promoted.
