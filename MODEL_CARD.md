---
license: apache-2.0
language:
- en
pipeline_tag: text-generation
library_name: juniper1
tags:
- base-model
- from-scratch
- pytorch
- safetensors
- cinqic
- juniper-1
datasets:
- CinqicX/juniper1-base-corpus-v0.2.2
model-index:
- name: juniper1-base-v1
  results:
  - task:
      type: text-generation
    dataset:
      name: j1-base-v0.2.2 validation split (2,576 documents, 12,311,466 UTF-8 bytes)
      type: CinqicX/juniper1-base-corpus-v0.2.2
      split: validation
    metrics:
    - type: bits_per_byte
      value: 0.916381
      name: bits per UTF-8 byte (fp32, full split; training-time monitoring set)
    - type: perplexity
      value: 11.358
      name: token perplexity (fp32, full split)
---

# Juniper 1 Base (`juniper1-base-v1`)

Cinqic's first original base language model, trained from scratch: 142,771,456 parameters, one epoch over
3.41 billion tokens of the `j1-base-v0.2.2` corpus, released under Apache-2.0 for the Cinqic-owned weights,
tokenizer and code. Code, evidence and reproduction guide: <https://github.com/Cinqic-Research/Juniper-1-Base>.

**This is a base next-token model.** It has no chat template, instruction tuning, tool use, reasoning mode or
safety tuning. Cinqic's post-training attempts on it did not qualify and are recorded as closed negative results
(see "Post-training" below). Do not deploy it as an assistant.

## Files

| File | SHA-256 | Size |
|---|---|---|
| `model.safetensors` (float32, exact tensors of the frozen checkpoint) | `9dcf74b5db154951d1edbeea6801c04193facb8efc4d5b8b2796e6e675301952` | 571,107,560 B |
| `config.json` | see `MANIFEST.json` | |
| `tokenizer.json` (`cand5-j1tok-24k`) | `176a89e2197c24d3f024fbb60ce6f346635e7cba912b3579d93b37d486c9adf1` | 680,825 B |
| `MANIFEST.json` | lists the hashes above and the frozen identity | |
| `full-state/` | the unmodified original checkpoint directory `ckpt-000052034-336e5967` (`model.pt`, `optim.pt`, `trainer.pt`, `manifest.json`, `COMPLETE`) for resumable-training research; PyTorch pickles, loadable with `weights_only=True` through `training/j1bench` |

Tensor-tree digest of the weights (reproduced by the loader): `1fd07c0d0c6c6b355460ab82ebe02530047b5682a4e6a41b9e4f0146323076e4`.
Source `model.pt` SHA-256: `71773e46db2511332af97891b6485b04bf414a448dab9c4b050988e0b532649c`.

## How to use

The model is a custom architecture (per-head QK-Norm, GQA, SwiGLU, tied embeddings). It is **not** loadable
with `AutoModelForCausalLM`, and no hosted inference or widget is provided. Use the native loader:

```bash
pip install "juniper1[hub] @ git+https://github.com/Cinqic-Research/Juniper-1-Base@v1.0.0"
juniper1 complete "The Pythagorean theorem states that" --max-new-tokens 48
```

```python
from juniper1 import load_model, complete, GenerationSettings
model, tok, info = load_model("CinqicX/juniper1-base")   # verifies hashes, config and tensor digest
print(complete(model, tok, "Photosynthesis is the process by which", GenerationSettings(max_new_tokens=32))["completion"])
```

Recorded greedy output (CPU float32) for the first prompt: *"... the square of the length of the hypotenuse is
equal to the sum of the squares of the lengths of the other two sides. In this case, the length of the
hypotenuse is 10 and the length of the other two sides is"*. CPU float32 greedy decoding is deterministic; CUDA
float32 logits differ by at most about 1e-4; bfloat16/float16 are derived variants that change logits.

Tokenizer: load `tokenizer.json` only through `juniper1.Juniper1Tokenizer` (or set
`encode_special_tokens = True` yourself). The JSON does not persist that flag, and a bare `Tokenizer.from_file`
turns a literal `<|user|>` in text into privileged control ID 3. Control IDs 1-63 were never trained.

## Architecture

| | |
|---|---|
| Type | dense decoder-only causal Transformer, `j1-wide-20x768` |
| Layers / width / FFN | 20 / 768 / SwiGLU 1,920 |
| Attention | grouped-query, 12 query heads, 6 KV heads, head_dim 64; learned per-head RMSNorm on q and k before RoPE |
| Positions | RoPE, θ = 100,000, rotate-half convention; 2,048-token context qualified (4,096 unqualified) |
| Norms | pre-RMSNorm (eps 1e-5) computed in fp32; final RMSNorm |
| Embeddings | tied input/output; vocabulary 24,576; no biases anywhere |
| Parameters | 142,771,456 total; 123,897,088 non-embedding |
| Init | normal std 0.02; output projections std/√(2·layers) |

## Training

| | |
|---|---|
| Data | `j1-base-v0.2.2` train split: 2,954,191 documents, 3,410,114,054 stream tokens (EOS-separated); 52,034 updates × 65,536 targets = 3,410,100,224 targets, exactly one epoch, no repeats; windows of 2,048 tokens over the EOS-separated stream, cross-document attention allowed, every position a target |
| Optimizer | AdamW (fused), β = (0.9, 0.95), eps 1e-8, weight decay 0.1 on 2-D projection weights only (embeddings and norms excluded), gradient clipping 1.0 |
| Schedule | peak LR 1.6971e-3, linear warmup 500 updates, cosine decay to 10 % of peak at update 52,034 |
| Batch | 32 sequences × 2,048 tokens per update (micro-batch 16, accumulation 2) |
| Precision | BF16 autocast, eager PyTorch (compiled execution failed an equivalence check and was not used); RMSNorm and losses in fp32 |
| Hardware | one NVIDIA A100-SXM4-80GB (RunPod, US-MD-1); torch 2.8.0+cu128, CUDA 12.8, Python 3.12 |
| Time | 52,025 s of active training (about 14.5 h), with one restart after a power loss at update 8,585 (replayed updates verified against identical batch fingerprints) |
| Seed | 1 (window permutation seeded by (seed, epoch)) |
| Code identity | Juniper-1-Train `j1bench` at commit `828edd19`, package SHA-256 `81a3a277…204e99`; published under `training/` |

Training and validation curves: `evidence/training_curve_j1-base-r2.json` and the CSVs beside it in the GitHub repository.

## Evaluation

Held-out language modeling on the full `j1-base-v0.2.2` validation split (1,572 windows of 2,048, 3,218,204
targets, 12,311,466 scored bytes), re-measured independently on 2026-10-12 from the frozen bytes:

| Checkpoint | bpb fp32 | loss (nats) | perplexity | logged (A100 bf16) |
|---|---:|---:|---:|---:|
| 50,000 | 0.917827 | 2.433787 | 11.402 | 0.917911 |
| 52,000 | 0.916198 | 2.429468 | 11.353 | 0.916280 |
| **52,034 (released)** | **0.916381** | **2.429953** | **11.358** | 0.916455 |

By source at 52,034 (logged, bpb): stackv2-edu 0.635, libretexts 0.861, finemath 0.885, arxiv 0.888,
oercommons 0.912, pressbooks 0.942, fineweb-edu 0.949, wikimedia 1.001, python-peps 1.043, stackexchange 1.088.

The validation split was observed during training as a monitoring set (never trained on). It supports the
checkpoint choice; it is not an unbiased capability estimate. The test split (2,604 documents) was never used.
52,000 is lower than 52,034 by 0.020 % relative, inside the pre-registered 0.1 % practical-tie threshold; 52,034
was selected as the run's terminal state with the stronger recoverability evidence. The selection protocol was
registered after the training run's logged values were known; this is disclosed in
`training/docs/audit_2026-10-12/base_checkpoint_selection_report.md`.

Small descriptive probes (greedy, n ≤ 20 each; not benchmarks): likelihood multiple choice on simple facts
17/20; executable Python with hidden asserts 3/8; few-shot two-digit addition 1/20; JSON prefix to valid JSON
1/6. Full prompts and outputs: `training/docs/audit_2026-10-12/results/behavior_cpu.json`.

## Training data and rights

Ten sources, all pinned to exact upstream revisions and documented in `data/docs/data_card.md` and
`data/configs/sources.toml`: FineWeb-Edu (SmolLM-Corpus dedup) 31.6 % of tokens, Stack v2 Edu (Common Pile,
permissive licenses) 17.2 %, Wikimedia (Common Pile) 13.9 %, Stack Exchange (Common Pile) 13.0 %, FineMath 4+
12.4 %, arXiv papers (Common Pile) 6.2 %, Pressbooks 3.1 %, LibreTexts 2.3 %, OER Commons 0.25 %, Python PEPs
0.08 %. English-first; multilingual content is incidental. Contamination controls: a layered benchmark-exclusion
index (0 hits on the accepted corpus) and the F9 adjudication that removed 7,008 documents matching
HumanEval+/MBPP+/MBPP test material; internal evaluation sets have low matchable fractions and remain a
documented coverage limit.

The owner approved these sources as training input on 2026-10-08 and authorized this public release on
2026-10-16. The model weights are licensed under Apache-2.0 by Cinqic. That license covers the weights, not the
third-party texts they were trained on: FineWeb-Edu and FineMath are ODC-By 1.0 databases over Common Crawl
content (notice under ODC-By §4.3 in `NOTICE`); Wikimedia and Stack Exchange text is CC BY-SA; the textbook and
arXiv sources carry per-document CC BY / CC BY-SA / CC0 / public-domain licenses; Stack v2 Edu code carries
permissive open-source licenses. Whether weights trained on ShareAlike text are adaptations of it is not settled
law; Cinqic releases the weights under Apache-2.0 and discloses the sources rather than asserting a conclusion.
The dataset release (`CinqicX/juniper1-base-corpus-v0.2.2`) carries each payload under its original license and
publishes reconstruction manifests, not text, for sources Cinqic does not redistribute.

## Post-training (closed, not released)

| Stage | Outcome |
|---|---|
| Instruction SFT pilot `j1-sft-v1.2` (protocol v3) and two replication seeds | `PILOT_CANDIDATE_QUALIFIED` as a research candidate; explicitly non-production |
| SFT v2 on dataset v1.3 | `FAILED`: generalization improved, but arithmetic, JSON, injection resistance and instruction-hierarchy regressed |
| SFT v2.1, v2.2 diagnostics | `STOP_FAILED` (hierarchy and JSON regressions against the registered non-inferiority gates) |
| Distillation v2 from a teacher model (three arms + control) | `NO_IMPROVEMENT_DETECTED` |

All candidates, protocols and decision records are preserved (private Cinqic archives; decision files under
`training/docs/posttrain/results/`). None of them is this release. Juniper 1 R&D was closed on 2026-10-16.

## Limitations and intended use

Intended for research on small from-scratch language models, reproducibility studies, tokenizer and training
infrastructure work, and as a base for others' experiments. Not intended for end-user applications without
further training and evaluation. It confabulates facts, cannot do arithmetic reliably, repeats itself under
greedy decoding, has no safety mitigations, and reflects the biases of web, forum, encyclopedia, textbook,
paper and code sources. Context beyond 2,048 tokens is unqualified.

## Citation

```bibtex
@misc{cinqic2026juniper1base,
  title  = {Juniper 1 Base: a 143M-parameter base language model trained from scratch},
  author = {Cinqic},
  year   = {2026},
  url    = {https://github.com/Cinqic-Research/Juniper-1-Base}
}
```
