# Juniper 1 Base

**Juniper 1 Base** (`juniper1-base-v1`) is Cinqic's first original base language model: a 142,771,456-parameter
decoder-only Transformer trained from scratch, by Cinqic, on one epoch of the 3.41-billion-token
`j1-base-v0.2.2` corpus. This repository holds the released weights' loader, the tokenizer package, a
text-continuation CLI, the training and data-pipeline source at their exact historical versions, the research
evidence behind the checkpoint selection, and a reproduction guide.

| | |
|---|---|
| Weights and tokenizer | Hugging Face [`CinqicX/juniper1-base`](https://huggingface.co/CinqicX/juniper1-base) (float32 safetensors, 571 MB) |
| Dataset | Hugging Face [`CinqicX/juniper1-base-corpus-v0.2.2`](https://huggingface.co/datasets/CinqicX/juniper1-base-corpus-v0.2.2): redistributable payloads with their original licenses, and reconstruction manifests for the rest (see [docs/DATASET.md](docs/DATASET.md)) |
| License | Apache-2.0 for the Cinqic-owned weights, tokenizer, code and documentation ([LICENSE](LICENSE), [NOTICE](NOTICE)). Third-party training data keeps its own licenses ([THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)). |
| Model card | [MODEL_CARD.md](MODEL_CARD.md) |
| Release record | [docs/RELEASE_RECORD.md](docs/RELEASE_RECORD.md) |

## What it is, and is not

Juniper 1 Base is a **next-token base model**. It continues text. It has no chat template, no instruction
tuning, no tool use, no reasoning mode and no safety tuning. Cinqic's post-training experiments on it
(instruction SFT v1.2 pilot, SFT v2, v2.1, v2.2, and a distillation stage) did not qualify for release and are
recorded as closed negative results in [docs/JUNIPER1_CLOSURE.md](docs/JUNIPER1_CLOSURE.md). Expect the behavior
of a small base model: fluent topical English and idiomatic Python, frequent factual confabulation, near-zero
arithmetic, and repetition under greedy decoding on long continuations.

The tokenizer reserves 64 control IDs (0-63). Only ID 0, `<|endoftext|>`, was used in pretraining (as the
document separator). IDs 1-63 (the `juniper-chat-v1` protocol tokens and reserved slots) were **never trained**;
the loader masks them out of generation, and text never encodes to them.

## Quickstart (CPU)

```bash
python -m venv .venv && . .venv/bin/activate
pip install "juniper1[hub] @ git+https://github.com/Cinqic-Research/Juniper-1-Base@v1.0.0"
juniper1 complete "The Pythagorean theorem states that" --max-new-tokens 48
```

The first run downloads `model.safetensors`, `config.json`, `MANIFEST.json` and `tokenizer.json` from
`CinqicX/juniper1-base` into the Hugging Face cache, verifies every file's SHA-256 against the manifest,
checks the configuration against the frozen one, reproduces the frozen tensor digest, loads strictly, and
prints the greedy continuation. Recorded output for that prompt (CPU, float32, greedy):

```text
The Pythagorean theorem states that the square of the length of the hypotenuse is equal to the sum of the
squares of the lengths of the other two sides. In this case, the length of the hypotenuse is 10 and the
length of the other two sides is
```

Deterministic settings: `--temperature 0` (greedy, the default) is deterministic on CPU float32. Sampling
(`--temperature 0.8 --top-k 40 --seed 0`) is deterministic for a fixed seed on the same CPU build. On CUDA
(`--device cuda`), float32 logits differ from CPU by about 1e-4 at most; greedy continuations matched on the
recorded prompts but are not guaranteed to across all inputs. `--dtype bfloat16` is a derived variant that
changes logits; the shipped weights are float32. Prompts must be under 2,048 tokens (the qualified context);
longer prompts are rejected rather than truncated.

Python:

```python
from juniper1 import load_model, complete, GenerationSettings
model, tok, info = load_model("CinqicX/juniper1-base", revision="<pinned commit>")
print(complete(model, tok, "Photosynthesis is the process by which", GenerationSettings(max_new_tokens=32))["completion"])
```

Local files: `juniper1 complete --model ./juniper1-base "..."` where the directory holds the four files above.
`juniper1 verify --model ...` reports the identity checks; `juniper1 tokenize "text"` shows token IDs.

## Identity

| Property | Value |
|---|---|
| Freeze ID | `juniper1-base-v1` = `j1-base-r2/ckpt-000052034-336e5967` (update 52,034, the run's terminal state) |
| Architecture | `j1-wide-20x768`: 20 layers, d_model 768, SwiGLU FFN 1,920, GQA 12 query / 6 KV heads, head_dim 64, pre-RMSNorm, learned per-head QK-RMSNorm, RoPE θ 100,000, tied embeddings, no biases |
| Parameters | 142,771,456 (123,897,088 non-embedding) |
| Context | 2,048 tokens qualified; 4,096 unqualified |
| Vocabulary | 24,576 IDs: 64 control, 256 byte-fallback, 24,256 learned (`cand5-j1tok-24k`) |
| Source `model.pt` SHA-256 | `71773e46db2511332af97891b6485b04bf414a448dab9c4b050988e0b532649c` |
| Tensor-tree digest | `1fd07c0d0c6c6b355460ab82ebe02530047b5682a4e6a41b9e4f0146323076e4` (reproduced by the loader) |
| `tokenizer.json` SHA-256 | `176a89e2197c24d3f024fbb60ce6f346635e7cba912b3579d93b37d486c9adf1` |
| Corpus | `j1-base-v0.2.2`, build identity `89b08d5754c46dfd18ec239fdacb21ae899c1cf9d7606392401b5224c54bffc5`; 2,954,191 train / 2,576 validation / 2,604 test documents; 3,413,019,479 document tokens; 3,410,100,224 training targets consumed |
| Validation (full split, fp32, re-measured 2026-10-12) | 0.916381 bits/byte, loss 2.429953 nats, perplexity 11.358 (a training-time monitoring set, not an unbiased test estimate; the test split was never used) |

## Repository layout

| Path | Contents |
|---|---|
| `juniper1/` | Release package: frozen model definition, tokenizer wrapper, verified loader, generation, CLI |
| `tokenizer/tokenizer.json` | The frozen tokenizer (same bytes as on the Hub) |
| `tests/` | Tokenizer, model-numerics and generation tests (CPU, no download) |
| `scripts/` | `convert_checkpoint_to_safetensors.py`, `parity_check.py` |
| `training/` | Juniper-1-Train source at the frozen code identity (`j1bench/`, tools, tests) and the research evidence: freeze record, audit and selection reports, training contract, protocols, logged results |
| `data/` | Juniper-1-Data pipeline source, configs, schemas, manifests and reports for `j1-base-v0.2.2`, data card, tokenizer stage |
| `evidence/` | Training and validation curves extracted from the raw run log; reference-versus-export parity record |
| `docs/` | Reproduction guide, dataset availability, release record, Juniper 1 R&D closure, export mapping |

## Reproduction levels

See [docs/REPRODUCTION.md](docs/REPRODUCTION.md). In short: exact artifact recovery and identity verification
are established; the inference path is tested from the published files; the data pipeline and training code are
published at their exact historical versions with pinned upstream sources; token shards regenerate to equal
token bytes but a metadata field differs; and a full pretraining replication was neither run nor claimed.

## Status

Juniper 1 R&D is closed (owner decision, 2026-10-16). This repository receives packaging, security and
documentation maintenance only. A future Juniper model would be a separate project and decision.
