# Juniper 1 Base v1.0.0 (`juniper1-base-v1`)

Cinqic's first original base language model, trained from scratch: **142,771,456 parameters**, dense
decoder-only Transformer (`j1-wide-20x768`: 20 layers, width 768, SwiGLU 1,920, GQA 12/6, per-head QK-RMSNorm,
RoPE θ 100,000, tied embeddings), one epoch over **3,410,100,224 training targets** of the `j1-base-v0.2.2`
corpus, 2,048-token context. Released under **Apache-2.0** (weights, tokenizer, code, documentation); the
third-party training data keeps its own licenses (`THIRD_PARTY_NOTICES.md`, `docs/RIGHTS.md`).

**This is a base next-token model.** No chat template, no instruction tuning, no tool use, no reasoning mode,
no safety tuning. Cinqic's post-training experiments (SFT v1.2 pilot, SFT v2 / v2.1 / v2.2, distillation v2) did
not qualify and are recorded as closed negative results. Juniper 1 R&D is closed with this release.

## Where things are

| | |
|---|---|
| Weights, tokenizer, model card | https://huggingface.co/CinqicX/juniper1-base |
| Dataset (text payloads + reconstruction manifests) | https://huggingface.co/datasets/CinqicX/juniper1-base-corpus-v0.2.2 |
| Code, evidence, reproduction guide | this repository, tag `v1.0.0` |

## Identity

- Frozen checkpoint `j1-base-r2/ckpt-000052034-336e5967` (update 52,034); source `model.pt` SHA-256
  `71773e46db2511332af97891b6485b04bf414a448dab9c4b050988e0b532649c`
- `model.safetensors` (float32, exact tensors) SHA-256 `9dcf74b5db154951d1edbeea6801c04193facb8efc4d5b8b2796e6e675301952`
- Tensor-tree digest `1fd07c0d0c6c6b355460ab82ebe02530047b5682a4e6a41b9e4f0146323076e4`
- `tokenizer.json` (`cand5-j1tok-24k`, 24,576 IDs) SHA-256 `176a89e2197c24d3f024fbb60ce6f346635e7cba912b3579d93b37d486c9adf1`
- Corpus build identity `89b08d5754c46dfd18ec239fdacb21ae899c1cf9d7606392401b5224c54bffc5`

## Quickstart

```bash
pip install "juniper1[hub] @ git+https://github.com/Cinqic-Research/Juniper-1-Base@v1.0.0"
juniper1 complete "The Pythagorean theorem states that" --max-new-tokens 48
```

Validation (full `j1-base-v0.2.2` validation split, fp32, re-measured independently): 0.916381 bits/byte, loss
2.429953, perplexity 11.358; a training-time monitoring set, not an unbiased test estimate.

## What is and is not reproducible

Exact artifact recovery and identity verification: established. Inference from the published files: tested from
a clean environment. Data pipeline and training code: published at their exact historical versions with pinned
upstream sources; token shards regenerate to equal token bytes but a metadata field differs; FineWeb-Edu,
FineMath and Stack v2 Edu text is published as reconstruction manifests, not text. A full pretraining
replication was neither run nor claimed. Details: `docs/REPRODUCTION.md`, `docs/DATASET.md`.

## Provenance

Clean export of Cinqic's private `Juniper-1-Train` (`4d39c0e5`) and `Juniper-1-Data` (`542bec3b`) repositories;
every exported file is listed with its source commit and SHA-256 in `EXPORT_MAPPING.json`. Release checks and
reviewer evidence: `docs/RELEASE_RECORD.md`.
