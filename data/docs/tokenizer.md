# Juniper 1 tokenizer stage (`j1.tokenizer.v1`)

This stage trains, evaluates, selects and freezes the Juniper 1 tokenizer, then produces
model-ready token-ID shards. It replaces the provisional measurement tokenizers
(`measure-24k/28k/32k`), which remain sizing tools only.

## Design

| Property | Value | Why |
|---|---|---|
| Model | BPE over Unicode characters with byte fallback (`<0x00>`–`<0xFF>`), no normalizer, spaces inside tokens | lossless for any string; common math and science symbols get their own tokens instead of 2–4 byte tokens |
| Required alphabet | printable ASCII plus a curated STEM alphabet (Greek, operators, arrows, super/subscripts, blackboard letters, ħ, hatted letters, combining accents, typography) seeded as `initial_alphabet`; alphabet limit 1,500 | STEM symbols never fall back to bytes |
| Pre-tokenizer | the measurement tokenizers' split: contractions, letters with an optional leading non-letter, **single digits**, punctuation runs, newline runs, whitespace runs | single-digit numbers help arithmetic; whitespace runs give indentation tokens |
| Control IDs | 0–63. Named: `<\|endoftext\|>`, `<\|system\|>`, `<\|developer\|>`, `<\|user\|>`, `<\|assistant\|>`, `<\|tool\|>`, `<\|analysis\|>`, `<\|final\|>`, `<\|tool_call\|>`, `<\|end\|>`, `<\|call\|>`, `<\|eot\|>`. `<\|reserved_12\|>`–`<\|reserved_63\|>` have no meaning | protocol `juniper-chat-v1` |
| Byte fallback IDs | 64–319 | fixed |
| Learned | 320 … total − 1 | trained |
| Total vocabulary | **inclusive** of control and byte IDs; candidates 24,576 / 28,672 / 32,768 (multiples of 64) | embedding shapes |
| Literal control strings | `encode_special_tokens = True`, so text never produces IDs 0–63. `tokenizer.json` does not persist this flag; load only with `j1data.jtok.load` | injection resistance |

## Procedure

```bash
export J1_DATA_ROOT=…/juniper1-data/data
python -m j1data.tokcli tokenizer-candidates --build-id j1-base-v0.2          # sample, train, evaluate
python -m j1data.tokcli tokenizer-freeze --build-id j1-base-v0.2 --candidate <name> --name <frozen> \
    --rationale "<rule outcome and evidence>"
python -m j1data.tokcli token-shards --build-id j1-base-v0.2 --tokenizer <frozen>
python -m j1data.tokcli token-shards --build-id j1-base-v0.2 --tokenizer <frozen> --verify-only
```

- **Training sample:** training-split documents with
  `unit_interval("j1.tokenizer-train-sample.v1", doc_id) < 0.03`. The selection does not
  depend on order, and the sample identity is the SHA-256 of the sorted doc IDs.
- **Evaluation set:** the entire validation split, by source. It is disjoint from the sample by
  construction. Domain probes cover Python, JSON and tool schemas, YAML/XML/Markdown/shell,
  paths and URLs, math, science, English, long identifiers, Unicode, and literal control
  strings.
  **Note (review D-16):** the frozen tokenizer's `freeze.json` evaluation block (`evaluation.domains`, freeze.json `08a2d898…edf7`) is the **`j1-base-v0.2`
  validation split** (2,583 documents; stackexchange 524, stackv2-edu 498), on which the
  candidates were evaluated and selected. The launch corpus `j1-base-v0.2.2` has a validation
  split of 2,576 documents (stackexchange 523, stackv2-edu 492): the 7 validation documents
  excluded by the F9 adjudication are absent. The two sets must not be confused: `freeze.json`
  records the selection evidence, the v0.2.2 validation split is the launch held-out set
  (scored bytes 12,311,466 under the Train rule). The frozen `tokenizer.json`
  (`176a89e2…adf1`) is unchanged by this difference.
- **Selection rule:** pre-registered in Juniper-1-Train `docs/prep/08_evaluation_framework.md`
  §3.1. Bits per byte of a small model trained on equal bytes is the deciding metric, so
  compression alone does not decide.
- **Shards:** uint16 little-endian streams per split, documents in keyed-hash order
  (`j1.pack-order.v1`) separated by ID 0, with a doc index (offset, length, source, domain)
  and SHA-256 per file. Equal-bytes training prefixes for pilots come from
  `token-shards-prefix`.

## Results on `j1-base-v0.2` (candidate set v3, 2026-10-09)

Training sample: 89,016 train-split documents, 391.1 MB (rate 0.03), doc-id set SHA-256 `f3ff84e3b4448a35…`, accepted manifest `1a99ef755084dcfb…`. tokenizers 0.23.2.

| Candidate | Total vocab | Learned | Bytes/token (validation) | Byte fallback | Round-trip failures | Control IDs from text | Embedding params at d=640 | tokenizer.json SHA-256 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| j1tok-24k | 24,576 | 24,256 | 3.8263 | 0.228 % | 0 | 0 | 15,728,640 | `174352e8a65f19ec…` |
| j1tok-28k | 28,672 | 28,352 | 3.888 | 0.234 % | 0 | 0 | 18,350,080 | `a50ee49930006f9b…` |
| j1tok-32k | 32,768 | 32,448 | 3.9343 | 0.234 % | 0 | 0 | 20,971,520 | `9e173919d4e704cb…` |

Bytes per token by validation source:

| Source | j1tok-24k | j1tok-28k | j1tok-32k |
|---|---:|---:|---:|
| arxiv-papers | 3.286 | 3.334 | 3.3691 |
| finemath-4plus | 3.3294 | 3.3689 | 3.3936 |
| fineweb-edu-dedup | 4.4474 | 4.521 | 4.5776 |
| libretexts | 4.4695 | 4.5428 | 4.6008 |
| oercommons | 4.255 | 4.3038 | 4.3345 |
| pressbooks | 4.1103 | 4.2031 | 4.2614 |
| python-peps | 4.2688 | 4.3123 | 4.4089 |
| stackexchange | 3.5644 | 3.6105 | 3.6448 |
| stackv2-edu | 3.4312 | 3.4914 | 3.5396 |
| wikimedia | 3.8192 | 3.8876 | 3.9446 |

Probe token counts (lower is more compact; all round-trip exactly):

| Probe | j1tok-24k | j1tok-28k | j1tok-32k |
|---|---:|---:|---:|
| control_literals | 62 | 61 | 60 |
| english | 55 | 55 | 55 |
| identifiers_rare | 90 | 84 | 81 |
| json_tools | 111 | 108 | 107 |
| math | 214 | 214 | 212 |
| paths_urls | 142 | 133 | 127 |
| python | 186 | 182 | 181 |
| science | 90 | 90 | 89 |
| structured | 129 | 126 | 125 |
| unicode_misc | 108 | 106 | 104 |

Indentation: 4, 8, 12 and 16 spaces are each a single token in every candidate.

History (preserved on disk; the selection rule was revised before any pilot ran):

| Set | Change | 32K overall fallback | STEM probe fallback |
|---|---|---:|---|
| v1 (`candidates-v1`) | alphabet limit 1,000, frequency only | 0.410 % | ħ, ψ, ∇, ⇔, ⟨⟩ … on bytes |
| v2 (`candidates-v2`) | + printable ASCII and STEM alphabet, limit 1,500 | 0.234 % | `Ĥ` on bytes (science probe) |
| v3 (`candidates`) | + hatted letters and combining accents | 0.234 % | none |

The remaining fallback is incidental scripts (CJK, Hangul, Sinhala, emoji) by design. It stays lossless.

Selection between the three candidates uses the A100 pilot (equal-bytes bits per byte, rule v1.1
in Juniper-1-Train `docs/prep/08_evaluation_framework.md`). Compression alone favours 32K by
2.8 %, while 24K saves 5.2M embedding parameters (3.5 % of the 148M model).

## Corpus recount per candidate (`j1-base-v0.2`, token shards verified 2026-10-09)

Token-ID shards are uint16, in keyed-hash document order, one `<|endoftext|>` after every document.

| Candidate | Train document tokens | Train stream (with separators) | Validation | Test | tokenizer.json SHA-256 |
|---|---:|---:|---:|---:|---|
| j1tok-24k | 3,435,832,033 | 3,438,793,874 | 3,227,367 | 2,657,053 | `174352e8a65f19ec…` |
| j1tok-28k | 3,384,119,981 | 3,387,081,822 | 3,176,152 | 2,615,915 | `a50ee49930006f9b…` |
| j1tok-32k | 3,343,943,878 | 3,346,905,719 | 3,138,789 | 2,583,900 | `9e173919d4e704cb…` |

Every candidate exceeds the ≈3B accepted-token target (directive §4) for the 2,961,841 training
documents. Building each set took ≈ 17 min on FLOWBOX (peak RSS 3.7 GB), and
`token-shards --verify-only` reports no problems for any of them.

**Pilot prefixes (`cand-<name>-prefix-700MB`):** the same 156,588 leading training documents
(699,996,069 text bytes) for every candidate, giving 183.1M (24K), 180.3M (28K) and 178.2M (32K)
stream tokens, plus the full validation split. This is the equal-bytes condition of selection
rule v1.1.
