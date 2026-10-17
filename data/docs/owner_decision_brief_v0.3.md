# Juniper 1 source-use decisions for the project owner

**Prepared:** 8 October 2026  
**Specification:** Juniper 1 Design Specification v0.3  
**Status:** Evidence brief only. All four decisions in `configs/owner_decisions.toml` remain `PENDING`.

This brief records what the pinned source material and upstream terms say, the limits
of that evidence, and the exact choices needed from Markus. It is not legal advice,
does not grant source rights, and does not authorize training or release. Training
from source text and a future Apache-2.0 weight release are separate decisions.

## Decisions requested

| Decision ID | Choice to record | Current gate |
| --- | --- | --- |
| `fineweb-edu-training-use` | For the pinned FineWeb-Edu-derived subset, authorize training input after reviewing the ODC-By database grant, the Common Crawl terms (including their AI/ML indemnity), and underlying page rights; or exclude it from a new corpus identity. Record any later weight-distribution position separately. | PENDING |
| `finemath-training-use` | For the pinned FineMath-4+ subset, make the same source-specific training decision after reviewing the compilation grant, Common Crawl terms, and underlying page rights; or exclude it from a new corpus identity. Record later weight treatment separately. | PENDING |
| `cc-by-sa-training-and-weights-treatment` | For CC BY-SA 2.5, 3.0, and 4.0 documents across the listed sources, choose to exclude them from a new corpus identity or retain them after legal review records the training basis, attribution approach, and treatment of publicly distributed weights under the intended Apache-2.0 project license. | PENDING |
| `stackv2-license-and-provenance` | For the pinned Common Pile Stack v2 educational release, choose to exclude it or retain it after reviewing automated license-detection limits, source-chain and bulk-access terms, Software Heritage principles and opt-outs, and the age of the pinned release. Record training and distribution treatment separately. | PENDING |

## Evidence and limitations

### FineWeb-Edu and FineMath

The pinned SmolLM-Corpus card labels its compilation ODC-By and describes the
`fineweb-edu-dedup` subset as Common Crawl educational pages filtered with an
educational-quality classifier. Juniper pins revision
`3ba9d605774198c5868892d7a8deda78031a781f` and selects only a subset of its files.
The pinned FineMath card also labels the compilation ODC-By. It describes
`finemath-4plus` as classifier-selected math material extracted from Common Crawl
WARCs, using URL sources that include FineWeb and additional math datasets. Juniper
pins revision `e92b25a616738fe95dc186b64dfb19f9c8525594`.

ODC-By covers rights in the database and database structure; its text explicitly
does not grant independent rights in individual contents. Its Section 4.3 also
describes a notice condition for publicly using a qualifying Produced Work. Whether
that condition applies to any Juniper artifact is not determined here. Common
Crawl's Terms of Use say crawled material may have separate owner terms, place
third-party rights responsibility on users, recommend legal counsel before use,
and contain an indemnity covering claims tied to AI/ML use, including training.
Those terms do not establish that every underlying page grants training rights.

**Decision limit:** A dataset-card ODC-By label alone does not settle rights in each
page or resolve obligations arising from Common Crawl terms. A source-use approval
must identify the exact pinned subset and intended use. A future weight license or
public release requires its own recorded assessment.

### CC BY-SA documents

The selected Common Pile sources include documents identified as CC BY-SA 2.5, 3.0,
or 4.0, alongside other per-document licenses. Creative Commons says the application
of copyright to AI training depends on jurisdiction and use. Its guidance describes
its conservative training recommendations as likely to over-comply; it does not
settle whether Juniper's weights are adaptations. The legal analysis distinguishes
training copies from publicly sharing works or adaptations. The source texts'
attribution and ShareAlike conditions therefore cannot be converted here into a
definitive conclusion about Juniper weights.

**Decision limit:** Do not infer either that Apache-2.0 weights necessarily violate
ShareAlike or that weights are necessarily outside it. The owner can choose to
exclude CC BY-SA items from a newly identified corpus, or ask counsel to assess the
specific training and release plan and record a supported path. Any choice affects
corpus identity, source mixture, and requalification.

### Stack v2 educational code

Juniper pins `common-pile/stackv2_edu_filtered` revision
`c354dbe88469a1153e97c6a63ac50591849654de`, a Common Pile release committed on
6 June 2025. Its card says the subset was filtered using the Stack v2 creators'
license detection, that repositories with multiple detected licenses were retained
only when all were on the Blue Oak list, and that per-document license metadata is
available. The same card warns that license laundering or inaccurate metadata can
lead to mistaken assignments.

The BigCode Stack v2 card describes the source archive as derived from Software
Heritage, says bulk download requires an agreement with Software Heritage and
INRIA, and its current changelog reports a v2.2.0 removal cutoff of 29 July 2026 for
opt-outs. Software Heritage's LLM principles call for suitable open model licensing
with documentation and tooling, precise identification of source inputs such as
SWHIDs, and author exclusion mechanisms where possible.

**Decision limit:** The pinned Common Pile release predates the later BigCode
opt-out cutoff. Public material reviewed does not establish whether Common Pile
independently applied later removals or whether the separate bulk-access agreement
flows through to this already-published derivative. Repository-level license
detection can miss file-level exceptions. This is a provenance and terms question,
not merely a `permissive` metadata check.

## Primary references

- [SmolLM-Corpus README at the pinned revision](https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus/blob/3ba9d605774198c5868892d7a8deda78031a781f/README.md)
- [FineMath README at the pinned revision](https://huggingface.co/datasets/HuggingFaceTB/finemath/blob/e92b25a616738fe95dc186b64dfb19f9c8525594/README.md)
- [ODC Attribution License 1.0](https://opendatacommons.org/licenses/by/1-0/)
- [Common Crawl Terms of Use](https://commoncrawl.org/terms-of-use)
- [Creative Commons guidance on CC-licensed works for AI training](https://creativecommons.org/using-cc-licensed-works-for-ai-training-2/)
- [Common Pile Stack v2 Edu README at the pinned revision](https://huggingface.co/datasets/common-pile/stackv2_edu_filtered/blob/c354dbe88469a1153e97c6a63ac50591849654de/README.md)
- [BigCode Stack v2 README and terms](https://huggingface.co/datasets/bigcode/the-stack-v2-train-full-ids/blob/main/README.md)
- [Software Heritage statement on LLMs for code](https://www.softwareheritage.org/2023/10/19/swh-statement-on-llm-for-code/)

## Current engineering consequence

All four statuses remain `PENDING`; the source-use gate must remain non-passing.
Corpus engineering can continue under a new, evidence-bound build identity, but no
result here clears the training gate. The v0.3 specification separately blocks
canonical training until the full save, restore, durability, and failure-injection
requirements pass on the qualified A100 stack.
