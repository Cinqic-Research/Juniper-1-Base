# Engineering decisions (j1-base v0.1)

Each entry records the decision, the evidence behind it and what would change it.
Authority order: current owner directive, Juniper 1 Design Specification v0.3
(8 Oct 2026), subsequent explicitly approved owner decisions, repository evidence,
historical research, external practice. The 7 Oct v0.1 and 8 Oct v0.2 specifications
are historical editing baselines.

## D1. A dedicated repository and storage layout

No Juniper 1 model or data repository existed (organization listing, 2026-10-07). The
retired Juniper-LM-1 and Juniper-Reference-4B repositories and the Juniper-LM-1.1
qualification repository are not reused. Code, configuration and committed manifests
live in `Juniper-1-Data`; corpus bytes live on the Cinqic Storage volume under
`J1_DATA_ROOT` (`raw/`, `exclusions/`, `work/`, `accepted/`, `token_counts/`, `tokenizer_stage/`,
`reports/`). The root SSD is not used for corpus data.

## D2. Sources: openly licensed first, web text for the educational foundation

Owner direction: established, large, high-quality open datasets are the raw
material; Cinqic's layer adds validation, filtering, privacy, global dedup,
decontamination, domain measurement, rebalancing and provenance. No public corpus
is rebuilt from individual websites. Every bulk source below is an existing
dataset; the only first-party source is the small Cinqic documentation allowlist,
which no dataset covers. Alternatives evaluated (Dolma v1.7, OLMo 2 mix, Dolmino,
DCLM-baseline, FineWeb, Proof-Pile-2, StarCoderData, The Stack) are recorded with
revisions and reasons in `configs/considered_sources.toml`. Only the shards that
contribute to the intended mixture are downloaded (keyed-hash subsets of
FineWeb-Edu, Stack Overflow and per-language code).

The design asks for contemporary American-English educational text as the foundation,
with more code, mathematics and science. Selected:

* FineWeb-Edu (SmolLM-Corpus deduplicated) and FineMath 4+ for the web foundation and
  mathematics. Both are ODC-By compilations of Common Crawl; page copyright stays with
  authors, so the state is `TRAINING_ONLY_NO_REDISTRIBUTION` and the training-use
  position is flagged for owner sign-off (design spec section 7, "Approve source
  inventory, licenses").
* Common Pile v0.1 sources with per-document open licenses (Wikimedia, Stack Exchange,
  arXiv open-license papers, LibreTexts, Pressbooks, OER Commons, PEPs) and Stack v2
  educational code restricted to Blue Oak permissive repositories. Every document's
  license field is checked on ingest; anything outside the allowlist is rejected with
  `license.document_not_allowed`. Upstream license evidence does not equal owner
  authorization: CC BY-SA treatment and Stack v2 provenance remain pending in
  `configs/owner_decisions.toml`.
* A small allowlist of Cinqic public Markdown documentation (Apache-2.0).

Not admitted, with reasons, in `configs/considered_sources.toml`: gated BigCode sets
(terms not accepted), Software-Heritage-ID-only code sets, GPL-including code,
synthetic Cosmopedia, OpenStax (relicensed NC-SA in March 2026), the official Stack
Exchange dump (no-LLM clause), unlicensed peS2o, and the Cinqic History Dataset
(private partitions, not approved for training distribution).

File subsets (FineWeb-Edu shards, Stack Overflow shards, per-language code shards) are
chosen by a keyed SHA-256 ordering of file paths (`j1.select.v1.*` namespaces), which is
reproducible and independent of listing order.

## D3. Two text representations

Stored text receives only meaning-preserving repairs (line endings, BOM, C0 controls,
unpaired surrogates, outer blank lines). NFKC, case folding, accent stripping and
punctuation removal are applied only to the matching signature used for hashing,
deduplication and decontamination; NFKC would rewrite superscripts, ligatures and
full-width operators in mathematical and programming text. Tests pin both behaviours.

## D4. Layered, domain-aware quality filtering

Documents are first assigned a kind (prose, math, technical, code) from source and
content (LaTeX density, code-line share), then checked against kind-specific thresholds:
Gopher rules for prose (Rae et al., 2021), relaxed symbol and alphabetic-word rules for
mathematics and technical text, and StarCoder file rules for code (line length,
alphanumeric share, generated or vendored files, data-heavy files). All failing reasons
are recorded, not only the first.

Calibration finding: the first implementation counted the *first* occurrence of a
repeated n-gram as duplicate text, roughly doubling Gopher's measure and rejecting
30% of LibreTexts. Gopher counts only repeat occurrences; the measure was corrected and
a regression test pins it.

## D5. Privacy

Credentials (provider-specific keys, private keys, tokens, credentialed URLs,
authorization headers, high-entropy password assignments) and high-risk identifiers
(SSNs with context, Luhn-valid grouped card numbers) reject the whole document;
partial redaction of a leaked credential still teaches its shape. Incidental e-mail
addresses, North-American phone numbers, public IPv4 addresses and home-directory user
names are replaced by fixed synthetic values (the FineWeb convention for e-mail and IP).
Documented example values (AWS `...EXAMPLE` keys, the jwt.io token, processor test
cards, `555` numbers, private and documentation IP ranges, version strings) are left
alone. Matched values are never logged or stored; rejected-text audit samples are never
retained for privacy rejections. Tests use credentials assembled at run time so no
credential-shaped literal exists in the repository.

Calibration findings: a free-form 13-19 digit card pattern and space-separated phone
pattern fired on number tables in textbooks; both were narrowed (grouped card formats
or a card keyword; phone numbers require phone punctuation).

## D6. Evaluation decontamination by item coverage

The exclusion registry (`configs/exclusions.toml`) pins GSM8K, ARC, BBH, HumanEval(+),
MBPP(+), MMLU, MATH, HellaSwag and every Cinqic evaluation set (JuniperBench-Code
v1/v1.1/v1.2, GPT-OSS qualification and anchors, Juniper Reference evaluation, Juniper
App wrapper evaluation, AAA exams and Erudition data) by exact revision. Records are
split into items (for example a GSM8K question and its answer separately). Long items
use normalized word 13-grams; complete 6–12 word items with at least 48 characters and
five informative words also use exact normalized word-sequence matching.

Calibration finding: "any shared 13-gram" (GPT-3 style) flagged documents for digit
runs, alphabet runs and zero padding shared with benchmark items, and for famous
quotations embedded in MMLU history passages. The long-item rule indexes only 13-grams
with at least six alphabetic words of two or more characters, and rejects a document
when it covers at least half of one item's 13-grams. The short-item rule requires the
complete normalized word sequence; canaries are checked separately. A known residual:
when a quotation is most of an MMLU item, documents quoting it are
removed; this costs a little data and never admits a copied item. AAA confirmation
evidence is protected by path exclusion in the Cinqic sources rather than n-gram
indexing. New benchmarks are added by extending the registry and re-checking accepted
shards; sources need not be re-acquired.

## D7. Corpus-wide deduplication

Identity, exact (SHA-256 of stored text), normalized-exact (SHA-256 of the signature)
and MinHash-LSH near deduplication run over all sources at once in a single global
order (source priority, source, unit, row), so the copy from the most trusted source is
retained regardless of which adapter delivered it. Word 5-gram shingles and 112
permutations follow FineWeb / datatrove. Full 32-bit signatures are stored, so the
banding (14 x 8, detection 50% at Jaccard 0.72) is applied at dedup time and was
calibrated on the real candidate pool (`docs/dedup.md`) without reprocessing text.
Calibration found clusters of unrelated code files joined by shared license headers;
code documents are therefore re-signed at dedup time without license and copyright
comment blocks (`j1.code-signatures.v1`), leaving stored text unchanged.

## D8. Splits by group, with a leakage check

Split membership is a keyed hash of a group key (web host, code repository, textbook,
paper or question URL), so related documents land together and assignment needs no
global state. After deduplication, a held-out 13-gram index removes from training any
document covering half of a held-out document (or whose own 13-grams are half
held-out), catching cross-split near copies below the LSH threshold.

## D9. No tokenizer is designed or frozen here; counts are provisional

Owner scope correction (7 Oct 2026): this work builds the corpus in tokenizer-
independent form. The Juniper 1 tokenizer (vocabulary size, merges, control-token
IDs, chat template, hashes, training shards) is designed and frozen in a separate
stage. To size the corpus against the 3B+ objective, byte-fallback BPE
*measurement* tokenizers at 24,576 / 28,672 / 32,768 IDs (256 byte tokens, no
special tokens, no normalizer, singly split digits) are trained on a keyed-hash
sample of the accepted training split, and every accepted document is counted
under each. These counts carry the measurement tokenizer's hash and are labelled
provisional everywhere. The same sample and a validation-split domain probe set
are saved under `tokenizer_stage/` as inputs for the tokenizer stage; no token-ID
shards are written.

## D10. Performance engineering

Profiling on FLOWBOX found quadratic regular expressions (a letter-free 32k-character
token took 8 s; cross-line `\s*` in format detection) and expensive case-insensitive
alternations scanned at every position. These were replaced by linear scans, literal
prefilters and anchored patterns, with regression tests on adversarial 200k-character
inputs; process segments also record any document that takes longer than 2 s.

## D11. Per-source budgets select the corpus from a larger clean pool

Owner directive (8 Oct 2026): about 3B+ accepted tokens, 15B a hard ceiling. Only the
units needed for headroom are processed (`configs/build_scope.toml`; the file hash is
part of the build identity), and after deduplication `j1data select` fills per-source
token budgets (`configs/selection.toml`, about 3.4B provisional tokens, converted to
bytes with measured bytes per token). FineWeb-Edu and FineMath are filled in order
of their upstream quality scores; other sources by a keyed-hash uniform sample, with
per-language budgets for code; small curated sources are taken whole. Unselected
documents are counted and recorded, not deleted, so budgets can be revised without
reprocessing. Budgets are provisional and versioned with the selection identity.

## D12. Memory-bounded release processing

The first release attempt was stopped when 16 workers (1.3-1.9 GB each) exhausted
memory and swap on FLOWBOX (`reports/aborted/j1-base-v0.1-attempt1/`). Workers now
stream signatures to disk in blocks, cut Parquet buffers by bytes and bound the word
hash cache (peak about 0.5 GB per worker); finalize reads 256-row Arrow batches.
Heavy stages run inside a memory-capped systemd scope so an overrun ends the job, not
the machine. The v0.1 unit markers did not make every stage safely resumable; that
claim was too broad and is superseded by D13 and the v0.3 compliance record.

## D13. Transactional data-stage checkpointing (v0.3)

The v0.3 specification makes tested recovery mandatory throughout the project
lifecycle. Processing now commits immutable candidate, reject, and signature
segments every five active minutes, with a journal bound to the build, source
revision, raw-file hash, exclusion-index identity, cursor, ID samples and hashes,
and output hashes. Release-bound resume re-hashes committed outputs before reuse.
The compressed Dolma adapter must decompress through a saved line cursor, but it
does not reapply filtering to committed records. A deterministic fixture interrupts
the journal after committed segments, resumes the suffix, and checks that committed
IDs and files remain unchanged.

This decision covers the process stage only. Deduplication, selection, finalization,
and provisional token measurement still require explicit 15-minute recovery
qualification; no current test establishes their full v0.3 recovery contract. The
external training stack's durable replicas, complete model/optimizer/data state,
restore qualification and failure-injection evidence are also not implemented in
this repository. Canonical training remains blocked until those stages pass their
own gates.
