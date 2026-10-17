# Quality audit: j1-base-v0.1 accepted corpus

Method: `j1data audit` drew a keyed-hash sample of accepted documents (four per
source and kind), boundary cases near the English-function-word and code-line
thresholds, and three audit-retained rejected documents per rejection reason. The
pack (`reports/j1-base-v0.1/audit_pack.md` in the data root; not committed because it
contains corpus text) was read in full for admitted samples and one excerpt per
rejection reason. Corpus-wide prevalence of artifacts found in the sample was then
measured over every accepted document.

## Admitted samples

The sampled documents read as intended: educational web articles (science,
history, computing), Wikipedia/Wikibooks/Wikinews articles, math and programming
Stack Exchange questions with answers, FineMath worked problems, open-textbook
sections, arXiv papers with front matter, PEPs, Cinqic documentation, and Python /
other-language source files and READMEs (machine learning, data processing, SDKs).

Observations:

| Observation | Corpus-wide prevalence | Assessment |
| --- | --- | --- |
| `<EMAIL_ADDRESS>` redaction marker from the Common Pile release | arXiv 70%, PEPs 94%, Pressbooks 2.4%, LibreTexts 2.0%, Stack Exchange 1.2%, code 0.7% of documents | Harmless, but a different placeholder convention from Cinqic's (`email@example.com`). Left unchanged; recorded as a limitation. |
| SlideShare / "presentation transcript" page chrome | FineMath 1.3% of documents | Low-value chrome inside otherwise mathematical pages; below the threshold for a source-specific rule. |
| "Click here" phrasing | FineWeb-Edu 0.85%, FineMath 0.55% | Mostly incidental instructions inside educational pages. |
| Document *kind* labels | Lines ending in `;` (learning-objective bullets, hymn verses) count as code lines, so some prose is labelled "technical" | Kind is a measurement and a choice of threshold set; technical thresholds are slightly more permissive. |
| Format detection | `yaml` matches any `key: value` line | The `yaml` share (26% of tokens) overstates real YAML; treat format shares as upper bounds. |
| A Markdown file with Devanagari verses in the code source | single sample | Non-Latin share below the technical-text limit. |

## Rejected samples

Reviewed one excerpt for each of 52 rejection reasons. The rules fire on what they
are meant to catch: template stubs and disambiguation pages, page chrome, minified,
vendored or generated code, data-heavy code, non-English text, corrupted decoding,
repetitive pages and "percent calculator" pages. Known false-positive classes:

- `quality.spam_terms` rejects some legitimate Wikipedia articles about
  cryptocurrency or gambling (for example a biography mentioning Coinbase). The
  lexicon applies to prose and math kinds only.
- `quality.prose.not_english` on Wikipedia marks list-like pages (surname lists,
  season tables), not foreign text.
- `privacy.pii.payment_card` (1,952 documents) and `privacy.secret.password_assignment`
  (5,357) are conservative: tutorials with realistic-looking example values are
  rejected rather than admitted.

These losses are small against a pool roughly ten times the corpus budget; none
removes a domain.

## Statistics (measure-32k, provisional)

- Spelling evidence (documents with US or UK variant spellings): 918,614 US-leaning
  versus 242,271 UK-leaning (79% US); 1,806,032 documents have no signal.
- English function-word share of accepted documents: median 0.357, p5 0.074 (code),
  p95 0.457.
- Document length (tokens): FineWeb-Edu median 674 (p99 7,867); code median 555;
  Wikimedia median 478; arXiv median 16,226 (p99 72,394).
- Source concentration: largest source 31.3% (FineWeb-Edu); Herfindahl index 0.185
  (13 sources).
