# Filter calibration record (2026-10-07)

Every processing rule was run on real units of each source before the release build
(`work/calib-*` builds on FLOWBOX). Each finding below follows observe -> diagnose ->
minimal intervention -> re-measure. Numbers are document counts for one unit unless
noted. Negative and superseded findings are kept.

| # | Source / unit | Observation | Diagnosis | Intervention | Effect |
| --- | --- | --- | --- | --- | --- |
| 1 | OER Commons, LibreTexts | 26-30% rejected by `duplicate_ngrams` | Repeated n-gram share counted the *first* occurrence too, roughly doubling Gopher's measure | Count repeat occurrences only (Gopher definition); regression test | LibreTexts admission 69% -> 80% |
| 2 | PEPs, OER Commons | HumanEval / MBPP / MMLU "contamination" in PEPs and history lessons | Matches were digit runs (`1 2 3 ... 13`), alphabet runs, zero padding, and famous quotations inside MMLU passages | Index only 13-grams with >= 6 alphabetic words; require >= 50% coverage of one evaluation item | PEPs 7 -> 0, OER 28 -> 3 flagged; residual: a quotation that is most of an MMLU item still flags (accepted, conservative) |
| 3 | LibreTexts | 112 phone redactions, 49 card rejections | Space-separated digit groups in tables; free-form 13-19 digit pattern | Phone numbers need phone punctuation; cards must be grouped 4-4-4-4 / 4-6-5 or follow a card keyword | Tables no longer match |
| 4 | LibreTexts, Pressbooks | Long processing on some documents | Quadratic regexes: `\S*[letter]\S*` on long letter-free tokens (8 s for 32k characters), cross-line `\s*` in format patterns | Linear patterns; adversarial 200k-character regression tests; slow-document log in unit markers | All adversarial shapes < 0.2 s |
| 5 | LibreTexts | `duplicate_lines` on `|` and `-` lines; site chrome on thousands of pages | Lines without letters or digits counted as duplicates; LibreTexts page chrome (`- Save as PDF`, `- Last updated`, `Skip to main content`, ...) | Symbol-only lines ignored for the duplicate-line count; exact chrome lines removed for LibreTexts (count kept per document) | see 6 |
| 6 | LibreTexts | 5,000 remaining repetition rejects of good pages | Learning objectives, key points and summaries restate body sentences; share flat across n (0.22-0.29) | `curated` profile for LibreTexts / Pressbooks / OER Commons: flat 0.35 repeated-n-gram limit, all other rules unchanged | LibreTexts 80% -> 94%, Pressbooks 91%, OER 85% |
| 7 | Pressbooks, arXiv | 1,875 / 3,108 `license.document_not_allowed` | CC0 written as `Creative Commons Zero - Public Domain - ...`, missed by the allowlist pattern | Allowlist both spellings | CC0 documents admitted |
| 8 | LibreTexts, Pressbooks | `alpha_words` rejects of good pages | Markdown table pipes and bullets count as letter-free words | Pure layout tokens (`|`, `*`, `#`, `>`, `-`, dashes, `:`, `===`) excluded from the ratio's denominator; operators still count | — |
| 9 | Stack v2 Markdown | 9.5% rejected by `code.long_lines` | READMEs judged by code-file rules (one paragraph per line) | Markdown in the code source uses technical-text rules | — |
| 10 | Stack v2 code | 1,000-1,800 `spam_terms` per unit | Web-spam lexicon (`bitcoin`, `crypto`, `betting`) fires on legitimate code and Q&A | Spam lexicon applies to prose and math kinds only | — |
| 11 | arXiv | 27% `math.duplicate_ngrams` | Repeated notation (`x_ p y_ p`, `tilde r _ a`), not repeated prose | For math kind, repeated n-grams are found among words of >= 4 characters that are not notation commands; formatting macros (`mbox`, `boldmath`) still count so broken conversions are rejected | 100 of 122 audited rejects now admitted; the remaining 22 are broken macro expansions or lemma-heavy repetition |
| 12 | arXiv | 134 `payment_card` | 4-4-4-4 digit groups in numeric tables pass Luhn 1 in 10 | Card numbers considered only when the document mentions cards (`card`, `visa`, `credit`, `cvv`, `expir`, ...) | — |
| 13 | arXiv | 2-3 s "slow documents" | Profiled in isolation: 0.03 s; CPU contention with 12+ workers | None | — |
| 14 | FineMath | 70.6% admitted | Notation filter divided by the filtered character total, inflating fractions on short pages | Fractions use the whole document's characters | 70.6% -> 79.5% |
| 15 | FineMath | 20% still rejected | Audit: about half clearly low value (search listings, OEIS boilerplate, forum chrome, betting-odds spam), half borderline exercise lists and forum threads | None: standard thresholds kept; quality over quota | 79.5% |
| 17 | Wikimedia (English Wikipedia units) | 71-87% of main-namespace articles admitted; `too_short`, `top_ngram_repetition`, `not_english` dominate | Audit: stubs under 50 words, disambiguation pages, and list or table pages (surname lists, season tables, taxonomy lists) whose English function-word share is low; `not_english` here marks list-like pages, not foreign text | None: low prose value; recorded | — |
| 16 | Stack v2 | Score-2 files are 54-75% of code bytes | This release keeps classifier scores below the published Stack-Edu cut | `min_int_score = 3` (Stack-Edu / SmolLM2) | Code volume at the intended share |

FineWeb-Edu (already filtered upstream) admits 97.8% of a 100,000-document unit;
the main residual rejects are repeated n-grams (1.3%) and bullet or ellipsis
listings.
