# Source exclusion: internal Cinqic repository documents (criteria, 2026-10-10)

Written before the successor `j1-base-v0.2.2` is derived. Engineer decision under the owner work order of 2026-10-10
(§6: "use source exclusion or versioned successor data when more reliable than ambiguous document-level filtering").

- **Problem:** the accepted corpus includes documents from Cinqic's own repositories (`cinqic-aaa-docs`,
  `cinqic-juniper-app-docs`, `cinqic-calculator-docs`). The internal evaluation suites (AAA exams/Erudition,
  App wrapper, reference and qualification suites) are built from the same repositories, and the strengthened
  contamination layers match only 6-21 % of those items (qualification 2026-10-10 §4). Document-level filtering
  therefore cannot certify that these sources are free of internal evaluation content or answers.
- **Rule:** every document of the three `cinqic-*` sources is excluded from base-pretraining data, whatever its scan
  result. Public sources are unaffected. The list is `reports/j1-base-v0.2.2/excluded_internal_source_doc_ids.json`
  (all train split; validation and test hold none).
- **Size:** 100 documents, 141,281 tokens under `j1tok-24k` (0.004 % of train tokens).
- **Scope of effect:** the protocol-18 architecture comparison uses the v0.2.1 960 MB prefix, which holds 9 of these
  documents (4 AAA, 5 App). They are not prohibited by any adjudication and the comparison metric is validation
  bits per byte on the unchanged validation split, so the comparison stays applicable; the dependency is recorded
  in the training contract's invalidation map. These sources may return only in later stages with evaluation
  suites held out by construction (separate allocation and decision).
