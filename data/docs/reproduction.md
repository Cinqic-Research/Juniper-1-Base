# Reproducing a build

All commands run from the repository root with the package installed
(`pip install -e '.[dev]'`) and `J1_DATA_ROOT` pointing at large storage. Processing
commits immutable segments and a transactional journal every five active minutes.
Release-build resumes always re-hash committed outputs; `--verify` enables the same
check for exploratory builds. Changed source manifests, build scope, selection,
split rules, or code require a new build id. Finalization refuses a stale or
out-of-scope marker inventory. Partial source or unit runs require a distinct
exploratory build id and cannot target the configured release build.

```bash
export J1_DATA_ROOT="/path/to/large-storage/juniper1-data/data"
```

`git_clones.json` maps `owner/repo` to local clones of the Cinqic repositories used by
the Cinqic documentation sources and the Cinqic evaluation exclusions. Clones are read
only at the pinned commits named in `configs/`, never from their working trees.

## 1. Acquire pinned sources

```bash
python -m j1data acquire --workers 4 --git-clones "$J1_DATA_ROOT/git_clones.json"
```

Hugging Face files are fetched at the pinned commit, resumed with HTTP Range requests
and accepted only when size and upstream LFS SHA-256 match. Acquisition manifests are
written to `$J1_DATA_ROOT/source_manifests/` (copied to `manifests/sources/`).

## 2. Build the evaluation-exclusion index

```bash
python -m j1data exclusions --git-clones "$J1_DATA_ROOT/git_clones.json"
```

The index binds the hashes of the selected benchmark source files into its identity.
It uses per-item 13-gram coverage for long items and complete normalized matching
for eligible short items; protected material that cannot be matched remains a
documented coverage limit.

## 3. Process, deduplicate, finalize

```bash
python -m j1data process --workers 12 --git-clones "$J1_DATA_ROOT/git_clones.json"
python -m j1data dedup
python -m j1data select
python -m j1data finalize --workers 10
```

`process` refuses to run on a dirty source tree and binds `work/<build id>/build.json`
to the build identity (pipeline configuration, stable source-file identities, release
scope, registries, exclusion source hashes and index, and output-affecting code).
Before reading any selected source file, it re-hashes Hugging Face inputs and compares
their pinned Git tree/blob identities. A changed input requires a new build id;
finished builds are never overwritten.
Run one `process` invocation per build at a time; on FLOWBOX run it inside a
memory-capped scope (`systemd-run --user --scope -p MemoryMax=12G ...`).

For each logical source unit, `process` publishes completed candidate, reject, and
signature segments before advancing its progress journal. On restart, the journal's
build, source, raw-file, exclusion-index, and output identities are checked, then the
pipeline resumes from the first uncommitted record. Dolma gzip data is decompressed
through the recorded line cursor, but already committed records are not filtered or
written again. A corrupt committed segment is moved under the build's quarantine
directory and that segment and its suffix are replayed.

`process` covers the units named in `configs/build_scope.toml`; `select` fills the
per-source budgets of `configs/selection.toml` from the deduplicated pool, and
`finalize` writes only selected documents. Do not reuse an accepted output directory
after changing source, scope, selection, split, or code inputs; use a new build id.

## 4. Provisional token counts (corpus sizing only)

```bash
python -m j1data measure-tokenizers
python -m j1data count-tokens
```

`measure-tokenizers` trains the provisional measurement tokenizers and saves the
tokenizer-stage sample and probes; `count-tokens` writes per-document count tables.
Neither produces a Juniper 1 tokenizer or token-ID shards.

## 5. Reports, audit and verification

```bash
python -m j1data report
python -m j1data audit
python -m j1data calibrate-dedup
python -m j1data verify --contamination-sample-rate 0.0005 \
    --git-clones "$J1_DATA_ROOT/git_clones.json"
```

## Tests and static checks

```bash
ruff format --check src tests && ruff check src tests && mypy && pytest
```

The end-to-end test (`tests/test_end_to_end.py`) runs the whole pipeline on planted
fixtures, checks every routing decision and count, corrupts a shard to test resume
verification, and rebuilds to confirm byte-identical outputs.
