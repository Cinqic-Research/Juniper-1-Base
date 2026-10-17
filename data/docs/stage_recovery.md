# CPU data-stage recovery (gate J1-13)

Status: **PRE-REGISTRATION** (written 2026-10-10 before any code change on branch
`opus/stage-recovery-2026-10-10`; results are appended in the "Results" section below
and never edit the plan or pass criteria above it).

Owner rule (v0.3, 15-minute rule): persist CPU-stage progress at most every 15 minutes and
at committed transaction boundaries, for deduplication, selection, finalization, audits,
tokenizer training, counting, token-shard generation and any new long operation. Where a
library cannot resume its internals, bound the restartable unit and preserve upstream
inputs/outputs; an unresumable full restart is not called exact recovery.

## Vocabulary

* **Unit**: the smallest piece of work whose output is committed as a whole.
* **Commit**: outputs written to a temporary name, fsynced, renamed; then a marker or
  journal entry that lists each output with SHA-256 (and row/token counts) is written
  atomically (write-temp, fsync, rename, fsync directory). The marker is always written
  last ("manifest-last"); an output without a verifying marker is never reused.
* **Sub-unit resume (exact)**: after a crash, work resumes from the last committed unit
  inside the stage; outputs are byte-identical to an uninterrupted run.
* **Bounded restart**: the stage has no internal resume point, but its whole run is
  reliably shorter than the 15-minute interval at corpus scale, publishes atomically
  (manifest-last), leaves its upstream inputs untouched, and is re-run from those inputs.
  Lost work is at most one stage run. This is reported as BOUNDED-RESTART-ONLY, never as
  exact sub-unit recovery.

## Per-stage inventory (before changes)

Real-scale figures come from the v0.2 build logs on FLOWBOX (`data/logs/*.log`, GNU time
`-v`): 20,196,491 candidates, 543 finalize units, 1,345 accepted shards (518 train),
3.44 B train stream tokens (`cand5-j1tok-24k`). "15B" means the hard corpus ceiling,
about 4.5x the current corpus.

| Stage | Module | Unit of work | Current persistence / resume | Real wall time, peak RSS | Honest claim today |
| --- | --- | --- | --- | --- | --- |
| Processing | `process.py` | unit split into segments committed every 300 s | Journal per unit, segment outputs hashed, quarantine and suffix replay; tested (`test_processing_checkpoints_are_immutable_and_contiguous`) | 4 h 11 min total; segments <= 300 s | Exact sub-unit resume (existing evidence, code unchanged here) |
| Code re-signing (dedup prep) | `code_signatures.py` | one code unit | Atomic `.npy` per unit; reuse checks only the array shape, not a hash | part of dedup log (66 units) | Unit-level, but reuse is not hash-verified |
| Deduplication | `dedup.py` | whole corpus (3 exact passes + 14 LSH bands + output) | None. `summary.json` written last, but `keep/*.npz` are written before it and `np.savez` embeds wall-clock timestamps (not byte-reproducible) | 12 min 50 s, 2.93 GB (O(N) arrays) | Full restart only; at 15B (~4.5x) projected >15 min, so the rule is NOT met at ceiling scale |
| Selection | `select.py` | per source | Masks written per source with atomic rename, `summary.json` last; no per-source marker, rerun recomputes everything; a stale summary from an earlier run is not invalidated before masks are rewritten | 2 min 46 s, 2.96 GB | Bounded restart (stage < 15 min even at 15B by linear projection, ~12.5 min, which is too close to call) |
| Finalization | `finalize.py` | held-out index (global), then per-unit accepted shards | Per-unit marker with hashes, verified before reuse; held-out index rebuilt every run with non-atomic `np.save`; manifest last | 20 min 12 s total (4 workers), 1.73 GB; per unit seconds | Unit-level resume exists but has no recorded kill drill |
| Contamination audit scan | `audit_scan.py` | one accepted shard | Per-shard report + digest-bound progress journal; tampered/stale reports recomputed (tested) | ~30 min, not recorded | Unit-level resume, no real kill drill |
| Sample audit pack | `audit.py` | whole pack | Non-atomic `write_text` of the Markdown pack | seconds | Bounded restart, but a partial pack could be left in place |
| Measurement tokenizers | `cli.py measure-tokenizers` | one provisional tokenizer | HF `tokenizers` training cannot resume; `tokenizer.save` not atomic; always retrains | minutes | Bounded restart per tokenizer (library limit) |
| Provisional counting | `token_counts.py` | one accepted shard | Per-shard marker, output hash verified on reuse | 31 min 10 s, 2.6 GB (357 shards) | Unit-level resume, no kill drill |
| Tokenizer candidates | `tokstage.py` | one candidate (train + eval) | `eval.json` written last per candidate, but reuse trusts its existence without checking the tokenizer hash or the training-sample identity | 2 min 44 s for 3 candidates, 3.1 GB | Bounded restart per candidate (library limit); reuse not verified |
| Tokenizer freeze | `tokstage.freeze` | one directory | `.partial` dir renamed last | 0.4 s | Atomic publish |
| Token-shard generation | `tokshards.py` | whole build (3 splits, 2 passes) | None: `.partial` directory, spill restarted from zero; orphan `cand4-*.partial` directories exist on the real root; all document metadata kept in memory; `shards.json` embeds wall-clock seconds | 16 min 42 s single (train 1,002 s), 50 min with three concurrent builds; 3.78 GB | Full restart only; exceeds 15 min; NOT meeting the rule |
| Prefix export | `tokshards.export_prefix*` | one directory | `.partial` dir renamed last | 56 s | Atomic publish, bounded restart |
| Successor derivation | `tools/successor_manifest.py` | one directory | Output must not exist; writes in place | ~1 min | Bounded restart; partial output blocks rerun |
| Verify / report / gates | `verify.py`, `report.py`, `gates.py` | read-only recomputation | No outputs to recover beyond a result file | 0-11 min | Read-only, rerun from inputs |

## Plan

1. **Token-shard generation** (largest gap). Replace the monolithic two-pass build with
   committed units: pass 1 per accepted shard appends records (doc id, source, domain,
   key, tokens) to per-bucket spill logs with a journal that records each bucket's
   committed byte length (commits at most every `CHECKPOINT_SECONDS`, default 120 s, and
   at split end); restart truncates every spill log to its committed length and continues
   with the next shard. Pass 2 per bucket writes the bucket's tokens directly into their
   final positions in pre-sized token files (offsets known from pass-1 totals) and a
   per-bucket index part, each committed with region SHA-256. Finishing hashes and renames
   token files, writes the document index, then `shards.json`, then renames the directory.
   Wall-clock seconds move out of `shards.json` into `timings.json` so `shards.json` is
   byte-reproducible. Memory becomes O(bucket), not O(corpus documents).
2. **Deduplication**. A journal in `dedup/state/` saves the pass state (`alive`,
   `removed_kind`, `retained_by`, union-find `parent`, next band) after each exact pass
   and at most every `CHECKPOINT_SECONDS` (default 300 s) during the LSH bands; restart
   resumes at the next band. Union-find state after band *b* is fully captured by
   `parent`, so the result is identical. Keep files are written deterministically
   (fixed zip timestamps) and listed with hashes in `summary.json`, which stays last.
   Code re-signing reuse becomes hash-verified through a per-unit marker.
3. **Selection**. Per-source commit marker (mask hashes + source summary); invalidate the
   stage summary before any mask is rewritten; resume skips sources whose marker verifies.
4. **Finalization**. Held-out index written atomically with a hash marker and reused only
   when it verifies; existing unit markers kept.
5. **Tokenizer candidates**. Reuse a candidate only when `eval.json` matches the current
   `tokenizer.json` hash and the current sample identity; write candidate files atomically.
   Training remains a bounded restart (HF `tokenizers` has no resumable state).
6. **Audit pack** and measurement tokenizers: atomic publish (bounded restart).
7. **Fault hook**: `J1_FAULT_POINT=<name>` (+ optional `J1_FAULT_AFTER=<n>`) sends SIGKILL
   to the current process at the n-th pass through a named commit point. No-op otherwise.

This branch changes eight `build.BUILD_MODULES` files relative to the v0.2 build code
(base `00555d3`, whose 24 module hashes equal `work/j1-base-v0.2/build.json`
`inputs.build_code`): `build.py`, `code_signatures.py`, `cli.py`, `dedup.py`, `finalize.py`,
`select.py`, `shards.py` and `token_counts.py`, and adds two new modules to the list,
`faults.py` and `pools.py`. The build code fingerprint therefore differs from the v0.2 one, so
existing work directories cannot resume under this code; any corpus built from scratch under
it receives a new identity. j1-base-v0.2.2 is a successor derivation (lineage identity
`89b08d57…bffc5`, four lineage keys only), which this code change leaves unchanged. Outputs
are intended to be semantically identical to the previous code (checked against the real
v0.2 dedup outputs where feasible; see "Reproducibility of the v0.2.2 token shards" below).

Recovery bindings (review D-02): the dedup journal/summary and the per-source selection
markers bind `build.build_code_fingerprint()` (the per-module hashes), not the git commit.
An unrelated commit between a crash and the restart (docs, reports, drill evidence) no
longer discards a committed journal; a changed build module does. `code_revision()` stays
in the summaries as an informational field, and `build.prepare` still refuses a dirty tree
unless `--allow-dirty` is given.

## Tests (pre-registered)

Automated (`tests/test_stage_recovery.py`), for each of dedup, selection, finalization,
counting, audit scan, tokenizer candidates and token shards where applicable:

* T1 kill mid-unit via the fault hook (subprocess, SIGKILL), restart, outputs
  byte-identical to an uninterrupted run;
* T2 corrupt or truncated committed output is rejected and recomputed (or the run refuses);
* T3 missing marker or journal entry causes recomputation, never reuse;
* T4 replay of a completed stage is a no-op (all output hashes unchanged);
* T5 disk-full simulation (ENOSPC injected at a write) fails non-zero without
  publishing, and a following run completes byte-identically.

Real-operation drills on a representative real subset (inputs hard-linked read-only from
the real data root into the scratch root; outputs only in scratch; scratch <= 8 GB): run
uninterrupted; run with SIGKILL at a random time (uniform over the reference wall time),
restart, compare every output file byte-for-byte; record wall time, peak RSS
(`/usr/bin/time -v`), kill time, units redone, and the maximum interval between commits.

## Pass criteria (pre-registered)

* P1 the restarted run exits 0;
* P2 every output file is byte-identical to the uninterrupted run (no exclusions are
  expected after timings move to sidecar files; any excluded field is listed explicitly);
* P3 no `.tmp`/`.partial` file remains in a published location and no consumer accepts one;
* P4 progress lost on a kill <= one unit and <= 15 minutes: the maximum observed interval
  between commits is <= 900 s at the drill scale and the unit duration projected to the
  15B ceiling is <= 900 s;
* P5 replay of a completed stage changes no output hash;
* P6 corrupted outputs are detected and never silently reused.

Bounded memory (corpus-scale concern): for token-shard generation, peak RSS at two corpus
fractions (target 5% and 25% of the v0.2.1 train split) must grow by less than 1.5x while
input grows 5x. For deduplication, which keeps O(N) global arrays by design, peak RSS is
measured at 25%, 50% and 100% of the v0.2 candidates and fitted per document; it passes
only if the projection is <= 4 GiB at 3B tokens and <= 12 GiB at the 15B ceiling,
otherwise the concern stays OPEN with the measured slope.

Verdict labels: VERIFIED (existing evidence, unchanged code), IMPLEMENTED+TESTED, 
BOUNDED-RESTART-ONLY, NOT DONE.

## Results (2026-10-10, appended after the pre-registration commit `3b19a87`)

Implementation commits: `def3c45` (committed units, fault hook, tests, drill tool) and
`c10d566` (document index written from single-chunk columns). Evidence JSON, logs and
GNU time records: `juniper1-archive/autonomous-2026-10-10/results/stage_recovery/` on the
FLOWBOX HDD; compact copies in `reports/stage_recovery/`.

### What changed

* `tokshards.build` is now a sequence of committed units (spill commits at most every
  120 s, one unit per bucket in pass 2, per-file finishing), with an identity file that
  refuses to resume foreign partial output, verification before publish, and wall-clock
  timings moved to `build_log.json`. Token files are byte-identical to the previous
  implementation (fixture and the 5% real subset).
* `dedup.run` commits its pass state after each exact pass and at most every 300 s across
  the 14 LSH bands, in generation-numbered files so a crash during a commit leaves the
  previous commit intact; keep files are byte-reproducible and listed with SHA-256 in the
  summary; a verified summary makes replay a no-op. Code re-signing reuse is hash-verified.
* `select.run` commits one marker per source and invalidates the stage summary first.
* Finalization, processing and code re-signing use a fail-fast pool (`j1data.pools`):
  previously a worker killed by SIGKILL or the OOM killer left `multiprocessing.Pool`
  waiting forever. Workers exit if their parent dies.
* Tokenizer candidates are reused only when `eval.json` matches the tokenizer hash, spec
  and sample identity; the audit pack and measurement tokenizers publish atomically.
* `j1data.faults` (`J1_FAULT_POINT`, `J1_FAULT_AFTER`, `J1_FAULT_ACTION=kill|enospc`).

### Automated fault tests (`tests/test_stage_recovery.py`, 36 tests after the review fixes)

Child-process SIGKILL at 7 token-shard points, 6 dedup points, 2 selection points, a
finalize worker and the finalize parent, 3 counting points, the audit scan and tokenizer
training; ENOSPC injected in token shards and dedup; corrupt spill chunk, corrupt placed
token region, missing journal, foreign partial identity, corrupt dedup state, damaged keep
file, corrupt selection mask, corrupt accepted shard plus missing finalize marker, corrupt
count table, tampered tokenizer. Every case finishes byte-identical to an uninterrupted run
(timing files excluded), leaves no temporary file, and replay of a completed stage is a
no-op or an immutability refusal.

### Per-stage verdict

| Stage | Verdict | Unit and bound at the 15B ceiling | Evidence |
| --- | --- | --- | --- |
| Processing | VERIFIED (code unchanged except the fail-fast pool) | segment, committed every 300 s | existing `test_processing_checkpoints_are_immutable_and_contiguous`; no new real drill |
| Code re-signing | IMPLEMENTED (no dedicated fault test) | one code unit, marker with SHA-256 | `code_signatures.resign` |
| Deduplication | IMPLEMENTED+TESTED; memory OPEN | exact pass or <= 300 s of bands; a band at 15B is estimated at a few minutes | pytest; real drill (10% subset) PASS |
| Selection | IMPLEMENTED+TESTED | one source (largest at v0.2 well under the 166 s stage total) | pytest; real drill PASS |
| Finalization | IMPLEMENTED+TESTED (held-out index rebuilt per run: bounded restart) | one process segment | pytest (worker and parent kill, corruption, missing marker); real drill not completed |
| Counting | IMPLEMENTED+TESTED | one accepted shard | pytest; real drill not run |
| Contamination audit scan | IMPLEMENTED+TESTED | one accepted shard | pytest; real drill not run |
| Tokenizer candidates | BOUNDED-RESTART-ONLY (library has no resumable state) | one candidate (train + evaluate) | pytest; real drill not run |
| Measurement tokenizers, audit pack, freeze, prefix export, successor derivation | BOUNDED-RESTART-ONLY (atomic publish, no fault test) | whole command (seconds to minutes) | code inspection |
| Token-shard generation | IMPLEMENTED+TESTED | <= 120 s of spill plus one shard; one bucket in pass 2 | pytest; real kill + resume PASS; memory stress PASS |

### Real-operation drills (hard-linked real inputs, outputs in scratch)

| Stage | Input | Reference wall, peak RSS | Kill | Restart | Files compared | Result |
| --- | --- | --- | --- | --- | --- | --- |
| Deduplication | 10% of v0.2 units (1,857,537 candidates, 14 bands) | 77 s, 498 MiB | SIGKILL of the process group at 36 s (random) | resumed at `near:0` (three exact passes kept), 65 s | 50 | byte-identical, PASS |
| Selection | same subset, configured quotas | 63 s, 2,059 MiB | SIGKILL at 25 s (random) | 29 s (committed sources reused) | 54 | byte-identical, PASS |
| Token shards | v0.2 accepted, 5% of train shards (148,641 documents, 149.8 M train tokens) + all held-out shards; frozen `cand5-j1tok-24k` | uninterrupted build of the same inputs | real SIGKILL of the process group in pass 2 (14 of 64 train buckets committed) by the disk-space guard, not at a planned time | 45 s, 364 MiB, resumed at bucket 14 | 7 (incl. `shards.json`) | byte-identical, PASS |

The finalization drill was stopped by the disk-space guard during its reference run
(the HDD fell below the 30 GB floor because of other jobs), and the counting, audit-scan
and tokenizer drills did not start. Those stages have only the automated fault tests above;
no real-data kill drill was recorded for them, and the full-scale (20.2 M candidate) dedup
kill drill was cancelled for time.

Old versus new token-shard implementation on the 5% real subset (commit `c10d566`): all
token files and all three document indexes are byte-identical; `shards.json` differs only
by the removed `seconds` field.

### Reproducibility of the v0.2.2 token shards (review D-03)

The launch token shards `token_shards/j1-base-v0.2.2/cand5-j1tok-24k` were produced by the
previous `tokshards` writer (the v0.2 build code era, base `00555d3` / `e3f243e`; derivation
on `opus/autonomous-2026-10-10`, which does not contain this branch). Their `shards.json`
(SHA-256 `1cf5e232…3aef`, bound in the recount record, `MANIFEST.sha256` and the HF upload)
contains per-split `seconds` fields, which the new writer no longer emits (timings go to
`build_log.json`). The v0.2.2 `shards.json` is therefore **not bit-reproducible** by the new
`tokshards.build`: a rerun would yield a `shards.json` equal except for `seconds`
(`reports/stage_recovery/memory.json`: `shards_json_equal_without_old_seconds`). Byte identity
of the token files and document indexes between the old and new writer was shown on a 5 %
subset of the v0.2 train shards (148,641 documents, 149,814,008 train stream tokens) plus all
held-out shards, **not** on the full v0.2.2 set. Any regenerated artifact (token files,
`shards.json`, document indexes) is a new artifact with its own hashes and must receive a new
identity and new records; the launch records bind the existing files only.

### Bounded memory

Token-shard generation (GNU time, `MemoryMax=4G` scope):

| Train documents | Previous implementation | New implementation |
| --- | --- | --- |
| 148,641 (5% of shards) | 1,218 MiB | 1,097 MiB |
| 598,982 (20% of shards; the 25% shard draw) | 2,476 MiB (2.03x) | 1,211 MiB (1.10x) |

Pre-registered criterion: < 1.5x while the input grows 5x (5% to 25%). The measured input
grew 4.03x by documents (5% to 20%: the 25% shard draw holds 20% of the documents); the new
implementation grew 1.10x, and a linear extrapolation to 5x gives about 1.14x: PASS for the
new implementation, on a 4.03x rather than 5x measurement. Its memory is one bucket plus the
tokenizer; the previous one grew with the document count. Wall time, both pairs measured
while other jobs shared the machine: at 5%, 182 s new versus 41.5 s previous (4.4x); at 20%,
274 s (198 s spill) new versus 211 s previous. The per-bucket fsync and commit overhead is
real at small scale and is not characterised at full scale. The longest commit interval in
that run was 124.6 s (spill, `CHECKPOINT_SECONDS` = 120 s plus one shard).

Deduplication keeps O(N) global arrays (128-bit keys, union-find parent, sort orders):

| Candidates | Peak RSS | Source |
| --- | --- | --- |
| 1,857,537 | 498 MiB | drill reference (new code) |
| 7,698,532 | 1,368 MiB | uninterrupted run, new code |
| 20,196,491 | 2,866 MiB | historical v0.2 run (previous code, same arrays) |

That is about 125-155 bytes per candidate above a ~0.2 GiB base. Projection: about
2.7 GiB at the current ~3.4 B-token corpus (PASS against <= 4 GiB) and about 11-13 GiB at
the 15B ceiling (about 90 M candidates), which straddles the pre-registered 12 GiB bound and
exceeds the 4 GiB job cap used on FLOWBOX. **The corpus-scale memory concern for
deduplication remains OPEN.** The 50% point (12.26 M candidates) was stopped by the
disk-space guard. A partitioned design (exact passes by key-prefix partition; LSH bands
already processed one at a time with only the 8-byte union-find array global) would bound
the per-pass arrays.

### Progress-loss bound (P4)

* Token shards: <= 120 s of spill plus one accepted shard (about 1.5 s at v0.2), or one
  bucket in pass 2 (about 1/64 of the split; seconds at v0.2, an estimated tens of seconds at
  15B). Measured maximum interval 124.6 s.
* Deduplication: one exact pass, or <= 300 s of bands plus one band. At v0.2 a band takes
  about 40 s and an exact pass under a minute; at 15B both are estimated at a few minutes,
  inside 15 minutes, but this was not measured at that scale (and memory does not fit).
* Selection, counting, audit scan: one source or one accepted shard. Finalization: one
  process segment, plus a held-out index rebuild per run (bounded restart, not measured
  separately).
* Tokenizer candidates: one candidate (14-22 s training at the v0.2 sample; full restart of
  that candidate, not exact recovery).

### What remains

1. Real-data kill drills for finalization, counting, the audit scan and tokenizer
   candidates (automated fault tests exist; `tools/stage_drill.py drill` runs them, about
   2-6 minutes each on the 10% subset), and a full-scale dedup kill drill.
2. Dedup memory at the 15B ceiling (OPEN), and a measured band/pass duration at that scale.
3. Code re-signing has hash-verified reuse but no dedicated fault test; processing has no
   new real-data drill (its existing evidence is reused unchanged).
4. Plan deviation (plan item 4): the held-out index is not published atomically with a hash
   marker; `finalize.build_heldout_index` still rebuilds it with `np.save` on every run
   (bounded restart; a torn file is always overwritten before use). Its rebuild time at the
   15B ceiling is not measured.
5. The freeze gate for aggregate CPU/data recovery stays NOT VERIFIED.
6. The changed modules alter the build code fingerprint: existing work directories cannot
   resume under this code. j1-base-v0.2.2 was built with the previous code; the next build
   takes a new identity. The v0.2.2 `shards.json` is not bit-reproducible by the new writer
   (see "Reproducibility of the v0.2.2 token shards").
7. `shards.remove_stale_temporaries` (review D-15) sweeps `*.tmp` / `*.tmp.npy` recursively at
   stage start. This is safe only under a **single-writer-per-directory** assumption: at most
   one process works in a stage directory (`work/<build>/dedup`, `…/selection`, an accepted
   or token-count output directory, a token-shard `.partial`) at a time. No lock enforces it;
   a second process started in the same directory would delete the first process's live
   temporaries and could then race on the committed files. Operators run one stage command
   per directory; the three concurrent token-shard builds in the inventory used distinct
   directories. Adding an `flock` on the stage directory is a possible hardening and would
   change `shards.py` (a build module).

### Independent review (2026-10-10, head `543ddec`): APPROVE_WITH_FIXES

| # | Sev | Finding | Disposition |
| --- | --- | --- | --- |
| 1 | HIGH | J1-13 row in `spec_compliance_v0.3.md` overstated ("IMPLEMENTED AND TESTED") | Fixed: PARTIAL with the open items |
| 2 | MED | Kill between dropping `.identity.json` and the final rename stranded a finished partial; no directory fsync after the rename | Fixed: a partial whose `shards.json` verifies and matches the inputs is accepted and re-identified; the identity is dropped last and the parent directory is fsynced; fault point `tokshards.publish.identity_dropped` tested |
| 3 | MED | Kill inside `write_json_atomic` left `<name>.<pid>.tmp`, later published | Fixed: `remove_stale_temporaries` at the start of token shards, dedup, selection, finalization and counting; token shards refuse to publish unexpected files; fault point `json.atomic.tmp` tested |
| 4 | MED | Finish phase hashed regions and index parts without checking their commit hashes | Fixed: every bucket region and index part is re-checked against the journal before publication; tests damage a published token file and an index part after commit |
| 5 | MED | Partial identity did not bind the code | Fixed: SHA-256 of `tokshards.py` and `jtok.py` in the identity |
| 6 | MED | Kill tests proved identical bytes, not resumption | Fixed: token-shard tests assert the resumed phase and committed progress and no spill discard; selection, finalization and counting tests assert committed outputs were reused (unchanged mtime) |
| 7 | MED | Plan item 4 (held-out index marker) not implemented and not listed | Recorded as a plan deviation (What remains 4) |
| 8 | LOW | Memory growth stated as "about 4-5x"; 5% timing pair omitted | Fixed above (4.03x stated; both timing pairs) |
| 9 | LOW | `pools.py` batches by `workers*N` jobs (stragglers idle workers) | Recorded; not measured |
| 10 | LOW | Code re-signing marker binds rows/unit/hash, not MinHash config or candidate hash; no fault test | Recorded (What remains 3) |
| 11 | LOW | `audit.py`/`cli.py` atomic publish without fsync before rename | Recorded |
| 12 | LOW | pytest imported another worktree's editable install | Fixed: `tests/conftest.py` puts this checkout's `src` first |

Re-review of `ea393f7`: **APPROVE**. Remaining LOW findings:

| # | Finding | Disposition |
| --- | --- | --- |
| R1 | Re-identifying a finished partial did not compare the order namespace (the record holds no code hash) | Fixed: namespace compared; a finished, verifying partial from older code with identical inputs can still be adopted (the record is complete and self-consistent) |
| R2 | Temporary sweep ran inside a foreign partial before it was refused | Fixed: sweep runs after the identity is accepted |
| R3 | Audit scan (`mkstemp` dot-files) and tokenizer candidate `eval.json.<pid>.tmp` are not swept | Recorded (tokenizer: harmless, freeze copies only `tokenizer.json`) |
| R4 | Reuse assertion accepts any unchanged file, not every committed unit | Recorded |
| R5 | Publication reads the token stream about four times (I/O only, ~30 GB per read at 15B) | Recorded |
| R6 | A failed finish re-check fails closed and needs a manual partial removal | Recorded (P6 allows fail-closed) |

j1-base-v0.2.2 was derived from the Data worktree on `opus/autonomous-2026-10-10` (archive
`stage/derive_v022.sh`), which does not contain this branch.
