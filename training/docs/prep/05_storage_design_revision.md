# Storage design revision (proposed for spec v0.6): interim topology on FLOWBOX HDD + GitHub + Hugging Face

**Status:** PROPOSAL requiring owner approval. It does not silently weaken v0.4/v0.5. It
records which v0.4/v0.5 storage guarantees the currently available resources can meet,
which they cannot, and when work must stop.

## 1. Measured facts (2026-10-08)

| Resource | Observation |
|---|---|
| FLOWBOX HDD | `/dev/sda1`, ext4, UUID `3b337640-…`, HGST HTS545050A7E380 (500 GB 2.5″, 5400 rpm), 458 GB usable, **217 GB free** before the v0.2 rebuild. SMART: passing, 0 bad sectors, 0 failed attributes, 30 °C, **32,704 power-on hours (~3.7 years)**. It is a single drive with no RAID |
| FLOWBOX uplink | Network bursts of **~30 MB/s** to the Hugging Face Xet CAS (interface counters, 6 × 10 s samples), but end-to-end push throughput of the v0.1 corpus was only **1.3 MB/s effective**: 5.21 GB in 4,033 s across one failed and one resumed attempt, while the HDD was shared with the corpus rebuild. Client-side chunk hashing on the 5400-rpm HDD, not the network, dominates. SSH to RunPod (proxy) measured 2.4–2.5 MB/s and the downlink 12.6–13.0 MB/s |
| Hugging Face | Free tier: **100 GB private storage** (per user or org, Hub docs "Storage limits", read 2026-10-08), best-effort public storage, files < 200 GB recommended. Private repos live under org `CinqicX` (admin). There is no quota API, so the tool counts bytes it stored (`store status` → `hf_bytes`) |
| GitHub | Private repos (Juniper-1-Data 781 KB, Juniper-1-Train 97 KB). Git LFS is not used. GitHub holds source, configuration, manifests and small evidence only |
| RunPod | Global volume `653478ff…` (~9.5 GB, beta, geesefs, console-only attachment) at ≈ $0.07/GB-month. No network volumes in Secure A100-SXM datacenters. Container disk is wiped on every stop |

## 2. Roles

| Location | Holds | Failure domain |
|---|---|---|
| FLOWBOX HDD `juniper1-archive/` and `juniper1-data/data/` | canonical local archive: raw acquisitions, accepted builds, work evidence, tokenizer, shards, checkpoints, receipts, audit log | one aged physical disk in the owner's home |
| Hugging Face private repos (`CinqicX/juniper1-*`) | immutable copies of accepted corpora, token shards, tokenizer releases, selected checkpoints, each with its manifest | third-party cloud, account-scoped |
| GitHub private repos | code, configs, manifests, decisions, compact evidence, artifact hashes | third-party cloud |
| RunPod global volume / container disk | training working storage only | provider; container disk is ephemeral |

### Repository layout

| Repo | Type | Content |
|---|---|---|
| `CinqicX/juniper1-corpus` | private dataset | `<build>/accepted/…` accepted Parquet shards plus `_manifests/` |
| `CinqicX/juniper1-shards` | private dataset | `<build>/<tokenizer>/…` token-ID shards plus manifests |
| `CinqicX/juniper1-tokenizer` | private model | frozen tokenizer releases (`tokenizer.json`, `freeze.json`) |
| `CinqicX/juniper1-checkpoints` | private model | resumable checkpoints required by retention, by run id |

Nothing is published publicly. `HFRepo.ensure_repo` refuses to store to a public repo.

## 3. Proof levels

Every copy has a receipt (`j1bench/store.py`) at one of three levels. They are not
interchangeable:

| Level | What it proves | What it does not prove |
|---|---|---|
| Git manifest committed | the expected bytes (paths, sizes, SHA-256) are known | that any copy of the bytes exists |
| `COMMITTED` | the destination accepted an upload | that the bytes are intact or retrievable |
| `REMOTE_HASH` | the Hub's stored LFS/Xet metadata matches the manifest SHA-256 and size | that this account can download them now |
| `BYTES_VERIFIED` | the bytes were read back into a fresh location and re-hashed | durability beyond the moment of verification |

An artifact counts as **independently backed up** only when it has `BYTES_VERIFIED` on the
FLOWBOX HDD **and** at least `REMOTE_HASH` on Hugging Face, plus a periodic `BYTES_VERIFIED`
restore drill of a representative sample.

## 4. Requirement-by-requirement status under this topology

| v0.4/v0.5 requirement | Interim status | Residual risk |
|---|---|---|
| FLOWBOX HDD canonical archive | **Met** (path, device identity, receipts, hash-chained audit log) | single aged disk; no RAID; no offline copy |
| Independent cloud backup | **Partially met**: HF private repos are a separate failure domain for corpus, shards, tokenizer and selected checkpoints | free-tier storage is "best-effort" policy-wise for public data, with a 100 GB private cap; no SLA; account compromise affects every copy |
| Backup of the FLOWBOX archive itself (v0.4 §7) | **Not met for raw acquisitions (95 GB)** and v0.1/v0.2 work directories (≈48 GB each) | raw sources are recoverable from pinned upstream revisions with LFS SHA-256 (acquisition manifests). If upstream deletes or gates them, only the HDD copy remains |
| Two pod-independent durable destinations for active training (v0.3 §6) | **Not met as specified.** The primary is the RunPod global volume (pod independence not demonstrated, G10). The secondary is the FLOWBOX pull replica, then HF push from FLOWBOX | at 1.3 MB/s effective, a 1.78 GB checkpoint reaches HF ≈ 23 min after arriving on FLOWBOX; global volume is beta |
| No cloud deletion before local + independent verification | **Met procedurally** (store receipts; no deletions in this assignment) | — |
| Quantitative sync deadlines | **Proposed** below | — |

## 5. Capacity budget (bytes)

| Item | HDD now | HDD after this preparation (projected) | HF (projected) |
|---|---:|---:|---:|
| Raw acquisitions (PRESERVE, upstream-recoverable) | 95 GB | 95 GB | 0 |
| v0.1 work + accepted (RETAIN_UNTIL PR #1 review closes) | 53 GB | 53 GB | 5.2 GB (accepted only; pushed 2026-10-08) |
| v0.2 work + accepted (PRESERVE accepted; RETAIN work) | 0 | ≈ 53 GB | ≈ 5.2 GB |
| Tokenizer candidates and frozen tokenizer | 0 | < 0.1 GB | < 0.01 GB |
| Token shards, 3 candidates (frozen one PRESERVE; others RECOMPUTABLE) | 0 | ≈ 20 GB | ≈ 6.8 GB (frozen tokenizer only) |
| Pilot checkpoints (selected, investigation) | 5.6 GB | ≈ 10 GB | ≈ 3 GB |
| Other Cinqic material on the same disk (AAA, models, GPT-OSS qualification, …) | ≈ 65 GB | ≈ 65 GB | — |
| **Free space** | **217 GB** | **≈ 140 GB** | **≈ 80 GB of 100 GB remaining** |
| Canonical run retention (148M: 1.78 GB/ckpt; ≈ 15 retained for a ≈ 9 h run) | — | +27 GB | +27 GB (if retained to HF) |

The minimum free-space reserve is **20 GB** on the HDD (`--reserve-gb`, enforced by
`LocalDir.put` and `pull`). Work stops before an operation would go below it.

## 6. Proposed sync deadlines and stop conditions (to freeze in the run configuration)

| Rule | Value |
|---|---|
| Max age of an unarchived regular checkpoint on FLOWBOX | 60 min of active training |
| Max unsynchronized checkpoints (not on FLOWBOX) | 2 |
| Stage-boundary / best / release-candidate checkpoint | training pauses at the next boundary until `BYTES_VERIFIED` on FLOWBOX **and** `REMOTE_HASH` on HF |
| FLOWBOX unreachable | continue only while the global-volume primary verifies and both limits above hold; then stop at a checkpoint boundary |
| HDD free < 20 GB + projected retention | refuse new runs; stop at the next boundary |
| HF push failure (auth, quota, network) | retries are idempotent; after 3 failures, alert and stop at the next stage boundary |
| HF private bytes > 85 GB | stop adding HF copies; owner decision needed (paid storage or pruning per retention) |

The HF copy remains the slow step. At the measured 1.3 MB/s effective rate, a 9-hour
canonical run producing ≈ 15 retained checkpoints (27 GB) needs ≈ 5.8 h of push time.
That fits inside the run only because checkpoints arrive over 9 h; burst rates suggest
it can be much faster when the HDD is not shared. The run configuration must use the
measured rate and must stop on backlog, never on assumption.

## 7. Conditions that must stop work

- Any `StoreError` refusal: conflicting bytes, manifest mismatch, capacity, or an unmounted
  archive.
- A SMART status change on the HDD: reallocated or pending sectors > 0, or a failing
  attribute.
- Any attempt to delete a cloud copy that has no `BYTES_VERIFIED` HDD receipt plus an
  independent copy.

## 8. Migration path once storage is purchased

1. Add a second local disk or NAS as a distinct failure domain. Run
   `store push` with `local:` destinations for every PRESERVE manifest, and convert the
   receipts.
2. Back up raw acquisitions (95 GB) to that disk first. They are the largest asset with a
   single Cinqic-controlled copy.
3. Optionally add an S3-compatible bucket as a third destination. The `Destination`
   protocol needs only `stat`, `put`, `fetch` and `free_bytes`.
4. Move the training primary to a network volume or object store that a replacement pod
   can attach by API, and re-run G10.
