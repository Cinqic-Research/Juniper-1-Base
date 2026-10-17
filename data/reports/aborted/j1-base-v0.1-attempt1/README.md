# Aborted attempt 1 of j1-base-v0.1 (2026-10-07, 17:16-19:47 EDT)

Stopped deliberately after 43 of 312 units: per-worker memory grew to 1.3-1.9 GB
(12 workers on 15 GB RAM), swap filled and the machine began thrashing.

Causes found: MinHash signatures were held as a Python list of small arrays and
stacked at the end of a unit (about 0.9 GB peak on million-document Stack Exchange
units); Parquet buffers flushed every 4,096 rows regardless of size (hundreds of MB
for arXiv papers); a 2M-entry word-hash cache.

Fix (commit "Bound per-worker memory ..."): signatures streamed to disk and
transposed block-wise, row groups cut at 32 MB, 300k-entry cache; measured peak
about 0.5 GB per worker. The build restarts under the same build id with a new
identity (the processing-module fingerprint changed); these markers are kept as
evidence and were not reused.
