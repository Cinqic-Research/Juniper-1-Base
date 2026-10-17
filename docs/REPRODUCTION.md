# Reproducing Juniper 1 Base

This guide separates what can be reproduced, at which level, with what evidence. Claims are made only at
the levels Cinqic established; the last level (bit-for-bit retraining) was neither run nor claimed.

| Level | Status | How to check |
|---|---|---|
| 1. Exact artifact recovery | **Established.** The released `model.safetensors` holds the exact float32 tensors of the frozen checkpoint; the loader reproduces the tensor-tree digest `1fd07c0d…076e4`, the config matches the freeze record, the parameter count is 142,771,456. | `juniper1 verify --model CinqicX/juniper1-base` |
| 2. Inference from published files | **Established** (tested from a clean virtual environment with only the public package and Hub access, 2026-10-16; see `docs/RELEASE_RECORD.md`). Reference loader (`training/j1bench/frozen.py` on `model.pt`) versus the release loader: identical logits on CPU fp32 and identical greedy continuations on the recorded prompts. | `evidence/parity_reference_vs_safetensors.json`; `scripts/parity_check.py` |
| 3. Dataset reconstruction | **Pipeline and identities published; full rebuild not re-executed for this release.** The pipeline (`data/`) is deterministic given the pinned upstream files, the configs and the code at the recorded commits; the accepted-manifest identity, the per-document hashes and the exclusion lists are published, so a rebuild can be checked document by document. The v0.2.2 token shards regenerate to equal token bytes and document indexes but the regenerated `shards.json` differs in a `seconds` metadata field (`data/docs/stage_recovery.md`). The upstream sources are public at their pinned revisions; FineWeb-Edu and FineMath text is not redistributed by Cinqic and must be re-acquired from upstream. | `docs/DATASET.md`, `data/docs/reproduction.md`, `data/reports/j1-base-v0.2.2/` |
| 4. Runnable training recipe | **Published.** `training/j1bench` at the exact code identity recorded in the checkpoint; the training contract pins every hyperparameter; `training/README.md` gives the command line. A CPU smoke test of the training loop on a synthetic stream and the save/resume equivalence test run in CI (`tests/test_resume.py`). | `training/docs/contract/training_contract_v2.json` |
| 5. Same-environment resume equivalence | **Established at training time** (full-state checkpoints; a restart after the update-8,585 power loss replayed updates with identical batch fingerprints, verified by the 2026-10-12 audit) and by the CPU resume tests. | `training/docs/audit_2026-10-12/INDEPENDENT_AUDIT_REPORT.md` |
| 6. Numerical reproduction of the reported validation numbers | **Established** on a different GPU (RTX 2060, fp16 and fp32) to within 1e-4 bpb of the A100 bf16 logged values. | `training/docs/audit_2026-10-12/results/full_fp32.json`; `training/tools/verify_bpb.py` |
| 7. Bit-for-bit full pretraining replication | **Not run, not claimed.** Expected tolerance for a rerun on the same hardware class with the same seed: not measured. BF16 autocast on an A100 with the SDPA flash backend is not bitwise deterministic across kernels or hardware. | — |

## Environment

Recorded training environment: one NVIDIA A100-SXM4-80GB, driver 580.126.16, CUDA 12.8, torch 2.8.0+cu128,
numpy 2.1.2, Python 3.12.3 (`training/env/flowbox-venv.lock.txt` is the FLOWBOX verification environment;
`training/env/pod-overlay-requirements.txt` the pod overlay). Inference needs only `torch>=2.4`,
`tokenizers`, `safetensors`, `numpy`.

## Data (level 3)

1. Install the pipeline: `cd data && pip install -e '.[dev]'` and set `J1_DATA_ROOT` to large storage (the full
   build needs about 150 GB of raw downloads and 20 GB of outputs; FLOWBOX stage timings are in
   `data/docs/data_card.md`).
2. `python -m j1data acquire` downloads the pinned upstream files (exact revisions and SHA-256 in
   `data/manifests/j1-base-v0.2/sources/*.acquisition.json`). The three internal Cinqic documentation sources in
   the registry were excluded from v0.2.2 and are not needed.
3. `exclusions`, `process`, `dedup`, `select`, `finalize` as in `data/docs/reproduction.md`, then derive the
   v0.2.2 successor by applying the published exclusion list
   (`data/reports/j1-base-v0.2.2/excluded_doc_ids_union.json`, 7,108 documents) to the v0.2 build as recorded in
   `data/reports/j1-base-v0.2.2/successor_lineage.json`.
4. Check: the accepted manifest SHA-256 must be `54a3adb9650c96bf67b44b721386722383caeb774b48b0bf970ad6f9634d4185`
   and the per-document `sha256`/`norm_sha256` must match the published reconstruction manifests.
5. `python -m j1data.tokcli token-shards --build-id j1-base-v0.2.2 --tokenizer cand5-j1tok-24k`: token files and
   document indexes are expected byte-identical to the launch shards (shown on a 5 % subset plus all held-out
   shards, not on the full set); `shards.json` will differ in `seconds`. Launch `shards.json` SHA-256:
   `1cf5e2320ad35e930f84ca0edbf5a9d188a2f524702dc4623e492dab043e3aef`.

## Training (level 4)

The recorded run (`j1-base-r2`) used, in `training/`:

```bash
python -m j1bench.train --preset j1-wide-20x768 --vocab 24576 --seq-len 2048 \
  --micro-bs 16 --accum 1 --steps 52034 --lr 0.0016971 --warmup 500 --min-lr-ratio 0.1 \
  --wd 0.1 --wd-policy matrices --clip 1.0 --opt fused --no-compile --seed 1 \
  --tokens <J1_DATA_ROOT>/token_shards/j1-base-v0.2.2/cand5-j1tok-24k --limit-tokens 3410100225 \
  --preload-data --verify-data --tokenizer <tokenizer.json> \
  --eval-every 500 --eval-windows 0 --ckpt-root <root> --ckpt-every-steps 500 --ckpt-every-seconds 900 \
  --run-id j1-base-r2 --metrics <metrics.jsonl>
```

The exact flags, including the data-order and evaluation identities, are bound in the checkpoint manifest
(`full-state/manifest.json` on the Hub, `run_contract`) and in `training/docs/contract/training_contract_v2.json`
(effective batch 32 × 2,048 = 65,536 targets per update; the launch used micro-batch 16 with accumulation 2 as
recorded there). Data order: windows of the EOS-separated train stream in keyed-hash document order, visited
in a permutation seeded by (seed = 1, epoch = 0); the stream prefix of 3,410,100,225 tokens drops the last six
windows. Interruption and resume: full-state checkpoints (model, AdamW moments, scheduler, RNG, data cursor)
every 500 updates or 900 active seconds; `--resume auto` restores the latest verified checkpoint.

## Evaluation (level 6)

`python training/tools/verify_bpb.py` with the validation shards, tokenizer and a checkpoint recomputes bits
per byte under the protocol in `training/docs/audit_2026-10-12/base_evaluation_protocol_v1.md` (fixed 2,048-token
windows over the EOS-separated validation stream, cross-entropy in fp32, sums in fp64, bytes from the
tokenizer's decoded UTF-8 length).
