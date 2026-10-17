# Juniper 1 — A100 / RunPod workflow qualification

Date: 2026-10-08 (UTC times throughout). Run from FLOWBOX; the agent never ran inside a GPU pod.
Authority: owner directive of 2026-10-08 and Juniper 1 Design Specification v0.3, v0.4 and v0.5
(v0.4 and v0.5 arrived during the run; their additions are addressed or listed as open below).

**Verdict: PARTIALLY QUALIFIED.** Compute, correctness, performance, checkpoint/restore, failure
injection, cross-pod recovery and autonomous pod control pass with evidence. The primary
checkpoint store's pod independence is not demonstrated (G10), and the v0.4 storage topology
(independent cloud backup, backup of the FLOWBOX archive) does not exist yet. Nothing here
authorizes canonical training: the corpus (Juniper-1-Data PR #1) is CHANGES REQUIRED, no
tokenizer is frozen, and the architecture is a leading candidate only. All data was synthetic;
every model artifact is marked non-production.

## Acceptance gates

| Gate | Disposition | Evidence |
|---|---|---|
| G01 Context | PASS | 8 org repos inventoried (no existing Juniper 1 trainer anywhere); spec v0.1–v0.5 read; Juniper-1-Data PR #1 review state recorded |
| G02 Authentication | PASS | RunPod MCP list/get/billing/actions all succeeded; SSH key added with owner approval |
| G03 Hardware | PASS | A100-SXM4-80GB, CC 8.0, 108 SMs, MIG disabled, ECC on with 0 errors; `results/env/` |
| G04 Execution | PASS | BF16 training on the 148.3M-parameter model, eager and compiled, 30+ runs |
| G05 Correctness | PASS (synthetic scope) | causal/GQA/gradient/loss-mask unit tests; fused attention vs FP64 forward and backward; learning to the entropy floor; bitwise run-to-run determinism in strict mode |
| G06 Performance | PASS | three repeats per leading config, spread < 0.3%; profiler breakdown |
| G07 Storage | PASS | semantics of each path measured; hashes verified on pod disk, global volume and FLOWBOX; global-volume data survived two stop/start cycles |
| G08 Checkpoint | PASS | full state saved and restored; bitwise continuation in a fresh process |
| G09 Failure recovery | PASS | 8/8 on-pod fault tests plus 6 replica/transfer cases and a replacement-pod restore |
| G10 Durability | NOT VERIFIED | primary (global volume) and FLOWBOX replica both verified, and the replica was restored on another pod; but no other pod could mount the global volume (console-only attachment), so its pod independence is unproven; v0.4 cloud backup and archive backup absent |
| G11 Pod controls | PASS | stop, start and re-stop through the RunPod API, each verified by status, system log, SSH and GPU identity |
| G12 Shutdown | PASS | final state EXITED, container removed 18:15:25, SSH unreachable from 18:15:31 |
| G13 Costs | PASS | estimated $3.55 (target $5, cap $8); provider billing lags and showed $1.86 at report time |
| G14 Reproducibility | PASS | fresh pod rebuilt from a pinned commit and image digest reproduced the reference bit for bit |
| G15 Review readiness | PASS | ruff, ruff format, mypy and 24 tests clean; PR open for independent review |

## A. Infrastructure

| Item | Observed |
|---|---|
| Pod | `4i13u1p447t6e7` "Juniper 1", Secure Cloud, US-MD-1, created 2026-10-08 16:22:22 by the owner |
| GPU | NVIDIA A100-SXM4-80GB, UUID `GPU-7528250f-6ccc-2d13-7f10-a1d3f4d128ab`, bus 07:00.0, 81,920 MiB (79.25 GiB usable by PyTorch), VBIOS 92.00.9E.00.16, 400 W limit, max SM clock 1410 MHz, PCIe Gen4 host link |
| Driver / CUDA | 580.126.16, driver CUDA 13.0; runtime CUDA 12.8, cuDNN 9.10.2, NCCL 2.27.3 |
| Image | `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404` = `sha256:0a360022e8de4375af99430f84e8b38951acc397252163a37ceac7204d01be35` |
| Software | Ubuntu 24.04.3, kernel 6.8.0-100, Python 3.12.3, torch 2.8.0+cu128, triton 3.4.0, numpy 2.1.2 (`results/env/pip-freeze.txt`) |
| Host | 2× AMD EPYC 7742; 128 CPUs visible but cgroup quota 13.6 CPUs (RunPod says 16 vCPU); memory limit 250 GB; 117 GB `/dev/shm`; no swap |
| Disks | `/` overlay 1000 GB container disk (wiped on every stop, restart and env update); `/workspace` = owner's RunPod **global volume** `653478ff-ba87-4cab-b64f-154ee0c7fe62` (fuse.geesefs, object-backed, beta); no pod volume, no network volume |
| Price | $1.79/h GPU (API); owner screenshot adds ~$0.139/h container disk |
| Access | SSH key `flowbox-juniper1-runpod` (ed25519, FLOWBOX `~/.ssh/runpod_juniper1`) in pod env and account; the public port changes on every container start |

Measured basics (`results/env/cuda_qual.out`): dense 8192² matmul 242.6 TFLOP/s BF16, 120.8 TF32,
19.1 FP32. BF16 matmul relative error vs FP64 is 2.9e-3. Fused and foreach AdamW agree to 7.5e-9.
Pinned host→device copies run at 21.4 GB/s (pageable 5.1 GB/s). A device-side assert and an
OOM both exit nonzero.

Storage characteristics (`results/env/storage_qual_timed.out`):

- **Container disk:** 1 GiB write in 0.76 s; hash read 1.07 s.
- **Global volume:** 1 GiB write in 2.0 s (write-back cache), hash read 7.7 s. Rename returns
  success, but RunPod documents it as non-atomic. `chmod` is refused. Listings are cached for
  60 s, and consistency is eventual.
- **Attachment:** RunPod documents global-volume attachment only through the console at deploy
  time. The v2 API rejects it both as `mounts.network` ("Network volume … not found") and as any
  other `mounts` key.
- **Network volumes:** RunPod refuses them in US-MD-1. None of the three datacenters with Secure
  A100-SXM capacity (EUR-IS-1, US-MD-1, US-WA-1) is on RunPod's list of network-volume
  datacenters.

## B. Benchmark suite

The harness is `j1bench/` (model, data, checkpoints, training loop, replicator), with
`tools/bench_matrix.sh`, `tools/recovery_suite.py`, `tools/summarize_bench.py`, and in `qual/`
`cuda_qual.py`, `sdpa_correctness.py` and `profile_step.py`. Reproduction commands are in
`docs/operations.md`; the machine-readable results are `results/bench_summary.{csv,json}`.
Revisions cited in evidence predate a message-only history rewrite; `results/revision_map.json`
maps them to published commits with identical trees.

The model is a benchmark implementation of `j1-core-28x640-v0.2`, built to the spec's stated
values with several choices left open. It has 28 layers at width 640, SwiGLU 1728, 10 query and
5 KV heads with head_dim 64, pre-RMSNorm, per-head RMSNorm QK-Norm before RoPE (θ = 100,000),
tied embeddings, and context 2048. Instantiated parameters are 148,315,264 at vocab 32,768;
24,576 and 28,672 are covered by a closed-form test. QK-Norm form, vocabulary, initialization,
weight-decay exclusions (currently decay applies to all parameters) and the cross-document
attention policy (currently allowed within a packed row) are **not frozen**, and each is recorded
in every run contract.

## C. Performance (synthetic tokens, seq 2048, vocab 32,768)

| Profile | Tokens/s (median, 3 repeats) | Step p50 / p99 | Peak VRAM | MFU† |
|---|---|---|---|---|
| Eager BF16, mb16 (flash SDPA) | 56,178 | 583 / 584 ms | 53.2 GiB | 0.20 |
| **Reference**: eager BF16, strict deterministic, mb16 | 51,350 | 638 / 639 ms | 53.3 GiB | 0.18 |
| **Optimized**: `torch.compile` default, mb16 | 115,249 (115,000–115,319) | 284 / 286 ms | 29.6 GiB | 0.41 |
| Optimized, mb32 | 119,306 (119,237–119,314) | 549 / 552 ms | 57.2 GiB | 0.42 |
| Optimized, mb16 × accum 2, strict deterministic | 102,708 | 638 / 640 ms | 30.2 GiB | 0.37 |
| Optimized, mb16 × accum 4 | 115,979 | 1130 / 1132 ms | 30.2 GiB | 0.41 |
| compile `reduce-overhead` (CUDA graphs), mb16 | 115,668 | 283 / 284 ms | 31.5 GiB | 0.41 |
| FP32 reference (no TF32), mb8 | 8,304 | 1973 ms | 70.9 GiB | 0.03 |

† MFU = (6N + 6·L·H·d·T)·tokens/s ÷ 312 TFLOP/s; an estimate, not a target.

Findings:

- `torch.compile` doubles throughput. The eager step is dominated by unfused elementwise work and
  copies (67% of GPU time). The compiled step is 52% GEMM, 26% fused Triton, 17% flash attention
  and 2.4% optimizer, with the GPU busy 82% of wall time even under the profiler
  (`results/v05/profile_*.json`).
- Compiled runs sit at the 400 W power limit (390–400 W mean). SM clocks average 1321–1344 MHz
  against 1410 for eager, and the driver logged 107 s of SW power capping. The workload is
  power-bound, not memory-bound.
- First-step compile costs 56–66 s cold and about 5 s with a warm inductor cache. The cache lives
  on the container disk, so every stop costs a cold compile (observed 64 s after restart).
- Eager microbatches 32 and 48 run out of memory; at microbatch 32 the failing allocation is the
  4 GiB logits tensor. Compiled microbatch 32 fits at 57 GiB. Memory-efficient SDPA has no GQA kernel ("No available kernel");
  the failure is preserved and nothing falls back silently. Flash and cuDNN perform equally.
- The data path is not a bottleneck: pre-tokenized memmap streaming matched synthetic generation
  (115,905 vs 115,250 tokens/s), with data time 0.8–5 ms per step. Host RSS peaks at 1.2–1.6 GiB.
- `reduce-overhead` is dominated: no gain, slower compile, +2 GiB.
- Strict determinism costs 11% (102.7K vs 115.7K tokens/s at the same configuration). Without
  it, two identical runs differ by up to 1.2e-4 in loss, because the flash-attention backward
  accumulates nondeterministically.

Correctness (`results/v05/sdpa_correctness*.json`, `results/v05/pilot_*`):

- **Fused attention vs FP64:** at the model's attention geometry, flash, cuDNN and math kernels
  have BF16 forward relative error ≤ 3.4e-3 and gradient error ≤ 4.8e-3.
- **Causality:** perturbing future positions changes no prefix output or gradient (exactly 0
  under strict determinism; the 1.2e-4 seen without it is accumulation order). cuDNN attention
  is unavailable in strict mode.
- **Learning:** on a learnable synthetic task (fixed permutation chain, 10% noise) the full
  model fell from 10.52 to 1.46 nats in 200 steps. The reference and optimized profiles saw
  identical batches and learning rates and finished within 0.03% of each other (mean |Δloss|
  over the last 50 steps: 0.001).
- **First task not learnable:** the original affine-progression task stayed near ln V and could
  not demonstrate learning, so it was replaced for the pilots. That pilot is kept in the results.

Projections (`results/projections.json`; estimates only, as no exposure schedule is frozen).
Overheads assumed: 3.3% checkpointing (measured: 29.7 s per 15 min), 2–5% evaluation (not
measured), 0–10% interruption allowance, and −10% throughput sensitivity. Rate: $1.93/h.

| Profile | 3B exposure | 6B | 15B (ceiling) |
|---|---|---|---|
| Optimized, mb16 | 7.2 h raw; 7.6–9.6 h; $15–18 | 14.5 h; 15.2–19.2 h; $29–37 | 36.2 h; 38.1–47.9 h; $73–92 |
| Optimized, strict deterministic | 8.1 h; 8.6–10.8 h; $16–21 | 16.2 h; 17.1–21.5 h; $33–41 | 40.6 h; 42.7–53.8 h; $82–104 |
| Reference eager | 16.2 h; 17.1–21.5 h; $33–42 | 32.5 h; 34.2–43.0 h; $66–83 | 81.1 h; 85.5–107.6 h; $165–207 |

The 32K vocabulary is the most expensive option under evaluation, real tokenized data may pack
differently, and checkpoint overhead depends on tokens per update (see D).

## D. Recovery

Checkpoint state (`j1bench/ckpt.py`, `j1bench/train.py`):

- **State saved:** model, AdamW state, scheduler, Python/NumPy/torch CPU/CUDA RNG, data cursor
  (document index, offset, batch count, epoch), update/useful/total token counters, cumulative
  active training time, best-eval value, full effective config, and run contract with its SHA-256.
- **Manifest also records:** code file hashes and git revision, dependency versions, container
  image, launch command, numerics, dataset/tokenizer/eval identities, lineage (parent, corrective
  continuation) and save reason.
- **Mixed precision and accumulation:** BF16 autocast needs no loss scaler (recorded explicitly).
  Saves happen only at optimizer boundaries, so no accumulated gradient needs preserving.

Publication protocol: rename can't be trusted on the global volume, so publication doesn't use it.

1. Each checkpoint gets a fresh `ckpt-<step>-<uid>` directory; nothing is ever overwritten.
2. Payload is serialized to local staging, then hashed.
3. Payload is copied to the primary and re-hashed there.
4. `manifest.json` (self-hashed) is written.
5. `COMPLETE` (holding the manifest hash) is written last.

Readers accept a checkpoint only if the marker, the manifest self-hash, and every file's size and
hash verify. Auto-resume picks the newest compatible verified checkpoint across the primary and
fallback roots. It refuses to start fresh over invalid or incompatible history, and a corrective
continuation requires a new run id and is recorded in lineage. After two failed save attempts
training stops (exit 3) at that boundary.

Cost on the global volume, over 29 full saves of 1.78 GB: median 29.7 s (27.7–68.3), of which
copy plus verify is 20.9 s. The same save to container disk took 6.6 s. Restore takes a median
7.5 s (max 16.7). Overhead at the production cadence therefore depends on tokens per update:

| Tokens per update | 500 updates at 115K tokens/s | Trigger that fires first | Save overhead |
|---|---|---|---|
| 32,768 (mb16, no accumulation) | 142 s | 500 updates | ~21% (needs async publish or shorter saves) |
| 524,288 (e.g. mb16 × accum 16) | 38 min | 15 minutes | ~3.3% |

All fault tests ran on the full model, compiled, in strict deterministic mode, against the
global volume (`results/recovery/results.json`):

| Test | Expected | Observed | Result |
|---|---|---|---|
| Save, stop, fresh-process exact continuation | Steps 16–40 identical to the uninterrupted run | 0 mismatches in loss bits, batch fingerprints, LR and token counters; final parameter and optimizer hashes equal; restore 6.7 s | PASS |
| Abrupt termination (SIGKILL at step 25) | Resume from 20 and replay with identical batches | Resumed at 20; 21–40 identical; final state equal | PASS |
| Interrupted serialization (`os._exit` mid `model.pt`) | Partial directory rejected; fall back to 10 | Rejected ("manifest missing"); resumed from 10; final state equal | PASS |
| Interrupted publication (before `COMPLETE`) | Rejected | Rejected ("completion marker missing"); final state equal | PASS |
| Corrupted file (one bit flipped in `model.pt`) | Rejected with reason | "sha256 mismatch model.pt"; resumed from 10; matched reference | PASS |
| Missing file (`optim.pt` deleted) | Rejected with reason | "missing file optim.pt"; matched reference | PASS |
| Storage exhaustion (injected ENOSPC) | Two attempts, stop at boundary, 10 remains latest | Exit 3 after 2 attempts; trained to 20 only; latest-valid 10 | PASS |
| Incompatible contract (LR changed) | Exact resume refused; corrective continuation recorded | Refused (exit 1); corrective run under new id records parent and both contract hashes | PASS |

Replica tests (FLOWBOX archive, `results/replication/`):

| Test | Observed | Result |
|---|---|---|
| Pull and verify | 4 t1 checkpoints VERIFIED, 12.6–13.0 MB/s; receipts; hash-chained audit log intact | PASS |
| Interrupted transfer (replicator killed mid `optim.pt`) | Partial copy fails verification on size; no final directory; no receipt; archive latest-valid unchanged | PASS |
| Resume of interrupted transfer | Partial reused; VERIFIED | PASS |
| Archive unavailable (unmounted path) | Refused, exit 1; nothing created | PASS |
| Source unreachable | Exit 1; audit entry SOURCE_UNREACHABLE (added after the first run) | PASS |
| Idempotent re-run | ALREADY_ARCHIVED_VERIFIED; no overwrite | PASS |
| Replica missing / primary missing (CPU harness) | Missing fallback ignored; with primary gone, resumed from replica | PASS |
| **Replacement-pod recovery** | New pod `qp11wt2e84lnyx` (US-WA-1; another physical A100, driver 595.91.07), code at the checkpoint's revision `28c0bfc`, FLOWBOX replica uploaded in 706 s and restored in 2.8 s; steps 16–40 bitwise identical to the original-pod reference, including final parameter and optimizer hashes | PASS |

The test exposed a defect, fixed and covered by regression: rsync transfers files alphabetically,
so `COMPLETE` and the manifest arrived before the payload. Verification rejected the partial copy
anyway, but the replicator now sends payload first and markers last.

Retention (`retention_plan`) implements the v0.3 defaults:

- initialization, the latest three, and every stage-boundary, best, release-candidate,
  published-result, pinned, incident and planned-shutdown checkpoint;
- hourly over the last 24 *active-training* hours, and daily for the whole run;
- invalid checkpoints kept as quarantined evidence, never latest-valid or best.

It is unit-tested with fixture timestamps. Pruning is not automated.

Not verified:

- A replacement pod reading the **global volume**, because the API cannot attach it.
- Asynchronous publication.
- Retention behaviour over real elapsed time.
- Planned-shutdown checkpoints triggered by a RunPod stop (SIGTERM handler implemented, not
  exercised).
- The v0.4 independent cloud backup and the backup of the FLOWBOX archive.

## E. Autonomous lifecycle

| Step | Operation | Independent verification | Result |
|---|---|---|---|
| Discovery | `list-pods` / `get-pod` | One pod, owner-created; GPU, ports, image, env | PASS |
| Remote execution | SSH from FLOWBOX | All benchmarks ran remotely; agent off-pod | PASS |
| Stop 1 | `pod-action stop` at 17:51:5x | `EXITED`, `runtime: null`, log "stop container / remove container" 17:51:53, SSH timeout | PASS |
| Start | `pod-action start` at 18:09:46 | Container ready 18:09:51; same GPU UUID, bus and driver; new SSH port 18079 and new host key; same global-volume mount | PASS |
| Post-restart | Volume marker and 266 evidence hashes; restore t1 step 35 from the global volume plus one update | Volume data intact; container-disk files gone as expected; step 36 bitwise equal to the reference | PASS |
| Stop 2 (final) | `pod-action stop` at ~18:15:05 | API said `EXITED` immediately, but SSH still answered until 18:15:21; container stopped 18:15:25; SSH dead from 18:15:31 | PASS |

**The stop response is not proof of a completed stop.** On the final stop the API reported
`EXITED` about 20 s before the container stopped, and `get-pod` kept a stale `runtime` block.
Verification must use the system log and SSH, as done here.

Final state: **VERIFIED STOPPED** (EXITED, container removed). Manual shutdown required: **NO**.
The replacement pod was terminated (it was created by this run) at about 18:09:20 and no longer
appears in `list-pods`.

## Cost

| Item | Basis | Amount |
|---|---|---|
| Owner pod GPU | 16:22:22–17:51:53 plus 18:09:46–18:15:25 = 95.2 min at $1.79/h | $2.84 (estimate) |
| Owner pod container disk | 1.59 h at ~$0.139/h (screenshot rate) | $0.22 (estimate) |
| Replacement pod | 17:53:05–~18:09:20 = 16.3 min at $1.79/h, plus 100 GB disk | $0.49 (estimate) |
| **Total compute** | | **≈ $3.55 (estimate)**, within the $5 target and $8 cap |
| Provider-confirmed so far | `list-billing` day bucket at report time (lagging; replacement pod not yet posted) | $1.86 |
| Ongoing | Global volume, 9.5 GB after approved cleanup, at $0.09/GB-month plus request charges | ≈ $0.85/month (estimate) |

Approved cleanup deleted 49.1 GB of recomputable fixture checkpoints and storage-test files from
the global volume, with logs in `results/lifecycle/global_volume_cleanup.txt` and on the volume.
Kept: `j1ckpt/t1`, `j1code`, `qual-evidence`.

## Remaining blockers before canonical training

1. **Data, tokenizer, architecture.** Juniper-1-Data PR #1 is CHANGES REQUIRED, with four owner
   source-use decisions pending. No Juniper 1 tokenizer exists. The architecture is not frozen
   (vocabulary, QK-Norm form, RoPE θ, weight-decay exclusions, cross-document attention policy).
2. **Durable storage topology (v0.3/v0.4).**
   - The global volume is beta. RunPod advises against it for checkpoint-heavy writes, and it
     can't be attached by API, so automated replacement-pod recovery has to come from FLOWBOX
     (12 min upload per 1.78 GB checkpoint at 2.4–2.5 MB/s).
   - Network volumes aren't offered where Secure A100-SXM capacity exists.
   - The owner must choose a second independent cloud destination (for example S3-compatible
     object storage) and an independent backup for the FLOWBOX archive.
3. **Checkpoint overhead.** A 30 s synchronous publish to the global volume is 21% overhead if
   500 updates take only 142 s. Before launch, either use a large global batch, or implement and
   qualify asynchronous publication from an immutable snapshot (v0.3 allows it with that
   condition) and drop the redundant pre-marker re-hash (~6 s).
4. **v0.5 optimization protocol.** The optimized profile meets the throughput and synthetic
   learning-equivalence evidence. Held-out domain quality pilots need real tokenized data and
   owner-approved thresholds, so the profile stays **experimental**. The candidates are named
   above: reference = eager strict-deterministic, optimized = `torch.compile` default.
5. **Operational.**
   - Move the inductor cache to persistent storage, or accept about 65 s per restart.
   - Read the SSH port from the API after each start.
   - Pin `OMP_NUM_THREADS ≤ 13` because of the cgroup CPU quota.
   - The account's single SSH key was set by this run; rotate if desired.
