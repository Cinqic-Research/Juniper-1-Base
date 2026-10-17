# Operations and reproduction guide

Experimental qualification harness. Synthetic data only; every artifact it
produces is non-production (`"non_production": true` in each manifest).

## Hosts and paths

| Role | Location |
|---|---|
| Control host (agent, replicator, archive) | FLOWBOX |
| Source of truth for code | this repository; deployed as `git archive` of a clean commit |
| Pod working copy | `/root/j1train` (container disk, lost on stop) |
| Pod Python env | `/root/venv` (`--system-site-packages` over the image's torch 2.8.0+cu128; adds pytest 8.4.2) |
| Primary checkpoints | `/workspace/j1ckpt/<run_id>/ckpt-<step>-<uid>/` (RunPod global volume, geesefs) |
| Code bytes behind each code identity | `/workspace/j1code/<git sha>.tar` |
| FLOWBOX archive (replica) | `FLOWBOX_ARCHIVE_ROOT`, see `config/flowbox_archive.json` |

The pod's SSH port changes on every container restart. Read it from the RunPod
API (`get-pod` → `ssh.direct`) and update the `j1pod` entry in `~/.ssh/config`.

## Environment

```bash
ssh j1pod 'python -m venv --system-site-packages /root/venv && /root/venv/bin/pip install pytest==8.4.2'
tools/deploy.sh j1pod            # refuses a dirty tree; records REVISION
ssh j1pod 'cd /root/j1train && /root/venv/bin/python -m pytest -q tests -p no:cacheprovider'
```

The container exposes 128 host CPUs but a cgroup quota of 13.6 CPUs. Set
`OMP_NUM_THREADS=12`; without it, CPU-side PyTorch work oversubscribes and runs
an order of magnitude slower.

## Minimal training test

```bash
ssh j1pod 'cd /root/j1train && OMP_NUM_THREADS=12 /root/venv/bin/python -m j1bench.train \
  --steps 20 --micro-bs 8 --seq-len 2048 --warmup 5 --summary /root/runs/smoke.json'
```

## Benchmarks

```bash
ssh j1pod 'C="--steps 80 --warmup 10 --seq-len 2048 --compile --warmup-steps-excluded 10"; \
  /root/j1train/tools/bench_matrix.sh /root/runs/repeat c_mb16 "$C --micro-bs 16" c_mb32 "$C --micro-bs 32"'
rsync -a j1pod:/root/runs/ results/raw/
python3 tools/summarize_bench.py results/bench results/raw/screen results/raw/repeat results/raw/determinism
```

## Checkpoint, restore and failure injection

```bash
ssh j1pod 'cd /root/j1train && OMP_NUM_THREADS=12 PY=/root/venv/bin/python /root/venv/bin/python \
  tools/recovery_suite.py /root/runs/recovery /workspace/j1ckpt --seq-len 2048 --micro-bs 16 --compile --deterministic'
```

Resume a run from the newest verified compatible checkpoint (primary first,
then any `--fallback-root`):

```bash
python -m j1bench.train <same contract flags> --run-id <run> --ckpt-root /workspace/j1ckpt --resume auto
```

`--resume auto` refuses to start fresh if checkpoints exist but none is valid
and compatible. A changed contract needs `--corrective-continuation REASON`
and a new `--run-id`; the parent run and contract change are written to lineage.

Verify one checkpoint by hand:

```bash
python3 -c "from j1bench.ckpt import verify_checkpoint; print(verify_checkpoint('/workspace/j1ckpt/<run>/<ckpt>'))"
```

Fault injection uses `J1_FAULT`: `die_mid_file:<i>`, `die_after_files:<n>`,
`die_before_complete`, `enospc_after_bytes:<n>`.

## Replication to FLOWBOX

```bash
PYTHONPATH=. python3 -m j1bench.replicate --remote-root /workspace/j1ckpt --run-id <run> \
  --archive "/media/cinqic/Cinqic Storage/juniper1-archive"
```

Receipts land in `<archive>/receipts/<run>/`, and `<archive>/audit.log` is
hash-chained. A nonzero exit means at least one checkpoint did not verify.

Restore from the FLOWBOX replica into a pod:

```bash
rsync -a "<archive>/checkpoints/<run>/<ckpt>/" j1pod:/root/restore/<run>/<ckpt>/
ssh j1pod 'cd /root/j1train && /root/venv/bin/python -m j1bench.train <contract flags> --run-id <run> \
  --ckpt-root /workspace/j1ckpt --resume /root/restore/<run>/<ckpt>'
```

## Pod lifecycle (RunPod API via MCP, from FLOWBOX)

- Start: `pod-action {"action": "start"}`. The container is recreated: the
  container disk is empty, the SSH host key and public port change, and the
  inductor compile cache is gone (first compiled step ~65 s). The global volume
  is remounted at `/workspace`. Confirm the GPU UUID with `nvidia-smi` before
  trusting the pod.
- Stop: `pod-action {"action": "stop"}`. The response can say `EXITED` up to
  ~20 s before the container actually stops, and `get-pod` may keep a stale
  `runtime` block. Treat a stop as complete only when the system log shows
  `stop container` / `remove container` (`stream-pod-logs`, source `system`)
  and SSH no longer connects.
- Global volumes cannot be attached through the API; a replacement pod created
  by API restores from the FLOWBOX archive (upload ~2.4 MB/s from FLOWBOX).

## Safe shutdown

1. Pull `/root/runs` to FLOWBOX and copy it to `/workspace/qual-evidence/`;
   both are lost with the container disk otherwise. Reconcile with a sha256
   manifest.
2. Confirm the final checkpoint verifies on the primary and has a VERIFIED
   FLOWBOX receipt.
3. Stop through the RunPod API and verify as above. Never use `terminate` on a
   pod whose storage consequences have not been checked.
4. Cloud deletions need logged owner approval (spec v0.4 section 7); record
   each target, its file count and bytes, and reconcile afterwards.
