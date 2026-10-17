# A100 pilot session runbook (ready to execute once the owner restores RunPod authorization)

**Authorization state (2026-10-08 ~22:00Z):** the owner suspended RunPod use. The session may
start only after the owner explicitly restores permission, and it is capped at **$10
incremental** over the billing baseline at that time. The owner prefers the A100 over FLOWBOX
for all heavy *GPU* work. CPU-bound data preparation stays on FLOWBOX (spec §6: do not pay for
GPU time while doing CPU work).

## 0. After the FLOWBOX restart (before anything paid)

```bash
mountpoint -q "/media/cinqic/Cinqic Storage" || udisksctl mount -b /dev/sda1
nvidia-smi                                   # should work again after the restart (NVML mismatch)
tail -3 "/media/cinqic/Cinqic Storage/juniper1-data/data/logs/v0.2-stages.log"
# If the corpus rebuild has not logged "done finalize rc=0":
"/media/cinqic/Cinqic Storage/juniper1-data/rebuild_v02_resume.sh"
```

A new Claude session should read the memory file `juniper1-pretraining-prep.md`, this runbook,
and `docs/prep/00_investigation_report.md` first.

## 1. Preconditions (all on FLOWBOX, free)

| # | Item | Command / evidence |
|---|---|---|
| 1 | `j1-base-v0.2` finalized and verified (gates, verify, full contamination audit) | Juniper-1-Data `verify`, `gates`, `audit_scan` |
| 2 | Tokenizer candidates trained and evaluated (24K/28K/32K) | `python -m j1data tokenizer-candidates` |
| 3 | Full token shards for each candidate (corpus recount per candidate) | `python -m j1data token-shards --tokenizer <cand>` (the candidate is temporarily frozen as `cand-<name>`; see §5) |
| 4 | Equal-bytes prefixes (≈ 700 MB text each) plus validation | `python -m j1data token-shards-prefix --max-bytes 700000000` |
| 5 | Transfer bundle staged with a manifest: prefixes, `tokenizer.json` and `eval.json` per candidate | `python -m j1bench.store manifest …` (bundle ≈ 1.1 GB) |
| 6 | Code: Juniper-1-Train `opus/pretraining-prep` clean and pushed; tests pass on FLOWBOX | `pytest`, `ruff`, `mypy` |
| 7 | Rehearsal of `tools/prep_pilots.sh` passed | done 2026-10-08 (synthetic data) |

### Status of preconditions (2026-10-09 03:15Z)

| # | Status |
|---|---|
| 1 | DONE: `j1-base-v0.2` built (00:30Z); verify clean; full audit 0 hits; gates 19 PASS / 3 NOT VERIFIED / 1 N/A (`Juniper-1-Data` data card) |
| 2 | DONE: candidates v3 (STEM alphabet). All pass rule v1.1 hard requirements |
| 3 | DONE: full shards for all three candidates, `--verify-only` clean (24K 3.436B, 28K 3.384B, 32K 3.344B train tokens) |
| 4 | DONE: 700 MB equal-bytes prefixes (156,588 documents; 178–183M tokens) |
| 5 | DONE: bundle `juniper1-a100/bundle-2026-10-09` (1.116 GB, 21 files, manifest `e937ada6416e3eb3341dc1c6fd587afa08a025c9619b47cc4c73629c3a560dc3`) |
| 6 | DONE: tests green and branch pushed |
| 7 | DONE twice: synthetic rehearsal, and a **real-bundle rehearsal on FLOWBOX** (fp16 and expanded GQA overrides, 6-update caps). The real-bundle rehearsal ran end to end, the time logs audited clean, and the report tool ran. It found and fixed three bugs: overrides were not last, the eval interval could round to 0, and there was no final-step evaluation |

**Constraint:** `bench_matrix.sh` word-splits run arguments, so `DATA` and `REPO` must contain no
spaces. On the pod they are `/root/data` and `/root/j1train`.

## 2. Session (agent-operated; every step logged under spec v0.6 §8)

```bash
T="/media/cinqic/Cinqic Storage/juniper1-archive/timelogs"; RUN=a100-pilots-<date>
python -m j1bench.timelog --log-root "$T" --run-id $RUN clock-check
```

1. **Baseline:** `list-billing` (RunPod tool). Record it with `timelog ledger billing_snapshot`.
   Cap = baseline + $10.00; stop at +$8.00.
2. **Start:** `pod-action start` on `4i13u1p447t6e7` (RunPod tool; this is the only paid
   action). Record `resource_request` and observed `resource_state` events. Read the SSH
   port from `get-pod` and update the `j1pod` entry in `~/.ssh/config`.
3. **Guard** (background on FLOWBOX, for the whole session):
   ```bash
   python tools/pod_guard.py --log-root "$T" --run-id $RUN --resource runpod:pod:4i13u1p447t6e7 \
     --usd-per-hour 1.93 --cap-usd 10 --reserve-usd 2 --prior-usd 0
   ```
   Exit status 10 means the reserve was reached and SIGTERM was sent; stop the pod immediately.
4. **Environment:**
   ```bash
   ssh j1pod 'python -m venv --system-site-packages /root/venv'
   tools/deploy.sh j1pod
   ssh j1pod '/root/venv/bin/pip install -r /root/j1train/env/pod-overlay-requirements.txt'
   ```
   Then run `pytest` (tests) and `qual/timelog_qual.py` on the pod (A100 logger
   qualification).
5. **Data:** `scp -r bundle-2026-10-09/data` to `/root/data` (plus `manifest.json`) (≈ 7–8 min at 2.4–2.5 MB/s), then
   `python -m j1bench.store verify` on the pod. The transfer runs in parallel with step 4.
6. **Pilots:**
   ```bash
   ssh j1pod 'cd /root/j1train && nohup env J1_TIME_LOG_ROOT=/root/timelogs bash tools/prep_pilots.sh > /root/runs/session.out 2>&1 &'
   ```
   Defaults are the A100 contract: micro-batch 16 × accum 2, BF16, compiled, native GQA.
   Monitor with `tail` plus the guard. Apply the cut order in `09_pilot_plan.md` if the
   projected cost exceeds the stop line.
7. **Pull results:** `rsync` `/root/runs`, `/root/timelogs`, and the final checkpoints of
   `fam-j1-core-28x640` and the selected small model into the FLOWBOX archive.
   `j1bench.replicate` produces receipts for the checkpoints.
8. **Stop:** `pod-action stop`. Verify with `get-pod` (EXITED, `runtime: null`), an SSH
   failure, and the system log; the API can report EXITED about 20 s early. Record the
   `resource_state` events.
9. **Billing:** re-read `list-billing` immediately and again after ≥ 1 h (billing lags).
   Record both snapshots and run the `timeaudit rollup` for the cost reconciliation.

## 3. After the session (FLOWBOX)

1. Apply the recorded tokenizer selection (`select_tokenizer.json`): `tokenizer-freeze` the
   chosen candidate with the rule output as the rationale.
2. Build the final shards for the frozen tokenizer, verify them, and push them to HF
   (`CinqicX/juniper1-shards`). Push the v0.2 accepted corpus (`CinqicX/juniper1-corpus`) and
   the tokenizer (`CinqicX/juniper1-tokenizer`).
3. `tools/scaling_report.py` on the pulled runs, then the cost-model refresh with the measured
   A100 throughput per shape.
4. GGUF export of the trained core checkpoint, parity (F16 gate, Q8/Q4 reported), and
   llama.cpp latency L1–L4 on FLOWBOX (the deployment target, so it runs on FLOWBOX by
   definition).
5. Readiness matrix, spec v0.7 delta update, handoff, PRs (not merged).

## 4. Estimated cost

The cost is ≈ $6.0 at $1.93/h (`09_pilot_plan.md`), with the hard stop at $8.00 incremental
and a $2 reserve for billing lag and residual charges.

## 5. Candidate freezes before selection

The token-shard builder accepts only frozen tokenizers, which makes the shard provenance
explicit. Each candidate is therefore frozen under a candidate name (`cand-j1tok-24k`, …)
with the rationale "pilot candidate, not selected". The canonical freeze is a separate,
later record (`j1tok-v1`) that points to the selected candidate's identical bytes.
