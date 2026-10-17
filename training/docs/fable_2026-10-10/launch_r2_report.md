# j1-base-r2 — canary and limited tranche report (owner order 2026-10-10, §11)

Executor: Claude Fable 5.1 (goal mode). Run: `j1-base-r2`, stage `base-tranche-1`, ~150M text-only model
(`j1-wide-20x768`, 142,771,456 params) trained from scratch on j1-base-v0.2.2 (3,410,114,054 stream tokens;
limit 3,410,100,225; 52,034-update horizon; one exact epoch), tokenizer j1tok-24k, one A100 SXM 80GB (pod
`pncbyxaf2dt2jl`, GPU `GPU-889ddbcd-118e-b2d0-0771-80b3c1a10451`), eager profile (compiled profile FAILED the
registered equivalence rule; see `compile_equivalence_v2.json`). Deployed code `f0b4dea81fb6003ae15ace0f9dbe3f5ecf780d10`.

## 1. Authorization chain (all bound by hash)

| Item | Value |
| --- | --- |
| Launch manifest | `launch_manifest.json` sha256 `b9c1257af92128ef65a0c917928796e0d638fcd94b3bc3cd905c57ff7e942731` |
| Readiness matrix (authorizing run 12:16Z) | sha256 `90c6246973ba82a1b5c30c05185724cae835f36d65a7ac082eb85900e1f9f391` (ready; all gates PASS; committed 2176f06) |
| Gate definitions | `readiness_gates_v2.json` sha256 `1459155b95b0f9b0bf60d30cba10994dcc34f6223526ee7c657d2c3db5dde45b` |
| Funding contract | v6 (FC-6b: LAUNCH admitted only with readiness ready), sha256 `3834c3d6af67c425be91fab7fb16515c9d56fd1440089363a04d56f84c6f51da` |
| Ledger job | `launch-tranche-r2b` (LAUNCH allocation, cap 3.40, 1.75 h, deadline 2026-10-11T14:02:06Z) |
| LAUNCH_AUTHORIZED | manifest · matrix · source head `2e666f6572c278137c84653116fb80f7163148ae` · expires 2026-10-11T14:02:06Z · gates sha (5 fields) |
| Supervision | on-pod controller armed 12:14:24Z (soft 3.60 / hard 4.10 absolute within the 11:57:49Z session, deadline 14:02:06Z, stall 900 s); off-pod watchdog (CPU pod `v247r25kcigu04`) session `j1-base-r2` from 12:15:40Z with the same limits |
| Preconditions | OK at 12:16:27Z (manifest, matrix, code, GPU, pod, expiry all matched) |

Attempt history (recorded, nothing trained): 12:12Z refused (pod restarted on a different A100 than the qualified
host; host record redone, manifest rebuilt, re-admitted as `launch-tranche-r2b`); 12:15:17Z refused
"supervision not armed" because the watchdog had no session event for `j1-base-r2` after the re-arm (the
re-arm script rewrites the watchdog config but does not restart it); the watchdog was restarted, a fresh
authorization issued, and the runner restarted. Both refusals are the readiness chain failing closed as designed.

## 2. Canary (50 updates) — ACCEPTED

Window 12:17:06Z–12:18:37Z (first trainer process after the authorization; warm page cache, data verify+load
1.76 s for 6.82 GB; model init 1.83 s). Evidence: HDD archive `evidence/launch_session/canary/` (canary.json,
canary.jsonl, canary_acceptance.json, supervision_pre/post_canary.json, preconditions.json, replica_status.json).

| Metric | Value |
| --- | --- |
| Updates / useful targets | 50 / 3,276,800 (65,536 per update, 16×2048 micro × accum 2) |
| Loss update 1 → 50 | 10.2563 → 7.3756 (hex `0x1.d809720000000p+2`), no skipped updates, grad-norm 0.665 at 50 |
| LR at 50 | 1.6971e-4 (= 50/500 of peak 0.0016971, warmup as scheduled) |
| Full-split evaluation at 50 | loss 7.3718, **2.7801 bpb** (2,576 docs, 12,311,466 scored bytes) |
| Throughput | 65,514 tok/s median (reference 65,316 × 0.9 allowance), step 1000 ms median / 1001 p90, first step 1.36 s |
| Peak VRAM / host RSS | 44.2 GiB / recorded in canary.json |
| Checkpoints | `ckpt-000000000-ac36adac` (init) and `ckpt-000000050-536f263d` (eval, best), manifest-last, verified on the volume and replicated+verified on the HDD before acceptance |
| Acceptance C1–C9 | all true at 12:21:15Z (exit/steps, finite and learning, useful targets, throughput, init+canary checkpoints, evaluation, replica, supervision alive before and after, identities GPU/code) |

## 3. Limited tranche (to 2,000 updates)

Started 12:21:58Z: resume from the exact canary checkpoint `ckpt-000000050-536f263d` (same schedule and data
order), guard loop armed (instability rule after update 1,000 at 2 % rel., replica lag limit 1,000 updates /
600 s, volume cap 45 GB). Trainer finished at update 2,000 at 12:56:33Z with rc=0. Disposition:
**LIMITED_TRANCHE_COMPLETE**. Evidence: HDD archive `evidence/launch_session/tranche/` (tranche.json, tranche.jsonl,
launch-guard.jsonl, guard.out, launch.log, supervision_pre_tranche.json, tranche.host.txt, tranche.gpu.csv),
`evidence/launch_session/ctl/`, `evidence/launch_session/wd/`, `evidence/launch_session/timelogs/j1-base-r2/`.

| Metric | Value |
| --- | --- |
| Updates in the tranche / total | 1,950 (51–2,000) / **2,000 of 52,034** |
| Valid targets | 131,072,000 total (= 2,000 × 65,536; 127,795,200 in the resumed segment), useful == total (no padding) |
| Skipped / non-finite updates | 0 / 0 |
| Restart (resume) cost | process entry → first update 12.8 s (restore 3.6 s, data verify+load 1.75 s, exact-epoch accounting 1.77 s) |
| Throughput | 65,519 tok/s median; step 1000.3 ms median, 1001.7 p90, 1003.0 p99, 1029.9 max |
| Peak VRAM / host RSS | 44.26 GiB / 7.98 GiB |
| LR schedule | 1.731e-4 at 51 → peak 1.6971e-3 at 500 (warmup) → 1.6939e-3 at 2,000 (cosine, horizon 52,034) |
| Training loss (per-batch) | 7.29 (51), 6.54 (100), 5.52 (250), 4.50 (500), 3.96 (750), 3.64 (1000), 3.47 (1250), 3.28 (1500), 3.18 (1750), 3.22 (2000); grad-norm median 0.22, max 6.15 (early) |
| Full-split evaluation (2,576 docs, 12,311,466 bytes) | 500: loss 4.3719 / **1.6487 bpb**; 1000: 3.6106 / **1.3616 bpb**; 1500: 3.3759 / **1.2731 bpb**; 2000: 3.2426 / **1.2228 bpb** |
| Cross-check at 1,000 | eval loss 3.6106 lies inside the registered eager spread of the equivalence check on the previous host (eager 3.6229, eager replicate 3.6174, compiled 3.6080) |
| Checkpoints | 500 `ckpt-000000500-c4be7e7f`, 1000 `ckpt-000001000-4d4593eb`, 1500 `ckpt-000001500-da8a5cb5`, 2000 `ckpt-000002000-ca0a2f88` (each ~9 s publish incl. copy-verify; manifest-last; all tagged eval+best) |
| Final hashes | params `4e363a7e51d70807ea23541dc49406b9f8daaf873d288390b444f5affcc98f9f`, optimizer `7d66fd895f4c6d7e34fa4394ae00d9a3c99a513e8e4ea20e68e39f720f0c3031` |
| Guard | 70 polls, no reason ever raised; `guard_end: trainer finished` 12:56:45Z |
| Replicas (HDD) | all 6 checkpoints (0, 50, 500, 1000, 1500, 2000) replicated and verified (`ckpt/j1-base-r2/receipts.jsonl`); final replica verified before the stop request |

## 4. Shutdown and money

- 12:59:04Z `launch_tranche.sh` wrote the stop request ("tranche complete at update 2000; final checkpoint
  replicated"); 12:59:24Z controller `soft_shutdown`; 12:59:55Z `stop_requested` → `stop_call_ok` (GraphQL);
  provider status **EXITED** confirmed at 13:00Z by the lead and by the off-pod watchdog (`observed_stopped`
  13:00:17Z, `stop_verified` 13:01:18Z). No stop by limit: soft 3.60 / hard 4.10 never approached.
- Controller session estimate (pod session from 11:57:49Z, rate 1.90/h): **2.0583** (3,900 billed s); the launch
  job's share = 2.0583 − 0.6017 (launch start basis) = **1.46**, closed in the ledger as `launch-tranche-r2b`
  actual 1.46 (cap 3.40). CPU watchdog pod `v247r25kcigu04` stopped 13:02Z (`support-cpu-pod-3` closed 0.08).
- Ledger after closure (contract v6): committed 6.64 of the 13.70 admissible envelope, remaining 7.06; the
  OPTIONAL_CORE_LR allocation (2.00) was not used. Provider-billed account spend 2026-10-08 → 13:02Z 2026-10-11:
  52.91 (pods 52.33, storage 0.58), i.e. ~4.1 since the 03:30Z reading of 48.83 — all inside the order's
  $150 cumulative ceiling (no deposits, no payment-method changes, no auto-pay).

## 5. Preservation — COMPLETE 14:19:09Z

`preserve_ckpts.py` (FLOWBOX systemd unit) copies each HDD-verified checkpoint to Drive
(`cinqic-drive:juniper1/checkpoints/j1-base-r2/`) and the private HF repo `CinqicX/juniper1-checkpoints`
(`checkpoints/j1-base-r2/`), with read-back and immutability checks, and records `offsite_receipts.jsonl` and
`preservation_status.json`. Result: `fully_preserved_steps` = [0, 50, 500, 1000, 1500, 2000], pending = [];
every Drive copy `rclone check` clean and every HF file read back at its commit; lags from HDD verification
(Drive / HF): 257/305 s, 837/1,568 s, 1,607/2,326 s, 2,423/3,132 s, 3,274/3,981 s, 4,100/4,813 s — all inside
the LP-3 4 h deadline. Each 1.63 GiB checkpoint took ~630 s to Drive and ~585 s to HF. Copies of the status and
receipt files are in the HDD archive `evidence/launch_session/tranche/`. The checkpoints also remain on the
network volume `/vol/ckpt/j1-base-r2` (both pods EXITED; the volume persists).

## 6. Disposition

- Canary: ACCEPTED (C1–C9). Limited tranche: **LIMITED_TRANCHE_COMPLETE** (2,000 updates, 131,072,000 valid
  targets, evaluated at 500/1000/1500/2000, all checkpoints replicated).
- Next: **READY_FOR_OWNER_CONTINUATION_DECISION** — the executor stops here per the order; see
  `continuation_package_r2.md`. The remaining 50,034 updates are not started.

## 7. Continuation to the full epoch (owner decision 2026-10-11 ~15:55Z) — COMPLETE (see §7b)

Authority: the owner's chat decision after the limited-tranche report (continue the full epoch in one session,
eager, every 10th checkpoint off-site) and the owner's autonomous-continuation directive of the same day;
funding contract v7 (balance 42.26 after a $30 reload, envelope 37.91; CONTINUATION cap 33.00); launch policy
LP-4; review ledger `findings/review_continuation_fddf206.md` (APPROVE_WITH_FIXES → re-check APPROVE at 51fb198);
review record `launch_review_v3.json` for the deployed code head 828edd1.

| Item | Value |
| --- | --- |
| Pod / GPU | replacement pod `dxnsn6g7n5s4fw` (the launch pod could not restart: no free GPU on its host), A100 SXM 80GB `GPU-5749bb50-4e2e-e39c-9bd7-d2f9abd5d98c`, same image, volume and data; host record `evidence/host_dxnsn6g7n5s4fw-fable/` |
| Deployed code | `828edd192fd97057c5126d6731191d2118638538` (tar `3c93db5c…`); j1bench tree unchanged since f0b4dea |
| Cold B17 | 13.27 s process entry → first update (STARTUP_QUALIFIED), `evidence/b17_startup_a100.json` |
| Manifest | `ef5b37aee80144cd2af70fe564072b0fcb43d0d40fd9c40e3da6a9a560197f0a`: continuation from update 2000 (`ckpt-000002000-ca0a2f88`, manifest sha `ae81100e…`, canary acceptance `79b546e9…` reused), stop 52034 (full horizon), retention every 5,000 + {0, 2000, 52034} |
| Limits | soft 30.32 / hard 33.32 absolute (launch start basis 0.3167 at 16:34Z), deadline 2026-10-12T10:05:16Z, stall 900 s; controller armed 16:36:30Z, watchdog session 16:36:40Z (CPU pod `v247r25kcigu04`) |
| Readiness / authorization | matrix `2f412631…` ready (gates `92c050fe…`), LAUNCH_AUTHORIZED 5 fields; ledger `continuation-epoch-r2` admitted (cap 33.00, 17.3 h) |
| Launch | preconditions OK 16:39:55Z; supervision ok; verify-continuation accepted the exact checkpoint (hash-verified; nothing newer on the volume) 16:40:36Z; trainer resumed 16:41:14Z (restore 3.7 s, data verify+load 7.5 s, first update 1.5 s); update 2017 at 16:41:49Z, loss 3.2377, LR 1.6938e-3, 1.0 s/update |
| Retention at start | HDD: non-keepers 50/500/1000 released by the replicator (1500 and 2000 kept as the newest two); volume: 18 GB used before the run |
| Expected | ~50,034 updates ≈ 13.9 h + evaluations/checkpoints ≈ 0.8 h → finish ≈ 07:30Z 2026-10-12, ≈ $28 |

Notes: the pod bootstrap's stale-attempt rule renamed `CANARY_ACCEPTED`; it was restored unchanged before the launch
(content bound by hash in the policy and manifest), and the rule is to be narrowed in a follow-up. The runner
appends to the existing `tranche.jsonl`, so the file holds the 51–2000 segment followed by a second
`startup_complete` record and the continuation segment. Results are appended when the run closes.

### 7a. Interruption and resume (2026-10-11 18:37Z → 2026-10-12 00:58Z)

FLOWBOX (the HDD replica host) lost power at about 18:27Z (the owner's household, not the run). Replication
stalled, and at 18:37:39Z the launch guard requested a stop for replica lag (primary 8500 vs replica 5000 for
> 600 s); the trainer wrote a planned-shutdown checkpoint at update 8585, the on-pod controller soft-stopped the pod
(stop_call_ok 18:38:40Z) and the off-pod watchdog verified EXITED (18:39:55Z). Evaluations before the stop:
7500 1.0806 → 8500 1.0708 bpb (loss 2.8393). Session cost 4.24 (job `continuation-epoch-r2` closed at 3.93).
After FLOWBOX came back (00:30Z): units restarted and replicas caught up (7500–8500); the pod restarted on a
different A100 (`GPU-c3966d38`; the first restart at 00:34Z was stopped by the watchdog because no new session had
been armed — correct behaviour); bootstrap, host record, cold B17 13.2 s on 828edd1; policy LP-5 (resume from the
eval checkpoint `ckpt-000008500-205ec566`, limits 26.0/29.0/15.2 h; the 85 updates after 8500 are re-trained; the
8585 checkpoint was archived to `/vol/ckpt/attempts/` as an operator decision because the resume point must be a
500-update boundary); manifest `db0921ce…` (soft 26.25 / hard 29.25 absolute, deadline 2026-10-12T16:05:30Z);
readiness ready; job `continuation-epoch-r2b` (cap 29.00) admitted; authorization from the docs-only branch
`fable/resume-r2-auth` (deployed code head 828edd1 unchanged); the contract file was kept at the deployed bytes
(`e6dfe275…`, the FC-7b deadline note lives in `funding_contract_v7_deadline_note.md`). Preconditions OK 00:57:24Z,
verify-continuation accepted 8500 at 00:58:06Z, trainer resumed 00:58:43Z (update 8552 at 00:59:47Z, 1.0 s/update).
Expected finish ≈ 13:50Z 2026-10-12.

### 7b. Epoch complete — EPOCH_COMPLETE (2026-10-12 13:49:27Z)

| Item | Value |
| --- | --- |
| Final update | **52,034 of 52,034** (3,410,100,224 valid targets, one exact epoch of j1-base-v0.2.2) |
| Resume segment | 8,502 → 52,034 (43,533 updates, 00:58:43Z–13:49:27Z), 0 skipped, 0 non-finite; 65,600 tok/s median, 999 ms/update, peak VRAM 44.26 GiB |
| Full-split evaluation (bpb) | 2,000: 1.2228 · 5,000: 1.1138 · 10,000: 1.0601 · 15,000: 1.0335 · 20,000: 1.0129 · 50,000: 0.9179 · **52,034: 0.9165** (loss 2.4301; best 2.4297 at 52,000) |
| Final checkpoint | `ckpt-000052034-336e5967` (params sha256 `1fd07c0d0c6c6b355460ab82ebe02530047b5682a4e6a41b9e4f0146323076e4`), tagged final; on the volume, the HDD, Drive and HF (read back) |
| Keepers preserved off-site | 0, 50, 500, 1000, 1500, 2000 (tranche), 5000, 10000, …, 50000, 52034 — `preservation_status.json` pending = [] |
| Shutdown | runner stop request 13:49:27Z → controller soft shutdown 13:49:44Z → provider stop 13:50:14Z → watchdog stop_verified 13:51:44Z; pod `dxnsn6g7n5s4fw` EXITED; CPU pod `v247r25kcigu04` stopped 14:31Z |
| Money (contract v7) | continuation-epoch-r2 3.93 (interrupted segment) + continuation-epoch-r2b 24.61 + preparation 0.67 + CPU pod 1.35 = **30.56 of the 37.91 envelope** (remaining 7.35); nothing exceeded a cap. Provider billing 2026-10-11 15:52Z → 2026-10-12 14:30Z: ≈30.46 (day buckets 13.81 + 24.18 minus 7.53 before the snapshot), so the balance should read ≈11.80; order-cumulative provider spend since 2026-10-08 ≈ 83.4, inside the $150 ceiling |
| Evidence | HDD archive `evidence/continuation_session/` (tranche records, guard log, both ledgers, time logs, pod environment, preservation receipts) |

Disposition: the base pretraining epoch of j1-base-r2 is complete and preserved. The run never required a
by-hand intervention on the pod; both interruptions (the planned stop at 2,000 and the FLOWBOX power loss) were
handled by the registered safety chain. Next owner decisions: post-training (T02), release evaluation, and
retention of the volume copies (the volume `72al92fys2` keeps the keepers; ~$0.03/day).

The 5,000, 7,500 and 8,500 bpb figures above were corrected on 2026-10-12 against the preserved raw
`evidence/continuation_session/tranche/tranche.jsonl` `eval_detail` records in the FLOWBOX archive.
For each step, bpb is `nll_sum / (ln(2) × scored_bytes)` with `scored_bytes = 12,311,466`:
5,000 = 1.1138314352861858, 7,500 = 1.080638972773633, and 8,500 = 1.0707591482145966.
The original training log and checkpoint artifacts were not changed.
