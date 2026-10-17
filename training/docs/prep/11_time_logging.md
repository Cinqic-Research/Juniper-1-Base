# Spec v0.6 §8 time logging — implementation and qualification record

Owner v0.6 arrived 2026-10-08 at 21:29Z, during this assignment. Work before that moment was
not instrumented. It is recorded as **backfill** (estimated from its own logs and receipts,
`status: "estimated"`) in the project log `juniper1-archive/timelogs/prep-2026-10-08/`, never
as measured.

## Components

| Component | File | Role |
|---|---|---|
| Event writer, spans, heartbeats, clock checks, command wrapper, ledger CLI | `j1bench/timelog.py` | producer side |
| Audit, supervisor repair, rollup, billing reconciliation | `j1bench/timeaudit.py` | supervisor side |
| Trainer instrumentation | `j1bench/train.py` (`run`, `_span`, segments) | run / init / compile / train-segment / eval / checkpoint / restore spans |
| Paid-session guard | `tools/pod_guard.py` | 60 s pod probes, ledger, idle alert, budget-reserve SIGTERM |
| Logger qualification | `qual/timelog_qual.py` | known-duration controls, event cost, audit |

## Frozen rules

- **Schema:** `j1.time-event.v1`. Required fields: `schema_version`, `event_id`, `project_id`,
  `run_id`, `producer_id`, `seq`, `event_type`, `timestamp_utc`, `host_id`, `clock_domain_id`,
  `monotonic`, `monotonic_unit`, `status`, `stage_id`, `attempt_id`, `span_id`,
  `parent_span_id`. A null field carries a reason in `null_reasons`. Terminal events also carry
  `elapsed_seconds`, `outcome`, `exit_code`, `failure_reason`, `counters`, `artifacts` and
  `evidence`.
- **Clocks:**
  - Timestamps are UTC RFC 3339 with milliseconds and `Z`.
  - Durations come only from `CLOCK_MONOTONIC` (resolution 1 ns reported, excludes suspend)
    within `host:boot_id:CLOCK_MONOTONIC`.
  - A cross-domain end is refused.
  - A wall/monotonic divergence above 2 s is recorded as `clock_anomaly`. Earlier timestamps
    are never rewritten.
  - The permitted absolute offset is 1,000 ms. Clock checks run before every wrapped command
    and trainer run, and hourly on the pod (guard).
- **Durability:** start, terminal and incident events are fsynced immediately. Routine events
  are written per line and fsynced within 10 s, and heartbeats are fsynced. A write failure
  raises `TimeLogError`. The trainer then stops at its next optimizer boundary and the
  wrapper SIGTERMs its child. Logging is never silently disabled.
- **Heartbeats:** at most 30 s in the trainer and wrapper (hard limit 60 s), carrying progress
  counters and the last progress time separately from liveness. The audit flags a gap above 3
  intervals.
- **Categories:** `initialization`, `compilation_warmup`, `cpu_data_processing`,
  `gpu_compute`, `data_io_wait`, `evaluation`, `checkpoint`, `replication_archival`,
  `restoration`, `transfer`, `manual_review`, `planned_idle`, `unexpected_interruption`,
  `supervision`. Leaf spans that declare `exclusive` form a resource timeline that must not
  overlap. Concurrent workers use separate resources (`cpu:wN`). Overlapping services
  (transfers) are non-exclusive and reported with their overlap.
- **Trainer data wait:** per-step data time is summed as the `data_wait_s` counter inside each
  `gpu_compute` train segment. One span per step would cost more than it explains.
- **Reconciliation tolerances (logger qualification):**
  - Span elapsed E must enclose the operation's own interval R, with
    `R ≤ E ≤ R + W_start + 5 ms`.
  - A wrapped `sleep 2` must report 2.0–2.5 s.
  - Cost estimates are bounded by the provider-state observation spacing and reconciled
    against provider `list-billing` snapshots, which lag.

## §8 acceptance cases → evidence

| Case | Test / evidence | Result on FLOWBOX |
|---|---|---|
| Successful job | `test_successful_and_failed_spans…`, `test_wrapper_success_failure_retry_and_concurrency` | PASS |
| Exception | same (failed outcome with reason) | PASS |
| Abrupt kill | `test_abrupt_kill_supervisor_appends_honest_interruption` (SIGKILL the process group; the supervisor appends `interrupted` with last-observed and detection times and only a lower-bound duration) | PASS |
| Retry | `retry` event linking the new attempt to its predecessor | PASS |
| Exact resume | `test_tokens_training_eval_and_exact_resume` (two run segments, both completed, audit clean; bitwise model equality) | PASS |
| Reboot / host change | `test_clock_domain_change_refuses_cross_domain_duration` | PASS |
| UTC clock jump | `test_wall_clock_jump_recorded_and_durations_unaffected` | PASS |
| Clock sync unavailable | `test_clock_check_unavailable_marks_ordering_uncertain` | PASS |
| Concurrent workers | 4 concurrent wrapped jobs on separate resource timelines; a shared exclusive timeline is flagged | PASS |
| Overlapping checkpoint/transfer | `test_overlap_accounting_and_unclassified_time` | PASS |
| Buffered-tail loss | `test_buffered_tail_loss_is_bounded_by_flush_interval` (events are written per line, so loss is bounded by the OS page cache and the ≤10 s fsync cadence) | PASS (bounded; real power loss not simulated) |
| Disk full | `test_disk_full_raises_instead_of_silently_dropping` (injected ENOSPC) | PASS |
| Failed replication | store push failure audited (real Xet network error, 2026-10-08 20:56Z); wrapped pushes record `failed` | PASS (store audit log); wrapper path PASS |
| Missing / duplicate events, tampering | `test_missing_duplicate_and_tampered_events_detected` | PASS |
| Resource-stop delay | `test_ledger_bounds_and_stop_delay` (requested stop ≠ observed stop; cost bounds) | PASS |
| Budget-triggered safe shutdown | `test_budget_reserve_triggers_safe_shutdown` (SIGTERM delivered to `python -m j1bench.train`, exit 10 for the operator stop) | PASS (local); A100: see pilot session |
| Known-duration controls / overhead | `results/prep/timelog/flowbox_hdd.json`, `flowbox_nvme.json` | PASS. Durable event: 44 ms median on the contended HDD, 0.27 ms on NVMe |

## Deviations and limits (explicit)

1. **Provider polling.** §8 asks for polling paid resources at least every 60 s. FLOWBOX holds
   no RunPod credentials, so the guard polls the **pod itself** over SSH every 60 s.
   Provider-API states (`get-pod`) and billing (`list-billing`) are read by the operator's RunPod
   tool calls at every transition and periodically, and are appended with
   `timelog ledger`. The provider-API cadence is therefore below 60 s only at transitions.
   Billed idle time is caught by the SSH probe (GPU utilisation and training-process count).
2. **Rebuild of `j1-base-v0.2`.** It started 19:43Z, before v0.6, so its stages are not wrapped.
   Its durations come from `/usr/bin/time -v` and the stage log, and are recorded as backfill.
3. **Authenticity.** Sealed segment hashes, an independent copy on FLOWBOX and pushes to the
   private HF repo give tamper evidence, not proof of authenticity.
