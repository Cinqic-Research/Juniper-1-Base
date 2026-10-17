"""Training loop: BF16 autocast, AdamW, gradient accumulation, full-state
checkpointing on the step/time cadence, exact resume.

EXPERIMENTAL. Synthetic data, fixtures, or real ``j1.token-shards.v1`` pilot data;
every artifact is non-production until a canonical run contract is frozen.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import platform
import random
import signal
import sys
import threading
import time
import uuid
from contextlib import nullcontext
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from . import ckpt as C
from .data import (
    MemmapPacked,
    SyntheticPacked,
    TokenStream,
    batches_fingerprint,
    consumed_prefix_record,
    full_split_eval,
    token_byte_lengths,
)
from .model import FAMILY, ModelConfig, Transformer
from .timelog import Span, TimeLog, TimeLogError, record_clock_check

PRESETS = {
    "j1-28x640": {},
    **FAMILY,
    # Same code paths, small enough for CPU tests and fast failure injection.
    "tiny": {
        "n_layer": 2,
        "d_model": 64,
        "n_head": 4,
        "n_kv_head": 2,
        "head_dim": 16,
        "d_ff": 128,
        "vocab_size": 512,
        "max_seq_len": 256,
    },
}


_IMPORT_PERF = time.perf_counter()


def process_elapsed_s() -> float:
    """Wall seconds since this process started (Linux: /proc/self/stat start time against /proc/uptime), so the
    B17 startup measurement includes interpreter start and imports; elsewhere, seconds since this module loaded."""
    try:
        stat = Path("/proc/self/stat").read_text()
        start_ticks = int(stat[stat.rindex(")") + 2 :].split()[19])  # field 22, after the comm field
        uptime = float(Path("/proc/uptime").read_text().split()[0])
        return uptime - start_ticks / os.sysconf("SC_CLK_TCK")
    except (OSError, ValueError, IndexError):
        return time.perf_counter() - _IMPORT_PERF


def code_identity() -> dict:
    root = Path(__file__).resolve().parent
    files = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.glob("*.py"))}
    rev = (root.parent / "REVISION").read_text().strip() if (root.parent / "REVISION").exists() else None
    agg = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    return {"package_sha256": agg, "files": files, "git_revision": rev}


def deps_identity() -> dict:
    d = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
    }
    if torch.cuda.is_available():
        d["device"] = torch.cuda.get_device_name(0)
    return d


def tensor_tree_sha(sd) -> str:
    h = hashlib.sha256()

    def walk(o, key=""):
        if isinstance(o, torch.Tensor):
            h.update(key.encode())
            h.update(
                o.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes() if o.numel() else b""
            )
        elif isinstance(o, dict):
            for k in sorted(o, key=str):
                walk(o[k], f"{key}/{k}")
        elif isinstance(o, (list, tuple)):
            for i, v in enumerate(o):
                walk(v, f"{key}/{i}")
        else:
            h.update(f"{key}={o!r}".encode())

    walk(sd)
    return h.hexdigest()


def lr_lambda(warmup: int, total: int, min_ratio: float):
    def f(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        prog = min(1.0, (step - warmup) / max(1, total - warmup))
        return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * prog))

    return f


def build_parser() -> argparse.ArgumentParser:
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument("--preset", default="j1-28x640", choices=PRESETS)
    a.add_argument("--vocab", type=int, default=None)
    a.add_argument("--seq-len", type=int, default=2048)
    a.add_argument("--micro-bs", type=int, default=8)
    a.add_argument("--accum", type=int, default=1)
    a.add_argument("--steps", type=int, default=100, help="total optimizer updates for the schedule and run")
    a.add_argument("--stop-at", type=int, default=None, help="exit cleanly after this update (schedule unchanged)")
    a.add_argument("--lr", type=float, default=3e-4)
    a.add_argument("--min-lr-ratio", type=float, default=0.1)
    a.add_argument("--warmup", type=int, default=10)
    a.add_argument("--wd", type=float, default=0.1)
    a.add_argument("--clip", type=float, default=1.0)
    a.add_argument("--opt", default="fused", choices=["fused", "foreach"])
    a.add_argument(
        "--precision",
        default="bf16",
        choices=["bf16", "fp16", "fp32"],
        help="bf16 = A100 canonical candidate; fp16 (autocast + dynamic loss scaling) only for GPUs without BF16 "
        "tensor cores (FLOWBOX RTX 2060 pilots)",
    )
    a.add_argument("--compile", action=argparse.BooleanOptionalAction, default=False)
    a.add_argument(
        "--compile-mode", default="default", choices=["default", "reduce-overhead", "max-autotune-no-cudagraphs"]
    )
    a.add_argument(
        "--sdpa",
        default="default",
        choices=["default", "flash", "efficient", "cudnn", "math"],
        help="restrict scaled_dot_product_attention to one backend",
    )
    a.add_argument(
        "--data",
        default="synthetic",
        help="synthetic | synthetic-padded | synthetic-copy | synthetic-perm | memmap:PATH[:SHA256] | tokens:DIR",
    )
    a.add_argument("--limit-tokens", type=int, default=None, help="tokens: train on this stream prefix only")
    a.add_argument("--tokenizer", default=None, help="tokenizer.json matching the shards (bytes per token for bpb)")
    a.add_argument("--eval-windows", type=int, default=64, help="validation windows per evaluation (tokens: data)")
    a.add_argument(
        "--eval-set",
        default="windows",
        choices=["windows", "full-split"],
        help="windows: first --eval-windows token windows (legacy, differs across tokenizers); "
        "full-split: every document of the validation split, identical source bytes for every tokenizer",
    )
    a.add_argument(
        "--exact-epoch",
        action="store_true",
        help="tokens: truncate the prefix to a whole number of updates and require --steps to consume "
        "every window exactly once; the consumed targets and source bytes are recorded",
    )
    a.add_argument("--wd-policy", default="matrices", choices=["matrices", "all"])
    a.add_argument(
        "--time-log",
        default=os.environ.get("J1_TIME_LOG_ROOT"),
        help="spec v0.6 section 8 event log root (default: $J1_TIME_LOG_ROOT, else <ckpt-root>/../timelogs, "
        "else ~/j1-timelogs)",
    )
    a.add_argument("--stage-id", default="pilot", help="time-log stage id")
    a.add_argument(
        "--gqa-kernel",
        default="native",
        choices=["native", "expand"],
        help="expand = repeat K/V heads before SDPA (GPUs without fused GQA kernels, e.g. Turing)",
    )
    a.add_argument("--preload-data", action="store_true", help="read token files into RAM (no random disk I/O)")
    a.add_argument("--verify-data", action="store_true", help="re-hash every token file before training")
    a.add_argument(
        "--spike-factor",
        type=float,
        default=10.0,
        help="alert when grad norm exceeds this multiple of its running median",
    )
    a.add_argument("--seed", type=int, default=1234)
    a.add_argument("--deterministic", action="store_true")
    a.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a.add_argument("--run-id", default=None)
    a.add_argument("--ckpt-root", default=None, help="primary checkpoint root; omit to disable checkpointing")
    a.add_argument("--stage-dir", default="/tmp/j1stage")
    a.add_argument("--ckpt-every-steps", type=int, default=500)
    a.add_argument("--ckpt-every-seconds", type=float, default=900.0)
    a.add_argument("--no-init-ckpt", action="store_true")
    a.add_argument(
        "--ckpt-prune",
        default="none",
        choices=["none", "pilot"],
        help="pilot: after each save keep only latest, newest best and final checkpoints (investigation runs only)",
    )
    a.add_argument("--resume", default="none", help="none | auto | PATH to a checkpoint directory")
    a.add_argument("--fallback-root", action="append", default=[], help="extra run roots searched on resume (replicas)")
    a.add_argument(
        "--corrective-continuation",
        default=None,
        help="allow resuming across a changed run contract; the reason is recorded in lineage",
    )
    a.add_argument("--eval-every", type=int, default=0)
    a.add_argument("--metrics", default=None, help="JSONL output path")
    a.add_argument("--summary", default=None, help="JSON summary output path")
    a.add_argument("--warmup-steps-excluded", type=int, default=5)
    return a


class Trainer:
    def __init__(self, args):
        self.a = args
        # Held open for the whole run; every record is flushed immediately.
        self.log_f = Path(args.metrics).open("a") if args.metrics else None  # noqa: SIM115
        self.stop_requested = False
        # Startup instrumentation (B17): distinct phases with real counters from process entry to update 1.
        # ``process_elapsed_s`` is measured from the process start time, so it includes interpreter and
        # import time that no in-process timer can see.
        self.startup: dict = {
            "process_elapsed_at_trainer_init_s": round(process_elapsed_s(), 3),
            "phases": [],
            "counters": {"bytes_read": 0, "bytes_hashed": 0, "tokens_histogrammed": 0, "files_loaded": 0},
            "complete": False,
        }
        self._startup_t = time.perf_counter()
        self._startup_seq = 0
        self._startup_last_emit = 0.0
        self._startup_lock = threading.Lock()  # the loader pool reports from several threads
        self.tl: TimeLog | None = None
        self.startup_span: Span | None = None
        if args.run_id:
            # The run identity is known, so the time log opens before any expensive work and a "startup" span
            # with heartbeats covers hashing, loading, accounting and initialisation (spec v0.7 section 8).
            self.tl = TimeLog(self._time_log_root(), args.run_id, producer_name="train")
            self.startup_span = self.tl.span(
                "operation",
                "startup",
                stage_id=args.stage_id,
                attempt_id=f"{args.run_id}-startup-{uuid.uuid4().hex[:8]}",
                category="initialization",
                resource="host",
                parent_span_id_null_reason="startup precedes the run span",
            )
            self.startup_span.start_heartbeat(30.0, lambda: dict(self.startup["counters"]))
        try:
            self._initialise(args)
        except BaseException as e:
            # R-07: a failed startup (the B17 failure mode) must leave a terminal outcome on the startup span and a
            # closed time log, not an unterminated span whose heartbeat thread dies with the process.
            self._abort_startup(e)
            raise

    def _abort_startup(self, e: BaseException) -> None:
        if self.startup_span is not None:
            outcome = "cancelled" if isinstance(e, KeyboardInterrupt) else "failed"
            try:
                self.startup_span.progress(**self.startup["counters"])
                self.startup_span.end(
                    outcome,
                    failure_reason=f"{type(e).__name__}: {e}"[:500],
                    phases=self.startup["phases"],
                )
            except (TimeLogError, ValueError) as te:
                print(f"[startup] time log: could not end the startup span: {te}", file=sys.stderr)
            self.startup_span = None
        if self.tl is not None:
            try:
                self.tl.close()
            except (TimeLogError, OSError) as te:
                print(f"[startup] time log: could not close: {te}", file=sys.stderr)
        if self.log_f is not None:
            self.log_f.close()

    def _initialise(self, args):
        """Everything after the startup span opens: model, data, accounting, optimizer (see ``__init__``)."""
        dev = torch.device(args.device)
        self.dev = dev
        if args.deterministic:
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            # Strict: warn_only=True leaves the FlashAttention backward non-deterministic.
            torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        kw = dict(PRESETS[args.preset])
        if args.data.startswith("tokens:"):
            meta = json.loads((Path(args.data[7:]) / "shards.json").read_text())
            kw["vocab_size"] = meta["tokenizer"]["total_vocab"]
        if args.vocab:
            kw["vocab_size"] = args.vocab
        kw["max_seq_len"] = max(kw.get("max_seq_len", 2048), args.seq_len)
        self.cfg = ModelConfig(**kw)
        self._seed_all(args.seed)
        self.model = Transformer(self.cfg).to(dev)
        self.model.set_gqa_kernel(args.gqa_kernel)
        self.n_params = self.model.num_params()
        self._startup_phase("model-init", n_params=self.n_params, device=str(dev))
        self.data = self._make_data()
        if isinstance(self.data, TokenStream):
            self._startup_phase(
                "data-verify-load",
                files=len(self.data.files),
                tokens=int(self.data.starts[-1]),
                bytes_read=self.data.bytes_read,
                bytes_hashed=self.data.bytes_hashed,
                verify=bool(args.verify_data),
                preload=bool(args.preload_data),
            )
        self.consumed = None
        if args.exact_epoch:
            self._exact_epoch()
        self.opt = torch.optim.AdamW(
            self.model.param_groups(args.wd, args.wd_policy),
            lr=args.lr,
            betas=(0.9, 0.95),
            eps=1e-8,
            weight_decay=args.wd,
            fused=args.opt == "fused" and dev.type == "cuda",
            foreach=args.opt == "foreach" or dev.type != "cuda",
        )
        self.scaler = torch.amp.GradScaler("cuda") if args.precision == "fp16" else None
        self.skipped_updates, self._consecutive_skips = 0, 0
        self.sched = torch.optim.lr_scheduler.LambdaLR(self.opt, lr_lambda(args.warmup, args.steps, args.min_lr_ratio))
        self.fwd = torch.compile(self.model, mode=args.compile_mode) if args.compile else self.model
        self._startup_phase("optimizer-init", fused=bool(args.opt == "fused" and dev.type == "cuda"))
        self.step = 0
        self.tokens_useful = 0
        self.tokens_total = 0
        self.parent = None
        self.lineage_note = None
        self.contract = self._contract()
        self.contract_sha = hashlib.sha256(C.canonical(self.contract)).hexdigest()
        self.run_id = args.run_id or f"run-{self.contract_sha[:8]}"
        self.eval_batches = None
        self.seg: Span | nullcontext | None = None
        self.seg_counts: dict = {}
        self.last_eval_detail: dict | None = None
        self.best_eval = None
        self.last_eval = None
        self.active_total = 0.0

    # --- identities -------------------------------------------------------
    def _make_data(self):
        a = self.a
        if a.data in ("synthetic", "synthetic-padded"):
            return SyntheticPacked(
                self.cfg.vocab_size,
                a.seq_len,
                a.micro_bs,
                seed=a.seed,
                max_len=max(64, min(3000, 2 * a.seq_len)),
                pad_mode=a.data == "synthetic-padded",
            )
        if a.data in ("synthetic-copy", "synthetic-perm"):
            return SyntheticPacked(
                self.cfg.vocab_size,
                a.seq_len,
                a.micro_bs,
                seed=a.seed,
                min_len=64,
                max_len=512,
                kind=a.data.split("-")[1],
            )
        if a.data.startswith("tokens:"):
            return TokenStream(
                a.data[7:],
                "train",
                a.seq_len,
                a.micro_bs,
                seed=a.seed,
                limit_tokens=a.limit_tokens,
                verify=a.verify_data,
                preload=a.preload_data,
                progress=self._startup_progress,
            )
        if a.data.startswith("memmap:"):
            parts = a.data.split(":")
            return MemmapPacked(
                parts[1], self.cfg.vocab_size, a.seq_len, a.micro_bs, file_sha256=parts[2] if len(parts) > 2 else None
            )
        raise SystemExit(f"unknown --data {a.data}")

    def _exact_epoch(self) -> None:
        """Truncates the prefix to whole updates; one epoch then consumes every window once."""
        a = self.a
        if not isinstance(self.data, TokenStream):
            raise SystemExit("--exact-epoch needs --data tokens:")
        per_update = a.micro_bs * a.accum
        windows = (self.data.n_windows // per_update) * per_update
        if a.steps * per_update != windows:
            raise SystemExit(
                f"--exact-epoch: --steps {a.steps} x {per_update} windows != {windows} windows in the truncated prefix"
            )
        a.limit_tokens = windows * a.seq_len + 1
        # B17: move the logical bounds of the stream already built (and verified) instead of rebuilding it,
        # which hashed and preloaded every token file a second time.
        self.data.truncate(a.limit_tokens)
        assert self.data.n_windows == windows
        self._startup_phase("exact-epoch-truncate", windows=windows, limit_tokens=a.limit_tokens)
        self.consumed = consumed_prefix_record(self.data, self._token_bytes(), progress=self._startup_progress)
        self._startup_phase(
            "exact-epoch-accounting",
            target_tokens=self.consumed["target_tokens"],
            eos_targets=self.consumed["eos_targets"],
            target_bytes=self.consumed.get("target_bytes"),
            tokens_histogrammed=self.startup["counters"]["tokens_histogrammed"],
        )

    # --- startup instrumentation (B17) ---------------------------------------
    def _time_log_root(self) -> Path:
        a = self.a
        root = a.time_log or (str(Path(a.ckpt_root).parent / "timelogs") if a.ckpt_root else "~/j1-timelogs")
        return Path(root).expanduser()

    def _startup_progress(self, kind: str, **counters) -> None:
        """Progress callback for the loader and the accounting pass: real counters, emitted at most every 10 s
        (every file for the loader) so a stalled phase cannot keep resetting a supervisor's no-progress timer."""
        c = self.startup["counters"]
        if kind == "token-file":
            c["files_loaded"] += 1
            c["bytes_read"] += counters["bytes"] if (counters.get("loaded") or counters.get("hashed")) else 0
            c["bytes_hashed"] += counters["bytes"] if counters.get("hashed") else 0
            self._startup_emit("load-file", **counters, **c)
        elif kind == "histogram":
            c["tokens_histogrammed"] = max(c["tokens_histogrammed"], int(counters["tokens"]))
            now = time.perf_counter()
            if now - self._startup_last_emit >= 10.0:
                self._startup_emit("histogram", tokens=int(counters["tokens"]), of=int(counters["of"]))

    def _startup_emit(self, phase: str, **fields) -> None:
        with self._startup_lock:
            if self.startup["complete"]:
                return
            self._startup_seq += 1
            self._startup_last_emit = time.perf_counter()
            fields = {k: v for k, v in fields.items() if k not in ("event", "step", "phase")}  # reserved keys
            rec = {
                "event": "startup_phase",
                "step": self._startup_seq,
                "phase": phase,
                "since_trainer_init_s": round(time.perf_counter() - self._startup_t, 3),
                "process_elapsed_s": round(process_elapsed_s(), 3),
                **fields,
            }
            self.log(rec)
            print(f"[startup] {phase} +{rec['since_trainer_init_s']:.1f}s {json.dumps(fields)[:200]}", flush=True)
            if self.startup_span is not None:
                try:
                    self.startup_span.event("startup_progress", **fields, phase=phase)
                    self.startup_span.progress(**{k: v for k, v in fields.items() if isinstance(v, int | float)})
                except TimeLogError as e:
                    self._log_failure(e)

    def _startup_phase(self, name: str, **counters) -> None:
        """Ends one distinct startup phase: duration since the previous phase end plus its counters."""
        if self.startup["complete"]:
            return
        now = time.perf_counter()
        prev_end = self.startup["phases"][-1]["end_since_init_s"] if self.startup["phases"] else 0.0
        ph = {
            "phase": name,
            "seconds": round(max(0.0, now - self._startup_t - prev_end), 3),
            "end_since_init_s": round(now - self._startup_t, 3),
            **counters,
        }
        self.startup["phases"].append(ph)
        self._startup_emit(name, seconds=ph["seconds"], **counters)

    def _startup_complete(self, first_step_s: float) -> None:
        """Called once after optimizer/scheduler update 1: the B17 measurement record."""
        self._startup_phase("first-update", trainer_step=self.step, step_s=round(first_step_s, 3))
        rss_kib = 0
        with Path("/proc/self/status").open() as f:
            for line in f:
                if line.startswith("VmHWM"):
                    rss_kib = int(line.split()[1])
        self.startup.update(
            {
                "complete": True,
                "process_elapsed_s": round(process_elapsed_s(), 3),
                "trainer_init_to_first_update_s": round(time.perf_counter() - self._startup_t, 3),
                "peak_rss_gib": round(rss_kib / 2**20, 3),
                "process_entry_clock": "/proc/self/stat start time vs /proc/uptime (wall seconds since process start)",
            }
        )
        self.log({"event": "startup_complete", **{k: v for k, v in self.startup.items() if k != "complete"}})
        st = self.startup
        print(
            f"[startup] complete: process {st['process_elapsed_s']:.1f}s, "
            f"trainer {st['trainer_init_to_first_update_s']:.1f}s, peak RSS {st['peak_rss_gib']:.2f} GiB",
            flush=True,
        )
        if self.startup_span is not None:
            try:
                self.startup_span.progress(**self.startup["counters"])
                self.startup_span.end(
                    "completed",
                    phases=self.startup["phases"],
                    process_elapsed_s=self.startup["process_elapsed_s"],
                    peak_rss_gib=self.startup["peak_rss_gib"],
                )
            except TimeLogError as e:
                self._log_failure(e)
            self.startup_span = None
        elif getattr(self, "run_span", None) is not None:
            # No run identity at construction: the phases were measured in-process and are attached after the fact.
            self.run_span.event(
                "startup_summary",
                logged_after_the_fact=True,
                phases=self.startup["phases"],
                process_elapsed_s=self.startup["process_elapsed_s"],
                peak_rss_gib=self.startup["peak_rss_gib"],
            )

    def _contract(self) -> dict:
        a = self.a
        return {
            "model": self.cfg.to_dict(),
            "seq_len": a.seq_len,
            "micro_bs": a.micro_bs,
            "accum": a.accum,
            "optimizer": {
                "name": "AdamW",
                "impl": a.opt,
                "lr": a.lr,
                "betas": [0.9, 0.95],
                "eps": 1e-8,
                "weight_decay": a.wd,
                "clip": a.clip,
            },
            "schedule": {"kind": "warmup_cosine", "warmup": a.warmup, "total": a.steps, "min_ratio": a.min_lr_ratio},
            "precision": a.precision,
            "compile": a.compile,
            "compile_mode": a.compile_mode,
            "deterministic": a.deterministic,
            "sdpa": a.sdpa,
            "gqa_kernel": a.gqa_kernel,
            "attention_policy": (
                "plain causal over each packed row; cross-document attention allowed (not a frozen decision)"
            ),
            "weight_decay_policy": {
                "matrices": "2-D projection weights only; norms and the tied embedding excluded",
                "all": "applied to all parameters including norms and embeddings (qualification harness)",
            }[a.wd_policy],
            "data": self.data.identity(),
            "seed": a.seed,
            "tokenizer": (
                self.data.identity()["tokenizer_json_sha256"]
                if isinstance(self.data, TokenStream)
                else "none: synthetic token ids"
            ),
            "code": code_identity()["package_sha256"],
        }

    def _seed_all(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    def log(self, rec: dict):
        rec = {"t": time.time(), **rec}
        if self.log_f:
            self.log_f.write(json.dumps(rec) + "\n")
            self.log_f.flush()

    # --- checkpoint state -------------------------------------------------
    def rng_state(self) -> dict:
        s = {"python": random.getstate(), "numpy": np.random.get_state(), "torch_cpu": torch.get_rng_state()}
        if self.dev.type == "cuda":
            s["torch_cuda"] = torch.cuda.get_rng_state_all()
        return s

    def set_rng_state(self, s: dict):
        random.setstate(s["python"])
        np.random.set_state(s["numpy"])
        torch.set_rng_state(s["torch_cpu"])
        if self.dev.type == "cuda" and "torch_cuda" in s:
            torch.cuda.set_rng_state_all(s["torch_cuda"])

    def save(self, tags=()):
        a = self.a
        if not a.ckpt_root:
            return None
        trainer_state = {
            "step": self.step,
            "tokens_useful": self.tokens_useful,
            "tokens_total": self.tokens_total,
            "scheduler": self.sched.state_dict(),
            "lr": self.sched.get_last_lr()[0],
            "rng": self.rng_state(),
            "data": self.data.state_dict(),
            # Saves happen only at optimizer boundaries, so there is never a
            # partially accumulated gradient to preserve.
            "grad_accum_position": 0,
            # BF16 autocast uses no loss scaler; recorded explicitly.
            "amp": {
                "precision": a.precision,
                "grad_scaler": self.scaler.state_dict() if self.scaler is not None else None,
                "skipped_updates": self.skipped_updates,
            },
            "best_eval": self.best_eval,
            "active_train_s": self.active_total,
            # Authoritative timing lives in the event log; this records where this state sits in it.
            "timing": {
                "time_log_producer": self.tl.producer_id if getattr(self, "tl", None) else None,
                "time_log_seq": self.tl.seq if getattr(self, "tl", None) else None,
                "attempt_id": getattr(self, "attempt", None),
                "active_train_s_cumulative": self.active_total,
            },
            "config": vars(a) | {"model": self.cfg.to_dict()},
        }
        payload = {"model.pt": self.model.state_dict(), "optim.pt": self.opt.state_dict(), "trainer.pt": trainer_state}
        meta = {
            "run_id": self.run_id,
            "run_contract_sha256": self.contract_sha,
            "run_contract": self.contract,
            "tokens_useful": self.tokens_useful,
            "tokens_total": self.tokens_total,
            "code_identity": code_identity(),
            "deps": deps_identity(),
            "dataset_identity": self.data.identity(),
            "tokenizer_identity": self.contract["tokenizer"],
            "eval_identity": self._eval_identity() if a.eval_every else None,
            "lineage": {"parent": self.parent, "note": self.lineage_note},
            "tags": list(tags),
            "save_reason": ",".join(tags) or "cadence",
            "eval_loss": self.last_eval if tags and "eval" in tags else None,
            "stage": a.stage_id,
            "epoch": self.data.state_dict().get("epoch", 0),
            "active_train_s": self.active_total,
            "accounting": {
                "exposure_tokens_useful": self.tokens_useful,
                "exposure_tokens_total": self.tokens_total,
                "padding_tokens": self.tokens_total - self.tokens_useful,
                "padding_treatment": "labels=-100, excluded from loss and useful count",
                "supervised_target_tokens": None,
                "accepted_corpus_tokens": None,
            },
            "launch_command": sys.argv,
            "container_image": os.environ.get("J1_CONTAINER_IMAGE"),
            "numerics": {
                "precision": a.precision,
                "tf32": False,
                "deterministic": a.deterministic,
                "compile": a.compile,
                "sdpa": a.sdpa,
                "gqa_kernel": a.gqa_kernel,
            },
            "known_defects": [
                "non-production pilot/qualification harness"
                + ("; synthetic data" if not isinstance(self.data, TokenStream) else "; real j1-base-v0.2 token shards")
            ],
            "state_sha256": {
                "model": tensor_tree_sha(self.model.state_dict()),
                "optim": tensor_tree_sha(self.opt.state_dict()),
            },
            "non_production": True,
        }
        last_err = None
        for attempt in (1, 2):
            try:
                r = C.save_checkpoint(
                    Path(a.ckpt_root),
                    Path(a.stage_dir),
                    step=self.step,
                    payload=payload,
                    meta=meta,
                    torch_save=torch.save,
                )
                self.parent = r.manifest["name"]
                self.lineage_note = None
                self.log(
                    {
                        "event": "checkpoint",
                        "step": self.step,
                        "name": r.manifest["name"],
                        "tags": list(tags),
                        **{f"ckpt_{k}": v for k, v in r.timings.items()},
                    }
                )
                if a.ckpt_prune == "pilot":
                    pruned = C.prune_pilot(Path(r.path).parent)
                    if pruned:
                        self.log({"event": "pruned", "step": self.step, "deleted": pruned})
                mib, secs = r.timings["bytes"] / 2**20, r.timings["total_s"]
                print(
                    f"[ckpt] step {self.step} -> {r.path} ({mib:.1f} MiB, {secs:.2f}s)",
                    flush=True,
                )
                return r
            except (OSError, C.CheckpointError) as e:
                last_err = e
                self.log({"event": "checkpoint_failed", "step": self.step, "attempt": attempt, "error": repr(e)})
                print(f"[ckpt] FAILED attempt {attempt} at step {self.step}: {e!r}", flush=True)
        # Training without a durable checkpoint violates the cadence contract: stop.
        print("[ckpt] CHECKPOINT_FAILED: stopping; previous valid checkpoint remains latest", flush=True)
        self.log({"event": "abort", "reason": "checkpoint_failed", "error": repr(last_err)})
        sys.exit(3)

    def restore(self, path: Path):
        ok, why = C.verify_checkpoint(path)
        if not ok:
            raise C.CheckpointError(f"refusing to restore {path}: {why}")
        m = C.read_manifest(path)
        assert m is not None  # verify_checkpoint succeeded, so the manifest parses
        if m["run_contract_sha256"] != self.contract_sha:
            if not self.a.corrective_continuation:
                raise C.CheckpointError(
                    "run contract differs from checkpoint; exact resume impossible "
                    "(use --corrective-continuation REASON to continue under a new contract)"
                )
            if m["run_id"] == self.run_id:
                raise C.CheckpointError(
                    "a corrective continuation must use a new --run-id (new run/branch with parent linkage)"
                )
            self.lineage_note = {
                "corrective_continuation": self.a.corrective_continuation,
                "parent_run": m["run_id"],
                "from_contract": m["run_contract_sha256"],
                "to_contract": self.contract_sha,
            }
        t0 = time.perf_counter()
        ms = torch.load(path / "model.pt", map_location=self.dev, weights_only=True)
        os_ = torch.load(path / "optim.pt", map_location=self.dev, weights_only=True)
        # Restricted unpickler: tensors plus the numpy types that carry RNG state; no arbitrary code.
        # The manifest and payload hashes detect corruption, not a malicious replacement
        # of the entire checkpoint, so arbitrary pickle globals stay disabled on restore.
        np_globals = [
            importlib.import_module("numpy._core.multiarray")._reconstruct,
            np.ndarray,
            np.dtype,
            type(np.dtype(np.uint32)),
            type(np.dtype(np.float64)),
        ]
        with torch.serialization.safe_globals(np_globals):
            ts = torch.load(path / "trainer.pt", map_location="cpu", weights_only=True)
        self.model.load_state_dict(ms)
        self.opt.load_state_dict(os_)
        if tensor_tree_sha(self.model.state_dict()) != m["state_sha256"]["model"]:
            raise C.CheckpointError("restored model state hash differs from manifest")
        if tensor_tree_sha(self.opt.state_dict()) != m["state_sha256"]["optim"]:
            raise C.CheckpointError("restored optimizer state hash differs from manifest")
        self.sched.load_state_dict(ts["scheduler"])
        if self.lineage_note is not None:
            # Corrective continuation (spec v0.7 §6): optimizer moments and the scheduler position are carried
            # over explicitly; the schedule is re-based to this run's peak LR (the restored base_lrs belong to
            # the parent contract). Both carryovers are recorded in the lineage, never silently reset.
            old = list(self.sched.base_lrs)
            self.sched.base_lrs = [self.a.lr] * len(old)
            for g in self.opt.param_groups:
                g["initial_lr"] = self.a.lr
            for g, lr in zip(self.opt.param_groups, self.sched.get_lr(), strict=True):
                g["lr"] = lr
            self.sched._last_lr = [g["lr"] for g in self.opt.param_groups]
            self.lineage_note["scheduler_carryover"] = {
                "position": self.sched.last_epoch,
                "base_lr_from": old,
                "base_lr_to": list(self.sched.base_lrs),
                "lr_after_restore": self.sched.get_last_lr()[0],
            }
            self.lineage_note["optimizer_carryover"] = "AdamW moments and per-parameter step counts restored unchanged"
        if self.scaler is not None:
            self.scaler.load_state_dict(ts["amp"]["grad_scaler"])
            self.skipped_updates = ts["amp"].get("skipped_updates", 0)
        self.step, self.tokens_useful, self.tokens_total = ts["step"], ts["tokens_useful"], ts["tokens_total"]
        self.data.load_state_dict(ts["data"])
        self.set_rng_state(ts["rng"])
        self.best_eval = ts.get("best_eval")
        self.active_total = ts.get("active_train_s", 0.0)
        self.parent = m["name"]
        dt = time.perf_counter() - t0
        self.log(
            {
                "event": "restored",
                "from": str(path),
                "step": self.step,
                "restore_s": dt,
                "lr": self.sched.get_last_lr()[0],
                "corrective": self.lineage_note,
            }
        )
        print(
            f"[resume] restored {path} at step {self.step} in {dt:.2f}s (lr={self.sched.get_last_lr()[0]:.6g})",
            flush=True,
        )

    # --- training ---------------------------------------------------------
    def _autocast(self):
        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(self.a.precision)
        return torch.autocast(self.dev.type, dtype=dtype or torch.float32, enabled=dtype is not None)

    def _token_bytes(self) -> np.ndarray | None:
        if not self.a.tokenizer:
            return None
        tj = Path(self.a.tokenizer)
        if hashlib.sha256(tj.read_bytes()).hexdigest() != self.data.identity()["tokenizer_json_sha256"]:
            raise SystemExit("--tokenizer does not match the shards' tokenizer")
        # Control IDs have no text; byte-fallback tokens are one byte; learned tokens are literal strings.
        return token_byte_lengths(tj, self.cfg.vocab_size)

    def _ensure_full_split(self) -> None:
        if self.eval_batches is not None:
            return
        ev = TokenStream(self.a.data[7:], "validation", self.a.seq_len, 1, shuffle=False, verify=True)
        self.eval_batches = full_split_eval(ev)
        self.token_bytes = self._token_bytes()
        e = self.eval_batches
        ident = dict(e["identity"])
        ident["logits_precision"] = f"autocast {self.a.precision}; cross-entropy in float32; sums in float64"
        if self.token_bytes is not None:
            ident["scored_bytes"] = int(self.token_bytes[e["y"][e["y"] != -100]].sum())
        self.eval_identity = ident

    def evaluate_full_split(self) -> float:
        """Summed NLL over every validation-split target; bits per byte = NLL / (ln 2 * source bytes)."""
        self._ensure_full_split()
        e = self.eval_batches
        names = e["source_names"]
        n_src = len(names)
        loss_sum = np.zeros(n_src, dtype=np.float64)
        count = np.zeros(n_src, dtype=np.int64)
        nbytes = np.zeros(n_src, dtype=np.int64)
        self.model.eval()
        with torch.no_grad(), self._autocast():
            for start in range(0, len(e["x"]), self.a.micro_bs):
                x = torch.from_numpy(e["x"][start : start + self.a.micro_bs]).to(self.dev)
                y = e["y"][start : start + self.a.micro_bs]
                logits = self.model(x)
                per_tok = (
                    torch.nn.functional.cross_entropy(
                        logits.float().reshape(-1, logits.size(-1)),
                        torch.from_numpy(y).to(self.dev).reshape(-1),
                        reduction="none",
                        ignore_index=-100,
                    )
                    .double()
                    .cpu()
                    .numpy()
                )
                yv = y.reshape(-1)
                m = yv != -100
                src = e["src"][start : start + self.a.micro_bs].reshape(-1)[m]
                np.add.at(loss_sum, src, per_tok[m])
                np.add.at(count, src, 1)
                if self.token_bytes is not None:
                    np.add.at(nbytes, src, self.token_bytes[yv[m]])
        self.model.train()
        by_source = {
            name: {
                "tokens": int(count[i]),
                "nll_sum": float(loss_sum[i]),
                "loss": float(loss_sum[i] / count[i]),
                **({"bytes": int(nbytes[i]), "bpb": float(loss_sum[i] / math.log(2) / nbytes[i])} if nbytes[i] else {}),
            }
            for i, name in enumerate(names)
            if count[i]
        }
        overall = float(loss_sum.sum() / count.sum())
        rec = {
            "event": "eval_detail",
            "step": self.step,
            "loss": overall,
            "nll_sum": float(loss_sum.sum()),
            "scored_targets": int(count.sum()),
            "eval_set": self.eval_identity,
            "by_source": by_source,
        }
        if nbytes.sum():
            if int(nbytes.sum()) != self.eval_identity["scored_bytes"]:
                raise RuntimeError("scored bytes differ from the eval-set identity")
            rec["scored_bytes"] = int(nbytes.sum())
            rec["bpb"] = float(loss_sum.sum() / math.log(2) / nbytes.sum())
        self.log(rec)
        self.last_eval_detail = rec
        return overall

    def evaluate_stream(self) -> float:
        """Validation loss per token, plus per-source loss and bits per byte when the tokenizer is known."""
        if self.a.eval_set == "full-split":
            return self.evaluate_full_split()
        if self.eval_batches is None:
            ev = TokenStream(self.a.data[7:], "validation", self.a.seq_len, 1, shuffle=False)
            self.eval_batches = ev.eval_windows(self.a.eval_windows)
            self.eval_sources = ev.source_names
            self.token_bytes = self._token_bytes()
        n_src = len(self.eval_sources)
        loss_sum = np.zeros(n_src)
        count = np.zeros(n_src, dtype=np.int64)
        nbytes = np.zeros(n_src, dtype=np.int64)
        self.model.eval()
        with torch.no_grad(), self._autocast():
            for start in range(0, len(self.eval_batches), self.a.micro_bs):
                chunk = self.eval_batches[start : start + self.a.micro_bs]
                toks = torch.from_numpy(np.stack([c[0] for c in chunk])).to(self.dev)
                src = np.stack([c[1] for c in chunk]).ravel()
                logits = self.model(toks[:, :-1])
                per_tok = (
                    torch.nn.functional.cross_entropy(
                        logits.float().reshape(-1, logits.size(-1)), toks[:, 1:].reshape(-1), reduction="none"
                    )
                    .cpu()
                    .numpy()
                )
                np.add.at(loss_sum, src, per_tok)
                np.add.at(count, src, 1)
                if self.token_bytes is not None:
                    np.add.at(nbytes, src, self.token_bytes[toks[:, 1:].cpu().numpy().ravel()])
        self.model.train()
        by_source = {
            name: {
                "tokens": int(count[i]),
                "loss": float(loss_sum[i] / count[i]),
                **({"bytes": int(nbytes[i]), "bpb": float(loss_sum[i] / math.log(2) / nbytes[i])} if nbytes[i] else {}),
            }
            for i, name in enumerate(self.eval_sources)
            if count[i]
        }
        overall = float(loss_sum.sum() / count.sum())
        rec = {"event": "eval_detail", "step": self.step, "loss": overall, "by_source": by_source}
        if nbytes.sum():
            rec["bpb"] = float(loss_sum.sum() / math.log(2) / nbytes.sum())
        self.log(rec)
        self.last_eval_detail = rec
        return overall

    def evaluate(self) -> float:
        if isinstance(self.data, TokenStream):
            return self.evaluate_stream()
        if self.eval_batches is None:
            src = SyntheticPacked(self.cfg.vocab_size, self.a.seq_len, self.a.micro_bs, seed=self.a.seed + 10_000)
            self.eval_batches = [src.next_batch() for _ in range(2)]
        self.model.eval()
        tot, n = 0.0, 0
        with torch.no_grad(), self._autocast():
            for x, y, u in self.eval_batches:
                _, ls = self.model(x.to(self.dev), y.to(self.dev))
                tot += ls.item()
                n += u
        self.model.train()
        return tot / n

    def train_step(self):
        if self.a.sdpa == "default":
            return self._train_step()
        from torch.nn.attention import SDPBackend, sdpa_kernel

        be = {
            "flash": SDPBackend.FLASH_ATTENTION,
            "efficient": SDPBackend.EFFICIENT_ATTENTION,
            "cudnn": SDPBackend.CUDNN_ATTENTION,
            "math": SDPBackend.MATH,
        }[self.a.sdpa]
        with sdpa_kernel(be):
            return self._train_step()

    def _train_step(self):
        a, dev = self.a, self.dev
        t0 = time.perf_counter()
        batches = [self.data.next_batch() for _ in range(a.accum)]
        useful = sum(b[2] for b in batches)
        fp = batches_fingerprint(batches)  # every microbatch of the update, in order (N-06)
        t_data = time.perf_counter() - t0
        loss_tot = torch.zeros((), device=dev)
        for x, y, _ in batches:
            x = x.pin_memory().to(dev, non_blocking=True) if dev.type == "cuda" else x
            y = y.pin_memory().to(dev, non_blocking=True) if dev.type == "cuda" else y
            with self._autocast():
                _, loss_sum = self.fwd(x, y)
            scaled = loss_sum / useful
            (self.scaler.scale(scaled) if self.scaler is not None else scaled).backward()
            loss_tot += loss_sum.detach()
        if self.scaler is not None:
            self.scaler.unscale_(self.opt)
        gnorm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), a.clip)
        lr = self.sched.get_last_lr()[0]
        skipped = False
        if self.scaler is not None and not math.isfinite(gnorm.item()):
            # FP16 overflow: the scaler skips this optimizer step and lowers the loss scale.
            skipped = True
            self.skipped_updates += 1
            self._consecutive_skips += 1
            if self._consecutive_skips > 20:
                raise FloatingPointError(f"more than 20 consecutive fp16 overflows at update {self.step + 1}")
        elif not math.isfinite(gnorm.item()):
            # Non-finite gradients: the optimizer step is skipped and the run stops at this boundary.
            self.opt.zero_grad(set_to_none=True)
            self.log({"event": "abort", "reason": "non_finite_grad_norm", "step": self.step + 1})
            raise FloatingPointError(f"non-finite gradient norm at update {self.step + 1}")
        if not (math.isfinite(lr) and 0.0 <= lr <= a.lr * (1 + 1e-6)):
            raise FloatingPointError(f"learning rate {lr} outside [0, peak] at update {self.step + 1}")
        if self.scaler is not None:
            self.scaler.step(self.opt)
            self.scaler.update()
            if not skipped:
                self._consecutive_skips = 0
        else:
            self.opt.step()
        self.sched.step()
        self.opt.zero_grad(set_to_none=True)
        loss = (loss_tot / useful).item()  # synchronizes the device
        dt = time.perf_counter() - t0
        self.step += 1
        total = a.accum * a.micro_bs * a.seq_len
        self.tokens_useful += useful
        self.tokens_total += total
        if not math.isfinite(loss):
            self.log({"event": "abort", "reason": "non_finite_loss", "step": self.step})
            raise FloatingPointError(f"non-finite loss at step {self.step}")
        rec = {
            "step": self.step,
            "loss": loss,
            "loss_hex": float(loss).hex(),
            "lr": lr,
            "grad_norm": gnorm.item(),
            "skipped_update": skipped,
            "step_s": dt,
            "data_s": t_data,
            "useful_tokens": useful,
            "total_tokens": total,
            "batch_fp": fp,
        }
        self._watch(rec)
        self.log(rec)
        return rec

    def _watch(self, rec: dict) -> None:
        """Gradient-norm and loss spike alerts (diagnostic; they do not stop training by themselves)."""
        hist = self.__dict__.setdefault("_gn_hist", [])
        if len(hist) >= 20:
            med = float(np.median(hist[-200:]))
            if rec["grad_norm"] > self.a.spike_factor * med:
                rec["alert"] = f"grad_norm {rec['grad_norm']:.3g} > {self.a.spike_factor}x running median {med:.3g}"
                if getattr(self, "tl", None) is not None:
                    self.run_span.event(
                        "alert",
                        kind="grad_norm_spike",
                        step=rec["step"],
                        grad_norm=rec["grad_norm"],
                        running_median=med,
                    )
        hist.append(rec["grad_norm"])

    def run(self):
        a = self.a
        if self.tl is None:
            self.tl = TimeLog(self._time_log_root(), self.run_id, producer_name="train")
        record_clock_check(self.tl)
        self._startup_phase("clock-check")
        self.res = f"gpu:{self.dev}"
        self.attempt = f"{self.run_id}-{uuid.uuid4().hex[:8]}"
        self.run_span = self.tl.span(
            "run",
            self.run_id,
            stage_id=a.stage_id,
            attempt_id=self.attempt,
            category="supervision",
            resource=self.res,
            contract_sha256=self.contract_sha,
            code=code_identity()["package_sha256"],
            data=self.data.identity(),
            resume=a.resume,
            n_params=self.n_params,
        )
        self.run_span.start_heartbeat(30.0, lambda: {"step": self.step, "tokens_useful": self.tokens_useful})
        try:
            summary = self._run_body()
        except BaseException as e:
            self._close_segment()
            outcome = "cancelled" if isinstance(e, KeyboardInterrupt) else "failed"
            if self.startup_span is not None and not self.startup_span.ended:
                self.startup_span.end(outcome, failure_reason=f"{type(e).__name__}: {e}"[:500])
            self.run_span.end(outcome, failure_reason=f"{type(e).__name__}: {e}"[:500], step=self.step)
            self.tl.close()
            raise
        if self.startup_span is not None and not self.startup_span.ended:
            # No update ran (e.g. --stop-at already reached): the startup span still gets its terminal outcome.
            self.startup_span.end("completed", phases=self.startup["phases"], first_update=False)
            self.startup_span = None
        self.run_span.end(
            "cancelled" if self.stop_requested else "completed",
            failure_reason="SIGTERM: planned shutdown at an optimizer boundary" if self.stop_requested else None,
            step=self.step,
            tokens_useful=self.tokens_useful,
            active_train_s=self.active_total,
            evidence=[a.summary] if a.summary else [],
        )
        self.tl.close()
        return summary

    def _span(self, category: str, name: str, **fields) -> Span | nullcontext:
        if self.tl is None:
            return nullcontext()
        try:
            return self.tl.span(
                "operation",
                name,
                stage_id=self.a.stage_id,
                attempt_id=self.attempt,
                category=category,
                resource=self.res,
                parent=self.run_span,
                **fields,
            )
        except TimeLogError as e:
            self._log_failure(e)
            return nullcontext()

    def _log_failure(self, e: Exception) -> None:
        # Spec v0.6 section 8: never continue silently unlogged; stop at the next boundary.
        print(f"[timelog] {e}; stopping at the next optimizer boundary", flush=True)
        self.stop_requested = True

    def _open_segment(self) -> None:
        self.seg = self._span("gpu_compute", "train-segment", first_step=self.step + 1)
        self.seg_counts = {"steps": 0, "useful_tokens": 0, "data_wait_s": 0.0, "step_s": 0.0}

    def _close_segment(self) -> None:
        seg = getattr(self, "seg", None)
        if isinstance(seg, Span) and not seg.ended:
            try:
                seg.progress(**self.seg_counts)
                seg.end(last_step=self.step)
            except TimeLogError as e:
                self._log_failure(e)
        self.seg = None

    def _run_body(self):
        a = self.a
        signal.signal(signal.SIGTERM, self._on_term)
        resumed = False
        if a.resume != "none":
            report = []
            if a.resume == "auto":
                roots = [Path(a.ckpt_root) / self.run_id] + [Path(r) / self.run_id for r in a.fallback_root]
                p, report = C.select_latest_valid(
                    roots, contract_sha=None if a.corrective_continuation else self.contract_sha
                )
                self.log({"event": "resume_scan", "report": report})
            else:
                p = Path(a.resume)
            if p is not None:
                with self._span("restoration", "restore", source=str(p)):
                    self.restore(p)
                self._startup_phase("restore", trainer_step=self.step, source=str(p))
                resumed = True
            elif a.resume == "auto" and report:
                # Never silently restart over existing (invalid or incompatible) history.
                raise C.CheckpointError(
                    f"checkpoints exist under {self.run_id} but none is valid and compatible; "
                    "inspect them or start a new --run-id"
                )
            else:
                print("[resume] no checkpoint found; starting fresh", flush=True)
        if a.ckpt_root and not resumed and not a.no_init_ckpt:
            with self._span("checkpoint", "save-init", step=self.step):
                self.save(tags=("init",))
            self._startup_phase("init-checkpoint", trainer_step=self.step)
        last_ckpt_step, active = self.step, 0.0
        times, peak_mem = [], 0
        if self.dev.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        t_start = time.perf_counter()
        first_step_s = None
        stop_at = a.stop_at if a.stop_at is not None else a.steps
        while self.step < min(a.steps, stop_at):
            if first_step_s is None:
                with self._span("compilation_warmup" if a.compile else "gpu_compute", "first-step"):
                    rec = self.train_step()
                self._open_segment()
            else:
                rec = self.train_step()
                self.seg_counts["steps"] += 1
                self.seg_counts["useful_tokens"] += rec["useful_tokens"]
                self.seg_counts["data_wait_s"] += rec["data_s"]
                self.seg_counts["step_s"] += rec["step_s"]
            if first_step_s is None:
                first_step_s = rec["step_s"]
                if not self.startup["complete"]:
                    self._startup_complete(first_step_s)
            times.append(rec["step_s"])
            active += rec["step_s"]
            self.active_total += rec["step_s"]
            if self.step % 10 == 0 or self.step == 1:
                print(
                    f"step {self.step} loss {rec['loss']:.4f} lr {rec['lr']:.3g} {rec['step_s'] * 1000:.0f}ms",
                    flush=True,
                )
            tags = []
            if a.eval_every and self.step % a.eval_every == 0:
                self._close_segment()
                with self._span("evaluation", "eval", step=self.step):
                    self.last_eval = self.evaluate()
                self.log(self._eval_event(t_start))
                if self.best_eval is None or self.last_eval < self.best_eval:
                    self.best_eval = self.last_eval
                    tags += ["eval", "best"]
            due = (self.step - last_ckpt_step >= a.ckpt_every_steps) or (active >= a.ckpt_every_seconds) or tags
            if self.stop_requested:
                tags.append("planned_shutdown")
            if a.ckpt_root and (due or self.stop_requested):
                self._close_segment()
                with self._span("checkpoint", "save", step=self.step, tags=list(tags)):
                    self.save(tags=tuple(tags))
                last_ckpt_step, active = self.step, 0.0
            if self.seg is None and self.step < min(a.steps, stop_at) and not self.stop_requested:
                self._open_segment()
            if self.stop_requested:
                print("[signal] SIGTERM: checkpointed at optimizer boundary, exiting", flush=True)
                break
        self._close_segment()
        final_best = False
        if a.eval_every and self.step % a.eval_every and not self.stop_requested:
            # Final-step evaluation, so reported final quality is never from an earlier step.
            with self._span("evaluation", "eval-final", step=self.step):
                self.last_eval = self.evaluate()
            self.log(self._eval_event(t_start))
            if self.best_eval is None or self.last_eval < self.best_eval:
                self.best_eval = self.last_eval
                final_best = True
        if a.exact_epoch and self.step == a.steps:
            d = self.data
            if not (d.epoch == 0 and d.cursor == d.n_windows):
                raise RuntimeError(f"exact epoch violated: epoch {d.epoch} cursor {d.cursor} of {d.n_windows}")
        if a.ckpt_root and self.step != last_ckpt_step and not self.stop_requested:
            with self._span("checkpoint", "save-final", step=self.step):
                self.save(tags=("final", "eval", "best") if final_best else ("final",))
        wall = time.perf_counter() - t_start
        if self.dev.type == "cuda":
            peak_mem = torch.cuda.max_memory_allocated()
        summary = self._summary(times, first_step_s, wall, peak_mem)
        if a.summary:
            Path(a.summary).write_text(json.dumps(summary, indent=1))
        print(
            json.dumps(
                {
                    k: summary[k]
                    for k in (
                        "steps_run",
                        "final_step",
                        "tokens_per_s_median",
                        "step_ms_median",
                        "peak_vram_gib",
                        "final_param_sha256",
                    )
                }
            ),
            flush=True,
        )
        return summary

    def _eval_identity(self) -> dict:
        if isinstance(self.data, TokenStream):
            if self.a.eval_set == "full-split":
                self._ensure_full_split()
                return self.eval_identity
            return {"kind": "first_windows_v1", "split": "validation", "windows": self.a.eval_windows}
        return {"kind": "synthetic_heldout_v1", "seed": self.a.seed + 10_000}

    def _eval_event(self, t_start: float) -> dict:
        """Eval record with the timing boundaries used for equal-compute comparisons:
        ``train_seconds`` sums synchronized optimizer-update times (warm-up/compile included,
        evaluation and checkpoint time excluded); ``wall_seconds`` is the process-local monotonic
        time since the training loop started (everything included)."""
        rec = {
            "event": "eval",
            "step": self.step,
            "eval_loss": self.last_eval,
            "train_seconds": self.active_total,
            "wall_seconds": time.perf_counter() - t_start,
            "tokens_useful": self.tokens_useful,
        }
        if self.last_eval_detail and "bpb" in self.last_eval_detail:
            rec["eval_bpb"] = self.last_eval_detail["bpb"]
        return rec

    def _on_term(self, *_):
        self.stop_requested = True

    def _summary(self, times, first_step_s, wall, peak_mem):
        a = self.a
        steady = times[a.warmup_steps_excluded :] if len(times) > a.warmup_steps_excluded else times
        tok_per_step = a.accum * a.micro_bs * a.seq_len
        med = float(np.median(steady)) if steady else None
        rss_kib = 0
        with Path("/proc/self/status").open() as f:
            for line in f:
                if line.startswith("VmHWM"):
                    rss_kib = int(line.split()[1])
        return {
            "utc": datetime.now(UTC).isoformat(),
            "run_id": self.run_id,
            "contract_sha256": self.contract_sha,
            "n_params": self.n_params,
            "config": vars(a),
            "model": self.cfg.to_dict(),
            "deps": deps_identity(),
            "code": code_identity(),
            "steps_run": len(times),
            "final_step": self.step,
            "steady_steps": len(steady),
            "first_step_s": first_step_s,
            "wall_s": wall,
            "step_ms_median": med * 1000 if med else None,
            "step_ms_p90": float(np.percentile(steady, 90)) * 1000 if steady else None,
            "step_ms_p99": float(np.percentile(steady, 99)) * 1000 if steady else None,
            "step_ms_max": float(np.max(steady)) * 1000 if steady else None,
            "tokens_per_step": tok_per_step,
            "tokens_per_s_median": tok_per_step / med if med else None,
            "tokens_per_s_mean": tok_per_step * len(steady) / sum(steady) if steady else None,
            "tokens_useful": self.tokens_useful,
            "tokens_total": self.tokens_total,
            "peak_vram_gib": peak_mem / 2**30,
            "peak_vram_reserved_gib": (torch.cuda.max_memory_reserved() / 2**30) if self.dev.type == "cuda" else 0,
            "peak_host_rss_gib": rss_kib / 2**20,
            "last_eval": self.last_eval,
            "last_eval_detail": self.last_eval_detail,
            "eval_set": getattr(self, "eval_identity", None),
            "data_identity": self.data.identity(),
            "device_name": torch.cuda.get_device_name(self.dev) if self.dev.type == "cuda" else "cpu",
            "consumed": self.consumed,
            "startup": self.startup,
            "train_seconds": self.active_total,
            "train_bytes_per_s_median": (
                self.consumed["target_bytes"] / a.steps / med
                if self.consumed and "target_bytes" in self.consumed and med
                else None
            ),
            "best_eval": self.best_eval,
            "final_param_sha256": tensor_tree_sha(self.model.state_dict()),
            "final_optim_sha256": tensor_tree_sha(self.opt.state_dict()),
        }


def main(argv=None):
    args = build_parser().parse_args(argv)
    Trainer(args).run()


if __name__ == "__main__":
    main()
