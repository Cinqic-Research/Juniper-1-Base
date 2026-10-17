"""Supervised fine-tuning (instruction SFT) of the frozen Juniper 1 base.

A separate optimization stage, not a continuation of pretraining: the stage starts from the
weights of an immutable freeze record (``j1bench.frozen.load_frozen_base``) with a fresh AdamW
state and its own schedule. Nothing from the base's optimizer, scheduler or data cursor is
reused, and the base checkpoint is only read.

Contract (``j1.sft-contract.v1``, JSON): base freeze record + pinned digest, dataset manifest
digest, tokenizer digest, optimizer, schedule, batch, precision, seed, update count, checkpoint
cadence, evaluation cadence. Its canonical SHA-256 is the run contract; an exact resume requires
the identical contract (a changed dataset, tokenizer, batch structure, precision, world size or
optimizer setting is refused, never silently resumed).

Loss: per update, the sum of token cross-entropies over supervised targets divided by the
update's supervised-target count (exact under gradient accumulation and padding). Targets come
pre-shifted from the dataset and are verified by ``j1bench.sftdata.check_alignment``.

Checkpoints: full state (model, AdamW, scheduler, RNG of Python/NumPy/Torch/CUDA, data cursor,
exposure counters) through ``j1bench.ckpt.save_checkpoint`` (staged, hashed, manifest-last,
``COMPLETE`` marker). A checkpoint is published at the earlier of ``every_updates`` (≤ 500) completed
updates or ``every_seconds`` (≤ 900) since the previous publication, at the end, and before
exiting on SIGTERM/SIGINT (after the update in flight completes). A failed save is retried
once; if it fails again the run stops with exit code 3 and the previous verified checkpoint
remains the latest valid state.

Exit codes: 0 completed, 3 checkpoint failure, 4 stopped on signal after a verified checkpoint,
5 numerical abort (non-finite loss/gradient; no update applied, nothing saved), 6 stopped by
``--stop-after`` (qualification only; nothing saved at that point).

Only fp32 is qualified (the bf16/fp16 code paths are refused by contract validation), one process
per run directory (exclusive lock; ``WORLD_SIZE`` must be 1), and an exact resume also requires the
same trainer code (package digest); a different library/device stack is logged.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib
import json
import math
import os
import random
import shutil
import signal
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from . import ckpt as C
from .frozen import load_frozen_base, read_freeze_record
from .sftdata import SFTDataset, micro_batches, plan_fingerprint, update_plan
from .train import code_identity, deps_identity, lr_lambda, tensor_tree_sha

CONTRACT_SCHEMA = "j1.sft-contract.v1"
MAX_EVERY_UPDATES = 500
MAX_EVERY_SECONDS = 900.0
EXIT_CKPT_FAILED, EXIT_SIGNAL, EXIT_NUMERIC, EXIT_STOPPED = 3, 4, 5, 6


class ContractError(ValueError):
    pass


def contract_sha(contract: dict) -> str:
    return hashlib.sha256(C.canonical(contract)).hexdigest()


def validate_contract(c: dict) -> None:
    if c.get("schema") != CONTRACT_SCHEMA:
        raise ContractError(f"unknown contract schema {c.get('schema')!r}")
    need = (
        "stage_id",
        "run_id",
        "base",
        "dataset",
        "optimizer",
        "schedule",
        "batch",
        "precision",
        "seed",
        "checkpoint",
    )
    missing = [k for k in need if k not in c]
    if missing:
        raise ContractError(f"contract lacks {missing}")
    ck = c["checkpoint"]
    if not 1 <= ck["every_updates"] <= MAX_EVERY_UPDATES or not 0 < ck["every_seconds"] <= MAX_EVERY_SECONDS:
        raise ContractError("checkpoint cadence must be <= 500 updates and <= 900 seconds")
    if c["precision"] != "fp32":
        # bf16/fp16 paths exist but are not qualified (fp16 GradScaler skips would still advance the
        # schedule and data cursor); a later contract version must qualify them first.
        raise ContractError(f"precision {c['precision']!r} is not qualified for SFT v1 (fp32 only)")
    if c.get("world_size", 1) != 1:
        raise ContractError("only single-device SFT is qualified")
    if int(c.get("log_every", 10)) < 1 or int(c.get("dev_every", 0)) < 0:
        raise ContractError("log_every must be >= 1 and dev_every >= 0")
    s = c["schedule"]
    if not 0 <= s["warmup_updates"] <= s["total_updates"] or s["total_updates"] < 1:
        raise ContractError("schedule updates out of range")
    o = c["optimizer"]
    if o.get("name") != "adamw" or not 0 < o["lr"] < 1e-2:
        raise ContractError("optimizer must be AdamW with 0 < lr < 1e-2")
    init = c.get("init")
    if init is not None and (
        init.get("kind") != "sft_checkpoint"
        or not all(init.get(k) for k in ("checkpoint", "manifest_sha256", "model_state_sha256"))
    ):
        raise ContractError("init must name an sft_checkpoint with its manifest and model-state digests")


@torch.no_grad()
def dev_loss(model, dev, token_budget: int, device) -> tuple[float, int]:
    """Mean CE over every dev target in a fixed order (64-episode chunks); leaves the model in eval mode."""
    model.eval()
    tot, n = 0.0, 0
    idx = np.arange(dev.n)
    for start in range(0, dev.n, 64):
        for x, y, _, cnt in micro_batches(dev, idx[start : start + 64], token_budget):
            _, ls = model(x.to(device), y.to(device))
            tot += float(ls)
            n += cnt
    return tot / n, n


class SFTTrainer:
    def __init__(
        self,
        contract: dict,
        *,
        freeze_record: Path,
        base_checkpoint: Path,
        tokenizer: Path,
        dataset_dir: Path,
        out_root: Path,
        staging: Path,
        device: str,
        init_checkpoint: Path | None = None,
        model_override=None,
        clock=time.monotonic,
        log=print,
    ):
        validate_contract(contract)
        if os.environ.get("WORLD_SIZE", "1") != "1":
            raise ContractError("WORLD_SIZE != 1: only single-process SFT is qualified")
        self.c = contract
        self.sha = contract_sha(contract)
        self.run_id = contract["run_id"]
        self.out_root, self.staging = Path(out_root), Path(staging)
        self.dev = torch.device(device)
        self.clock = clock
        self.print = log
        self._stop_signal: int | None = None
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        if contract.get("deterministic", True):
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.benchmark = False
        if model_override is not None:  # synthetic qualification only: an in-memory fixture model
            self.model = model_override
            self.freeze: dict[str, Any] = {"freeze_id": "synthetic", "freeze_record_sha256": "synthetic"}
        else:
            base = contract["base"]
            self.model, self.freeze = load_frozen_base(
                freeze_record,
                base_checkpoint,
                tokenizer_path=tokenizer,
                expected_record_sha256=base["freeze_record_sha256"],
            )
            if self.freeze["freeze_id"] != base["freeze_id"]:
                raise ContractError("freeze record ID differs from the contract")
        self.init_identity: dict[str, Any] | None = None
        if contract.get("init") is not None:
            self.init_identity = self._load_init(contract["init"], init_checkpoint)
        elif init_checkpoint is not None:
            raise ContractError("an init checkpoint was given but the contract declares none")
        tok_sha = C.sha256_file(tokenizer)
        if tok_sha != contract["dataset"]["tokenizer_sha256"]:
            raise ContractError("tokenizer bytes differ from the contract")
        self.data = SFTDataset(
            dataset_dir,
            expected_manifest_sha256=contract["dataset"]["manifest_sha256"],
            tokenizer_sha256=tok_sha,
            vocab_size=self.model.cfg.vocab_size,
        )
        self.model.to(self.dev).train()
        if self.dev.type == "cuda" and contract.get("gqa_kernel", "expand") == "expand":
            self.model.set_gqa_kernel("expand")
        o, s = contract["optimizer"], contract["schedule"]
        self.opt = torch.optim.AdamW(
            self.model.param_groups(o["weight_decay"], o.get("decay_policy", "matrices")),
            lr=o["lr"],
            betas=tuple(o["betas"]),
            eps=o["eps"],
            foreach=False,
        )
        self.sched = torch.optim.lr_scheduler.LambdaLR(
            self.opt, lr_lambda(s["warmup_updates"], s["total_updates"], s["min_lr_ratio"])
        )
        self.scaler = torch.amp.GradScaler("cuda") if contract["precision"] == "fp16" else None
        self.update = 0
        self.epoch = 0
        self.update_in_epoch = 0
        self.exposure = {"episodes": 0, "target_tokens": 0, "input_tokens": 0, "batch_cells": 0, "pad_tokens": 0}
        self.parent: str | None = None
        self.last_pub_time = self.clock()
        self.last_pub_update = 0
        self.dev_history: list[dict] = []
        self.run_dir = self.out_root / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._lock = (self.run_dir / ".lock").open("w")
        try:
            fcntl.flock(self._lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ContractError(f"{self.run_dir} is in use by another trainer process") from None
        self._check_batch_contract()
        self.log_f = (self.run_dir / "sft_log.jsonl").open("a")
        self._seed_all(int(contract["seed"]))

    def _load_init(self, init: dict, path: Path | None) -> dict[str, Any]:
        """Weights of a verified SFT checkpoint that descends from this stage's freeze record; the optimizer,
        schedule and data cursor still start fresh (a new lineage, recorded in every checkpoint manifest)."""
        if path is None:
            raise ContractError("the contract declares an init checkpoint; pass --init-checkpoint")
        path = Path(path)
        ok, why = C.verify_checkpoint(path)
        if not ok:
            raise ContractError(f"init checkpoint does not verify: {why}")
        m = C.read_manifest(path)
        if m is None:
            raise ContractError("init checkpoint has no readable manifest")
        if m["name"] != init["checkpoint"] or m["manifest_sha256"] != init["manifest_sha256"]:
            raise ContractError("init checkpoint identity differs from the contract")
        if m.get("base_freeze_record_sha256") != self.c["base"]["freeze_record_sha256"]:
            raise ContractError("init checkpoint does not descend from the contract's freeze record")
        sd = torch.load(path / "model.pt", map_location="cpu", weights_only=True)
        if tensor_tree_sha(sd) != init["model_state_sha256"]:
            raise ContractError("init model state differs from the contract")
        self.model.load_state_dict(sd, strict=True)
        return {"checkpoint": m["name"], "manifest_sha256": m["manifest_sha256"], "run_id": m.get("run_id"),
                "model_state_sha256": init["model_state_sha256"]}  # fmt: skip

    def _check_batch_contract(self) -> None:
        """Fail at startup, not mid-run, when the batch contract cannot serve this dataset."""
        b = self.c["batch"]
        longest = max(int(s.lengths().max()) for s in self.data.splits.values())
        if longest > b["token_budget"]:
            raise ContractError(f"longest episode ({longest} tokens) exceeds token_budget {b['token_budget']}")
        if longest > self.model.cfg.max_seq_len:
            raise ContractError(f"longest episode ({longest}) exceeds the model context {self.model.cfg.max_seq_len}")
        if not 1 <= b["episodes_per_update"] <= self.data.splits["train"].n:
            raise ContractError("episodes_per_update must be between 1 and the number of train episodes")

    # --- bookkeeping ------------------------------------------------------------------------
    def log(self, rec: dict) -> None:
        if self.log_f.closed:
            return
        self.log_f.write(json.dumps({"t": time.time(), **rec}) + "\n")
        self.log_f.flush()

    def _seed_all(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    def rng_state(self) -> dict:
        s = {"python": random.getstate(), "numpy": np.random.get_state(), "torch_cpu": torch.get_rng_state()}
        if self.dev.type == "cuda":
            s["torch_cuda"] = torch.cuda.get_rng_state_all()
        return s

    def set_rng_state(self, s: dict) -> None:
        random.setstate(s["python"])
        np.random.set_state(s["numpy"])
        torch.set_rng_state(s["torch_cpu"])
        if self.dev.type == "cuda" and "torch_cuda" in s:
            torch.cuda.set_rng_state_all(s["torch_cuda"])

    def identities(self) -> dict:
        return {
            "base_freeze_id": self.freeze["freeze_id"],
            "base_freeze_record_sha256": self.freeze["freeze_record_sha256"],
            "base_checkpoint": (self.freeze.get("source_checkpoint") or {}).get("name"),
            "dataset": self.data.identity(),
            "tokenizer_sha256": self.c["dataset"]["tokenizer_sha256"],
            "code_identity": code_identity(),
            "deps": deps_identity(),
        }

    # --- checkpointing ------------------------------------------------------------------------
    def save(self, reason: str) -> C.SaveResult | None:
        now = self.clock()
        trainer_state = {
            "update": self.update,
            "epoch": self.epoch,
            "update_in_epoch": self.update_in_epoch,
            "exposure": dict(self.exposure),
            "scheduler": self.sched.state_dict(),
            "rng": self.rng_state(),
            "grad_scaler": self.scaler.state_dict() if self.scaler is not None else None,
            "dev_history": list(self.dev_history),
            "contract_sha256": self.sha,
            "grad_accum_position": 0,  # saves happen only at optimizer boundaries
        }
        payload = {"model.pt": self.model.state_dict(), "optim.pt": self.opt.state_dict(), "trainer.pt": trainer_state}
        meta = {
            "run_id": self.run_id,
            "run_contract_sha256": self.sha,
            "run_contract": self.c,
            "stage": self.c["stage_id"],
            "kind": "j1.sft-checkpoint.v1",
            "model_config": self.model.cfg.to_dict(),
            **self.identities(),
            "lineage": {"parent": self.parent, "root": self.freeze["freeze_id"], "init": self.init_identity},
            "updates": self.update,
            "data_cursor": {"epoch": self.epoch, "update_in_epoch": self.update_in_epoch, "seed": self.c["seed"]},
            "exposure": dict(self.exposure),
            "cadence": {
                "reason": reason,
                "updates_since_previous": self.update - self.last_pub_update,
                "seconds_since_previous": round(now - self.last_pub_time, 3),
                "limits": {
                    "updates": self.c["checkpoint"]["every_updates"],
                    "seconds": self.c["checkpoint"]["every_seconds"],
                },
            },
            "numerics": {
                "precision": self.c["precision"],
                "tf32": False,
                "deterministic": self.c.get("deterministic", True),
            },
            "state_sha256": {
                "model": tensor_tree_sha(self.model.state_dict()),
                "optim": tensor_tree_sha(self.opt.state_dict()),
            },
            "last_dev": self.dev_history[-1] if self.dev_history else None,
            "non_production": True,
        }
        for attempt in (1, 2):
            try:
                r = C.save_checkpoint(
                    self.out_root, self.staging, step=self.update, payload=payload, meta=meta, torch_save=torch.save
                )
            except (OSError, RuntimeError, C.CheckpointError) as e:  # torch.save raises RuntimeError on write errors
                for leftover in self.staging.glob(f"ckpt-{self.update:09d}-*"):
                    shutil.rmtree(leftover, ignore_errors=True)  # staged copy only; never the published directory
                self.log({"event": "checkpoint_failed", "update": self.update, "attempt": attempt, "error": repr(e)})
                self.print(f"[sft] checkpoint FAILED attempt {attempt} at update {self.update}: {e!r}")
                continue
            self.parent = r.manifest["name"]
            self.log(
                {
                    "event": "checkpoint",
                    "update": self.update,
                    "name": r.manifest["name"],
                    "reason": reason,
                    "updates_since_previous": self.update - self.last_pub_update,
                    "seconds_since_previous": round(now - self.last_pub_time, 3),
                    **{f"ckpt_{k}": v for k, v in r.timings.items()},
                }
            )
            self.last_pub_time, self.last_pub_update = self.clock(), self.update
            return r
        self.log({"event": "abort", "reason": "checkpoint_failed", "update": self.update})
        return None

    def restore(self, path: Path) -> None:
        ok, why = C.verify_checkpoint(path)
        if not ok:
            raise C.CheckpointError(f"refusing to restore {path}: {why}")
        m = C.read_manifest(path)
        assert m is not None
        if m.get("kind") != "j1.sft-checkpoint.v1":
            raise C.CheckpointError("not an SFT checkpoint")
        if m["run_contract_sha256"] != self.sha:
            raise C.CheckpointError("contract differs from the checkpoint's; exact resume is impossible")
        if m["base_freeze_record_sha256"] != self.freeze["freeze_record_sha256"]:
            raise C.CheckpointError("checkpoint descends from a different frozen base")
        if m["dataset"]["manifest_sha256"] != self.data.manifest["manifest_sha256"]:
            raise C.CheckpointError("checkpoint was trained on a different dataset")
        if m["code_identity"]["package_sha256"] != code_identity()["package_sha256"]:
            raise C.CheckpointError("trainer code differs from the checkpoint's; exact resume is impossible")
        if m["deps"] != deps_identity():
            self.log({"event": "resume_environment_differs", "checkpoint": m["deps"], "now": deps_identity()})
        ms = torch.load(path / "model.pt", map_location=self.dev, weights_only=True)
        os_ = torch.load(
            path / "optim.pt", map_location="cpu", weights_only=True
        )  # AdamW moves moments; step stays on CPU
        np_globals = [
            importlib.import_module("numpy._core.multiarray")._reconstruct,
            np.ndarray,
            np.dtype,
            type(np.dtype(np.uint32)),
            type(np.dtype(np.float64)),
        ]
        with torch.serialization.safe_globals(np_globals):
            ts = torch.load(path / "trainer.pt", map_location="cpu", weights_only=True)
        self.model.load_state_dict(ms, strict=True)
        self.opt.load_state_dict(os_)
        if tensor_tree_sha(self.model.state_dict()) != m["state_sha256"]["model"]:
            raise C.CheckpointError("restored model state differs from the manifest")
        if tensor_tree_sha(self.opt.state_dict()) != m["state_sha256"]["optim"]:
            raise C.CheckpointError("restored optimizer state differs from the manifest")
        self.sched.load_state_dict(ts["scheduler"])
        if self.scaler is not None:
            self.scaler.load_state_dict(ts["grad_scaler"])
        self.update, self.epoch, self.update_in_epoch = ts["update"], ts["epoch"], ts["update_in_epoch"]
        self.exposure = dict(ts["exposure"])
        self.dev_history = list(ts["dev_history"])
        self.set_rng_state(ts["rng"])
        self.parent = m["name"]
        self.last_pub_update = self.update
        self.last_pub_time = self.clock()
        self.log({"event": "restored", "from": str(path), "update": self.update, "lr": self.sched.get_last_lr()[0]})
        self.print(f"[sft] restored {path.name} at update {self.update}")

    # --- training -----------------------------------------------------------------------------
    def _autocast(self):
        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(self.c["precision"])
        return torch.autocast(self.dev.type, dtype=dtype or torch.float32, enabled=dtype is not None)

    def next_update_batches(self) -> list[tuple]:
        b = self.c["batch"]
        train = self.data.splits["train"]
        plan = update_plan(train.n, int(self.c["seed"]), self.epoch, b["episodes_per_update"])
        if self.update_in_epoch >= len(plan):
            self.epoch, self.update_in_epoch = self.epoch + 1, 0
            plan = update_plan(train.n, int(self.c["seed"]), self.epoch, b["episodes_per_update"])
        if self.update_in_epoch == 0:
            dropped = train.n - plan.size
            self.log({"event": "epoch_start", "epoch": self.epoch, "episodes_skipped_this_epoch": dropped})
        return micro_batches(train, plan[self.update_in_epoch], b["token_budget"])

    def train_update(self) -> dict:
        t0 = time.perf_counter()
        batches = self.next_update_batches()
        fp = plan_fingerprint(batches)
        n_targets = sum(b[3] for b in batches)
        loss_sum_total = torch.zeros((), device=self.dev, dtype=torch.float64)
        fault = os.environ.get("J1_SFT_FAULT", "")
        for i, (x, y, _, _) in enumerate(batches):
            if fault == f"kill_mid_update:{self.update + 1}" and i == 1:
                os._exit(137)  # failure injection: process dies with a partially accumulated gradient
            x, y = x.to(self.dev), y.to(self.dev)
            with self._autocast():
                _, loss_sum = self.model(x, y)
            scaled = loss_sum / n_targets
            (self.scaler.scale(scaled) if self.scaler is not None else scaled).backward()
            loss_sum_total += loss_sum.detach().double()
        if self.scaler is not None:
            self.scaler.unscale_(self.opt)
        gnorm = float(torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.c["optimizer"]["clip"]))
        loss = float(loss_sum_total / n_targets)
        lr = self.sched.get_last_lr()[0]
        if not math.isfinite(loss) or (not math.isfinite(gnorm) and self.scaler is None):
            self.opt.zero_grad(set_to_none=True)
            self.log(
                {
                    "event": "abort",
                    "reason": "non_finite",
                    "update": self.update + 1,
                    "loss": repr(loss),
                    "grad_norm": repr(gnorm),
                }
            )
            raise FloatingPointError(f"non-finite loss/gradient at update {self.update + 1}")
        if self.scaler is not None:
            self.scaler.step(self.opt)
            self.scaler.update()
        else:
            self.opt.step()
        self.sched.step()
        self.opt.zero_grad(set_to_none=True)
        self.update += 1
        self.update_in_epoch += 1
        n_inputs = sum(int(sum(len(self.data.splits["train"].episode(e)[0]) for e in b[2])) for b in batches)
        cells = sum(int(b[0].numel()) for b in batches)
        self.exposure["episodes"] += sum(len(b[2]) for b in batches)
        self.exposure["target_tokens"] += n_targets
        self.exposure["input_tokens"] += n_inputs
        self.exposure["batch_cells"] += cells
        self.exposure["pad_tokens"] += cells - n_inputs
        rec = {
            "update": self.update,
            "epoch": self.epoch,
            "loss": loss,
            "lr": lr,
            "grad_norm": gnorm,
            "targets": n_targets,
            "micro_batches": len(batches),
            "batch_cells": cells,
            "batch_fp": fp,
            "step_s": round(time.perf_counter() - t0, 4),
        }
        self.log(rec)
        return rec

    def evaluate_dev(self) -> dict:
        loss, n = dev_loss(self.model, self.data.splits["dev"], self.c["batch"]["token_budget"], self.dev)
        self.model.train()
        rec = {"update": self.update, "dev_loss": loss, "dev_targets": n}
        self.dev_history.append(rec)
        self.log({"event": "dev_eval", **rec})
        return rec

    def _due(self) -> str | None:
        ck = self.c["checkpoint"]
        if self.update - self.last_pub_update >= ck["every_updates"]:
            return "cadence_updates"
        if self.clock() - self.last_pub_time >= ck["every_seconds"]:
            return "cadence_time"
        return None

    def _on_signal(self, signum, _frame) -> None:
        self._stop_signal = signum

    def _refuse_foreign_checkpoints(self) -> None:
        """A run directory belongs to one contract; never start or resume over another contract's states."""
        for p in C.scan(self.run_dir):
            m = C.read_manifest(p)
            if m is not None and m.get("run_contract_sha256") not in (None, self.sha):
                raise ContractError(
                    f"{self.run_dir} holds {p.name} from contract {m['run_contract_sha256'][:16]}; "
                    "a changed contract needs a new run_id (new lineage)"
                )

    def run(self, *, resume: bool = True, stop_after: int | None = None) -> int:
        self._refuse_foreign_checkpoints()
        if not resume and C.scan(self.run_dir):
            raise ContractError(f"{self.run_dir} already holds checkpoints; refusing to restart over them")
        if resume:
            latest, report = C.select_latest_valid([self.run_dir], contract_sha=self.sha, log=lambda *_: None)
            self.log({"event": "resume_scan", "candidates": report})
            if latest is not None:
                self.restore(latest)
                due = self.c.get("dev_every") and self.update % self.c["dev_every"] == 0 and self.update > 0
                if due and not any(d["update"] == self.update for d in self.dev_history):
                    self.evaluate_dev()  # the checkpoint was saved before this update's dev eval ran
        old = {s: signal.signal(s, self._on_signal) for s in (signal.SIGTERM, signal.SIGINT)}
        total = self.c["schedule"]["total_updates"]
        self.log({"event": "start", "update": self.update, "contract_sha256": self.sha, **self.identities()})
        self.last_pub_time = self.clock()
        try:
            while self.update < total:
                if stop_after is not None and self.update >= stop_after:
                    self.log({"event": "stopped_after", "update": self.update})
                    return EXIT_STOPPED
                try:
                    rec = self.train_update()
                except FloatingPointError as e:
                    self.print(f"[sft] numerical abort: {e}")
                    return EXIT_NUMERIC
                if self.update % self.c.get("log_every", 10) == 0:
                    self.print(
                        f"[sft] update {rec['update']} loss {rec['loss']:.4f} lr {rec['lr']:.3g} "
                        f"gn {rec['grad_norm']:.3f}"
                    )
                if self.update >= total:
                    break  # the final block evaluates dev and publishes the final checkpoint
                # Cadence is checked before the (slow) dev evaluation so a save is never delayed by it.
                reason = self._due()
                if self._stop_signal is not None:
                    reason = f"signal_{self._stop_signal}"
                if reason and self.save(reason) is None:
                    return EXIT_CKPT_FAILED
                if self._stop_signal is not None:
                    self.log({"event": "stopped_on_signal", "update": self.update})
                    return EXIT_SIGNAL
                if self.c.get("dev_every") and self.update % self.c["dev_every"] == 0:
                    self.evaluate_dev()
            if not self.dev_history or self.dev_history[-1]["update"] != self.update:
                self.evaluate_dev()
            if self.last_pub_update != self.update and self.save("final") is None:
                return EXIT_CKPT_FAILED
            self.log({"event": "done", "update": self.update, "exposure": self.exposure})
            return 0
        finally:
            for s, h in old.items():
                signal.signal(s, h)
            self.close()

    def close(self) -> None:
        """Release the run-directory lock and the log (a trainer object runs at most once)."""
        if not self.log_f.closed:
            self.log_f.close()
        if not self._lock.closed:
            fcntl.flock(self._lock, fcntl.LOCK_UN)
            self._lock.close()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--contract", type=Path, required=True)
    ap.add_argument("--freeze-record", type=Path, required=True)
    ap.add_argument("--base-checkpoint", type=Path, required=True)
    ap.add_argument("--tokenizer", type=Path, required=True)
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--staging", type=Path, required=True)
    ap.add_argument("--init-checkpoint", type=Path, help="the SFT checkpoint the contract's init names")
    ap.add_argument("--device", help="cuda (default; refused when unavailable) or an explicit cpu")
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--stop-after", type=int, default=None, help="stop (without saving) at this update (qualification)")
    a = ap.parse_args(argv)
    if a.device is None:
        if not torch.cuda.is_available():
            # A broken GPU host must not silently train on the CPU (a different environment).
            raise SystemExit("no usable CUDA device; pass --device cpu to run on the CPU deliberately")
        a.device = "cuda"
    contract = json.loads(a.contract.read_text())
    read_freeze_record(a.freeze_record)
    t = SFTTrainer(
        contract,
        freeze_record=a.freeze_record,
        base_checkpoint=a.base_checkpoint,
        tokenizer=a.tokenizer,
        dataset_dir=a.dataset,
        out_root=a.out_root,
        staging=a.staging,
        device=a.device,
        init_checkpoint=a.init_checkpoint,
    )
    t.print(f"[sft] {datetime.now(UTC).isoformat()} contract {t.sha[:16]} run {t.run_id} on {a.device}")
    return t.run(resume=not a.no_resume, stop_after=a.stop_after)


if __name__ == "__main__":
    sys.exit(main())
