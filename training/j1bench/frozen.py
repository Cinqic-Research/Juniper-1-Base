"""Loading a frozen Juniper 1 base by its immutable freeze record (never by ``latest`` or ``best``).

A freeze record (``juniper1.base-freeze.v1``, e.g. ``docs/freeze/juniper1-base-v1.json``) binds a
versioned freeze ID to one authentic full-state checkpoint: its directory name, manifest SHA-256,
``model.pt`` file SHA-256, model tensor-tree SHA-256, model configuration, parameter count and
tokenizer SHA-256. ``load_frozen_base`` refuses to return weights unless every one of those
identities matches the bytes on disk. It reads only; it never writes or re-saves a checkpoint.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import torch

from j1bench.ckpt import canonical, sha256_file, verify_checkpoint
from j1bench.model import ModelConfig, Transformer

FREEZE_SCHEMA = "juniper1.base-freeze.v1"
FREEZE_ID = re.compile(r"juniper1-base-v[1-9][0-9]*")
DIGEST_KEY = "freeze_record_sha256"


class FreezeError(RuntimeError):
    """The freeze record or the checkpoint bytes do not establish the frozen identity."""


def record_digest(record: dict[str, Any]) -> str:
    return hashlib.sha256(canonical({k: v for k, v in record.items() if k != DIGEST_KEY})).hexdigest()


def read_freeze_record(path: Path) -> dict[str, Any]:
    rec = json.loads(Path(path).read_text())
    if rec.get("schema") != FREEZE_SCHEMA:
        raise FreezeError(f"unknown freeze schema {rec.get('schema')!r}")
    if not isinstance(rec.get("freeze_id"), str) or FREEZE_ID.fullmatch(rec["freeze_id"]) is None:
        raise FreezeError(f"freeze_id {rec.get('freeze_id')!r} is not a versioned juniper1-base-vN identity")
    if rec.get(DIGEST_KEY) != record_digest(rec):
        raise FreezeError("freeze record self-hash mismatch")
    if rec.get("decision") != "BASE_FREEZE_APPROVED":
        raise FreezeError(f"freeze decision is {rec.get('decision')!r}, not BASE_FREEZE_APPROVED")
    return rec


def load_frozen_base(
    record_path: Path, checkpoint_dir: Path, *, tokenizer_path: Path, expected_record_sha256: str
) -> tuple[Transformer, dict[str, Any]]:
    """Return ``(model in eval mode on CPU, freeze record)`` after verifying every bound identity.

    ``expected_record_sha256`` pins the record itself: its self-hash only proves internal
    consistency, so a caller (a stage contract) must state which freeze record it inherits.
    """
    rec = read_freeze_record(record_path)
    if rec[DIGEST_KEY] != expected_record_sha256:
        raise FreezeError("freeze record is not the one the caller pinned (expected_record_sha256)")
    src = rec["source_checkpoint"]
    ckpt_dir = Path(checkpoint_dir)
    if ckpt_dir.name != src["name"]:
        raise FreezeError(f"checkpoint directory {ckpt_dir.name!r} is not the frozen {src['name']!r}")
    ok, why = verify_checkpoint(ckpt_dir)
    if not ok:
        raise FreezeError(f"checkpoint verification failed: {why}")
    man = json.loads((ckpt_dir / "manifest.json").read_text())
    if man["manifest_sha256"] != src["manifest_sha256"] or man["run_id"] != src["run_id"] or man["step"] != src["step"]:
        raise FreezeError("checkpoint manifest identity differs from the freeze record")
    if sha256_file(ckpt_dir / "model.pt") != src["files"]["model.pt"]["sha256"]:
        raise FreezeError("model.pt bytes differ from the freeze record")
    if sha256_file(Path(tokenizer_path)) != rec["tokenizer"]["sha256"]:
        raise FreezeError("tokenizer bytes differ from the freeze record")
    cfg_dict = man["run_contract"]["model"]
    if cfg_dict != rec["model"]["config"]:
        raise FreezeError("checkpoint model configuration differs from the freeze record")
    sd = torch.load(ckpt_dir / "model.pt", map_location="cpu", weights_only=True)
    from j1bench.train import tensor_tree_sha  # local import: train pulls in the optimizer stack

    if tensor_tree_sha(sd) != rec["model"]["state_sha256"]:
        raise FreezeError("model tensor-tree digest differs from the freeze record")
    model = Transformer(ModelConfig(**cfg_dict))
    model.load_state_dict(sd, strict=True)
    if model.num_params() != rec["model"]["parameters"]:
        raise FreezeError("parameter count differs from the freeze record")
    return model.eval(), rec
