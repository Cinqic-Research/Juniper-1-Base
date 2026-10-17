"""Identity checks for a Juniper 1 Base release directory: file hashes, configuration, tensor digest."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from .model import FROZEN_CONFIG, FROZEN_PARAMETERS

FREEZE_ID = "juniper1-base-v1"
SOURCE_MODEL_PT_SHA256 = "71773e46db2511332af97891b6485b04bf414a448dab9c4b050988e0b532649c"
TENSOR_TREE_SHA256 = "1fd07c0d0c6c6b355460ab82ebe02530047b5682a4e6a41b9e4f0146323076e4"
RELEASE_FILES = ("model.safetensors", "config.json")


class VerifyError(RuntimeError):
    pass


def sha256_file(path: Path, bufsize: int = 16 << 20) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while b := f.read(bufsize):
            h.update(b)
    return h.hexdigest()


def tensor_tree_sha(sd: dict) -> str:
    """SHA-256 over sorted (key, raw little-endian bytes) of a state dict; equals Juniper-1-Train's
    ``j1bench.train.tensor_tree_sha`` for a flat dict of tensors."""
    h = hashlib.sha256()
    for k in sorted(sd, key=str):
        t = sd[k]
        h.update(f"/{k}".encode())
        h.update(t.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes() if t.numel() else b"")
    return h.hexdigest()


def read_config(directory: Path) -> dict:
    cfg = json.loads((Path(directory) / "config.json").read_text())
    model_cfg = {k: cfg.get(k) for k in FROZEN_CONFIG}
    if model_cfg != FROZEN_CONFIG:
        diff = {k: (model_cfg[k], FROZEN_CONFIG[k]) for k in FROZEN_CONFIG if model_cfg[k] != FROZEN_CONFIG[k]}
        raise VerifyError(f"config.json differs from the frozen juniper1-base-v1 configuration: {diff}")
    if cfg.get("freeze_id") != FREEZE_ID:
        raise VerifyError(f"config.json freeze_id {cfg.get('freeze_id')!r} != {FREEZE_ID!r}")
    return cfg


def verify_release_dir(directory: Path, *, check_manifest: bool = True) -> dict:
    """Return the manifest after checking that every listed file hashes as recorded."""
    d = Path(directory)
    cfg = read_config(d)
    result = {"config": "ok", "files": {}}
    if check_manifest:
        man_path = d / "MANIFEST.json"
        if not man_path.exists():
            raise VerifyError("MANIFEST.json missing")
        man = json.loads(man_path.read_text())
        if man.get("freeze_id") != FREEZE_ID or man.get("tensor_tree_sha256") != TENSOR_TREE_SHA256:
            raise VerifyError("MANIFEST.json identity differs from juniper1-base-v1")
        for name, ent in man["files"].items():
            p = d / name
            if not p.exists():
                raise VerifyError(f"{name} missing")
            got = sha256_file(p)
            if got != ent["sha256"] or p.stat().st_size != ent["size"]:
                raise VerifyError(f"{name}: SHA-256/size differ from MANIFEST.json")
            result["files"][name] = got
        result["manifest"] = man
    if cfg.get("tensor_tree_sha256") != TENSOR_TREE_SHA256:
        raise VerifyError("config.json tensor_tree_sha256 differs from juniper1-base-v1")
    result["parameters_expected"] = FROZEN_PARAMETERS
    return result
