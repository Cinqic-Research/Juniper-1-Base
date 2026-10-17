"""Read-only, fixed-window audit of authentic j1-base-r2 checkpoints.

The selection protocol is docs/audit_2026-10-12/base_evaluation_protocol_v1.md.
This does not train or alter checkpoint files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from j1bench.ckpt import verify_checkpoint
from j1bench.data import TokenStream, full_split_eval, token_byte_lengths
from j1bench.model import ModelConfig, Transformer, analytic_params
from j1bench.train import tensor_tree_sha

NAMES = (
    "ckpt-000050000-e4775a69",
    "ckpt-000052000-6b13028e",
    "ckpt-000052034-336e5967",
)
EXPECTED_SHARDS = "1cf5e2320ad35e930f84ca0edbf5a9d188a2f524702dc4623e492dab043e3aef"
EXPECTED_TOKENIZER = "176a89e2197c24d3f024fbb60ce6f346635e7cba912b3579d93b37d486c9adf1"
EXPECTED_VALIDATION = "03efc04fc4c1bf1b1960f950cf98de229a9494dca786bb9472410bc9d44e13df"
EXPECTED_DOCS = "42e057d7771fc1223c9f59895edb5d4d2bb8cbfacdca5dabe6a91ec37482f787"


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while b := f.read(16 << 20):
            h.update(b)
    return h.hexdigest()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoints", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--tokenizer", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--smoke", action="store_true", help="run only the first window as a device check")
    mode.add_argument("--full", action="store_true", help="score all full-split validation windows")
    p.add_argument("--names", default=",".join(NAMES), help="comma-separated subset of the registered candidates")
    p.add_argument(
        "--precision", choices=("fp16", "fp32"), default="fp16", help="CUDA numerics (fp16 autocast or full fp32)"
    )
    a = p.parse_args()
    names = a.names.split(",")
    if not names or any(n not in NAMES for n in names) or len(set(names)) != len(names):
        raise ValueError(f"--names must be distinct registered candidates: {names}")

    hashes = {
        "shards": file_hash(a.data / "shards.json"),
        "tokenizer": file_hash(a.tokenizer),
        "validation": file_hash(a.data / "validation-tokens-00000.bin"),
        "docs": file_hash(a.data / "validation-docs.parquet"),
    }
    if hashes != {
        "shards": EXPECTED_SHARDS,
        "tokenizer": EXPECTED_TOKENIZER,
        "validation": EXPECTED_VALIDATION,
        "docs": EXPECTED_DOCS,
    }:
        raise ValueError(f"input identity mismatch: {hashes}")

    # The same dataset producer is used here for window layout. Byte/target and
    # source totals are recomputed below independently of the training log.
    data = TokenStream(a.data, split="validation", seq_len=2048, batch_size=1, seed=1, shuffle=False, verify=True)
    table = full_split_eval(data)
    if table["identity"]["windows"] != 1572 or table["identity"]["scored_targets"] != 3218204:
        raise ValueError("validation layout differs from the registered protocol")
    chosen = (
        np.array([0], dtype=np.int64)
        if a.smoke
        else np.arange(1572, dtype=np.int64)
        if a.full
        else np.linspace(0, 1571, 32, dtype=np.int64)
    )
    if len(set(chosen.tolist())) != len(chosen):
        raise AssertionError("duplicate selected windows")
    y_all = table["y"][chosen]
    src_all = table["src"][chosen]
    token_bytes = token_byte_lengths(a.tokenizer, 24576)
    valid = y_all != -100
    source_names = table["source_names"]
    source_tokens = np.bincount(src_all[valid], minlength=len(source_names))
    source_bytes = np.bincount(src_all[valid], weights=token_bytes[y_all[valid]], minlength=len(source_names)).astype(
        int
    )
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    device = "cuda" if torch.cuda.is_available() else "cpu"
    autocast = device == "cuda" and a.precision == "fp16"
    mode = "cuda_autocast_fp16" if autocast else f"{device}_fp32"

    out = {
        "protocol": "base_evaluation_protocol_v1",
        "mode": "smoke" if a.smoke else "full_split" if a.full else "fixed_32_window_sample",
        "input_hashes": hashes,
        "device": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
        "torch": torch.__version__,
        "numerics": mode,
        "gqa_kernel": "expand" if device == "cuda" else "native",
        "windows": chosen.tolist(),
        "targets": int(valid.sum()),
        "source_tokens": {n: int(v) for n, v in zip(source_names, source_tokens, strict=True)},
        "source_bytes": {n: int(v) for n, v in zip(source_names, source_bytes, strict=True)},
        "candidates": [],
    }

    for name in names:
        path = a.checkpoints / name
        t0 = time.monotonic()
        ok, why = verify_checkpoint(path)
        if not ok:
            raise ValueError(f"invalid checkpoint {name}: {why}")
        manifest = json.loads((path / "manifest.json").read_text())
        cfg = ModelConfig(**manifest["run_contract"]["model"])
        if analytic_params(cfg) != 142771456:
            raise ValueError(f"unexpected model shape: {name}")
        sd = torch.load(path / "model.pt", map_location="cpu", weights_only=True)
        state_hash = tensor_tree_sha(sd)
        if state_hash != manifest["state_sha256"]["model"]:
            raise ValueError(f"state digest mismatch: {name}")
        model = Transformer(cfg)
        model.load_state_dict(sd, strict=True)
        if model.num_params() != 142771456:
            raise ValueError(f"parameter count mismatch: {name}")
        del sd
        model.eval().to(device)
        if device == "cuda":
            model.set_gqa_kernel("expand")
        sums = np.zeros(len(source_names), dtype=np.float64)
        counts = np.zeros(len(source_names), dtype=np.int64)
        nonfinite = 0
        for idx in chosen:
            x = torch.as_tensor(table["x"][idx : idx + 1], device=device)
            y = torch.as_tensor(table["y"][idx : idx + 1], device=device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=autocast):
                logits = model(x)
            nonfinite += int((~torch.isfinite(logits)).sum().item())
            per_tok = F.cross_entropy(
                logits.float().reshape(-1, cfg.vocab_size), y.reshape(-1), reduction="none", ignore_index=-100
            )
            nll = per_tok.double().cpu().numpy().reshape(-1)
            labels = table["y"][idx].reshape(-1)
            src = table["src"][idx].reshape(-1)
            keep = labels != -100
            np.add.at(sums, src[keep], nll[keep])
            np.add.at(counts, src[keep], 1)
            del x, y, logits, per_tok
        if nonfinite or not np.isfinite(sums).all():
            raise ValueError(f"nonfinite inference: {name}, {nonfinite} logits")
        if not np.array_equal(counts, source_tokens):
            raise AssertionError("source count mismatch")
        candidate = {
            "name": name,
            "step": manifest["step"],
            "manifest_sha256": manifest["manifest_sha256"],
            "model_file_sha256": manifest["files"]["model.pt"]["sha256"],
            "model_state_sha256": state_hash,
            "loss": float(sums.sum() / counts.sum()),
            "perplexity": float(math.exp(sums.sum() / counts.sum())),
            "bpb": float(sums.sum() / math.log(2) / source_bytes.sum()),
            "nll_sum": float(sums.sum()),
            "by_source": {
                n: {"tokens": int(counts[i]), "bytes": int(source_bytes[i]), "nll_sum": float(sums[i])}
                for i, n in enumerate(source_names)
                if counts[i]
            },
            "elapsed_s": round(time.monotonic() - t0, 2),
            "nonfinite_logits": nonfinite,
        }
        out["candidates"].append(candidate)
        print(name, candidate["loss"], candidate["bpb"], candidate["elapsed_s"], flush=True)
        a.output.write_text(json.dumps(out, indent=2, allow_nan=False) + "\n")
        del model
        if device == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
