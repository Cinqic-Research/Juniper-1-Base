"""Base-language retention: validation-split bits per byte for the frozen base or an SFT checkpoint.

Uses the registered layout of ``docs/audit_2026-10-12/base_evaluation_protocol_v1.md`` (the same
input hashes, 1,572 windows of 2,048 targets, fp32 or fp16 autocast) so SFT results are directly
comparable with the freeze record's full-split numbers. ``--mode fixed32`` scores the protocol's
32 fixed windows (fast diagnostic); ``--mode full`` scores every window. Read-only.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # before CUDA initializes: deterministic cuBLAS

from j1bench.data import TokenStream, full_split_eval, token_byte_lengths
from j1bench.frozen import load_frozen_base
from j1bench.generate import load_checkpoint_model
from tools.audit_base_candidates import (
    EXPECTED_DOCS,
    EXPECTED_SHARDS,
    EXPECTED_TOKENIZER,
    EXPECTED_VALIDATION,
    file_hash,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--freeze-record", type=Path, help="score the checkpoint as the frozen base (verified load)")
    ap.add_argument("--record-sha256", required=True)
    ap.add_argument("--data", type=Path, required=True, help="token_shards/j1-base-v0.2.2/cand5-j1tok-24k")
    ap.add_argument("--tokenizer", type=Path, required=True)
    ap.add_argument("--mode", choices=("fixed32", "full"), default="fixed32")
    ap.add_argument("--precision", choices=("fp32", "fp16"), default="fp32")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
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
        raise SystemExit(f"input identity mismatch: {hashes}")
    if a.freeze_record:
        model, rec = load_frozen_base(
            a.freeze_record, a.checkpoint, tokenizer_path=a.tokenizer, expected_record_sha256=a.record_sha256
        )
        ident = {"kind": "frozen_base", "freeze_id": rec["freeze_id"], "checkpoint": a.checkpoint.name}
    else:
        model, ident = load_checkpoint_model(a.checkpoint)
        man = json.loads((a.checkpoint / "manifest.json").read_text())
        if man.get("base_freeze_record_sha256") != a.record_sha256:
            raise SystemExit("checkpoint does not descend from the pinned freeze record")
        ident = {"kind": "sft_checkpoint", **ident}
    data = TokenStream(a.data, split="validation", seq_len=2048, batch_size=1, seed=1, shuffle=False, verify=True)
    table = full_split_eval(data)
    if table["identity"]["windows"] != 1572 or table["identity"]["scored_targets"] != 3218204:
        raise SystemExit("validation layout differs from the registered protocol")
    chosen = np.arange(1572) if a.mode == "full" else np.linspace(0, 1571, 32, dtype=np.int64)
    token_bytes = token_byte_lengths(a.tokenizer, 24576)
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval().to(dev)
    if dev == "cuda":
        model.set_gqa_kernel("expand")
    nll, n_tok, n_bytes = 0.0, 0, 0
    t0 = time.monotonic()
    for idx in chosen:
        x = torch.as_tensor(table["x"][idx : idx + 1], device=dev)
        y = torch.as_tensor(table["y"][idx : idx + 1], device=dev)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=dev == "cuda" and a.precision == "fp16"):
            logits = model(x)
        per = F.cross_entropy(logits.float().reshape(-1, 24576), y.reshape(-1), reduction="none", ignore_index=-100)
        keep = table["y"][idx].reshape(-1) != -100
        nll += float(per.double().sum())
        n_tok += int(keep.sum())
        n_bytes += int(token_bytes[table["y"][idx].reshape(-1)[keep]].sum())
    if not math.isfinite(nll):
        raise SystemExit("non-finite loss")
    out = {
        "schema": "j1.sft-retention.v1",
        "model": ident,
        "mode": a.mode,
        "precision": a.precision,
        "windows": len(chosen),
        "targets": n_tok,
        "bytes": n_bytes,
        "loss": nll / n_tok,
        "bpb": nll / math.log(2) / n_bytes,
        "input_hashes": hashes,
        "device": torch.cuda.get_device_name(0) if dev == "cuda" else "cpu",
        "seconds": round(time.monotonic() - t0, 1),
    }
    a.out.write_text(json.dumps(out, indent=1, sort_keys=True))
    print(json.dumps({k: out[k] for k in ("mode", "loss", "bpb", "seconds")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
