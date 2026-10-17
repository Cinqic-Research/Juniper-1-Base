"""Convert the frozen Juniper 1 Base checkpoint (model.pt) to safetensors, preserving tensors exactly.

Reads only. Verifies: model.pt SHA-256, tensor-tree digest, parameter count, config; writes
model.safetensors (fp32, exact tensors), config.json, and a manifest with SHA-256 of outputs.
"""

import hashlib
import json
import sys
import time
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

CKPT = Path(sys.argv[1])
OUT = Path(sys.argv[2])
FREEZE = Path(sys.argv[3])
rec = json.loads(FREEZE.read_text())
src = rec["source_checkpoint"]


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while b := f.read(16 << 20):
            h.update(b)
    return h.hexdigest()


def tensor_tree_sha(sd):
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


t0 = time.time()
assert sha256_file(CKPT / "model.pt") == src["files"]["model.pt"]["sha256"], "model.pt sha mismatch"
sd = torch.load(CKPT / "model.pt", map_location="cpu", weights_only=True)
assert isinstance(sd, dict) and all(isinstance(v, torch.Tensor) for v in sd.values())
digest = tensor_tree_sha(sd)
assert digest == rec["model"]["state_sha256"], f"tensor digest {digest}"
n = sum(v.numel() for v in sd.values())
assert n == rec["model"]["parameters"], n
dtypes = sorted({str(v.dtype) for v in sd.values()})
print("keys", len(sd), "params", n, "dtypes", dtypes, "digest ok")
OUT.mkdir(parents=True, exist_ok=True)
meta = {
    "format": "pt",
    "juniper1_freeze_id": rec["freeze_id"],
    "source_checkpoint": src["name"],
    "source_model_pt_sha256": src["files"]["model.pt"]["sha256"],
    "tensor_tree_sha256": digest,
    "dtype": "float32",
    "architecture": "j1-wide-20x768",
}
tensors = {k: v.contiguous() for k, v in sd.items()}
save_file(tensors, OUT / "model.safetensors", metadata=meta)
back = load_file(OUT / "model.safetensors")
assert set(back) == set(sd)
for k in sd:
    assert back[k].dtype == sd[k].dtype and torch.equal(back[k], sd[k]), k
assert tensor_tree_sha(back) == digest
cfg = dict(rec["model"]["config"])
config = {
    "model_type": "juniper1",
    "architecture": "j1-wide-20x768",
    "freeze_id": rec["freeze_id"],
    "parameters": n,
    "torch_dtype": "float32",
    "context_tokens_qualified": 2048,
    **cfg,
    "attention": {
        "type": "GQA",
        "n_head": cfg["n_head"],
        "n_kv_head": cfg["n_kv_head"],
        "head_dim": cfg["head_dim"],
        "causal": True,
    },
    "qk_norm_description": "rmsnorm_per_head: learned RMSNorm over head_dim on q and k, before RoPE",
    "rope": {
        "theta": cfg["rope_theta"],
        "convention": "rotate-half over the two halves of head_dim, applied after QK-Norm",
    },
    "norm": f"pre-RMSNorm (eps {cfg['norm_eps']:g}), computed in fp32",
    "ffn": f"SwiGLU: down(silu(gate(x)) * up(x)), d_ff {cfg['d_ff']}",
    "tie_embeddings": True,
    "bias": False,
    "init_std": cfg["init_std"],
    "residual_init": "output projections (attn.o, mlp.down) std/sqrt(2*n_layer)",
    "tokenizer": {
        "name": "cand5-j1tok-24k",
        "tokenizer_json_sha256": rec["tokenizer"]["sha256"],
        "vocab_size": 24576,
        "control_ids": "0-63",
        "byte_fallback_ids": "64-319",
        "learned_ids": "320-24575",
        "eos_token_id": 0,
        "pad_token_id": 0,
        "encode_special_tokens": True,
    },
    "state_dict_keys": sorted(sd),
    "tensor_tree_sha256": digest,
}
(OUT / "config.json").write_text(json.dumps(config, indent=1) + "\n")
man = {
    "schema": "juniper1.release-manifest.v1",
    "freeze_id": rec["freeze_id"],
    "freeze_record_sha256": rec.get("freeze_record_sha256"),
    "source": {
        "checkpoint": src["name"],
        "run_id": src["run_id"],
        "step": src["step"],
        "model_pt_sha256": src["files"]["model.pt"]["sha256"],
        "model_pt_size": src["files"]["model.pt"]["size"],
    },
    "tensor_tree_sha256": digest,
    "parameters": n,
    "files": {
        f: {"sha256": sha256_file(OUT / f), "size": (OUT / f).stat().st_size}
        for f in ("model.safetensors", "config.json")
    },
    "conversion": {
        "tool": "convert_to_safetensors.py",
        "torch": torch.__version__,
        "dtype": "float32 exact (no conversion)",
        "seconds": round(time.time() - t0, 1),
    },
}
(OUT / "MANIFEST.json").write_text(json.dumps(man, indent=1) + "\n")
print(json.dumps(man["files"], indent=1))
print("done in", round(time.time() - t0, 1), "s")
