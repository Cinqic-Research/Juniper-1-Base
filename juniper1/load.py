"""Load Juniper 1 Base from a local release directory or from the Hugging Face Hub, verifying identity."""

from __future__ import annotations

from pathlib import Path

import torch
from safetensors.torch import load_file

from .model import FROZEN_CONFIG, FROZEN_PARAMETERS, ModelConfig, Transformer
from .tokenizer import Juniper1Tokenizer
from .verify import TENSOR_TREE_SHA256, VerifyError, tensor_tree_sha, verify_release_dir

HUB_REPO = "CinqicX/juniper1-base"
HUB_FILES = ["model.safetensors", "config.json", "MANIFEST.json", "tokenizer.json"]


def resolve(source: str | Path, *, revision: str | None = None) -> Path:
    """A local directory, or a Hub repo id (``owner/name``) downloaded with ``huggingface_hub``."""
    p = Path(source)
    if p.is_dir():
        return p
    if "/" in str(source) and not str(source).startswith((".", "/")):
        from huggingface_hub import snapshot_download

        return Path(snapshot_download(repo_id=str(source), revision=revision, allow_patterns=HUB_FILES))
    raise FileNotFoundError(f"{source} is neither a directory nor a Hub repo id")


def load_model(
    source: str | Path = HUB_REPO,
    *,
    revision: str | None = None,
    device: str = "cpu",
    dtype: torch.dtype | None = None,
    verify_digest: bool = True,
) -> tuple[Transformer, Juniper1Tokenizer, dict]:
    """Return ``(model.eval(), tokenizer, manifest)``.

    Verification: config.json equals the frozen configuration; every file in MANIFEST.json hashes as
    recorded; the loaded tensors reproduce the frozen tensor-tree digest (``verify_digest``); the
    parameter count is 142,771,456; the state dict loads strictly. ``dtype`` other than float32 is
    a derived numerical variant (bf16/fp16 change logits slightly); the shipped weights are float32.
    """
    d = resolve(source, revision=revision)
    info = verify_release_dir(d)
    sd = load_file(str(d / "model.safetensors"))
    if verify_digest and tensor_tree_sha(sd) != TENSOR_TREE_SHA256:
        raise VerifyError("loaded tensors differ from the frozen juniper1-base-v1 tensor digest")
    model = Transformer(ModelConfig(**FROZEN_CONFIG))
    model.load_state_dict(sd, strict=True)  # raises on missing or unexpected keys
    if model.num_params() != FROZEN_PARAMETERS:
        raise VerifyError(f"parameter count {model.num_params()} != {FROZEN_PARAMETERS}")
    tok_path = d / "tokenizer.json"
    tok = Juniper1Tokenizer(tok_path) if tok_path.exists() else Juniper1Tokenizer()
    if dtype is not None:
        model = model.to(dtype)
    model = model.to(device).eval()
    if device.startswith("cuda"):
        model.set_gqa_kernel("expand")  # portable kernel path; identical mathematics
    info["device"], info["dtype"] = device, str(dtype or torch.float32)
    return model, tok, info
