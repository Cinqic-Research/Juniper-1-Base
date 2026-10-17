"""Exports a Juniper 1 model to GGUF using llama.cpp's ``qwen3`` architecture (inference qualification).

``j1-core`` maps exactly onto llama.cpp's qwen3 graph:
- pre-RMSNorm blocks
- GQA attention with per-head RMSNorm on q and k before NEOX-style (rotate-half) RoPE
- SwiGLU feed-forward
- final RMSNorm
- tied output embedding (qwen3 falls back to ``token_embd`` when ``output.weight`` is absent)

The model's single ``norm_eps`` is used for all RMSNorms, as in llama.cpp.

Tokenizer metadata uses llama.cpp's SentencePiece-style vocabulary:
- control IDs as CONTROL, ``<0xNN>`` as BYTE, learned tokens as NORMAL
- spaces in token text rewritten to U+2581
- learned-token scores ``-rank``

**Limitation (recorded, not hidden):** llama.cpp's SPM tokenizer does not apply Juniper's
pre-tokenizer split (single digits, punctuation runs), so its text tokenization can differ
from ``j1data.jtok``. Qualification therefore feeds **token IDs** to llama.cpp. A Juniper
pre-tokenizer must be registered in the runtime before text-in deployment (P17). Exports carry
``juniper.io_contract = "token-ids-only"`` so hosts can refuse every text path (A-02).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from j1bench.model import FAMILY, ModelConfig, Transformer

GGUF_PY = Path("/media/cinqic/Cinqic Storage/Juniper-GPT-OSS-Qualification/runtime/llama.cpp/gguf-py")


def load_model(args: argparse.Namespace) -> tuple[Transformer, ModelConfig, str]:
    if args.checkpoint:
        from j1bench.ckpt import verify_checkpoint
        from j1bench.train import tensor_tree_sha

        ck = Path(args.checkpoint)
        ok, why = verify_checkpoint(ck)
        if not ok:
            raise SystemExit(f"checkpoint failed verification: {why}")
        man = json.loads((ck / "manifest.json").read_text())
        cfg = (
            ModelConfig(**man["model_config"]) if "model_config" in man else ModelConfig(**man["run_contract"]["model"])
        )
        m = Transformer(cfg)
        sd = torch.load(ck / "model.pt", map_location="cpu", weights_only=True)
        if tensor_tree_sha(sd) != man["state_sha256"]["model"]:
            raise SystemExit("model state digest differs from the checkpoint manifest")
        m.load_state_dict(sd, strict=True)
        args.source_manifest_sha256 = man["manifest_sha256"]
        return m, cfg, f"checkpoint {man.get('name', ck.name)} (manifest {man['manifest_sha256'][:16]})"
    cfg = ModelConfig(vocab_size=args.vocab, max_seq_len=args.ctx, **FAMILY[args.preset])
    torch.manual_seed(args.seed)
    return Transformer(cfg), cfg, f"random init seed {args.seed} (latency/compatibility only)"


def vocab_entries(cfg: ModelConfig, tokenizer: Path | None) -> tuple[list[bytes], list[float], list[int]]:
    from gguf import TokenType

    if tokenizer is None:
        names = [f"<|c{i}|>" for i in range(64)] + [f"<0x{b:02X}>" for b in range(256)]
        names += [f"t{i}" for i in range(cfg.vocab_size - 320)]
    else:
        vocab = json.loads(tokenizer.read_text())["model"]["vocab"]
        names = [None] * len(vocab)
        for t, i in vocab.items():
            names[i] = t
        if len(names) != cfg.vocab_size or any(n is None for n in names):
            raise SystemExit("tokenizer vocabulary does not match the model")
    toks, scores, types = [], [], []
    for i, n in enumerate(names):
        if i < 64:
            types.append(TokenType.CONTROL)
        elif i < 320:
            types.append(TokenType.BYTE)
        else:
            types.append(TokenType.NORMAL)
        toks.append((n.replace(" ", "▁") if i >= 320 else n).encode("utf-8"))
        scores.append(0.0 if i < 320 else -float(i - 320))
    return toks, scores, [int(t) for t in types]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--preset", default="j1-core-28x640", choices=sorted(FAMILY))
    ap.add_argument("--vocab", type=int, default=32768)
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tokenizer", type=Path, default=None)
    ap.add_argument("--dtype", choices=["f16", "f32"], default="f16")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    sys.path.insert(0, str(GGUF_PY))
    import gguf

    m, cfg, origin = load_model(a)
    sd = {k: v.float().numpy() for k, v in m.state_dict().items()}
    w = gguf.GGUFWriter(str(a.out), "qwen3")
    w.add_name("juniper-1 (experimental export)")
    w.add_description(
        f"Juniper 1 {a.preset if not a.checkpoint else ''} exported for inference qualification; {origin}"
    )
    w.add_context_length(cfg.max_seq_len)
    w.add_embedding_length(cfg.d_model)
    w.add_block_count(cfg.n_layer)
    w.add_feed_forward_length(cfg.d_ff)
    w.add_head_count(cfg.n_head)
    w.add_head_count_kv(cfg.n_kv_head)
    w.add_key_length(cfg.head_dim)
    w.add_value_length(cfg.head_dim)
    w.add_rope_freq_base(cfg.rope_theta)
    w.add_layer_norm_rms_eps(cfg.norm_eps)
    w.add_file_type(gguf.LlamaFileType.MOSTLY_F16 if a.dtype == "f16" else gguf.LlamaFileType.ALL_F32)
    toks, scores, types = vocab_entries(cfg, a.tokenizer)
    w.add_tokenizer_model("llama")
    w.add_tokenizer_pre("default")
    w.add_token_list(toks)
    w.add_token_scores(scores)
    w.add_token_types(types)
    w.add_bos_token_id(0)
    w.add_eos_token_id(0)  # <|endoftext|>; chat turns end at <|eot|>
    w.add_eot_token_id(11)
    w.add_add_bos_token(False)
    w.add_add_eos_token(False)
    # A-02: llama.cpp's text tokenizer does not reproduce j1tok. These keys let hosts (Juniper App's
    # gguf_guard) refuse every text-mode path; only a host that tokenizes with j1tok may serve this file.
    w.add_string("juniper.io_contract", "token-ids-only")
    w.add_string("juniper.protocol", "juniper-chat-v1")
    if a.tokenizer is not None:
        import hashlib

        w.add_string("juniper.tokenizer_sha256", hashlib.sha256(a.tokenizer.read_bytes()).hexdigest())
    if getattr(a, "source_manifest_sha256", None):
        w.add_string("juniper.source_checkpoint_manifest_sha256", a.source_manifest_sha256)

    def put(name: str, arr: np.ndarray, keep_f32: bool = False) -> None:
        w.add_tensor(name, arr.astype(np.float32 if keep_f32 or a.dtype == "f32" else np.float16))

    put("token_embd.weight", sd["embed.weight"])
    put("output_norm.weight", sd["final_norm.weight"], keep_f32=True)
    names = {
        "attn_norm.weight": "attn_norm.weight",
        "attn.q.weight": "attn_q.weight",
        "attn.k.weight": "attn_k.weight",
        "attn.v.weight": "attn_v.weight",
        "attn.o.weight": "attn_output.weight",
        "attn.q_norm.weight": "attn_q_norm.weight",
        "attn.k_norm.weight": "attn_k_norm.weight",
        "mlp_norm.weight": "ffn_norm.weight",
        "mlp.gate.weight": "ffn_gate.weight",
        "mlp.up.weight": "ffn_up.weight",
        "mlp.down.weight": "ffn_down.weight",
    }
    for i in range(cfg.n_layer):
        for src, dst in names.items():
            arr = sd[f"blocks.{i}.{src}"]
            put(f"blk.{i}.{dst}", arr, keep_f32=arr.ndim == 1)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(json.dumps({"out": str(a.out), "bytes": a.out.stat().st_size, "params": m.num_params(), "origin": origin}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
