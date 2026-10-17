"""``juniper1`` command line: complete, verify, tokenize."""

from __future__ import annotations

import argparse
import json
import sys
import time

import torch


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="juniper1", description="Juniper 1 Base: text continuation and identity checks")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("complete", help="continue a text prompt (base model; no chat template)")
    c.add_argument("prompt", nargs="?", help="prompt text; omit to read stdin")
    c.add_argument("--model", default="CinqicX/juniper1-base", help="release directory or Hub repo id")
    c.add_argument("--revision", default=None, help="Hub revision (commit) to pin")
    c.add_argument("--max-new-tokens", type=int, default=64)
    c.add_argument("--temperature", type=float, default=0.0, help="0 = greedy (default)")
    c.add_argument("--top-k", type=int, default=0)
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--no-stop-at-eos", action="store_true")
    c.add_argument("--device", default="cpu", help="cpu (default) or cuda")
    c.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"])
    c.add_argument("--json", action="store_true", help="print a JSON record instead of plain text")
    c.add_argument("--no-verify-digest", action="store_true", help="skip the tensor digest check (faster start)")
    v = sub.add_parser("verify", help="check a release directory against the frozen identity")
    v.add_argument("--model", default="CinqicX/juniper1-base")
    v.add_argument("--revision", default=None)
    t = sub.add_parser("tokenize", help="show token IDs and pieces for a text")
    t.add_argument("text")
    t.add_argument("--tokenizer", default=None)
    a = ap.parse_args(argv)

    if a.cmd == "tokenize":
        from .tokenizer import Juniper1Tokenizer

        tok = Juniper1Tokenizer(a.tokenizer) if a.tokenizer else Juniper1Tokenizer()
        ids = tok.encode(a.text)
        print(json.dumps({"ids": ids, "pieces": tok.token_strings(ids), "roundtrip_ok": tok.decode(ids) == a.text}))
        return 0
    if a.cmd == "verify":
        from .load import load_model

        t0 = time.time()
        _model, _tok, info = load_model(a.model, revision=a.revision)
        info.pop("manifest", None)
        print(json.dumps({**info, "tensor_digest": "ok", "seconds": round(time.time() - t0, 1)}, indent=1))
        return 0
    from .generate import GenerationSettings, complete
    from .load import load_model

    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[a.dtype]
    model, tok, _ = load_model(a.model, revision=a.revision, device=a.device, dtype=dtype, verify_digest=not a.no_verify_digest)
    prompt = a.prompt if a.prompt is not None else sys.stdin.read()
    s = GenerationSettings(a.max_new_tokens, a.temperature, a.top_k, a.seed, not a.no_stop_at_eos)
    t0 = time.time()
    rec = complete(model, tok, prompt, s)
    rec["seconds"] = round(time.time() - t0, 2)
    if a.json:
        print(json.dumps(rec, ensure_ascii=False))
    else:
        sys.stdout.write(prompt + rec["completion"] + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
