"""Checks that llama.cpp (llama-server, qwen3 graph) computes the same next-token distribution as PyTorch.

The prompt is given as **token IDs**, so no tokenizer is involved. Two quantities are compared
at several prompt lengths:

* top-1 token ID
* log-probabilities of llama.cpp's top-k tokens, recomputed in PyTorch FP32 from the same weights

Frozen tolerance for an F16 GGUF vs FP32 PyTorch: top-1 agreement on every prompt, and a maximum
absolute log-probability difference ≤ 0.05 nats. Quantized formats (Q8_0, Q4_K_M) are
reported, not gated.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from j1bench.model import FAMILY, ModelConfig, Transformer

BIN = Path("/media/cinqic/Cinqic Storage/AAA/juniper1/llama-build/bin")


def query(port: int, ids: list[int], k: int) -> dict:
    body = json.dumps(
        {
            "prompt": ids,
            "n_predict": 1,
            "n_probs": k,
            "temperature": 0.0,
            "cache_prompt": False,
            "post_sampling_probs": False,
        }
    ).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/completion", body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", type=Path, required=True)
    ap.add_argument("--preset", default="j1-core-28x640")
    ap.add_argument("--vocab", type=int, default=32768)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--port", type=int, default=18187)
    ap.add_argument("--ngl", type=int, default=0)
    ap.add_argument("--gate", action="store_true", help="apply the F16 tolerance")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    if a.checkpoint:
        ck = Path(a.checkpoint)
        man = json.loads((ck / "manifest.json").read_text())
        cfg = ModelConfig(**man["run_contract"]["model"])
        m = Transformer(cfg)
        m.load_state_dict(torch.load(ck / "model.pt", map_location="cpu", weights_only=True))
    else:
        cfg = ModelConfig(vocab_size=a.vocab, **FAMILY[a.preset])
        torch.manual_seed(a.seed)
        m = Transformer(cfg)
    m.eval()
    env = {"LD_LIBRARY_PATH": str(BIN), "PATH": "/usr/bin:/bin"}
    srv = subprocess.Popen(
        [
            str(BIN / "llama-server"),
            "-m",
            str(a.gguf),
            "--port",
            str(a.port),
            "-ngl",
            str(a.ngl),
            "-c",
            "2048",
            "-np",
            "1",
            "--no-webui",
        ],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        for _ in range(120):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{a.port}/health", timeout=2)
                break
            except OSError:
                if srv.poll() is not None:
                    raise SystemExit("llama-server exited: " + srv.stderr.read()[-2000:]) from None
                time.sleep(1)
        g = torch.Generator().manual_seed(123)
        rows, ok = [], True
        for n in (1, 7, 64, 300, 1000):
            ids = torch.randint(320, cfg.vocab_size, (n,), generator=g).tolist()
            r = query(a.port, ids, 10)
            probs = r["completion_probabilities"][0]["top_logprobs"]
            with torch.no_grad():
                lp = torch.log_softmax(m(torch.tensor([ids]))[0, -1].double(), -1)
            diffs = [abs(lp[p["id"]].item() - p["logprob"]) for p in probs]
            top_torch = int(lp.argmax())
            row = {
                "prompt_tokens": n,
                "top1_llama": probs[0]["id"],
                "top1_torch": top_torch,
                "top1_agree": probs[0]["id"] == top_torch,
                "max_abs_logprob_diff": round(max(diffs), 5),
            }
            rows.append(row)
            ok &= row["top1_agree"] and row["max_abs_logprob_diff"] <= 0.05
        res = {
            "gguf": str(a.gguf),
            "gguf_bytes": a.gguf.stat().st_size,
            "ngl": a.ngl,
            "rows": rows,
            "gate": "F16 vs FP32: top-1 agree and max |dlogp| <= 0.05" if a.gate else "reported only",
            "pass": bool(ok) if a.gate else None,
        }
    finally:
        srv.terminate()
        srv.wait(timeout=30)
    text = json.dumps(res, indent=1)
    if a.out:
        a.out.write_text(text + "\n")
    print(text)
    return 0 if (not a.gate or ok) else 1


if __name__ == "__main__":
    sys.exit(main())
