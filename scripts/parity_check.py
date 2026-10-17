"""Reference (Juniper-1-Train j1bench.frozen loader on model.pt) versus release export (juniper1 package on
model.safetensors): logits and greedy continuations on fixed inputs. Writes a JSON record.

Usage: PYTHONPATH=<Juniper-1-Train checkout> python parity_check.py <freeze.json> <ckpt dir> <tokenizer.json> <release dir> <out.json>
"""

import json
import platform
import sys
from pathlib import Path

import torch

freeze, ckpt, tokp, rel, out = map(Path, sys.argv[1:6])
from j1bench.frozen import load_frozen_base  # reference

from juniper1.generate import GenerationSettings, continue_ids
from juniper1.load import load_model

rec = json.loads(freeze.read_text())
ref, _ = load_frozen_base(freeze, ckpt, tokenizer_path=tokp, expected_record_sha256=rec["freeze_record_sha256"])
exp, tok, info = load_model(rel)
PROMPTS = [
    "The Pythagorean theorem states that",
    'def fibonacci(n):\n    """Return the first n Fibonacci numbers."""\n',
    "Photosynthesis is the process by which",
    "Q: What is the capital of France?\nA:",
    "∫₀¹ x² dx =",
]
result = {
    "torch": torch.__version__,
    "platform": platform.platform(),
    "release_files": info.get("files"),
    "tolerance": {"logits_abs_max": 0.0, "greedy_ids": "identical"},
    "prompts": [],
}
torch.manual_seed(0)
worst = 0.0
for p in PROMPTS:
    ids = tok.encode(p)
    x = torch.tensor([ids])
    with torch.no_grad():
        lr = ref(x)[0]
        le = exp(x)[0]
    d = float((lr - le).abs().max())
    worst = max(worst, d)
    gr, _ = continue_ids(ref, ids, GenerationSettings(max_new_tokens=48))
    ge, reason = continue_ids(exp, ids, GenerationSettings(max_new_tokens=48))
    result["prompts"].append(
        {
            "prompt": p,
            "prompt_tokens": len(ids),
            "logits_abs_max_diff": d,
            "greedy_identical": gr == ge,
            "generated_ids": ge,
            "stop": reason,
            "completion": tok.decode_text(ge),
        }
    )
result["logits_abs_max_diff_overall"] = worst
result["all_greedy_identical"] = all(r["greedy_identical"] for r in result["prompts"])
if torch.cuda.is_available():
    dev = torch.device("cuda")
    expc = exp.to(dev)
    expc.set_gqa_kernel("expand")
    cuda = []
    for r in result["prompts"]:
        x = torch.tensor([tok.encode(r["prompt"])], device=dev)
        with torch.no_grad():
            lc = expc(x)[0].float().cpu()
            lcpu = exp.to("cpu")(x.cpu())[0]
        g, _ = continue_ids(expc, tok.encode(r["prompt"]), GenerationSettings(max_new_tokens=48))
        cuda.append(
            {
                "logits_abs_max_diff_cuda_vs_cpu_fp32": float((lc - lcpu).abs().max()),
                "greedy_identical_to_cpu": g == r["generated_ids"],
            }
        )
        expc = exp.to(dev)
    result["cuda_fp32_vs_cpu_fp32"] = {"device": torch.cuda.get_device_name(0), "per_prompt": cuda}
out.write_text(json.dumps(result, indent=1, ensure_ascii=False) + "\n")
print(json.dumps({k: v for k, v in result.items() if k != "prompts"}, indent=1))
for r in result["prompts"]:
    print("---", repr(r["prompt"])[:60], "| diff", r["logits_abs_max_diff"], "| same greedy", r["greedy_identical"])
    print(r["completion"][:300].replace("\n", "\\n"))
