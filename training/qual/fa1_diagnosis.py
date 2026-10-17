"""FA-1 diagnosis: is the flash-SDPA backward prefix-gradient difference causal leakage or accumulation noise?

``sdpa_correctness.py`` perturbs q/k/v after position p and compares prefix gradients of two separate backward
passes. On A100/4090 the flash kernel showed a nonzero difference (1.2e-4 to 4.9e-4 absolute, bf16) in the default
mode and exactly 0 under ``torch.use_deterministic_algorithms(True)``. Two hypotheses:

* H_noise: flash backward accumulates dQ (and dK/dV across query blocks) with atomics in an unspecified order, so two
  backward passes differ even on identical inputs; the prefix is not influenced by the suffix.
* H_leak: suffix values leak into prefix gradients (a causal-mask defect).

Registered decision (written before this script first ran on an A100; docs/prep/18 §9):
FA-1 is EXPLAINED as accumulation-order noise iff, for flash on the j1 attention geometry,
  (a) deterministic mode: prefix-gradient difference under suffix perturbation is exactly 0 in every repeat, and
  (b) default mode: repeats on IDENTICAL inputs give a nonzero prefix-gradient difference whose maximum over repeats is
      at least 1/4 of the maximum perturbed-input difference (same magnitude: the perturbation adds nothing
      measurable beyond run-to-run noise), and
  (c) every prefix-gradient difference in (b) is below 2 bf16 ulps of the largest prefix-gradient magnitude.
Otherwise FA-1 stays OPEN and training must use a backend that passes the causal backward check exactly.
Both geometries (core 10q/5kv heads, wide 12q/6kv heads; head_dim 64; T 2048) are tested.
"""

import json
import os
import sys

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

REPEATS = 5
P = 1000


def run(hq: int, hkv: int, deterministic: bool, backend) -> dict:
    torch.use_deterministic_algorithms(deterministic)
    torch.manual_seed(0)
    b, t, d = 2, 2048, 64
    base = [torch.randn(b, h, t, d, device="cuda", dtype=torch.bfloat16) for h in (hq, hkv, hkv)]
    go = torch.randn(b, hq, t, d, device="cuda", dtype=torch.bfloat16)
    go[:, :, P + 1 :] = 0
    pert = [x.clone() for x in base]
    for x in pert:
        x[:, :, P + 1 :] = torch.randn_like(x[:, :, P + 1 :])

    def grads(inp):
        q, k, v = [x.clone().requires_grad_() for x in inp]
        with sdpa_kernel(backend):
            o = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
            o.backward(go)
        return [g[:, :, : P + 1].float() for g in (q.grad, k.grad, v.grad)], o[:, :, : P + 1].float()

    ref_g, ref_o = grads(base)
    same, perturbed, fwd_pert = [], [], []
    for _ in range(REPEATS):
        g1, _ = grads(base)
        same.append(max((a - r).abs().max().item() for a, r in zip(g1, ref_g, strict=True)))
        g2, o2 = grads(pert)
        perturbed.append(max((a - r).abs().max().item() for a, r in zip(g2, ref_g, strict=True)))
        fwd_pert.append((o2 - ref_o).abs().max().item())
    gmax = max(g.abs().max().item() for g in ref_g)
    ulp = 2.0 ** (int(torch.tensor(gmax).log2().floor().item()) - 7)  # bf16 has 8 significant bits
    return {
        "same_input_prefix_grad_maxdiff": same,
        "perturbed_prefix_grad_maxdiff": perturbed,
        "perturbed_prefix_fwd_maxdiff": fwd_pert,
        "prefix_grad_absmax": gmax,
        "bf16_ulp_at_absmax": ulp,
    }


def main() -> int:
    out: dict = {"torch": torch.__version__, "cuda": torch.version.cuda, "device": torch.cuda.get_device_name()}
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    verdicts = []
    for name, hq, hkv in (("core-10q5kv", 10, 5), ("wide-12q6kv", 12, 6)):
        rec = {}
        for be_name, be in (("flash", SDPBackend.FLASH_ATTENTION), ("cudnn", SDPBackend.CUDNN_ATTENTION)):
            try:
                rec[be_name] = {"default": run(hq, hkv, False, be), "deterministic": run(hq, hkv, True, be)}
            except RuntimeError as e:
                rec[be_name] = {"status": "unavailable", "error": str(e).splitlines()[0][:200]}
        f = rec.get("flash", {})
        if "default" in f:
            det, dft = f["deterministic"], f["default"]
            a = all(x == 0.0 for x in det["perturbed_prefix_grad_maxdiff"]) and all(
                x == 0.0 for x in det["perturbed_prefix_fwd_maxdiff"]
            )
            b = max(dft["same_input_prefix_grad_maxdiff"]) > 0 and max(dft["same_input_prefix_grad_maxdiff"]) >= 0.25 * max(
                dft["perturbed_prefix_grad_maxdiff"]
            )
            c = max(dft["perturbed_prefix_grad_maxdiff"] + dft["same_input_prefix_grad_maxdiff"]) < 2 * dft[
                "bf16_ulp_at_absmax"
            ]
            fwd0 = all(x == 0.0 for x in dft["perturbed_prefix_fwd_maxdiff"])
            rec["criteria"] = {"a_det_exact": a, "b_noise_same_magnitude": b, "c_below_2_ulp": c, "fwd_exact": fwd0}
            verdicts.append(a and b and c and fwd0)
        else:
            verdicts.append(False)
        out[name] = rec
    out["verdict"] = "EXPLAINED_ACCUMULATION_NOISE" if all(verdicts) else "OPEN"
    print(json.dumps(out, indent=1))
    return 0 if all(verdicts) else 1


if __name__ == "__main__":
    sys.exit(main())
