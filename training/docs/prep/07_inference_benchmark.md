# Inference benchmark: objectives L1–L4 on FLOWBOX (2026-10-09)

**Model:** `j1-core-28x640` at 32,768 vocabulary (148,315,264 parameters), **random-init weights**.
Decode and prefill cost do not depend on weight values, so these are valid latency and memory
measurements. Quality is evaluated separately, and quantization *quality* is re-measured on
trained weights. Export: `tools/export_gguf.py` writes llama.cpp's `qwen3` graph. Parity with
PyTorch: F16 PASS (top-1 agrees on 5/5 prompts up to 1,000 tokens, max |Δ log p| 0.004);
`results/prep/gguf/`.

**Host:** FLOWBOX, AMD Ryzen 7 5700G (8C/16T), 16 GB DDR4, NVIDIA RTX 2060 6 GB, Linux Mint. The
machine was otherwise idle (corpus jobs finished).

**Runtimes:**

- llama.cpp `llama-bench`: a CUDA build of the llama.cpp revision Juniper App ships with GPT-OSS,
  with 5 repetitions (3 for 2,048-token runs), reported as mean ± sd.
- PyTorch 2.8 eager with KV cache (`qual/infer_bench.py`), as a reference.

**CPU-only** runs hide the GPU (`CUDA_VISIBLE_DEVICES=""`). A first attempt with only `-ngl 0`
let llama.cpp offload prompt batches to the GPU and overstated CPU prefill; those files are kept as
`bench-cpu-*.json` and are **not** used below.

## Results (llama.cpp, 8 threads on CPU, all layers on GPU)

| Format | Size | CPU prefill 256 (tok/s) | CPU decode (tok/s) | GPU prefill 256 (tok/s) | GPU decode (tok/s) |
|---|---:|---:|---:|---:|---:|
| F16 | 297 MB | 2,211 ± 60 | 104.8 ± 0.2 | 21,693 ± 3,635 | 385.8 ± 9.5 |
| Q8_0 | 158 MB | 2,024 ± 16 | 187.8 ± 0.4 | 16,447 ± 2,150 | 493.5 ± 4.8 |
| Q4_K_M | 118 MB | 1,604 ± 7 | 240.0 ± 1.7 | 15,917 ± 1,857 | 515.3 ± 10.9 |

End-to-end prompt + generation (`-pg`), seconds for the whole request:

| Format | CPU 256+256 | CPU 2,048+256 | GPU 256+256 | GPU 2,048+256 |
|---|---:|---:|---:|---:|
| F16 | 2.70 | 4.85 | 0.68 | 0.81 |
| Q8_0 | 1.63 | 3.96 | 0.54 | 0.68 |
| Q4_K_M | 1.35 | 3.94 | 0.52 | 0.66 |

Peak process RSS at 2,048 context: F16 0.57 GB (CPU and GPU runs), Q4_K_M 0.39 GB (CPU) /
0.44 GB (GPU). GPU memory could not be read because NVML is broken until a reboot. The analytic
footprint is weights (≤ 297 MB) plus a 73 MB KV cache at 2,048 tokens plus compute buffers.

PyTorch eager reference (idle machine; batch 1; 256 prompt + 256 generated):

| Device | Precision | TTFT median | Decode tok/s | Peak memory |
|---|---|---:|---:|---|
| RTX 2060 | FP16 | 0.028 s | 37.9 | 314 MiB allocated |
| RTX 2060 | FP32 | 0.030 s | 50.3 | 614 MiB allocated |
| CPU, 8 threads | FP32 | 0.269 s | 23.9 (64 generated) | 1,122 MiB RSS |

PyTorch eager decode is bound by Python and kernel launches; FP16 is *slower* than FP32 here. It is
a correctness reference, not a deployment runtime.

## Objectives (provisional thresholds from `01_model_definition.md` §6)

| ID | Threshold | Measured (llama.cpp) | Status |
|---|---|---|---|
| L1 GPU decode | ≥ 30 tok/s | 386–515 tok/s (RTX 2060, 6 GB) | met on the 6 GB stand-in for the 8 GB class |
| L2 GPU TTFT, 256-token prompt | median ≤ 2.0 s | prefill 256 tokens ≈ 12–16 ms plus one decode step ≈ 2–3 ms | met |
| L3 CPU decode | ≥ 5 tok/s | 105 (F16) – 240 (Q4_K_M) tok/s | met |
| L4 stable at qualified context | no OOM at 2,048 + 256 | completes; peak RSS ≤ 0.57 GB | met (2,048); 4,096 not qualified (no 4K training) |

**Limits:**

- **Text runtime.** llama.cpp's tokenizer path needs a registered Juniper pre-tokenizer before
  text-in deployment. These runs used token IDs (P17 item).
- **Quantization quality.** Q4_K_M quality on *trained* weights is unmeasured. On random weights it
  drifts up to 0.18 nats and once changed the top-1 token (`results/prep/gguf/`). Q8_0 drifts
  ≤ 0.025 nats.
- **Thresholds stay provisional.** A model this small clears them by an order of magnitude; the
  owner may tighten them.
