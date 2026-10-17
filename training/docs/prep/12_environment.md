# Reproducible environments (directive §13.3)

## A100 pod

| Item | Pinned identity | Evidence |
|---|---|---|
| Hardware | 1× NVIDIA A100-SXM4-80GB, Secure Cloud US-MD-1, pod `4i13u1p447t6e7` (GPU UUID recorded per session) | `results/env/nvidia-smi-q.txt` |
| Driver / CUDA | 580.126.16 / driver CUDA 13.0; runtime CUDA 12.8, cuDNN 9.10.2 | `results/env/env_inventory.txt` |
| Image | `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404` @ `sha256:0a360022e8de4375af99430f84e8b38951acc397252163a37ceac7204d01be35` | qualification report |
| Python stack | torch 2.8.0+cu128, triton 3.4.0, numpy 2.1.2 (from the image) | `results/env/pip-freeze.txt` |
| Overlay | `env/pod-overlay-requirements.txt` (pyarrow 25.0.1, pytest 8.4.2) installed in a `--system-site-packages` venv | this file |
| Code | `git archive` of a clean commit, with its SHA in `/root/j1train/REVISION`, deployed by `tools/deploy.sh` | deploy output |
| Runtime flags | `OMP_NUM_THREADS=12` (cgroup quota 13.6 CPUs); TF32 disabled in code; BF16 autocast; `torch.compile` default mode for the optimized profile; `CUBLAS_WORKSPACE_CONFIG=:4096:8` only under `--deterministic` | `j1bench/train.py` |

Recreate:

```bash
ssh j1pod 'python -m venv --system-site-packages /root/venv && /root/venv/bin/pip install -r /root/j1train/env/pod-overlay-requirements.txt'
tools/deploy.sh j1pod
ssh j1pod 'cd /root/j1train && /root/venv/bin/python -m pytest -q -p no:cacheprovider tests'
```

## FLOWBOX

| Item | Identity |
|---|---|
| Host | Ryzen 7 5700G, 16 GB DDR4 (14 GiB usable), RTX 2060 6 GB (sm_75), Linux Mint, kernel NVIDIA module 595.91.07 |
| Known defect | `nvidia-smi` / NVML fails ("Driver/library version mismatch": userspace 595.99 vs kernel 595.91.07). CUDA via PyTorch works. A reboot (owner action) should realign them |
| Train venv | `juniper1-a100/venv`, exact lock `env/flowbox-venv.lock.txt` (torch 2.8.0+cu128, numpy 2.1.2, triton 3.4.0 — same as the pod; tokenizers 0.23.2, huggingface_hub 1.33.0, hf-xet 1.7.0, pyarrow 25.0.1) |
| Data venv | `juniper1-data/venv` (Juniper-1-Data `pyproject.toml`; Python 3.12.3) |
| Clock | systemd-timesyncd, NTP synchronized (offset ≈ 0.1–1 ms, root distance ≈ 6–8 ms at check time) |

Recreate the Train venv:

```bash
python3 -m venv venv && venv/bin/pip install "torch==2.8.0" --index-url https://download.pytorch.org/whl/cu128
venv/bin/pip install -r j1train/env/flowbox-venv.lock.txt --extra-index-url https://download.pytorch.org/whl/cu128
```

FLOWBOX is not an interchangeable mid-run resume target for A100 training (spec §1). It runs
evaluation, inference and latency benchmarks, verification, and archiving.
