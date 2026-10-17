"""Stage-3 recovery comparison: run A (uninterrupted) vs run B (SIGTERM, verified checkpoint, resume).

Bitwise when the final model and optimizer tensor-tree digests match and the batch fingerprints and
losses of every update after B's restore equal A's. Otherwise reports the maximum absolute parameter
difference (numerical continuation, not exact recovery).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from j1bench import ckpt as C


def _published(p: Path) -> bool:
    # Manifest self-hash and completion marker only: for runs whose weight files were not preserved.
    m = C.read_manifest(p)
    body = {k: v for k, v in (m or {}).items() if k != "manifest_sha256"}
    marker = (p / C.COMPLETE).read_text().strip() if (p / C.COMPLETE).is_file() else None
    return m is not None and m.get("manifest_sha256") == hashlib.sha256(C.canonical(body)).hexdigest() == marker


def final(run_dir: Path, weights: bool = True) -> Path:
    ok = [p for p in C.scan(run_dir) if (C.verify_checkpoint(p)[0] if weights else _published(p))]
    cands = [(C.read_manifest(p)["updates"], p) for p in ok]
    return max(cands)[1]


def log(run_dir: Path) -> list[dict]:
    return [json.loads(x) for x in (run_dir / "sft_log.jsonl").read_bytes().split(b"\n") if x]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-a", type=Path, required=True)
    ap.add_argument("--run-b", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--manifest-only",
        action="store_true",
        help="weights not preserved: compare the manifests' recorded digests (bitwise verdict only)",
    )
    a = ap.parse_args()
    fa, fb = final(a.run_a, not a.manifest_only), final(a.run_b, not a.manifest_only)
    ma, mb = C.read_manifest(fa), C.read_manifest(fb)
    lb = log(a.run_b)
    restores = [e for e in lb if e.get("event") == "restored"]
    stop = [e for e in lb if e.get("event") == "stopped_on_signal"]
    resumed_from = restores[-1]["update"] if restores else None
    la = {e["update"]: e for e in log(a.run_a) if "batch_fp" in e}
    after = [e for e in lb[lb.index(restores[-1]) :] if "batch_fp" in e] if restores else []
    batches_match = bool(after) and all(la.get(e["update"], {}).get("batch_fp") == e["batch_fp"] for e in after)
    losses_match = bool(after) and all(la.get(e["update"], {}).get("loss") == e["loss"] for e in after)
    fp_match = batches_match and losses_match
    same_length = ma["updates"] == mb["updates"]
    covered = bool(after) and resumed_from is not None and len(after) == mb["updates"] - resumed_from
    evidence = bool(stop) and bool(restores) and same_length and covered and batches_match
    bitwise = ma["state_sha256"] == mb["state_sha256"] and evidence and losses_match
    res = {
        "run_a_final": ma["name"],
        "run_b_final": mb["name"],
        "updates": [ma["updates"], mb["updates"]],
        "b_stopped_on_signal_at": stop[-1]["update"] if stop else None,
        "b_resumed_from_update": resumed_from,
        "updates_compared_after_resume": len(after),
        "fingerprints_and_losses_match": fp_match,
        "batch_fingerprints_match": batches_match,
        "interruption_evidence_complete": evidence,
        "state_sha256_a": ma["state_sha256"],
        "state_sha256_b": mb["state_sha256"],
        "bitwise": bitwise,
    }
    if a.manifest_only:
        res["evidence"] = "manifests and logs only (weights not preserved); file digests not re-hashed"
    elif not bitwise:
        sa = torch.load(fa / "model.pt", map_location="cpu", weights_only=True)
        sb = torch.load(fb / "model.pt", map_location="cpu", weights_only=True)
        res["max_abs_param_diff"] = max(float((sa[k].float() - sb[k].float()).abs().max()) for k in sa)
        # Numerical continuation only with a recorded stop, a restore, identical batches for every
        # replayed update and equal lengths; anything less is a recovery failure.
        res["numerical_only"] = evidence
    a.out.write_text(json.dumps(res, indent=1, sort_keys=True))
    print(json.dumps(res))
    return 0 if bitwise else 1


if __name__ == "__main__":
    sys.exit(main())
