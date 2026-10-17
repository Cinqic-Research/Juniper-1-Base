"""Builds the ``juniper1.base-freeze.v1`` record for an independently selected j1-base-r2 checkpoint.

Every identity is read from bytes on disk (checkpoint manifest and payload hashes, tensor-tree digest,
tokenizer, shard index) or from committed audit results; nothing is typed in by hand. The record is
self-hashed (``freeze_record_sha256``) and is what ``j1bench.frozen.load_frozen_base`` verifies. It
reads only; the checkpoint directory is never written.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from j1bench.ckpt import sha256_file, verify_checkpoint
from j1bench.frozen import DIGEST_KEY, FREEZE_SCHEMA, record_digest
from j1bench.model import ModelConfig, Transformer
from j1bench.train import tensor_tree_sha


def jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--tokenizer", type=Path, required=True)
    ap.add_argument("--shards", type=Path, required=True, help="token-shard directory (shards.json)")
    ap.add_argument("--results", type=Path, required=True, help="committed docs/audit_2026-10-12/results")
    ap.add_argument("--offsite-receipts", type=Path, required=True, help="original run offsite_receipts.jsonl")
    ap.add_argument("--train-commit", required=True)
    ap.add_argument("--data-commit", required=True)
    ap.add_argument("--restore-receipts", type=Path, required=True, help="independent restore drill receipts (jsonl)")
    ap.add_argument("--decision", choices=("BASE_FREEZE_APPROVED",), required=True)
    ap.add_argument("--decided-utc", required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    ck = a.checkpoint
    ok, why = verify_checkpoint(ck)
    if not ok:
        raise SystemExit(f"checkpoint invalid: {why}")
    man = json.loads((ck / "manifest.json").read_text())
    sd = torch.load(ck / "model.pt", map_location="cpu", weights_only=True)
    state = tensor_tree_sha(sd)
    if state != man["state_sha256"]["model"]:
        raise SystemExit("tensor-tree digest differs from the checkpoint manifest")
    cfg = man["run_contract"]["model"]
    model = Transformer(ModelConfig(**cfg))
    model.load_state_dict(sd, strict=True)
    tok_sha = sha256_file(a.tokenizer)
    if tok_sha != man["tokenizer_identity"]:
        raise SystemExit("tokenizer differs from the checkpoint's tokenizer identity")
    shards = json.loads((a.shards / "shards.json").read_text())

    def result(name: str) -> dict:
        return json.loads((a.results / name).read_text())

    expected_inputs = {"shards": sha256_file(a.shards / "shards.json"), "tokenizer": tok_sha}

    def scores(name: str) -> dict:
        r = result(name)
        if r["mode"] != "full_split" or any(r["input_hashes"][k] != v for k, v in expected_inputs.items()):
            raise SystemExit(f"{name}: not a full-split result on this checkpoint's data and tokenizer")
        mine = [c for c in r["candidates"] if c["name"] == ck.name]
        if len(mine) != 1 or mine[0]["model_state_sha256"] != state:
            raise SystemExit(f"{name}: does not score this checkpoint's exact weights")
        return {
            "numerics": r["numerics"],
            "device": r["device"],
            "targets": r["targets"],
            "by_candidate": {
                c["name"]: {"bpb": c["bpb"], "loss": c["loss"], "perplexity": c["perplexity"]} for c in r["candidates"]
            },
        }

    restores = [r for r in jsonl(a.restore_receipts) if r["name"] == ck.name]
    if {r["replica"] for r in restores} != {"hf", "drive"}:
        raise SystemExit("an independent restore receipt is required from both the HF and the Drive replica")
    for r in restores:
        if not (
            r["verify_checkpoint"][0]
            and r["manifest_sha256"] == man["manifest_sha256"]
            and r["model_state_sha256"] == state
        ):
            raise SystemExit(f"restore receipt does not reproduce the identity: {r['replica']}")
    offsite = [r for r in jsonl(a.offsite_receipts) if r.get("name") == ck.name and r.get("ok")]
    if {r["destination"] for r in offsite} != {"hf", "drive"}:
        raise SystemExit("training-time off-site receipts for both HF and Drive are required")

    rec = {
        "schema": FREEZE_SCHEMA,
        "freeze_id": "juniper1-base-v1",
        "decision": a.decision,
        "decided_utc": a.decided_utc,
        "decided_by": "independent review (Claude Opus 5.5, continuing the GPT-6 Sol assignment) under the owner instruction of 2026-10-12",  # noqa: E501
        "selection_report": "docs/audit_2026-10-12/base_checkpoint_selection_report.md",
        "selection_protocol": "docs/audit_2026-10-12/base_evaluation_protocol_v1.md (registered 305ea96)",
        "source_checkpoint": {
            "name": ck.name,
            "run_id": man["run_id"],
            "step": man["step"],
            "created_utc": man["created_utc"],
            "manifest_sha256": man["manifest_sha256"],
            "run_contract_sha256": man["run_contract_sha256"],
            "files": man["files"],
            "original_labels": {
                "stage": man.get("stage"),
                "tags": man.get("tags"),
                "non_production": man.get("non_production"),
            },
            "full_state": "model.pt + optim.pt (AdamW state) + trainer.pt (scheduler/RNG/data cursor); preserved unmodified",  # noqa: E501
        },
        "model": {
            "architecture": "j1-wide-20x768 dense decoder-only causal Transformer (pre-RMSNorm, per-head QK-Norm, RoPE, SwiGLU, GQA 12/6, tied embeddings, no biases)",  # noqa: E501
            "config": cfg,
            "parameters": model.num_params(),
            "state_sha256": state,
            "optimizer_state_sha256": man["state_sha256"]["optim"],
            "inference_weights": "model.pt inside the source checkpoint (no separate export is canonical)",
            "context_tokens_qualified": 2048,
        },
        "tokenizer": {
            "name": "j1tok-24k (cand5-j1tok-24k)",
            "sha256": tok_sha,
            "vocab_size": cfg["vocab_size"],
            "load_requirement": "j1data.jtok.load (sets encode_special_tokens; tokenizer.json alone does not)",
        },
        "protocol": {
            "name": "juniper-chat-v1",
            "source": f"Juniper-1-Data src/j1data/protocol.py @ {a.data_commit}",
            "trained_into_base": False,
        },
        "dataset": {
            "build": shards["build"],
            "build_identity": "89b08d5754c46dfd18ec239fdacb21ae899c1cf9d7606392401b5224c54bffc5",
            "accepted_manifest_sha256": shards["accepted_manifest_sha256"],
            "shards_json_sha256": sha256_file(a.shards / "shards.json"),
            "dataset_identity": man["dataset_identity"],
            "documents": {k: v["documents"] for k, v in shards["splits"].items()},
            "train_targets_consumed": man["accounting"]["exposure_tokens_useful"],
            "test_split_used_for_selection": False,
        },
        "training": {
            "run_id": man["run_id"],
            "optimizer_updates": man["step"],
            "epochs": 1,
            "precision_profile": man["run_contract"]["precision"],
            "compile": man["run_contract"]["compile"],
            "code_identity": man["code_identity"],
        },
        "evaluation": {
            "training_logged_eval_identity": man["eval_identity"],
            "independent_full_split_fp16": scores("full_fp16.json"),
            "independent_full_split_fp32": scores("full_fp32.json"),
            "results_sha256": {p.name: sha256_file(p) for p in sorted(a.results.iterdir()) if p.is_file()},
        },
        "recovery": {
            "flowbox_hdd": f"/media/cinqic/Cinqic Storage/juniper1-archive/fable-2026-10-10/ckpt/j1-base-r2/{ck.name}",
            "offsite_receipts_at_training": [
                {k: r.get(k) for k in ("destination", "remote", "repo", "path_prefix", "commit", "ok", "verified_utc")}
                for r in offsite
            ],
            "independent_restore_drills": restores,
        },
        "code_references": {"juniper_1_train": a.train_commit, "juniper_1_data": a.data_commit},
        "limitations": [
            "Base language model only: no instruction, chat, tool-use or safety training; not a product model.",
            "Qualified context is 2,048 tokens; 4,096 is unqualified.",
            "Validation split was observed during training; full-split bpb checks the checkpoint choice but is not an unbiased capability estimate. The test split remains unused.",  # noqa: E501
            "Objective diagnostics (MC, addition, Python, JSON) are small descriptive probes, not benchmarks.",
            "Text-in/text-out via the experimental GGUF (llama.cpp SPM vocabulary) does not reproduce j1tok tokenization; only token-ID inference is parity-qualified.",  # noqa: E501
            "The original manifest labels this artifact stage 'base-tranche-1' and non_production=true (launch-harness labels); this freeze record, not those labels, is the canonical identity.",  # noqa: E501
            "Weights are private; no public release is authorized.",
        ],
    }
    rec[DIGEST_KEY] = record_digest(rec)
    a.out.write_text(json.dumps(rec, indent=2, sort_keys=False) + "\n")
    print(a.out, rec[DIGEST_KEY])


if __name__ == "__main__":
    main()
