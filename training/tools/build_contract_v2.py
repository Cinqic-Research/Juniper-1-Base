# ruff: noqa: E501  (contract text: long literal lines are intended)
"""Builds docs/contract/training_contract_v2.json (successor of v1) from committed records; nothing is hand-typed.

The model is instantiated from the selected preset with the frozen vocabulary to record the exact parameter count
and the actual weight-decay groups (parameter names), so prose ("matrices only") is checked against the code.
Every identity comes from a committed file whose SHA-256 is recorded; the lock (``training_contract_v2.lock.json``)
names the executable commits, and the evidence index refers to the lock (no self-referential hashes).

Status (review T-18) is derived, never chosen: ``FROZEN`` only if the readiness matrix passed in is bound to this
manifest (``bindings.manifest_sha256``) and to the committed gate file (``source_sha256``), says
``BASE_PRETRAINING_LAUNCH.ready == true``, and the compiled profile is either equivalence-PASS or not used
(``execution_profile.compile == false``); otherwise ``FROZEN_LAUNCH_BLOCKED``. An explicit ``--status FROZEN`` that
the inputs do not support is refused. Build identities must agree across the recount, the manifest and the lineage
record. ``known_limitations`` come from ``--limitations`` (docs/fable_2026-10-10/contract_limitations.json).

Usage: build_contract_v2.py --train-root . --data-root DATA_CHECKOUT --manifest docs/autonomous_2026-10-10/launch_manifest.json
       --readiness-matrix MATRIX --limitations docs/fable_2026-10-10/contract_limitations.json
       [--status FROZEN|FROZEN_LAUNCH_BLOCKED] --out docs/contract/training_contract_v2.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from j1bench.model import FAMILY, ModelConfig, Transformer

D10 = "docs/autonomous_2026-10-10"


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load(p: Path) -> dict:
    return json.loads(p.read_text())


def head(root: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()


def ref(root: Path, rel: str) -> dict:
    p = root / rel
    return {"path": rel, "sha256": sha(p)} if p.exists() else {"path": rel, "missing": True}


def model_record(preset: str, vocab: int) -> dict:
    cfg = ModelConfig(**{**FAMILY[preset], "vocab_size": vocab})
    m = Transformer(cfg)
    groups = m.param_groups(0.1, "matrices")
    names = {id(p): n for n, p in m.named_parameters()}
    decay = sorted(names[id(p)] for p in groups[0]["params"])
    no_decay = sorted(names[id(p)] for p in groups[1]["params"])
    nonemb = sum(p.numel() for n, p in m.named_parameters() if not n.startswith("embed."))
    return {
        "preset": preset,
        "config": cfg.to_dict(),
        "n_params": m.num_params(),
        "n_params_non_embedding": nonemb,
        "attention": f"GQA {cfg.n_head} query / {cfg.n_kv_head} KV heads, head_dim {cfg.head_dim}, causal, QK-Norm {cfg.qk_norm} before RoPE (theta {cfg.rope_theta})",
        "embeddings": "tied input/output embedding" if cfg.tie_embeddings else "untied",
        "norm": f"RMSNorm (eps {cfg.norm_eps}), computed in fp32",
        "bias": "no biases",
        "init": f"normal std {cfg.init_std}",
        "weight_decay": {
            "policy": "matrices: 2-D projection weights decay 0.1; 1-D parameters and the tied embedding are excluded",
            "decayed": {
                "count": len(decay),
                "elements": sum(p.numel() for p in groups[0]["params"]),
                "names_sha256": hashlib.sha256("\n".join(decay).encode()).hexdigest(),
                "examples": decay[:4],
            },
            "excluded": {
                "count": len(no_decay),
                "elements": sum(p.numel() for p in groups[1]["params"]),
                "names": [n for n in no_decay if not n.startswith("blocks.")]
                + [f"blocks.*.{x}" for x in sorted({n.split(".", 2)[2] for n in no_decay if n.startswith("blocks.")})],
            },
        },
    }


def derive_status(matrix: dict, man: dict, ce: dict, *, manifest_sha256: str, gates_sha256: str) -> tuple[str, dict]:
    """``FROZEN`` only when the readiness matrix is this manifest's, fresh against the committed gate file, ready for
    the base launch, and the compiled profile is PASS or unused; otherwise ``FROZEN_LAUNCH_BLOCKED``."""
    base = (matrix.get("stages") or {}).get("BASE_PRETRAINING_LAUNCH") or {}
    binds = matrix.get("bindings") or {}
    basis = {
        "matrix_valid": matrix.get("valid") is True,
        "matrix_bound_to_manifest": binds.get("manifest_sha256") == manifest_sha256,
        "matrix_source_is_committed_gates_file": matrix.get("source_sha256") == gates_sha256,
        "base_launch_ready": base.get("ready") is True,
        "base_launch_blocking": base.get("blocking"),
        "compile_equivalence_status": ce.get("status"),
        "execution_profile_compile": man["execution_profile"]["compile"],
        "profile_admissible": ce.get("status") == "PASS" or man["execution_profile"]["compile"] is False,
    }
    frozen = all(
        basis[k] for k in ("matrix_valid", "matrix_bound_to_manifest", "matrix_source_is_committed_gates_file",
                           "base_launch_ready", "profile_admissible")
    )  # fmt: skip
    return ("FROZEN" if frozen else "FROZEN_LAUNCH_BLOCKED"), basis


def check_build_identities(rc: dict, man: dict, lin: dict) -> dict:
    """The recount, the manifest and the lineage record must name the same corpus build (review T-18)."""
    ids = {
        "build": (lin.get("build"), man["data"]["build"]),
        "build_identity": (rc["inputs"]["build_identity"], lin.get("build_identity")),
        "accepted_manifest_sha256": (
            rc["inputs"]["accepted_manifest_sha256"],
            man["data"]["accepted_manifest_sha256"],
            lin.get("manifest_sha256"),
        ),
        "shards_json_sha256": (rc["inputs"].get("shards_json_sha256"), man["data"]["shards_json_sha256"]),
        "tokenizer_json_sha256": (rc["inputs"].get("tokenizer_json_sha256"), man["tokenizer"]["sha256"]),
    }
    bad = {k: v for k, v in ids.items() if len(set(v)) != 1 or v[0] in (None, "")}
    if bad:
        raise SystemExit(f"build identities disagree across recount / manifest / lineage: {bad}")
    return {k: v[0] for k, v in ids.items()}


def build(a) -> tuple[dict, dict, dict]:
    T, Dd = a.train_root, a.data_root
    man = load(a.manifest)
    matrix = load(a.readiness_matrix)
    sel = load(T / D10 / "selection_18b/select_architecture_v2.json")
    rc = load(Dd / "reports/j1-base-v0.2.2/full_recount_cand5-j1tok-24k.json")
    lin = load(Dd / "reports/j1-base-v0.2.2/successor_lineage.json")
    ce = load(T / D10 / "compile_equivalence.json")
    host = load(T / D10 / "evidence/host_pncbyxaf2dt2jl/host_qualification.json")
    pol = load(T / D10 / "launch_policy.json")
    bud = load(T / D10 / "budget_contract.json")
    limitations = load(a.limitations)
    if not isinstance(limitations.get("limitations"), list) or not all(
        isinstance(x, str) and x for x in limitations["limitations"]
    ):
        raise SystemExit(f"{a.limitations}: 'limitations' must be a non-empty list of strings")
    gates = a.gates_file or (T / D10 / "readiness_gates_v2.json")
    status, basis = derive_status(matrix, man, ce, manifest_sha256=sha(a.manifest), gates_sha256=sha(gates))
    if a.status and a.status != status:
        if a.status == "FROZEN":
            raise SystemExit(f"--status FROZEN refused: the inputs derive {status} ({basis})")
        status = a.status  # an explicit downgrade is always admissible
    identities = check_build_identities(rc, man, lin)
    model = model_record(man["model"]["preset"], man["tokenizer"]["total_vocab"])
    if model["n_params"] != man["model"]["n_params"]:
        raise SystemExit(f"instantiated parameter count {model['n_params']} != manifest {man['model']['n_params']}")
    contract = {
        "schema": "j1.training-contract.v2",
        "supersedes": "docs/contract/training_contract_v1.json (PROPOSED_FOR_OWNER_FREEZE, 2026-10-10)",
        "status": status,
        "status_basis": {
            **basis,
            "readiness_matrix": ref(T, str(a.readiness_matrix)),
            "readiness_gates": ref(T, str(gates)),
            "rule": "FROZEN only if the matrix is bound to this manifest and the committed gate file, "
            "BASE_PRETRAINING_LAUNCH is ready, and the compiled profile is PASS or unused",
        },
        "decision_record": "engineer-selected under the owner work order of 2026-10-10 (delegated selection and freeze); no owner signature is implied",
        "authority": ref(T, f"{D10}/authority.json"),
        "code": {
            "train_head_at_build": head(T),
            "data_head_at_build": head(Dd),
            "deployed_train_commit": man["code"]["train_commit"],
            "deploy_tar_sha256": man["code"]["deploy_tar_sha256"],
        },
        "corpus": {
            "build": identities["build"],
            "build_identity": identities["build_identity"],
            "accepted_manifest_sha256": identities["accepted_manifest_sha256"],
            "identity_sources": "equal across the recount, the launch manifest and the successor lineage (checked)",
            "lineage": lin,
            "source_permissions": ref(Dd, "reports/j1-base-v0.2.2/source_permissions.json"),
            "exclusions": {
                "union_list": ref(Dd, "reports/j1-base-v0.2.2/excluded_doc_ids_union.json"),
                "f9": ref(Dd, "reports/contamination_f9/summary_f9.json"),
                "internal_sources": ref(Dd, "docs/source_exclusion_internal_2026-10-10.md"),
            },
            "model_ready_shards": {
                "shards_json_sha256": man["data"]["shards_json_sha256"],
                "train_stream_tokens": man["data"]["train_stream_tokens"],
                "train_documents": man["data"]["train_documents"],
            },
            "independent_recount": {
                "record": ref(Dd, "reports/j1-base-v0.2.2/full_recount_cand5-j1tok-24k.json"),
                "pass": rc["pass"],
                "splits": {
                    s: {k: v[k] for k in ("documents", "document_tokens", "separator_tokens", "stream_tokens")}
                    for s, v in rc["splits"].items()
                },
                "bpe_sample_checked": rc["phase_a"]["bpe_sample_checked"],
                "literal_control_docs": rc["phase_a"]["literal_control_docs"],
            },
            "preservation": ref(Dd, "reports/j1-base-v0.2.2/preservation_receipts.json"),
        },
        "tokenizer": {
            "name": "j1tok-24k",
            "tokenizer_json_sha256": man["tokenizer"]["sha256"],
            "total_vocab": man["tokenizer"]["total_vocab"],
            "controls": "IDs 0-63 reserved (0 <|endoftext|> document separator; 1-11 juniper-chat-v1 roles/channels; 12-63 reserved)",
            "loading": "j1data.jtok.load (encode_special_tokens on): mandatory for every consumer; a plain Tokenizer.from_str turns literal control spellings into control IDs (22 corpus documents)",
        },
        "model": model,
        "training": {
            "context": 2048,
            "packing": "EOS-separated documents, cross-document attention allowed, every position a target incl. EOS, no padding",
            "batch": {
                "global_tokens_per_update": 65536,
                "sequences_per_update": 32,
                "micro_batch": 16,
                "accumulation": 2,
            },
            "optimizer": {
                "type": "AdamW (fused)",
                "betas": [0.9, 0.95],
                "eps": 1e-8,
                "weight_decay": 0.1,
                "decay_groups": "see model.weight_decay",
                "clip_grad_norm": 1.0,
            },
            "schedule": man["schedule"],
            "precision": "BF16 autocast; RMSNorm and losses in fp32; eval cross-entropy fp32 with fp64 sums",
            "kernels": "SDPA default backend (flash on A100); FA-1 explained as accumulation-order noise (evidence/host_*/fa1_diagnosis.json)",
            "execution_profile": {
                "optimized": man["execution_profile"],
                "reference": "eager BF16, --deterministic (bitwise-resumable profile used by the recovery suite)",
                "compile_equivalence": {"status": ce["status"], "record": ref(T, f"{D10}/compile_equivalence.json")},
            },
            "exposure": {
                "tranche": f"{man['schedule']['tranche_valid_targets']} valid targets (no repeats)",
                "full_epoch_horizon_updates": man["schedule"]["horizon_updates"],
                "continuation": "separate owner allocation required",
            },
        },
        "architecture_selection": {
            "rule": sel["rule"],
            "status": sel["status"],
            "decision": sel["decision"],
            "basis": sel.get("basis"),
            "record": ref(T, f"{D10}/selection_18b/select_architecture_v2.json"),
            "protocol": ref(T, "docs/prep/18_architecture_followup_protocol.md"),
        },
        "host": {
            "qualification": host,
            "record": ref(T, f"{D10}/evidence/host_pncbyxaf2dt2jl/host_qualification.json"),
        },
        "evaluation": {
            "base_launch": "full validation split of j1-base-v0.2.2 (bpb from summed NLL over scored UTF-8 bytes, by source); test split untouched",
            "validation": man["validation"],
            "thresholds": {k: pol[k] for k in ("instability_after_update", "instability_rel", "canary_updates")},
            "product_evaluations": "POSTTRAINING/RELEASE gates (pending)",
        },
        "checkpoints": {
            **man["checkpoints"],
            "retention_policy": "v0.7: init, latest three, hourly for 24 active hours, daily, stage, best, release lineage; tranche keeps all",
            "publication": "manifest-last COMPLETE marker, full received-byte verification",
            "recovery": ref(T, f"{D10}/evidence/recovery_vol.json"),
        },
        "preservation": {
            "destinations": "FLOWBOX HDD canonical; Google Drive (owner-approved unencrypted base remote, juniper1/); GitHub source/evidence; Hugging Face private CinqicX/juniper1-corpus",
            "checkpoint_replication": "FLOWBOX pull with full rehash (tools/replicate_ckpts.py)",
            "lag_limits": {
                k: man["limits"][k] for k in ("max_replica_lag_updates", "max_replica_lag_seconds", "volume_bytes_max")
            },
            "key_and_access_recovery": "R02 BLOCKED on owner actions (recovery sheet custody, Drive OAuth publishing); base-launch artifacts are private and unencrypted (no key on the recovery path)",
        },
        "supervision": {
            "on_pod_controller": "tools/pod_controller.py (metrics progress)",
            "off_pod_watchdog": "tools/remote_watchdog.py on CPU pod vls66uj40yrv2c",
            "live_drill": ref(T, f"{D10}/controller_live_drill.md"),
            "limits": man["limits"],
            "stop_commands": [
                "touch /vol/runs/STOP_REQUESTED (graceful)",
                "RunPod pod-action stop pncbyxaf2dt2jl (provider)",
            ],
        },
        "budget": {
            "contract": ref(T, f"{D10}/budget_contract.json"),
            "B_task_usd": bud["B_task_usd"],
            "allocations": bud["allocations"],
            "reserve_usd": bud["reserve_usd"],
        },
        "time_logging": {
            "schema": "j1.time-event.v1 (j1bench/timelog.py)",
            "qualification": [
                ref(T, f"{D10}/evidence/timelog_qual_flowbox.json"),
                ref(T, f"{D10}/evidence/timelog_qual_flowbox_hdd.json"),
            ],
        },
        "invalidation_map": {
            "corpus or exclusion change": [
                "corpus identity",
                "recount",
                "preservation receipts",
                "launch manifest",
                "architecture comparison applicability (re-check prefix identity)",
            ],
            "tokenizer change": ["everything above", "all architecture and LR evidence"],
            "model shape change": [
                "architecture selection",
                "LR",
                "throughput references",
                "compile equivalence",
                "launch manifest",
            ],
            "precision/kernel/compiler/driver/image change": [
                "host qualification",
                "FA-1",
                "compile equivalence",
                "throughput references",
                "recovery evidence",
            ],
            "schedule, batch or optimizer change": [
                "LR evidence",
                "launch manifest",
                "resume semantics of existing checkpoints (a corrective successor branch is required)",
            ],
            "host change": ["host qualification", "throughput references", "comparison timing (protocol 18 §6)"],
        },
        "known_limitations": limitations["limitations"],
        "known_limitations_record": ref(T, str(a.limitations)),
        "launch_manifest": ref(T, str(a.manifest)),
    }
    lock = {
        "schema": "j1.training-contract-lock.v2",
        "train_commit": head(T),
        "data_commit": head(Dd),
        "deployed_train_commit": man["code"]["train_commit"],
        "status": status,
    }
    return contract, lock, model


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-root", type=Path, default=Path())
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--readiness-matrix", type=Path, required=True, help="readiness_v2.py output for this manifest")
    ap.add_argument("--limitations", type=Path, required=True, help="JSON with a 'limitations' list of strings")
    ap.add_argument("--gates-file", type=Path, help="committed gate definitions (default: train root's)")
    ap.add_argument("--status", choices=["FROZEN", "FROZEN_LAUNCH_BLOCKED"], help="refused when the inputs derive less")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    contract, lock, model = build(a)
    a.out.write_text(json.dumps(contract, indent=1) + "\n")
    lock = {"schema": lock["schema"], "contract_sha256": sha(a.out), **{k: v for k, v in lock.items() if k != "schema"}}
    a.out.with_suffix(".lock.json").write_text(json.dumps(lock, indent=1) + "\n")
    print(
        json.dumps(
            {
                "status": contract["status"],
                "n_params": model["n_params"],
                "decayed": model["weight_decay"]["decayed"]["count"],
                "excluded": model["weight_decay"]["excluded"]["count"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
