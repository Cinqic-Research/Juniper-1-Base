"""Write SFT stage contracts (``j1.sft-contract.v1``) derived from a registered pilot protocol.

``probe --lr X``: a stage-1 learning-rate probe (diagnostic, never a candidate).
``pilot --lr X``: the stage-2 pilot (the single candidate).
``recovery``: the stage-3 recovery contract (30 updates); runs A and B use it under different out-roots
and B is interrupted by the operator after update 15.
``replicate --seed N``: a seed replication of the pilot registered in
``docs/posttrain/sft_replication_protocol_v1.json``; only the seed and the run identity differ from ``pilot``.
``v2 --seed N``: a registered seed of the SFT v2 stage (``docs/posttrain/sft_v2_protocol_v1.json``);
every field comes from that protocol's ``training`` block, its dataset and its run registry.
``v21 --seed N``: a registered seed of the SFT v2.1 stage (``docs/posttrain/sft_v21_protocol_v1.json``),
built exactly as ``v2`` from that protocol's own training block, dataset and run registry.
``v22 --seed N``: a registered seed of the SFT v2.2 stage (``docs/posttrain/sft_v22_protocol_v1.json``), built
the same way (it shares the v2.1 protocol schema).
``distill --arm NAME``: an arm of the distillation stage file (``docs/posttrain/distill_v2_stage.json``),
initialized from the SFT checkpoint it pins.
``diagnostic --arm NAME``: a diagnostic arm of ``docs/posttrain/sft_knowledge_diag_v1.json`` (never a
candidate); the arms differ only in their dataset manifest.

Every field comes from ``docs/posttrain/sft_pilot_protocol_v3.json`` (and, for ``replicate``, the
replication protocol); nothing is chosen here.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from j1bench.sft import contract_sha, validate_contract

TOKENIZER_SHA = "176a89e2197c24d3f024fbb60ce6f346635e7cba912b3579d93b37d486c9adf1"


def _build(proto: dict, kind: str, lr: float) -> dict:
    p = proto["pilot"]
    lrs = proto["lr_probes"]
    if kind == "probe" and lr not in lrs["lrs"]:
        raise SystemExit(f"lr {lr} is not one of the registered probe LRs {lrs['lrs']}")
    updates = {"probe": lrs["updates"], "pilot": p["updates"], "recovery": proto["recovery"]["run_a_updates"]}[kind]
    warmup = {"probe": lrs["warmup_updates"], "pilot": p["warmup_updates"], "recovery": 15}[kind]
    min_ratio = {"probe": lrs["min_lr_ratio"], "pilot": p["min_lr_ratio"], "recovery": p["min_lr_ratio"]}[kind]
    run_id = {
        "probe": f"j1-sft-v1-probe-lr{lr:g}",
        "pilot": "j1-sft-v1-pilot",
        "recovery": "j1-sft-v1-recovery",  # A and B share the contract; they differ only in out-root
    }[kind]
    return {
        "schema": "j1.sft-contract.v1",
        "stage_id": f"juniper1-sft-v1-{kind}",
        "run_id": run_id,
        "protocol": "docs/posttrain/sft_pilot_protocol_v3.json",
        "base": dict(proto["base"]),
        "dataset": {"manifest_sha256": proto["dataset_manifest_sha256"], "tokenizer_sha256": TOKENIZER_SHA},
        "optimizer": {**p["optimizer"], "lr": lr},
        "schedule": {"warmup_updates": warmup, "total_updates": updates, "min_lr_ratio": min_ratio},
        "batch": {"episodes_per_update": p["episodes_per_update"], "token_budget": p["token_budget"]},
        "precision": p["precision"],
        "seed": p["seed"],
        "checkpoint": dict(p["checkpoint"]),
        "dev_every": {"probe": updates, "pilot": p["dev_every"], "recovery": 0}[kind],
        "log_every": 10,
        "deterministic": True,
        "world_size": 1,
    }


def build(proto: dict, kind: str, lr: float) -> dict:
    c = _build(proto, kind, lr)
    validate_contract(c)
    return c


def build_replication(proto: dict, rep: dict, seed: int, rep_path: str) -> dict:
    """The pilot contract with a registered replication seed and its registered run identity."""
    if rep.get("schema") != "j1.sft-replication-protocol.v1":
        raise SystemExit(f"not a replication protocol: {rep.get('schema')!r}")
    if rep["parent_protocol"]["schema"] != proto["schema"]:
        raise SystemExit("replication protocol names a different parent protocol")
    if seed not in rep["seeds"]["registered"] or seed in rep["seeds"]["used_before"]:
        raise SystemExit(f"seed {seed} is not a registered replication seed {rep['seeds']['registered']}")
    c = _build(proto, "pilot", rep["lr"])
    c.update(stage_id=rep["stage_id"], run_id=rep["runs"][str(seed)], protocol=rep_path, seed=seed)
    validate_contract(c)
    return c


def build_diagnostic(proto: dict, diag: dict, arm: str, diag_path: str) -> dict:
    if diag.get("schema") != "j1.sft-diagnostic-protocol.v1":
        raise SystemExit(f"not a diagnostic protocol: {diag.get('schema')!r}")
    if arm not in diag["arm_datasets"]:
        raise SystemExit(f"arm {arm!r} is not registered {sorted(diag['arm_datasets'])}")
    sch = diag["arms"]["schedule"]
    c = _build(proto, "pilot", sch["lr"])
    c.update(
        stage_id="juniper1-sft-v1-diagnostic",
        run_id=diag["arms"]["run_ids"][arm],
        protocol=diag_path,
        seed=sch["seed"],
        schedule={
            "warmup_updates": sch["warmup_updates"],
            "total_updates": sch["total_updates"],
            "min_lr_ratio": sch["min_lr_ratio"],
        },
        dev_every=sch["dev_every"],
    )
    c["dataset"] = {**c["dataset"], "manifest_sha256": diag["arm_datasets"][arm]["manifest_sha256"]}
    validate_contract(c)
    return c


def build_v2(v2: dict, seed: int, v2_path: str, smoke: bool = False) -> dict:
    if v2.get("schema") not in ("j1.sft-v2-protocol.v1", "j1.sft-v21-protocol.v1"):
        raise SystemExit(f"not an SFT v2 or v2.1 protocol: {v2.get('schema')!r}")
    if seed not in v2["seeds"]:
        raise SystemExit(f"seed {seed} is not registered {v2['seeds']}")
    t = v2["training"]
    c = {
        "schema": "j1.sft-contract.v1",
        "stage_id": v2["stage_id"],
        "run_id": v2["smoke"]["run_id"] if smoke else v2["runs"][str(seed)],
        "protocol": v2_path,
        "base": dict(v2["base"]),
        "dataset": {"manifest_sha256": v2["dataset_manifest_sha256"], "tokenizer_sha256": TOKENIZER_SHA},
        "optimizer": {**t["optimizer"], "lr": t["lr"]},
        "schedule": {
            "warmup_updates": v2["smoke"]["warmup_updates"] if smoke else t["warmup_updates"],
            "total_updates": v2["smoke"]["updates"] if smoke else t["updates"],
            "min_lr_ratio": t["min_lr_ratio"],
        },
        "batch": {"episodes_per_update": t["episodes_per_update"], "token_budget": t["token_budget"]},
        "precision": t["precision"],
        "seed": seed,
        "checkpoint": dict(t["checkpoint"]),
        "dev_every": 0 if smoke else t["dev_every"],
        "log_every": 10,
        "deterministic": True,
        "world_size": 1,
    }
    validate_contract(c)
    return c


def build_distill(stage: dict, arm: str, stage_path: str) -> dict:
    if stage.get("schema") != "j1.sft-distill-stage.v1":
        raise SystemExit(f"not a distillation stage file: {stage.get('schema')!r}")
    if arm not in stage["arms"]:
        raise SystemExit(f"arm {arm!r} is not registered {sorted(stage['arms'])}")
    t = stage["training"]
    c = {
        "schema": "j1.sft-contract.v1",
        "stage_id": stage["stage_id"],
        "run_id": stage["arms"][arm]["run_id"],
        "protocol": stage_path,
        "base": dict(stage["base"]),
        "init": dict(stage["init"]),
        "dataset": {"manifest_sha256": stage["arms"][arm]["manifest_sha256"], "tokenizer_sha256": TOKENIZER_SHA},
        "optimizer": {**t["optimizer"], "lr": t["lr"]},
        "schedule": {
            "warmup_updates": t["warmup_updates"],
            "total_updates": t["updates"],
            "min_lr_ratio": t["min_lr_ratio"],
        },
        "batch": {"episodes_per_update": t["episodes_per_update"], "token_budget": t["token_budget"]},
        "precision": t["precision"],
        "seed": t["seed"],
        "checkpoint": dict(t["checkpoint"]),
        "dev_every": t["dev_every"],
        "log_every": 10,
        "deterministic": True,
        "world_size": 1,
    }
    validate_contract(c)
    return c


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    root = Path(__file__).resolve().parents[1]
    kinds = ("probe", "pilot", "recovery", "replicate", "diagnostic", "v2", "v21", "v22", "distill")
    ap.add_argument("kind", choices=kinds)
    ap.add_argument("--lr", type=float, help="probe/pilot/recovery: the learning rate")
    ap.add_argument("--seed", type=int, help="replicate: a seed registered in the replication protocol")
    ap.add_argument("--arm", help="diagnostic: an arm registered in the diagnostic protocol")
    ap.add_argument("--smoke", action="store_true", help="v2/v21: the protocol's smoke/recovery contract")
    ap.add_argument("--distill-stage", type=Path, default=root / "docs/posttrain/distill_v2_stage.json")
    ap.add_argument("--v2-protocol", type=Path, default=root / "docs/posttrain/sft_v2_protocol_v1.json")
    ap.add_argument("--v21-protocol", type=Path, default=root / "docs/posttrain/sft_v21_protocol_v1.json")
    ap.add_argument("--v22-protocol", type=Path, default=root / "docs/posttrain/sft_v22_protocol_v1.json")
    ap.add_argument("--diagnostic", type=Path, default=root / "docs/posttrain/sft_knowledge_diag_v1.json")
    ap.add_argument("--protocol", type=Path, default=root / "docs/posttrain/sft_pilot_protocol_v3.json")
    ap.add_argument("--replication", type=Path, default=root / "docs/posttrain/sft_replication_protocol_v1.json")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    proto = json.loads(a.protocol.read_text())
    if a.kind == "distill":
        if a.arm is None or a.lr is not None or a.seed is not None:
            ap.error("distill takes --arm only")
        sp = a.distill_stage.resolve().relative_to(root).as_posix()
        c = build_distill(json.loads(a.distill_stage.read_text()), a.arm, sp)
    elif a.kind in ("v2", "v21", "v22"):
        if a.seed is None or a.lr is not None or a.arm is not None:
            ap.error(f"{a.kind} takes --seed only")
        path = {"v2": a.v2_protocol, "v21": a.v21_protocol, "v22": a.v22_protocol}[a.kind]
        rel = path.resolve().relative_to(root).as_posix()
        c = build_v2(json.loads(path.read_text()), a.seed, rel, smoke=a.smoke)
    elif a.kind == "diagnostic":
        if a.arm is None or a.lr is not None or a.seed is not None:
            ap.error("diagnostic takes --arm only")
        diag_path = a.diagnostic.resolve().relative_to(root).as_posix()
        c = build_diagnostic(proto, json.loads(a.diagnostic.read_text()), a.arm, diag_path)
    elif a.kind == "replicate":
        if a.seed is None or a.lr is not None:
            ap.error("replicate takes --seed and no --lr (the replication protocol registers the lr)")
        rep_path = a.replication.resolve().relative_to(root).as_posix()
        c = build_replication(proto, json.loads(a.replication.read_text()), a.seed, rep_path)
    else:
        if a.lr is None or a.seed is not None:
            ap.error(f"{a.kind} takes --lr and no --seed")
        c = build(proto, a.kind, a.lr)
    a.out.write_text(json.dumps(c, indent=1, sort_keys=True) + "\n")
    print(a.out, contract_sha(c))
    return 0


if __name__ == "__main__":
    sys.exit(main())
