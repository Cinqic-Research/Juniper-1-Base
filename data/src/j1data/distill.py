"""Distillation protocol v2 (docs/distillation/gptoss_teacher_protocol_v2.md): teacher prompt pool and arms.

The pool is the GSM8K ``main`` train problems that ``j1-sft-v1.3`` left out **only** because of its mix cap
(rejection reason "mix cap"): their human solutions passed the v1.3 annotation checks, and none of them is in
v1.3 train or dev, so a teacher arm adds supervision on problems the student has not seen. A seeded sample of
``n`` problems is written as ``j1.teacher-prompt`` records for ``j1data.teacher.run_batch``; the matched
no-teacher control (C0) uses the human solutions of the same problems.

``arms`` builds one dataset per arm (C0 human solution, A1 teacher answer, A2 concise verified rationale,
A3 verified equations; ``j1data.teacherverify``) over the matched problems P that have a target in every
arm, plus the same replay sample of v1.3 train episodes (``replay`` per arm episode) copied byte for byte.
Dev rows are v1.3's. Each arm is its own self-hashed dataset identity; none is ever merged into v1.3.
"""

from __future__ import annotations

import argparse
import json
import sys
import tomllib
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from j1data import jtok, sftbuild, sftsources, teacherverify
from j1data.dedup import save_npz_deterministic
from j1data.hashing import canonical_json, sha256_file, sha256_json, unit_interval
from j1data.sftgen import Item, Turn

VERSION = "j1.distill.v1"
MIX_CAP_REASON = "mix cap: GSM8K limited to train_cap episodes"


def _records(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_bytes().split(b"\n") if x]


def pool(
    build: Path, gsm8k_train: Path, gsm8k_sha256: str, teacher_cfg: Path, n: int, seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Teacher prompts (``teacher-record`` schema) and, per prompt, the human solution for the C0 control."""
    if sha256_file(gsm8k_train) != gsm8k_sha256:
        raise SystemExit("GSM8K train parquet differs from the pinned digest")
    used = {
        r["source_record"]
        for split in ("train", "dev")
        for r in _records(build / f"{split}.jsonl")
        if r["source_id"] == "openai/gsm8k"
    }
    capped = sorted(
        r["source_record"]
        for r in _records(build / "rejections.jsonl")
        if r["family"] == "gsm8k" and r["reason"] == MIX_CAP_REASON
    )
    if used & set(capped):
        raise SystemExit("a mix-capped problem is also in train/dev")
    rows = pq.read_table(gsm8k_train).to_pylist()
    ranked = sorted(capped, key=lambda rec: unit_interval(f"j1.distill.pool.{seed}", rec))[:n]
    settings = tomllib.loads(teacher_cfg.read_text())
    template, spec = settings["template"], settings["prompts"]["gsm8k-train"]
    out: list[dict[str, Any]] = []
    controls: list[dict[str, Any]] = []
    for rec in sorted(ranked, key=lambda r: int(r.rsplit("/", 1)[1])):
        row = rows[int(rec.rsplit("/", 1)[1])]
        out.append(
            {
                "prompt_id": f"gsm8k-train-{rec.rsplit('/', 1)[1]}",
                "source": "gsm8k-train",
                "source_license": spec["license"],
                "template_version": template["version"],
                "system": template["system"],
                "developer": template["developer"],
                "user": row["question"],
                "oracle": {"kind": "known_answer", "answer": row["answer"].split("####")[-1].strip()},
            }
        )
        controls.append(
            {"prompt_id": out[-1]["prompt_id"], "source_record": rec, "human_solution": row["answer"]}
        )
    return out, controls


ARMS = ("C0", "A1", "A2", "A3")
TEACHER_LICENSE = "LicenseRef-gpt-oss-20b-output"


def _arm_item(
    arm: str, rec: str, question: str, target: str, answer: str, checks: list[dict[str, Any]]
) -> Item:
    teacher = arm != "C0"
    return Item(
        f"distill.gsm8k.{arm.lower()}",
        f"distill:{arm}:{rec}",
        [Turn("user", question), Turn("assistant", target, {"type": "number", "value": answer})],
        ["direct_answer", "concise_reasoning", "no_tool"],
        factuality="VERIFIED_KNOWN_ANSWER",
        difficulty="medium",
        source_id="gpt-oss-20b-teacher" if teacher else "openai/gsm8k",
        source_revision="6cee5e81ee83917806bbde320786a8fb61efebee"
        if teacher
        else "740312add88f781978c0658806c59bc2815b9866",
        source_record=rec,
        license=TEACHER_LICENSE if teacher else "MIT",
        origin="teacher" if teacher else "human",
        split="train",
        extra_checks=checks,
    )


def arm_items(
    records: Path, controls: Path, gsm8k_train: Path
) -> tuple[dict[str, list[Item]], list[dict[str, Any]]]:
    """Per arm, the matched problems' items (same order in every arm) and the per-problem ledger."""
    human = {
        it.source_record: it for it in sftsources.gsm8k_items(gsm8k_train, split="train", revision="")[0]
    }
    ctl = {c["prompt_id"]: c for c in _records(controls)}
    ledger, rows = [], []
    for r in _records(records):
        pid = r["prompt"]["prompt_id"]
        rec = ctl[pid]["source_record"]
        entry: dict[str, Any] = {"prompt_id": pid, "source_record": rec, "decision": r["decision"]["status"]}
        if r["decision"]["status"] != "accepted" or rec not in human:
            entry["matched"] = False
            entry["reason"] = (
                "teacher rejected" if r["decision"]["status"] != "accepted" else "no verified human solution"
            )
            ledger.append(entry)
            continue
        cand = r["candidates"][r["decision"]["selected_index"]]
        answer = r["prompt"]["oracle"]["answer"].replace(",", "")
        targets = teacherverify.arm_targets(cand["output"], answer)
        check = teacherverify.check_rationale(teacherverify.strip_answer_lines(cand["output"]))
        entry |= {"arms": {k: v is not None for k, v in targets.items()}, "rationale": check.as_dict()}
        entry["matched"] = all(v is not None for v in targets.values())
        ledger.append(entry)
        if entry["matched"]:
            rows.append((rec, human[rec], targets, r["record_id"]))
    out: dict[str, list[Item]] = {a: [] for a in ARMS}
    for rec, h, targets, rid in rows:
        q = h.turns[0].text
        known = {"name": "final_answer_matches_dataset_label", "oracle": "known_answer", "passed": True}
        out["C0"].append(
            _arm_item("C0", rec, q, h.turns[-1].text, (h.turns[-1].spec or {})["value"], list(h.extra_checks))
        )
        for arm in ("A1", "A2", "A3"):
            checks = [known | {"detail": f"teacher record {rid}"}]
            if arm != "A1":
                checks.append({"name": "every_equality_reevaluated", "oracle": "execution", "passed": True,
                               "detail": teacherverify.VERSION})  # fmt: skip
            target = targets[arm]
            assert target is not None and h.turns[-1].spec is not None  # matched problems have every arm
            out[arm].append(_arm_item(arm, rec, q, target, h.turns[-1].spec["value"], checks))
    return out, ledger


def write_arm(
    build: Path,
    root: Path,
    cfg_path: Path,
    arm: str,
    items: list[Item],
    replay: list[int],
    out: Path,
    meta: dict[str, Any],
) -> dict[str, Any]:
    cfg = tomllib.loads(cfg_path.read_text())
    manifest = json.loads((build / "manifest.json").read_text())
    tok = jtok.load((root / sftbuild.TOKENIZER_REL).read_text())
    decon = manifest["decontamination_index_identity"]
    lines = [x for x in (build / "train.jsonl").read_bytes().split(b"\n") if x]
    with np.load(build / "train_rows.npz") as z:
        inputs, targets, offsets, ids = z["inputs"], z["targets"], z["offsets"], z["episode_ids"]
    rows_in = [inputs[offsets[i] : offsets[i + 1]] for i in replay]
    rows_tg = [targets[offsets[i] : offsets[i + 1]] for i in replay]
    recs = [lines[i] for i in replay]
    new_ids = [ids[i] for i in replay]
    cfg["build"]["approved_licenses"] = [*cfg["build"]["approved_licenses"], TEACHER_LICENSE]
    for it in items:
        ep = sftbuild.to_episode(it, decon)
        x, y = sftbuild.encode(tok, ep, cfg, decon)
        rows_in.append(np.asarray(x, dtype=inputs.dtype))
        rows_tg.append(np.asarray(y, dtype=targets.dtype))
        rec = sftbuild._record(it, "train", ep, x, y)
        recs.append(canonical_json({**rec, "episode": ep}))
        new_ids.append(np.frombuffer(bytes.fromhex(ep["episode_id"]), dtype=np.uint8))
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} is not empty")
    out.mkdir(parents=True, exist_ok=True)
    lens = np.array([len(r) for r in rows_in], dtype=np.int64)
    save_npz_deterministic(
        out / "train_rows.npz",
        inputs=np.concatenate(rows_in),
        targets=np.concatenate(rows_tg),
        offsets=np.concatenate([[0], np.cumsum(lens)]).astype(np.int64),
        episode_ids=np.stack(new_ids),
    )
    (out / "train.jsonl").write_bytes(b"".join(r + b"\n" for r in recs))
    for name in ("dev.jsonl", "dev_rows.npz", "NOTICE"):
        (out / name).write_bytes((build / name).read_bytes())
    (out / "rejections.jsonl").write_bytes(b"")
    (out / "stats.json").write_text(
        json.dumps({"arm": arm, "matched_problems": len(items), "replay_episodes": len(replay)}, indent=1)
    )
    files = {
        n: {"sha256": sha256_file(out / n), "size": (out / n).stat().st_size}
        for n in sorted(p.name for p in out.iterdir())
    }
    tg = np.concatenate(rows_tg)
    body = {k: v for k, v in manifest.items() if k != "manifest_sha256"}
    body |= {
        "version": f"{manifest['version']}-distill-{arm.lower()}",
        "files": files,
        "counts": {
            **manifest["counts"],
            "train": {
                "episodes": len(recs),
                "target_tokens": int((tg != -100).sum()),
                "tokens": int(lens.sum()) + len(recs),
            },
        },
        "distill": {
            "builder": VERSION,
            "arm": arm,
            "parent_manifest_sha256": manifest["manifest_sha256"],
            **meta,
        },
        "approved_licenses": cfg["build"]["approved_licenses"],
    }
    body["manifest_sha256"] = sha256_json(body)
    (out / "manifest.json").write_bytes(canonical_json(body) + b"\n")
    return body


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m j1data.distill", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pool")
    p.add_argument("--build", type=Path, required=True, help="the j1-sft-v1.3 build directory")
    p.add_argument("--gsm8k-train", type=Path, required=True)
    p.add_argument("--gsm8k-sha256", required=True)
    p.add_argument("--teacher-config", type=Path, required=True)
    p.add_argument("--n", type=int, default=1200)
    p.add_argument("--seed", type=int, default=20261013)
    p.add_argument("--out-dir", type=Path, required=True)
    m = sub.add_parser("arms")
    m.add_argument("--build", type=Path, required=True)
    m.add_argument("--root", type=Path, required=True)
    m.add_argument("--config", type=Path, required=True, help="configs/sft_v13.toml")
    m.add_argument("--records", type=Path, required=True, help="teacher run records.jsonl")
    m.add_argument("--controls", type=Path, required=True)
    m.add_argument("--gsm8k-train", type=Path, required=True)
    m.add_argument("--replay", type=int, default=3, help="replay episodes per arm episode")
    m.add_argument("--seed", type=int, default=20261013)
    m.add_argument("--out-root", type=Path, required=True)
    a = ap.parse_args(argv)
    if a.cmd == "arms":
        items, ledger = arm_items(a.records, a.controls, a.gsm8k_train)
        n = len(items["C0"])
        if n == 0:
            raise SystemExit("no matched problems")
        lines = [x for x in (a.build / "train.jsonl").read_bytes().split(b"\n") if x]
        replay = sorted(
            sorted(range(len(lines)), key=lambda i: unit_interval(f"j1.distill.replay.{a.seed}", str(i)))[
                : a.replay * n
            ]
        )
        meta = {
            "records_sha256": sha256_file(a.records),
            "controls_sha256": sha256_file(a.controls),
            "matched": n,
            "replay_ids_sha256": sha256_json(replay),
            "replay_per_arm_episode": a.replay,
        }
        a.out_root.mkdir(parents=True, exist_ok=True)
        (a.out_root / "ledger.jsonl").write_text(
            "".join(json.dumps(x, sort_keys=True) + "\n" for x in ledger)
        )
        res = {
            arm: write_arm(
                a.build, a.root, a.config, arm, items[arm], replay, a.out_root / arm.lower(), meta
            )["manifest_sha256"]
            for arm in ARMS
        }
        print(json.dumps({"matched": n, "replay": len(replay), "manifests": res}, indent=1))
        return 0
    prompts, controls = pool(a.build, a.gsm8k_train, a.gsm8k_sha256, a.teacher_config, a.n, a.seed)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    res = {}
    for name, rows in (("prompts.jsonl", prompts), ("controls.jsonl", controls)):
        path = a.out_dir / name
        if path.exists():
            raise SystemExit(f"{path} exists; prompt pools are immutable")
        path.write_text("".join(json.dumps(x, sort_keys=True) + "\n" for x in rows))
        res[name] = {"rows": len(rows), "sha256": sha256_file(path)}
    print(json.dumps(res))
    return 0


if __name__ == "__main__":
    sys.exit(main())
