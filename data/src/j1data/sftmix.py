"""Exposure-controlled mixtures of an admitted SFT dataset (j1.sft-mix.v1).

``python -m j1data.sftmix --dataset DIR --config TOML --out DIR`` materializes one training epoch whose
episode draws per group are exactly the counts the config registers. The trainer then samples it
with its unchanged per-epoch permutation: with ``updates * episodes_per_update`` draws, one epoch is
the whole registered exposure and no episode is skipped.

Each config group selects parent train episodes (by category or family prefix; every episode must
belong to exactly one group) and gives a draw count. Inside a group, draws are split over families in
proportion to their episode counts (largest remainder). Inside a family every episode is repeated
``draws // n`` times, and a seeded permutation picks the remaining ``draws % n`` episodes once more.
Repeats are byte-identical rows with derived episode IDs (the trainer requires unique IDs);
``mixture_plan.jsonl`` maps every row to its parent episode and copy number.

Nothing new is admitted. Dev rows, the evaluation bundle identity and every token of every episode
come from the parent, whose manifest and file hashes are checked first. The new manifest names the
parent, the config, the realized per-group and per-family draws, and the approved-license list. That
list may only narrow the parent's, and every train record's license must stay inside it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tomllib
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from j1data.dedup import save_npz_deterministic
from j1data.hashing import canonical_json, sha256_file, sha256_json

VERSION = "j1.sft-mix.v1"
IGNORE = -100
COPIED = ("dev.jsonl", "dev_rows.npz", "NOTICE")


def _derived_id(parent: bytes, copy: int) -> bytes:
    if copy == 0:
        return parent
    return hashlib.sha256(b"j1.sft-mix.v1\0" + parent + copy.to_bytes(4, "big")).digest()[:16]


def _member(record: dict[str, Any], selector: dict[str, Any]) -> bool:
    if record.get("_source", "parent") != selector.get("source", "parent"):
        return False
    if "category" in selector:
        return bool(record["category"] == selector["category"])
    if "categories" in selector:
        return record["category"] in selector["categories"]
    prefix = str(selector["family_prefix"])
    family = str(record["family"])
    return family == prefix or family.startswith(prefix + ".")


def _largest_remainder(total: int, weights: dict[str, int]) -> dict[str, int]:
    whole = sum(weights.values())
    exact = {k: total * w / whole for k, w in weights.items()}
    out = {k: int(v) for k, v in exact.items()}
    left = total - sum(out.values())
    for k in sorted(weights, key=lambda k: (-(exact[k] - out[k]), k))[:left]:
        out[k] += 1
    return out


def plan(records: list[dict[str, Any]], cfg: dict[str, Any]) -> tuple[list[tuple[int, int]], dict[str, Any]]:
    """Return ``(rows, realized)``: rows are ``(record index, copy number)`` in a canonical order. Records of
    the parent come first, then admitted replay records (``_source = "replay"``)."""
    groups = cfg["groups"]
    names = sorted(groups)
    owner: list[str | None] = []
    for r in records:
        hits = [g for g in names if _member(r, groups[g]["select"])]
        if len(hits) != 1:
            raise SystemExit(f"episode {r['episode_id']} belongs to {len(hits)} groups: {hits}")
        owner.append(hits[0])
    total = sum(int(groups[g]["draws"]) for g in names)
    if total != int(cfg["draws_total"]):
        raise SystemExit(f"group draws sum to {total}, not draws_total {cfg['draws_total']}")
    seed = int(cfg["seed"])
    rows: list[tuple[int, int]] = []
    realized: dict[str, Any] = {}
    for gi, g in enumerate(names):
        members = [i for i, o in enumerate(owner) if o == g]
        if not members:
            raise SystemExit(f"group {g} selects no episodes")
        by_family: dict[str, list[int]] = defaultdict(list)
        for i in members:
            by_family[records[i]["family"]].append(i)
        share = _largest_remainder(int(groups[g]["draws"]), {f: len(v) for f, v in by_family.items()})
        fam_out = {}
        for fi, fam in enumerate(sorted(by_family)):
            idx, draws = by_family[fam], share[fam]
            full, extra = divmod(draws, len(idx))
            rng = np.random.default_rng([seed, gi, fi])
            picked = sorted(idx[j] for j in rng.permutation(len(idx))[:extra])
            for i in idx:
                rows.extend((i, c) for c in range(full))
            rows.extend((i, full) for i in picked)
            fam_out[fam] = {"episodes": len(idx), "draws": draws}
        realized[g] = {
            "select": groups[g]["select"],
            "episodes": len(members),
            "draws": int(groups[g]["draws"]),
            "passes": round(int(groups[g]["draws"]) / len(members), 6),
            "families": fam_out,
        }
    rows.sort()
    return rows, realized


def _concat(a: tuple[Any, Any, Any, Any], b: tuple[Any, Any, Any, Any]) -> tuple[Any, Any, Any, Any]:
    inputs = np.concatenate([a[0], b[0]])
    targets = np.concatenate([a[1], b[1]])
    offsets = np.concatenate([a[2], a[2][-1] + b[2][1:]]).astype(np.int64)
    return inputs, targets, offsets, np.concatenate([a[3], b[3]])


def _load_replay(
    cfg: dict[str, Any],
    dataset: Path | None,
    admission: Path | None,
    parent: dict[str, Any],
    approved: list[str],
) -> tuple[dict[str, Any], list[dict[str, Any]], tuple[Any, Any, Any, Any]]:
    """Admitted replay records (``j1data.sftreplay``) of a frozen dataset with the parent's row format."""
    if dataset is None or admission is None:
        raise SystemExit(
            "the mixture config has a [replay] section: give --replay-dataset and --replay-admission"
        )
    rc = cfg["replay"]
    m = json.loads((dataset / "manifest.json").read_text())
    if sha256_json({k: v for k, v in m.items() if k != "manifest_sha256"}) != m["manifest_sha256"]:
        raise SystemExit("replay manifest self-hash mismatch")
    if m["manifest_sha256"] != rc["dataset_manifest_sha256"]:
        raise SystemExit("replay dataset is not the one the mixture config pins")
    for k in ("protocol", "max_tokens", "tokenizer_sha256", "decontamination_index_identity", "stage"):
        if m.get(k) != parent.get(k):
            raise SystemExit(f"replay and parent differ in {k}")
    for fname in ("train.jsonl", "train_rows.npz"):
        if sha256_file(dataset / fname) != m["files"][fname]["sha256"]:
            raise SystemExit(f"replay {fname}: bytes differ from its manifest")
    adm = json.loads(admission.read_text())
    if sha256_json({k: v for k, v in adm.items() if k != "admission_sha256"}) != adm["admission_sha256"]:
        raise SystemExit("replay admission self-hash mismatch")
    if (
        adm["admission_sha256"] != rc["admission_sha256"]
        or adm["v12_manifest_sha256"] != m["manifest_sha256"]
    ):
        raise SystemExit("replay admission is not the pinned one, or describes another dataset")
    if adm["v13_manifest_sha256"] != parent["manifest_sha256"]:
        raise SystemExit("replay admission was checked against another parent")
    admitted = set(adm["admitted_episode_ids"])
    lines = [x for x in (dataset / "train.jsonl").read_bytes().split(b"\n") if x]
    with np.load(dataset / "train_rows.npz") as z:
        inputs, targets, offsets, ids = z["inputs"], z["targets"], z["offsets"], z["episode_ids"]
    if len(offsets) - 1 != len(lines):
        raise SystemExit("replay train rows and records disagree")
    keep, recs = [], []
    for i, line in enumerate(lines):
        r = json.loads(line)
        if ids[i].tobytes().hex() != r["episode_id"]:
            raise SystemExit(f"replay train row {i} is not record {r['episode_id']}")
        if r["episode_id"] in admitted:
            if r["license"] not in approved:
                raise SystemExit(
                    f"replay record {r['episode_id']} has license {r['license']!r} outside the list"
                )
            keep.append(i)
            recs.append({**r, "_source": "replay"})
    if len(keep) != len(admitted):
        raise SystemExit("admitted replay IDs missing from the replay dataset")
    seg_in = [inputs[offsets[i] : offsets[i + 1]] for i in keep]
    seg_tg = [targets[offsets[i] : offsets[i + 1]] for i in keep]
    lens = np.array([len(x) for x in seg_in], dtype=np.int64)
    arrays = (
        np.concatenate(seg_in),
        np.concatenate(seg_tg),
        np.concatenate([[0], np.cumsum(lens)]).astype(np.int64),
        ids[keep],
    )
    info = {
        "dataset_manifest_sha256": m["manifest_sha256"],
        "dataset_version": m["version"],
        "admission_sha256": adm["admission_sha256"],
        "admission_tool": adm["schema"],
        "admitted_episodes": len(keep),
        "train_records": {"file": "train.jsonl", **m["files"]["train.jsonl"]},
    }
    return info, recs, arrays


def mix(
    dataset: Path,
    config_path: Path,
    out: Path,
    replay_dataset: Path | None = None,
    replay_admission: Path | None = None,
) -> dict[str, Any]:
    cfg = tomllib.loads(config_path.read_text())
    manifest = json.loads((dataset / "manifest.json").read_text())
    body = {k: v for k, v in manifest.items() if k != "manifest_sha256"}
    if sha256_json(body) != manifest["manifest_sha256"]:
        raise SystemExit("parent manifest self-hash mismatch")
    if manifest["manifest_sha256"] != cfg["parent_manifest_sha256"]:
        raise SystemExit("parent dataset is not the one the mixture config pins")
    if manifest.get("diagnostic"):
        raise SystemExit("a diagnostic subset cannot be a mixture parent")
    for fname, rec in manifest["files"].items():
        if sha256_file(dataset / fname) != rec["sha256"]:
            raise SystemExit(f"{fname}: bytes differ from the parent manifest")
    approved = list(cfg["approved_licenses"])
    if not set(approved) <= set(manifest["approved_licenses"]):
        raise SystemExit("a mixture may only narrow the parent's approved licenses")
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} is not empty")

    lines = [x for x in (dataset / "train.jsonl").read_bytes().split(b"\n") if x]
    records = [json.loads(x) for x in lines]
    with np.load(dataset / "train_rows.npz") as z:
        inputs, targets, offsets, ids = z["inputs"], z["targets"], z["offsets"], z["episode_ids"]
    if len(offsets) - 1 != len(records):
        raise SystemExit("train rows and records disagree")
    for i, r in enumerate(records):
        if ids[i].tobytes().hex() != r["episode_id"]:
            raise SystemExit(f"train row {i} is not record {r['episode_id']}")
        if r["license"] not in approved:
            raise SystemExit(
                f"train record {r['episode_id']} has license {r['license']!r} outside the mixture list"
            )

    replay_info = None
    if "replay" in cfg:
        replay_info, r_records, r_arrays = _load_replay(
            cfg, replay_dataset, replay_admission, manifest, approved
        )
        records = records + r_records
        inputs, targets, offsets, ids = _concat((inputs, targets, offsets, ids), r_arrays)
    elif replay_dataset is not None or replay_admission is not None:
        raise SystemExit("replay files were given but the mixture config has no [replay] section")
    rows, realized = plan(records, cfg)
    per_update = int(cfg["episodes_per_update"])
    if len(rows) != int(cfg["updates"]) * per_update:
        raise SystemExit("draws_total must equal updates * episodes_per_update")
    new_ids = np.stack([np.frombuffer(_derived_id(ids[i].tobytes(), c), dtype=np.uint8) for i, c in rows])
    if len({x.tobytes() for x in new_ids}) != len(rows):
        raise SystemExit("derived episode IDs collide")
    with np.load(dataset / "dev_rows.npz") as z:
        dev_ids = {x.tobytes() for x in z["episode_ids"]}
    if dev_ids & {x.tobytes() for x in new_ids}:
        raise SystemExit("a derived train ID equals a dev ID")

    out.mkdir(parents=True, exist_ok=True)
    seg_in = [inputs[offsets[i] : offsets[i + 1]] for i, _ in rows]
    seg_tg = [targets[offsets[i] : offsets[i + 1]] for i, _ in rows]
    lens = np.array([len(s) for s in seg_in], dtype=np.int64)
    tg = np.concatenate(seg_tg)
    save_npz_deterministic(
        out / "train_rows.npz",
        inputs=np.concatenate(seg_in).astype(inputs.dtype),
        targets=tg.astype(targets.dtype),
        offsets=np.concatenate([[0], np.cumsum(lens)]).astype(np.int64),
        episode_ids=new_ids,
    )
    with (out / "mixture_plan.jsonl").open("wb") as f:
        for (i, c), nid in zip(rows, new_ids, strict=True):
            f.write(
                canonical_json(
                    {
                        "copy": c,
                        "episode_id": nid.tobytes().hex(),
                        "group": next(
                            g
                            for g in sorted(cfg["groups"])
                            if _member(records[i], cfg["groups"][g]["select"])
                        ),
                        "parent_episode_id": records[i]["episode_id"],
                        "parent_row": i,
                    }
                    | ({"source": records[i].get("_source", "parent")} if replay_info else {})
                )
                + b"\n"
            )
    for fname in COPIED:
        shutil.copyfile(dataset / fname, out / fname)
    shutil.copyfile(config_path, out / "mixture_config.toml")

    files = {f: dict(manifest["files"][f]) for f in ("dev.jsonl", "dev_rows.npz", "NOTICE")}
    for fname in ("train_rows.npz", "mixture_plan.jsonl", "mixture_config.toml"):
        p = out / fname
        files[fname] = {"sha256": sha256_file(p), "size": p.stat().st_size}
    files["train_rows.npz"]["rows"] = files["mixture_plan.jsonl"]["rows"] = len(rows)
    new = {
        **{k: v for k, v in body.items() if k not in ("files", "counts", "code_sha256", "config", "builder")},
        "builder": VERSION,
        "version": cfg["version"],
        "approved_licenses": approved,
        "code_sha256": {"sftmix.py": sha256_file(Path(__file__))},
        "mixture": {
            "parent_manifest_sha256": manifest["manifest_sha256"],
            "parent_version": manifest["version"],
            "parent_builder": manifest.get("builder"),
            "parent_code_sha256": manifest.get("code_sha256"),
            "parent_config": manifest.get("config"),
            "parent_train_records": {"file": "train.jsonl", **manifest["files"]["train.jsonl"]},
            "config_sha256": files["mixture_config.toml"]["sha256"],
            "seed": int(cfg["seed"]),
            "updates": int(cfg["updates"]),
            "episodes_per_update": per_update,
            "groups": realized,
            "rationale": cfg.get("rationale", ""),
            "note": (
                "train rows repeat parent (and admitted replay) rows per the plan; parent_row indexes parent "
                "records then replay records; dev, eval bundle and NOTICE are the parent's"
            ),
        }
        | ({"replay": replay_info} if replay_info else {}),
        "files": files,
    }
    new["counts"] = {
        **{k: v for k, v in manifest["counts"].items() if k != "rejected"},
        "train": {
            "episodes": len(rows),
            "unique_parent_episodes": len({i for i, _ in rows}),
            "target_tokens": int((tg != IGNORE).sum()),
            "tokens": int(lens.sum()) + len(rows),
        },
    }
    new["manifest_sha256"] = sha256_json(new)
    (out / "manifest.json").write_bytes(canonical_json(new) + b"\n")
    return new


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m j1data.sftmix", description=__doc__)
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--replay-dataset", type=Path)
    ap.add_argument("--replay-admission", type=Path)
    a = ap.parse_args(argv)
    m = mix(a.dataset, a.config, a.out, a.replay_dataset, a.replay_admission)
    summary = {g: {k: v for k, v in r.items() if k != "families"} for g, r in m["mixture"]["groups"].items()}
    print(
        json.dumps(
            {
                "version": m["version"],
                "manifest_sha256": m["manifest_sha256"],
                "counts": m["counts"]["train"],
                "groups": summary,
            },
            indent=1,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
