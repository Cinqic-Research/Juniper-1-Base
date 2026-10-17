"""Diagnostic subsets of an admitted SFT dataset (controlled ablations; never candidate data).

``python -m j1data.sftsubset --dataset DIR --out DIR --name NAME [--exclude-family PREFIX ...]
[--exclude-factuality LABEL ...]`` copies the dataset with every train episode of the excluded family
prefixes or factuality labels (e.g. ``VERIFIED_SOURCE``: episodes whose targets carry Wikidata labels,
including facts embedded in hierarchy and formatting tasks) removed. Dev rows, the evaluation
bundle identity and every admitted byte of the kept episodes are unchanged. The new manifest is
self-hashed, names its parent manifest and the exclusions, and carries ``"diagnostic": true``: a stage
protocol may train a diagnostic arm on it, but no candidate may come from it.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np

from j1data.dedup import save_npz_deterministic
from j1data.hashing import canonical_json, sha256_file, sha256_json

IGNORE = -100


def subset(
    dataset: Path, out: Path, name: str, exclude: list[str], exclude_factuality: list[str] | None = None
) -> dict[str, Any]:
    drop_fact = set(exclude_factuality or [])
    manifest = json.loads((dataset / "manifest.json").read_text())
    body = {k: v for k, v in manifest.items() if k != "manifest_sha256"}
    if sha256_json(body) != manifest["manifest_sha256"]:
        raise SystemExit("parent manifest self-hash mismatch")
    for fname, rec in manifest["files"].items():
        if sha256_file(dataset / fname) != rec["sha256"]:
            raise SystemExit(f"{fname}: bytes differ from the parent manifest")
    if not exclude and not drop_fact:
        raise SystemExit("a diagnostic subset must exclude something")
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} is not empty")
    out.mkdir(parents=True, exist_ok=True)

    lines = (dataset / "train.jsonl").read_bytes().split(b"\n")
    lines = [x for x in lines if x]
    recs = [json.loads(x) for x in lines]
    keep = [
        i
        for i, r in enumerate(recs)
        if not any(r["family"] == p or r["family"].startswith(p + ".") for p in exclude)
        and r["factuality"] not in drop_fact
    ]
    with np.load(dataset / "train_rows.npz") as z:
        inputs, targets, offsets, ids = z["inputs"], z["targets"], z["offsets"], z["episode_ids"]
    if len(offsets) - 1 != len(lines):
        raise SystemExit("train rows and records disagree")
    seg_in = [inputs[offsets[i] : offsets[i + 1]] for i in keep]
    seg_tg = [targets[offsets[i] : offsets[i + 1]] for i in keep]
    lens = np.array([len(s) for s in seg_in], dtype=np.int64)
    new_offsets = np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)
    save_npz_deterministic(
        out / "train_rows.npz",
        inputs=np.concatenate(seg_in).astype(inputs.dtype),
        targets=np.concatenate(seg_tg).astype(targets.dtype),
        offsets=new_offsets,
        episode_ids=ids[keep],
    )
    (out / "train.jsonl").write_bytes(b"".join(lines[i] + b"\n" for i in keep))
    for fname in ("dev.jsonl", "dev_rows.npz", "rejections.jsonl", "stats.json"):
        shutil.copyfile(dataset / fname, out / fname)

    files = dict(manifest["files"])
    for fname in ("train.jsonl", "train_rows.npz"):
        p = out / fname
        files[fname] = {"rows": len(keep), "sha256": sha256_file(p), "size": p.stat().st_size}
    tg = np.concatenate(seg_tg)
    new = {
        **body,
        "version": f"{manifest['version']}-diag-{name}",
        "diagnostic": True,
        "subset": {
            "parent_manifest_sha256": manifest["manifest_sha256"],
            "excluded_family_prefixes": sorted(exclude),
            "excluded_factuality": sorted(drop_fact),
            "removed_train_episodes": len(lines) - len(keep),
            "note": "diagnostic ablation; stats.json and rejections.jsonl are the parent's",
        },
        "files": files,
    }
    new["counts"] = {
        **manifest["counts"],
        "train": {
            "episodes": len(keep),
            "target_tokens": int((tg != IGNORE).sum()),
            "tokens": int(lens.sum()) + len(keep),
        },
    }
    new["manifest_sha256"] = sha256_json(new)
    (out / "manifest.json").write_bytes(canonical_json(new) + b"\n")
    return new


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m j1data.sftsubset", description=__doc__)
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--exclude-family", action="append", default=[])
    ap.add_argument("--exclude-factuality", action="append", default=[])
    a = ap.parse_args(argv)
    m = subset(a.dataset, a.out, a.name, a.exclude_family, a.exclude_factuality)
    print(
        json.dumps(
            {"version": m["version"], "manifest_sha256": m["manifest_sha256"], **m["subset"]}, indent=1
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
