"""Derive a successor accepted build from an existing one by removing adjudicated documents.

The parent's shards are never modified. Shards that contain excluded documents are rewritten without those
rows (same schema, zstd); all other shards are hard-linked. The successor manifest keeps the parent's
structure, updates per-shard SHA-256 and row counts and the per-split totals, and records its lineage:
parent manifest SHA-256, the exclusion list SHA-256 and the adjudication-criteria commit. Its build identity
is derived from that lineage (`j1data.successor.successor_identity`), never copied from the parent, so
parent-bound records cannot validate for the successor. Blocks copied unchanged from the parent are listed
under `inherited_from_parent` (their per-source counts describe the parent, not the successor).

`--criteria-commit` must be the full 40-hex SHA of a commit that exists in this repository (`--repo`,
default: the repository holding this tool); it is verified with `git rev-parse --verify` and abbreviations
are refused, because the value is hashed into the build identity. `successor_of.derived_by` records the
code revision that performed the derivation (`build.code_revision()`); it is outside the lineage keys, so
identities of earlier successors are unchanged.

`exclusion_list_sha256` is the SHA-256 of `json.dumps(sorted_ids)` with Python's default separators
(", "); the exact bytes are also written as `excluded_doc_ids.json` next to the manifest.

Unchanged shards are hard links to the parent's files: both trees are immutable and must never be written in
place.

Usage: successor_manifest.py PARENT_DIR OUT_DIR --exclude IDS.json --criteria-commit FULL_SHA --build-id NAME
       [--repo REPO_DIR]
Fails closed if an excluded ID is still present, if any excluded ID is absent from the parent, or if the
output exists.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from j1data.build import code_revision
from j1data.successor import successor_identity, successor_problem

REPO_ROOT = Path(__file__).resolve().parents[1]


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        while b := f.read(1 << 24):
            h.update(b)
    return h.hexdigest()


def resolve_criteria_commit(value: str, repo: Path) -> str:
    """The full SHA of an existing commit; abbreviations and unknown objects are refused."""
    if not re.fullmatch(r"[0-9a-f]{40}", value):
        raise SystemExit(f"--criteria-commit must be a full 40-hex commit SHA, got {value!r}")
    result = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "--verify", "--quiet", f"{value}^{{commit}}"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0 or result.stdout.strip() != value:
        raise SystemExit(f"--criteria-commit {value} is not a commit in {repo}")
    return value


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("parent", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--exclude", type=Path, required=True)
    ap.add_argument("--criteria-commit", required=True)
    ap.add_argument("--build-id", required=True)
    ap.add_argument("--repo", type=Path, default=REPO_ROOT, help="repository holding the criteria commit")
    a = ap.parse_args()
    criteria_commit = resolve_criteria_commit(a.criteria_commit, a.repo)
    if a.out.exists():
        raise SystemExit(f"output exists (immutable): {a.out}")
    excl = sorted(set(json.loads(a.exclude.read_text())))
    excl_set = set(excl)
    parent_bytes = (a.parent / "manifest.json").read_bytes()
    man = json.loads(parent_bytes)
    partial = a.out.with_name(a.out.name + ".partial")
    partial.mkdir(parents=True)
    removed: dict[str, int] = {}
    shards = []
    for e in man["shards"]:
        src = a.parent / e["file"]
        dst = partial / e["file"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        if sha(src) != e["sha256"]:
            raise SystemExit(f"parent shard hash mismatch: {e['file']}")
        ids = pq.read_table(src, columns=["doc_id"]).column("doc_id")
        hit = pc.is_in(ids, value_set=pa.array(excl)) if excl else None
        n_hit = int(pc.sum(hit).as_py() or 0) if hit is not None else 0
        if n_hit == 0:
            os.link(src, dst)
            shards.append(dict(e))
            continue
        t = pq.read_table(src)
        keep = pc.invert(pc.is_in(t.column("doc_id"), value_set=pa.array(excl)))
        t2 = t.filter(keep)
        if t2.num_rows:
            pq.write_table(t2, dst, compression="zstd")
            shards.append({**e, "rows": t2.num_rows, "sha256": sha(dst)})
        removed[e["split"]] = removed.get(e["split"], 0) + n_hit
    if sum(removed.values()) != len(excl):
        raise SystemExit(f"{len(excl) - sum(removed.values())} excluded IDs are not in the parent build")
    # Re-check: no excluded ID anywhere; recompute per-split document counts.
    per_split: dict[str, int] = {}
    for e in shards:
        ids = pq.read_table(partial / e["file"], columns=["doc_id"]).column("doc_id").to_pylist()
        if excl_set.intersection(ids):
            raise SystemExit(f"excluded ID survived in {e['file']}")
        per_split[e["split"]] = per_split.get(e["split"], 0) + len(ids)
    inherited = sorted(
        k for k in man if k not in {"shards", "build_id", "build_identity", "accepted_by_split"}
    )
    lineage = {
        "build_id": man["build_id"],
        "build_identity": man["build_identity"],
        "manifest_sha256": hashlib.sha256(parent_bytes).hexdigest(),
        "exclusion_list_sha256": hashlib.sha256(json.dumps(excl).encode()).hexdigest(),
        "excluded_documents": len(excl),
        "removed_by_split": dict(sorted(removed.items())),
        "criteria_commit": criteria_commit,
        "rule": "docs/contamination_adjudication_criteria_v2.md (corpus-scale successor rule)",
        # Outside successor.LINEAGE_KEYS: informational, not part of the identity.
        "derived_by": {"tool": "tools/successor_manifest.py", **code_revision()},
    }
    out = {
        **{k: v for k, v in man.items() if k != "shards"},
        "build_id": a.build_id,
        "build_identity": successor_identity(lineage),
        "accepted_by_split": dict(sorted(per_split.items())),
        "inherited_from_parent": inherited,
        "shards": shards,
        "successor_of": lineage,
    }
    if (problem := successor_problem(out)) is not None:
        raise SystemExit(problem)
    (partial / "excluded_doc_ids.json").write_text(json.dumps(excl))
    (partial / "manifest.json").write_text(json.dumps(out, indent=1, sort_keys=True))
    partial.rename(a.out)
    print(
        json.dumps(
            {
                "build_id": a.build_id,
                "accepted_by_split": out["accepted_by_split"],
                "removed": removed,
                "manifest_sha256": sha(a.out / "manifest.json"),
            }
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
