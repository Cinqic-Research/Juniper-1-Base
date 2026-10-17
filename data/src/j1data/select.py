"""Quota-driven selection of the accepted corpus from the deduplicated pool.

The processed pool is larger than the corpus target (3B+ tokens, owner
directive 2026-10-08) so that each source can be filled with its best
documents. For every source in configs/selection.toml:

* sources marked ``all`` keep every deduplicated document;
* otherwise documents are ranked by an upstream quality score when the source
  has one (FineWeb-Edu and FineMath classifier scores), ties and unscored
  sources ordered by a keyed hash of the doc id (a reproducible uniform sample),
  and taken until the source's byte budget is filled;
* an optional ``group_field`` splits a source budget into sub-budgets (for
  example code by language), with ``"*"`` covering every other value.

Budgets are given in provisional tokens and converted to bytes with a
per-source bytes-per-token figure. After provisional token counting, the
measured figures replace the initial estimates and selection is re-run; the
summary records which figures were used.

Recovery (v0.3 15-minute rule): each source is a committed unit. Its masks are written
atomically, then ``_sources/<source>.json`` lists every mask with SHA-256 and binds the
source's quota, unit inventory, keep-file hashes, build identity and build code fingerprint
(``build.BUILD_MODULES`` hashes, not the git commit, so an unrelated commit does not discard a
committed source). A restart reuses a source only when that marker's binding matches and every
mask verifies. The stage summary is removed before any mask is rewritten and written last; the
git revision (``code``) is recorded in the marker and the summary for information only.
"""

from __future__ import annotations

import json
import os
import tomllib
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from j1data import faults
from j1data.dedup import UnitRef, build_code_binding
from j1data.hashing import sha256_file, sha256_json, unit_interval
from j1data.registry import REPO_ROOT
from j1data.shards import remove_stale_temporaries, verify_files, write_json_atomic

VERSION = "j1.selection.v1"


def load_config(path: Path | None = None) -> dict[str, Any]:
    with (path or REPO_ROOT / "configs" / "selection.toml").open("rb") as handle:
        return tomllib.load(handle)


def _dig(meta: dict[str, Any], dotted: str) -> Any:
    value: Any = meta
    for part in dotted.split("."):
        value = value.get(part) if isinstance(value, dict) else None
    return value


def run(
    refs: list[UnitRef],
    keep_dir: Path,
    out_dir: Path,
    config: dict[str, Any],
    code: dict[str, Any] | None = None,
    build_identity: str | None = None,
) -> dict[str, Any]:
    namespace = config["selection"]["namespace"]
    quotas = {q["source"]: q for q in config["quota"]}
    by_source: dict[str, list[UnitRef]] = defaultdict(list)
    for ref in refs:
        by_source[ref.source_id].append(ref)
    summary: dict[str, Any] = {}
    remove_stale_temporaries(out_dir)
    (out_dir / "summary.json").unlink(missing_ok=True)  # invalidate before any mask is rewritten
    build_code = build_code_binding()
    for source, units in sorted(by_source.items()):
        quota = quotas.get(source)
        if quota is None:
            raise SystemExit(f"{source}: no selection quota configured")
        binding = {
            "selection_version": VERSION,
            "build_identity": build_identity,
            "build_code": build_code,
            "namespace": namespace,
            "quota": quota,
            "units": [[ref.unit_id, ref.rows] for ref in units],
            "keep_sha256": sha256_json(
                [
                    sha256_file(keep_dir / ref.source_id / f"{ref.unit_id}.npz") if ref.rows else None
                    for ref in units
                ]
            ),
        }
        marker_path = out_dir / "_sources" / f"{source}.json"
        if marker_path.exists():
            marker = json.loads(marker_path.read_text())
            if marker.get("binding") == binding and not verify_files(out_dir, marker.get("masks", [])):
                summary[source] = marker["summary"]
                continue
            marker_path.unlink()
        columns = ["doc_id", "n_bytes"] + (["meta"] if quota.get("rank") or quota.get("group_field") else [])
        budgets = _budgets(quota)
        bucket_names = sorted(budgets)
        parts: dict[str, list[np.ndarray]] = defaultdict(list)
        for position, ref in enumerate(units):
            if not ref.rows:
                continue
            with np.load(keep_dir / ref.source_id / f"{ref.unit_id}.npz") as npz:
                rows = np.flatnonzero(npz["keep"])
            table = pq.read_table(ref.candidates, columns=columns).take(rows).to_pydict()
            metas = [json.loads(m) for m in table["meta"]] if "meta" in table else [{}] * len(rows)
            scores = [_dig(m, quota["rank"]) if quota.get("rank") else None for m in metas]
            groups = [str(_dig(m, quota["group_field"])) if quota.get("group_field") else "*" for m in metas]
            parts["position"].append(np.full(len(rows), position, dtype=np.int32))
            parts["row"].append(rows.astype(np.int64))
            parts["bytes"].append(np.asarray(table["n_bytes"], dtype=np.int64))
            parts["neg_score"].append(
                np.array(
                    [-float(x) if isinstance(x, (int, float)) else 0.0 for x in scores], dtype=np.float64
                )
            )
            parts["key"].append(
                np.array([unit_interval(namespace, d) for d in table["doc_id"]], dtype=np.float64)
            )
            parts["bucket"].append(
                np.array(
                    [
                        bucket_names.index(g)
                        if g in budgets
                        else bucket_names.index("*")
                        if "*" in budgets
                        else -1
                        for g in groups
                    ],
                    dtype=np.int32,
                )
            )
        arrays = {k: np.concatenate(v) for k, v in parts.items()} if parts else {}
        selected: dict[int, list[int]] = defaultdict(list)
        used: dict[str, int] = {}
        available: dict[str, int] = {}
        chosen_docs: dict[str, int] = {}
        if arrays:
            order = np.lexsort((arrays["key"], arrays["neg_score"]))
            for index, name in enumerate(bucket_names):
                members = order[arrays["bucket"][order] == index]
                sizes = arrays["bytes"][members]
                budget = budgets[name]
                if budget is None:
                    taken = members
                else:
                    before = np.concatenate(([0], np.cumsum(sizes)[:-1]))
                    taken = members[before < budget]
                used[name] = int(arrays["bytes"][taken].sum())
                available[name] = int(sizes.sum())
                chosen_docs[name] = int(taken.size)
                for doc in taken:
                    selected[int(arrays["position"][doc])].append(int(arrays["row"][doc]))
        masks = []
        for position, ref in enumerate(units):
            mask = np.zeros(ref.rows, dtype=bool)
            mask[selected.get(position, [])] = True
            relative = f"{ref.source_id}/{ref.unit_id}.npy"
            path = out_dir / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_name(path.name + ".tmp.npy")
            with tmp.open("wb") as handle:
                np.save(handle, mask)
                handle.flush()
                os.fsync(handle.fileno())
            tmp.replace(path)
            masks.append({"file": relative, "sha256": sha256_file(path)})
            faults.point("select.mask")
        summary[source] = {
            "mode": "all" if quota.get("all") else "quota",
            "rank": quota.get("rank") or "keyed hash",
            "bytes_per_token": quota.get("bytes_per_token"),
            "budget_bytes": {k: v for k, v in budgets.items() if v is not None},
            "selected_bytes": dict(used),
            "selected_documents": dict(chosen_docs),
            "available_bytes_after_dedup": dict(available),
        }
        write_json_atomic(
            marker_path, {"binding": binding, "masks": masks, "summary": summary[source], "code": code}
        )
        faults.point("select.source.commit")
    record = {
        "schema": "j1.selection-summary.v1",
        "selection_version": VERSION,
        "build_identity": build_identity,
        "config": config,
        "code": code,
        "sources": summary,
    }
    write_json_atomic(out_dir / "summary.json", record)
    return record


def _budgets(quota: dict[str, Any]) -> dict[str, int | None]:
    if quota.get("all"):
        return {"*": None}
    ratio = float(quota["bytes_per_token"])
    if "groups" in quota:
        return {group: int(tokens * ratio) for group, tokens in quota["groups"].items()}
    return {"*": int(quota["tokens"] * ratio)}
