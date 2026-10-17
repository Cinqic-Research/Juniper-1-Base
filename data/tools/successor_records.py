"""Launch-readiness records for an accepted successor build (readiness v2 gates B03 and B05).

``source_permissions.json``: every source present in the build with its training basis: an APPROVED owner
decision (``configs/owner_decisions.toml``, required by ``gates.REQUIRED_OWNER_DECISIONS``) or an APPROVED
source license that permits training (``configs/sources.toml``). ``successor_lineage.json``: the successor's
identity, parent and exclusion list hash, and a full check that no excluded document ID remains in any split.

Usage: successor_records.py ACCEPTED_DIR EXCLUDED_IDS.json OUT_DIR
"""

from __future__ import annotations

import hashlib
import json
import sys
import tomllib
from pathlib import Path

import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from j1data.gates import REQUIRED_OWNER_DECISIONS

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    acc, excl_path, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    man = json.loads((acc / "manifest.json").read_text())
    sources = sorted({s["source_id"] for s in man["shards"]})
    decisions = tomllib.loads((ROOT / "configs/owner_decisions.toml").read_text())["decision"]
    sources_toml = tomllib.loads((ROOT / "configs/sources.toml").read_text())["source"]
    src_cfg = {s["source_id"]: s for s in sources_toml}
    rows = {}
    for s in sources:
        required = [d for d, governed in REQUIRED_OWNER_DECISIONS.items() if s in governed]
        dec = [d for d in decisions if d["decision_id"] in required and s in d.get("sources", [])]
        lic = src_cfg.get(s, {}).get("license", {})
        if required:
            ok = bool(dec) and all(d["status"] == "APPROVED" for d in dec)
            basis = f"owner decision {required[0]}: {dec[0]['status'] if dec else 'MISSING'}"
        else:
            ok = lic.get("state") == "APPROVED" and str(lic.get("training_use", "")).startswith("Permitted")
            basis = f"source license {lic.get('identifiers')} ({lic.get('state')}): {lic.get('training_use')}"
        rows[s] = {"training_approved": ok, "basis": basis}
    perms = {
        "schema": "j1.source-permissions.v1",
        "build": man["build_id"],
        "build_identity": man["build_identity"],
        "sources": rows,
        "all_sources_training_approved": all(r["training_approved"] for r in rows.values()),
        "release_decisions": (
            "PENDING (weights license treatment, dataset redistribution): not training gates"
        ),
    }
    excluded = set(json.loads(excl_path.read_text()))
    present = set()
    for sh in man["shards"]:
        present.update(pq.read_table(acc / sh["file"], columns=["doc_id"]).column("doc_id").to_pylist())
    so = man.get("successor_of", {})
    lineage = {
        "schema": "j1.successor-lineage-check.v1",
        "build": man["build_id"],
        "build_identity": man["build_identity"],
        "manifest_sha256": hashlib.sha256((acc / "manifest.json").read_bytes()).hexdigest(),
        "parent": {k: so.get(k) for k in ("build_id", "build_identity", "manifest_sha256")},
        "criteria_commit": so.get("criteria_commit"),
        "exclusion_list_sha256": so.get("exclusion_list_sha256"),
        "exclusion_list_sha256_recomputed": hashlib.sha256(json.dumps(sorted(excluded)).encode()).hexdigest(),
        "excluded_documents": len(excluded),
        "removed_by_split": so.get("removed_by_split"),
        "accepted_by_split": man["accepted_by_split"],
        "documents_checked": len(present),
        "exclusions_absent_from_successor": not (present & excluded),
    }
    lineage["exclusion_hash_consistent"] = (
        lineage["exclusion_list_sha256"] == lineage["exclusion_list_sha256_recomputed"]
    )
    out.mkdir(parents=True, exist_ok=True)
    (out / "source_permissions.json").write_text(json.dumps(perms, indent=1) + "\n")
    (out / "successor_lineage.json").write_text(json.dumps(lineage, indent=1) + "\n")
    print(json.dumps({"permissions_ok": perms["all_sources_training_approved"], **{k: lineage[k] for k in (
        "exclusions_absent_from_successor", "exclusion_hash_consistent", "documents_checked")}}))  # fmt: skip
    return 0 if perms["all_sources_training_approved"] and lineage["exclusions_absent_from_successor"] else 1


if __name__ == "__main__":
    sys.exit(main())
