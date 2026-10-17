"""Identity of a successor accepted build derived from a parent by removing adjudicated documents.

A successor never inherits its parent's build identity: its identity hashes the parent identity, the parent
manifest, the exclusion list and the adjudication-criteria commit, so records bound to the parent (gate,
dedup, selection, statistics) cannot validate for the successor.
"""

from __future__ import annotations

from typing import Any

from j1data.hashing import sha256_json

SUCCESSOR_IDENTITY_VERSION = "j1.successor-identity.v1"
LINEAGE_KEYS = ("build_identity", "manifest_sha256", "exclusion_list_sha256", "criteria_commit")


def successor_identity(lineage: dict[str, Any]) -> str:
    missing = [key for key in LINEAGE_KEYS if not isinstance(lineage.get(key), str) or not lineage[key]]
    if missing:
        raise ValueError(f"successor lineage incomplete: {missing}")
    return sha256_json({"version": SUCCESSOR_IDENTITY_VERSION, **{key: lineage[key] for key in LINEAGE_KEYS}})


def successor_problem(manifest: dict[str, Any]) -> str | None:
    """None when the manifest is not a successor or carries its own derived identity; else the problem."""
    lineage = manifest.get("successor_of")
    if lineage is None:
        return None
    if not isinstance(lineage, dict):
        return "successor_of is not an object"
    try:
        expected = successor_identity(lineage)
    except ValueError as error:
        return str(error)
    identity = manifest.get("build_identity")
    if identity == lineage.get("build_identity"):
        return "successor carries its parent's build identity"
    if identity != expected:
        return f"successor build identity {identity} != derived {expected}"
    return None
