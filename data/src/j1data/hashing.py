"""Content hashes and stable identifiers.

Identifiers are derived only from logical source identity (source id plus the
upstream record key), never from file paths or processing order, so a record
keeps its id when the corpus is reprocessed or re-sharded.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

DOC_ID_NAMESPACE = "j1.doc.v1"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path, chunk_size: int = 1 << 22) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_json(value))


def doc_id(source_id: str, source_key: str) -> str:
    """Stable 128-bit document id rendered as 32 hex characters."""
    payload = f"{DOC_ID_NAMESPACE}\x00{source_id}\x00{source_key}".encode()
    return hashlib.sha256(payload).hexdigest()[:32]


def unit_interval(namespace: str, key: str) -> float:
    """Map a key to a deterministic point in [0, 1) using 53 bits of SHA-256."""
    digest = hashlib.sha256(f"{namespace}\x00{key}".encode()).digest()
    return (int.from_bytes(digest[:8], "big") >> 11) / float(1 << 53)
