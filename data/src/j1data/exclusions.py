"""Build the evaluation-exclusion index from configs/exclusions.toml.

Hugging Face benchmark files are fetched at pinned revisions and checked
against their LFS SHA-256; Cinqic evaluation files are read from pinned commits
of local clones. The resulting index and a manifest (per benchmark: revision,
file hashes, item and n-gram counts) are written under
``<data root>/exclusions/<index identity>/``.
"""

from __future__ import annotations

import fnmatch
import json
import re
import subprocess
import tomllib
from pathlib import Path
from typing import Any
from urllib.parse import quote

import pyarrow.parquet as pq

from j1data.acquire import HF_ENDPOINT, download_file, git_blob_oid, list_hf_files
from j1data.contamination import ContaminationIndex
from j1data.hashing import sha256_bytes, sha256_json
from j1data.paths import DataLayout, safe_child
from j1data.registry import REPO_ROOT


def load_config(path: Path | None = None) -> dict[str, Any]:
    with (path or REPO_ROOT / "configs" / "exclusions.toml").open("rb") as handle:
        return tomllib.load(handle)


def _field_texts(record: dict[str, Any], dotted: str) -> list[str]:
    value: Any = record
    for part in dotted.split("."):
        if isinstance(value, dict):
            value = value.get(part)
        else:
            return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [v for v in value if isinstance(v, str)]
    return []


def _string_leaves(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [s for v in value.values() for s in _string_leaves(v)]
    if isinstance(value, list):
        return [s for v in value for s in _string_leaves(v)]
    return []


def _hf_items(entry: dict[str, Any], layout: DataLayout) -> tuple[list[str], list[dict[str, Any]]]:
    listing = {e["path"]: e for e in list_hf_files(entry["repo"], entry["revision"])}
    paths = sorted(p for p in listing if any(fnmatch.fnmatch(p, pat) for pat in entry["files"]))
    if not paths:
        raise RuntimeError(f"{entry['id']}: no files matched")
    items: list[str] = []
    files = []
    for path in paths:
        info = listing[path]
        raw_dir = layout.exclusions() / "raw" / entry["id"] / entry["revision"]
        target = safe_child(raw_dir, path)
        url = f"{HF_ENDPOINT}/datasets/{entry['repo']}/resolve/{entry['revision']}/{quote(path, safe='/')}"
        upstream = (info.get("lfs") or {}).get("oid")
        digest = download_file(url, target, int(info["size"]), upstream)
        upstream_git_oid = info.get("oid")
        if upstream is None and (
            not isinstance(upstream_git_oid, str)
            or git_blob_oid(target, upstream_git_oid) != upstream_git_oid
        ):
            target.unlink(missing_ok=True)
            raise RuntimeError(f"{entry['id']}: Git blob mismatch for {path}")
        files.append(
            {
                "path": path,
                "sha256": digest,
                "upstream_sha256": upstream,
                "git_blob_oid": upstream_git_oid,
                "size": int(info["size"]),
            }
        )
        table = pq.read_table(target)
        for record in table.to_pylist():
            for group in entry["items"]:
                items.append("\n".join(t for name in group for t in _field_texts(record, name)))
    return items, files


def _git_items(entry: dict[str, Any], clone: Path) -> tuple[list[str], list[dict[str, Any]]]:
    tree = subprocess.run(
        ["git", "-C", str(clone), "ls-tree", "-r", "--format=%(objectname) %(path)", entry["revision"]],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    items: list[str] = []
    files = []
    for line in tree:
        oid, path = line.split(" ", 1)
        if not any(fnmatch.fnmatch(path, pat) for pat in entry["files"]):
            continue
        blob = subprocess.run(
            ["git", "-C", str(clone), "cat-file", "blob", oid], check=True, capture_output=True
        ).stdout
        try:
            text = blob.decode("utf-8")
        except UnicodeDecodeError:
            continue
        files.append({"path": path, "git_blob_oid": oid, "sha256": sha256_bytes(blob), "size": len(blob)})
        if path.endswith(".jsonl"):
            for row in text.splitlines():
                if row.strip():
                    items.extend(_string_leaves(json.loads(row)))
        elif path.endswith(".json"):
            items.extend(_string_leaves(json.loads(text)))
        else:
            items.extend(block for block in re.split(r"\n\s*\n", text) if block.strip())
    if not files:
        raise RuntimeError(f"{entry['id']}: no files matched at {entry['revision']}")
    return [i for i in items if i.strip()], files


def collect_items(layout: DataLayout, clones: dict[str, str]) -> dict[str, list[str]]:
    return {name: texts for name, (texts, _) in _collect(layout, clones).items()}


def _collect(layout: DataLayout, clones: dict[str, str]) -> dict[str, tuple[list[str], dict[str, Any]]]:
    config = load_config()
    out: dict[str, tuple[list[str], dict[str, Any]]] = {}
    for entry in config["benchmark"]:
        if entry["kind"] == "huggingface":
            texts, files = _hf_items(entry, layout)
        else:
            if entry["repo"] not in clones:
                raise SystemExit(f"{entry['id']}: no local clone given for {entry['repo']}")
            texts, files = _git_items(entry, Path(clones[entry["repo"]]))
        source = {k: entry[k] for k in ("kind", "repo", "revision", "files", "license") if k in entry}
        out[entry["id"]] = (texts, source | {"file_hashes": files})
    return out


def build(
    layout: DataLayout,
    ngram: int,
    min_coverage: float,
    min_informative_words: int,
    short_min_words: int,
    short_min_chars: int,
    short_min_informative_words: int,
    clones: dict[str, str],
) -> tuple[str, dict[str, Any]]:
    config = load_config()
    collected = _collect(layout, clones)
    items = {name: texts for name, (texts, _) in collected.items()}
    sources = {name: source for name, (_, source) in collected.items()}
    source_fingerprint = sha256_json(sources)
    index, stats = ContaminationIndex.build(
        items,
        config["canaries"],
        ngram,
        min_coverage,
        min_informative_words,
        short_min_words,
        short_min_chars,
        short_min_informative_words,
        source_fingerprint,
    )
    identity = index.identity()
    directory = layout.exclusions() / identity[:16]
    index.save(directory)
    manifest = {
        "schema": "j1.exclusion-manifest.v1",
        "index_identity": identity,
        "ngram": ngram,
        "min_coverage": min_coverage,
        "min_informative_words": min_informative_words,
        "short_exact": {
            "min_words": short_min_words,
            "min_chars": short_min_chars,
            "min_informative_words": short_min_informative_words,
        },
        "canaries": config["canaries"],
        "source_fingerprint": source_fingerprint,
        "benchmarks": {name: {**sources[name], **stats[name]} for name in sorted(sources)},
        "total_ngrams": int(index.hashes.size),
        "total_items": int(index.item_ngrams.size),
    }
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    return identity, manifest
