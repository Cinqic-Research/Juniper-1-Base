"""Build identity and configuration loading.

A build directory (``work/<build id>``) is bound to the identity of the full
corpus pipeline: output-affecting configuration, acquisition manifests,
registries, the exclusion index, and the code for processing through
finalization. Every unit marker repeats the identity so changed scope, source
pins, selection, split rules, or implementation cannot silently resume or mix
with existing outputs.
"""

from __future__ import annotations

import json
import re
import subprocess
import tomllib
from pathlib import Path
from typing import Any

from j1data import __version__
from j1data.acquire import manifest_identity
from j1data.contamination import ContaminationIndex
from j1data.hashing import sha256_file, sha256_json
from j1data.minhash import MinHashConfig
from j1data.paths import DataLayout
from j1data.process import Settings
from j1data.quality import QualityConfig
from j1data.registry import REPO_ROOT, load_registry
from j1data.shards import write_json_atomic

BUILD_CONFIG_SECTIONS = (
    "build",
    "normalize",
    "privacy",
    "quality",
    "minhash",
    "contamination",
    "dedup",
    "splits",
)
BUILD_MODULES = (
    "build.py",
    "adapters/__init__.py",
    "acquire.py",
    "code_signatures.py",
    "contamination.py",
    "cli.py",
    "dedup.py",
    "domains.py",
    "exclusions.py",
    "faults.py",
    "finalize.py",
    "hashing.py",
    "minhash.py",
    "ngrams.py",
    "normalize.py",
    "privacy.py",
    "process.py",
    "quality.py",
    "paths.py",
    "pools.py",
    "registry.py",
    "select.py",
    "shards.py",
    "splits.py",
    "token_counts.py",
    "tokenization.py",
)


def build_code_fingerprint() -> dict[str, str]:
    package = Path(__file__).resolve().parent
    return {name: sha256_file(package / name) for name in BUILD_MODULES}


def load_pipeline_config(path: Path | None = None) -> dict[str, Any]:
    with (path or REPO_ROOT / "configs" / "pipeline.toml").open("rb") as handle:
        return tomllib.load(handle)


def code_revision() -> dict[str, Any]:
    def git(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(REPO_ROOT), *args], capture_output=True, text=True
        ).stdout.strip()

    status = git("status", "--porcelain", "--untracked-files=no", "--", "src", "configs", "schemas")
    return {"commit": git("rev-parse", "HEAD") or None, "dirty": bool(status), "package_version": __version__}


def current_exclusion_dir(layout: DataLayout) -> Path:
    pointer = layout.exclusions() / "CURRENT"
    if not pointer.exists():
        raise SystemExit("no exclusion index has been built (run `j1data exclusions`)")
    identity_prefix = pointer.read_text().strip()
    if not re.fullmatch(r"[0-9a-f]{16}", identity_prefix):
        raise ValueError("exclusion CURRENT pointer is not a 16-character lowercase identity prefix")
    return layout.exclusions() / identity_prefix


def _identity_inputs(layout: DataLayout, config: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    exclusion_dir = current_exclusion_dir(layout)
    index = ContaminationIndex.load(exclusion_dir)
    index_identity = index.identity()
    exclusion_manifest_path = exclusion_dir / "manifest.json"
    exclusion_manifest = json.loads(exclusion_manifest_path.read_text())
    contamination = config["contamination"]
    expected_exclusion_settings = {
        "ngram": contamination["ngram"],
        "min_coverage": contamination["min_coverage"],
        "min_informative_words": contamination["min_informative_words"],
        "short_exact": {
            "min_words": contamination["short_min_words"],
            "min_chars": contamination["short_min_chars"],
            "min_informative_words": contamination["short_min_informative_words"],
        },
    }
    actual_exclusion_settings = {key: exclusion_manifest.get(key) for key in expected_exclusion_settings}
    if actual_exclusion_settings != expected_exclusion_settings:
        raise ValueError("exclusion index settings do not match the current pipeline configuration")
    if (
        exclusion_manifest.get("index_identity") != index_identity
        or exclusion_manifest.get("source_fingerprint") != index.source_fingerprint
    ):
        raise ValueError("exclusion manifest does not match the loaded index")
    acquisition_manifests: dict[str, str | None] = {}
    registry = load_registry()
    for source_id, source in registry.items():
        path = layout.acquisition_manifest(source_id)
        if path.exists():
            manifest = json.loads(path.read_text())
            acquisition_manifests[source_id] = manifest_identity(source, manifest)
        else:
            acquisition_manifests[source_id] = None
    return exclusion_dir, {
        "pipeline": {section: config[section] for section in BUILD_CONFIG_SECTIONS},
        "build_scope_sha256": sha256_file(REPO_ROOT / "configs" / "build_scope.toml"),
        "selection_config_sha256": sha256_file(REPO_ROOT / "configs" / "selection.toml"),
        "acquisition_manifests": acquisition_manifests,
        "sources_sha256": sha256_file(REPO_ROOT / "configs" / "sources.toml"),
        "exclusions_config_sha256": sha256_file(REPO_ROOT / "configs" / "exclusions.toml"),
        "exclusion_index": index_identity,
        "exclusion_sources": exclusion_manifest.get("source_fingerprint"),
        "build_code": build_code_fingerprint(),
    }


def expected_build_identity(layout: DataLayout) -> str:
    """Compute the current identity without creating or mutating a build record."""
    _, inputs = _identity_inputs(layout, load_pipeline_config())
    return sha256_json(inputs)


def prepare(layout: DataLayout, allow_dirty: bool = False, build_id: str | None = None) -> Settings:
    """``build_id`` overrides the configured id for exploratory runs, which then
    live in their own work directory and can never be mistaken for the release build."""
    config = load_pipeline_config()
    build_id = build_id or config["build"]["id"]
    revision = code_revision()
    if revision["dirty"] and not allow_dirty:
        raise SystemExit("source tree has uncommitted changes under src/, configs/ or schemas/; commit first")
    exclusion_dir, identity_inputs = _identity_inputs(layout, config)
    identity = sha256_json(identity_inputs)
    record_path = layout.work(build_id) / "build.json"
    if record_path.exists():
        recorded = json.loads(record_path.read_text())
        if recorded["identity"] != identity:
            raise SystemExit(
                f"{build_id} was started under identity {recorded['identity'][:12]}; "
                f"current inputs give {identity[:12]}. Use a new build id; "
                "finished builds are never overwritten."
            )
    else:
        write_json_atomic(
            record_path,
            {
                "build_id": build_id,
                "identity": identity,
                "inputs": identity_inputs,
                "started_at_code": revision,
            },
        )
    mh = config["minhash"]
    return Settings(
        layout=layout,
        build_id=build_id,
        quality=QualityConfig.from_dict({k: v for k, v in config["quality"].items() if k != "version"}),
        minhash=MinHashConfig(mh["num_perm"], mh["shingle_words"], mh["seed"]),
        contamination_dir=exclusion_dir,
        audit_reject_text_rate=float(config["build"]["audit_reject_text_rate"]),
        identity=identity,
    )
