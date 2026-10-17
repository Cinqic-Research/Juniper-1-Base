"""Successor builds carry their own lineage-derived identity and fail closed on incomplete exclusions."""

import json
import subprocess
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from j1data.hashing import sha256_file
from j1data.successor import LINEAGE_KEYS, successor_identity, successor_problem

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "successor_manifest.py"
V022_MANIFEST = ROOT / "reports" / "j1-base-v0.2.2" / "manifest.json"
V022_IDENTITY = "89b08d5754c46dfd18ec239fdacb21ae899c1cf9d7606392401b5224c54bffc5"
V022_MANIFEST_SHA256 = "54a3adb9650c96bf67b44b721386722383caeb774b48b0bf970ad6f9634d4185"


def _head() -> str:
    result = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "--verify", "HEAD"], capture_output=True, text=True
    )
    if result.returncode != 0:
        pytest.skip("not a git checkout")
    return result.stdout.strip()


def _parent(tmp_path: Path) -> Path:
    parent = tmp_path / "parent"
    shards = []
    for name, split, ids in (("a.parquet", "train", ["d1", "d2", "d3"]), ("b.parquet", "validation", ["d4"])):
        (parent / "shards").mkdir(parents=True, exist_ok=True)
        path = parent / "shards" / name
        pq.write_table(pa.table({"doc_id": ids, "text": [f"text {i}" for i in ids]}), path)
        shards.append(
            {"file": f"shards/{name}", "split": split, "rows": len(ids), "sha256": sha256_file(path)}
        )
    manifest = {
        "build_id": "p",
        "build_identity": "parent-identity",
        "accepted_by_split": {"train": 3},
        "selection": {"kept": 4},
        "shards": shards,
    }
    (parent / "manifest.json").write_text(json.dumps(manifest))
    return parent


def _run(
    tmp_path: Path, parent: Path, excluded: list[str], out: str = "succ", commit: str | None = None
) -> subprocess.CompletedProcess:
    excl = tmp_path / f"{out}-excl.json"
    excl.write_text(json.dumps(excluded))
    return subprocess.run(
        [
            sys.executable,
            "-I",
            str(TOOL),
            str(parent),
            str(tmp_path / out),
            "--exclude",
            str(excl),
            "--criteria-commit",
            commit or _head(),
            "--build-id",
            "s",
        ],
        capture_output=True,
        text=True,
    )


def test_successor_gets_derived_identity_and_parent_is_untouched(tmp_path):
    parent = _parent(tmp_path)
    before = {p.name: sha256_file(p) for p in (parent / "shards").iterdir()}
    result = _run(tmp_path, parent, ["d2"])
    assert result.returncode == 0, result.stderr
    manifest = json.loads((tmp_path / "succ" / "manifest.json").read_text())
    assert manifest["build_identity"] != "parent-identity"
    assert manifest["build_identity"] == successor_identity(manifest["successor_of"])
    assert successor_problem(manifest) is None
    assert manifest["accepted_by_split"] == {"train": 2, "validation": 1}
    assert manifest["inherited_from_parent"] == ["selection"]
    assert before == {p.name: sha256_file(p) for p in (parent / "shards").iterdir()}
    lineage = manifest["successor_of"]
    assert lineage["criteria_commit"] == _head()
    assert lineage["derived_by"]["tool"] == "tools/successor_manifest.py"
    assert set(lineage["derived_by"]) >= {"tool", "commit", "dirty", "package_version"}
    assert "derived_by" not in LINEAGE_KEYS
    assert successor_identity({k: lineage[k] for k in LINEAGE_KEYS}) == manifest["build_identity"]


@pytest.mark.parametrize(
    "commit",
    ["c0ffee", "869a208", "0" * 40, "G" * 40, "HEAD"],
    ids=["short", "abbrev", "unknown", "hex?", "ref"],
)
def test_criteria_commit_must_be_a_full_existing_sha(tmp_path, commit):
    if commit == "869a208":
        commit = _head()[:7]  # an abbreviation of a real commit is still refused
    result = _run(tmp_path, _parent(tmp_path), ["d2"], commit=commit)
    assert result.returncode != 0 and "--criteria-commit" in result.stderr
    assert not (tmp_path / "succ").exists() and not (tmp_path / "succ.partial").exists()


def test_committed_v022_identity_recomputes_from_its_lineage():
    """derived_by sits outside LINEAGE_KEYS, so the launch identity is unchanged by the new field."""
    assert sha256_file(V022_MANIFEST) == V022_MANIFEST_SHA256  # the committed copy is the HDD manifest
    manifest = json.loads(V022_MANIFEST.read_text())
    lineage = manifest["successor_of"]
    assert manifest["build_identity"] == V022_IDENTITY == successor_identity(lineage)
    assert successor_problem(manifest) is None
    assert successor_identity({**lineage, "derived_by": {"commit": "anything"}}) == V022_IDENTITY


def test_excluded_id_absent_from_parent_fails_closed(tmp_path):
    result = _run(tmp_path, _parent(tmp_path), ["d2", "not-in-parent"])
    assert result.returncode != 0 and "not in the parent build" in result.stderr
    assert not (tmp_path / "succ").exists()


@pytest.mark.parametrize(
    ("identity", "fragment"), [("parent-identity", "parent's build identity"), ("other", "!= derived")]
)
def test_successor_problem_rejects_inherited_or_unrelated_identity(identity, fragment):
    lineage = {
        "build_identity": "parent-identity",
        "manifest_sha256": "m",
        "exclusion_list_sha256": "e",
        "criteria_commit": "c",
    }
    assert fragment in successor_problem({"build_identity": identity, "successor_of": lineage})
    assert successor_problem({"build_identity": "x"}) is None
