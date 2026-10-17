import json
import os
from pathlib import Path

import pytest

from j1bench import store
from j1bench.replicate import audit_verify


def _tree(root: Path) -> Path:
    (root / "a").mkdir(parents=True)
    (root / "a" / "x.bin").write_bytes(os.urandom(5000))
    (root / "y.txt").write_text("hello")
    return root


def _manifest(tmp_path: Path) -> tuple[dict, Path]:
    src = _tree(tmp_path / "src")
    return store.build_manifest(src, "art-1", "PRESERVE"), src


def test_manifest_self_hash_and_tamper(tmp_path):
    m, _ = _manifest(tmp_path)
    store.check_manifest(m)
    bad = json.loads(json.dumps(m))
    bad["files"][0]["sha256"] = "0" * 64
    with pytest.raises(store.StoreError, match="self-hash"):
        store.check_manifest(bad)


def test_manifest_rejects_unsafe_paths(tmp_path):
    m, _ = _manifest(tmp_path)
    body = {k: v for k, v in m.items() if k != "manifest_sha256"}
    body["files"] = [{"path": "../evil", "size": 1, "sha256": "0" * 64}]
    body["manifest_sha256"] = store.hashlib.sha256(store._canonical(body)).hexdigest()
    with pytest.raises(store.StoreError, match="unsafe path"):
        store.check_manifest(body)


def test_retain_until_requires_date(tmp_path):
    with pytest.raises(store.StoreError):
        store.build_manifest(_tree(tmp_path / "s"), "x", "RETAIN_UNTIL")


def test_local_push_pull_roundtrip_and_idempotence(tmp_path):
    m, src = _manifest(tmp_path)
    dest = store.LocalDir(tmp_path / "archive", reserve_bytes=0)
    (tmp_path / "archive").mkdir()
    rec = store.push(m, src, dest, tmp_path / "r")
    assert rec["level"] == "BYTES_VERIFIED" and rec["uploaded_files"] == 2
    again = store.push(m, src, dest, tmp_path / "r")
    assert again["uploaded_files"] == 0 and again["already_present"] == 2
    out = tmp_path / "restore"
    store.pull(m, dest, out, tmp_path / "r", reserve_bytes=0)
    assert store.verify_tree(m, out) == []
    assert audit_verify(tmp_path / "r")[0]
    assert store.status(tmp_path / "r")[f"art-1@{m['manifest_sha256'][:12]}"][dest.label] == "BYTES_VERIFIED"


def test_corrupted_source_refused(tmp_path):
    m, src = _manifest(tmp_path)
    (src / "y.txt").write_text("HELLO")
    (tmp_path / "archive").mkdir()
    with pytest.raises(store.StoreError, match="source does not match"):
        store.push(m, src, store.LocalDir(tmp_path / "archive", 0), tmp_path / "r")


def test_conflicting_destination_bytes_refused(tmp_path):
    m, src = _manifest(tmp_path)
    arc = tmp_path / "archive"
    arc.mkdir()
    (arc / "y.txt").write_text("other")
    with pytest.raises(store.StoreError, match="conflicting bytes"):
        store.push(m, src, store.LocalDir(arc, 0), tmp_path / "r")
    assert (arc / "y.txt").read_text() == "other"


def test_corrupted_destination_detected_on_pull(tmp_path):
    m, src = _manifest(tmp_path)
    arc = tmp_path / "archive"
    arc.mkdir()
    dest = store.LocalDir(arc, 0)
    store.push(m, src, dest, tmp_path / "r")
    (arc / "a" / "x.bin").write_bytes(b"\0" * 5000)  # same size, different bytes
    with pytest.raises(store.StoreError, match="do not match"):
        store.pull(m, dest, tmp_path / "out", tmp_path / "r", reserve_bytes=0)
    assert not (tmp_path / "out").exists()


def test_missing_object_and_unmounted_destination(tmp_path):
    m, src = _manifest(tmp_path)
    arc = tmp_path / "archive"
    arc.mkdir()
    dest = store.LocalDir(arc, 0)
    store.push(m, src, dest, tmp_path / "r")
    (arc / "y.txt").unlink()
    with pytest.raises(store.StoreError, match="missing"):
        store.pull(m, dest, tmp_path / "out", tmp_path / "r", reserve_bytes=0)
    with pytest.raises(store.StoreError, match="unavailable"):
        store.push(m, src, store.LocalDir(tmp_path / "not-mounted", 0), tmp_path / "r")


def test_full_destination_refused(tmp_path):
    m, src = _manifest(tmp_path)
    (tmp_path / "archive").mkdir()
    huge = store.LocalDir(tmp_path / "archive", reserve_bytes=1 << 62)
    with pytest.raises(store.StoreError, match="insufficient capacity"):
        store.push(m, src, huge, tmp_path / "r")


def test_interrupted_pull_resumes(tmp_path, monkeypatch):
    m, src = _manifest(tmp_path)
    arc = tmp_path / "archive"
    arc.mkdir()
    dest = store.LocalDir(arc, 0)
    store.push(m, src, dest, tmp_path / "r")
    calls = {"n": 0}
    real = dest.fetch

    def flaky(rel, dst):
        calls["n"] += 1
        if calls["n"] == 2:
            raise ConnectionError("link dropped")
        real(rel, dst)

    monkeypatch.setattr(dest, "fetch", flaky)
    with pytest.raises(ConnectionError):
        store.pull(m, dest, tmp_path / "out", tmp_path / "r", reserve_bytes=0)
    assert not (tmp_path / "out").exists()
    rec = store.pull(m, dest, tmp_path / "out", tmp_path / "r", reserve_bytes=0)
    assert rec["reused_partial"] == 1


def test_cli_failure_is_audited(tmp_path):
    m, src = _manifest(tmp_path)
    mp = tmp_path / "m.json"
    mp.write_text(json.dumps(m))
    rc = store.main(["--receipts", str(tmp_path / "r"), "push", str(mp), f"local:{tmp_path / 'nope'}", str(src)])
    assert rc == 2
    lines = (tmp_path / "r" / "audit.log").read_text().splitlines()
    assert json.loads(lines[-1])["event"] == "FAILED"


class _FakeInfo:
    def __init__(self, path, size, sha):
        self.path, self.size = path, size
        self.lfs = type("L", (), {"sha256": sha})() if sha else None


class _FakeApi:
    """Stores bytes in memory; reports LFS hashes only for files above 10 bytes."""

    def __init__(self):
        self.files: dict[str, bytes] = {}
        self.private = True
        self.fail_next_commit = False

    def create_repo(self, *_a, **_k):
        pass

    def repo_info(self, *_a, **_k):
        return type("I", (), {"private": self.private})()

    def get_paths_info(self, _repo_id, paths, repo_type=None, expand=False):  # noqa: ARG002
        out = []
        for p in paths:
            if p in self.files:
                b = self.files[p]
                out.append(_FakeInfo(p, len(b), store.hashlib.sha256(b).hexdigest() if len(b) > 10 else None))
        return out

    def create_commit(self, _repo_id, ops, commit_message="", repo_type=None):  # noqa: ARG002
        if self.fail_next_commit:
            self.fail_next_commit = False
            raise ConnectionError("upload interrupted")
        for op in ops:
            self.files[op.path_in_repo] = Path(op.path_or_fileobj).read_bytes()
        return type("C", (), {"oid": "c" * 40})()


def test_hf_push_levels_resume_and_public_refusal(tmp_path):
    m, src = _manifest(tmp_path)
    api = _FakeApi()
    dest = store.HFRepo("Org/repo", "dataset", "p", api=api)
    dest.attempts, dest.backoff_s = 1, 0.0
    api.fail_next_commit = True
    with pytest.raises(ConnectionError):
        store.push(m, src, dest, tmp_path / "r")
    dest.attempts = 2
    api.fail_next_commit = True  # retried within the push this time
    rec = store.push(m, src, dest, tmp_path / "r")
    assert len(rec["retries"]) == 1
    # y.txt is 5 bytes: the fake keeps no hash for it, so the receipt cannot claim REMOTE_HASH.
    assert rec["level"] == "COMMITTED" and rec["unhashed_paths"] == ["y.txt"]
    api.files["p/a/x.bin"] = b"x" * 5000
    with pytest.raises(store.StoreError, match="conflicting"):
        store.push(m, src, dest, tmp_path / "r")
    api.private = False
    with pytest.raises(store.StoreError, match="public"):
        store.push(m, src, store.HFRepo("Org/other", "dataset", api=api), tmp_path / "r")
    assert store.hf_bytes(tmp_path / "r") == {"Org": m["total_bytes"]}
