"""Resumable, checksum-verified downloads against a local HTTP server."""

import hashlib
import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import ClassVar

import pytest

from j1data.acquire import download_file, manifest_identity, verify_acquired_source
from j1data.paths import DataLayout
from j1data.registry import Source

PAYLOAD = bytes(range(256)) * 4096  # 1 MiB


class _Handler(BaseHTTPRequestHandler):
    fail_after: int | None = None
    ranges_seen: ClassVar[list[str]] = []

    def do_GET(self) -> None:
        start = 0
        header = self.headers.get("Range")
        if header:
            _Handler.ranges_seen.append(header)
            start = int(header.removeprefix("bytes=").split("-")[0])
            self.send_response(206)
        else:
            self.send_response(200)
        body = PAYLOAD[start:]
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if _Handler.fail_after is not None:
            cut = _Handler.fail_after
            _Handler.fail_after = None
            self.wfile.write(body[:cut])
            self.wfile.flush()
            self.connection.close()
            return
        self.wfile.write(body)

    def log_message(self, *args: object) -> None:
        pass


@pytest.fixture
def server() -> Iterator[str]:
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}/file.bin"
    httpd.shutdown()


def test_interrupted_download_resumes_with_range(server: str, tmp_path: Path) -> None:
    _Handler.ranges_seen = []
    _Handler.fail_after = 300_000
    target = tmp_path / "file.bin"
    digest = download_file(server, target, len(PAYLOAD), hashlib.sha256(PAYLOAD).hexdigest(), attempts=4)
    assert target.read_bytes() == PAYLOAD
    assert digest == hashlib.sha256(PAYLOAD).hexdigest()
    assert _Handler.ranges_seen and _Handler.ranges_seen[0].startswith("bytes=")


def test_partial_file_from_a_previous_run_is_resumed(server: str, tmp_path: Path) -> None:
    _Handler.ranges_seen = []
    target = tmp_path / "file.bin"
    (tmp_path / "file.bin.part").write_bytes(PAYLOAD[:123_456])
    download_file(server, target, len(PAYLOAD), hashlib.sha256(PAYLOAD).hexdigest())
    assert target.read_bytes() == PAYLOAD
    assert _Handler.ranges_seen == ["bytes=123456-"]


def test_checksum_mismatch_is_fatal_and_leaves_no_file(server: str, tmp_path: Path) -> None:
    target = tmp_path / "file.bin"
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        download_file(server, target, len(PAYLOAD), "0" * 64)
    assert not target.exists() and not (tmp_path / "file.bin.part").exists()


def test_verified_existing_file_is_kept(server: str, tmp_path: Path) -> None:
    target = tmp_path / "file.bin"
    target.write_bytes(PAYLOAD)
    _Handler.ranges_seen = []
    download_file(server, target, len(PAYLOAD), hashlib.sha256(PAYLOAD).hexdigest())
    assert _Handler.ranges_seen == []


def fixture_source() -> Source:
    return Source(
        "fixture",
        {
            "acquisition": {
                "mechanism": "huggingface",
                "repo": "owner/dataset",
                "revision": "a" * 40,
                "files": {"include": ["*.bin"], "exclude": []},
            },
            "license": {"state": "APPROVED"},
            "processing": {"priority": 1},
        },
    )


def fixture_manifest(payload: bytes) -> dict[str, object]:
    source = fixture_source()
    git_oid = hashlib.sha1(f"blob {len(payload)}\0".encode() + payload).hexdigest()
    return {
        "schema": "j1.acquisition-manifest.v1",
        "acquire_version": "j1.acquire.v1",
        "source_id": source.source_id,
        "mechanism": source.raw["acquisition"]["mechanism"],
        "repo": source.repo,
        "revision": source.revision,
        "selection": source.raw["acquisition"]["files"],
        "upstream_file_count": 1,
        "selected_file_count": 1,
        "selected_bytes": len(payload),
        "acquired_utc": "2026-10-08T00:00:00+00:00",
        "files": [
            {
                "path": "part.bin",
                "size": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "upstream_sha256": None,
                "git_blob_oid": git_oid,
                "retrieved_utc": "2026-10-08T00:00:00+00:00",
            }
        ],
    }


def test_manifest_identity_ignores_retrieval_timestamps_but_binds_file_hash() -> None:
    manifest = fixture_manifest(PAYLOAD)
    identity = manifest_identity(fixture_source(), manifest)
    timestamp_only = json.loads(json.dumps(manifest))
    timestamp_only["acquired_utc"] = "2026-10-09T00:00:00+00:00"
    timestamp_only["files"][0]["retrieved_utc"] = "2026-10-09T00:00:00+00:00"  # type: ignore[index]
    assert manifest_identity(fixture_source(), timestamp_only) == identity
    changed_file = json.loads(json.dumps(manifest))
    changed_file["files"][0]["sha256"] = "0" * 64  # type: ignore[index]
    assert manifest_identity(fixture_source(), changed_file) != identity


def test_raw_source_preflight_detects_changed_bytes_and_symlinks(tmp_path: Path) -> None:
    source = fixture_source()
    layout = DataLayout(tmp_path)
    raw = layout.raw_dir(source.source_id, source.revision)
    raw.mkdir(parents=True)
    path = raw / "part.bin"
    path.write_bytes(PAYLOAD)
    manifest = fixture_manifest(PAYLOAD)
    verified = verify_acquired_source(source, layout, manifest)
    assert verified["files"] == 1 and verified["bytes"] == len(PAYLOAD)

    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="acquired size mismatch"):
        verify_acquired_source(source, layout, manifest)

    path.unlink()
    (raw / "copy.bin").write_bytes(PAYLOAD)
    path.symlink_to(raw / "copy.bin")
    with pytest.raises(ValueError, match="symlink"):
        verify_acquired_source(source, layout, manifest)
