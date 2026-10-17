"""tools/successor_prefix.py: --check-against names both inputs and the compared token count (D-11)."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

from test_full_recount import _build

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "successor_prefix.py"


def _tool(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-I", str(TOOL), *map(str, args)], capture_output=True, text=True)


def test_check_record_is_self_describing(tmp_path):
    acc, shards, _tok = _build(tmp_path)
    excl = tmp_path / "excl.json"
    excl.write_text("[]")
    prefix = tmp_path / "prefix-1k"
    result = _tool(shards, acc / "train", prefix, "--max-bytes", "1000", "--exclude", excl)
    assert result.returncode == 0, result.stderr
    record = tmp_path / "check.json"
    result = _tool(shards, acc / "train", prefix, "--check-against", shards, "--record", record)
    assert result.returncode == 0, result.stderr
    rec = json.loads(record.read_text())
    assert rec == json.loads(result.stdout)
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()  # noqa: E731
    p = json.loads((prefix / "shards.json").read_text())
    assert rec["ok"] is True
    assert rec["prefix"] == "prefix-1k" and rec["official"] == shards.name
    assert rec["prefix_shards_json_sha256"] == sha(prefix / "shards.json")
    assert rec["official_shards_json_sha256"] == sha(shards / "shards.json")
    assert rec["compared_train_tokens"] == p["splits"]["train"]["stream_tokens"] > 0
    assert rec["prefix_train_documents"] == p["splits"]["train"]["documents"] > 0
    assert rec["official_train_stream_tokens"] > rec["compared_train_tokens"]
    for key in (
        "validation_token_files_equal",
        "validation_doc_index_equal",
        "train_prefix_equals_official_head",
    ):
        assert rec[key] is True

    # A damaged official head is detected and the record still names both inputs.
    o = json.loads((shards / "shards.json").read_text())
    f = shards / o["splits"]["train"]["token_files"][0]["file"]
    b = bytearray(f.read_bytes())
    b[2] ^= 1
    f.write_bytes(bytes(b))
    result = _tool(shards, acc / "train", prefix, "--check-against", shards)
    rec = json.loads(result.stdout)
    assert result.returncode == 1 and rec["ok"] is False
    assert rec["train_prefix_equals_official_head"] is False and rec["validation_token_files_equal"] is True
    assert rec["official_shards_json_sha256"] == sha(shards / "shards.json")
