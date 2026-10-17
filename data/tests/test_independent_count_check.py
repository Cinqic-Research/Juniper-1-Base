"""tools/independent_count_check.py loads the frozen tokenizer with jtok.load semantics (D-06)."""

import json
import subprocess
import sys
from pathlib import Path

from test_full_recount import TEXTS, _build

ROOT = Path(__file__).resolve().parents[1]


def test_sampled_check_passes_on_a_literal_control_spelling(tmp_path):
    assert any("<|user|>" in t for t in TEXTS["train"])  # the fixture holds a literal control spelling
    acc, out, tok = _build(tmp_path, {"build_identity": "fixture-identity"})
    record = tmp_path / "check.json"
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            str(ROOT / "tools/independent_count_check.py"),
            str(acc),
            "--cand",
            f"frozen={out}={tok}",
            "--rate",
            "1.0",
            "--min-per-stratum",
            "1",
            "--out",
            str(record),
        ],
        capture_output=True,
        text=True,
    )
    rec = json.loads(record.read_text())
    train = rec["candidates"]["frozen"]["splits"]["train"]
    assert train["sample"]["checked"] == 400  # every train document, including the literal-control one
    assert train["sample"] == {
        "checked": 400,
        "missing": 0,
        "count_mismatches": 0,
        "id_mismatches": 0,
        "examples": [],
    }, train["sample"]
    assert result.returncode == 0 and rec["ok"], result.stdout + result.stderr
