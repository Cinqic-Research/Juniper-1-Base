"""tools/full_recount.py: a clean shard set passes; corrupted, incomplete or mismatched inputs fail."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from j1data import tokshards, tokstage

ROOT = Path(__file__).resolve().parents[1]
TEXTS = {
    "train": [f"def f{i}(x):\n    return x * {i}  # doc {i}\n" for i in range(200)]
    + [
        f"Paragraph {i}: the colour of the sky is blue. <|user|> literal"
        if i == 7
        else f"Paragraph {i}: sky."
        for i in range(200)
    ],
    "validation": ["Validation text about behavior.", "    print('valid')\n"],
    "test": ["Test document."],
}


def _build(tmp_path: Path, manifest_extra: dict | None = None) -> tuple[Path, Path, Path]:
    root = tmp_path / "data"
    acc = root / "accepted" / "b1"
    shards, counts = [], {}
    for split, texts in TEXTS.items():
        counts[split] = len(texts)
        for part in range(2):
            rows = texts[part::2]
            if not rows:
                continue
            rel = f"{split}/src{part}/s.parquet"
            (acc / rel).parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(
                pa.table(
                    {
                        "doc_id": [f"{split}-{part}-{k}" for k in range(len(rows))],
                        "text": rows,
                        "source_id": [f"src{part}"] * len(rows),
                        "domain": ["computing"] * len(rows),
                        "sha256": [hashlib.sha256(t.encode()).hexdigest() for t in rows],
                        "n_bytes": [len(t.encode()) for t in rows],
                        "split": [split] * len(rows),
                    }
                ),
                acc / rel,
            )
            shards.append({"file": rel, "split": split, "source_id": f"src{part}"})
    manifest = {"shards": shards, "accepted_by_split": counts, **(manifest_extra or {})}
    (acc / "manifest.json").write_text(json.dumps(manifest))
    tokstage.run(root, "b1", rate=1.0, sizes=[384], out_dir=tmp_path / "tok", stem_alphabet=False)
    frozen = tmp_path / "frozen"
    tokstage.freeze(tmp_path / "tok" / "j1tok-0k", frozen, "test", {})
    out = tmp_path / "shards"
    tokshards.build(root, "b1", frozen, out)
    return acc, out, frozen / "tokenizer.json"


def _recount(acc, out, tok, tmp, rate="1.0"):
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()  # noqa: E731
    r = subprocess.run(
        [sys.executable, str(ROOT / "tools/full_recount.py"), str(acc), str(out), str(tok),
         "--expect-tokenizer-sha256", sha(tok), "--expect-manifest-sha256", sha(acc / "manifest.json"),
         "--out", str(tmp), "--workers", "1", "--bpe-sample-rate", rate],
        capture_output=True, text=True,
    )  # fmt: skip
    rec = tmp / "recount.json"
    return r.returncode, (json.loads(rec.read_text()) if rec.exists() else None), r.stdout + r.stderr


def test_clean_set_passes_with_full_bpe_cross_check(tmp_path):
    acc, out, tok = _build(tmp_path)
    rc, rec, log = _recount(acc, out, tok, tmp_path / "rc")
    assert rc == 0 and rec["pass"], log
    a = rec["phase_a"]
    assert a["bpe_sample_checked"] == a["documents"] == 403 and a["bpe_sample_mismatch"] == 0
    assert a["literal_control_docs"] == 1 and a["literal_control_docs_plain_load_gives_control_ids"] == 1
    assert rec["splits"]["train"]["documents"] == 400
    # D-01: the record names document, separator and stream totals; phase_a.tokens is document tokens.
    totals = rec["totals"]
    assert rec["schema"] == "j1.full-recount.v2"
    assert totals["document_tokens_total"] == a["tokens"]
    assert totals["separator_tokens_total"] == a["documents"] == 403
    assert totals["stream_tokens_total"] == a["tokens"] + 403
    assert totals["stream_tokens_total"] == sum(s["stream_tokens"] for s in rec["splits"].values())
    assert totals["consistent_with_phase_a"] is True


def test_corrupted_token_byte_fails(tmp_path):
    acc, out, tok = _build(tmp_path)
    meta = json.loads((out / "shards.json").read_text())
    f = out / meta["splits"]["train"]["token_files"][0]["file"]
    b = bytearray(f.read_bytes())
    b[10] ^= 1
    f.write_bytes(bytes(b))
    rc, rec, _ = _recount(acc, out, tok, tmp_path / "rc", rate="0")
    errs = " ".join(rec["splits"]["train"]["errors"])
    assert rc == 1 and not rec["pass"] and "sha256 differs" in errs and "stored IDs differ" in errs


def test_missing_document_and_wrong_tokenizer_fail(tmp_path):
    acc, out, tok = _build(tmp_path)
    p = acc / "train/src1/s.parquet"
    t = pq.read_table(p)
    pq.write_table(t.slice(1), p)
    rc, rec, _ = _recount(acc, out, tok, tmp_path / "rc", rate="0")
    assert rc == 1 and any("not in accepted text" in e for e in rec["splits"]["train"]["errors"])
    other = tmp_path / "other.json"
    other.write_text(tok.read_text().replace('"<|reserved_63|>"', '"<|reserved_6x|>"'))
    rc, rec, log = _recount(acc, out, other, tmp_path / "rc2", rate="0")
    assert rc == 2 and "shards.json is bound to a different tokenizer" in log


def test_phase_a_reuse_recomputes_a_changed_input_file(tmp_path):
    """D-07: a resumed --out must not reuse a per-file result whose input bytes changed."""
    acc, out, tok = _build(tmp_path)
    rc, rec, log = _recount(acc, out, tok, tmp_path / "rc")
    assert rc == 0 and rec["pass"], log
    key = hashlib.sha256(b"train/src1/s.parquet").hexdigest()[:20]
    summary = tmp_path / "rc" / "files" / f"{key}.summary.json"
    before = json.loads(summary.read_text())
    # Same relative path, different content (one document's text, hash and byte count rewritten).
    p = acc / "train/src1/s.parquet"
    t = pq.read_table(p)
    texts = t.column("text").to_pylist()
    texts[0] = texts[0] + " changed after the first recount"
    t = t.set_column(t.column_names.index("text"), "text", pa.array(texts))
    t = t.set_column(
        t.column_names.index("sha256"),
        "sha256",
        pa.array([hashlib.sha256(x.encode()).hexdigest() for x in texts]),
    )
    t = t.set_column(t.column_names.index("n_bytes"), "n_bytes", pa.array([len(x.encode()) for x in texts]))
    pq.write_table(t, p)
    (acc / "manifest.json").write_text((acc / "manifest.json").read_text())  # manifest hash unchanged
    rc, rec, _ = _recount(acc, out, tok, tmp_path / "rc", rate="0")
    after = json.loads(summary.read_text())
    assert after["file_sha256"] != before["file_sha256"]
    assert rc == 1 and not rec["pass"]
    assert any("stored IDs differ" in e or "n_tokens" in e for e in rec["splits"]["train"]["errors"])
