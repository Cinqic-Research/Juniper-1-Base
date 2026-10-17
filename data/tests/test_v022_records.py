"""Committed j1-base-v0.2.2 launch records agree with each other (D-05, D-12)."""

import hashlib
import json
from pathlib import Path

RECORDS = Path(__file__).resolve().parents[1] / "reports" / "j1-base-v0.2.2"


def _manifest_entries() -> dict[str, str]:
    entries = {}
    for line in (RECORDS / "MANIFEST.sha256").read_text().splitlines():
        digest, _, path = line.partition("  ")
        entries[path.removeprefix("./")] = digest
    return entries


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_hf_readback_receipt_matches_manifest_and_preservation_receipt():
    manifest = _manifest_entries()
    assert len(manifest) == 18
    receipt = json.loads((RECORDS / "hf_receipt.json").read_text())
    receipts = json.loads((RECORDS / "preservation_receipts.json").read_text())
    assert receipt["repo"] == receipts["hf"]["repo"] == "CinqicX/juniper1-corpus"
    assert receipt["commit"] == receipts["hf"]["commit"]
    assert receipt["all_ok"] is True and all(f["readback_ok"] is True for f in receipt["files"])
    by_path = {f["path_in_repo"].removeprefix("j1-base-v0.2.2/"): f["sha256"] for f in receipt["files"]}
    assert len(by_path) == len(receipt["files"]) == 19
    assert by_path.pop("MANIFEST.sha256") == _sha(RECORDS / "MANIFEST.sha256") == receipts["manifest_sha256"]
    assert by_path == manifest
    # The small files committed here are the bytes the receipt describes (size and SHA-256).
    described = {f["path_in_repo"].removeprefix("j1-base-v0.2.2/"): f for f in receipt["files"]}
    for name in ("MANIFEST.sha256", "accepted/manifest.json", "accepted/excluded_doc_ids_union.json"):
        local = RECORDS / Path(name).name
        assert described[name]["size"] == local.stat().st_size
        assert described[name]["sha256"] == _sha(local)


def test_receipt_note_and_totals_note_bind_their_records():
    note = json.loads((RECORDS / "preservation_receipts_note.json").read_text())
    assert note["annotated_record_sha256"] == _sha(RECORDS / "preservation_receipts.json")
    assert note["manifest"]["sha256"] == _sha(RECORDS / "MANIFEST.sha256")
    manifest = _manifest_entries()
    assert set(note["canonical_hdd_root"]["files_absent"]) == {"./accepted/excluded_doc_ids_union.json"}
    assert note["canonical_hdd_root"]["files_present"] + len(note["canonical_hdd_root"]["files_absent"]) == 18
    assert manifest["accepted/excluded_doc_ids_union.json"] == manifest["accepted/excluded_doc_ids.json"]
    assert manifest["accepted/excluded_doc_ids_union.json"] == _sha(RECORDS / "excluded_doc_ids_union.json")
    assert (
        note["hf_readback_receipt"]["commit"]
        == json.loads((RECORDS / "hf_receipt.json").read_text())["commit"]
    )

    totals = json.loads((RECORDS / "recount_totals_note.json").read_text())
    record = json.loads((RECORDS / "full_recount_cand5-j1tok-24k.json").read_text())
    assert totals["annotated_record_sha256"] == _sha(RECORDS / "full_recount_cand5-j1tok-24k.json")
    assert totals["document_tokens"]["total"] == record["phase_a"]["tokens"] == 3_413_019_479
    for split in ("train", "validation", "test"):
        assert totals["document_tokens"][split] == record["splits"][split]["document_tokens"]
        assert totals["separator_tokens"][split] == record["splits"][split]["separator_tokens"]
        assert totals["stream_tokens_from_file_sizes"][split] == record["splits"][split]["stream_tokens"]
    assert totals["stream_tokens_from_file_sizes"]["total"] == 3_415_978_850
    assert totals["stream_tokens_from_file_sizes"]["total"] == (
        totals["document_tokens"]["total"] + totals["separator_tokens"]["total"]
    )


def test_docs_name_the_launch_corpus():
    """D-04: the qualification record, data card and recount doc carry the v0.2.2 identities and counts."""
    docs = Path(__file__).resolve().parents[1] / "docs"
    manifest = json.loads((RECORDS / "manifest.json").read_text())
    record = json.loads((RECORDS / "full_recount_cand5-j1tok-24k.json").read_text())
    identity = manifest["build_identity"]
    manifest_sha = _sha(RECORDS / "manifest.json")
    counts = {f"{manifest['accepted_by_split'][s]:,}" for s in ("train", "validation", "test")}
    for name in ("qualification_2026-10-10.md", "data_card.md", "full_recount_2026-10-10.md"):
        text = (docs / name).read_text()
        assert "j1-base-v0.2.2" in text, name
        assert identity in text or identity[:8] in text, name
        assert manifest_sha in text or manifest_sha[:8] in text, name
        assert all(c in text for c in counts), name
        assert f"{record['phase_a']['tokens']:,}" in text, name
    assert "superseded" in (docs / "qualification_2026-10-10.md").read_text()
