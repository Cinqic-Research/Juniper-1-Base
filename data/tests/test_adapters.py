import gzip
import json
from pathlib import Path

from j1data.adapters import DolmaJsonlAdapter, _license_skip
from j1data.registry import load_registry


def test_license_wildcards_reject_noncommercial_and_no_derivatives_variants() -> None:
    registry = load_registry()
    restricted = [
        ("libretexts", "Creative Commons - Attribution - NonCommercial - 4.0"),
        ("libretexts", "Creative Commons - Attribution - NoDerivatives - 4.0"),
        ("libretexts", "Creative Commons - Attribution - NonCommercial - NoDerivatives - 4.0"),
        ("arxiv-papers", "CC BY-NC 4.0"),
        ("arxiv-papers", "CC BY-ND 4.0"),
        ("arxiv-papers", "https://creativecommons.org/licenses/by-nc-sa/4.0/"),
    ]
    for source_id, license_value in restricted:
        source = registry[source_id]
        assert _license_skip(source, {"metadata": {"license": license_value}}) == (
            "license.document_not_allowed"
        )


def test_license_allowlist_keeps_attribution_and_sharealike_variants() -> None:
    registry = load_registry()
    values = [
        ("libretexts", "Creative Commons - Attribution - https://creativecommons.org/licenses/by/4.0/"),
        (
            "libretexts",
            "Creative Commons - Attribution Share-Alike - https://creativecommons.org/licenses/by-sa/4.0/",
        ),
        ("arxiv-papers", "CC BY 4.0"),
        ("arxiv-papers", "CC BY-SA 4.0"),
    ]
    for source_id, license_value in values:
        assert _license_skip(registry[source_id], {"metadata": {"license": license_value}}) is None


def write(path: Path, records: list[dict]) -> None:  # type: ignore[type-arg]
    with gzip.open(path, "wt") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def test_code_source_routes_markdown_to_text_rules_and_rejects_vendored(tmp_path: Path) -> None:
    source = load_registry()["stackv2-edu"]
    meta = {"license_type": "permissive", "repo_name": "o/r", "is_generated": False, "is_vendor": False}
    scored = {"int_score": 3}
    write(
        tmp_path / "f.json.gz",
        [
            {
                **scored,
                "id": "1",
                "text": "# Title\n\nSome prose.",
                "metadata": {**meta, "blob_id": "b1", "language": "Markdown", "path": "/README.md"},
            },
            {
                **scored,
                "id": "2",
                "text": "x = 1",
                "metadata": {**meta, "blob_id": "b2", "language": "Python", "path": "/a.py"},
            },
            {
                **scored,
                "id": "3",
                "text": "y",
                "metadata": {
                    **meta,
                    "blob_id": "b3",
                    "language": "Python",
                    "path": "/v.py",
                    "is_vendor": True,
                },
            },
            {
                **scored,
                "id": "4",
                "text": "z",
                "metadata": {**meta, "blob_id": "b4", "language": "Python", "license_type": "no_license"},
            },
        ],
    )
    adapter = DolmaJsonlAdapter(source, tmp_path, {"files": [{"path": "f.json.gz"}]})
    records = list(adapter.read(adapter.units()[0]))
    assert [r.kind for r in records[:2]] == ["technical", None]
    assert records[0].group == "repo:o/r" and records[0].key == "b1"
    assert [r.skip_reason for r in records] == [
        None,
        None,
        "selection.vendored_code",
        "license.document_not_allowed",
    ]


def test_wikimedia_selection_and_libretexts_chrome_removal(tmp_path: Path) -> None:
    registry = load_registry()
    lic = "Creative Commons - Attribution Share-Alike - https://creativecommons.org/licenses/by-sa/4.0/"
    write(
        tmp_path / "w.json.gz",
        [
            {
                "id": "1",
                "text": "Article",
                "metadata": {"wiki": "wikipedia.com", "namespace": "0", "license": lic, "url": "u1"},
            },
            {
                "id": "2",
                "text": "Talk",
                "metadata": {"wiki": "wikipedia.com", "namespace": "1", "license": lic, "url": "u2"},
            },
            {
                "id": "3",
                "text": "Entry",
                "metadata": {"wiki": "wiktionary.com", "namespace": "0", "license": lic, "url": "u3"},
            },
        ],
    )
    wiki = DolmaJsonlAdapter(registry["wikimedia"], tmp_path, {"files": [{"path": "w.json.gz"}]})
    assert [r.skip_reason for r in wiki.read(wiki.units()[0])] == [
        None,
        "selection.namespace",
        "selection.wiki",
    ]

    cc0 = "Creative Commons Zero - Public Domain - https://creativecommons.org/publicdomain/zero/1.0/"
    write(
        tmp_path / "l.json.gz",
        [
            {
                "id": "u",
                "text": "Skip to main content\n- Save as PDF\nReal line\n-\nMore",
                "metadata": {"license": cc0, "url": "u"},
            }
        ],
    )
    libre = DolmaJsonlAdapter(registry["libretexts"], tmp_path, {"files": [{"path": "l.json.gz"}]})
    (record,) = list(libre.read(libre.units()[0]))
    assert record.text == "Real line\nMore" and record.meta["removed_boilerplate_lines"] == 3
    assert record.skip_reason is None
