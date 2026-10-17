"""Source adapters: turn pinned raw files into ``RawRecord`` streams.

Adapters own source-specific selection (namespaces, sites, per-document
licenses, vendored files) and identity (record keys, split groups). They do
not filter on quality; that is the job of the shared processing stage.
"""

from __future__ import annotations

import fnmatch
import gzip
import json
import re
import subprocess
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import pyarrow.parquet as pq

from j1data.paths import safe_child
from j1data.registry import Source

ROWS_PER_PARQUET_UNIT = 100_000


@dataclass(frozen=True)
class UnitSpec:
    source_id: str
    unit_id: str
    file: str
    row_start: int = 0
    row_end: int = -1


@dataclass
class RawRecord:
    key: str
    text: str
    group: str
    meta: dict[str, Any] = field(default_factory=dict)
    path: str | None = None
    skip_reason: str | None = None
    kind: str | None = None


def _dig(record: dict[str, Any], dotted: str) -> Any:
    value: Any = record
    for part in dotted.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(part)
    return value


def url_host(url: str | None) -> str:
    if not url:
        return ""
    host = urlsplit(url).hostname or ""
    return host.removeprefix("www.")


def _license_skip(source: Source, record: dict[str, Any]) -> str | None:
    field_name = source.raw["license"].get("per_document_license_field")
    if not field_name:
        return None
    value = _dig(record, field_name)
    allowed = source.raw["license"].get("allowed_document_licenses", [])
    if not isinstance(value, str):
        return "license.document_not_allowed"
    # Some upstream metadata uses broad strings such as "Creative Commons -
    # Attribution - *" or "CC BY*". Reject NC and ND terms before applying
    # those patterns so a wildcard cannot admit a restricted variant.
    restricted = re.search(
        r"non[\s-]*commercial|no[\s-]*derivatives|"
        r"(?<![a-z])(?:cc[\s-]*)?by[\s-]*n[cd](?:[\s-]*sa)?(?![a-z])",
        value,
        flags=re.IGNORECASE,
    )
    if restricted or not any(fnmatch.fnmatchcase(value, pattern) for pattern in allowed):
        return "license.document_not_allowed"
    return None


class Adapter:
    def __init__(self, source: Source, raw_dir: Path, manifest: dict[str, Any]) -> None:
        self.source = source
        self.raw_dir = raw_dir
        self.manifest = manifest
        self.options = source.options
        if "drop_exact_lines" in self.options:
            self.options["drop_exact_lines"] = frozenset(self.options["drop_exact_lines"])

    def units(self) -> list[UnitSpec]:
        raise NotImplementedError

    def read(self, unit: UnitSpec, start_at: int | None = None) -> Iterator[RawRecord]:
        raise NotImplementedError


class HFParquetAdapter(Adapter):
    """Parquet shards with one document per row (FineWeb-Edu, FineMath)."""

    def units(self) -> list[UnitSpec]:
        units = []
        # Units are runs of whole row groups so that no row group is decoded twice.
        for entry in self.manifest["files"]:
            metadata = pq.ParquetFile(safe_child(self.raw_dir, entry["path"])).metadata
            stem = Path(entry["path"]).name.removesuffix(".parquet")
            start = row = 0
            for group_index in range(metadata.num_row_groups):
                row += metadata.row_group(group_index).num_rows
                if row - start >= ROWS_PER_PARQUET_UNIT or group_index == metadata.num_row_groups - 1:
                    units.append(
                        UnitSpec(self.source.source_id, f"{stem}.r{start:08d}", entry["path"], start, row)
                    )
                    start = row
        return units

    def read(self, unit: UnitSpec, start_at: int | None = None) -> Iterator[RawRecord]:
        text_column = self.options.get("text_column", "text")
        key_columns = self.options.get("key_columns", [self.options.get("key_column", "id")])
        meta_columns = self.options.get("meta_columns", [])
        columns = sorted({text_column, *key_columns, *meta_columns})
        parquet = pq.ParquetFile(safe_child(self.raw_dir, unit.file))
        lower = max(unit.row_start, unit.row_start if start_at is None else start_at)
        groups, first_row, row = [], None, 0
        for group_index in range(parquet.metadata.num_row_groups):
            group_rows = parquet.metadata.row_group(group_index).num_rows
            if row + group_rows > lower and row < unit.row_end:
                groups.append(group_index)
                first_row = row if first_row is None else first_row
            row += group_rows
        absolute = first_row or 0
        for batch in parquet.iter_batches(
            batch_size=1024, row_groups=groups, columns=columns, use_threads=False
        ):
            for record in batch.to_pylist():
                absolute += 1
                if not lower <= absolute - 1 < unit.row_end:
                    continue
                meta = {name: record[name] for name in meta_columns}
                if isinstance(meta.get("metadata"), dict):
                    meta = {**meta.pop("metadata"), **meta}
                url = meta.get("url")
                key = "|".join(str(record[c]) for c in key_columns)
                text = record[text_column]
                yield RawRecord(
                    key=key,
                    text=text if isinstance(text, str) else "",
                    group=f"host:{url_host(url)}" if url else f"key:{key}",
                    meta=meta,
                )


class DolmaJsonlAdapter(Adapter):
    """Common Pile Dolma-format ``*.json.gz`` files (one JSON document per line)."""

    def units(self) -> list[UnitSpec]:
        return [
            UnitSpec(self.source.source_id, Path(e["path"]).name.removesuffix(".json.gz"), e["path"])
            for e in self.manifest["files"]
        ]

    def read(self, unit: UnitSpec, start_at: int | None = None) -> Iterator[RawRecord]:
        lower = max(unit.row_start, unit.row_start if start_at is None else start_at)
        with gzip.open(safe_child(self.raw_dir, unit.file), "rt", encoding="utf-8", newline="") as handle:
            for line_number, line in enumerate(handle):
                if line_number < lower:
                    continue
                if unit.row_end >= 0 and line_number >= unit.row_end:
                    break
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    yield RawRecord(
                        key=f"{unit.file}:{line_number}",
                        text="",
                        group="",
                        skip_reason="extraction.json_decode",
                    )
                    continue
                yield self._record(record, unit, line_number)

    def _record(self, record: dict[str, Any], unit: UnitSpec, line_number: int) -> RawRecord:
        meta = dict(record.get("metadata") or {})
        for name in ("id", "created", "added", "source"):
            if name in record:
                meta[f"doc_{name}"] = record[name]
        url = meta.get("url")
        key = str(url or record.get("id") or f"{unit.file}:{line_number}")
        if self.source.raw["family"] == "code":
            key = str(meta.get("blob_id") or record.get("id"))
        text = record.get("text")
        text = text if isinstance(text, str) else ""
        drop = self.options.get("drop_exact_lines")
        if drop and text:
            kept = [line for line in text.split("\n") if line.strip() not in drop]
            removed = text.count("\n") + 1 - len(kept)
            if removed:
                meta["removed_boilerplate_lines"] = removed
                text = "\n".join(kept)
        raw = RawRecord(key=key, text=text, group=self._group(meta, key), meta=meta)
        if self.source.raw["family"] == "code":
            raw.path = f"{meta.get('repo_name', '')}{meta.get('path', '')}"
            if meta.get("language") in self.options.get("text_languages", ()):
                raw.kind = "technical"
        raw.skip_reason = self._skip(record, meta)
        return raw

    def _group(self, meta: dict[str, Any], key: str) -> str:
        family = self.source.raw["family"]
        if family == "code" and meta.get("repo_name"):
            return f"repo:{meta['repo_name']}"
        if family == "textbooks" and meta.get("book_url"):
            return f"book:{meta['book_url']}"
        return f"key:{key}"

    def _skip(self, record: dict[str, Any], meta: dict[str, Any]) -> str | None:
        options = self.options
        if "allowed_wikis" in options and meta.get("wiki") not in options["allowed_wikis"]:
            return "selection.wiki"
        if (
            "allowed_namespaces" in options
            and str(meta.get("namespace")) not in options["allowed_namespaces"]
        ):
            return "selection.namespace"
        site = str(meta.get("site", ""))
        if site in options.get("excluded_sites", []):
            return "selection.site"
        if options.get("exclude_meta_sites") and (site.startswith("meta.") or ".meta." in site):
            return "selection.meta_site"
        minimum = options.get("min_int_score")
        if minimum is not None and (record.get("int_score") is None or record["int_score"] < minimum):
            return "selection.edu_score"
        if meta.get("is_generated") is True:
            return "selection.generated_code"
        if meta.get("is_vendor") is True:
            return "selection.vendored_code"
        return _license_skip(self.source, record)


class CinqicGitAdapter(Adapter):
    """Allowlisted files read from a pinned commit of a local clone."""

    def __init__(self, source: Source, raw_dir: Path, manifest: dict[str, Any], clone: Path) -> None:
        super().__init__(source, raw_dir, manifest)
        self.clone = clone

    def units(self) -> list[UnitSpec]:
        return [UnitSpec(self.source.source_id, "all", "")]

    def read(self, unit: UnitSpec, start_at: int | None = None) -> Iterator[RawRecord]:
        deny = [t.lower() for t in self.options.get("deny_terms", [])]
        lower = max(unit.row_start, unit.row_start if start_at is None else start_at)
        upper = unit.row_end
        for index, entry in enumerate(self.manifest["files"]):
            if index < lower:
                continue
            if upper >= 0 and index >= upper:
                break
            oid = entry.get("git_blob_oid")
            if (
                not isinstance(oid, str)
                or len(oid) not in (40, 64)
                or any(c not in "0123456789abcdef" for c in oid)
            ):
                raise ValueError(f"invalid git blob id for {entry.get('path')!r}")
            blob = subprocess.run(
                ["git", "-C", str(self.clone), "cat-file", "blob", oid],
                check=True,
                capture_output=True,
            ).stdout
            key = f"{self.source.repo}:{entry['path']}"
            meta = {"repo": self.source.repo, "revision": self.source.revision, "path": entry["path"]}
            try:
                text = blob.decode("utf-8")
            except UnicodeDecodeError:
                yield RawRecord(key, "", f"repo:{self.source.repo}", meta, skip_reason="extraction.decode")
                continue
            lowered = text.lower()
            skip = "selection.deny_term" if any(term in lowered for term in deny) else None
            yield RawRecord(key, text, f"repo:{self.source.repo}", meta, path=entry["path"], skip_reason=skip)


def make_adapter(
    source: Source, raw_dir: Path, manifest: dict[str, Any], clone: Path | None = None
) -> Adapter:
    if source.adapter == "hf_parquet":
        return HFParquetAdapter(source, raw_dir, manifest)
    if source.adapter == "dolma_jsonl":
        return DolmaJsonlAdapter(source, raw_dir, manifest)
    if source.adapter == "cinqic_git":
        if clone is None:
            raise ValueError(f"{source.source_id}: a local clone is required")
        return CinqicGitAdapter(source, raw_dir, manifest, clone)
    raise ValueError(f"unknown adapter {source.adapter}")
