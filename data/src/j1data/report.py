"""Corpus statistics: stage accounting, mixture, distributions, concentration.

Everything here is computed from the build's own markers and shards; nothing
is typed in by hand. Raw-document and rejected-document token counts are not
measurable without tokenizing text that was never admitted, so rejected volume
is reported in bytes, with token *estimates* derived from each source's
measured accepted bytes-per-token ratio and labelled as estimates.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from j1data.hashing import sha256_file
from j1data.paths import DataLayout

STAGES = ("selection", "license", "extraction", "privacy", "contamination", "quality")


def process_summary(layout: DataLayout, build_id: str) -> dict[str, Any]:
    by_source: dict[str, dict[str, Any]] = {}
    for path in sorted((layout.work(build_id) / "units").glob("*/*.json")):
        marker = json.loads(path.read_text())
        entry = by_source.setdefault(
            marker["source_id"],
            {
                "units": 0,
                "docs_in": 0,
                "bytes_in": 0,
                "seconds": 0.0,
                "stages": Counter(),
                "bytes_by_stage": Counter(),
                "reasons": Counter(),
                "redactions": Counter(),
                "kinds": Counter(),
                "contamination_benchmarks": Counter(),
            },
        )
        entry["units"] += 1
        entry["docs_in"] += marker["docs_in"]
        entry["bytes_in"] += marker["bytes_in"]
        entry["seconds"] += marker["seconds"]
        for key in ("stages", "bytes_by_stage", "reasons", "redactions", "kinds", "contamination_benchmarks"):
            entry[key].update(marker[key])
    for entry in by_source.values():
        entry["mb_per_cpu_second"] = entry["bytes_in"] / 1e6 / max(entry["seconds"], 1e-9)
        for key in ("stages", "bytes_by_stage", "reasons", "redactions", "kinds", "contamination_benchmarks"):
            entry[key] = dict(sorted(entry[key].items(), key=lambda kv: (-kv[1], kv[0])))
    return dict(sorted(by_source.items()))


def _percentiles(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {}
    points = (1, 5, 25, 50, 75, 95, 99)
    return {f"p{p}": float(np.percentile(values, p)) for p in points} | {
        "mean": float(values.mean()),
        "max": float(values.max()),
    }


_GROUP_KEYS = (
    ("source", ("source_id",)),
    ("split", ("split",)),
    ("domain", ("domain",)),
    ("kind", ("kind",)),
    ("source_split", ("source_id", "split")),
    ("source_domain", ("source_id", "domain")),
)


def _accumulate(
    target: dict[str, dict[str, Counter[str]]],
    name: str,
    table: pa.Table,
    keys: tuple[str, ...],
    measures: list[str],
) -> None:
    grouped = table.group_by(list(keys)).aggregate([(m, "sum") for m in measures])
    columns = grouped.to_pydict()
    for row in range(grouped.num_rows):
        label = "|".join(str(columns[k][row]) for k in keys)
        for measure in measures:
            target[measure].setdefault(name, Counter())[label] += int(columns[f"{measure}_sum"][row])


def accepted_statistics(
    accepted_root: Path, counts_root: Path | None, manifest: dict[str, Any]
) -> dict[str, Any]:
    """Aggregate accepted metadata and, when available, provisional per-document token counts
    (Arrow group-by per shard; memory bounded by one shard plus per-source length arrays)."""
    totals: dict[str, dict[str, Counter[str]]] = defaultdict(dict)
    lengths: dict[str, list[np.ndarray]] = defaultdict(list)
    english: list[np.ndarray] = []
    spelling: Counter[str] = Counter()
    clusters: Counter[int] = Counter()
    tokenizer_names: list[str] = []
    columns = [
        "doc_id",
        "source_id",
        "split",
        "domain",
        "kind",
        "formats",
        "n_bytes",
        "us_spellings",
        "uk_spellings",
        "dup_cluster_size",
        "english_fw_frac",
    ]
    for shard in manifest["shards"]:
        table = pq.read_table(accepted_root / shard["file"], columns=columns)
        table = table.append_column("documents", pa.array(np.ones(table.num_rows, dtype=np.int64)))
        measures = ["documents", "n_bytes"]
        if counts_root is not None:
            counts = pq.read_table(counts_root / "counts" / shard["file"])
            if counts.column("doc_id").to_pylist() != table.column("doc_id").to_pylist():
                raise RuntimeError(f"{shard['file']}: count table does not align with the accepted shard")
            tokenizer_names = sorted(
                n.removeprefix("tokens_") for n in counts.column_names if n.startswith("tokens_")
            )
            for name in tokenizer_names:
                table = table.append_column(f"tokens_{name}", counts.column(f"tokens_{name}"))
                measures.append(f"tokens_{name}")
                lengths[f"{name}|{shard['source_id']}"].append(counts.column(f"tokens_{name}").to_numpy())
        for group, keys in _GROUP_KEYS:
            _accumulate(totals, group, table, keys, measures)
        parents = pc.list_parent_indices(table.column("formats"))
        if len(parents):
            exploded = (
                table.select(measures)
                .take(parents)
                .append_column("format", pc.list_flatten(table.column("formats")))
            )
            _accumulate(totals, "format", exploded, ("format",), measures)
        us = table.column("us_spellings").to_numpy()
        uk = table.column("uk_spellings").to_numpy()
        spelling.update(
            {"us": int((us > uk).sum()), "uk": int((uk > us).sum()), "neutral": int((us == uk).sum())}
        )
        sizes = np.minimum(table.column("dup_cluster_size").to_numpy(), 10)
        clusters.update(dict(zip(*np.unique(sizes, return_counts=True), strict=True)))
        english.append(table.column("english_fw_frac").to_numpy())
    documents = totals["documents"]
    result: dict[str, Any] = {
        "build_identity": manifest.get("build_identity"),
        "accepted_manifest_sha256": sha256_file(accepted_root / "manifest.json")
        if (accepted_root / "manifest.json").is_file()
        else None,
        "documents": {g: dict(c) for g, c in documents.items()},
        "bytes": {g: dict(c) for g, c in totals["n_bytes"].items()},
        "tokens": {n: {g: dict(c) for g, c in totals[f"tokens_{n}"].items()} for n in tokenizer_names},
        "spelling_variety": dict(spelling),
        "retained_cluster_size": {str(int(k)): int(v) for k, v in sorted(clusters.items())},
        "english_function_word_share": _percentiles(np.concatenate(english)) if english else {},
        "tokenizers": tokenizer_names,
        "token_length_percentiles": {
            key: _percentiles(np.concatenate(parts)) for key, parts in sorted(lengths.items())
        },
    }
    if tokenizer_names:
        name = tokenizer_names[-1]
        shares = np.array(list(totals[f"tokens_{name}"]["source"].values()), dtype=np.float64)
        shares = shares / shares.sum() if shares.sum() else shares
        result["source_concentration"] = {
            "tokenizer": name,
            "herfindahl_index": float((shares**2).sum()),
            "largest_source_share": float(shares.max()) if shares.size else 0.0,
        }
    return result


def rejected_token_estimates(
    process: dict[str, Any], stats: dict[str, Any], tokenizer: str
) -> dict[str, Any]:
    """Estimated tokens per rejection stage, from each source's accepted bytes per token."""
    estimates: dict[str, Any] = {}
    accepted_bytes = stats["bytes"]["source"]
    accepted_tokens = stats["tokens"][tokenizer]["source"]
    for source, entry in process.items():
        if not accepted_tokens.get(source):
            continue
        bytes_per_token = accepted_bytes[source] / accepted_tokens[source]
        estimates[source] = {
            "bytes_per_token": bytes_per_token,
            "raw_tokens_estimate": entry["bytes_in"] / bytes_per_token,
            "rejected_tokens_estimate_by_stage": {
                stage: entry["bytes_by_stage"].get(stage, 0) / bytes_per_token for stage in STAGES
            },
        }
    return estimates


_SOURCE_COLUMNS = (
    "Source",
    "Family",
    "License state",
    "Revision",
    "Files",
    "Raw bytes",
    "Raw docs",
    "Candidates",
    "Retrieved",
)


def sources_markdown(
    registry: dict[str, Any],
    considered: list[dict[str, Any]],
    acquisitions: dict[str, dict[str, Any]],
    process: dict[str, Any] | None,
) -> str:
    """Render the source registry, acquisition facts and processing counts as Markdown."""
    lines = [
        "# Source registry",
        "",
        "Generated by `j1data sources-doc` from `configs/sources.toml`, `configs/considered_sources.toml`,",
        "the acquisition manifests and the processing markers. Do not edit by hand.",
        "",
        "## Admitted sources",
        "",
        "| " + " | ".join(_SOURCE_COLUMNS) + " |",
        "|" + " --- |" * len(_SOURCE_COLUMNS),
    ]
    for source_id, source in sorted(registry.items(), key=lambda kv: kv[1].priority):
        raw = source.raw
        acquisition = acquisitions.get(source_id, {})
        counts = (process or {}).get(source_id, {})
        size = acquisition.get("selected_bytes")
        cells = [
            f"`{source_id}`",
            raw["family"],
            raw["license"]["state"],
            f"`{source.revision[:12]}`",
            str(acquisition.get("selected_file_count", "")),
            f"{size / 1e9:.2f} GB" if size is not None else "",
            str(counts.get("docs_in", "")),
            str(counts.get("stages", {}).get("candidate", "")),
            acquisition.get("acquired_utc", "")[:10],
        ]
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["", "## Licensing, attribution and risks", ""]
    for source_id, source in sorted(registry.items(), key=lambda kv: kv[1].priority):
        raw = source.raw
        lic = raw["license"]
        lines += [
            f"### {raw['source_name']} (`{source_id}`)",
            "",
            raw["description"],
            "",
            f"- Origin: `{source.repo}` at `{source.revision}` ({raw['acquisition']['mechanism']})",
            f"- License state: **{lic['state']}**; identifiers: {', '.join(lic['identifiers'])}",
            f"- Training use: {lic['training_use']}",
            f"- Redistribution: {lic['redistribution']}",
            f"- Attribution: {lic['attribution']}",
        ]
        if lic.get("per_document_license_field"):
            lines.append(
                f"- Per-document license field `{lic['per_document_license_field']}` must match: "
                + ", ".join(f"`{p}`" for p in lic["allowed_document_licenses"])
            )
        lines += [f"- Evidence: {', '.join(lic['evidence'])}"]
        lines += [f"- Risk: {risk}" for risk in raw.get("risks", [])]
        lines += [f"- Known limitation: {item}" for item in raw.get("known_limitations", [])]
        lines.append("")
    lines += ["## Considered but not admitted", "", "| Source | State | Reason |", "| --- | --- | --- |"]
    for entry in considered:
        lines.append(f"| `{entry['source_id']}` | {entry['state']} | {entry['reason']} |")
    return "\n".join(lines) + "\n"


def _table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    def cell(value: Any) -> str:
        if isinstance(value, float):
            return f"{value:,.4f}" if value < 1 else f"{value:,.1f}"
        if isinstance(value, int):
            return f"{value:,}"
        return str(value)

    return ["| " + " | ".join(headers) + " |", "|" + " --- |" * len(headers)] + [
        "| " + " | ".join(cell(v) for v in row) + " |" for row in rows
    ]


def summary_markdown(
    build_id: str,
    process: dict[str, Any],
    dedup: dict[str, Any],
    manifest: dict[str, Any],
    stats: dict[str, Any],
    estimates: dict[str, Any] | None,
) -> str:
    """Generated corpus summary: every number comes from build outputs."""
    names = stats.get("tokenizers", [])
    lines = [
        f"# Corpus summary: {build_id}",
        "",
        "Generated by `j1data report`. Token counts use **provisional measurement tokenizers**",
        "(byte-fallback BPE trained only for corpus sizing; not the Juniper 1 tokenizer).",
        "",
        "## Totals by split",
        "",
    ]
    rows = []
    for split in ("train", "validation", "test"):
        row: list[Any] = [
            split,
            stats["documents"]["split"].get(split, 0),
            stats["bytes"]["split"].get(split, 0),
        ]
        row += [stats["tokens"][n]["split"].get(split, 0) for n in names]
        rows.append(row)
    lines += _table(["split", "documents", "bytes", *[f"tokens ({n})" for n in names]], rows)
    lines += ["", "## Per source: raw input, stage outcomes and accepted volume", ""]
    rows = []
    removed = dedup["removed_by_source"]
    for source, entry in process.items():
        stages = entry["stages"]
        dedup_counts = removed.get(source, {})
        row = [
            source,
            entry["docs_in"],
            entry["bytes_in"],
            stages.get("selection", 0) + stages.get("license", 0) + stages.get("extraction", 0),
            stages.get("privacy", 0),
            stages.get("contamination", 0),
            stages.get("quality", 0),
            dedup_counts.get("exact", 0) + dedup_counts.get("identity", 0),
            dedup_counts.get("normalized", 0),
            dedup_counts.get("near", 0),
            stats["documents"]["source"].get(source, 0),
            stats["bytes"]["source"].get(source, 0),
        ]
        row += [stats["tokens"][n]["source"].get(source, 0) for n in names]
        rows.append(row)
    lines += _table(
        [
            "source",
            "raw docs",
            "raw bytes",
            "selection/license/extraction",
            "privacy",
            "contamination",
            "quality",
            "exact dup",
            "normalized dup",
            "near dup",
            "accepted docs",
            "accepted bytes",
            *[f"tokens ({n})" for n in names],
        ],
        rows,
    )
    if names:
        name = names[-1]
        total = sum(stats["tokens"][name]["source"].values()) or 1
        lines += ["", f"## Mixture (share of accepted tokens, {name})", ""]
        for group in ("source", "domain", "kind"):
            ranked = sorted(stats["tokens"][name][group].items(), key=lambda kv: -kv[1])
            lines += [*_table([group, "tokens", "share"], [[k, v, v / total] for k, v in ranked]), ""]
        formats = sorted(stats["tokens"][name].get("format", {}).items(), key=lambda kv: -kv[1])
        lines += [
            "Documents containing a structured format (tokens of those documents; formats overlap):",
            "",
        ]
        lines += [*_table(["format", "tokens", "share"], [[k, v, v / total] for k, v in formats]), ""]
        concentration = stats.get("source_concentration", {})
        lines += [
            f"Source concentration: Herfindahl {concentration.get('herfindahl_index', 0):.3f}, "
            f"largest source share {concentration.get('largest_source_share', 0):.3f}.",
            "",
        ]
    lines += ["## Rejection reasons (top 12 per source)", ""]
    for source, entry in process.items():
        top = list(entry["reasons"].items())[:12]
        lines.append(f"- **{source}**: " + ", ".join(f"`{k}` {v:,}" for k, v in top))
    lines += ["", "## Privacy redactions (documents admitted with redacted values)", ""]
    for source, entry in process.items():
        if entry["redactions"]:
            lines.append(
                f"- **{source}**: " + ", ".join(f"{k} {v:,}" for k, v in entry["redactions"].items())
            )
    lines += ["", "## Contamination removals by benchmark", ""]
    for source, entry in process.items():
        if entry["contamination_benchmarks"]:
            lines.append(
                f"- **{source}**: "
                + ", ".join(f"{k} {v:,}" for k, v in entry["contamination_benchmarks"].items())
            )
    lines += [
        "",
        "## Deduplication",
        "",
        f"Candidates {dedup['candidates']:,}; retained {dedup['retained']:,}; "
        + ", ".join(f"{k} {v:,}" for k, v in dedup["removed"].items())
        + ".",
        "",
        "Cross-source removals (removed source -> retained source):",
        "",
    ]
    lines += [f"- {k}: {v:,}" for k, v in list(dedup["cross_source_removals"].items())[:20]]
    lines += ["", f"Split-leakage removals: {manifest['split_leakage_removed']:,}.", ""]
    if estimates:
        lines += ["## Raw and rejected volume in tokens (ESTIMATES from accepted bytes per token)", ""]
        rows = [
            [
                s,
                e["bytes_per_token"],
                e["raw_tokens_estimate"],
                sum(e["rejected_tokens_estimate_by_stage"].values()),
            ]
            for s, e in estimates.items()
        ]
        lines += _table(["source", "bytes/token", "raw tokens (est.)", "rejected before dedup (est.)"], rows)
    lines += ["", "## Spelling variety (documents)", ""]
    lines += _table(["variety", "documents"], [[k, v] for k, v in stats["spelling_variety"].items()])
    return "\n".join(lines) + "\n"
