"""Structured sample audits.

Writes a Markdown audit pack for human review: deterministic random samples
of admitted documents per source and kind, rejected documents per reason
(only those whose text was retained for audit), and boundary cases whose
measured English function-word share or code-line share sits close to a
decision threshold. Rejected text is never retained for privacy or
contamination rejections, so those reasons are listed by count only.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from j1data.hashing import unit_interval
from j1data.paths import DataLayout

NAMESPACE = "j1.audit-sample.v1"


def _excerpt(text: str, limit: int) -> str:
    flat = text[:limit].replace("\n", " ⏎ ")
    return flat + (" …" if len(text) > limit else "")


def build_pack(
    layout: DataLayout,
    build_id: str,
    out: Path,
    per_group: int = 4,
    per_reason: int = 3,
    excerpt_chars: int = 600,
    sample_rate: float = 0.002,
    boundaries: dict[str, tuple[float, float]] | None = None,
    admitted_root: Path | None = None,
) -> dict[str, Any]:
    """``admitted_root`` points the admitted samples at accepted shards instead of candidates."""
    work = layout.work(build_id)
    boundaries = boundaries or {}
    admitted: dict[str, list[tuple[float, dict[str, Any]]]] = defaultdict(list)
    boundary: dict[str, list[tuple[float, dict[str, Any]]]] = defaultdict(list)
    rejected: dict[str, list[tuple[float, dict[str, Any]]]] = defaultdict(list)
    columns = [
        "doc_id",
        "source_id",
        "kind",
        "domain",
        "text",
        "english_fw_frac",
        "code_line_frac",
        "n_chars",
    ]
    admitted_files = (
        sorted(admitted_root.glob("*/*/*.parquet"))
        if admitted_root
        else sorted((work / "candidates").glob("*/*.parquet"))
    )
    for path in admitted_files:
        light = pq.read_table(path, columns=["doc_id", "english_fw_frac", "code_line_frac"]).to_pydict()
        picked = [
            i
            for i, did in enumerate(light["doc_id"])
            if unit_interval(NAMESPACE, did) < sample_rate
            or any(abs(light[f][i] - t) <= m for f, (t, m) in boundaries.items())
        ]
        if not picked:
            continue
        table = pq.read_table(path, columns=columns).take(picked).to_pylist()
        for row in table:
            key = unit_interval(NAMESPACE, row["doc_id"])
            admitted[f"{row['source_id']} / {row['kind']}"].append((key, row))
            for feature, (threshold, margin) in boundaries.items():
                if abs(row[feature] - threshold) <= margin:
                    boundary[f"{feature} near {threshold}"].append((key, row))
        for group in (admitted, boundary):
            for name in group:
                group[name] = sorted(group[name], key=lambda kv: kv[0])[:per_group]
    for path in sorted((work / "rejects").glob("*/*.parquet")):
        for row in pq.read_table(path).to_pylist():
            if row["audit_text"] is None:
                continue
            for reason in row["reasons"]:
                rejected[reason].append((unit_interval(NAMESPACE, row["doc_id"] + reason), row))
        for reason in rejected:
            rejected[reason] = sorted(rejected[reason], key=lambda kv: kv[0])[:per_reason]

    lines = [f"# Audit pack: {build_id}", ""]
    for title, group, text_field in (
        ("Admitted samples", admitted, "text"),
        ("Boundary cases", boundary, "text"),
        ("Rejected samples (audit-retained text)", rejected, "audit_text"),
    ):
        lines += [f"## {title}", ""]
        for name in sorted(group):
            lines.append(f"### {name}")
            for _, row in group[name]:
                facts = {
                    k: row[k]
                    for k in ("source_id", "kind", "domain", "reasons", "english_fw_frac", "code_line_frac")
                    if k in row
                }
                lines.append(f"- `{row['doc_id']}` {json.dumps(facts, default=str)}")
                lines.append(f"  > {_excerpt(row[text_field], excerpt_chars)}")
            lines.append("")
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".partial")
    tmp.write_text("\n".join(lines) + "\n")
    tmp.replace(out)  # atomic publish: a crash never leaves a truncated pack in place
    return {
        "admitted_groups": len(admitted),
        "boundary_groups": len(boundary),
        "reject_reasons": len(rejected),
    }


def repeated_lines(
    layout: DataLayout, build_id: str, sample_rate: float, min_chars: int = 30, top: int = 40
) -> dict[str, list[tuple[int, str]]]:
    """Most frequent whole lines per source in a keyed-hash document sample.

    Lines that recur across many unrelated documents are site boilerplate
    (navigation, cookie notices, licence footers); this measurement decides
    whether corpus-level line removal is justified for a source."""
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    sampled: dict[str, int] = defaultdict(int)
    for path in sorted((layout.work(build_id) / "candidates").glob("*/*.parquet")):
        ids = pq.read_table(path, columns=["doc_id"]).column("doc_id").to_pylist()
        picked = [i for i, did in enumerate(ids) if unit_interval(NAMESPACE + ".lines", did) < sample_rate]
        if not picked:
            continue
        table = pq.read_table(path, columns=["source_id", "text"]).take(picked).to_pydict()
        for source, text in zip(table["source_id"], table["text"], strict=True):
            sampled[source] += 1
            for line in {line.strip() for line in text.split("\n")}:
                if len(line) >= min_chars:
                    counts[source][line] += 1
    result = {}
    for source, lines in sorted(counts.items()):
        ranked = sorted(lines.items(), key=lambda kv: (-kv[1], kv[0]))[:top]
        result[source] = [(count, text[:160]) for text, count in ranked]
        result[source].insert(0, (sampled[source], "<documents sampled>"))
    return result
