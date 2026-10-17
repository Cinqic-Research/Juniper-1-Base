"""Near-duplicate threshold calibration on the real candidate pool.

1. A permissive LSH pass (many short bands) over every stored signature
   collects candidate pairs well below any threshold under consideration.
2. A deterministic sample of those pairs is drawn across similarity strata
   (estimated from signatures), their texts are read back, and the true word
   5-gram Jaccard similarity is computed from the text.
3. For each LSH setting under consideration, whether the pair would have been
   linked is computed from the stored signatures, giving empirical detection
   curves against true similarity.
4. A smaller stratified subset is written with text excerpts for manual
   duplicate / not-duplicate labelling (docs/dedup.md records the labels).
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pyarrow.parquet as pq

from j1data.dedup import UnitRef
from j1data.hashing import unit_interval
from j1data.minhash import LSHConfig, band_keys, jaccard
from j1data.ngrams import ngram_hashes, word_hashes
from j1data.normalize import signature_words

NAMESPACE = "j1.dedup-calibration.v1"
MAX_SCORED_PAIRS = 200_000


def candidate_pairs(refs: list[UnitRef], probe: LSHConfig, limit_per_band: int) -> npt.NDArray[np.int64]:
    pairs: list[npt.NDArray[np.int64]] = []
    for band in range(probe.bands):
        keys = np.concatenate(
            [band_keys(np.load(r.signatures, mmap_mode="r"), band, probe.rows) for r in refs if r.rows]
        )
        order = np.argsort(keys, kind="stable")
        sorted_keys = keys[order]
        same = np.flatnonzero(sorted_keys[1:] == sorted_keys[:-1])
        found = np.stack([order[same], order[same + 1]], axis=1)
        if found.shape[0] > limit_per_band:
            pick = np.linspace(0, found.shape[0] - 1, limit_per_band).astype(np.int64)
            found = found[pick]
        pairs.append(found)
    joined = np.concatenate(pairs) if pairs else np.empty((0, 2), dtype=np.int64)
    joined.sort(axis=1)
    return np.unique(joined, axis=0)


def _signature_matrix(refs: list[UnitRef], docs: npt.NDArray[np.int64]) -> dict[int, npt.NDArray[np.uint32]]:
    """Signatures of ``docs``, gathered unit by unit with one memory map per unit."""
    offsets = np.array([r.offset for r in refs])
    wanted = np.unique(docs)
    owners = np.searchsorted(offsets, wanted, side="right") - 1
    out: dict[int, npt.NDArray[np.uint32]] = {}
    for position in np.unique(owners):
        ref = refs[int(position)]
        members = wanted[owners == position]
        block = np.asarray(np.load(ref.signatures, mmap_mode="r")[:, members - ref.offset])
        for column, doc in enumerate(members):
            out[int(doc)] = block[:, column]
    return out


def _texts(refs: list[UnitRef], docs: list[int]) -> dict[int, tuple[str, str, str]]:
    offsets = np.array([r.offset for r in refs])
    wanted: dict[int, list[int]] = defaultdict(list)
    for doc in docs:
        wanted[int(np.searchsorted(offsets, doc, side="right") - 1)].append(doc)
    out = {}
    for position, members in wanted.items():
        ref = refs[position]
        parquet = pq.ParquetFile(ref.candidates)
        starts = np.cumsum(
            [0] + [parquet.metadata.row_group(i).num_rows for i in range(parquet.num_row_groups)]
        )
        for doc in members:
            row = doc - ref.offset
            group = int(np.searchsorted(starts, row, side="right") - 1)
            table = parquet.read_row_group(group, columns=["doc_id", "source_id", "text"])
            record = table.slice(row - int(starts[group]), 1).to_pylist()[0]
            out[doc] = (record["doc_id"], record["source_id"], record["text"])
    return out


def linked(sig_a: npt.NDArray[np.uint32], sig_b: npt.NDArray[np.uint32], lsh: LSHConfig) -> bool:
    rows = lsh.rows
    return any(
        np.array_equal(sig_a[b * rows : (b + 1) * rows], sig_b[b * rows : (b + 1) * rows])
        for b in range(lsh.bands)
    )


def calibrate(
    refs: list[UnitRef],
    settings: list[LSHConfig],
    probe: LSHConfig,
    sample_pairs: int,
    label_pairs: int,
    out: Path,
) -> dict[str, Any]:
    pairs = candidate_pairs(refs, probe, limit_per_band=200_000)
    if pairs.shape[0] > MAX_SCORED_PAIRS:
        ranked = sorted(
            range(pairs.shape[0]),
            key=lambda i: unit_interval(NAMESPACE + ".pairs", f"{pairs[i][0]}-{pairs[i][1]}"),
        )
        pairs = pairs[np.sort(np.asarray(ranked[:MAX_SCORED_PAIRS]))]
    signatures = _signature_matrix(refs, pairs.reshape(-1)) if pairs.size else {}
    estimates = (
        np.array([float(np.mean(signatures[a] == signatures[b])) for a, b in pairs])
        if pairs.size
        else np.empty(0)
    )
    strata = np.minimum((estimates * 10).astype(int), 9)
    chosen: list[int] = []
    per_stratum = max(1, sample_pairs // 10)
    for stratum in range(10):
        members = np.flatnonzero(strata == stratum)
        ranked = sorted(
            members.tolist(), key=lambda i: unit_interval(NAMESPACE, f"{pairs[i][0]}-{pairs[i][1]}")
        )
        chosen.extend(ranked[:per_stratum])
    texts = _texts(refs, sorted({int(d) for i in chosen for d in pairs[i]}))
    records: list[dict[str, Any]] = []
    for i in chosen:
        a, b = (int(x) for x in pairs[i])
        shingles = [ngram_hashes(word_hashes(signature_words(texts[d][2])), 5) for d in (a, b)]
        records.append(
            {
                "pair": [texts[a][0], texts[b][0]],
                "sources": [texts[a][1], texts[b][1]],
                "signature_estimate": float(estimates[i]),
                "true_jaccard": jaccard(shingles[0], shingles[1]),
                "linked": {f"{s.bands}x{s.rows}": linked(signatures[a], signatures[b], s) for s in settings},
            }
        )
    curves: dict[str, dict[str, Any]] = {}
    edges = np.linspace(0, 1, 11)
    truth = np.array([r["true_jaccard"] for r in records])
    for setting in settings:
        name = f"{setting.bands}x{setting.rows}"
        hit = np.array([r["linked"][name] for r in records])
        bucket = np.minimum(np.digitize(truth, edges) - 1, 9)
        curves[name] = {
            "threshold_at_half_detection": setting.threshold,
            "detection_by_true_jaccard": {
                f"{edges[k]:.1f}-{edges[k + 1]:.1f}": [int(hit[bucket == k].sum()), int((bucket == k).sum())]
                for k in range(10)
            },
        }
    labels = sorted(records, key=lambda r: unit_interval(NAMESPACE + ".label", "|".join(r["pair"])))
    label_set: list[dict[str, Any]] = []
    for low in np.arange(0.3, 1.0, 0.1):
        label_set.extend(
            [r for r in labels if low <= r["true_jaccard"] < low + 0.1][: max(1, label_pairs // 7)]
        )
    by_id = {did: text for did, _, text in texts.values()}
    out.mkdir(parents=True, exist_ok=True)
    lines = ["# Near-duplicate pairs for manual labelling", ""]
    for number, record in enumerate(label_set, 1):
        lines.append(
            f"## Pair {number}: true Jaccard {record['true_jaccard']:.3f} ({' / '.join(record['sources'])})"
        )
        lines.append(f"linked: {record['linked']}")
        for did in record["pair"]:
            excerpt = by_id[did][:700].replace("\n", " ")
            lines.append(f"- `{did}`: {excerpt}")
        lines.append("")
    (out / "label_pairs.md").write_text("\n".join(lines))
    (out / "pairs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    summary = {
        "probe": f"{probe.bands}x{probe.rows}",
        "candidate_pairs": int(pairs.shape[0]),
        "sampled_pairs": len(records),
        "curves": curves,
        "label_candidates": [r["pair"] for r in label_set],
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    return summary
