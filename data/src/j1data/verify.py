"""Independent verification of a finished build.

Each check recomputes a critical quantity by a different route than the
pipeline used to produce it, and reports disagreement instead of trusting the
markers:

* shard integrity  re-hash every accepted and token shard; row counts from
                   Parquet metadata rather than markers
* uniqueness       no doc id, text hash or signature hash occurs twice among
                   accepted documents (an end-to-end check of dedup)
* splits           every split recomputed from its group key; no group spans
                   two splits
* accounting       raw documents = rejections + candidates; candidates =
                   duplicates + leakage removals + accepted
* token counts     provisional count tables aligned with accepted shards; a
                   sample re-encoded one document at a time, unbatched
* contamination    a sample re-scanned with plain Python sets of word 13-grams
                   built directly from the evaluation items
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tokenizers import Tokenizer

from j1data import splits
from j1data.hashing import sha256_file, unit_interval
from j1data.normalize import signature_words
from j1data.paths import DataLayout, safe_child

SAMPLE_NAMESPACE = "j1.verify-sample.v1"


def _hex_prefix(values: list[str]) -> np.ndarray:
    raw = b"".join(bytes.fromhex(v[:32]) for v in values)
    return np.frombuffer(raw, dtype=[("hi", ">u8"), ("lo", ">u8")])


def check_shards(root: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    problems = []
    rows: Counter[str] = Counter()
    for shard in manifest["shards"]:
        try:
            path = safe_child(root, shard["file"])
        except (KeyError, TypeError, ValueError):
            problems.append(f"unsafe shard path {shard.get('file')!r}")
            continue
        if not path.is_file() or path.is_symlink():
            problems.append(f"missing or non-regular shard {shard['file']}")
            continue
        if sha256_file(path) != shard["sha256"]:
            problems.append(f"sha256 mismatch {shard['file']}")
            continue
        try:
            actual = pq.ParquetFile(path).metadata.num_rows
        except (OSError, pa.ArrowException) as error:
            problems.append(f"invalid Parquet shard {shard['file']}: {error}")
            continue
        rows[shard["split"]] += actual
        if actual != shard["rows"]:
            problems.append(f"row count {shard['file']}: {actual} != {shard['rows']}")
    return {"shards": len(manifest["shards"]), "rows_by_split_from_parquet": dict(rows), "problems": problems}


def check_uniqueness_and_splits(
    root: Path, manifest: dict[str, Any], config: splits.SplitConfig
) -> dict[str, Any]:
    ids, texts, sigs, groups, split_codes = [], [], [], [], []
    mismatched = 0
    problems: list[str] = []
    code = {name: i for i, name in enumerate(splits.SPLITS)}
    for shard in manifest["shards"]:
        try:
            path = safe_child(root, shard["file"])
            table = pq.read_table(path, columns=["doc_id", "sha256", "norm_sha256", "group_key", "split"])
        except (KeyError, TypeError, ValueError, OSError, pa.ArrowException) as error:
            problems.append(f"cannot read shard {shard.get('file')!r}: {error}")
            continue
        columns = table.to_pydict()
        ids.append(_hex_prefix(columns["doc_id"]))
        texts.append(_hex_prefix(columns["sha256"]))
        sigs.append(_hex_prefix(columns["norm_sha256"]))
        for group, split in zip(columns["group_key"], columns["split"], strict=True):
            if splits.assign(group, config) != split:
                mismatched += 1
        groups.append(_hex_prefix([hashlib.sha256(g.encode()).hexdigest() for g in columns["group_key"]]))
        split_codes.append(np.array([code[s] for s in columns["split"]], dtype=np.int8))

    def duplicates(parts: list[np.ndarray]) -> int:
        if not parts:
            return 0
        values = np.concatenate(parts)
        return int(values.size - np.unique(values).size)

    group_values = np.concatenate(groups) if groups else np.empty(0, dtype=[("hi", ">u8"), ("lo", ">u8")])
    codes = np.concatenate(split_codes) if split_codes else np.empty(0, dtype=np.int8)
    order = np.argsort(group_values, kind="stable")
    sorted_groups, sorted_codes = group_values[order], codes[order]
    same_group = sorted_groups[1:] == sorted_groups[:-1]
    groups_spanning_splits = int(np.count_nonzero(same_group & (sorted_codes[1:] != sorted_codes[:-1])))
    return {
        "duplicate_doc_ids": duplicates(ids),
        "duplicate_text_hashes": duplicates(texts),
        "duplicate_signature_hashes": duplicates(sigs),
        "split_mismatches": mismatched,
        "groups_spanning_splits": groups_spanning_splits,
        "problems": problems,
    }


def check_accounting(layout: DataLayout, build_id: str, manifest: dict[str, Any]) -> dict[str, Any]:
    problems: list[str] = []
    markers: list[dict[str, Any]] = []
    for path in sorted((layout.work(build_id) / "units").glob("*/*.json")):
        try:
            marker = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            problems.append(f"cannot read unit marker {path.name}: {error}")
            continue
        if not isinstance(marker, dict):
            problems.append(f"malformed unit marker {path.name}")
        else:
            markers.append(marker)
    try:
        if not markers:
            raise ValueError("unit marker inventory is empty")
        docs_in = sum(int(marker["docs_in"]) for marker in markers)
        stage_total = sum(sum(int(value) for value in marker["stages"].values()) for marker in markers)
        candidates = sum(int(marker["stages"]["candidate"]) for marker in markers)
        candidate_rows = 0
        for marker in markers:
            recorded = marker.get("candidate_rows")
            if isinstance(recorded, int):
                candidate_rows += recorded
            elif isinstance(marker.get("segments"), list):
                candidate_rows += sum(int(segment["outputs"][0]["rows"]) for segment in marker["segments"])
            else:
                candidate_rows += int(marker["outputs"][0]["rows"])
        dedup = json.loads((layout.work(build_id) / "dedup" / "summary.json").read_text())
        accepted = sum(int(value) for value in manifest["accepted_by_split"].values())
        removed = sum(int(value) for value in dedup["removed"].values())
        not_selected = int(manifest.get("not_selected", 0))
        leakage_removed = int(manifest["split_leakage_removed"])
        finalize_removed = int(manifest["deduplicated_away"])
        dedup_candidates = int(dedup["candidates"])
    except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError) as error:
        problems.append(f"incomplete accounting inputs: {error}")
        docs_in = 0
        stage_total = candidates = candidate_rows = accepted = removed = 0
        not_selected = leakage_removed = finalize_removed = dedup_candidates = -1
    return {
        "raw_documents": docs_in,
        "raw_equals_stage_total": bool(not problems and docs_in > 0 and docs_in == stage_total),
        "candidates_equal_candidate_rows": bool(not problems and candidates == candidate_rows),
        "dedup_saw_all_candidates": bool(not problems and dedup_candidates == candidates),
        "candidates_equal_removed_plus_unselected_plus_leaked_plus_accepted": bool(
            not problems and candidates == removed + not_selected + leakage_removed + accepted
        ),
        "finalize_dedup_agree": bool(not problems and finalize_removed == removed),
        "problems": problems,
    }


def check_token_counts(
    accepted_root: Path,
    counts_root: Path,
    manifest: dict[str, Any],
    tokenizers: dict[str, Tokenizer],
    sample_rate: float,
) -> dict[str, Any]:
    """Provisional token counts: table rows align with accepted shards, and a sample
    re-encoded one document at a time reproduces the batched counts."""
    problems: list[str] = []
    totals: Counter[str] = Counter()
    sampled = 0
    for shard in manifest["shards"]:
        stem = shard["file"].removesuffix(".parquet")
        try:
            counts_path = safe_child(counts_root / "counts", f"{stem}.parquet")
            accepted_path = safe_child(accepted_root, shard["file"])
            counts = pq.read_table(counts_path).to_pydict()
            accepted_ids = pq.read_table(accepted_path, columns=["doc_id"]).column("doc_id").to_pylist()
        except (KeyError, TypeError, ValueError, OSError, pa.ArrowException) as error:
            problems.append(f"cannot read count or accepted shard {shard.get('file')!r}: {error}")
            continue
        if counts["doc_id"] != accepted_ids:
            problems.append(f"{stem}: count rows do not align with the accepted shard")
            continue
        for name in tokenizers:
            totals[name] += sum(counts[f"tokens_{name}"])
        chosen = [i for i, d in enumerate(accepted_ids) if unit_interval(SAMPLE_NAMESPACE, d) < sample_rate]
        if not chosen:
            continue
        texts = pq.read_table(accepted_path, columns=["text"]).take(chosen).column("text").to_pylist()
        for text, position in zip(texts, chosen, strict=True):
            for name, tokenizer in tokenizers.items():
                if (
                    len(tokenizer.encode(text, add_special_tokens=False).ids)
                    != counts[f"tokens_{name}"][position]
                ):
                    problems.append(
                        f"{stem}: re-encoded length differs for {accepted_ids[position]} with {name}"
                    )
            sampled += 1
    return {"token_totals": dict(totals), "reencoded_sample_documents": sampled, "problems": problems}


def check_contamination_sample(
    accepted_root: Path,
    manifest: dict[str, Any],
    items: dict[str, list[str]],
    ngram: int,
    min_coverage: float,
    min_informative_words: int,
    sample_rate: float,
    short_min_words: int = 6,
    short_min_chars: int = 48,
    short_min_informative_words: int = 5,
) -> dict[str, Any]:
    # Python's tuple hash is unrelated to the pipeline's xxh64/SplitMix n-gram hashing.
    gram_items: dict[int, list[int]] = {}
    sizes: list[int] = []
    short_sequences: dict[int, set[tuple[str, ...]]] = {}
    for texts in items.values():
        for text in texts:
            words = signature_words(text)
            grams = {
                hash(tuple(words[i : i + ngram]))
                for i in range(len(words) - ngram + 1)
                if sum(1 for w in words[i : i + ngram] if len(w) >= 2 and not w.isdigit())
                >= min_informative_words
            }
            if not grams:
                if (
                    short_min_words <= len(words) < ngram
                    and len(" ".join(words)) >= short_min_chars
                    and sum(len(word) >= 2 and not word.isdigit() for word in words)
                    >= short_min_informative_words
                ):
                    short_sequences.setdefault(len(words), set()).add(tuple(words))
                continue
            for gram in grams:
                gram_items.setdefault(gram, []).append(len(sizes))
            sizes.append(len(grams))
    scanned = flagged = short_flagged = 0
    for shard in manifest["shards"]:
        try:
            path = safe_child(accepted_root, shard["file"])
            table = pq.read_table(path, columns=["doc_id", "text"]).to_pydict()
        except (KeyError, TypeError, ValueError, OSError, pa.ArrowException) as error:
            raise ValueError(f"cannot read accepted shard {shard.get('file')!r}: {error}") from error
        for doc, text in zip(table["doc_id"], table["text"], strict=True):
            if unit_interval(SAMPLE_NAMESPACE + ".contamination", doc) >= sample_rate:
                continue
            scanned += 1
            words = signature_words(text)
            hits: Counter[int] = Counter()
            for gram in {hash(tuple(words[i : i + ngram])) for i in range(len(words) - ngram + 1)}:
                for item in gram_items.get(gram, ()):
                    hits[item] += 1
            long_hit = any(count / sizes[item] >= min_coverage for item, count in hits.items())
            short_hit = False
            for length, protected in short_sequences.items():
                for start in range(max(0, len(words) - length + 1)):
                    if tuple(words[start : start + length]) in protected:
                        short_hit = True
                        break
                if short_hit:
                    break
            if long_hit or short_hit:
                flagged += 1
                short_flagged += int(short_hit)
    return {
        "scanned_documents": scanned,
        "contaminated_documents_found": flagged,
        "short_exact_documents_found": short_flagged,
        "short_exact_eligible_items": sum(len(values) for values in short_sequences.values()),
        "limitations": "sampled independently; zero hits do not certify a clean corpus",
    }
