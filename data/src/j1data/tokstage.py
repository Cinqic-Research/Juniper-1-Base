"""Tokenizer stage: deterministic sample, candidate training, evaluation, freeze record.

Inputs come only from a finalized accepted build:

* Training sample: training-split documents selected by a keyed hash of their id
  (``TRAIN_NAMESPACE``), so selection does not depend on file order.
* Evaluation set: the whole validation split, grouped by source. It never overlaps the
  training sample, because splits are disjoint by construction.

Outputs go under ``$J1_DATA_ROOT/tokenizer/<build>/``. The freeze record binds the
tokenizer bytes, the sample identity, the evaluation results of every candidate, and
the selection rationale.

Recovery: HF ``tokenizers`` training has no resumable internal state, so each candidate
(train + evaluate) is one bounded restartable unit. ``tokenizer.json`` is written
atomically and ``eval.json`` (which records the tokenizer hash and training-sample
identity) is written last; a candidate is reused only when that record verifies.
Upstream inputs (the accepted build) are never modified.
"""

from __future__ import annotations

import json
import os
import platform
import resource
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import tokenizers

from j1data import faults, jtok
from j1data.hashing import sha256_file, sha256_json, unit_interval
from j1data.shards import write_json_atomic

TRAIN_NAMESPACE = "j1.tokenizer-train-sample.v1"


def _accepted(root: Path, build: str) -> tuple[Path, dict[str, Any]]:
    acc = root / "accepted" / build
    manifest = json.loads((acc / "manifest.json").read_text())
    return acc, manifest


def train_sample(
    root: Path, build: str, rate: float, exclude: frozenset[str] = frozenset()
) -> tuple[list[str], dict[str, Any]]:
    acc, manifest = _accepted(root, build)
    texts, ids, excluded = [], [], []
    by_source: dict[str, int] = defaultdict(int)
    for shard in manifest["shards"]:
        if shard["split"] != "train":
            continue
        t = pq.read_table(acc / shard["file"], columns=["doc_id", "text", "source_id"])
        for did, text, src in zip(
            *(t.column(c).to_pylist() for c in ("doc_id", "text", "source_id")), strict=True
        ):
            if unit_interval(TRAIN_NAMESPACE, did) < rate:
                if did in exclude:
                    excluded.append(did)
                    continue
                texts.append(text)
                ids.append(did)
                by_source[src] += len(text.encode("utf-8"))
    info = {
        "namespace": TRAIN_NAMESPACE,
        "rate": rate,
        "build": build,
        "accepted_manifest_sha256": sha256_file(acc / "manifest.json"),
        "documents": len(ids),
        "bytes": sum(by_source.values()),
        "bytes_by_source": dict(sorted(by_source.items())),
        "doc_ids_sha256": sha256_json(sorted(ids)),
    }
    if exclude:
        info["excluded_doc_ids"] = sorted(excluded)
        info["exclusion_list_sha256"] = sha256_json(sorted(exclude))
    return texts, info


def eval_set(root: Path, build: str) -> dict[str, list[str]]:
    acc, manifest = _accepted(root, build)
    out: dict[str, list[str]] = defaultdict(list)
    for shard in manifest["shards"]:
        if shard["split"] != "validation":
            continue
        t = pq.read_table(acc / shard["file"], columns=["text", "source_id"])
        for text, src in zip(t.column("text").to_pylist(), t.column("source_id").to_pylist(), strict=True):
            out[src].append(text)
    return dict(out)


def _write_text_atomic(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".partial")
    with tmp.open("w") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    tmp.replace(path)


def _reusable_candidate(cdir: Path, spec: jtok.CandidateSpec, info: dict[str, Any]) -> dict[str, Any] | None:
    """A candidate is one restartable unit (tokenizer training cannot resume inside the library).

    Its ``eval.json`` is reused only when it names the same spec and training sample and the
    tokenizer file still hashes to the recorded value; otherwise the candidate is retrained."""
    ev_path, tok_path = cdir / "eval.json", cdir / "tokenizer.json"
    if not ev_path.is_file() or not tok_path.is_file():
        return None
    try:
        ev: dict[str, Any] = json.loads(ev_path.read_text())
    except json.JSONDecodeError:
        return None
    if (
        ev.get("spec") != jtok.spec_dict(spec)
        or ev.get("sample_doc_ids_sha256") != info["doc_ids_sha256"]
        or ev.get("tokenizer_json_sha256") != sha256_file(tok_path)
    ):
        return None
    return ev


def run(
    root: Path,
    build: str,
    rate: float,
    sizes: list[int],
    out_dir: Path,
    stem_alphabet: bool = True,
    exclude: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    texts, info = train_sample(root, build, rate, exclude)
    write_json_atomic(out_dir / "sample.json", info)
    evals = eval_set(root, build)
    results = {}
    for size in sizes:
        spec = jtok.CandidateSpec(f"j1tok-{size // 1024}k", size, stem_alphabet=stem_alphabet)
        cdir = out_dir / spec.name
        reused = _reusable_candidate(cdir, spec, info)
        if reused is not None:
            results[spec.name] = reused
            continue
        cdir.mkdir(exist_ok=True)
        (cdir / "eval.json").unlink(missing_ok=True)  # the evaluation record is the unit's marker
        start = time.time()
        tok = jtok.train(spec, texts)
        train_s = time.time() - start
        _write_text_atomic(cdir / "tokenizer.json", tok.to_str())
        faults.point("tokstage.candidate.tokenizer")
        ev = jtok.evaluate(tok, evals)
        ev["spec"] = jtok.spec_dict(spec)
        ev["sample_doc_ids_sha256"] = info["doc_ids_sha256"]
        ev["train_seconds"] = round(train_s, 1)
        ev["tokenizer_json_sha256"] = sha256_file(cdir / "tokenizer.json")
        ev["peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        write_json_atomic(cdir / "eval.json", ev)
        faults.point("tokstage.candidate.commit")
        results[spec.name] = ev
    summary = {
        "version": jtok.VERSION,
        "sample": info,
        "eval_set": {
            k: {"documents": len(v), "bytes": sum(len(t.encode()) for t in v)} for k, v in evals.items()
        },
        "candidates": {
            k: {
                "tokenizer_json_sha256": v["tokenizer_json_sha256"],
                "overall_bytes_per_token": v["overall_bytes_per_token"],
                "byte_fallback_rate": v["byte_fallback_rate"],
                "roundtrip_failures": sum(d["roundtrip_failures"] for d in v["domains"].values())
                + sum(p["roundtrip_failures"] for p in v["probes"].values()),
                "control_ids_from_text": sum(d["control_ids_from_text"] for d in v["domains"].values())
                + sum(p["control_ids_from_text"] for p in v["probes"].values()),
                "train_seconds": v["train_seconds"],
            }
            for k, v in results.items()
        },
        "environment": {"tokenizers": tokenizers.__version__, "python": platform.python_version()},
        "seconds": round(time.time() - t0, 1),
    }
    write_json_atomic(out_dir / "summary.json", summary)
    return summary


def freeze(cand_dir: Path, out_dir: Path, rationale: str, extra: dict[str, Any]) -> dict[str, Any]:
    """Copies the selected candidate into an immutable freeze directory with its record."""
    if out_dir.exists():
        raise SystemExit(f"freeze directory exists (immutable): {out_dir}")
    tok = jtok.load((cand_dir / "tokenizer.json").read_text())
    ev = json.loads((cand_dir / "eval.json").read_text())
    if ev["tokenizer_json_sha256"] != sha256_file(cand_dir / "tokenizer.json"):
        raise SystemExit("candidate tokenizer.json does not match its evaluation record")
    partial = out_dir.with_name(out_dir.name + ".partial")
    partial.mkdir(parents=True)
    (partial / "tokenizer.json").write_bytes((cand_dir / "tokenizer.json").read_bytes())
    record = {
        "version": jtok.VERSION,
        "name": ev["spec"]["name"],
        "tokenizer_json_sha256": ev["tokenizer_json_sha256"],
        "total_vocab": tok.get_vocab_size(),
        "layout": {
            "control": [0, 63],
            "byte_fallback": [64, 319],
            "learned": [320, tok.get_vocab_size() - 1],
        },
        "control_tokens": {t: i for i, t in enumerate(jtok.CONTROL_TOKENS)},
        "named_control_semantics": "see j1data.protocol (juniper-chat-v1); reserved IDs have no meaning",
        "document_separator_id": jtok.control_id("<|endoftext|>"),
        "pad_id": jtok.control_id("<|endoftext|>"),
        "load_requirement": (
            "encode_special_tokens=True (not persisted by tokenizer.json); use j1data.jtok.load"
        ),
        "protocol": "juniper-chat-v1",
        "evaluation": ev,
        "rationale": rationale,
        **extra,
    }
    write_json_atomic(partial / "freeze.json", record)
    partial.rename(out_dir)
    return record
