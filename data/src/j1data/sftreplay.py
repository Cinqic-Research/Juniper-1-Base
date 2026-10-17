"""Admission of v1.2 replay episodes for a v1.3-based mixture (j1.sft-replay.v1).

``python -m j1data.sftreplay admit --v12 DIR --v13 DIR --confirm DIR --tokenizer FILE --categories hierarchy
injection --out FILE`` reads the frozen j1-sft-v1.2 build and admits a train episode of the requested
categories only if all of these hold. Everything is checked against hash-verified files:

* no task key is a v1.3 test, v1.3 dev, v1.2 test (protected) or confirmation-bundle key;
* its user text does not occur in any v1.3 test/dev or confirmation prompt (decoded prompt text);
* none of the checked v1.3 evaluation-only phrasings occurs in its text: no reserved paraphrase of a
  trained operation, no phrasing of a held-out operation, and neither of the two test-only prompt frames;
* it is not byte-identical content of a v1.3 train episode, and its primary task key is not a v1.3 train
  episode's primary key (the same task, differently worded, is already trained and shares the episode ID).
  Embedded sub-task keys shared with v1.3 *train* are allowed; only evaluation keys are protected above.

v1.2 and v1.3 share the chat protocol, the 1,025-token row limit, the tokenizer and the decontamination index
identity (checked here), so admitted rows are used as they are. The output lists admitted episode IDs and
every rejection with its reason; nothing is copied or re-encoded.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from j1data import sftops
from j1data.hashing import canonical_json, sha256_file, sha256_json

VERSION = "j1.sft-replay.v1"
# The phrasing patterns cover operation paraphrases, held-out operations and test frames, which is what the
# hierarchy and injection generators can carry. heldout.if templates, heldout.json and held-out compose pairs
# are not pattern-checked, so other categories are refused until those checks exist.
SUPPORTED_CATEGORIES = frozenset({"hierarchy", "injection"})
_PLACEHOLDER = re.compile(r"\\\{[a-zA-Z_]+\\\}")


def _rows(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("rb") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def _norm(text: str) -> str:
    return " ".join(text.lower().split())


def _keys(rec: dict[str, Any]) -> set[str]:
    keys = set(rec.get("task_keys_sha256") or [])
    if rec.get("task_key_sha256"):
        keys.add(rec["task_key_sha256"])
    return keys


def _manifest(path: Path, key: str) -> dict[str, Any]:
    m: dict[str, Any] = json.loads(path.read_text())
    body = {k: v for k, v in m.items() if k != key}
    if sha256_json(body) != m[key]:
        raise SystemExit(f"{path}: manifest self-hash mismatch")
    return m


def _check_files(base: Path, files: dict[str, Any], names: tuple[str, ...]) -> None:
    for n in names:
        if sha256_file(base / n) != files[n]["sha256"]:
            raise SystemExit(f"{base / n}: bytes differ from its manifest")


def heldout_patterns() -> list[tuple[str, re.Pattern[str]]]:
    """Regexes for every v1.3 evaluation-only phrasing and the two test-only frames."""
    out: list[tuple[str, re.Pattern[str]]] = []

    def add(label: str, template: str) -> None:
        text = _norm(template)
        body = _PLACEHOLDER.sub(r".{0,120}?", re.escape(text))
        if len(re.sub(r"\.\{0,120\}\?", "", body)) >= 8:  # ignore near-empty templates
            out.append((label, re.compile(body)))

    for op in sftops.TRAIN_OPS:
        for i, phr in enumerate(op.phrasings):
            if sftops.reserved_phrasing(op, i):
                add(f"reserved phrasing {op.name}#{i}", phr)
    for op in sftops.HELDOUT_OPS:
        for i, phr in enumerate(op.phrasings):
            add(f"held-out op {op.name}#{i}", phr)
    out.append(("test frame 8", re.compile(r"here are some words:")))
    out.append(("test frame 9", re.compile(r'words: ".*" instruction:')))
    return out


def _texts(rec: dict[str, Any]) -> tuple[str, str]:
    """(user text, all prompt-side text) of an episode record."""
    user, every = [], []
    for m in rec["episode"]["messages"]:
        t = "\n".join(c.get("text", "") for c in m["content"] if c.get("type") == "text")
        if m["role"] == "user":
            user.append(t)
        if m["role"] != "assistant":
            every.append(t)
    return _norm("\n".join(user)), _norm("\n".join(every))


def admit(v12: Path, v13: Path, confirm: Path, tokenizer: Path, categories: list[str]) -> dict[str, Any]:
    from j1data import jtok

    unsupported = set(categories) - SUPPORTED_CATEGORIES
    if unsupported:
        raise SystemExit(
            f"replay admission is only qualified for {sorted(SUPPORTED_CATEGORIES)}: {sorted(unsupported)}"
        )
    m12 = _manifest(v12 / "manifest.json", "manifest_sha256")
    m13 = _manifest(v13 / "manifest.json", "manifest_sha256")
    for k in ("protocol", "max_tokens", "tokenizer_sha256", "decontamination_index_identity"):
        if m12.get(k) != m13.get(k):
            raise SystemExit(f"v1.2 and v1.3 differ in {k}; rows are not interchangeable")
    _check_files(v12, m12["files"], ("train.jsonl", "train_rows.npz"))
    _check_files(v13, m13["files"], ("train.jsonl", "dev.jsonl"))
    e12 = _manifest(v12 / "eval/manifest.json", "bundle_sha256")
    e13 = _manifest(v13 / "eval/manifest.json", "bundle_sha256")
    ec = _manifest(confirm / "manifest.json", "bundle_sha256")
    _check_files(v12 / "eval", e12["files"], ("test_records.jsonl",))
    _check_files(v13 / "eval", e13["files"], ("test_records.jsonl", "prompts.jsonl"))
    _check_files(confirm, ec["files"], ("test_records.jsonl", "prompts.jsonl"))
    if sha256_file(tokenizer) != m13["tokenizer_sha256"]:
        raise SystemExit("tokenizer differs from the datasets' tokenizer")
    tok = jtok.load(tokenizer.read_text())

    protected: set[str] = set()
    for path in (
        v13 / "eval/test_records.jsonl",
        v12 / "eval/test_records.jsonl",
        confirm / "test_records.jsonl",
    ):
        for r in _rows(path):
            protected |= _keys(r)
    for r in _rows(v13 / "dev.jsonl"):
        protected |= _keys(r)
    prompts: list[str] = []
    for path in (v13 / "eval/prompts.jsonl", confirm / "prompts.jsonl"):
        for row in _rows(path):
            prompts.append(
                _norm(str(tok.decode([int(t) for t in row["prompt_ids"]], skip_special_tokens=True)))
            )
    for r in _rows(v13 / "dev.jsonl"):
        prompts.append(_texts(r)[0])
    big = "\n".join(prompts)
    v13_content: set[str] = set()
    v13_train_tasks: set[str] = set()
    for r in _rows(v13 / "train.jsonl"):
        v13_content.add(r["content_sha256"])
        v13_train_tasks.add(r["task_key_sha256"])
    patterns = heldout_patterns()

    admitted: list[str] = []
    rejected: list[dict[str, Any]] = []
    seen_cat: Counter[str] = Counter()
    for r in _rows(v12 / "train.jsonl"):
        if r["category"] not in categories:
            continue
        seen_cat[r["category"]] += 1
        user, every = _texts(r)
        reasons = []
        if _keys(r) & protected:
            reasons.append("task key in a v1.3 test/dev, v1.2 test or confirmation set")
        if user and user in big:
            reasons.append("user text occurs in an evaluation or dev prompt")
        hits = [label for label, pat in patterns if pat.search(every)]
        if hits:
            reasons.append("evaluation-only phrasing: " + ", ".join(sorted(hits)[:3]))
        if r["content_sha256"] in v13_content:
            reasons.append("identical to a v1.3 train episode")
        elif r["task_key_sha256"] in v13_train_tasks:
            reasons.append("same task as a v1.3 train episode (different wording)")
        if reasons:
            rejected.append({"episode_id": r["episode_id"], "category": r["category"], "reasons": reasons})
        else:
            admitted.append(r["episode_id"])
    return {
        "schema": VERSION,
        "v12_manifest_sha256": m12["manifest_sha256"],
        "v13_manifest_sha256": m13["manifest_sha256"],
        "confirmation_bundle_sha256": ec["bundle_sha256"],
        "categories": sorted(categories),
        "candidates": dict(sorted(seen_cat.items())),
        "admitted_by_category": dict(
            sorted(
                Counter(
                    r["category"] for r in _rows(v12 / "train.jsonl") if r["episode_id"] in set(admitted)
                ).items()
            )
        ),
        "rejected_by_reason": dict(sorted(Counter(x for r in rejected for x in r["reasons"]).items())),
        "heldout_patterns": len(patterns),
        "code_sha256": {
            "sftreplay.py": sha256_file(Path(__file__)),
            "sftops.py": sha256_file(Path(sftops.__file__)),
        },
        "admitted_episode_ids": sorted(admitted),
        "rejected": rejected,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m j1data.sftreplay", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("admit")
    a.add_argument("--v12", type=Path, required=True)
    a.add_argument("--v13", type=Path, required=True)
    a.add_argument("--confirm", type=Path, required=True)
    a.add_argument("--tokenizer", type=Path, required=True)
    a.add_argument("--categories", nargs="+", required=True)
    a.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    res = admit(args.v12, args.v13, args.confirm, args.tokenizer, args.categories)
    res["admission_sha256"] = sha256_json(res)
    args.out.write_bytes(canonical_json(res) + b"\n")
    print(
        json.dumps(
            {k: res[k] for k in ("candidates", "admitted_by_category", "rejected_by_reason")}, indent=1
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
