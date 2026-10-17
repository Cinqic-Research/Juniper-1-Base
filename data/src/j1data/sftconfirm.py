"""Sealed confirmation evaluation bundle (j1-sft-confirm-v1) from a fresh-seed scratch build.

``python -m j1data.sftconfirm build`` reads the evaluation bundle of a scratch ``j1data.sftbuild``
build (``configs/sft_confirm_v1.toml``: the v1.3 design with a new seed) and writes a bundle in the
unchanged ``j1.sft-eval-bundle.v1`` format holding only test items that were never exposed before
(from j1-sft-confirm-v2 on, also not to the v1.2 baseline's train/dev tasks):

* an item is removed when any of its task keys (``task_keys_sha256`` plus ``task_key_sha256``) is a
  task key of the v1.3 train or dev split, of the v1.3 test set, or of the protected v1.2 test set
  (``build.protected_test_records`` of the config);
* an item is removed when the text of any of its prompts (chat or base few-shot) equals the text of a
  v1.3 test prompt (compared by SHA-256 of the decoded prompt text and of the prompt token IDs);
* the scratch build's dev prompts are not part of a confirmation bundle and are dropped.

Kept rows keep their bytes and order (prompt/answer pairing, IDs, stop and forbidden IDs unchanged).
The manifest records the parent build identity, the exclusion-set digests, per-family kept/removed
counts and a freshness note. The tool never prints item text; its output is deterministic.
"""

from __future__ import annotations

import argparse
import json
import sys
import tomllib
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import Any

from j1data.hashing import canonical_json, sha256_bytes, sha256_file, sha256_json

VERSION = "j1.sft-confirm.v1"
BUNDLE_VERSION = "j1-sft-confirm-v1"
SCHEMA = "j1.sft-eval-bundle.v1"
BUNDLE_FILES = ("NOTICE", "answers.jsonl", "prompts.jsonl", "test_records.jsonl")
REPO_ROOT = Path(__file__).resolve().parents[2]

# Fixed-pool sources: a new seed cannot draw new items from them (name -> family prefix, description).
FIXED_POOLS = {
    "wikidata_facts": ("facts.", "test split fixed by subject QID hash"),
    "gsm8k": ("ext.gsm8k", "seeded 500-item subsample of the fixed 1,319 public test problems"),
    "mbpp": ("ext.mbpp", "the full fixed 500-problem public test split"),
    "humaneval": ("ext.humaneval", "the full fixed 164-problem public set"),
    "oasst1": ("chat.oasst", "fixed filtered test conversations; open-ended, unscored"),
}
REASONS = (
    "v13_train_dev_key",
    "v13_test_key",
    "v12_test_key",
    "v12_train_dev_key",
    "v13_test_prompt_text",
    "v12_eval_prompt_text",
)

Decode = Callable[[list[int]], str]


def _rows(path: Path) -> Iterator[dict[str, Any]]:
    """JSONL rows split on b"\\n" only (U+2028/U+0085 may occur inside strings)."""
    with path.open("rb") as fh:
        for line in fh:
            line = line.rstrip(b"\n")
            if line:
                yield json.loads(line)


def _keys(rec: dict[str, Any]) -> set[str]:
    keys = set(rec.get("task_keys_sha256", []))
    if rec.get("task_key_sha256"):
        keys.add(rec["task_key_sha256"])
    return keys


def _set_digest(values: Iterable[str]) -> dict[str, Any]:
    vals = sorted(set(values))
    return {"count": len(vals), "sha256": sha256_bytes("\n".join(vals).encode())}


def _prompt_hashes(row: dict[str, Any], decode: Decode) -> set[str]:
    ids = [int(t) for t in row["prompt_ids"]]
    return {"text:" + sha256_bytes(decode(ids).encode()), "ids:" + sha256_json(ids)}


def _check_manifest(path: Path, self_key: str) -> dict[str, Any]:
    m: dict[str, Any] = json.loads(path.read_text())
    if sha256_json({k: v for k, v in m.items() if k != self_key}) != m[self_key]:
        raise SystemExit(f"{path}: manifest self-hash mismatch")
    return m


def _check_files(base: Path, files: dict[str, dict[str, Any]], names: Iterable[str]) -> None:
    for name in names:
        if sha256_file(base / name) != files[name]["sha256"]:
            raise SystemExit(f"{base / name}: bytes differ from its manifest")


def exclusion_sets(
    v13: Path, protected: Path, protected_sha256: str, decode: Decode, v12_build: Path | None = None
) -> dict[str, set[str]]:
    """The exclusion sets, read from hash-verified files. With ``v12_build`` (j1-sft-confirm-v2 onward),
    task keys of the frozen j1-sft-v1.2 train and dev splits are excluded too: the v1.2-recipe baseline
    trained on them, so items sharing them are not fresh for a v1.2 comparison."""
    m = _check_manifest(v13 / "manifest.json", "manifest_sha256")
    _check_files(v13, m["files"], ("train.jsonl", "dev.jsonl"))
    em = _check_manifest(v13 / "eval/manifest.json", "bundle_sha256")
    if em["bundle_sha256"] != m["eval_bundle_sha256"]:
        raise SystemExit("v1.3 eval bundle is not the one its dataset manifest names")
    _check_files(v13 / "eval", em["files"], ("prompts.jsonl", "test_records.jsonl"))
    if sha256_file(protected) != protected_sha256:
        raise SystemExit("protected v1.2 test records differ from the configured digest")
    train_dev: set[str] = set()
    for split in ("train", "dev"):
        for rec in _rows(v13 / f"{split}.jsonl"):
            train_dev |= _keys(rec)
    test: set[str] = set()
    for rec in _rows(v13 / "eval/test_records.jsonl"):
        test |= _keys(rec)
    v12: set[str] = set()
    for rec in _rows(protected):
        v12 |= _keys(rec)
    prompts: set[str] = set()
    for row in _rows(v13 / "eval/prompts.jsonl"):
        if row["split"] == "test":
            prompts |= _prompt_hashes(row, decode)
    sets = {
        "v13_train_dev_key": train_dev,
        "v13_test_key": test,
        "v12_test_key": v12,
        "v13_test_prompt_text": prompts,
    }
    if v12_build is not None:
        m12 = _check_manifest(v12_build / "manifest.json", "manifest_sha256")
        _check_files(v12_build, m12["files"], ("train.jsonl", "dev.jsonl"))
        v12_td: set[str] = set()
        for split in ("train", "dev"):
            for rec in _rows(v12_build / f"{split}.jsonl"):
                v12_td |= _keys(rec)
        sets["v12_train_dev_key"] = v12_td
        # v1.2 evaluation prompts (dev and test, inspected during v1.2 scoring): an identical prompt is not
        # fresh even when its task key differs (e.g. heldout.if.count_vowels).
        e12 = _check_manifest(v12_build / "eval/manifest.json", "bundle_sha256")
        _check_files(v12_build / "eval", e12["files"], ("prompts.jsonl",))
        v12_prompts: set[str] = set()
        for row in _rows(v12_build / "eval/prompts.jsonl"):
            v12_prompts |= _prompt_hashes(row, decode)
        sets["v12_eval_prompt_text"] = v12_prompts
    return sets


def _source(family: str) -> str | None:
    return next(
        (
            name
            for name, (pre, _) in FIXED_POOLS.items()
            if family == pre.rstrip(".") or family.startswith(pre)
        ),
        None,
    )


def freshness(families: dict[str, dict[str, Any]]) -> dict[str, Any]:
    pools: dict[str, dict[str, int]] = {}
    for fam, c in families.items():
        src = _source(fam)
        if src is not None:
            p = pools.setdefault(src, {"kept": 0, "removed": 0})
            p["kept"] += c["kept"]
            p["removed"] += c["removed"]
    emptied = sorted(src for src, c in pools.items() if c["kept"] == 0)
    partial = sorted(src for src, c in pools.items() if c["kept"] > 0)
    synthetic_emptied = sorted(f for f, c in families.items() if _source(f) is None and c["kept"] == 0)
    synthetic_mostly = sorted(
        f for f, c in families.items() if _source(f) is None and 0 < c["kept"] < c["removed"]
    )
    lines = [
        "Synthetic (Cinqic-generated) families are fresh instances of the same held-out design: held-out "
        "phrasings, operations, frames and compose pairs are fixed by hash in j1data.sftops, so the new "
        "seed draws new items of the unchanged design, not a new design.",
    ]
    if synthetic_emptied:
        lines.append(f"Synthetic families emptied by the exclusions: {', '.join(synthetic_emptied)}.")
    if synthetic_mostly:
        lines.append(
            "Synthetic families with fewer items kept than removed (small key spaces, or embedded "
            "fixed-pool tasks such as Wikidata facts, which keep their v1.3 test keys): "
            + ", ".join(f"{f} ({families[f]['kept']} kept)" for f in synthetic_mostly)
            + "."
        )
    if emptied:
        lines.append(
            "Entirely removed (fixed pools: every item is a v1.2 or v1.3 test task, so none is fresh): "
            + "; ".join(f"{s} ({FIXED_POOLS[s][1]})" for s in emptied)
            + "."
        )
    if partial:
        lines.append(
            "Kept in part but not fresh instances (the same fixed public pool; only items absent from the "
            "v1.2 and v1.3 test sets remain, and they may occur in public pretraining data): "
            + "; ".join(f"{s} ({FIXED_POOLS[s][1]}): {pools[s]['kept']} kept" for s in partial)
            + "."
        )
    return {
        "note": " ".join(lines),
        "fixed_pools": {s: pools[s] for s in sorted(pools)},
        "fixed_pools_emptied": emptied,
        "fixed_pools_partial": partial,
        "synthetic_families_emptied": synthetic_emptied,
        "synthetic_families_mostly_removed": synthetic_mostly,
    }


def confirm(
    scratch: Path,
    v13: Path,
    protected: Path,
    protected_sha256: str,
    out: Path,
    decode: Decode,
    tokenizer_sha256: str,
    code_sha256: str | None = None,
    *,
    v12_build: Path | None = None,
    version: str = BUNDLE_VERSION,
) -> dict[str, Any]:
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} is not empty; a bundle is never overwritten")
    sm = _check_manifest(scratch / "manifest.json", "manifest_sha256")
    em = _check_manifest(scratch / "eval/manifest.json", "bundle_sha256")
    if em["bundle_sha256"] != sm["eval_bundle_sha256"] or em["schema"] != SCHEMA:
        raise SystemExit("scratch eval bundle is not the one its dataset manifest names")
    if em["tokenizer_sha256"] != tokenizer_sha256 or sm["tokenizer_sha256"] != tokenizer_sha256:
        raise SystemExit("scratch build tokenizer identity mismatch")
    _check_files(scratch / "eval", em["files"], BUNDLE_FILES)
    excl = exclusion_sets(v13, protected, protected_sha256, decode, v12_build)

    records = list(_rows(scratch / "eval/test_records.jsonl"))
    prompts = list(_rows(scratch / "eval/prompts.jsonl"))
    answers = list(_rows(scratch / "eval/answers.jsonl"))
    if [p["id"] for p in prompts] != [a["id"] for a in answers]:
        raise SystemExit("scratch prompts and answers are not paired row by row")
    by_item: dict[str, set[str]] = defaultdict(set)
    for p in prompts:
        if p["split"] == "test":
            by_item[p["id"].removesuffix("#fewshot")] |= _prompt_hashes(p, decode)
    if set(by_item) != {r["episode_id"] for r in records} or len(records) != len(by_item):
        raise SystemExit("scratch test records and test prompts do not correspond one to one")

    families: dict[str, dict[str, Any]] = {}
    keep: set[str] = set()
    for rec in records:
        keys = _keys(rec)
        hits = [r for r in REASONS if r.endswith("_key") and r in excl and keys & excl[r]]
        hits += [
            r
            for r in REASONS
            if r.endswith("_prompt_text") and r in excl and by_item[rec["episode_id"]] & excl[r]
        ]
        c = families.setdefault(
            rec["family"], {"kept": 0, "removed": 0, "removed_by": dict.fromkeys(REASONS, 0)}
        )
        if hits:
            c["removed"] += 1
            for h in hits:
                c["removed_by"][h] += 1
        else:
            c["kept"] += 1
            keep.add(rec["episode_id"])
    families = {f: families[f] for f in sorted(families)}

    kept_rows = [
        i for i, p in enumerate(prompts) if p["split"] == "test" and p["id"].removesuffix("#fewshot") in keep
    ]
    out_files: dict[str, list[dict[str, Any]]] = {
        "prompts.jsonl": [prompts[i] for i in kept_rows],
        "answers.jsonl": [answers[i] for i in kept_rows],
        "test_records.jsonl": [r for r in records if r["episode_id"] in keep],
    }
    out.mkdir(parents=True, exist_ok=True)
    files: dict[str, dict[str, Any]] = {}
    notice = (scratch / "eval/NOTICE").read_bytes()
    (out / "NOTICE").write_bytes(notice)
    files["NOTICE"] = {"sha256": sha256_file(out / "NOTICE"), "size": len(notice)}
    for name, rows in out_files.items():
        dst = out / name
        dst.write_bytes(b"".join(canonical_json(r) + b"\n" for r in rows))
        files[name] = {"sha256": sha256_file(dst), "size": dst.stat().st_size, "rows": len(rows)}

    variants = Counter(f"{p['split']}/{p['variant']}" for p in out_files["prompts.jsonl"])
    manifest: dict[str, Any] = {
        **{k: em[k] for k in ("schema", "protocol", "tokenizer_sha256", "stop_ids", "forbidden_ids", "note")},
        "version": version,
        "files": files,
        "builder": {"tool": VERSION, "code_sha256": code_sha256},
        "parent": {
            "scratch_build_version": sm["version"],
            "scratch_manifest_sha256": sm["manifest_sha256"],
            "scratch_config": sm["config"],
            "scratch_code_sha256": sm["code_sha256"],
            "scratch_eval_bundle_sha256": em["bundle_sha256"],
            "scratch_eval_files": em["files"],
            "dropped_dev_prompts": sum(1 for p in prompts if p["split"] != "test"),
        },
        "exclusions": {
            "v13_dataset_manifest_sha256": json.loads((v13 / "manifest.json").read_text())["manifest_sha256"],
            "v13_eval_bundle_sha256": json.loads((v13 / "eval/manifest.json").read_text())["bundle_sha256"],
            "v12_protected_test_records_sha256": protected_sha256,
            "v12_dataset_manifest_sha256": (
                json.loads((v12_build / "manifest.json").read_text())["manifest_sha256"]
                if v12_build
                else None
            ),
            "sets": {name: _set_digest(vals) for name, vals in excl.items()},
            "set_digest": "SHA-256 of the sorted unique values joined by newlines; prompt-text values are "
            "'text:'+SHA-256(decoded prompt) and 'ids:'+SHA-256(canonical JSON token IDs)",
        },
        "counts": {
            "test_items_parent": len(records),
            "test_items_kept": len(keep),
            "test_items_removed": len(records) - len(keep),
            "prompts": dict(sorted(variants.items())),
        },
        "families": families,
        "freshness": freshness(families),
    }
    manifest["bundle_sha256"] = sha256_json(manifest)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True))
    return manifest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m j1data.sftconfirm")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument(
        "--scratch", type=Path, required=True, help="scratch sftbuild output (configs/sft_confirm_v1.toml)"
    )
    b.add_argument("--v13", type=Path, required=True, help="frozen j1-sft-v1.3 build directory")
    b.add_argument("--v12", type=Path, help="frozen j1-sft-v1.2 build directory (exclude its train/dev keys)")
    b.add_argument("--config", type=Path, default=REPO_ROOT / "configs/sft_confirm_v1.toml")
    b.add_argument("--root", type=Path, required=True, help="data root of build.protected_test_records")
    b.add_argument("--tokenizer", type=Path, required=True)
    b.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    from j1data import jtok

    cfg = tomllib.loads(a.config.read_text())
    tok_sha = sha256_file(a.tokenizer)
    if tok_sha != cfg["build"]["tokenizer_sha256"]:
        raise SystemExit("tokenizer bytes differ from the configured frozen tokenizer")
    tok = jtok.load(a.tokenizer.read_text())
    prot = cfg["build"]["protected_test_records"]

    def decode(ids: list[int]) -> str:
        return str(tok.decode(ids, skip_special_tokens=False))

    m = confirm(
        a.scratch,
        a.v13,
        a.root / prot["path"],
        prot["sha256"],
        a.out,
        decode,
        tok_sha,
        sha256_file(Path(__file__)),
        v12_build=a.v12,
        version=cfg["build"]["version"],
    )
    print(json.dumps({"bundle_sha256": m["bundle_sha256"], "counts": m["counts"]}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
