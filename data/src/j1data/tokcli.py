"""Tokenizer-stage commands (``python -m j1data.tokcli``).

These live outside ``cli.py`` because ``cli.py`` is part of the corpus build identity
(``build.BUILD_MODULES``). Tokenizer work is downstream of a finished build and must not
change the identity of the corpus it reads.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from j1data.cli import _build_id
from j1data.paths import DataLayout


def _tokenizer_config() -> dict[str, Any]:
    import tomllib

    with (Path(__file__).resolve().parents[2] / "configs" / "juniper_tokenizer.toml").open("rb") as handle:
        return tomllib.load(handle)


def _cmd_tokenizer_candidates(args: argparse.Namespace) -> int:
    from j1data import tokstage

    layout = DataLayout.from_env()
    build_id = _build_id(args)
    config = _tokenizer_config()
    out = layout.root / "tokenizer" / build_id / args.out_name
    if out.exists() and args.out_name != "candidates" and (out / "summary.json").exists():
        raise SystemExit(f"{out} already exists (candidate sets are immutable)")
    exclude = (
        frozenset(json.loads(Path(args.exclude_doc_ids).read_text())) if args.exclude_doc_ids else frozenset()
    )
    summary = tokstage.run(
        layout.root,
        build_id,
        config["sample"]["rate"],
        config["candidates"]["total_vocab"],
        out,
        exclude=exclude,
    )
    print(json.dumps(summary["candidates"], indent=1))
    return 0


def _cmd_tokenizer_freeze(args: argparse.Namespace) -> int:
    from j1data import tokstage
    from j1data.build import code_revision

    layout = DataLayout.from_env()
    build_id = _build_id(args)
    base = layout.root / "tokenizer" / build_id
    record = tokstage.freeze(
        base / args.candidates_dir / args.candidate,
        base / "frozen" / args.name,
        args.rationale,
        {
            "frozen_name": args.name,
            "code_revision": code_revision(),
            "candidates_summary": json.loads((base / args.candidates_dir / "summary.json").read_text()),
        },
    )
    print(json.dumps({k: record[k] for k in ("name", "tokenizer_json_sha256", "total_vocab")}))
    return 0


def _cmd_token_shards(args: argparse.Namespace) -> int:
    from j1data import tokshards

    layout = DataLayout.from_env()
    build_id = _build_id(args)
    tok_dir = layout.root / "tokenizer" / build_id / "frozen" / args.tokenizer
    out = layout.root / "token_shards" / build_id / args.tokenizer
    if args.verify_only:
        problems = tokshards.verify(out)
        print(json.dumps({"ok": not problems, "problems": problems}))
        return 0 if not problems else 2
    record = tokshards.build(layout.root, build_id, tok_dir, out)
    print(
        json.dumps(
            {
                s: {k: v[k] for k in ("documents", "document_tokens", "stream_tokens")}
                for s, v in record["splits"].items()
            }
        )
    )
    return 0 if not tokshards.verify(out) else 2


def _cmd_token_shards_prefix(args: argparse.Namespace) -> int:
    import pyarrow.parquet as pq

    from j1data import tokshards

    layout = DataLayout.from_env()
    build_id = _build_id(args)
    acc = layout.accepted(build_id)
    manifest = json.loads((acc / "manifest.json").read_text())
    doc_bytes: dict[str, int] = {}
    for shard in manifest["shards"]:
        if shard["split"] == "train":
            t = pq.read_table(acc / shard["file"], columns=["doc_id", "n_bytes"])
            doc_bytes.update(
                zip(t.column("doc_id").to_pylist(), t.column("n_bytes").to_pylist(), strict=True)
            )
    src = layout.root / "token_shards" / build_id / args.tokenizer
    dst = layout.root / "token_shards" / build_id / f"{args.tokenizer}-prefix-{args.max_bytes // 10**6}MB"
    if args.exclude_doc_ids:
        exclude = frozenset(json.loads(Path(args.exclude_doc_ids).read_text()))
        dst = dst.with_name(dst.name + f"-excl{len(exclude)}")
        rec = tokshards.export_prefix_excluding(src, dst, args.max_bytes, doc_bytes, exclude)
    else:
        rec = tokshards.export_prefix(src, dst, args.max_bytes, doc_bytes)
    print(
        json.dumps(
            {
                "dst": str(dst),
                **rec["prefix_of"],
                "train_stream_tokens": rec["splits"]["train"]["stream_tokens"],
            }
        )
    )
    return 0 if not tokshards.verify(dst) else 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m j1data.tokcli")
    sub = parser.add_subparsers(dest="command", required=True)

    tok_c = sub.add_parser("tokenizer-candidates", help="train and evaluate Juniper 1 tokenizer candidates")
    tok_c.add_argument("--build-id")
    tok_c.add_argument(
        "--out-name", default="candidates", help="candidate-set directory under tokenizer/<build>"
    )
    tok_c.add_argument(
        "--exclude-doc-ids", help="JSON list of train doc_ids removed from the sample (recorded)"
    )
    tok_c.set_defaults(func=_cmd_tokenizer_candidates)

    tok_f = sub.add_parser("tokenizer-freeze", help="freeze one evaluated candidate (immutable)")
    tok_f.add_argument("--build-id")
    tok_f.add_argument("--candidate", required=True)
    tok_f.add_argument("--name", required=True)
    tok_f.add_argument("--rationale", required=True)
    tok_f.add_argument("--candidates-dir", default="candidates")
    tok_f.set_defaults(func=_cmd_tokenizer_freeze)

    tshards = sub.add_parser("token-shards", help="model-ready uint16 token-ID shards for a frozen tokenizer")
    tshards.add_argument("--build-id")
    tshards.add_argument("--tokenizer", required=True, help="frozen tokenizer name")
    tshards.add_argument("--verify-only", action="store_true")
    tshards.set_defaults(func=_cmd_token_shards)

    tprefix = sub.add_parser("token-shards-prefix", help="equal-bytes train prefix + validation for pilots")
    tprefix.add_argument("--build-id")
    tprefix.add_argument("--tokenizer", required=True)
    tprefix.add_argument("--max-bytes", type=int, required=True)
    tprefix.add_argument(
        "--exclude-doc-ids", help="JSON list of doc_ids removed by contamination adjudication"
    )
    tprefix.set_defaults(func=_cmd_token_shards_prefix)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
