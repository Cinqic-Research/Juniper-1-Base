"""Real-operation recovery drills for the CPU data stages (gate J1-13, docs/stage_recovery.md).

``setup`` builds a scratch data root whose processing outputs are hard links to a fraction
of the real build's committed units (read-only use; every stage writes new files by
temp-and-rename, which never modifies a linked inode). ``stage`` runs one stage on a
scratch root through the library API (the CLI's ``prepare`` would bind the scratch root
to a new build identity). ``drill`` runs a stage uninterrupted, moves its outputs aside as
the reference, reruns it under SIGKILL of the whole process group at a random time,
restarts it, and compares every output file byte-for-byte.

Usage:
  stage_drill.py setup  --real ROOT --scratch ROOT --fraction F
  stage_drill.py stage  --root ROOT --stage NAME [--real ROOT]
  stage_drill.py drill  --root ROOT --stage NAME --real ROOT --evidence FILE [--seed N] [--mem 4G]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

BUILD = "j1-base-v0.2"
TOKENIZER = "cand5-j1tok-24k"
STAGES = ("dedup", "select", "finalize", "count", "audit-scan", "tokenizer", "token-shards")
# Files that record wall-clock time only; every other output must be byte-identical.
TIMING_FILES = ("recovery_log.json", "build_log.json")
TIMING_FIELDS = ("train_seconds", "peak_rss_kib", "seconds")


def _identity(real: Path) -> str:
    return str(json.loads((real / "work" / BUILD / "build.json").read_text())["identity"])


def setup(real: Path, scratch: Path, fraction: float) -> dict[str, Any]:
    from j1data.hashing import unit_interval

    src_work, dst_work = real / "work" / BUILD, scratch / "work" / BUILD
    if dst_work.exists():
        raise SystemExit(f"{dst_work} exists")
    linked = units = rows = 0
    for marker_path in sorted((src_work / "units").glob("*/*.json")):
        relative = marker_path.relative_to(src_work).as_posix()
        if fraction < 1 and unit_interval("j1.stage-drill.v1", relative) >= fraction:
            continue
        marker = json.loads(marker_path.read_text())
        target = dst_work / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(marker_path, target)
        units += 1
        for segment in marker["segments"]:
            rows += segment["outputs"][0]["rows"]
            for entry in segment["outputs"]:
                destination = dst_work / entry["file"]
                destination.parent.mkdir(parents=True, exist_ok=True)
                os.link(src_work / entry["file"], destination)
                linked += 1
            override = (
                src_work
                / "dedup"
                / "code_signatures"
                / marker["source_id"]
                / f"{marker['unit_id']}~part-{segment['index']:06d}.npy"
            )
            if override.exists():
                destination = dst_work / override.relative_to(src_work)
                destination.parent.mkdir(parents=True, exist_ok=True)
                os.link(override, destination)
                linked += 1
    shutil.copyfile(src_work / "build.json", dst_work / "build.json")
    info = {"fraction": fraction, "units": units, "candidate_rows": rows, "hard_links": linked}
    (scratch / "drill_setup.json").write_text(json.dumps(info, indent=1) + "\n")
    return info


def setup_accepted(real: Path, scratch: Path, fraction: float) -> dict[str, Any]:
    """A scratch accepted build: a keyed fraction of the real train shards plus every held-out shard."""
    from j1data.hashing import unit_interval

    source, target = real / "accepted" / BUILD, scratch / "accepted" / BUILD
    if target.exists():
        raise SystemExit(f"{target} exists")
    manifest = json.loads((source / "manifest.json").read_text())
    kept = [
        shard
        for shard in manifest["shards"]
        if shard["split"] != "train" or unit_interval("j1.stage-drill.v1", shard["file"]) < fraction
    ]
    for shard in kept:
        destination = target / shard["file"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.link(source / shard["file"], destination)
    manifest["shards"] = kept
    (target / "manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    info = {
        "fraction": fraction,
        "shards": len(kept),
        "train_documents": sum(s["rows"] for s in kept if s["split"] == "train"),
    }
    (scratch / "drill_setup.json").write_text(json.dumps(info, indent=1) + "\n")
    return info


def run_stage(root: Path, stage: str, real: Path) -> None:
    from j1data import dedup, finalize, select, splits
    from j1data.build import load_pipeline_config
    from j1data.minhash import LSHConfig
    from j1data.paths import DataLayout
    from j1data.registry import load_registry

    layout = DataLayout(root)
    identity = _identity(real)
    work = layout.work(BUILD)
    priorities = {s: src.priority for s, src in load_registry().items()}
    config = load_pipeline_config()
    if stage == "dedup":
        lsh = LSHConfig(config["dedup"]["bands"], config["dedup"]["rows"])
        summary = dedup.run(
            layout, BUILD, priorities, lsh, None, work / "dedup" / "code_signatures", identity
        )
        print(json.dumps({k: summary[k] for k in ("candidates", "retained", "removed")}))
        return
    refs = dedup.collect_units(layout, BUILD, priorities)
    if stage == "select":
        select.run(refs, work / "dedup" / "keep", work / "selection", select.load_config(), None, identity)
    elif stage == "finalize":
        split = config["splits"]
        settings = finalize.FinalizeSettings(
            layout,
            BUILD,
            splits.SplitConfig(split["namespace"], split["validation_fraction"], split["test_fraction"]),
            split["leak_fraction"],
            identity,
        )
        manifest = finalize.run(settings, refs, 4, None, work / "selection")
        print(json.dumps(manifest["accepted_by_split"]))
    elif stage == "count":
        from tokenizers import Tokenizer

        from j1data import tokenization
        from j1data.token_counts import count_tokens

        manifest = json.loads((layout.accepted(BUILD) / "manifest.json").read_text())
        tokenizers, hashes = {}, {}
        for path in sorted((real / "measurement_tokenizers" / "j1-base-v0.1").glob("*/tokenizer.json")):
            tokenizers[path.parent.name] = Tokenizer.from_file(str(path))
            hashes[path.parent.name] = tokenization.identity(tokenizers[path.parent.name])
        count_tokens(layout.accepted(BUILD), layout.token_counts(BUILD), manifest, tokenizers, hashes)
    elif stage == "audit-scan":
        from j1data.audit_scan import run_scan

        current = (real / "exclusions" / "CURRENT").read_text().strip()
        print(run_scan(root, BUILD, real / "exclusions" / current, workers=2))
    elif stage == "tokenizer":
        import tomllib

        from j1data import tokstage

        config_path = Path(__file__).resolve().parents[1] / "configs" / "juniper_tokenizer.toml"
        tok = tomllib.loads(config_path.read_text())
        out = root / "tokenizer" / BUILD / "candidates"
        tokstage.run(root, BUILD, tok["sample"]["rate"], tok["candidates"]["total_vocab"], out)
    elif stage == "token-shards":
        from j1data import tokshards

        out = root / "token_shards" / BUILD / TOKENIZER
        tokshards.build(root, BUILD, real / "tokenizer" / BUILD / "frozen" / TOKENIZER, out)
    else:
        raise SystemExit(f"unknown stage {stage}")


def outputs(root: Path, stage: str) -> list[Path]:
    work = root / "work" / BUILD
    return {
        "dedup": [
            work / "dedup" / "keep",
            work / "dedup" / "removed.parquet",
            work / "dedup" / "summary.json",
            work / "dedup" / "recovery_log.json",
            work / "dedup" / "state",
        ],
        "select": [work / "selection"],
        "finalize": [root / "accepted" / BUILD, work / "heldout_index"],
        "count": [root / "token_counts" / BUILD],
        "audit-scan": [root / "audits"],
        "tokenizer": [root / "tokenizer" / BUILD / "candidates"],
        "token-shards": [
            root / "token_shards" / BUILD / TOKENIZER,
            root / "token_shards" / BUILD / f"{TOKENIZER}.partial",
        ],
    }[stage]


def _digest(path: Path) -> str:
    if path.name == "eval.json" or (path.name == "summary.json" and "tokenizer" in path.parts):
        value = json.loads(path.read_text())

        def strip(node: Any) -> Any:
            if isinstance(node, dict):
                return {k: strip(v) for k, v in node.items() if k not in TIMING_FIELDS}
            if isinstance(node, list):
                return [strip(v) for v in node]
            return node

        return (
            "json-without-timings:"
            + hashlib.sha256(json.dumps(strip(value), sort_keys=True).encode()).hexdigest()
        )
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 22):
            digest.update(chunk)
    return digest.hexdigest()


def tree(paths: list[Path], base: Path) -> dict[str, str]:
    result = {}
    for top in paths:
        if top.is_file():
            files = [top]
        else:
            files = sorted(p for p in top.rglob("*") if p.is_file()) if top.is_dir() else []
        for path in files:
            if path.name in TIMING_FILES or "state" in path.relative_to(base).parts[-2:-1]:
                continue
            result[path.relative_to(base).as_posix()] = _digest(path)
    return result


def _command(root: Path, stage: str, real: Path, mem: str, time_file: Path) -> list[str]:
    return [
        "systemd-run", "--user", "--scope", "--quiet", "-p", f"MemoryMax={mem}", "-p", "MemorySwapMax=0",
        "/usr/bin/time", "-v", "-o", str(time_file),
        sys.executable, str(Path(__file__).resolve()), "stage", "--root", str(root), "--stage", stage,
        "--real", str(real),
    ]  # fmt: skip


def _time_record(path: Path) -> dict[str, Any]:
    text = path.read_text() if path.exists() else ""
    wall = re.search(r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\): (\S+)", text)
    rss = re.search(r"Maximum resident set size \(kbytes\): (\d+)", text)
    status = re.search(r"Exit status: (\d+)", text)
    seconds = None
    if wall:
        parts = [float(p) for p in wall.group(1).split(":")]
        seconds = sum(p * 60**i for i, p in enumerate(reversed(parts)))
    return {
        "wall_seconds": seconds,
        "peak_rss_kib": int(rss.group(1)) if rss else None,
        "exit_status": int(status.group(1)) if status else None,
    }


def _run(cmd: list[str], log: Path, kill_after: float | None = None) -> dict[str, Any]:
    started = time.monotonic()
    with log.open("ab") as handle:
        process = subprocess.Popen(cmd, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
        killed = False
        if kill_after is not None:
            try:
                process.wait(timeout=kill_after)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                killed = True
        returncode = process.wait()
    return {"returncode": returncode, "killed": killed, "elapsed": round(time.monotonic() - started, 2)}


def drill(root: Path, stage: str, real: Path, evidence: Path, seed: int, mem: str) -> dict[str, Any]:
    rng = random.Random(seed)
    logs = evidence.with_suffix("")
    logs.mkdir(parents=True, exist_ok=True)
    reference_dir = root / "drill-reference" / stage
    if reference_dir.exists():
        shutil.rmtree(reference_dir)
    for path in outputs(root, stage):
        if path.exists():
            raise SystemExit(f"{path} exists: drills start from absent stage outputs")
    repo = Path(__file__).resolve().parents[1]
    head = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout
    dirty = subprocess.run(
        ["git", "-C", str(repo), "status", "--porcelain", "src"], capture_output=True, text=True
    )
    record: dict[str, Any] = {
        "schema": "j1.stage-recovery-drill.v1",
        "stage": stage,
        "root": str(root),
        "seed": seed,
        "code_commit": head.strip(),
        "src_dirty": bool(dirty.stdout.strip()),
    }
    ref = _run(_command(root, stage, real, mem, logs / "reference.time"), logs / "reference.log")
    record["reference"] = {**ref, **_time_record(logs / "reference.time")}
    if ref["returncode"]:
        raise SystemExit(f"reference run failed: {logs / 'reference.log'}")
    reference_tree = tree(outputs(root, stage), root)
    reference_dir.mkdir(parents=True)
    for path in outputs(root, stage):
        if path.exists():
            shutil.move(str(path), str(reference_dir / path.name))
    wall = record["reference"]["wall_seconds"] or ref["elapsed"]
    kill_after = rng.uniform(0.05, 0.9) * wall
    killed = _run(_command(root, stage, real, mem, logs / "killed.time"), logs / "killed.log", kill_after)
    record["killed"] = {**killed, "kill_after_seconds": round(kill_after, 2)}
    restart = _run(_command(root, stage, real, mem, logs / "restart.time"), logs / "restart.log")
    record["restart"] = {**restart, **_time_record(logs / "restart.time")}
    resumed_tree = tree(outputs(root, stage), root)
    differ = sorted(
        k for k in set(reference_tree) | set(resumed_tree) if reference_tree.get(k) != resumed_tree.get(k)
    )
    leftovers = [
        p.relative_to(root).as_posix()
        for top in outputs(root, stage) if top.exists()
        for p in ([top] if top.is_file() else top.rglob("*"))
        if p.name.endswith((".tmp", ".partial")) or ".tmp." in p.name
    ]  # fmt: skip
    record["comparison"] = {
        "files": len(reference_tree),
        "identical": not differ and bool(reference_tree),
        "differing_files": differ[:50],
        "leftover_temporary_files": leftovers[:50],
        "excluded_timing_files": list(TIMING_FILES),
        "excluded_timing_json_fields": list(TIMING_FIELDS) if stage == "tokenizer" else [],
    }
    for name in ("recovery_log.json", "build_log.json"):
        for top in outputs(root, stage):
            candidate = top / name if top.is_dir() else None
            if candidate is not None and candidate.exists():
                record.setdefault("stage_recovery_logs", {})[name] = json.loads(candidate.read_text())
    record["pass"] = bool(
        killed["killed"]
        and restart["returncode"] == 0
        and record["comparison"]["identical"]
        and not leftovers
    )
    shutil.rmtree(reference_dir)
    evidence.parent.mkdir(parents=True, exist_ok=True)
    evidence.write_text(json.dumps(record, indent=1, sort_keys=True) + "\n")
    return record


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("setup")
    s.add_argument("--real", type=Path, required=True)
    s.add_argument("--scratch", type=Path, required=True)
    s.add_argument("--fraction", type=float, required=True)
    sa = sub.add_parser("setup-accepted")
    sa.add_argument("--real", type=Path, required=True)
    sa.add_argument("--scratch", type=Path, required=True)
    sa.add_argument("--fraction", type=float, required=True)
    st = sub.add_parser("stage")
    st.add_argument("--root", type=Path, required=True)
    st.add_argument("--stage", choices=STAGES, required=True)
    st.add_argument("--real", type=Path, required=True)
    d = sub.add_parser("drill")
    d.add_argument("--root", type=Path, required=True)
    d.add_argument("--stage", choices=STAGES, required=True)
    d.add_argument("--real", type=Path, required=True)
    d.add_argument("--evidence", type=Path, required=True)
    d.add_argument("--seed", type=int, default=20261010)
    d.add_argument("--mem", default="4G")
    args = parser.parse_args()
    if args.command == "setup":
        print(json.dumps(setup(args.real, args.scratch, args.fraction)))
    elif args.command == "setup-accepted":
        print(json.dumps(setup_accepted(args.real, args.scratch, args.fraction)))
    elif args.command == "stage":
        run_stage(args.root, args.stage, args.real)
    else:
        result = drill(args.root, args.stage, args.real, args.evidence, args.seed, args.mem)
        print(json.dumps({k: result[k] for k in ("stage", "pass", "killed", "comparison")}, default=str))
        return 0 if result["pass"] else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
