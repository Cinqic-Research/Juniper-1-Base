"""Command-line entry point: ``python -m j1data <stage> ...``."""

from __future__ import annotations

import argparse
import fnmatch
import json
import sys
from pathlib import Path
from typing import Any

from j1data.paths import DataLayout
from j1data.registry import load_registry


def _cmd_acquire(args: argparse.Namespace) -> int:
    from j1data.acquire import acquire_git, acquire_hf, list_hf_files

    layout = DataLayout.from_env()
    registry = load_registry()
    wanted = args.sources or [s for s, src in registry.items() if src.admissible]
    clones = _clones(args.git_clones)
    for source_id in wanted:
        source = registry[source_id]
        if not source.admissible:
            print(f"{source_id}: skipped (license state {source.license_state})", file=sys.stderr)
            continue
        mechanism = source.raw["acquisition"]["mechanism"]
        if args.dry_run:
            if mechanism == "huggingface":
                listing = list_hf_files(source.repo, source.revision)
                chosen = source.selection().apply([e["path"] for e in listing])
                size = sum(e["size"] for e in listing if e["path"] in set(chosen))
                print(f"{source_id}: {len(chosen)}/{len(listing)} files, {size / 1e9:.2f} GB")
            continue
        if mechanism == "huggingface":
            manifest = acquire_hf(source, layout, workers=args.workers)
        else:
            if source.repo not in clones:
                raise SystemExit(f"{source_id}: no local clone given for {source.repo} (--git-clones)")
            manifest = acquire_git(source, layout, Path(clones[source.repo]))
        print(
            f"{source_id}: {manifest['selected_file_count']} files, "
            f"{manifest['selected_bytes'] / 1e9:.3f} GB",
            flush=True,
        )
    return 0


def _clones(path: str | None) -> dict[str, str]:
    return dict(json.loads(Path(path).read_text())) if path else {}


def _cmd_exclusions(args: argparse.Namespace) -> int:
    from j1data.build import load_pipeline_config
    from j1data.exclusions import build

    layout = DataLayout.from_env()
    config = load_pipeline_config()["contamination"]
    identity, manifest = build(
        layout,
        config["ngram"],
        config["min_coverage"],
        config["min_informative_words"],
        config["short_min_words"],
        config["short_min_chars"],
        config["short_min_informative_words"],
        _clones(args.git_clones),
    )
    (layout.exclusions() / "CURRENT").write_text(identity[:16] + "\n")
    for name, info in manifest["benchmarks"].items():
        print(
            f"{name}: {info['items']} items, {info['ngrams']} n-grams, "
            f"{info['unmatchable_items']} unmatchable"
        )
    print(f"index {identity[:16]}: {manifest['total_ngrams']} n-grams")
    return 0


def _process_job(job: tuple[Any, ...]) -> dict[str, Any]:
    from j1data.process import process_unit

    return process_unit(*job)


def _validate_process_sources(registry: dict[str, Any], wanted: list[str]) -> list[str]:
    """Fail closed when an explicit process request names an unknown or inadmissible source."""
    for source_id in wanted:
        source = registry.get(source_id)
        if source is None:
            raise SystemExit(f"unknown source {source_id!r}")
        if not source.admissible:
            raise SystemExit(
                f"{source_id}: processing refused (license state {source.license_state}); "
                "only explicitly admissible source states may be processed"
            )
    if len(set(wanted)) != len(wanted):
        raise SystemExit("duplicate source id in --sources")
    return wanted


def _require_exploratory_build(
    build_id: str,
    release_build_id: str,
    selected_sources: list[str],
    release_sources: list[str],
    unit_glob: str | None,
    limit_units: int | None,
) -> None:
    partial = bool(unit_glob or limit_units) or set(selected_sources) != set(release_sources)
    if partial and build_id == release_build_id:
        raise SystemExit(
            "partial source or unit processing is not allowed in the configured release build; "
            "choose a distinct --build-id for exploratory work"
        )


def _apply_scope(source_id: str, units: list[Any]) -> list[Any]:
    """Restrict a source to the units named by configs/build_scope.toml (release builds only)."""
    import tomllib

    from j1data.hashing import unit_interval
    from j1data.registry import REPO_ROOT

    with (REPO_ROOT / "configs" / "build_scope.toml").open("rb") as handle:
        scope = tomllib.load(handle).get("sources", {}).get(source_id)
    if scope is None:
        return units
    if "include" in scope:
        return [u for u in units if u.unit_id in set(scope["include"])]
    ranked = sorted(units, key=lambda u: unit_interval(scope["namespace"], u.unit_id))
    keep = {u.unit_id for u in ranked[: scope["count"]]}
    return [u for u in units if u.unit_id in keep]


def _cmd_process(args: argparse.Namespace) -> int:

    from j1data.adapters import make_adapter
    from j1data.build import load_pipeline_config, prepare
    from j1data.process import init_worker, unit_complete

    layout = DataLayout.from_env()
    registry = load_registry()
    release_sources = sorted(
        (s for s, src in registry.items() if src.admissible), key=lambda s: registry[s].priority
    )
    wanted = args.sources or release_sources
    wanted = _validate_process_sources(registry, wanted)
    release_build_id = load_pipeline_config()["build"]["id"]
    requested_build_id = args.build_id or release_build_id
    _require_exploratory_build(
        requested_build_id,
        release_build_id,
        wanted,
        release_sources,
        args.unit_glob,
        args.limit_units,
    )
    settings = prepare(layout, allow_dirty=args.allow_dirty, build_id=args.build_id)
    verify_outputs = args.verify or settings.build_id == release_build_id
    clones = _clones(args.git_clones)
    jobs: list[tuple[Any, ...]] = []
    verified_manifest_info: dict[str, dict[str, Any]] = {}
    from j1data.shards import write_json_atomic

    preflight_path = layout.reports(settings.build_id) / "source_integrity_preflight.json"
    write_json_atomic(
        preflight_path,
        {
            "schema": "j1.source-integrity-preflight.v1",
            "build_identity": settings.identity,
            "verified_at_utc": None,
            "sources": {},
        },
    )
    for source_id in wanted:
        source = registry[source_id]
        manifest = json.loads(layout.acquisition_manifest(source_id).read_text())
        raw_dir = layout.raw_dir(source_id, source.revision)
        clone = Path(clones[source.repo]) if source.adapter == "cinqic_git" else None
        from j1data.acquire import verify_acquired_source

        verified = verify_acquired_source(source, layout, manifest, clone)
        if settings.verified_sources is None:
            settings.verified_sources = set()
        settings.verified_sources.add(source_id)
        verified_manifest_info[source_id] = verified
        print(
            f"{source_id}: acquisition identity {verified['manifest_identity'][:16]}, "
            f"{verified['files']} files verified ({verified['bytes'] / 1e9:.2f} GB)",
            flush=True,
        )
        units = make_adapter(source, raw_dir, manifest, clone).units()
        if args.unit_glob:
            units = [u for u in units if fnmatch.fnmatch(u.unit_id, args.unit_glob)]
        elif settings.build_id == release_build_id:
            units = _apply_scope(source_id, units)
        if args.limit_units:
            units = units[: args.limit_units]
        pending = [
            u
            for u in units
            if not unit_complete(layout, settings.build_id, u, settings.identity, verify_outputs)
        ]
        print(f"{source_id}: {len(units)} units, {len(pending)} pending", flush=True)
        jobs.extend((source, u, raw_dir, manifest, clone) for u in pending)
    from datetime import UTC, datetime

    write_json_atomic(
        preflight_path,
        {
            "schema": "j1.source-integrity-preflight.v1",
            "build_identity": settings.identity,
            "verified_at_utc": datetime.now(UTC).isoformat(),
            "sources": verified_manifest_info,
        },
    )
    if not jobs:
        return 0
    from j1data.pools import run_unordered

    results = run_unordered(
        _process_job, jobs, args.workers, initializer=init_worker, initargs=(settings,), max_tasks_per_child=8
    )
    for done, marker in enumerate(results, 1):
        kept = marker["stages"].get("candidate", 0)
        print(
            f"[{done}/{len(jobs)}] {marker['source_id']}/{marker['unit_id']}: {marker['docs_in']} docs, "
            f"{kept} candidates, {marker['bytes_in'] / 1e6:.0f} MB in {marker['seconds']:.0f}s",
            flush=True,
        )
    return 0


def _priorities() -> dict[str, int]:
    return {s: src.priority for s, src in load_registry().items()}


def _cmd_dedup(args: argparse.Namespace) -> int:
    from j1data import dedup
    from j1data.build import code_revision, load_pipeline_config, prepare

    layout = DataLayout.from_env()
    settings = prepare(layout, allow_dirty=args.allow_dirty, build_id=args.build_id)
    from j1data.code_signatures import resign
    from j1data.minhash import LSHConfig

    config = load_pipeline_config()
    lsh = LSHConfig(config["dedup"]["bands"], config["dedup"]["rows"])
    lsh.check(config["minhash"]["num_perm"])
    overrides = layout.work(settings.build_id) / "dedup" / "code_signatures"
    code_sources = {s for s, src in load_registry().items() if src.raw["family"] == "code"}
    refs = dedup.collect_units(layout, settings.build_id, _priorities())
    resigned = resign(refs, overrides, settings.minhash, code_sources, args.workers)
    print(f"re-signed {resigned} code units without license text", flush=True)
    summary = dedup.run(
        layout,
        settings.build_id,
        _priorities(),
        lsh,
        code_revision(),
        overrides,
        settings.identity,
    )
    print(json.dumps({k: summary[k] for k in ("candidates", "retained", "removed")}, indent=1))
    return 0


def _cmd_finalize(args: argparse.Namespace) -> int:
    from j1data import finalize, select, splits
    from j1data.build import code_revision, load_pipeline_config, prepare
    from j1data.dedup import collect_units
    from j1data.registry import load_registry

    layout = DataLayout.from_env()
    settings = prepare(layout, allow_dirty=args.allow_dirty, build_id=args.build_id)
    config = load_pipeline_config()["splits"]
    split_config = splits.SplitConfig(
        config["namespace"], config["validation_fraction"], config["test_fraction"]
    )
    final = finalize.FinalizeSettings(
        layout, settings.build_id, split_config, config["leak_fraction"], settings.identity
    )
    selection = layout.work(settings.build_id) / "selection"
    refs = collect_units(layout, settings.build_id, _priorities())
    admissible_sources = {source_id for source_id, source in load_registry().items() if source.admissible}
    inadmissible_sources = sorted({ref.source_id for ref in refs} - admissible_sources)
    if inadmissible_sources:
        raise PermissionError(f"cannot finalize unknown or inadmissible sources: {inadmissible_sources}")
    release_build_id = load_pipeline_config()["build"]["id"]
    require_selection = settings.build_id == release_build_id
    selection_dir = selection if (selection / "summary.json").exists() else None
    if require_selection and selection_dir is None:
        raise RuntimeError("release build cannot finalize before the quota selection stage")
    if selection_dir is not None:
        finalize.validate_selection(
            refs, selection_dir, settings.identity, code_revision(), select.load_config()
        )
    if args.rebuild:
        import shutil

        shutil.rmtree(layout.accepted(settings.build_id), ignore_errors=True)
    manifest = finalize.run(
        final,
        refs,
        args.workers,
        code_revision(),
        selection_dir,
        admissible_sources,
        require_selection,
        select.load_config(),
    )
    print(
        json.dumps(
            {
                k: manifest[k]
                for k in (
                    "candidates",
                    "deduplicated_away",
                    "not_selected",
                    "split_leakage_removed",
                    "accepted_by_split",
                )
            },
            indent=1,
        )
    )
    return 0


def _cmd_select(args: argparse.Namespace) -> int:
    from j1data import select
    from j1data.build import code_revision, prepare
    from j1data.dedup import collect_units
    from j1data.registry import load_registry

    layout = DataLayout.from_env()
    settings = prepare(layout, allow_dirty=args.allow_dirty, build_id=args.build_id)
    build_id = settings.build_id
    work = layout.work(build_id)
    refs = collect_units(layout, build_id, _priorities())
    admissible = {source_id for source_id, source in load_registry().items() if source.admissible}
    inadmissible = sorted({ref.source_id for ref in refs} - admissible)
    if inadmissible:
        raise PermissionError(f"cannot select unknown or inadmissible sources: {inadmissible}")
    summary = select.run(
        refs,
        work / "dedup" / "keep",
        work / "selection",
        select.load_config(),
        code_revision(),
        settings.identity,
    )
    for source, entry in summary["sources"].items():
        print(
            f"{source}: selected {sum(entry['selected_bytes'].values()) / 1e9:.2f} GB of "
            f"{sum(entry['available_bytes_after_dedup'].values()) / 1e9:.2f} GB"
        )
    return 0


def _measurement_config() -> dict[str, Any]:
    import tomllib

    from j1data.registry import REPO_ROOT

    with (REPO_ROOT / "configs" / "measurement_tokenizers.toml").open("rb") as handle:
        return tomllib.load(handle)


def _load_measurement_tokenizers(layout: DataLayout, build_id: str) -> tuple[dict[str, Any], dict[str, str]]:
    from tokenizers import Tokenizer

    from j1data import tokenization

    tokenizers, hashes = {}, {}
    for entry in _measurement_config()["tokenizer"]:
        path = layout.measurement_tokenizers(build_id) / entry["name"] / "tokenizer.json"
        tokenizer = Tokenizer.from_file(str(path))
        tokenizers[entry["name"]] = tokenizer
        hashes[entry["name"]] = tokenization.identity(tokenizer)
    return tokenizers, hashes


def _cmd_measure_tokenizers(args: argparse.Namespace) -> int:
    from j1data import tokenization
    from j1data.build import code_revision
    from j1data.shards import write_json_atomic
    from j1data.token_counts import build_probes, tokenizer_sample, write_sample

    layout = DataLayout.from_env()
    build_id = _build_id(args)
    accepted = layout.accepted(build_id)
    manifest = json.loads((accepted / "manifest.json").read_text())
    config = _measurement_config()
    texts, sample_info = tokenizer_sample(accepted, manifest, config["sample"]["rate"])
    print(f"sample: {sample_info['documents']} documents, {sample_info['bytes'] / 1e9:.2f} GB", flush=True)
    stage = layout.tokenizer_stage(build_id)
    write_sample(stage, texts, sample_info)
    write_json_atomic(stage / "probes.json", build_probes(accepted, manifest))
    for entry in config["tokenizer"]:
        spec = tokenization.MeasurementSpec(
            entry["name"], entry["total_vocab"], entry["limit_alphabet"], entry["min_frequency"]
        )
        tokenizer = tokenization.train(spec, texts)
        directory = layout.measurement_tokenizers(build_id) / spec.name
        directory.mkdir(parents=True, exist_ok=True)
        tokenizer.save(str(directory / "tokenizer.json.partial"))
        (directory / "tokenizer.json.partial").replace(directory / "tokenizer.json")
        info = {
            "schema": "j1.measurement-tokenizer.v1",
            "status": "PROVISIONAL MEASUREMENT ONLY - not a Juniper 1 tokenizer, not frozen",
            "version": tokenization.VERSION,
            "name": spec.name,
            "vocab_size": tokenizer.get_vocab_size(),
            "spec": entry,
            "pretokenize_pattern": tokenization.PRETOKENIZE_PATTERN,
            "sample": sample_info,
            "tokenizer_sha256": tokenization.identity(tokenizer),
            "code": code_revision(),
        }
        write_json_atomic(directory / "info.json", info)
        print(f"{spec.name}: {info['tokenizer_sha256'][:16]}", flush=True)
    return 0


def _cmd_count_tokens(args: argparse.Namespace) -> int:
    from j1data.token_counts import count_tokens

    layout = DataLayout.from_env()
    build_id = _build_id(args)
    manifest = json.loads((layout.accepted(build_id) / "manifest.json").read_text())
    tokenizers, hashes = _load_measurement_tokenizers(layout, build_id)
    markers = count_tokens(
        layout.accepted(build_id), layout.token_counts(build_id), manifest, tokenizers, hashes
    )
    totals: dict[str, int] = {}
    for marker in markers:
        for name, value in marker["tokens"].items():
            totals[name] = totals.get(name, 0) + value
    print(json.dumps(totals, indent=1))
    return 0


def _build_id(args: argparse.Namespace) -> str:
    from j1data.build import load_pipeline_config

    return str(args.build_id or load_pipeline_config()["build"]["id"])


def _cmd_audit(args: argparse.Namespace) -> int:
    from j1data.audit import build_pack
    from j1data.build import load_pipeline_config

    layout = DataLayout.from_env()
    build_id = _build_id(args)
    prose = load_pipeline_config()["quality"]
    boundaries = {
        "english_fw_frac": (prose["kinds"]["prose"]["min_english_function_word_frac"], 0.01),
        "code_line_frac": (prose["technical_code_line_frac"], 0.01),
    }
    out = layout.reports(build_id) / "audit_pack.md"
    accepted = layout.accepted(build_id)
    admitted_root = accepted if (accepted / "manifest.json").exists() else None
    summary = build_pack(
        layout,
        build_id,
        out,
        sample_rate=args.sample_rate,
        boundaries=boundaries,
        admitted_root=admitted_root,
    )
    print(json.dumps(summary))
    print(out)
    return 0


def _cmd_repeated_lines(args: argparse.Namespace) -> int:
    from j1data.audit import repeated_lines
    from j1data.shards import write_json_atomic

    layout = DataLayout.from_env()
    build_id = _build_id(args)
    result = repeated_lines(layout, build_id, args.sample_rate)
    write_json_atomic(layout.reports(build_id) / "repeated_lines.json", result)
    for source, rows in result.items():
        print(f"== {source} ({rows[0][0]} documents sampled)")
        for count, line in rows[1:16]:
            print(f"  {count:6d} {line[:100]}")
    return 0


def _cmd_calibrate_dedup(args: argparse.Namespace) -> int:
    from j1data.dedup import collect_units
    from j1data.dedup_calibration import calibrate
    from j1data.minhash import LSHConfig

    layout = DataLayout.from_env()
    build_id = _build_id(args)
    settings = [LSHConfig(*map(int, s.split("x"))) for s in args.settings]
    probe = LSHConfig(*map(int, args.probe.split("x")))
    refs = collect_units(layout, build_id, _priorities())
    summary = calibrate(
        refs, settings, probe, args.pairs, args.label_pairs, layout.reports(build_id) / "dedup_calibration"
    )
    print(json.dumps(summary["curves"], indent=1))
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    from j1data.report import accepted_statistics, process_summary, rejected_token_estimates, summary_markdown
    from j1data.shards import write_json_atomic

    layout = DataLayout.from_env()
    build_id = _build_id(args)
    out = layout.reports(build_id)
    process = process_summary(layout, build_id)
    write_json_atomic(out / "process_summary.json", process)
    manifest_path = layout.accepted(build_id) / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        counts = layout.token_counts(build_id)
        counts_root = counts if (counts / "counts").exists() else None
        stats = accepted_statistics(layout.accepted(build_id), counts_root, manifest)
        write_json_atomic(out / "accepted_statistics.json", stats)
        estimates = None
        if stats["tokenizers"]:
            estimates = rejected_token_estimates(process, stats, stats["tokenizers"][-1])
            write_json_atomic(out / "rejected_token_estimates.json", estimates)
        dedup_summary = json.loads((layout.work(build_id) / "dedup" / "summary.json").read_text())
        (out / "corpus_summary.md").write_text(
            summary_markdown(build_id, process, dedup_summary, manifest, stats, estimates)
        )
    print(out)
    return 0


def _cmd_verify(args: argparse.Namespace) -> int:
    from datetime import UTC, datetime

    from j1data import splits, verify
    from j1data.build import code_revision, load_pipeline_config, prepare
    from j1data.hashing import sha256_file
    from j1data.shards import write_json_atomic

    layout = DataLayout.from_env()
    build_id = _build_id(args)
    build_settings = prepare(layout, build_id=args.build_id)
    config = load_pipeline_config()
    accepted = layout.accepted(build_id)
    manifest_path = accepted / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    split_cfg = config["splits"]
    split_config = splits.SplitConfig(
        split_cfg["namespace"], split_cfg["validation_fraction"], split_cfg["test_fraction"]
    )
    result: dict[str, Any] = {
        "schema": "j1.verification.v2",
        "build_id": build_id,
        "build_identity": build_settings.identity,
        "accepted_manifest_sha256": sha256_file(manifest_path),
        "verified_at_utc": datetime.now(UTC).isoformat(),
        "code": code_revision(),
        "shards": verify.check_shards(accepted, manifest),
        "uniqueness_and_splits": verify.check_uniqueness_and_splits(accepted, manifest, split_config),
        "accounting": verify.check_accounting(layout, build_id, manifest),
    }
    if (layout.token_counts(build_id) / "counts").exists():
        tokenizers, _ = _load_measurement_tokenizers(layout, build_id)
        result["provisional_token_counts"] = verify.check_token_counts(
            accepted, layout.token_counts(build_id), manifest, tokenizers, args.token_sample_rate
        )
    if args.contamination_sample_rate > 0:
        from j1data.exclusions import collect_items

        contamination = config["contamination"]
        items = collect_items(layout, _clones(args.git_clones))
        result["contamination"] = verify.check_contamination_sample(
            accepted,
            manifest,
            items,
            contamination["ngram"],
            contamination["min_coverage"],
            contamination["min_informative_words"],
            args.contamination_sample_rate,
            contamination["short_min_words"],
            contamination["short_min_chars"],
            contamination["short_min_informative_words"],
        )
    write_json_atomic(layout.reports(build_id) / "verification.json", result)
    print(json.dumps(result, indent=1))
    return 0


def _cmd_sources_doc(args: argparse.Namespace) -> int:
    import tomllib

    from j1data.registry import REPO_ROOT
    from j1data.report import process_summary, sources_markdown

    layout = DataLayout.from_env()
    registry = load_registry()
    with (REPO_ROOT / "configs" / "considered_sources.toml").open("rb") as handle:
        considered = tomllib.load(handle)["considered"]
    acquisitions = {}
    for source_id in registry:
        path = layout.acquisition_manifest(source_id)
        if path.exists():
            acquisitions[source_id] = json.loads(path.read_text())
    build_id = _build_id(args)
    process = process_summary(layout, build_id) if (layout.work(build_id) / "units").exists() else None
    out = REPO_ROOT / "docs" / "sources.md"
    out.write_text(sources_markdown(registry, considered, acquisitions, process))
    print(out)
    return 0


def _teacher_settings() -> dict[str, Any]:
    import tomllib

    from j1data.registry import REPO_ROOT

    with (REPO_ROOT / "configs" / "teacher.toml").open("rb") as handle:
        return tomllib.load(handle)


def _cmd_teacher_prompts(args: argparse.Namespace) -> int:
    import pyarrow.parquet as pq

    from j1data.acquire import HF_ENDPOINT, download_file, list_hf_files
    from j1data.hashing import unit_interval

    layout = DataLayout.from_env()
    settings = _teacher_settings()
    spec = settings["prompts"][args.source]
    listing = {e["path"]: e for e in list_hf_files(spec["repo"], spec["revision"])}
    entry = listing[spec["file"]]
    target = layout.root / "synthetic" / "prompt_sources" / args.source / spec["revision"] / spec["file"]
    url = f"{HF_ENDPOINT}/datasets/{spec['repo']}/resolve/{spec['revision']}/{spec['file']}"
    download_file(url, target, int(entry["size"]), (entry.get("lfs") or {}).get("oid"))
    rows = pq.read_table(target).to_pylist()
    ranked = sorted(
        range(len(rows)), key=lambda i: unit_interval(f"j1.teacher-prompts.{args.source}", str(i))
    )
    template = settings["template"]
    prompts = []
    for i in ranked[: args.limit]:
        answer = rows[i]["answer"].split("####")[-1].strip()
        prompts.append(
            {
                "prompt_id": f"{args.source}-{i}",
                "source": args.source,
                "source_license": spec["license"],
                "template_version": template["version"],
                "system": template["system"],
                "developer": template["developer"],
                "user": rows[i]["question"],
                "oracle": {"kind": "known_answer", "answer": answer},
            }
        )
    out = layout.root / "synthetic" / "prompts" / f"{args.source}-{args.limit}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(p, sort_keys=True) + "\n" for p in prompts))
    print(out)
    return 0


def _cmd_teacher_run(args: argparse.Namespace) -> int:
    from j1data import teacher
    from j1data.build import current_exclusion_dir
    from j1data.contamination import ContaminationIndex

    layout = DataLayout.from_env()
    settings = _teacher_settings()
    t = settings["teacher"]
    config = teacher.TeacherConfig(
        endpoint=t["endpoint"],
        model=t["model"],
        upstream_revision=t["upstream_revision"],
        artifact_sha256=t["artifact_sha256"],
        runtime=t["runtime"],
        temperature=t["temperature"],
        top_p=t["top_p"],
        max_tokens=t["max_tokens"],
        reasoning_effort=t["reasoning_effort"],
        candidates=t["candidates"],
        base_seed=t["base_seed"],
        template_version=settings["template"]["version"],
        system=settings["template"]["system"],
        developer=settings["template"]["developer"],
    )
    prompts = [json.loads(line) for line in Path(args.prompts).read_text().splitlines() if line]
    index = ContaminationIndex.load(current_exclusion_dir(layout))
    counts = teacher.run_batch(prompts, config, index, layout.root / "synthetic" / "runs" / args.run_id)
    print(json.dumps(counts))
    return 0


def _cmd_gates(args: argparse.Namespace) -> int:
    from j1data import gates
    from j1data.shards import write_json_atomic

    layout = DataLayout.from_env()
    build_id = _build_id(args)
    if args.record_tests:
        print(json.dumps(gates.record_tests(layout, build_id)))
    results = gates.evaluate(layout, build_id)
    write_json_atomic(layout.reports(build_id) / "freeze_gates.json", results)
    for result in results:
        print(f"{result['status']:15s} {result['gate']}: {result['evidence']}")
    return 0 if all(r["status"] in (gates.PASS, gates.NOT_APPLICABLE) for r in results) else 1


def _cmd_export_manifests(args: argparse.Namespace) -> int:
    """Copy the small, reviewable build records from the data root into the repository."""
    import shutil

    from j1data.build import current_exclusion_dir
    from j1data.registry import REPO_ROOT
    from j1data.shards import write_json_atomic

    layout = DataLayout.from_env()
    build_id = _build_id(args)
    manifests = REPO_ROOT / "manifests" / build_id
    reports = REPO_ROOT / "reports" / build_id
    for source_id in load_registry():
        path = layout.acquisition_manifest(source_id)
        if path.exists():
            (manifests / "sources").mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, manifests / "sources" / path.name)
    copies = {
        layout.work(build_id) / "build.json": manifests / "build.json",
        current_exclusion_dir(layout) / "manifest.json": manifests / "exclusions.json",
        layout.work(build_id) / "dedup" / "summary.json": manifests / "dedup_summary.json",
        layout.work(build_id) / "selection" / "summary.json": manifests / "selection_summary.json",
        layout.accepted(build_id) / "manifest.json": manifests / "accepted_manifest.json",
        layout.tokenizer_stage(build_id) / "sample.json": manifests / "tokenizer_stage_sample.json",
    }
    for name in (
        "process_summary.json",
        "accepted_statistics.json",
        "rejected_token_estimates.json",
        "corpus_summary.md",
        "verification.json",
        "freeze_gates.json",
        "tests.json",
    ):
        copies[layout.reports(build_id) / name] = reports / name
    for source, target in copies.items():
        if source.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    for directory in sorted(layout.measurement_tokenizers(build_id).glob("*")):
        target = manifests / "measurement_tokenizers" / directory.name
        target.mkdir(parents=True, exist_ok=True)
        for name in ("tokenizer.json", "info.json"):
            shutil.copy2(directory / name, target / name)
    counts = layout.token_counts(build_id) / "_units"
    if counts.exists():
        rows = [json.loads(p.read_text()) for p in sorted(counts.rglob("*.json"))]
        write_json_atomic(
            manifests / "token_counts.json",
            {
                "schema": "j1.token-count-summary.v1",
                "status": "PROVISIONAL - measurement tokenizers, not a Juniper 1 tokenizer",
                "tokenizers": rows[0]["tokenizers"] if rows else {},
                "shards": [
                    {
                        k: r[k]
                        for k in (
                            "accepted_shard",
                            "accepted_sha256",
                            "split",
                            "source_id",
                            "documents",
                            "tokens",
                        )
                    }
                    | {"counts_sha256": r["outputs"][0]["sha256"]}
                    for r in rows
                ],
            },
        )
    print(manifests)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="j1data")
    sub = parser.add_subparsers(dest="command", required=True)

    acquire = sub.add_parser("acquire", help="download or pin admissible sources")
    acquire.add_argument("sources", nargs="*")
    acquire.add_argument("--dry-run", action="store_true")
    acquire.add_argument("--workers", type=int, default=3)
    acquire.add_argument("--git-clones", help="JSON file mapping owner/repo to a local clone path")
    acquire.set_defaults(func=_cmd_acquire)

    exclusions = sub.add_parser("exclusions", help="build the evaluation-exclusion index")
    exclusions.add_argument("--git-clones", required=True)
    exclusions.set_defaults(func=_cmd_exclusions)

    process = sub.add_parser("process", help="filter raw units into candidate shards")
    process.add_argument("sources", nargs="*")
    process.add_argument("--workers", type=int, default=12)
    process.add_argument("--git-clones")
    process.add_argument("--verify", action="store_true", help="re-hash outputs of completed units")
    process.add_argument("--limit-units", type=int, default=0)
    process.add_argument(
        "--unit-glob", help="only process units whose id matches this glob (exploratory runs)"
    )
    process.add_argument("--allow-dirty", action="store_true")
    process.add_argument("--build-id", help="override the configured build id (exploratory runs)")
    process.set_defaults(func=_cmd_process)

    for name, func, help_text in (
        ("dedup", _cmd_dedup, "corpus-wide exact and near deduplication"),
        ("finalize", _cmd_finalize, "assign splits, check leakage, write accepted shards"),
    ):
        stage = sub.add_parser(name, help=help_text)
        stage.add_argument("--allow-dirty", action="store_true")
        stage.add_argument("--build-id")
        stage.add_argument("--workers", type=int, default=10)
        stage.set_defaults(func=func)
        if name == "finalize":
            stage.add_argument(
                "--rebuild", action="store_true", help="discard accepted shards and finalize again"
            )

    select_parser = sub.add_parser(
        "select", help="choose the accepted corpus from the deduplicated pool by quota"
    )
    select_parser.add_argument("--build-id")
    select_parser.add_argument("--allow-dirty", action="store_true")
    select_parser.set_defaults(func=_cmd_select)

    measure = sub.add_parser(
        "measure-tokenizers", help="train provisional measurement tokenizers (sizing only)"
    )
    measure.add_argument("--build-id")
    measure.set_defaults(func=_cmd_measure_tokenizers)

    count = sub.add_parser("count-tokens", help="provisional per-document token counts for corpus sizing")
    count.add_argument("--build-id")
    count.set_defaults(func=_cmd_count_tokens)

    audit = sub.add_parser("audit", help="write a Markdown sample-audit pack")
    audit.add_argument("--build-id")
    audit.add_argument("--sample-rate", type=float, default=0.002)
    audit.set_defaults(func=_cmd_audit)

    lines = sub.add_parser("repeated-lines", help="measure cross-document repeated lines per source")
    lines.add_argument("--build-id")
    lines.add_argument("--sample-rate", type=float, default=0.02)
    lines.set_defaults(func=_cmd_repeated_lines)

    calibrate = sub.add_parser("calibrate-dedup", help="empirical LSH detection curves and labelling pairs")
    calibrate.add_argument("--build-id")
    calibrate.add_argument("--settings", nargs="+", default=["14x8", "20x5", "16x7", "9x12"])
    calibrate.add_argument("--probe", default="28x4")
    calibrate.add_argument("--pairs", type=int, default=2000)
    calibrate.add_argument("--label-pairs", type=int, default=42)
    calibrate.set_defaults(func=_cmd_calibrate_dedup)

    report = sub.add_parser("report", help="process, mixture and distribution statistics")
    report.add_argument("--build-id")
    report.set_defaults(func=_cmd_report)

    verify = sub.add_parser("verify", help="independent recounts of a finished build")
    verify.add_argument("--build-id")
    verify.add_argument("--token-sample-rate", type=float, default=0.0005)
    verify.add_argument("--contamination-sample-rate", type=float, default=0.0)
    verify.add_argument("--git-clones")
    verify.set_defaults(func=_cmd_verify)

    gates_parser = sub.add_parser("gates", help="evaluate dataset freeze gates from recorded evidence")
    gates_parser.add_argument("--build-id")
    gates_parser.add_argument(
        "--record-tests", action="store_true", help="run and record the test suite first"
    )
    gates_parser.set_defaults(func=_cmd_gates)

    export = sub.add_parser("export-manifests", help="copy build records into the repository for review")
    export.add_argument("--build-id")
    export.set_defaults(func=_cmd_export_manifests)

    sources_doc = sub.add_parser(
        "sources-doc", help="render docs/sources.md from the registries and manifests"
    )
    sources_doc.add_argument("--build-id")
    sources_doc.set_defaults(func=_cmd_sources_doc)

    prompts = sub.add_parser("teacher-prompts", help="build verifiable teacher prompts from a pinned source")
    prompts.add_argument("source")
    prompts.add_argument("--limit", type=int, default=100)
    prompts.set_defaults(func=_cmd_teacher_prompts)

    teacher_run = sub.add_parser(
        "teacher-run", help="generate and verify teacher outputs (needs llama-server)"
    )
    teacher_run.add_argument("prompts")
    teacher_run.add_argument("--run-id", required=True)
    teacher_run.set_defaults(func=_cmd_teacher_run)

    args = parser.parse_args(argv)
    result: int = args.func(args)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
