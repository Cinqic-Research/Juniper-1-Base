"""Dataset freeze gates.

Each gate is evaluated only from recorded evidence (manifests, markers, reports,
the test record, the git tree). A gate without evidence is ``NOT VERIFIED``,
never ``PASS``. Gates that the current scope deliberately leaves to a later
stage (the canonical tokenizer) are ``NOT APPLICABLE`` with the reason.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tomllib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from j1data.build import expected_build_identity
from j1data.hashing import sha256_file, sha256_json, unit_interval
from j1data.paths import DataLayout, safe_child
from j1data.registry import REPO_ROOT, load_registry
from j1data.successor import successor_problem

PASS, FAIL, NOT_VERIFIED, NOT_APPLICABLE = "PASS", "FAIL", "NOT VERIFIED", "NOT APPLICABLE"
REQUIRED_OWNER_DECISIONS = {
    "fineweb-edu-training-use": {"fineweb-edu-dedup"},
    "finemath-training-use": {"finemath-4plus"},
    "cc-by-sa-training-and-weights-treatment": {
        "wikimedia",
        "stackexchange",
        "arxiv-papers",
        "libretexts",
        "pressbooks",
        "oercommons",
    },
    "stackv2-license-and-provenance": {"stackv2-edu"},
}


def _load(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _git_head() -> str | None:
    result = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], capture_output=True, text=True
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _test_inputs_sha256() -> str:
    """Fingerprint the code, config, schemas, and tests exercised by the local suite."""
    roots = ["src/j1data", "tests", "configs", "schemas", ".github"]
    files = [REPO_ROOT / "pyproject.toml"]
    for root in roots:
        directory = REPO_ROOT / root
        if directory.exists():
            files.extend(path for path in directory.rglob("*") if path.is_file())
    digests = {
        path.relative_to(REPO_ROOT).as_posix(): sha256_file(path)
        for path in sorted(set(files))
        if "__pycache__" not in path.parts and path.suffix != ".pyc"
    }
    return sha256_json(digests)


def _current_test_record(layout: DataLayout, build_id: str) -> dict[str, Any] | None:
    build = _load(layout.work(build_id) / "build.json")
    if not build:
        return None
    expected = {
        "tested_commit": _git_head(),
        "test_inputs_sha256": _test_inputs_sha256(),
        "build_identity": build.get("identity"),
    }
    report_dir = layout.reports(build_id)
    candidates = sorted((report_dir / "test-runs").glob("*.json"), reverse=True)
    # Read legacy evidence too, but it will only qualify when it has the new
    # commit, input, and build bindings.
    legacy = report_dir / "tests.json"
    if legacy.exists():
        candidates.append(legacy)
    for path in candidates:
        try:
            record = _load(path)
        except (json.JSONDecodeError, OSError):
            continue
        if isinstance(record, dict) and all(record.get(key) == value for key, value in expected.items()):
            return record
    return None


def _test_record_passes(record: dict[str, Any] | None) -> bool:
    return bool(
        record
        and record.get("returncode") == 0
        and record.get("failed") == 0
        and record.get("passed", 0) > 0
        and record.get("end_to_end_returncode") == 0
        and record.get("end_to_end_passed") is True
    )


def _owner_decision_status(source_ids: set[str]) -> tuple[str, str]:
    path = REPO_ROOT / "configs" / "owner_decisions.toml"
    if not path.exists():
        return NOT_VERIFIED, "configs/owner_decisions.toml is absent"
    with path.open("rb") as handle:
        document = tomllib.load(handle)
    decisions = document.get("decision", [])
    applicable = [
        item
        for item in decisions
        if source_ids.intersection(REQUIRED_OWNER_DECISIONS.get(item.get("decision_id"), set()))
        or source_ids.intersection(item.get("sources", []))
    ]
    counts: dict[str, int] = {}
    for item in applicable:
        decision_id = item.get("decision_id")
        if isinstance(decision_id, str):
            counts[decision_id] = counts.get(decision_id, 0) + 1
    missing = [
        decision_id
        for decision_id, governed_sources in REQUIRED_OWNER_DECISIONS.items()
        if source_ids.intersection(governed_sources) and counts.get(decision_id, 0) == 0
    ]
    duplicates = sorted(decision_id for decision_id, count in counts.items() if count > 1)
    wrong_scope = [
        decision_id
        for decision_id, governed_sources in REQUIRED_OWNER_DECISIONS.items()
        if source_ids.intersection(governed_sources)
        and counts.get(decision_id, 0) == 1
        and not governed_sources.issubset(
            set(
                next(item.get("sources", []) for item in applicable if item.get("decision_id") == decision_id)
            )
        )
    ]
    if missing or duplicates or wrong_scope:
        details = []
        if missing:
            details.append(f"missing required decisions: {missing}")
        if duplicates:
            details.append(f"duplicate required decisions: {duplicates}")
        if wrong_scope:
            details.append(f"required decisions omit governed sources: {wrong_scope}")
        return NOT_VERIFIED, "; ".join(details)
    pending = [
        item["decision_id"]
        for item in applicable
        if item.get("status") == "PENDING" and source_ids.intersection(item.get("sources", []))
    ]
    rejected = [
        item["decision_id"]
        for item in applicable
        if item.get("status") == "REJECTED" and source_ids.intersection(item.get("sources", []))
    ]
    if rejected:
        return FAIL, f"owner rejected training use for decisions: {rejected}"
    if pending:
        return NOT_VERIFIED, f"owner decisions pending: {pending}"
    invalid = [
        item.get("decision_id", "<missing id>")
        for item in applicable
        if item.get("status") not in {"APPROVED", "NOT_APPLICABLE", "PENDING", "REJECTED"}
    ]
    if invalid:
        return NOT_VERIFIED, f"owner decision records have unknown status: {invalid}"
    return PASS, "no pending or rejected owner decisions apply to accepted sources"


def _raw_integrity_status(
    layout: DataLayout, build: dict[str, Any] | None, accepted_sources: set[str]
) -> tuple[str, str]:
    from j1data.acquire import manifest_identity

    if not accepted_sources:
        return NOT_VERIFIED, "accepted source set is empty"
    if not isinstance(build, dict) or not isinstance(build.get("build_id"), str):
        return NOT_VERIFIED, "build record is absent"
    record = _load(layout.reports(build["build_id"]) / "source_integrity_preflight.json")
    if not isinstance(record, dict) or record.get("build_identity") != build.get("identity"):
        return NOT_VERIFIED, "no source-integrity preflight is bound to the current build"
    sources = record.get("sources")
    identities = build.get("inputs", {}).get("acquisition_manifests", {})
    registry = load_registry()
    problems = []
    for source_id in sorted(accepted_sources):
        info = sources.get(source_id) if isinstance(sources, dict) else None
        source = registry.get(source_id)
        manifest = _load(layout.acquisition_manifest(source_id))
        expected_identity = identities.get(source_id) if isinstance(identities, dict) else None
        if not isinstance(info, dict) or source is None or not isinstance(manifest, dict):
            problems.append(f"{source_id}: missing source preflight or manifest")
            continue
        try:
            current_manifest_identity = manifest_identity(source, manifest)
        except (KeyError, TypeError, ValueError) as error:
            problems.append(f"{source_id}: acquisition manifest is invalid: {error}")
            continue
        if expected_identity != current_manifest_identity:
            problems.append(f"{source_id}: current manifest differs from the build source identity")
        if info.get("manifest_identity") != current_manifest_identity:
            problems.append(f"{source_id}: preflight does not match the build source identity")
        if info.get("files") != manifest.get("selected_file_count"):
            problems.append(f"{source_id}: verified file count differs from the manifest")
        if info.get("bytes") != manifest.get("selected_bytes"):
            problems.append(f"{source_id}: verified byte count differs from the manifest")
    if problems:
        return NOT_VERIFIED, "; ".join(problems[:8])
    return PASS, f"all {len(accepted_sources)} accepted sources passed byte-level preflight"


def _expected_unit_inventory(layout: DataLayout, build_id: str) -> tuple[set[tuple[str, str]] | None, str]:
    """Recompute release unit scope from the pinned source manifests and configs."""
    from j1data import adapters
    from j1data.build import load_pipeline_config

    config = load_pipeline_config()
    if build_id != config["build"]["id"]:
        return None, "build is exploratory; release-scope inventory does not apply"
    scope_path = REPO_ROOT / "configs" / "build_scope.toml"
    with scope_path.open("rb") as handle:
        scope = tomllib.load(handle).get("sources", {})
    registry = load_registry()
    expected: set[tuple[str, str]] = set()
    for source_id, source in registry.items():
        if not source.admissible:
            continue
        manifest = _load(layout.acquisition_manifest(source_id))
        if not isinstance(manifest, dict):
            return None, f"{source_id}: acquisition manifest is missing"
        if source.adapter == "cinqic_git":
            units = [adapters.UnitSpec(source_id, "all", "")]
        else:
            units = adapters.make_adapter(
                source, layout.raw_dir(source_id, source.revision), manifest
            ).units()
        source_scope = scope.get(source_id)
        if source_scope is not None:
            if "include" in source_scope:
                allowed = set(source_scope["include"])
                present = {unit.unit_id for unit in units}
                if not allowed.issubset(present):
                    absent = sorted(allowed - present)
                    return None, f"{source_id}: scoped units are absent from pinned manifest: {absent[:8]}"
                units = [unit for unit in units if unit.unit_id in allowed]
            elif "count" in source_scope:
                if source_scope["count"] > len(units):
                    return None, (
                        f"{source_id}: scope requests {source_scope['count']} units, only {len(units)} exist"
                    )
                ranked = sorted(
                    units,
                    key=lambda unit: unit_interval(source_scope["namespace"], unit.unit_id),
                )
                allowed = {unit.unit_id for unit in ranked[: source_scope["count"]]}
                units = [unit for unit in units if unit.unit_id in allowed]
        expected.update((unit.source_id, unit.unit_id) for unit in units)
    return expected, "expected release unit inventory reconstructed from source manifests and scope"


def _verification_matches_current(layout: DataLayout, build_id: str, build: dict[str, Any] | None) -> bool:
    if not isinstance(build, dict) or build.get("build_id") != build_id:
        return False
    accepted_manifest = layout.accepted(build_id) / "manifest.json"
    verification = _load(layout.reports(build_id) / "verification.json")
    return bool(
        isinstance(verification, dict)
        and verification.get("schema") == "j1.verification.v2"
        and verification.get("build_id") == build_id
        and verification.get("build_identity") == build.get("identity")
        and accepted_manifest.exists()
        and verification.get("accepted_manifest_sha256") == sha256_file(accepted_manifest)
    )


def _marker_inventory_matches(
    expected: set[tuple[str, str]] | None,
    units: set[tuple[str, str]],
    markers: list[dict[str, Any]],
    errors: list[str],
    identity: str | None,
    exclusion_identity: str | None = None,
) -> bool:
    return bool(
        expected is not None
        and identity
        and units == expected
        and len(units) == len(markers)
        and not errors
        and all(marker.get("build_identity") == identity for marker in markers)
        and (
            exclusion_identity is None
            or all(marker.get("exclusion_index_identity") == exclusion_identity for marker in markers)
        )
    )


def _expected_finalize_units(
    layout: DataLayout, build_id: str, expected: set[tuple[str, str]] | None
) -> set[tuple[str, str]] | None:
    """Finalize work units as ``dedup.collect_units`` derives them: a process marker with
    committed segments (process v2) yields ``<unit>~part-NNNNNN`` per segment, and a marker
    without segments yields the logical unit itself."""
    if expected is None:
        return None
    out: set[tuple[str, str]] = set()
    seen: set[tuple[str, str]] = set()
    for path in sorted((layout.work(build_id) / "units").glob("*/*.json")):
        marker = _load(path)
        if not isinstance(marker, dict):
            return None
        source_id, unit_id = marker.get("source_id"), marker.get("unit_id")
        if not isinstance(source_id, str) or not isinstance(unit_id, str):
            return None
        if (source_id, unit_id) not in expected:
            continue
        seen.add((source_id, unit_id))
        parts = marker.get("segments")
        if isinstance(parts, list):
            out.update((source_id, f"{unit_id}~part-{int(part['index']):06d}") for part in parts)
        else:
            out.add((source_id, unit_id))
    return out if seen == expected else None


def _accounting_passes(accounting: Any) -> bool:
    required = {
        "raw_equals_stage_total",
        "candidates_equal_candidate_rows",
        "dedup_saw_all_candidates",
        "candidates_equal_removed_plus_unselected_plus_leaked_plus_accepted",
        "finalize_dedup_agree",
    }
    return bool(
        isinstance(accounting, dict)
        and required.issubset(accounting)
        and all(accounting[key] is True for key in required)
    )


def _process_checkpoint_status(
    layout: DataLayout, build_id: str, build: dict[str, Any] | None, markers: list[dict[str, Any]]
) -> tuple[str, str]:
    from j1data.process import VERSION
    from j1data.shards import verify_files

    if not markers or not isinstance(build, dict):
        return NOT_VERIFIED, "process completion markers or build identity are absent"
    work = layout.work(build_id)
    problems: list[str] = []
    unverified: list[str] = []
    for marker in markers:
        source_id, unit_id = marker.get("source_id"), marker.get("unit_id")
        if not isinstance(source_id, str) or not isinstance(unit_id, str):
            problems.append("process marker has no source/unit identity")
            continue
        marker_name = f"{source_id}/{unit_id}"
        if marker.get("schema") != "j1.unit-marker.v2" or marker.get("process_version") != VERSION:
            problems.append(f"{marker_name}: checkpoint-qualified process marker is absent")
            continue
        progress_path = work / "progress" / "process" / source_id / f"{sha256_json(unit_id)[:24]}.json"
        progress = _load(progress_path)
        if not isinstance(progress, dict):
            unverified.append(f"{marker_name}: progress journal is missing or unreadable")
            continue
        expected_progress = {
            "schema": "j1.process-progress.v1",
            "process_version": VERSION,
            "build_identity": build.get("identity"),
            "source_id": source_id,
            "source_revision": marker.get("source_revision"),
            "unit_id": unit_id,
            "raw_file": marker.get("raw_file"),
            "raw_input_identity": marker.get("raw_input_identity"),
            "exclusion_index_identity": marker.get("exclusion_index_identity"),
        }
        if any(progress.get(key) != value for key, value in expected_progress.items()):
            problems.append(f"{marker_name}: progress journal identity does not match marker/build")
            continue
        segments = marker.get("segments")
        if not isinstance(segments, list) or not segments or progress.get("segments") != segments:
            problems.append(f"{marker_name}: progress journal segments do not match completion marker")
            continue
        row_range = marker.get("row_range")
        if not isinstance(row_range, list) or len(row_range) != 2 or not isinstance(row_range[0], int):
            problems.append(f"{marker_name}: malformed source row range")
            continue
        cursor = row_range[0]
        segment_outputs: list[dict[str, Any]] = []
        segment_docs = 0
        segment_candidates = 0
        for index, segment in enumerate(segments):
            if not isinstance(segment, dict):
                problems.append(f"{marker_name}: malformed segment {index}")
                break
            seconds = segment.get("seconds")
            docs = segment.get("docs_in")
            outputs = segment.get("outputs")
            if (
                segment.get("index") != index
                or segment.get("cursor_start") != cursor
                or not isinstance(docs, int)
                or docs < 0
                or not isinstance(segment.get("cursor_end"), int)
                or segment["cursor_end"] - cursor != docs
                or not isinstance(seconds, (int, float))
                or seconds < 0
                or seconds >= 900
                or not isinstance(outputs, list)
                or len(outputs) != 3
                or not isinstance(outputs[0], dict)
                or not isinstance(outputs[1], dict)
                or not isinstance(outputs[2], dict)
                or not isinstance(segment.get("stages"), dict)
                or not isinstance(segment.get("accepted_ids_sha256"), str)
                or len(segment["accepted_ids_sha256"]) != 64
                or not isinstance(segment.get("rejected_ids_sha256"), str)
                or len(segment["rejected_ids_sha256"]) != 64
                or not isinstance(segment.get("accepted_ids_sample"), list)
                or len(segment["accepted_ids_sample"]) > 16
                or not all(isinstance(item, str) for item in segment["accepted_ids_sample"])
                or not isinstance(segment.get("rejected_ids_sample"), list)
                or len(segment["rejected_ids_sample"]) > 16
                or not all(isinstance(item, str) for item in segment["rejected_ids_sample"])
            ):
                problems.append(f"{marker_name}: segment {index} violates the checkpoint contract")
                break
            cursor = segment["cursor_end"]
            segment_docs += docs
            candidates = segment["stages"].get("candidate", 0)
            if (
                not isinstance(candidates, int)
                or outputs[0].get("rows") != candidates
                or outputs[2].get("rows") != candidates
                or outputs[1].get("rows") != docs - candidates
            ):
                problems.append(
                    f"{marker_name}: segment {index} row accounting is inconsistent "
                    f"(docs={docs}, candidates={candidates}, outputs={outputs})"
                )
                break
            segment_candidates += candidates
            segment_outputs.extend(outputs)
            for output_index, (output_index_name, recorded_hash, recorded_sample) in enumerate(
                (
                    ("accepted", segment.get("accepted_ids_sha256"), segment.get("accepted_ids_sample")),
                    ("rejected", segment.get("rejected_ids_sha256"), segment.get("rejected_ids_sample")),
                )
            ):
                try:
                    output_entry = outputs[output_index]
                    output_path = safe_child(work, output_entry["file"])
                    actual_hash = hashlib.sha256()
                    actual_sample: list[str] = []
                    rows = 0
                    for batch in pq.ParquetFile(output_path).iter_batches(
                        columns=["doc_id"], batch_size=8192
                    ):
                        for doc_id in batch.column(0).to_pylist():
                            if not isinstance(doc_id, str):
                                raise ValueError("doc_id is not text")
                            actual_hash.update((doc_id + "\n").encode("utf-8"))
                            if len(actual_sample) < 16:
                                actual_sample.append(doc_id)
                            rows += 1
                except (OSError, ValueError, KeyError, TypeError, pa.ArrowException) as error:
                    problems.append(
                        f"{marker_name}: cannot independently read {output_index_name} IDs: {error}"
                    )
                    continue
                if actual_hash.hexdigest() != recorded_hash or actual_sample != recorded_sample:
                    problems.append(
                        f"{marker_name}: segment {index} {output_index_name} ID digest/sample differs "
                        "from Parquet rows"
                    )
                if rows != output_entry.get("rows"):
                    problems.append(
                        f"{marker_name}: segment {index} {output_index_name} ID rows "
                        "differ from output manifest"
                    )
        if (
            segment_docs != marker.get("docs_in")
            or segment_candidates != marker.get("candidate_rows")
            or progress.get("next_cursor") != cursor
        ):
            problems.append(f"{marker_name}: segment accounting or journal cursor is inconsistent")
        if segment_outputs != marker.get("outputs"):
            problems.append(f"{marker_name}: completion output inventory differs from segments")
        if verify_files(work, segment_outputs):
            problems.append(f"{marker_name}: one or more committed segment outputs failed re-hashing")
    if problems:
        return FAIL, "; ".join(problems[:8])
    if unverified:
        return NOT_VERIFIED, "; ".join(unverified[:8])
    return (
        PASS,
        f"{len(markers)} logical units have hash-verified journals, independently checked ID digests, "
        "and sub-15-minute immutable segments",
    )


def _v03_checkpoint_gates() -> list[dict[str, str]]:
    """Requirements beyond processing stay explicitly blocked until evidence exists."""
    return [
        {
            "gate": "v0.3 transactional recovery across every CPU/data stage",
            "status": NOT_VERIFIED,
            "evidence": (
                "process recovery is implemented and fixture-tested, but deduplication, selection, "
                "finalization, tokenizer sampling/probes, and token counting lack complete "
                "transactional recovery qualification at the required cadence"
            ),
        },
        {
            "gate": "v0.3 canonical A100 training checkpoint and restore qualification",
            "status": NOT_VERIFIED,
            "evidence": (
                "this repository has no qualified canonical trainer or A100 evidence for complete "
                "training state, independent durable replicas, fresh-process restore equivalence, "
                "rollback, and failure injection"
            ),
        },
    ]


def evaluate(layout: DataLayout, build_id: str) -> list[dict[str, str]]:
    gates: list[dict[str, str]] = []

    def gate(name: str, status: str, evidence: str) -> None:
        gates.append({"gate": name, "status": status, "evidence": evidence})

    registry = load_registry()
    admissible = {source_id for source_id, source in registry.items() if source.admissible}
    acquisitions = {s: _load(layout.acquisition_manifest(s)) for s in admissible}
    missing = [s for s, m in acquisitions.items() if m is None]
    gate(
        "source registry complete",
        PASS if not missing else FAIL,
        f"{len(admissible)} admissible sources validated against the schema; "
        f"missing acquisition manifests: {missing}",
    )
    manifest = _load(layout.accepted(build_id) / "manifest.json")
    accepted_manifest_path = layout.accepted(build_id) / "manifest.json"
    accepted_manifest_sha256 = (
        sha256_file(accepted_manifest_path) if accepted_manifest_path.is_file() else None
    )
    if isinstance(manifest, dict) and "successor_of" in manifest:
        problem = successor_problem(manifest)
        gate(
            "successor build identity derived from its lineage",
            FAIL if problem else PASS,
            problem
            or f"successor of {manifest['successor_of'].get('build_id')}; identity "
            f"{manifest.get('build_identity')} (parent-bound records cannot match)",
        )
    shards = manifest.get("shards", []) if isinstance(manifest, dict) else []
    accepted_sources: set[str] = {
        shard["source_id"]
        for shard in shards
        if isinstance(shard, dict) and isinstance(shard.get("source_id"), str)
    }
    marker_paths = sorted((layout.work(build_id) / "units").glob("*/*.json"))
    markers: list[dict[str, Any]] = []
    marker_errors: list[str] = []
    for path in marker_paths:
        try:
            marker = _load(path)
        except (OSError, json.JSONDecodeError):
            marker = None
        if not isinstance(marker, dict):
            marker_errors.append(path.name)
        else:
            markers.append(marker)
    marker_units: set[tuple[str, str]] = set()
    for marker in markers:
        source_id, unit_id = marker.get("source_id"), marker.get("unit_id")
        if isinstance(source_id, str) and isinstance(unit_id, str):
            marker_units.add((source_id, unit_id))
    encountered = accepted_sources | {source for source, _ in marker_units}
    inadmissible = [
        source for source in sorted(encountered) if source not in registry or not registry[source].admissible
    ]
    gate(
        "no rejected or review-required sources in build outputs",
        FAIL if inadmissible else PASS if encountered else NOT_VERIFIED,
        f"sources found in accepted manifest or process markers: {sorted(encountered)}; "
        f"inadmissible or unknown sources: {inadmissible}",
    )
    approval_status, approval_evidence = _owner_decision_status(accepted_sources)
    gate("training-use owner decisions", approval_status, approval_evidence)
    build = _load(layout.work(build_id) / "build.json")
    raw_status, raw_evidence = _raw_integrity_status(layout, build, accepted_sources)
    gate(
        "raw source identity recorded",
        raw_status if not missing else NOT_VERIFIED,
        raw_evidence if not missing else f"acquisition manifests missing: {missing}",
    )

    try:
        expected_identity = expected_build_identity(layout)
    except (OSError, SystemExit, ValueError, KeyError) as error:
        expected_identity = None
        identity_detail = f"current inputs could not be resolved: {error}"
    else:
        identity_detail = f"current input identity {expected_identity}"
    build_matches = bool(build and expected_identity and build.get("identity") == expected_identity)
    gate(
        "recorded build matches current pipeline inputs",
        PASS if build_matches else NOT_VERIFIED,
        f"recorded identity {build.get('identity') if build else 'absent'}; {identity_detail}",
    )

    tests = _current_test_record(layout, build_id)
    tests_ok = _test_record_passes(tests)
    gate(
        "secrets/PII filter validated",
        PASS if tests_ok else NOT_VERIFIED,
        "planted-credential and PII tests passed on current commit and inputs"
        if tests_ok
        else "no passing test record bound to the current commit, test inputs, and build identity",
    )

    try:
        expected_units, inventory_evidence = _expected_unit_inventory(layout, build_id)
    except (OSError, ValueError, KeyError, TypeError, SystemExit) as error:
        expected_units, inventory_evidence = None, f"release scope could not be reconstructed: {error}"
    exclusion_identity = (build or {}).get("inputs", {}).get("exclusion_index")
    scope_complete = _marker_inventory_matches(
        expected_units,
        marker_units,
        markers,
        marker_errors,
        (build or {}).get("identity"),
        exclusion_identity,
    )
    checkpoint_status, checkpoint_evidence = _process_checkpoint_status(layout, build_id, build, markers)
    gate("transactional process checkpoint recovery", checkpoint_status, checkpoint_evidence)
    gates.extend(_v03_checkpoint_gates())
    dedup = _load(layout.work(build_id) / "dedup" / "summary.json")
    try:
        candidates = sum(
            int(marker["stages"]["candidate"]) for marker in markers if isinstance(marker.get("stages"), dict)
        )
    except (KeyError, TypeError, ValueError):
        candidates = 0
        marker_errors.append("malformed-stage-count")
        scope_complete = False
    complete = bool(
        scope_complete
        and build_matches
        and candidates > 0
        and isinstance(dedup, dict)
        and dedup.get("build_identity") == (build or {}).get("identity")
        and dedup.get("candidates") == candidates
    )
    gate(
        "global exact deduplication complete",
        PASS if complete else NOT_VERIFIED,
        f"dedup saw {dedup.get('candidates', 0) if isinstance(dedup, dict) else 0} "
        f"of {candidates} candidates; "
        f"scope complete={scope_complete}; {inventory_evidence}",
    )
    gate(
        "near deduplication complete",
        PASS if complete and isinstance(dedup, dict) and dedup.get("lsh") else NOT_VERIFIED,
        f"MinHash-LSH {dedup.get('lsh') if isinstance(dedup, dict) else 'not run'}",
    )
    gate(
        "benchmark contamination scan complete",
        PASS if scope_complete and build_matches and exclusion_identity else NOT_VERIFIED,
        f"{len(markers)} of "
        f"{len(expected_units) if expected_units is not None else 'unknown'} expected units scanned; "
        f"invalid markers={marker_errors}; exclusion index "
        f"{str(exclusion_identity)[:16] if exclusion_identity else '-'}",
    )

    verification = _load(layout.reports(build_id) / "verification.json")
    verification_fresh = _verification_matches_current(layout, build_id, build)
    accepted_marker_paths = sorted((layout.accepted(build_id) / "_units").glob("*/*.json"))
    accepted_markers: list[dict[str, Any]] = []
    accepted_marker_errors: list[str] = []
    for path in accepted_marker_paths:
        marker = _load(path)
        if isinstance(marker, dict):
            accepted_markers.append(marker)
        else:
            accepted_marker_errors.append(path.name)
    accepted_marker_units: set[tuple[str, str]] = set()
    for marker in accepted_markers:
        source_id, unit_id = marker.get("source_id"), marker.get("unit_id")
        if isinstance(source_id, str) and isinstance(unit_id, str):
            accepted_marker_units.add((source_id, unit_id))
    expected_finalize_units = _expected_finalize_units(layout, build_id, expected_units)
    accepted_inventory_complete = (
        _marker_inventory_matches(
            expected_finalize_units,
            accepted_marker_units,
            accepted_markers,
            accepted_marker_errors,
            (build or {}).get("identity"),
        )
        and isinstance(manifest, dict)
        and manifest.get("build_identity") == (build or {}).get("identity")
    )
    gate(
        "accepted unit marker inventory complete",
        PASS if accepted_inventory_complete else NOT_VERIFIED,
        f"{len(accepted_markers)} of "
        f"{len(expected_finalize_units) if expected_finalize_units is not None else 'unknown'} expected "
        f"finalize work-unit markers "
        f"({len(expected_units) if expected_units is not None else 'unknown'} logical units); "
        f"invalid markers={accepted_marker_errors}",
    )
    uniqueness = verification.get("uniqueness_and_splits") if isinstance(verification, dict) else None
    splits_ok = bool(
        accepted_inventory_complete
        and scope_complete
        and verification_fresh
        and isinstance(uniqueness, dict)
        and uniqueness.get("problems") == []
        and all(
            uniqueness.get(key) == 0
            for key in (
                "duplicate_doc_ids",
                "duplicate_text_hashes",
                "duplicate_signature_hashes",
                "groups_spanning_splits",
                "split_mismatches",
            )
        )
        and isinstance(manifest, dict)
    )
    gate(
        "split leakage checks complete",
        PASS if splits_ok else NOT_VERIFIED,
        "held-out 13-gram leakage removal in finalize; independent split recomputation in verify",
    )
    stats = _load(layout.reports(build_id) / "accepted_statistics.json")
    gate(
        "source/domain mixture measured",
        PASS
        if isinstance(stats, dict)
        and stats.get("build_identity") == (build or {}).get("identity")
        and isinstance(manifest, dict)
        and stats.get("accepted_manifest_sha256") == accepted_manifest_sha256
        else NOT_VERIFIED,
        "reports/accepted_statistics.json must name the current build identity",
    )
    token_counts = verification.get("provisional_token_counts") if isinstance(verification, dict) else None
    tokens_ok = bool(
        accepted_inventory_complete
        and scope_complete
        and verification_fresh
        and isinstance(token_counts, dict)
        and token_counts.get("problems") == []
        and token_counts.get("reencoded_sample_documents", 0) > 0
    )
    gate(
        "token count independently verified (provisional measurement tokenizers)",
        PASS if tokens_ok else NOT_VERIFIED,
        "unbatched re-encoding of a keyed-hash sample against count tables",
    )
    gate(
        "canonical tokenizer identity recorded",
        NOT_APPLICABLE,
        "out of scope by owner decision: the Juniper 1 tokenizer is designed and frozen in a later stage",
    )
    shard_report = verification.get("shards") if isinstance(verification, dict) else None
    shards_ok = bool(
        accepted_inventory_complete
        and scope_complete
        and verification_fresh
        and isinstance(shard_report, dict)
        and shard_report.get("problems") == []
    )
    gate(
        "accepted shard hashes complete",
        PASS if shards_ok else NOT_VERIFIED,
        f"{shard_report.get('shards', 0) if isinstance(shard_report, dict) else 0} shards re-hashed",
    )
    accounting = verification.get("accounting") if isinstance(verification, dict) else None
    accounting_ok = bool(
        accepted_inventory_complete
        and scope_complete
        and verification_fresh
        and _accounting_passes(accounting)
    )
    gate(
        "document accounting closes",
        PASS if accounting_ok else NOT_VERIFIED,
        "raw = rejections + candidates; candidates = duplicates + leakage + accepted",
    )
    card = REPO_ROOT / "docs" / "data_card.md"
    card_ok = card.exists() and build is not None and build["identity"] in card.read_text()
    gate(
        "data card complete", PASS if card_ok else NOT_VERIFIED, "docs/data_card.md names the build identity"
    )
    gate(
        "known limitations documented",
        PASS if card_ok and "Known limitations" in card.read_text() else NOT_VERIFIED,
        "docs/data_card.md#known-limitations",
    )
    smoke = _test_record_passes(tests)
    gate(
        "pipeline smoke rebuild passes",
        PASS if smoke else NOT_VERIFIED,
        "tests/test_end_to_end.py exited successfully on the current commit and inputs",
    )
    status = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "status", "--porcelain"], capture_output=True, text=True
    )
    gate(
        "review branch clean",
        PASS if status.returncode == 0 and not status.stdout.strip() else FAIL,
        status.stdout.strip()[:200] or "git status clean",
    )
    return gates


def record_tests(layout: DataLayout, build_id: str) -> dict[str, Any]:
    """Run the test suite and record the outcome as gate evidence."""
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-o",
            "addopts=",
            "-q",
            "-p",
            "no:cacheprovider",
            str(REPO_ROOT / "tests"),
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    summary = result.stdout.strip().splitlines()[-1] if result.stdout.strip() else ""
    counts = {"passed": 0, "failed": 0}
    for part in summary.replace("=", "").split(","):
        words = part.split()
        if len(words) >= 2 and words[1] in ("passed", "failed", "error", "errors"):
            key = "failed" if words[1] != "passed" else "passed"
            counts[key] += int(words[0])
    end_to_end = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:cacheprovider",
            str(REPO_ROOT / "tests" / "test_end_to_end.py"),
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    build = _load(layout.work(build_id) / "build.json")
    record = {
        **counts,
        "summary": summary,
        "end_to_end_passed": end_to_end.returncode == 0,
        "returncode": result.returncode,
        "end_to_end_returncode": end_to_end.returncode,
        "tested_at_utc": datetime.now(UTC).isoformat(),
        "tested_commit": _git_head(),
        "test_inputs_sha256": _test_inputs_sha256(),
        "build_identity": build.get("identity") if build else None,
    }
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    out = layout.reports(build_id) / "test-runs" / f"{record['tested_commit'] or 'unknown'}-{stamp}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=1) + "\n")
    return record
