"""Transactional recovery of the CPU data stages (gate J1-13, docs/stage_recovery.md).

Each fault test runs the stage in a child process with the ``j1data.faults`` hook armed,
so the kill is a real SIGKILL (no ``finally`` blocks, no buffered flushes), then reruns
the stage without the fault and compares every output file byte-for-byte with an
uninterrupted run.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from j1data import tokshards, tokstage

SRC = Path(__file__).resolve().parents[1] / "src"
TESTS = Path(__file__).resolve().parent
KILLED = -9
SELECTION = {
    "selection": {"namespace": "j1.select-docs.v1"},
    "quota": [
        {"source": "fixture-a", "tokens": 2_000, "bytes_per_token": 1.0},
        {"source": "fixture-b", "all": True},
    ],
}


def child(code: str, fault: str | None = None, after: int = 1, action: str = "kill") -> int:
    env = {k: v for k, v in os.environ.items() if not k.startswith("J1_FAULT_")}
    env["PYTHONPATH"] = os.pathsep.join([str(SRC), str(TESTS)])
    if fault:
        env.update(J1_FAULT_POINT=fault, J1_FAULT_AFTER=str(after), J1_FAULT_ACTION=action)
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)], env=env, capture_output=True, text=True, timeout=300
    )
    if fault is None and result.returncode:
        raise AssertionError(result.stderr[-4000:])
    return result.returncode


def tree(root: Path, exclude: tuple[str, ...] = ()) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name not in exclude and not any(part in exclude for part in path.parts)
    }


def leftovers(root: Path) -> list[str]:
    return [p.name for p in root.rglob("*") if p.name.endswith((".tmp", ".partial")) or ".tmp." in p.name]


# --------------------------------------------------------------------------- token shards


TOKSHARDS = """
    from pathlib import Path
    from j1data import tokshards
    tokshards.TOKENS_PER_FILE = 1000
    tokshards.CHECKPOINT_SECONDS = 0.0
    tokshards.build(Path({root!r}), "b1", Path({frozen!r}), Path({out!r}))
"""


@pytest.fixture(scope="module")
def tok_env(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    from test_tokshards_tokstage import _fixture

    base = tmp_path_factory.mktemp("tokrec")
    root = _fixture(base)
    tokstage.run(root, "b1", rate=1.0, sizes=[384], out_dir=base / "tok", stem_alphabet=False)
    frozen = base / "frozen"
    tokstage.freeze(base / "tok" / "j1tok-0k", frozen, "test", {})
    reference = base / "reference"
    child(TOKSHARDS.format(root=str(root), frozen=str(frozen), out=str(reference)))
    return {"root": root, "frozen": frozen, "reference": tree(reference, ("build_log.json",)), "base": base}


def _tok_code(env: dict[str, Any], out: Path) -> str:
    return TOKSHARDS.format(root=str(env["root"]), frozen=str(env["frozen"]), out=str(out))


def _assert_same_as_reference(env: dict[str, Any], out: Path) -> dict[str, Any]:
    assert tree(out, ("build_log.json",)) == env["reference"]
    assert tokshards.verify(out) == []
    assert not leftovers(out) and not list(out.glob(".*"))
    return dict(json.loads((out / "build_log.json").read_text()))


@pytest.mark.parametrize(
    ("point", "after"),
    [
        ("tokshards.spill.batch", 1),
        ("tokshards.spill.commit", 2),
        ("tokshards.place.written", 5),
        ("tokshards.place.commit", 20),
        ("tokshards.finish.file", 2),
        ("tokshards.split.done", 1),
        ("tokshards.record.written", 1),
        ("tokshards.publish.identity_dropped", 1),
        ("json.atomic.tmp", 3),  # inside a journal write: the temporary is left behind
    ],
)
def test_token_shards_resume_after_sigkill_is_byte_identical(
    tok_env: dict[str, Any], tmp_path: Path, point: str, after: int
) -> None:
    out = tmp_path / "shards"
    assert child(_tok_code(tok_env, out), point, after) == KILLED
    assert not out.exists()  # nothing is published before completion
    child(_tok_code(tok_env, out))
    log = _assert_same_as_reference(tok_env, out)
    train = log["splits"].get("train", {})
    assert not any(e.get("event") == "spill discarded" for e in train.get("recovery_events", []))
    resumed = train.get("resumed_from", {})
    # the restart continued from committed units instead of rebuilding
    if point == "tokshards.spill.commit":
        assert resumed["phase"] == "spill" and resumed["next_shard"] >= 1
    if point.startswith("tokshards.place"):
        assert resumed["phase"] == "place" and resumed["placed"] >= after - 1
    if point == "tokshards.finish.file":
        assert resumed["phase"] == "finish" and resumed["placed"] == tokshards.N_BUCKETS
    if point in ("tokshards.split.done", "tokshards.record.written", "tokshards.publish.identity_dropped"):
        assert "train" not in log["splits"] or resumed["phase"] == "done"


@pytest.mark.parametrize("target", ["published-token-file", "index-part"])
def test_token_shards_finish_rejects_damage_after_commit(
    tok_env: dict[str, Any], tmp_path: Path, target: str
) -> None:
    out = tmp_path / "shards"
    assert child(_tok_code(tok_env, out), "tokshards.finish.file", 2) == KILLED
    partial = out.with_name(out.name + ".partial")
    if target == "published-token-file":
        path = partial / "train-tokens-00000.bin"  # already renamed by the killed finish phase
    else:
        path = partial / ".spill-train" / "index" / "b00.parquet"
    assert path.is_file()
    data = bytearray(path.read_bytes())
    data[-1] ^= 0xFF
    path.write_bytes(bytes(data))
    assert child(_tok_code(tok_env, out), "never.reached") not in (0, KILLED)
    assert not out.exists()


def _killed_in_place_phase(env: dict[str, Any], out: Path, after: int = 10) -> Path:
    assert child(_tok_code(env, out), "tokshards.place.commit", after) == KILLED
    return out.with_name(out.name + ".partial")


def test_token_shards_corrupt_spill_is_detected_and_rebuilt(tok_env: dict[str, Any], tmp_path: Path) -> None:
    out = tmp_path / "shards"
    partial = _killed_in_place_phase(tok_env, out, after=3)
    journal = json.loads((partial / ".progress-train.json").read_text())
    victim = max(range(tokshards.N_BUCKETS), key=lambda b: journal["bucket_bytes"][b])
    assert str(victim) not in journal["placed"]
    path = partial / ".spill-train" / f"b{victim:02d}.bin"
    data = bytearray(path.read_bytes())
    data[-1] ^= 0xFF
    path.write_bytes(bytes(data))
    child(_tok_code(tok_env, out))
    log = _assert_same_as_reference(tok_env, out)
    assert any(e.get("event") == "spill discarded" for e in log["splits"]["train"]["recovery_events"])


def test_token_shards_corrupt_placed_region_is_rewritten(tok_env: dict[str, Any], tmp_path: Path) -> None:
    out = tmp_path / "shards"
    partial = _killed_in_place_phase(tok_env, out)
    path = partial / "train-tokens-00000.bin.partial"
    data = bytearray(path.read_bytes())
    data[0] ^= 0xFF
    path.write_bytes(bytes(data))
    child(_tok_code(tok_env, out))
    log = _assert_same_as_reference(tok_env, out)
    events = log["splits"]["train"]["recovery_events"]
    assert any(e.get("event") == "placed bucket failed verification" for e in events)


def test_token_shards_missing_journal_restarts_split(tok_env: dict[str, Any], tmp_path: Path) -> None:
    out = tmp_path / "shards"
    partial = _killed_in_place_phase(tok_env, out)
    (partial / ".progress-train.json").unlink()
    child(_tok_code(tok_env, out))
    _assert_same_as_reference(tok_env, out)


def test_token_shards_disk_full_fails_closed_then_resumes(tok_env: dict[str, Any], tmp_path: Path) -> None:
    out = tmp_path / "shards"
    assert child(_tok_code(tok_env, out), "tokshards.spill.commit", 2, action="enospc") == 1
    assert not out.exists()
    child(_tok_code(tok_env, out))
    _assert_same_as_reference(tok_env, out)


def test_token_shards_refuse_foreign_partial_and_replay(tok_env: dict[str, Any], tmp_path: Path) -> None:
    out = tmp_path / "shards"
    partial = _killed_in_place_phase(tok_env, out)
    identity = json.loads((partial / ".identity.json").read_text())
    (partial / ".identity.json").write_text(json.dumps({**identity, "n_buckets": 1}))
    with pytest.raises(SystemExit, match="different inputs"):
        tokshards.build(tok_env["root"], "b1", tok_env["frozen"], out)
    shutil.rmtree(partial)
    child(_tok_code(tok_env, out))
    before = tree(out)
    with pytest.raises(SystemExit, match="immutable"):  # replay never rewrites a published build
        tokshards.build(tok_env["root"], "b1", tok_env["frozen"], out)
    assert tree(out) == before


# --------------------------------------------------------------------------- tokenizer candidates


def test_tokenizer_candidate_reuse_requires_matching_record(tok_env: dict[str, Any], tmp_path: Path) -> None:
    out = tmp_path / "cands"
    code = f"""
        from pathlib import Path
        from j1data import tokstage
        tokstage.run(Path({str(tok_env["root"])!r}), "b1", 1.0, [384], Path({str(out)!r}), False)
    """
    assert child(code, "tokstage.candidate.tokenizer", 1) == KILLED
    assert not (out / "j1tok-0k" / "eval.json").exists()  # an unevaluated tokenizer is never reused
    child(code)
    tok_path = out / "j1tok-0k" / "tokenizer.json"
    first = tok_path.read_bytes()
    stamp = (out / "j1tok-0k" / "eval.json").stat().st_mtime_ns
    child(code)  # replay: verified candidates are reused, not retrained
    assert (out / "j1tok-0k" / "eval.json").stat().st_mtime_ns == stamp
    tok_path.write_text(tok_path.read_text() + " ")
    child(code)  # a tokenizer that no longer matches its record is retrained
    assert (out / "j1tok-0k" / "eval.json").stat().st_mtime_ns != stamp
    assert tok_path.read_bytes() == first  # training is deterministic
    assert not leftovers(out)


# ------------------------------------------------------------- dedup, selection, finalize, counts


PIPELINE = """
    import json
    from pathlib import Path
    from j1data import dedup, finalize, select, splits, tokenization
    from j1data.minhash import LSHConfig
    from j1data.paths import DataLayout
    from j1data.token_counts import count_tokens
    from tokenizers import Tokenizer
    root = Path({root!r})
    layout = DataLayout(root)
    priorities = {{"fixture-a": 1, "fixture-b": 2}}
    work = layout.work("fixture-build")
    dedup.CHECKPOINT_SECONDS = 0.0
    stage = {stage!r}
    if stage == "dedup":
        dedup.run(layout, "fixture-build", priorities, LSHConfig(14, 8))
    refs = dedup.collect_units(layout, "fixture-build", priorities)
    if stage == "select":
        select.run(refs, work / "dedup" / "keep", work / "selection", json.loads({selection!r}),
                   build_identity="fixture-identity")
    if stage == "finalize":
        final = finalize.FinalizeSettings(
            layout, "fixture-build", splits.SplitConfig("j1.split.v1", 0.2, 0.0), 0.5, "fixture-identity"
        )
        finalize.run(final, refs, workers=1, selection_dir=work / "selection")
    if stage == "count":
        manifest = json.loads((layout.accepted("fixture-build") / "manifest.json").read_text())
        tok = Tokenizer.from_file(str(root / "tiny-tokenizer.json"))
        count_tokens(layout.accepted("fixture-build"), layout.token_counts("fixture-build"), manifest,
                     {{"tiny": tok}}, {{"tiny": tokenization.identity(tok)}})
"""

STAGE_OUTPUT = {
    "dedup": ("work/fixture-build/dedup", ("recovery_log.json", "state", "code_signatures")),
    "select": ("work/fixture-build/selection", ()),
    "finalize": ("accepted/fixture-build", ()),
    "count": ("token_counts/fixture-build", ()),
}


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    from test_end_to_end import run_build

    base = tmp_path_factory.mktemp("piperec")
    result = run_build(base / "root", selection=SELECTION)
    result["tokenizer"].save(str(base / "root" / "tiny-tokenizer.json"))
    references = {}
    for stage, (relative, exclude) in STAGE_OUTPUT.items():
        references[stage] = tree(base / "root" / relative, exclude)
    return {"root": base / "root", "references": references}


def _stage_copy(pipeline: dict[str, Any], tmp_path: Path, stage: str) -> Path:
    """A copy of the built fixture with ``stage`` and every later stage output removed."""
    root = tmp_path / "root"
    shutil.copytree(pipeline["root"], root)
    order = list(STAGE_OUTPUT)
    for later in order[order.index(stage) :]:
        relative, _ = STAGE_OUTPUT[later]
        target = root / relative
        if later == "dedup":
            for name in ("keep", "state", "removed.parquet", "summary.json", "recovery_log.json"):
                path = target / name
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink(missing_ok=True)
        else:
            shutil.rmtree(target, ignore_errors=True)
    return root


def _stage_code(root: Path, stage: str) -> str:
    return PIPELINE.format(root=str(root), stage=stage, selection=json.dumps(SELECTION))


def _assert_stage_matches(pipeline: dict[str, Any], root: Path, stage: str) -> None:
    relative, exclude = STAGE_OUTPUT[stage]
    assert tree(root / relative, exclude) == pipeline["references"][stage]
    assert not leftovers(root / relative)


@pytest.mark.parametrize(
    ("stage", "point", "after", "code"),
    [
        ("dedup", "dedup.state.arrays", 2, KILLED),
        ("dedup", "dedup.state.commit", 1, KILLED),
        ("dedup", "dedup.near.band", 5, KILLED),
        ("dedup", "dedup.state.commit", 9, KILLED),
        ("dedup", "dedup.output.keep", 1, KILLED),
        ("dedup", "dedup.summary.written", 1, KILLED),
        ("dedup", "dedup.state.arrays", 3, "enospc"),
        ("select", "select.mask", 1, KILLED),
        ("select", "select.source.commit", 1, KILLED),
        ("finalize", "finalize.unit.written", 1, 1),  # worker killed: the pool fails fast
        ("finalize", "finalize.parent.unit_done", 1, KILLED),
        ("count", "count.batch", 1, KILLED),
        ("count", "count.shard.written", 1, KILLED),
        ("count", "count.shard.commit", 1, KILLED),
    ],
)
def test_stage_resume_after_fault_is_byte_identical(
    pipeline: dict[str, Any], tmp_path: Path, stage: str, point: str, after: int, code: int | str
) -> None:
    root = _stage_copy(pipeline, tmp_path, stage)
    action = "enospc" if code == "enospc" else "kill"
    returncode = child(_stage_code(root, stage), point, after, action)
    assert returncode == (1 if code == "enospc" else code)
    relative, _ = STAGE_OUTPUT[stage]
    before = {path: path.stat().st_mtime_ns for path in (root / relative).rglob("*") if path.is_file()}
    child(_stage_code(root, stage))
    _assert_stage_matches(pipeline, root, stage)
    if point in ("select.source.commit", "finalize.parent.unit_done", "count.shard.commit"):
        # committed units were reused, not rewritten
        assert any(path.exists() and path.stat().st_mtime_ns == mtime for path, mtime in before.items())
    if stage == "dedup" and point.startswith(("dedup.state", "dedup.near")):
        log = json.loads((root / "work/fixture-build/dedup/recovery_log.json").read_text())
        assert any(event.startswith("resumed at") for event in log["events"])


def test_dedup_replay_is_a_noop_and_corrupt_state_restarts(pipeline: dict[str, Any], tmp_path: Path) -> None:
    root = _stage_copy(pipeline, tmp_path, "dedup")
    assert child(_stage_code(root, "dedup"), "dedup.state.commit", 6) == KILLED
    state = root / "work/fixture-build/dedup/state"
    victim = next(state.glob("parent.*.npy"))
    data = bytearray(victim.read_bytes())
    data[-1] ^= 0xFF
    victim.write_bytes(bytes(data))
    child(_stage_code(root, "dedup"))
    _assert_stage_matches(pipeline, root, "dedup")
    log = json.loads((root / "work/fixture-build/dedup/recovery_log.json").read_text())
    assert any("failed verification" in event for event in log["events"])
    summary = root / "work/fixture-build/dedup/summary.json"
    stamp = summary.stat().st_mtime_ns
    child(_stage_code(root, "dedup"))
    assert summary.stat().st_mtime_ns == stamp  # verified summary: nothing recomputed
    keep = next((root / "work/fixture-build/dedup/keep").rglob("*.npz"))
    keep.write_bytes(keep.read_bytes()[:-1])
    child(_stage_code(root, "dedup"))  # a damaged keep file invalidates the summary
    _assert_stage_matches(pipeline, root, "dedup")


def test_selection_and_finalize_reject_corrupt_or_unmarked_outputs(
    pipeline: dict[str, Any], tmp_path: Path
) -> None:
    root = _stage_copy(pipeline, tmp_path, "select")
    child(_stage_code(root, "select"))
    masks = sorted((root / "work/fixture-build/selection").glob("fixture-*/*.npy"))
    masks[0].write_bytes(masks[0].read_bytes()[:-1] + b"\x01")
    child(_stage_code(root, "select"))
    _assert_stage_matches(pipeline, root, "select")

    child(_stage_code(root, "finalize"))
    accepted = root / "accepted/fixture-build"
    shard = next(accepted.glob("train/*/*.parquet"))
    data = bytearray(shard.read_bytes())
    data[len(data) // 2] ^= 0xFF
    shard.write_bytes(bytes(data))
    next((accepted / "_units").glob("*/*.json")).unlink()  # and one unit loses its marker
    child(_stage_code(root, "finalize"))
    _assert_stage_matches(pipeline, root, "finalize")


def test_count_rejects_corrupt_output(pipeline: dict[str, Any], tmp_path: Path) -> None:
    root = _stage_copy(pipeline, tmp_path, "count")
    child(_stage_code(root, "count"))
    table = next((root / "token_counts/fixture-build/counts").rglob("*.parquet"))
    table.write_bytes(table.read_bytes()[:-3] + b"xyz")
    child(_stage_code(root, "count"))
    _assert_stage_matches(pipeline, root, "count")


# --------------------------------------------------------------------------- audit scan


def test_audit_scan_resume_after_sigkill(tmp_path: Path) -> None:
    from test_audit_scan import SYNTHETIC_ITEM, _write_scan_inputs

    data_root, index_dir = _write_scan_inputs(tmp_path, [SYNTHETIC_ITEM, "Plain text."])
    code = f"""
        from pathlib import Path
        from j1data.audit_scan import run_scan
        print(run_scan(Path({str(data_root)!r}), "build", Path({str(index_dir)!r}), workers=1))
    """
    assert child(code, "audit_scan.shard.commit", 1) == KILLED
    outputs = list((data_root / "audits").glob("*"))
    assert len(outputs) == 1 and not (outputs[0] / "result.json").exists()
    child(code)
    resumed = tree(outputs[0])
    shutil.rmtree(outputs[0])
    child(code)
    assert tree(outputs[0]) == resumed


# --------------------------------------------------------------------------- deterministic keep files


def test_keep_files_are_byte_reproducible(tmp_path: Path) -> None:
    from j1data.dedup import save_npz_deterministic

    keep = np.array([True, False, True])
    sizes = np.array([1, 2, 3], dtype=np.int32)
    first = save_npz_deterministic(tmp_path / "a.npz", keep=keep, cluster_size=sizes)
    second = save_npz_deterministic(tmp_path / "b.npz", keep=keep, cluster_size=sizes)
    assert first == second
    with np.load(tmp_path / "a.npz") as npz:
        assert np.array_equal(npz["keep"], keep) and np.array_equal(npz["cluster_size"], sizes)


# --------------------------------------------------- recovery binding: build code, not git commit (D-02)


BOUND = """
    import json
    from pathlib import Path
    from j1data import build, dedup, select
    from j1data.minhash import LSHConfig
    from j1data.paths import DataLayout
    root = Path({root!r})
    layout = DataLayout(root)
    priorities = {{"fixture-a": 1, "fixture-b": 2}}
    work = layout.work("fixture-build")
    dedup.CHECKPOINT_SECONDS = 0.0
    if {patch_build_code!r}:
        build.build_code_fingerprint = lambda: {{"dedup.py": "0" * 64}}
    code = {{"commit": {commit!r}, "dirty": False, "package_version": "0.1.0"}}
    stage = {stage!r}
    if stage == "dedup":
        dedup.run(layout, "fixture-build", priorities, LSHConfig(14, 8), code)
    if stage == "select":
        refs = dedup.collect_units(layout, "fixture-build", priorities)
        select.run(refs, work / "dedup" / "keep", work / "selection", json.loads({selection!r}), code,
                   build_identity="fixture-identity")
"""


def _bound_code(root: Path, stage: str, commit: str, patch_build_code: bool = False) -> str:
    return BOUND.format(
        root=str(root),
        stage=stage,
        commit=commit,
        patch_build_code=patch_build_code,
        selection=json.dumps(SELECTION),
    )


def _mtimes(root: Path, pattern: str) -> dict[Path, int]:
    return {path: path.stat().st_mtime_ns for path in sorted(root.glob(pattern))}


def test_dedup_and_selection_bindings_follow_build_code_not_commit(
    pipeline: dict[str, Any], tmp_path: Path
) -> None:
    root = _stage_copy(pipeline, tmp_path, "dedup")
    dedup_dir = root / "work/fixture-build/dedup"
    summary = dedup_dir / "summary.json"
    volatile = ("recovery_log.json", "state", "code_signatures", "summary.json")

    # Killed at one commit, restarted after an unrelated commit (different git revision): resumes.
    assert child(_bound_code(root, "dedup", "commit-a"), "dedup.state.commit", 3) == KILLED
    child(_bound_code(root, "dedup", "commit-b"))
    log = json.loads((dedup_dir / "recovery_log.json").read_text())
    assert any(event.startswith("resumed at") for event in log["events"]), log["events"]
    assert tree(dedup_dir, volatile) == tree(pipeline["root"] / "work/fixture-build/dedup", volatile)
    record = json.loads(summary.read_text())
    assert record["code"]["commit"] == "commit-b"  # informational
    assert "code" not in record["recovery_binding"]
    assert set(record["recovery_binding"]["build_code"]) == set(_build_modules())

    # Replay after yet another commit: the verified summary is reused, nothing recomputed.
    stamp = summary.stat().st_mtime_ns
    child(_bound_code(root, "dedup", "commit-c"))
    assert summary.stat().st_mtime_ns == stamp

    # A changed build module hash invalidates the committed stage.
    child(_bound_code(root, "dedup", "commit-c", patch_build_code=True))
    assert summary.stat().st_mtime_ns != stamp
    record = json.loads(summary.read_text())
    assert record["recovery_binding"]["build_code"] == {"dedup.py": "0" * 64}
    assert not any(event.startswith("resumed at") for event in json.loads(
        (dedup_dir / "recovery_log.json").read_text()
    )["events"])  # fmt: skip
    assert tree(dedup_dir, volatile) == tree(pipeline["root"] / "work/fixture-build/dedup", volatile)

    # Selection: committed source markers survive an unrelated commit, not a build-code change.
    selection = root / "work/fixture-build/selection"
    child(_bound_code(root, "select", "commit-a"))
    markers = _mtimes(selection, "_sources/*.json")
    assert markers
    child(_bound_code(root, "select", "commit-b"))
    assert _mtimes(selection, "_sources/*.json") == markers
    marker = json.loads(next(iter(markers)).read_text())
    assert marker["code"]["commit"] == "commit-a" and "code" not in marker["binding"]
    assert json.loads((selection / "summary.json").read_text())["code"]["commit"] == "commit-b"
    child(_bound_code(root, "select", "commit-b", patch_build_code=True))
    rewritten = _mtimes(selection, "_sources/*.json")
    assert rewritten.keys() == markers.keys()
    assert all(rewritten[path] != stamp for path, stamp in markers.items())
    masks = ("_sources", "summary.json")  # markers and summary carry the code and fingerprint
    assert tree(selection, masks) == tree(pipeline["root"] / "work/fixture-build/selection", masks)


def _build_modules() -> tuple[str, ...]:
    from j1data.build import BUILD_MODULES

    return BUILD_MODULES
