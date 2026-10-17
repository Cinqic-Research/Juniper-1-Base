"""Regressions for the October 14 independent review; synthetic inputs only."""

from __future__ import annotations

import json
import time
import tracemalloc
from pathlib import Path

import pytest

from j1data import codecheck as cc
from j1data import jtok, sandbox, sfteval, sftscore
from j1data.hashing import sha256_file, sha256_json

needs_sandbox = pytest.mark.skipif(not sandbox.available(), reason="bubblewrap unavailable")


@needs_sandbox
@pytest.mark.parametrize("fd", [1, 2])
def test_pipe_flood_is_bounded_in_host_and_fails_closed(fd):
    source = f"import os\nfor _ in range(10000): os.write({fd}, b'x' * 4096)\n"
    tracemalloc.start()
    try:
        result = sandbox.run_python(source, sandbox.Limits(output_bytes=65536, wall_seconds=3))
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert result.output_limited and not result.ok and not result.timed_out
    assert len(result.stdout) <= sandbox.MAX_OUTPUT_CHARS
    assert len(result.stderr) <= sandbox.MAX_OUTPUT_CHARS
    assert peak < 2 << 20  # 40 MiB attempted output must not be retained in the host


@needs_sandbox
@pytest.mark.parametrize("close_pipes", [False, True])
def test_descendant_is_killed_at_deadline_even_with_new_session_and_closed_pipes(close_pipes):
    source = "import os, time\nif os.fork() == 0:\n    os.setsid()\n"
    if close_pipes:
        source += "os.close(1)\nos.close(2)\n"
    source += "time.sleep(60)\n"
    before = time.monotonic()
    result = sandbox.run_python(source, sandbox.Limits(wall_seconds=0.3))
    assert result.timed_out and not result.ok
    assert time.monotonic() - before < 3


@pytest.mark.parametrize("purpose", ["adversarial_reward", "adversarial_qualification", "unknown"])
def test_unqualified_purposes_refused_before_execution(purpose, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("must refuse before execution")

    monkeypatch.setattr(sandbox, "run_python", forbidden)
    spec = {"entry_point": "f", "inputs": [[1]], "expected": [42]}
    with pytest.raises(cc.UnqualifiedOracle):
        cc.score_python_io(spec, "def f(x): return 0", purpose=purpose)
    with pytest.raises(cc.UnqualifiedOracle):
        cc.run_io("def f(x): return 0", "f", [[1]], purpose=purpose)
    suite = {"spec": {"tiers": ["own"], "purpose": purpose}}
    with pytest.raises(cc.UnqualifiedOracle):
        sftscore.score(
            {
                "type": "python_suite",
                "entry_point": "f",
                "suite": suite,
                "tiers": ["own"],
                "purpose": purpose,
            },
            "def f(x): return 0",
        )


@needs_sandbox
def test_frame_patch_is_documented_untrusted_output_not_a_return_oracle():
    spec = {"entry_point": "f", "inputs": [[1], [2], [3]], "expected": [42, 42, 42]}
    code = (
        "import sys\ndef f(x):\n    sys._getframe().f_back.f_globals['_enc'] = lambda *a: 42\n    return 0\n"
    )
    assert not cc.score_python_io(spec, "def f(x): return 0")[0]
    passed, detail = cc.score_python_io(spec, code)
    assert passed and "not authenticated function returns" in detail
    assert cc._code_identity()["purpose"] == "functional_diagnostic"
    with pytest.raises(cc.UnqualifiedOracle):
        cc.score_python_io(spec, code, purpose="adversarial_qualification")


@needs_sandbox
def test_valid_report_followed_by_pipe_flood_never_passes():
    spec = {"entry_point": "f", "inputs": [[1]], "expected": [42]}
    code = (
        "import sys, os\ndef f(x):\n    fr = sys._getframe().f_back\n"
        "    fr.f_globals['_emit'](fr.f_locals['i'], ['ok', 42])\n"
        "    for _ in range(10000): os.write(2, b'x' * 4096)\n    os._exit(0)\n"
    )
    assert not cc.score_python_io(spec, code, cc.RunConfig(limits=sandbox.Limits(output_bytes=65536)))[0]


def _bundle(tmp_path: Path, variant: str = "chat", value: str = "", duplicate: str | None = None):
    tok = jtok.train(
        jtok.CandidateSpec("review-fixture", 448, min_frequency=1, stem_alphabet=False),
        [
            "The quick brown fox jumps over the lazy dog. What is 12 + 7? The answer is 19.",
            '{"name": "Alice", "age": 30} I don\'t know. ```python\ndef f(x):\n    return x + 1\n```',
        ]
        * 50,
    )
    tokenizer = tmp_path / "tokenizer.json"
    tokenizer.write_text(tok.to_str())
    bundle, run_dir = tmp_path / "bundle", tmp_path / "run"
    bundle.mkdir()
    run_dir.mkdir()
    prompt = {
        "id": "synthetic",
        "split": "test",
        "variant": "chat",
        "family": "if.synthetic",
        "category": "instruction_following",
    }
    for name, row in (
        ("prompts.jsonl", prompt),
        ("answers.jsonl", {"id": "synthetic", "spec": {"type": "exact", "value": value}}),
    ):
        bundle.joinpath(name).write_text((json.dumps(row) + "\n") * (2 if duplicate == name else 1))
    manifest = {
        "tokenizer_sha256": sha256_file(tokenizer),
        "files": {n: {"sha256": sha256_file(bundle / n)} for n in ("prompts.jsonl", "answers.jsonl")},
    }
    manifest["bundle_sha256"] = sha256_json(manifest)
    bundle.joinpath("manifest.json").write_text(json.dumps(manifest))
    ids = (
        [jtok.control_id("<|final|>"), jtok.control_id("<|eot|>")]
        if not value
        else tok.encode(value, add_special_tokens=False).ids
    )
    run_dir.joinpath("generations.jsonl").write_text(
        json.dumps(
            {
                "id": "synthetic",
                "variant": variant,
                "generated_ids": ids,
                "stop": "synthetic",
                "new_tokens": len(ids),
            }
        )
        + "\n"
    )
    run_dir.joinpath("run.json").write_text(
        json.dumps(
            {
                "model": "synthetic",
                "bundle_sha256": manifest["bundle_sha256"],
                "generations_sha256": sha256_file(run_dir / "generations.jsonl"),
                "selection": {"split": "test", "variant": "chat", "families": "all"},
            }
        )
    )
    return bundle, run_dir, tokenizer, manifest["tokenizer_sha256"]


@needs_sandbox
def test_valid_computed_empty_answer_scores_and_invalid_empty_does_not(tmp_path):
    paths = _bundle(tmp_path)
    report = sfteval.score_run(*paths)
    assert report["items"][0]["strict"] and report["items"][0]["lenient"]
    assert report["scorer"] == "j1.sfteval.v3"
    assert "no adversarial qualification" in report["executable_scoring"]["limitation"]
    assert sfteval._content(jtok.load(paths[2].read_text()), [], "chat")[0] is False


@needs_sandbox
def test_generation_cannot_override_pinned_chat_variant(tmp_path):
    with pytest.raises(SystemExit, match="variant differs"):
        sfteval.score_run(*_bundle(tmp_path, variant="base_fewshot", value="42"))


@needs_sandbox
@pytest.mark.parametrize("duplicate", ["prompts.jsonl", "answers.jsonl"])
def test_bundle_duplicate_ids_fail_closed(tmp_path, duplicate):
    with pytest.raises(SystemExit, match="duplicated"):
        sfteval.score_run(*_bundle(tmp_path, duplicate=duplicate))


@needs_sandbox
def test_bundle_false_self_hash_fails_closed(tmp_path):
    paths = _bundle(tmp_path)
    p = paths[0] / "manifest.json"
    manifest = json.loads(p.read_text())
    manifest["bundle_sha256"] = "forged"
    p.write_text(json.dumps(manifest))
    with pytest.raises(SystemExit, match="self-hash"):
        sfteval.score_run(*paths)


def test_python_tests_refuses_adversarial_purposes() -> None:
    spec = {
        "type": "python_tests",
        "entry_point": "f",
        "tests": "assert f() == 1",
        "purpose": "adversarial_reward",
    }
    with pytest.raises(sftscore.SpecError):
        sftscore.score(spec, "def f():\n    return 1\n")
    with pytest.raises(sftscore.SpecError):
        sftscore.run_tests(
            "def f():\n    return 1\n", "", "assert f() == 1", purpose="adversarial_qualification"
        )


@pytest.mark.skipif(not sandbox.available(), reason="bubblewrap sandbox unavailable")
def test_reports_before_a_timeout_are_kept_without_rerunning_every_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[int] = []
    real = sandbox.run_python

    def counting(source: str, limits: sandbox.Limits | None = None) -> sandbox.Result:
        calls.append(1)
        return real(source, limits)

    monkeypatch.setattr(sandbox, "run_python", counting)
    code = (
        "import threading, time\n"
        "def f(x):\n"
        "    threading.Thread(target=time.sleep, args=(60,)).start()\n"
        "    return x\n"
    )
    got = cc.run_io(
        code, "f", [[i] for i in range(10)], config=cc.RunConfig(limits=sandbox.Limits(wall_seconds=3.0))
    )
    assert [g.kind for g in got] == ["ok"] * 10
    assert len(calls) == 1
