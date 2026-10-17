"""j1data.sftconfirm: exclusion by v1.3 train/dev/test and v1.2 test task keys and by v1.3 test prompt
text; dev prompts dropped; unchanged bundle format; deterministic bytes; fail-closed input checks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from j1data import sftconfirm
from j1data.hashing import canonical_json, sha256_bytes, sha256_file, sha256_json

TOK = "ab" * 32


def _k(name: str) -> str:
    return sha256_bytes(name.encode())


def _decode(ids: list[int]) -> str:
    return ",".join(map(str, ids))


def _jsonl(path: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"".join(canonical_json(r) + b"\n" for r in rows))
    return {"sha256": sha256_file(path), "size": path.stat().st_size, "rows": len(rows)}


def _seal(path: Path, body: dict[str, Any], key: str) -> dict[str, Any]:
    body[key] = sha256_json(body)
    path.write_text(json.dumps(body, indent=1, sort_keys=True))
    return body


def _v13(root: Path) -> Path:
    d = root / "v13"
    files = {
        "train.jsonl": _jsonl(
            d / "train.jsonl",
            [
                {"task_key_sha256": _k("t1"), "task_keys_sha256": [_k("t1")]},
                {"task_key_sha256": _k("t2"), "task_keys_sha256": [_k("t2"), _k("embedded")]},
            ],
        ),
        "dev.jsonl": _jsonl(d / "dev.jsonl", [{"task_key_sha256": _k("d1"), "task_keys_sha256": [_k("d1")]}]),
    }
    efiles = {
        "test_records.jsonl": _jsonl(
            d / "eval/test_records.jsonl",
            [{"episode_id": "x1", "task_key_sha256": _k("old-test"), "task_keys_sha256": [_k("other")]}],
        ),
        "prompts.jsonl": _jsonl(
            d / "eval/prompts.jsonl",
            [
                {"id": "x1", "split": "test", "variant": "chat", "prompt_ids": [9, 9, 9]},
                {"id": "x2", "split": "dev", "variant": "chat", "prompt_ids": [7, 7, 7]},
            ],
        ),
    }
    em = _seal(d / "eval/manifest.json", {"schema": sftconfirm.SCHEMA, "files": efiles}, "bundle_sha256")
    _seal(d / "manifest.json", {"files": files, "eval_bundle_sha256": em["bundle_sha256"]}, "manifest_sha256")
    return d


# (id, family, task_key, task_keys, chat prompt ids); expected outcome noted per row
ITEMS = [
    ("a", "if.op.x", "fresh-a", ["fresh-a"], [1, 2]),  # kept (also has a few-shot prompt)
    ("b", "if.op.x", "t1", ["t1"], [1, 3]),  # v1.3 train key
    ("c", "hierarchy.system", "fresh-c", ["fresh-c", "embedded"], [1, 4]),  # embedded v1.3 train key
    ("d", "json.kv", "d1", ["d1"], [1, 5]),  # v1.3 dev key
    ("e", "json.kv", "old-test", ["fresh-e"], [1, 6]),  # item key (not in the list) is a v1.3 test key
    ("f", "arith.basic", "v12", ["v12"], [1, 7]),  # protected v1.2 test key
    ("g", "arith.basic", "fresh-g", ["fresh-g"], [9, 9, 9]),  # prompt text equals a v1.3 test prompt
    ("h", "ext.mbpp", "other", ["other"], [2, 1]),  # fixed pool: v1.3 test key; family emptied
    ("i", "ext.gsm8k", "fresh-i", ["fresh-i"], [2, 2]),  # fixed pool, kept
    ("j", "ext.gsm8k", "t2", ["t2"], [2, 3]),  # fixed pool, removed
    ("k", "arith.basic", "fresh-k", ["fresh-k"], [3, 3]),  # kept: arith.basic is mostly removed
]


def _scratch(root: Path) -> Path:
    d = root / "scratch"
    prompts, answers, records = [], [], []
    for eid, fam, key, keys, ids in ITEMS:
        base = {"split": "test", "family": fam, "category": fam.split(".")[0], "max_new_tokens": 8}
        prompts.append({"id": eid, "variant": "chat", **base, "prompt_ids": ids})
        answers.append({"id": eid, "spec": {"type": "exact"}, "reference": "r", "turns": 1})
        if eid in ("a", "g"):
            prompts.append(
                {"id": eid + "#fewshot", "variant": "base_fewshot", **base, "prompt_ids": [5, *ids]}
            )
            answers.append({"id": eid + "#fewshot", "spec": {"type": "exact"}, "reference": "r", "turns": 1})
        records.append(
            {
                "episode_id": eid,
                "family": fam,
                "split": "test",
                "task_key_sha256": _k(key),
                "task_keys_sha256": sorted(_k(k) for k in keys),
            }
        )
    prompts.append({"id": "z", "variant": "chat", "split": "dev", "family": "json.kv", "prompt_ids": [3]})
    answers.append({"id": "z", "spec": {"type": "exact"}, "reference": "r", "turns": 1})
    (d / "eval").mkdir(parents=True)
    (d / "eval/NOTICE").write_text("notice\n")
    efiles = {
        "NOTICE": {"sha256": sha256_file(d / "eval/NOTICE"), "size": 7},
        "prompts.jsonl": _jsonl(d / "eval/prompts.jsonl", prompts),
        "answers.jsonl": _jsonl(d / "eval/answers.jsonl", answers),
        "test_records.jsonl": _jsonl(d / "eval/test_records.jsonl", records),
    }
    em = {
        "schema": sftconfirm.SCHEMA,
        "version": "j1-sft-eval-confirm-v1",
        "protocol": "juniper-chat-v1",
        "tokenizer_sha256": TOK,
        "stop_ids": [11, 10],
        "forbidden_ids": [0, 1, 2],
        "files": efiles,
        "note": "answers are read only by the scorer",
    }
    _seal(d / "eval/manifest.json", em, "bundle_sha256")
    _seal(
        d / "manifest.json",
        {
            "version": "j1-sft-confirm-v1",
            "config": {"path": "configs/x.toml", "sha256": "0" * 64},
            "code_sha256": {},
            "tokenizer_sha256": TOK,
            "eval_bundle_sha256": em["bundle_sha256"],
        },
        "manifest_sha256",
    )
    return d


@pytest.fixture
def env(tmp_path: Path) -> dict[str, Any]:
    prot = tmp_path / "v12_test_records.jsonl"
    _jsonl(prot, [{"task_key_sha256": _k("v12"), "task_keys_sha256": [_k("v12")]}])
    return {"v13": _v13(tmp_path), "scratch": _scratch(tmp_path), "prot": prot, "tmp": tmp_path}


def _run(env: dict[str, Any], out: Path) -> dict[str, Any]:
    return sftconfirm.confirm(
        env["scratch"], env["v13"], env["prot"], sha256_file(env["prot"]), out, _decode, TOK, "c" * 64
    )


def _read(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_bytes().split(b"\n") if x]


def test_exclusions_and_counts(env: dict[str, Any]) -> None:
    out = env["tmp"] / "out"
    m = _run(env, out)
    assert [r["episode_id"] for r in _read(out / "test_records.jsonl")] == ["a", "i", "k"]
    assert [p["id"] for p in _read(out / "prompts.jsonl")] == ["a", "a#fewshot", "i", "k"]
    assert [a["id"] for a in _read(out / "answers.jsonl")] == ["a", "a#fewshot", "i", "k"]
    fam = m["families"]
    assert fam["if.op.x"]["kept"] == 1 and fam["if.op.x"]["removed_by"]["v13_train_dev_key"] == 1
    assert fam["hierarchy.system"]["removed_by"]["v13_train_dev_key"] == 1
    assert fam["json.kv"]["removed"] == 2
    assert fam["json.kv"]["removed_by"]["v13_test_key"] == 1
    assert fam["arith.basic"]["removed_by"]["v12_test_key"] == 1
    assert fam["arith.basic"]["removed_by"]["v13_test_prompt_text"] == 1
    assert fam["ext.mbpp"] == {
        "kept": 0,
        "removed": 1,
        "removed_by": dict(dict.fromkeys(sftconfirm.REASONS, 0), v13_test_key=1),
    }
    assert m["counts"]["test_items_kept"] == 3 and m["counts"]["test_items_removed"] == 8
    assert m["parent"]["dropped_dev_prompts"] == 1
    fr = m["freshness"]
    assert fr["fixed_pools_emptied"] == ["mbpp"] and fr["fixed_pools_partial"] == ["gsm8k"]
    assert fr["fixed_pools"]["gsm8k"] == {"kept": 1, "removed": 1}
    assert {"json.kv", "hierarchy.system"} <= set(fr["synthetic_families_emptied"])
    assert fr["synthetic_families_mostly_removed"] == ["arith.basic"]
    assert "fresh instances of the same held-out design" in fr["note"] and "mbpp" in fr["note"]


def test_bundle_loads_like_score_run(env: dict[str, Any]) -> None:
    out = env["tmp"] / "out"
    m = _run(env, out)
    disk = json.loads((out / "manifest.json").read_text())
    assert disk == m and m["version"] == "j1-sft-confirm-v1" and m["schema"] == sftconfirm.SCHEMA
    assert sha256_json({k: v for k, v in disk.items() if k != "bundle_sha256"}) == disk["bundle_sha256"]
    old = json.loads((env["scratch"] / "eval/manifest.json").read_text())
    assert disk["bundle_sha256"] != old["bundle_sha256"]
    for k in ("tokenizer_sha256", "stop_ids", "forbidden_ids", "protocol", "note"):
        assert disk[k] == old[k]
    for name in sftconfirm.BUNDLE_FILES:
        assert sha256_file(out / name) == disk["files"][name]["sha256"]
    ids_p = [p["id"] for p in _read(out / "prompts.jsonl")]
    ids_a = [a["id"] for a in _read(out / "answers.jsonl")]
    assert len(set(ids_p)) == len(ids_p) and set(ids_p) == set(ids_a)


def test_deterministic_and_never_overwrites(env: dict[str, Any]) -> None:
    a, b = env["tmp"] / "a", env["tmp"] / "b"
    _run(env, a)
    _run(env, b)
    for name in (*sftconfirm.BUNDLE_FILES, "manifest.json"):
        assert (a / name).read_bytes() == (b / name).read_bytes()
    with pytest.raises(SystemExit, match="not empty"):
        _run(env, a)


def test_fails_closed_on_changed_inputs(env: dict[str, Any]) -> None:
    with (env["v13"] / "dev.jsonl").open("ab") as fh:
        fh.write(b"{}\n")
    with pytest.raises(SystemExit, match="differ"):
        _run(env, env["tmp"] / "out")


def test_fails_closed_on_protected_digest(env: dict[str, Any]) -> None:
    with pytest.raises(SystemExit, match="protected"):
        sftconfirm.confirm(env["scratch"], env["v13"], env["prot"], "0" * 64, env["tmp"] / "o", _decode, TOK)


def test_v12_train_dev_keys_are_excluded_when_given(env: dict[str, Any]) -> None:
    d = env["tmp"] / "v12"
    files = {
        "train.jsonl": _jsonl(
            d / "train.jsonl", [{"task_key_sha256": _k("fresh-i"), "task_keys_sha256": []}]
        ),
        "dev.jsonl": _jsonl(d / "dev.jsonl", [{"task_key_sha256": _k("zz"), "task_keys_sha256": [_k("zz")]}]),
    }
    # A v1.2 evaluation prompt identical to item k's chat prompt, under a different task key.
    efiles = {
        "prompts.jsonl": _jsonl(
            d / "eval/prompts.jsonl", [{"id": "q1", "split": "test", "variant": "chat", "prompt_ids": [3, 3]}]
        )
    }
    _seal(d / "eval/manifest.json", {"schema": sftconfirm.SCHEMA, "files": efiles}, "bundle_sha256")
    _seal(d / "manifest.json", {"files": files}, "manifest_sha256")
    out = env["tmp"] / "out2"
    m = sftconfirm.confirm(
        env["scratch"],
        env["v13"],
        env["prot"],
        sha256_file(env["prot"]),
        out,
        _decode,
        TOK,
        "c" * 64,
        v12_build=d,
        version="j1-sft-confirm-v2",
    )
    assert [r["episode_id"] for r in _read(out / "test_records.jsonl")] == ["a"]
    assert m["version"] == "j1-sft-confirm-v2"
    assert "v12_train_dev_key" in m["exclusions"]["sets"]
    assert m["families"]["ext.gsm8k"]["removed_by"]["v12_train_dev_key"] == 1
    assert m["families"]["arith.basic"]["removed_by"]["v12_eval_prompt_text"] == 1
    assert "v12_eval_prompt_text" in m["exclusions"]["sets"]
