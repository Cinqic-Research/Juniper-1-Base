"""j1data.sftmix: exact registered draws, byte-identical repeated rows, fail-closed parent checks."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from j1data import sftmix
from j1data.dedup import save_npz_deterministic
from j1data.hashing import canonical_json, sha256_file, sha256_json

FAMILIES = [
    ("math", "arith.add"),
    ("math", "arith.mul"),
    ("json", "json.kv"),
    ("instruction_following", "if.op.x"),
]


def _parent(root: Path, n_per: int = 5, fill: int = 7) -> Path:
    root.mkdir()
    records, seg_in, seg_tg, ids = [], [], [], []
    for k, (cat, fam) in enumerate(FAMILIES):
        for j in range(n_per):
            eid = bytes([k, j] + [fill] * 14)
            toks = np.array([100 + k, 200 + j, 9], dtype=np.int32)
            seg_in.append(toks[:-1])
            seg_tg.append(toks[1:])
            ids.append(np.frombuffer(eid, dtype=np.uint8))
            records.append(
                {"category": cat, "family": fam, "license": "Cinqic-generated", "episode_id": eid.hex()}
            )
    lens = [len(s) for s in seg_in]
    save_npz_deterministic(
        root / "train_rows.npz",
        inputs=np.concatenate(seg_in),
        targets=np.concatenate(seg_tg),
        offsets=np.concatenate([[0], np.cumsum(lens)]).astype(np.int64),
        episode_ids=np.stack(ids),
    )
    save_npz_deterministic(
        root / "dev_rows.npz",
        inputs=np.array([5, 6], dtype=np.int32),
        targets=np.array([6, 9], dtype=np.int32),
        offsets=np.array([0, 2], dtype=np.int64),
        episode_ids=np.full((1, 16), 255, dtype=np.uint8),
    )
    (root / "train.jsonl").write_bytes(b"".join(canonical_json(r) + b"\n" for r in records))
    (root / "dev.jsonl").write_text("{}\n")
    (root / "NOTICE").write_text("notice\n")
    files = {
        f: {"sha256": sha256_file(root / f)}
        for f in ("train_rows.npz", "dev_rows.npz", "train.jsonl", "dev.jsonl", "NOTICE")
    }
    body = {
        "schema": "j1.sft-dataset.v1",
        "stage": "agent-instruction-sft",
        "version": "parent",
        "approved_licenses": ["Cinqic-generated", "Apache-2.0"],
        "eval_bundle_sha256": "e" * 64,
        "tokenizer_sha256": "t" * 64,
        "counts": {"dev": {"episodes": 1}, "rejected": 0},
        "files": files,
    }
    (root / "manifest.json").write_bytes(
        canonical_json({**body, "manifest_sha256": sha256_json(body)}) + b"\n"
    )
    return root


def _config(path: Path, parent: Path, arith: int, rest: int, **extra: object) -> Path:
    sha = json.loads((parent / "manifest.json").read_text())["manifest_sha256"]
    lic = extra.get("licenses", ["Cinqic-generated"])
    path.write_text(
        f'version = "mix"\nparent_manifest_sha256 = "{sha}"\nseed = 3\nupdates = {(arith + rest) // 4}\n'
        f"episodes_per_update = 4\ndraws_total = {arith + rest}\napproved_licenses = {json.dumps(lic)}\n"
        f'[groups.arith]\nselect = {{ family_prefix = "arith" }}\ndraws = {arith}\n'
        f'[groups.rest]\nselect = {{ categories = ["json", "instruction_following"] }}\ndraws = {rest}\n'
    )
    return path


def test_exact_draws_identical_rows_and_determinism(tmp_path: Path) -> None:
    parent = _parent(tmp_path / "p")
    cfg = _config(tmp_path / "c.toml", parent, arith=23, rest=13)
    m = sftmix.mix(parent, cfg, tmp_path / "a")
    again = sftmix.mix(parent, cfg, tmp_path / "b")
    assert m["manifest_sha256"] == again["manifest_sha256"]
    assert m["approved_licenses"] == ["Cinqic-generated"]
    groups = m["mixture"]["groups"]
    assert groups["arith"]["families"] == {
        "arith.add": {"episodes": 5, "draws": 12},  # largest remainder: 11.5 -> 12 (tie broken by name)
        "arith.mul": {"episodes": 5, "draws": 11},
    }
    plan = [json.loads(x) for x in (tmp_path / "a" / "mixture_plan.jsonl").read_text().splitlines()]
    assert len(plan) == 36 and Counter(p["group"] for p in plan) == {"arith": 23, "rest": 13}
    per_parent = Counter(p["parent_episode_id"] for p in plan)
    assert set(per_parent.values()) <= {0, 1, 2, 3}
    with np.load(parent / "train_rows.npz") as z, np.load(tmp_path / "a" / "train_rows.npz") as y:
        assert len({bytes(x) for x in y["episode_ids"]}) == 36
        for row, p in enumerate(plan):
            a, b = z["offsets"][p["parent_row"]], z["offsets"][p["parent_row"] + 1]
            c, d = y["offsets"][row], y["offsets"][row + 1]
            assert np.array_equal(z["inputs"][a:b], y["inputs"][c:d])
            assert np.array_equal(z["targets"][a:b], y["targets"][c:d])
    assert (tmp_path / "a" / "dev_rows.npz").read_bytes() == (parent / "dev_rows.npz").read_bytes()


@pytest.mark.parametrize(
    "case",
    ["changed_parent", "draws_mismatch", "widened_license", "unassigned_episode"],
)
def test_fails_closed(tmp_path: Path, case: str) -> None:
    parent = _parent(tmp_path / "p")
    cfg = _config(tmp_path / "c.toml", parent, arith=20, rest=12)
    if case == "changed_parent":
        (parent / "NOTICE").write_text("edited\n")
    elif case == "draws_mismatch":
        cfg.write_text(cfg.read_text().replace("draws_total = 32", "draws_total = 36"))
    elif case == "widened_license":
        _config(cfg, parent, 20, 12, licenses=["Cinqic-generated", "GPL-3.0"])
    else:
        cfg.write_text(cfg.read_text().replace('"json", ', ""))
    with pytest.raises(SystemExit):
        sftmix.mix(parent, cfg, tmp_path / "out")


def _replay(root: Path, parent: Path, fill: int = 7) -> tuple[Path, Path]:
    """A second frozen dataset (v1.2 stand-in) plus an admission of two of its JSON episodes."""
    r = _parent(root, fill=fill)
    m = json.loads((r / "manifest.json").read_text())
    body = {k: v for k, v in m.items() if k != "manifest_sha256"} | {"version": "replay"}
    (r / "manifest.json").write_bytes(canonical_json({**body, "manifest_sha256": sha256_json(body)}) + b"\n")
    rm = json.loads((r / "manifest.json").read_text())
    pm = json.loads((parent / "manifest.json").read_text())
    recs = [json.loads(x) for x in (r / "train.jsonl").read_text().splitlines()]
    adm = {
        "schema": "j1.sft-replay.v1",
        "v12_manifest_sha256": rm["manifest_sha256"],
        "v13_manifest_sha256": pm["manifest_sha256"],
        "admitted_episode_ids": [x["episode_id"] for x in recs if x["family"] == "json.kv"][:2],
    }
    adm["admission_sha256"] = sha256_json(adm)
    (root.parent / "admission.json").write_bytes(canonical_json(adm) + b"\n")
    return r, root.parent / "admission.json"


def _replay_config(path: Path, parent: Path, replay: Path, admission: Path) -> Path:
    sha = json.loads((parent / "manifest.json").read_text())["manifest_sha256"]
    rsha = json.loads((replay / "manifest.json").read_text())["manifest_sha256"]
    asha = json.loads(admission.read_text())["admission_sha256"]
    path.write_text(
        f'version = "mix"\nparent_manifest_sha256 = "{sha}"\nseed = 3\nupdates = 9\nepisodes_per_update = 4\n'
        f'draws_total = 36\napproved_licenses = ["Cinqic-generated"]\n'
        f'[replay]\ndataset_manifest_sha256 = "{rsha}"\nadmission_sha256 = "{asha}"\n'
        f'[groups.arith]\nselect = {{ family_prefix = "arith" }}\ndraws = 20\n'
        f'[groups.rest]\nselect = {{ categories = ["json", "instruction_following"] }}\ndraws = 12\n'
        f'[groups.json_replay]\nselect = {{ source = "replay", category = "json" }}\ndraws = 4\n'
    )
    return path


def test_replay_source_draws_admitted_rows_only(tmp_path: Path) -> None:
    parent = _parent(tmp_path / "p")
    replay, admission = _replay(tmp_path / "r", parent)
    cfg = _replay_config(tmp_path / "c.toml", parent, replay, admission)
    # Replay IDs equal the parent's in this synthetic fixture: they must be refused as colliding.
    with pytest.raises(SystemExit, match="collide"):
        sftmix.mix(parent, cfg, tmp_path / "o0", replay, admission)


def test_replay_requires_pinned_admission_and_matching_parent(tmp_path: Path) -> None:
    parent = _parent(tmp_path / "p")
    replay, admission = _replay(tmp_path / "r", parent)
    cfg = _replay_config(tmp_path / "c.toml", parent, replay, admission)
    with pytest.raises(SystemExit, match="give --replay-dataset"):
        sftmix.mix(parent, cfg, tmp_path / "o1")
    adm = json.loads(admission.read_text())
    adm["v13_manifest_sha256"] = "0" * 64
    adm["admission_sha256"] = sha256_json({k: v for k, v in adm.items() if k != "admission_sha256"})
    admission.write_bytes(canonical_json(adm) + b"\n")
    with pytest.raises(SystemExit, match=r"not the pinned one|another parent"):
        sftmix.mix(parent, cfg, tmp_path / "o2", replay, admission)


def test_replay_rows_are_admitted_byte_identical_and_labelled(tmp_path: Path) -> None:
    parent = _parent(tmp_path / "p")
    replay, admission = _replay(tmp_path / "r", parent, fill=9)
    cfg = _replay_config(tmp_path / "c.toml", parent, replay, admission)
    m = sftmix.mix(parent, cfg, tmp_path / "o", replay, admission)
    assert m["mixture"]["replay"]["admitted_episodes"] == 2
    assert m["mixture"]["groups"]["json_replay"]["draws"] == 4
    plan = [json.loads(x) for x in (tmp_path / "o" / "mixture_plan.jsonl").read_text().splitlines()]
    rp = [p for p in plan if p["source"] == "replay"]
    admitted = set(json.loads(admission.read_text())["admitted_episode_ids"])
    assert len(rp) == 4 and {p["parent_episode_id"] for p in rp} == admitted
    with np.load(replay / "train_rows.npz") as z, np.load(tmp_path / "o" / "train_rows.npz") as y:
        src = {bytes(e).hex(): i for i, e in enumerate(z["episode_ids"])}
        for row, p in enumerate(plan):
            if p["source"] != "replay":
                continue
            i = src[p["parent_episode_id"]]
            a, b = z["offsets"][i], z["offsets"][i + 1]
            c, d = y["offsets"][row], y["offsets"][row + 1]
            assert np.array_equal(z["inputs"][a:b], y["inputs"][c:d])
