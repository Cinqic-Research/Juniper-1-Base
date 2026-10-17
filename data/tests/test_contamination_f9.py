"""F9 (HumanEval+ / MBPP+ test fields): windows, fail-closed items, offline pass 2, restart, adjudication."""

import importlib.util
import json
import random
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools"
for name in ("contamination_layers", "contamination_f9"):
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
L = sys.modules["contamination_layers"]
F = sys.modules["contamination_f9"]

HARNESS = (
    "def check(candidate):\n"
    "    for i, (inp, exp) in enumerate(zip(inputs, results)):\n"
    "        assert candidate(*inp) == exp\n"
)


def _test_field(seed: int) -> str:
    rng = random.Random(seed)
    inputs = [[rng.randint(-999, 999) for _ in range(4)] for _ in range(40)]
    results = [rng.random() < 0.5 for _ in range(40)]
    return f"inputs = {inputs}\nresults = {results}\n" + HARNESS


def _fixture(tmp_path, *, he_test=None, drop_col=False, corrupt=False):
    raw = tmp_path / "raw"
    rev_he, rev_mp, rev_mb = "a" * 40, "b" * 40, "c" * 40
    files = {}

    def write(bench, rev, rel, table):
        p = raw / bench / rev / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, p)
        files[bench] = {"path": rel, "sha256": L.sha256_file(p)}
        if corrupt and bench == "mbppplus":
            p.write_bytes(p.read_bytes() + b"x")

    he = {"task_id": ["HumanEval/0", "HumanEval/1"], "prompt": ["p", "q"]}
    he["test"] = he_test if he_test is not None else [_test_field(1), _test_field(2)]
    if drop_col:
        he.pop("test")
    write("humanevalplus", rev_he, "data/test.parquet", pa.table(he))
    write("mbppplus", rev_mp, "data/test.parquet", pa.table({"task_id": [2], "test": [_test_field(3)]}))
    write(
        "mbpp",
        rev_mb,
        "full/test.parquet",
        pa.table(
            {
                "task_id": [11, 12],
                "challenge_test_list": [['assert remove_occ("hellolloll","l") == "helollol"'], []],
                "test_setup_code": ["", ""],
            }
        ),
    )
    reg = {"benchmark": []}
    prov = {}
    for bench, rev, repo in (
        ("humanevalplus", rev_he, "evalplus/humanevalplus"),
        ("mbppplus", rev_mp, "evalplus/mbppplus"),
        ("mbpp", rev_mb, "google-research-datasets/mbpp"),
    ):
        reg["benchmark"].append({"id": bench, "repo": repo, "revision": rev})
        prov[bench] = {"repo": repo, "revision": rev, "file_hashes": [files[bench]]}
    return raw, prov, reg


def test_windows_cover_value_and_reproduce_its_grams():
    text = _test_field(5)
    toks = L.TOKEN_RE.findall(L.normalize(text))
    ws = F.windows(text)
    assert len(ws) == -(-len(toks) // F.W)
    assert all(len(L.TOKEN_RE.findall(w)) == F.W for w in ws)
    assert L.TOKEN_RE.findall(ws[-1]) == toks[-F.W :]
    whole = set(L.grams(text).tolist())
    assert set().union(*(set(L.grams(w).tolist()) for w in ws)) <= whole
    assert F.windows("short value only") == []


def test_collect_reads_added_fields_and_records_hashes(tmp_path):
    raw, prov, reg = _fixture(tmp_path)
    items, report = F.collect(raw, prov, reg)
    assert set(items) == set(F.SPECS)
    assert report["labels"]["humanevalplus-test"]["records"] == 2
    assert report["labels"]["humanevalplus-test"]["window_items"] > 0
    assert items["mbpp-challenge-tests"] == ['assert remove_occ("hellolloll","l") == "helollol"']
    assert _test_field(1) in items["humanevalplus-test"]


@pytest.mark.parametrize(
    ("kwargs", "msg"),
    [
        ({"corrupt": True}, "SHA-256"),
        ({"drop_col": True}, "lacks column"),
        ({"he_test": ["", _test_field(2)]}, "empty"),
        ({"he_test": [None, _test_field(2)]}, "unexpected type"),
    ],
)
def test_collect_fails_closed(tmp_path, kwargs, msg):
    raw, prov, reg = _fixture(tmp_path, **kwargs)
    with pytest.raises(F.F9Error, match=msg):
        F.collect(raw, prov, reg)


def test_collect_fails_closed_on_missing_file_and_revision_drift(tmp_path):
    raw, prov, reg = _fixture(tmp_path)
    reg["benchmark"][1]["revision"] = "d" * 40
    with pytest.raises(F.F9Error, match="revision"):
        F.collect(raw, prov, reg)
    raw, prov, reg = _fixture(tmp_path / "b")
    prov["mbpp"]["file_hashes"][0]["path"] = "full/missing.parquet"
    with pytest.raises(F.F9Error, match="missing input"):
        F.collect(raw, prov, reg)


def _index(tmp_path):
    raw, prov, reg = _fixture(tmp_path)
    items, report = F.collect(raw, prov, reg)
    d, meta = F.build_index(items, report, tmp_path / "excl")
    return L.Index(d), items, report, meta


def test_build_publishes_by_identity_and_never_overwrites(tmp_path):
    _, items, report, meta = _index(tmp_path)
    d2, meta2 = F.build_index(items, report, tmp_path / "excl")
    assert meta2["index_identity"] == meta["index_identity"]
    other = {**items, "mbppplus-test": [*items["mbppplus-test"], _test_field(9)]}
    _, meta3 = F.build_index(other, report, tmp_path / "excl")
    assert meta3["index_identity"] != meta["index_identity"]
    assert len(list((tmp_path / "excl").iterdir())) == 2
    (d2 / "index.json").write_text(json.dumps({**meta, "index_identity": "f" * 64}))
    with pytest.raises(F.F9Error, match="different index"):
        F.build_index(items, report, tmp_path / "excl")


def test_offline_pass2_equals_index_match(tmp_path):
    idx, items, _, _ = _index(tmp_path)
    rng = random.Random(3)
    texts = ["unrelated text about weather", items["humanevalplus-test"][3], "intro\n" + _test_field(2)[:900]]
    texts += [
        F.plant("a b c\nd e f", F.perturb(rng.choice(items["mbppplus-test"]), "truncated_80", rng), rng)
    ]
    common_sets = [None, np.sort(idx.G[::3]), np.unique(idx.G)]
    for text in texts:
        g = L.grams(text)
        lo = np.searchsorted(idx.G, g, side="left")
        g_in = g[np.searchsorted(idx.G, g, side="right") > lo]
        ids = sorted(L.idents(text) & idx.idents.keys())
        for common in common_sets:
            want = idx.match(text, F.MIN_COV, common)
            got = F.match_grams(idx, g_in, ids, F.MIN_COV, common)
            if want is None:
                assert got is None
            else:
                got.pop("common_only_benchmarks")
                assert got == want
    with pytest.raises(F.F9Error, match="not in index"):
        F.match_grams(idx, np.array([12345], dtype=np.uint64), [], F.MIN_COV, None)


def test_controls_detect_planted_items_and_spare_negatives(tmp_path):
    idx, items, _, _ = _index(tmp_path)
    hosts = ["A plain paragraph about gardens.\nSecond line here.", "Another host document."]
    res = F.controls(idx, items, hosts, ["nothing relevant here at all, just prose"], None, per_label=5)
    for label in ("humanevalplus-test", "mbppplus-test"):
        assert res["positive"][label]["verbatim"]["detected"] == res["positive"][label]["verbatim"]["planted"]
    assert res["negative"]["flagged"] == 0
    with pytest.raises(F.F9Error):
        F.controls(idx, items, [], [], None)


def _shard(tmp_path, docs):
    p = tmp_path / "shard.parquet"
    pq.write_table(pa.table({"doc_id": [d for d, _ in docs], "text": [t for _, t in docs]}), p)
    return p


def test_scan_unit_restart_and_hash_check(tmp_path):
    idx, items, _, meta = _index(tmp_path)
    F._init(str(Path(tmp_path / "excl") / f"layers-v1-f9-{meta['index_identity'][:16]}"))
    shard = _shard(tmp_path, [("d1", "clean prose"), ("d2", "x\n" + items["mbppplus-test"][0])])
    sha = L.sha256_file(shard)
    out = tmp_path / "units"
    out.mkdir()
    r = F.scan_unit((str(shard), str(out), sha, "train", "src"))
    assert r["documents"] == 2 and [h["doc_id"] for h in r["hits_uncalibrated"]] == ["d2"]
    assert [t["doc_id"] for t in r["touched"]] == ["d2"]
    assert F.scan_unit((str(shard), str(out), sha, "train", "src")).get("reused")
    unit = next(out.iterdir())
    unit.write_text(json.dumps({**json.loads(unit.read_text()), "tool_sha256": "0" * 64}))
    assert not F.scan_unit((str(shard), str(out), sha, "train", "src")).get("reused")
    with pytest.raises(F.F9Error, match="accepted manifest"):
        F.scan_unit((str(shard), str(out), "0" * 64, "train", "src"))
    common, hits = F.calibrate(idx, [r])
    assert len(common) == 0 and [h["doc_id"] for h in hits] == ["d2"]


def test_adjudication_rule_and_fail_closed():
    base = {
        "layer_a": {},
        "layer_a_benchmarks": [],
        "layer_a_common_only": {},
        "layer_c": [],
        "split": "train",
        "source_id": "s",
    }
    dec = F.adjudicate(
        [
            {**base, "doc_id": "a", "layer_a": {"1": 0.7}, "layer_a_benchmarks": ["mbppplus-test"]},
            {
                **base,
                "doc_id": "b",
                "layer_a_common_only": {"2": 1.0},
                "common_only_benchmarks": ["mbppplus-test"],
            },
            {**base, "doc_id": "c", "layer_c": ["x86_64-linux-gnu"]},
            {**base, "doc_id": "d", "layer_c": ["x86_64-linux-gnu", "abc123def456ghi789"]},
        ]
    )
    assert [d["decision"] for d in dec] == ["PROHIBITED", "PROHIBITED", "NOT_PROHIBITED", "PROHIBITED"]
    with pytest.raises(F.F9Error, match="empty hit"):
        F.adjudicate([{**base, "doc_id": "e"}])
    with pytest.raises(F.F9Error, match="unknown split"):
        F.adjudicate([{**base, "doc_id": "f", "split": "dev", "layer_c": ["zz"]}])
    with pytest.raises(F.F9Error, match="without"):
        F.adjudicate([{"doc_id": "g"}])


def _report_inputs():
    spec = importlib.util.spec_from_file_location(
        "contamination_f9_report", TOOLS / "contamination_f9_report.py"
    )
    rep = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rep)
    hit = {
        "doc_id": "d1",
        "split": "train",
        "source_id": "s",
        "layer_a": {"1": 0.9},
        "layer_a_benchmarks": ["mbppplus-test"],
        "layer_a_common_only": {},
        "common_only_benchmarks": [],
        "layer_c": [],
        "layer_c_benchmarks": [],
    }
    scan = {
        "index_identity": "i",
        "common_sha256": "c",
        "hit_documents_pass2": 1,
        "accepted_manifest_sha256": "m",
        "build_identity": "b",
        "tool_sha256": "t",
    }
    ctl = {"index_identity": "i", "positive": {}, "negative": {}}
    meta = {"index_identity": "i", "items": 1, "distinct_grams": 1, "identifiers": 0, "benchmarks": {"x": {}}}
    sources = {"labels": {"x": {"files": []}}}
    args = [scan, [hit], F.adjudicate([hit]), {**ctl, "common_sha256": None}, {**ctl, "common_sha256": "c"}]
    return rep, args, [meta, sources, {}, "abc"]


def test_report_assembles_and_fails_closed():
    rep, args, rest = _report_inputs()
    prohibited, summary = rep.assemble(*args, *rest)
    assert prohibited["counts"] == {"train": 1, "validation": 0, "test": 0}
    assert list(prohibited["prohibited"]["train"]["s"].values()) == [["d1"]]
    assert summary["hits_by_label"] == {"mbppplus-test": 1}
    for i, bad in (
        (0, {**args[0], "index_identity": "other"}),
        (2, []),
        (4, {**args[4], "common_sha256": "different"}),
        (3, {**args[3], "common_sha256": "c"}),
        (0, {**args[0], "hit_documents_pass2": 2}),
    ):
        broken = list(args)
        broken[i] = bad
        with pytest.raises(SystemExit):
            rep.assemble(*broken, *rest)
