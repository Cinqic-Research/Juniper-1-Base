import copy
import tomllib
from collections import Counter
from pathlib import Path

import jsonschema
import pyarrow as pa
import pytest

from j1data import splits
from j1data.registry import ADMISSIBLE_STATES, REPO_ROOT, FileSelection, load_registry
from j1data.shards import ShardWriter, verify_files


def test_registry_is_valid_and_complete() -> None:
    registry = load_registry()
    assert len(registry) >= 10
    for source in registry.values():
        assert source.license_state in ADMISSIBLE_STATES
        assert source.raw["license"]["evidence"]
        if source.raw["license"]["state"] != "APPROVED" or source.raw["family"] != "cinqic":
            assert source.raw["license"]["attribution"]


def test_schema_rejects_unknown_license_state() -> None:
    with (REPO_ROOT / "configs" / "sources.toml").open("rb") as handle:
        entry = copy.deepcopy(tomllib.load(handle)["source"][0])
    entry["license"]["state"] = "PUBLIC_SO_FINE"
    schema = __import__("json").loads((REPO_ROOT / "schemas" / "source.schema.json").read_text())
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(entry, schema)


def test_hash_selection_is_order_independent_and_sized() -> None:
    selection = FileSelection(include=("data/*.parquet",), hash_select=((("data/*.parquet",), "ns", 3),))
    paths = [f"data/f{i:03d}.parquet" for i in range(50)] + ["README.md"]
    chosen = selection.apply(paths)
    assert len(chosen) == 3
    assert chosen == selection.apply(list(reversed(paths)))
    other = FileSelection(include=("data/*.parquet",), hash_select=((("data/*.parquet",), "other", 3),))
    assert other.apply(paths) != chosen


def test_splits_are_deterministic_group_level_and_sized() -> None:
    config = splits.SplitConfig("j1.split.v1", 0.01, 0.01)
    groups = [f"host:site{i}.org" for i in range(50000)]
    first = [splits.assign(g, config) for g in groups]
    assert first == [splits.assign(g, config) for g in groups]
    counts = Counter(first)
    assert 400 < counts["validation"] < 600 and 400 < counts["test"] < 600
    assert splits.assign("repo:a/b", config) == splits.assign("repo:a/b", config)
    with pytest.raises(ValueError):
        splits.SplitConfig("x", 0.6, 0.5)


def test_shard_writer_is_atomic_and_verifiable(tmp_path: Path) -> None:
    schema = pa.schema([("doc_id", pa.string()), ("text", pa.large_string())])
    writer = ShardWriter(tmp_path / "s.parquet", schema, rows_per_group=2)
    for i in range(5):
        writer.append({"doc_id": str(i), "text": "x" * i})
    assert not (tmp_path / "s.parquet").exists()
    info = writer.close()
    assert info["rows"] == 5 and not (tmp_path / "s.parquet.tmp").exists()
    assert verify_files(tmp_path, [info]) == []

    data = (tmp_path / "s.parquet").read_bytes()
    (tmp_path / "s.parquet").write_bytes(data[:-10])
    assert verify_files(tmp_path, [info]) == ["sha256 mismatch s.parquet"]
    (tmp_path / "s.parquet").unlink()
    assert verify_files(tmp_path, [info]) == ["missing s.parquet"]


def test_aborted_writer_leaves_nothing(tmp_path: Path) -> None:
    writer = ShardWriter(tmp_path / "a.parquet", pa.schema([("x", pa.int64())]))
    writer.append({"x": 1})
    writer.abort()
    assert list(tmp_path.iterdir()) == []


def test_considered_sources_are_documented_and_disjoint() -> None:
    from j1data.registry import LICENSE_STATES

    with (REPO_ROOT / "configs" / "considered_sources.toml").open("rb") as handle:
        considered = tomllib.load(handle)["considered"]
    admitted = set(load_registry())
    for entry in considered:
        assert {"source_id", "origin", "revision", "state", "reason", "evidence"} <= set(entry)
        assert entry["state"] in LICENSE_STATES and entry["evidence"]
        assert entry["source_id"] not in admitted


def test_signature_sink_transposes_across_block_boundaries(tmp_path: Path) -> None:
    import numpy as np

    from j1data.shards import SignatureSink

    rng = np.random.default_rng(0)
    rows = rng.integers(0, 2**32, size=(SignatureSink.BLOCK * 2 + 17, 112), dtype=np.uint64).astype(np.uint32)
    sink = SignatureSink(tmp_path / "s.npy", 112)
    for row in rows:
        sink.append(row)
    info = sink.close()
    stored = np.load(tmp_path / "s.npy")
    assert info["rows"] == rows.shape[0] and stored.shape == (112, rows.shape[0])
    assert np.array_equal(stored, rows.T)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["s.npy"]
    empty = SignatureSink(tmp_path / "e.npy", 112)
    assert empty.close()["rows"] == 0 and np.load(tmp_path / "e.npy").shape == (112, 0)


def test_shard_writer_cuts_row_groups_by_size(tmp_path: Path) -> None:
    import pyarrow.parquet as pq

    writer = ShardWriter(
        tmp_path / "big.parquet", pa.schema([("text", pa.large_string())]), max_buffer_bytes=1000
    )
    for _ in range(10):
        writer.append({"text": "x" * 600})
    writer.close()
    assert pq.ParquetFile(tmp_path / "big.parquet").metadata.num_row_groups == 5
