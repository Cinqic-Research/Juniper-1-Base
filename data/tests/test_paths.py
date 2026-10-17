from pathlib import Path

import pytest

from j1data.paths import safe_child


def test_safe_child_accepts_relative_path_inside_root(tmp_path: Path) -> None:
    root = tmp_path / "raw"
    root.mkdir()
    assert safe_child(root, "subset/file.parquet") == root / "subset" / "file.parquet"


@pytest.mark.parametrize("path", ["../outside", "/etc/passwd", "subset/../../outside", "subset\\..\\outside"])
def test_safe_child_rejects_absolute_and_traversal_paths(tmp_path: Path, path: str) -> None:
    with pytest.raises(ValueError, match="unsafe relative path"):
        safe_child(tmp_path / "raw", path)


def test_safe_child_rejects_symlink_escape(tmp_path: Path) -> None:
    root = tmp_path / "raw"
    root.mkdir()
    target = tmp_path / "outside"
    target.mkdir()
    (root / "link").symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError, match="path escapes data root"):
        safe_child(root, "link/file.parquet")


def test_safe_child_rejects_internal_symlink(tmp_path: Path) -> None:
    root = tmp_path / "raw"
    root.mkdir()
    (root / "real").mkdir()
    (root / "real" / "file.parquet").write_bytes(b"data")
    (root / "alias").symlink_to(root / "real", target_is_directory=True)

    with pytest.raises(ValueError, match="symlink"):
        safe_child(root, "alias/file.parquet")
