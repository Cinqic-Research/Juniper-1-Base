"""Data-root layout.

All persistent identities are logical (source id, revision, unit id); absolute
paths appear only at run time and are never written into manifests. The data
root comes from ``J1_DATA_ROOT``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath


def safe_child(root: Path, relative: str) -> Path:
    """Resolve an upstream path, rejecting traversal and symlinked components."""
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise ValueError(f"unsafe relative path: {relative!r}")
    posix = PurePosixPath(relative)
    windows = PureWindowsPath(relative)
    if (
        not posix.parts
        or posix.is_absolute()
        or windows.is_absolute()
        or windows.drive
        or ".." in posix.parts
    ):
        raise ValueError(f"unsafe relative path: {relative!r}")
    base = root.resolve()
    candidate = base.joinpath(*posix.parts)
    resolved = candidate.resolve()
    try:
        resolved.relative_to(base)
    except ValueError as error:
        raise ValueError(f"path escapes data root: {relative!r}") from error
    current = base
    for part in posix.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"symlink component in relative path: {relative!r}")
    return resolved


@dataclass(frozen=True)
class DataLayout:
    root: Path

    @staticmethod
    def from_env() -> DataLayout:
        value = os.environ.get("J1_DATA_ROOT")
        if not value:
            raise SystemExit("J1_DATA_ROOT is not set (point it at the large-storage data root)")
        return DataLayout(Path(value))

    def raw_dir(self, source_id: str, revision: str) -> Path:
        return self.root / "raw" / source_id / revision

    def acquisition_manifest(self, source_id: str) -> Path:
        return self.root / "source_manifests" / f"{source_id}.acquisition.json"

    def work(self, build: str) -> Path:
        return self.root / "work" / build

    def exclusions(self) -> Path:
        return self.root / "exclusions"

    def accepted(self, release: str) -> Path:
        return self.root / "accepted" / release

    def measurement_tokenizers(self, release: str) -> Path:
        return self.root / "measurement_tokenizers" / release

    def token_counts(self, release: str) -> Path:
        return self.root / "token_counts" / release

    def tokenizer_stage(self, release: str) -> Path:
        """Prepared inputs (document sample, probes) for the separate tokenizer stage."""
        return self.root / "tokenizer_stage" / release

    def reports(self, release: str) -> Path:
        return self.root / "reports" / release

    def relative(self, path: Path) -> str:
        return path.relative_to(self.root).as_posix()
