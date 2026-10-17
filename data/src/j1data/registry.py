"""Source registry: what may enter the corpus, under which terms, and how.

The registry (configs/sources.toml) is validated against
schemas/source.schema.json on every load. Only sources whose license state is
in ``ADMISSIBLE_STATES`` can be processed; ``REVIEW_REQUIRED`` and ``REJECTED``
entries stay in the registry as a record of what was considered.
"""

from __future__ import annotations

import fnmatch
import json
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jsonschema

from j1data.hashing import unit_interval

LICENSE_STATES = (
    "APPROVED",
    "APPROVED_WITH_ATTRIBUTION",
    "TRAINING_ONLY_NO_REDISTRIBUTION",
    "REVIEW_REQUIRED",
    "REJECTED",
)
ADMISSIBLE_STATES = frozenset(LICENSE_STATES[:3])

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class FileSelection:
    include: tuple[str, ...]
    exclude: tuple[str, ...] = ()
    hash_select: tuple[tuple[tuple[str, ...], str, int], ...] = ()

    def apply(self, paths: list[str]) -> list[str]:
        """Filter upstream paths. Each ``hash_select`` entry (patterns, namespace, count) keeps
        the ``count`` matching files with the smallest keyed hash: an unbiased, reproducible
        subset that does not depend on listing order."""
        chosen = [
            p
            for p in sorted(paths)
            if any(fnmatch.fnmatch(p, pat) for pat in self.include)
            and not any(fnmatch.fnmatch(p, pat) for pat in self.exclude)
        ]
        for patterns, namespace, count in self.hash_select:
            group = [p for p in chosen if any(fnmatch.fnmatch(p, pat) for pat in patterns)]
            keep = set(sorted(group, key=lambda p: unit_interval(namespace, p))[:count])
            chosen = [p for p in chosen if p not in group or p in keep]
        return chosen


@dataclass(frozen=True)
class Source:
    source_id: str
    raw: dict[str, Any]

    @property
    def license_state(self) -> str:
        return str(self.raw["license"]["state"])

    @property
    def admissible(self) -> bool:
        return self.license_state in ADMISSIBLE_STATES

    @property
    def adapter(self) -> str:
        return str(self.raw["acquisition"]["adapter"])

    @property
    def priority(self) -> int:
        return int(self.raw["processing"]["priority"])

    @property
    def source_kind(self) -> str | None:
        kind = self.raw["processing"].get("kind")
        return str(kind) if kind else None

    @property
    def options(self) -> dict[str, Any]:
        return dict(self.raw["processing"].get("options", {}))

    @property
    def repo(self) -> str:
        return str(self.raw["acquisition"]["repo"])

    @property
    def revision(self) -> str:
        return str(self.raw["acquisition"]["revision"])

    def selection(self) -> FileSelection:
        files = self.raw["acquisition"]["files"]
        return FileSelection(
            include=tuple(files["include"]),
            exclude=tuple(files.get("exclude", [])),
            hash_select=tuple(
                (tuple(entry["patterns"]), entry["namespace"], int(entry["count"]))
                for entry in files.get("hash_select", [])
            ),
        )


def load_registry(path: Path | None = None) -> dict[str, Source]:
    path = path or REPO_ROOT / "configs" / "sources.toml"
    with path.open("rb") as handle:
        data = tomllib.load(handle)
    schema = json.loads((REPO_ROOT / "schemas" / "source.schema.json").read_text())
    sources: dict[str, Source] = {}
    for entry in data["source"]:
        jsonschema.validate(entry, schema)
        if entry["source_id"] in sources:
            raise ValueError(f"duplicate source_id {entry['source_id']}")
        sources[entry["source_id"]] = Source(entry["source_id"], entry)
    priorities = [s.priority for s in sources.values() if s.admissible]
    if len(priorities) != len(set(priorities)):
        raise ValueError(
            "admissible sources must have distinct priorities (dedup retention is priority-ordered)"
        )
    return sources
