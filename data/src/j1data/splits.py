"""Deterministic split assignment from stable group identity.

A document's split depends only on its group key (web host, repository,
textbook, question URL ...), hashed with a fixed namespace, so related
documents always land together and the assignment is reproducible without any
global state or ordering.
"""

from __future__ import annotations

from dataclasses import dataclass

from j1data.hashing import unit_interval

VERSION = "j1.split.v1"
SPLITS = ("train", "validation", "test")


@dataclass(frozen=True)
class SplitConfig:
    namespace: str
    validation_fraction: float
    test_fraction: float

    def __post_init__(self) -> None:
        if not 0 <= self.validation_fraction + self.test_fraction < 1:
            raise ValueError("held-out fractions must sum to less than 1")


def assign(group_key: str, config: SplitConfig) -> str:
    point = unit_interval(config.namespace, group_key)
    if point < config.validation_fraction:
        return "validation"
    if point < config.validation_fraction + config.test_fraction:
        return "test"
    return "train"
