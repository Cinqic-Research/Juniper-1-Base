"""Strengthened contamination layers: coverage over non-common grams, common-only reporting, controls."""

import importlib.util
import sys
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location(
    "contamination_layers", Path(__file__).resolve().parents[1] / "tools" / "contamination_layers.py"
)
L = importlib.util.module_from_spec(spec)
sys.modules["contamination_layers"] = L
spec.loader.exec_module(L)

ITEM = "Write a function to group a sequence of key value pairs into a dictionary of lists using python code"
BOILER = 'if __name__ == "__main__":\n    main_function_entry(argument_one, argument_two)\n'


def _index(tmp_path):
    items = {"mbpp": [ITEM], "code": [BOILER], "ids": ["run id 3f9a1c2b7d8e4f60a1b2 recorded"]}
    L.build(items, tmp_path / "idx", "fp")
    return L.Index(tmp_path / "idx")


def test_verbatim_and_transformed_copies_match(tmp_path):
    idx = _index(tmp_path)
    for text in (ITEM, ITEM.upper(), "Intro.\n\n" + ITEM.replace(" ", "   ") + "\n\nOutro."):
        m = idx.match(text, 0.6)
        assert m and m["layer_a_benchmarks"] == ["mbpp"]
    assert idx.match("an unrelated document about the weather and the sea", 0.6) is None
    assert idx.match("see 3F9A1C2B7D8E4F60A1B2 above", 0.6)["layer_c_benchmarks"] == ["ids"]


def test_coverage_is_over_non_common_grams_and_all_common_items_are_reported(tmp_path):
    idx = _index(tmp_path)
    item_grams = L.grams(ITEM)
    # Half of the grams are "common": a copy still covers 100 % of the non-common grams.
    common = np.sort(item_grams[: len(item_grams) // 2])
    m = idx.match(ITEM, 0.6, common)
    assert m and list(m["layer_a"].values()) == [1.0]
    # Every gram of the boilerplate item is common: it must surface as common_only, never silently vanish.
    all_common = np.sort(np.concatenate([L.grams(BOILER), common]))
    m2 = idx.match(BOILER, 0.6, all_common)
    assert m2 and m2["layer_a"] == {} and list(m2["layer_a_common_only"].values()) == [1.0]


def test_grams_require_three_alphanumeric_tokens():
    assert len(L.grams("_ _ _ _ _ _ _ _ _ _ _ _ _ _")) == 0
    assert len(L.grams("a b c d e f g h i j k")) > 0
