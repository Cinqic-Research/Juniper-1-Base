"""Distillation v2 arm matching (j1data.distill.arm_items) on synthetic records."""

import json

import pyarrow as pa
import pyarrow.parquet as pq

from j1data.distill import arm_items


def test_only_problems_with_every_arm_are_matched(tmp_path):
    rows = [
        {
            "question": "Ann has 3 bags of 4 apples. How many apples?",
            "answer": "3*4=<<3*4=12>>12 apples.\n#### 12",
        },
        {
            "question": "Bob has 10 pens and gives 4 away. How many?",
            "answer": "10-4=<<10-4=6>>6 pens.\n#### 6",
        },
        {"question": "Cat has 2 boxes of 5. How many?", "answer": "2*5=<<2*5=10>>10.\n#### 10"},
    ]
    gsm = tmp_path / "train-00000-of-00001.parquet"
    pq.write_table(pa.Table.from_pylist(rows), gsm)
    controls = tmp_path / "controls.jsonl"
    controls.write_text(
        "".join(
            json.dumps({"prompt_id": f"p{i}", "source_record": f"gsm8k/main/train/{i}"}) + "\n"
            for i in range(3)
        )
    )
    outs = ["She has 3 * 4 = 12 apples.\n#### 12", "Let x = 6, so x = 6.\n#### 6", "2 * 5 = 10\n#### 10"]
    records = tmp_path / "records.jsonl"
    records.write_text(
        "".join(
            json.dumps(
                {
                    "record_id": f"r{i}",
                    "prompt": {"prompt_id": f"p{i}", "oracle": {"answer": a}},
                    "decision": {"status": "accepted", "selected_index": 0},
                    "candidates": [{"output": o}],
                }
            )
            + "\n"
            for i, (o, a) in enumerate(zip(outs, ["12", "6", "10"], strict=True))
        )
    )
    items, ledger = arm_items(records, controls, gsm)
    assert [e["matched"] for e in ledger] == [True, False, True]  # p1's rationale is algebra: unverifiable
    assert [len(items[a]) for a in ("C0", "A1", "A2", "A3")] == [2, 2, 2, 2]
    assert [it.source_record for it in items["C0"]] == [it.source_record for it in items["A3"]]
    assert items["A1"][0].turns[-1].text == "#### 12"
    assert items["A3"][0].turns[-1].text == "3 * 4 = 12\n#### 12"
    assert items["C0"][0].origin == "human" and items["A2"][0].origin == "teacher"
