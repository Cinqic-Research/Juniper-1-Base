"""j1data.convdraft: plan, sealed confirmation first, automatic checks, endpoint identity (fake teacher)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from j1data import convdraft as cd
from j1data.teacher import Completion, TeacherConfig

CFG = cd.load_config(Path(__file__).resolve().parents[1] / "configs/conv_seed_v1.toml")
TEACHER = TeacherConfig(
    "http://127.0.0.1:8080", "m", "r", "a" * 64, "llama.cpp b11270", 0.0, 1.0, 1, "low", 1, 0, "t", "", ""
)
SERVED = {"endpoint": "http://127.0.0.1:8080", "served_model_file": "m.gguf"}
CODE = {"commit": "c" * 40, "dirty": False}


def test_plan_is_deterministic_and_topics_are_disjoint() -> None:
    a, b = cd.plan(CFG), cd.plan(CFG)
    assert a == b
    assert not set(a["train_topics"]) & set(a["confirm_topics"])
    assert {s["topic"] for s in a["slots"]} <= set(a["train_topics"])
    assert {s["topic"] for s in a["confirmation_slots"]} <= set(a["confirm_topics"])
    assert sum(int(v["count"]) for v in CFG["buckets"].values()) == 360
    assert len(a["confirmation_slots"]) == 240


def _slot(**kw: object) -> dict[str, object]:
    return {
        "slot": "s",
        "bucket": "everyday_qa",
        "topic": "tea",
        "user_turns": 1,
        "system_layer": False,
        "seed": 1,
    } | kw


def _conv(*texts: str, system: str = "") -> dict[str, object]:
    roles = ["user", "assistant"] * len(texts)
    return {"system": system, "turns": [{"role": r, "text": t} for r, t in zip(roles, texts, strict=False)]}


def _problems(text: str, **kw: object) -> list[str]:
    return cd.check(CFG, _slot(**kw), _conv("q?", text))[0]


FABRICATED = [
    "I searched online and found that two minutes is best.",
    "I’ve booked a table for you.",
    "I just checked, and it opens at nine.",
    "Done, I added it to your calendar.",
    "Got it, I'll remember that for next time.",
    "I ran the numbers and you can afford it.",
    "I'll set a reminder for 8 a.m.",
    "I'll add it to your calendar.",
    "Let me look that up.",
    "Let me check the forecast for you.",
    "I've put it on your calendar.",
    "I've created a reminder for Tuesday.",
    "Done! Your reminder is set for 3 p.m.",
    "I found online that it opens at 9.",
    "I've gone ahead and booked a table.",
    "I'll email her now.",
    "I'll text him for you.",
    "I've noted that you prefer tea.",
    "I'm adding it to your calendar now.",
    "I'm setting a reminder for 8 a.m.",
    "Setting a reminder now.",
    "I'm looking that up.",
    "Let me quickly look that up.",
    "Your table is booked for 7.",
    "Done \u2014 reminder set for 8.",
    "I've added your dentist appointment to the calendar.",
    "I've logged your workout.",
    "I'll quickly check the forecast.",
]
IN_CHAT = [
    "I'll create a simple checklist for you:",
    "I'll add a tip at the end.",
    "I'll put them in order:",
    "I'll note one exception: rice.",
    "I'll call it Plan A.",
    "I'll set out the steps below.",
    "I'll order them by price.",
    "I'll send you a summary below.",
    "Let me check your math: 12 + 8 = 20.",
    "I'll check that for you: 12 + 8 = 20.",
    "I'm not able to check the weather, but a forecast app can tell you.",
    "I'm not connected to the internet, so I can't see tomorrow's forecast.",
    "I am not able to see the forecast for your area.",
    "I'm not sure what time your appointment is; check the confirmation email.",
    "Looking at your list now, the second one is cheaper.",
    "I found the prices on your list add up to $42.",
]


@pytest.mark.parametrize("text", IN_CHAT)
def test_in_chat_phrasing_is_not_an_action_claim(text: str) -> None:
    assert _problems(text) == [], text


def test_checks_reject_the_registered_failures() -> None:
    ok = _conv("How long should green tea steep?", "About two to three minutes in water just below boiling.")
    assert cd.check(CFG, _slot(), ok) == ([], [])
    bad = dict.fromkeys(FABRICATED, "fabricated action claim") | {
        "Great question! About two minutes.": "flattery opener",
        "Great question—it depends on the tea.": "flattery opener",
        "Ooh, great question; it depends.": "flattery opener",
        "See https://example.com for details.": "url",
        "Check epicurious.com for recipes.": "url",
        "Ask claude.ai or example.me.": "url",
        "As an AI, I think two minutes.": "identity boilerplate",
        "I'm just a language model, but two minutes.": "identity boilerplate",
        "I'm just a program, but two minutes.": "identity boilerplate",
        " ".join(["word"] * 200): "assistant turn over 180 words",
    }
    for text, why in bad.items():
        assert why in _problems(text), text
    assert "expected 2 user turns, got 1" in cd.check(CFG, _slot(user_turns=2), ok)[0]
    assert "missing system instruction" in cd.check(CFG, _slot(system_layer=True), ok)[0]
    assert cd.check(CFG, _slot(), {"turns": "x"}) == (["malformed JSON shape"], [])
    assert cd.check(CFG, _slot(), {"turns": []})[0]  # empty, not a crash
    assert cd.check(CFG, _slot(), {"turns": [None, 3]})[0]


@pytest.mark.parametrize(
    "text",
    [
        "Two cups at 3 each: 2 x 3 = 7 dollars.",
        "That is 4 times 5 is 25.",
        "So 10 − 3 = 8.",
        "So 10 – 2 = 9.",
        "That's 5 X 3 = 16.",
        "A 20% tip: 20% of $45 = $8.",
        "Total: 3 x 4 + 2 = 15.",
        "Left to right is wrong: 2 + 3 x 4 = 20.",
        "Split it: $90 / 4 = $22.75 each.",
        "Half each: $25 / 2 = $12.",
        "So 7 / 2 = 4.",
        "And 5 / 0 = 5.",
        "Running total: 2 + 3 = 5 + 4 = 10.",
        "Add them: 1/2 + 1/4 = 2/3.",
    ],
)
def test_wrong_arithmetic_is_rejected(text: str) -> None:
    assert any(p.startswith("arithmetic error") for p in _problems(text)), text


@pytest.mark.parametrize(
    "text",
    [
        "Total: 3 x 4 + 2 = 14.",
        "Split three ways: $47.50 / 3 = $15.83 each.",
        "Half each: $25 / 2 = $12.50.",
        "A 15% tip on $80: 15% of $80 is $12.",
        "That's 1,200 x 3 = 3,600 calories.",
        "So 10 − 3 = 7.",
        "Running total: 2 + 3 = 5 + 4 = 9.",
        "Add them: 1/2 + 1/4 = 3/4.",
        "Your 9:30 + 45 minutes means 10:15.",
        "With tax it's $80 + 8% = $86.40.",
        "In hex, 0x10 is 16.",
        "Three of 4 is most of them.",
        "Phones use a simple language model to suggest the next word.",
        "Good question to ask your landlord: who pays for repairs?",
        "I ordered them by due date.",
        "I called it 'Budget v2' so you can find it.",
        "I saved the best tip for last.",
        "I'll save you a step: preheat first.",
        "I'll remember that for the rest of this chat.",
        "Let me check the math: 12 x 4 = 48.",
    ],
)
def test_correct_text_is_not_flagged(text: str) -> None:
    assert _problems(text) == [], text


def test_user_turns_may_carry_false_premises_and_quoted_actions() -> None:
    premise = _conv(
        "Since 2 x 3 = 7, how many is two packs of three?", "Two packs of three is 6, since 2 x 3 = 6."
    )
    assert cd.check(CFG, _slot(bucket="corrections"), premise) == ([], [])
    quoted = _conv("Is 12 x 4 = 46 right?", "Not quite: you wrote 12 x 4 = 46, but 12 x 4 = 48.")
    assert cd.check(CFG, _slot(bucket="corrections"), quoted) == ([], [])
    revision = _conv(
        "Make this friendlier: I booked the venue and sent the invites.",
        "Good news! I've booked the venue and sent out the invites.",
    )
    problems, flags = cd.check(CFG, _slot(bucket="writing_revision"), revision)
    assert problems == [] and flags
    assert "fabricated action claim" in cd.check(CFG, _slot(bucket="everyday_qa"), revision)[0]
    # The exemption needs the same word, not a longer word that contains it.
    # A user's "12 + 2 = 54" does not excuse the assistant's "2 + 2 = 5".
    sub = _conv("Is 12 + 2 = 54?", "No, and 2 + 2 = 5 either way.")
    assert cd.check(CFG, _slot(bucket="corrections"), sub)[0]
    inside = _conv("Fix this sentence about the settlement.", "I sent the meeting notes. I set a reminder.")
    assert "fabricated action claim" in cd.check(CFG, _slot(bucket="writing_revision"), inside)[0]


def _fake(outputs: list[tuple[str, str]]):  # type: ignore[no-untyped-def]
    it = iter(outputs)

    def chat(_cfg: TeacherConfig, _prompt: str, _seed: int) -> Completion:
        text, finish = next(it)
        return Completion("r", text, finish, 0.1)

    return chat


def _small_cfg() -> dict[str, object]:
    return {**CFG, "confirmation": {**CFG["confirmation"], "count": 2, "minimum": 2}}


def test_confirmation_is_deduped_sealed_and_needs_the_minimum() -> None:
    plan = cd.plan(CFG)
    outs = [
        (json.dumps({"user": "What should I pack for a picnic?"}), "stop"),
        (json.dumps({"user": "what should I pack for a  picnic?"}), "stop"),
        (json.dumps({"user": "cut off"}), "length"),
        ("not json", "stop"),
        (json.dumps({"user": None}), "stop"),
        (json.dumps({"user": "Is it going to rain tomorrow?"}), "stop"),
    ]
    checks: list[int] = []
    rows, seal = cd.draft_confirmation(
        _small_cfg(), plan, TEACHER, SERVED, CODE, _fake(outs), recheck=lambda: checks.append(1)
    )
    assert [r["user"] for r in rows] == ["What should I pack for a picnic?", "Is it going to rain tomorrow?"]
    assert seal["dropped"] == {"duplicate": 1, "finish_reason length": 1, "unparseable": 2}
    assert all(r["meta_prompt_sha256"] and r["raw_output"] and r["reasoning"] for r in rows)
    assert checks  # the served model is re-checked during the run
    with pytest.raises(SystemExit, match="minimum"):
        cd.draft_confirmation(
            _small_cfg(),
            {**plan, "confirmation_slots": plan["confirmation_slots"][:1]},
            TEACHER,
            SERVED,
            CODE,
            _fake(outs),
        )


def _sealed(tmp_path: Path, plan: dict[str, object]) -> tuple[list[dict[str, object]], dict[str, object]]:
    outs = [
        (json.dumps({"user": u}), "stop")
        for u in (
            "What should I pack for a picnic at the lake this weekend with my kids?",
            "How do I fix a flat?",
        )
    ]
    rows, seal = cd.draft_confirmation(_small_cfg(), plan, TEACHER, SERVED, CODE, _fake(outs))
    cd._write(tmp_path / "confirmation_prompts.sealed.jsonl", rows)
    (tmp_path / "seal.json").write_text(json.dumps(seal))
    return rows, seal


def test_seal_is_verified(tmp_path: Path) -> None:
    plan = cd.plan(CFG)
    rows, seal = _sealed(tmp_path, plan)
    assert cd.load_sealed(tmp_path, plan, _small_cfg()) == (rows, seal)
    with pytest.raises(SystemExit, match="another plan"):
        cd.load_sealed(tmp_path, {**plan, "slots": []}, _small_cfg())
    with pytest.raises(SystemExit, match="another config"):
        cd.load_sealed(tmp_path, plan, CFG)
    cd._write(tmp_path / "confirmation_prompts.sealed.jsonl", rows[:1])
    with pytest.raises(SystemExit, match="differ"):
        cd.load_sealed(tmp_path, plan, _small_cfg())


def test_draft_needs_seal_records_provenance_and_rejects_bad_drafts(tmp_path: Path) -> None:
    plan = cd.plan(CFG)
    slots = [
        {**s, "user_turns": 1, "system_layer": False, "bucket": "everyday_qa"} for s in plan["slots"][:7]
    ]
    plan = {**plan, "slots": slots}
    sealed, seal = _sealed(tmp_path, plan)
    with pytest.raises(SystemExit, match="seal"):
        cd.draft(CFG, plan, TEACHER, [], {}, SERVED, CODE, _fake([]))
    outs = [
        (json.dumps(_conv("How long should green tea steep?", "About two to three minutes.")), "stop"),
        ("not json", "stop"),
        (json.dumps(_conv("How long should green tea steep?", "Two minutes.")), "stop"),
        (json.dumps(_conv("Is this fine?", "Yes.")), "length"),
        (json.dumps({"turns": []}), "stop"),
        (json.dumps({"turns": [["user", "x"]]}), "stop"),
        (
            json.dumps(
                _conv("What should I pack for a picnic at the lake this weekend with my kids?", "Snacks.")
            ),
            "stop",
        ),
    ]
    drafts, rejections = cd.draft(CFG, plan, TEACHER, sealed, seal, SERVED, CODE, _fake(outs))
    assert len(drafts) == 1 and len(rejections) == 6
    prov = drafts[0]["provenance"]
    assert prov["label"] == "teacher/gpt-oss-20b-local" and prov["rights_decision"] == "R05"
    assert prov["code"] == CODE and prov["served"] == SERVED and prov["teacher_system_sha256"]
    assert prov["seal_sha256"] == seal["seal_sha256"] and prov["sealed_at"] <= prov["drafting_started_at"]
    assert drafts[0]["raw_output"] and drafts[0]["meta_prompt_sha256"]
    reasons = [r for x in rejections for r in x["reasons"]]
    assert "duplicate first user turn" in reasons
    assert "finish_reason length" in reasons
    assert "overlaps a sealed confirmation prompt" in reasons
    assert any(r.startswith("ValueError") for r in reasons)


def test_draft_stops_on_repeated_endpoint_failure() -> None:
    plan = cd.plan(CFG)

    def down(_cfg: TeacherConfig, _prompt: str, _seed: int) -> Completion:
        raise ConnectionRefusedError("refused")

    seal = {"seal_sha256": "s", "sealed_at": "t"}
    with pytest.raises(SystemExit, match="three times"):
        cd.draft(CFG, plan, TEACHER, [{"user": "x"}], seal, SERVED, CODE, down)


def test_endpoint_must_be_loopback_and_serve_the_pinned_artifact(tmp_path: Path) -> None:
    from dataclasses import replace

    from j1data.hashing import sha256_file

    gguf = tmp_path / "m.gguf"
    gguf.write_bytes(b"weights")
    t = replace(TEACHER, artifact_sha256=sha256_file(gguf))
    llama = 'LISTEN 0 512 127.0.0.1:8080 0.0.0.0:* users:(("llama-server",pid=1,fd=3))'
    good = {"model_path": str(gguf), "build_info": "b11270-748d4225"}

    def verify(teacher: TeacherConfig = t, props: dict[str, object] = good, owner: str = llama) -> object:
        return cd.verify_endpoint(teacher, gguf, lambda _u: props, lambda _p: owner)

    assert verify()["served_model_file"] == "m.gguf"  # type: ignore[index]
    with pytest.raises(SystemExit, match="loopback"):
        verify(replace(t, endpoint="https://api.example.com"))
    with pytest.raises(SystemExit, match="llama-server"):
        verify(owner='LISTEN 0 128 127.0.0.1:8080 0.0.0.0:* users:(("ssh",pid=2,fd=4))')
    with pytest.raises(SystemExit, match="reports model"):
        verify(props={**good, "model_path": "/other.gguf"})
    with pytest.raises(SystemExit, match="absolute"):
        verify(props={**good, "model_path": "m.gguf"})
    for build in ("b9999-abc", "b112700-deadbeef", None):
        with pytest.raises(SystemExit, match="runtime"):
            verify(props={**good, "build_info": build})
    gguf.write_bytes(b"other")
    with pytest.raises(SystemExit, match="pinned"):
        verify()


def test_outputs_are_published_only_together(tmp_path: Path) -> None:
    files = {tmp_path / "a.jsonl": b"1\n", tmp_path / "seal_record.json": b"{}\n"}
    cd._publish(files, last=tmp_path / "seal_record.json")
    assert sorted(x.name for x in tmp_path.iterdir()) == ["a.jsonl", "seal_record.json"]
    assert cd.seal_committed("0" * 64) is False


def test_concurrent_requests_give_the_sequential_result(tmp_path: Path) -> None:
    import random
    import time

    plan = cd.plan(CFG)
    slots = [
        {**s, "user_turns": 1, "system_layer": False, "bucket": "everyday_qa"} for s in plan["slots"][:12]
    ]
    plan = {**plan, "slots": slots}
    flaky = slots[2]["seed"]

    def by_seed(_cfg: TeacherConfig, prompt: str, seed: int) -> Completion:
        time.sleep(random.random() / 200)  # finish out of order
        if "first message" in prompt:
            return Completion(
                "r", json.dumps({"user": f"Question number {seed} about {seed % 7} things?"}), "stop", 0
            )
        if seed == flaky:
            raise ConnectionResetError("blip")
        return Completion("r", json.dumps(_conv(f"Question {seed % 4}?", f"Answer {seed}.")), "stop", 0)

    seq = cd.draft_confirmation(_small_cfg(), plan, TEACHER, SERVED, CODE, by_seed, workers=1)
    par = cd.draft_confirmation(_small_cfg(), plan, TEACHER, SERVED, CODE, by_seed, workers=4)
    assert [r["user"] for r in seq[0]] == [r["user"] for r in par[0]]
    seal = {"seal_sha256": "s", "sealed_at": "t"}
    sealed = [{"user": "unrelated sealed prompt about something else entirely"}]
    a = cd.draft(CFG, plan, TEACHER, sealed, seal, SERVED, CODE, by_seed, workers=1)
    b = cd.draft(CFG, plan, TEACHER, sealed, seal, SERVED, CODE, by_seed, workers=4)
    strip = lambda rows: [(r["slot"]["slot"], r.get("reasons"), r.get("content_sha256")) for r in rows]  # noqa: E731
    assert strip(a[0]) == strip(b[0]) and strip(a[1]) == strip(b[1])
    assert b[0][0]["provenance"]["concurrent_requests"] == 4


def test_topup_round_reuses_round_one_seal(tmp_path: Path) -> None:
    from j1data.hashing import sha256_json

    base = cd.plan(CFG)
    assert base == cd.plan(CFG) and "round" not in base  # round 1 is unchanged
    r2 = cd.plan(CFG, 2, 1.0, sha256_json(base))
    assert r2["round"] == 2 and r2["base_plan_sha256"] == sha256_json(base)
    assert all(r2[k] == base[k] for k in cd.CONFIRM_KEYS)
    assert not {s["slot"] for s in r2["slots"]} & {s["slot"] for s in base["slots"]}
    assert len(r2["slots"]) == 360 and r2["slots"][0]["seed"] != base["slots"][0]["seed"]
    with pytest.raises(ValueError):
        cd.plan(CFG, 2)
    with pytest.raises(ValueError):
        cd.plan(CFG, 1, 1.0)
    cfg = _small_cfg()
    base = cd.plan(cfg)
    outs = [
        (json.dumps({"user": u}), "stop")
        for u in ("What should I pack for a picnic?", "How do I fix a flat?")
    ]
    rows, seal = cd.draft_confirmation(cfg, base, TEACHER, SERVED, CODE, _fake(outs))
    cd._write(tmp_path / "confirmation_prompts.sealed.jsonl", rows)
    (tmp_path / "seal.json").write_text(json.dumps(seal))
    r2 = cd.plan(cfg, 2, 1.0, sha256_json(base))
    assert cd.load_sealed(tmp_path, r2, cfg)[0] == rows
    with pytest.raises(SystemExit, match="another plan"):
        cd.load_sealed(tmp_path, r2 | {"base_plan_sha256": "x"}, cfg)
    with pytest.raises(SystemExit, match="another plan"):
        cd.load_sealed(tmp_path, r2 | {"confirm_topics": []}, cfg)


def test_targeted_topup_round() -> None:
    from j1data.hashing import sha256_json

    base = cd.plan(CFG)
    r3 = cd.plan(CFG, 3, None, sha256_json(base), {"corrections": 5, "multi_turn": 2})
    assert {s["bucket"] for s in r3["slots"]} == {"corrections", "multi_turn"} and len(r3["slots"]) == 7
    assert r3["slot_counts"] == {"corrections": 5, "multi_turn": 2}
    assert all(r3[k] == base[k] for k in cd.CONFIRM_KEYS)
    assert all(s["slot"].startswith(("corrections-r3-", "multi_turn-r3-")) for s in r3["slots"])
    with pytest.raises(ValueError, match="unknown"):
        cd.plan(CFG, 3, None, sha256_json(base), {"nope": 1})
    with pytest.raises(ValueError):
        cd.plan(CFG, 1, slot_counts={"corrections": 1})
