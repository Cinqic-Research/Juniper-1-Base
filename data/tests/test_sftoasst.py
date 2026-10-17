"""OASST1 source (j1data.sftoasst): path selection, filters, trimming, splits, pins and the review sample."""

from __future__ import annotations

import hashlib
import json
import shutil
import tomllib
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from j1data import jtok, sftoasst
from j1data.protocol import Message, serialize
from j1data.sftgen import Item

REPO = Path(__file__).resolve().parents[1]
CFG = REPO / "configs/sft_oasst_v1.toml"
REV = "fdf72ae0827c1cda404aff25b6603abec9e3399b"
GOOD = {"quality": 0.8, "helpfulness": 0.8, "spam": 0.0, "toxicity": 0.0, "fails_task": 0.0}
LONG = (
    "Start with a quiet room and a clear goal for the session. Read the passage slowly, then write down "
    "the parts that felt confusing and ask yourself why they felt that way. Short breaks help you keep "
    "your attention, and a friendly tone with yourself makes the practice easier to keep up over time."
)


@pytest.fixture(scope="module")
def tokenizer():
    return jtok.train(
        jtok.CandidateSpec("oasst-fixture", 448, limit_alphabet=200, min_frequency=1, stem_alphabet=False),
        [LONG, "How do I stay calm before a talk? Breathe slowly and practice with a friend."] * 50,
    )


def msg(mid: str, parent: str | None, role: str, text: str, tree: str, **kw: Any) -> dict[str, Any]:
    labels = kw.pop("labels", GOOD if role == "assistant" else {"quality": 0.7, "spam": 0.0})
    row = {
        "message_id": mid,
        "parent_id": parent,
        "user_id": kw.pop("user", f"u-{mid}"),
        "created_date": "2023-02-01T00:00:00+00:00",
        "text": text,
        "role": role,
        "lang": "en",
        "review_count": 3,
        "review_result": True,
        "deleted": False,
        "rank": 0 if role == "assistant" else None,
        "synthetic": False,
        "model_name": None,
        "detoxify": None,
        "message_tree_id": tree,
        "tree_state": "ready_for_export",
        "emojis": None,
        "labels": {
            "name": list(labels),
            "value": [float(v) for v in labels.values()],
            "count": [3] * len(labels),
        },
    }
    row.update(kw)
    return row


def tree(tid: str, prompt: str, reply: str, **reply_kw: Any) -> list[dict[str, Any]]:
    return [
        msg(f"{tid}-p", None, "prompter", prompt, tid),
        msg(f"{tid}-a", f"{tid}-p", "assistant", reply, tid, **reply_kw),
    ]


def fixture_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    rows += tree("t-ok", "How can I focus better when I study?", LONG)
    # Multi-turn: rank-1 sibling ignored; two follow-ups, the higher-quality one is taken; third pair trims.
    rows += tree(
        "t-multi", "Can you help me write a short poem about rain?", "Rain taps softly on the window glass."
    )
    rows.append(msg("t-multi-a1", "t-multi-p", "assistant", "Rank one reply.", "t-multi", rank=1))
    rows.append(
        msg(
            "t-multi-f1",
            "t-multi-a",
            "prompter",
            "Make it longer please.",
            "t-multi",
            labels={"quality": 0.4},
        )
    )
    rows.append(
        msg("t-multi-f2", "t-multi-a", "prompter", "Now make it rhyme.", "t-multi", labels={"quality": 0.9})
    )
    rows.append(msg("t-multi-f1a", "t-multi-f1", "assistant", "Unused branch.", "t-multi"))
    rows.append(
        msg("t-multi-f2a", "t-multi-f2", "assistant", "Rain on the pane, again and again.", "t-multi")
    )
    rows.append(msg("t-multi-f3", "t-multi-f2a", "prompter", "Thanks! One more?", "t-multi"))
    rows.append(
        msg("t-multi-f3a", "t-multi-f3", "assistant", "Sure.", "t-multi", labels={**GOOD, "quality": 0.2})
    )
    rows += [msg("t-es-p", None, "prompter", "Hola, como estas?", "t-es", lang="es")]
    rows += tree("t-del", "Tell me a story.", "Once upon a time.", deleted=True)
    rows += tree("t-rank1", "Tell me a joke.", "Why did the chicken cross the road?", rank=1)
    rows += tree("t-spam", "Give me ideas for a party.", "Balloons and music.", labels={**GOOD, "spam": 0.34})
    rows += tree("t-lowq", "Describe a sunset.", "Orange.", labels={**GOOD, "quality": 0.45})
    rows += tree("t-ai", "Who are you?", "As an AI language model, I do not have a name.")
    rows += tree("t-auth", "Write a haiku about snow.", "Snow falls without sound.", user="bad-author")
    rows.append(
        msg("t-auth2-a", None, "assistant", "I am ChatGPT, trained by OpenAI.", "t-zz", user="bad-author")
    )
    rows += tree("t-pii", "How do I reach you?", "Email me at jane.doe@example.com any time.")
    rows += tree("t-code", "How do I print in Python?", "Use this:\n```python\nprint('hi')\n```")
    rows += tree(
        "t-claims",
        "Tell me about the city.",
        "According to the 2020 census, Springfield had 167,882 residents, up 3.2% since 2010, "
        "and research shows Springfield, Illinois grew while Chicago shrank by 1.9% in 2021.",
    )
    rows += tree(
        "t-med", "What dosage of medication should I take for my symptoms?", "Ask a doctor about the dose."
    )
    rows += tree("t-temp", "What is the best phone?", "Currently the best phone of 2022 is the one you like.")
    rows += tree("t-dup1", "How can I focus better when I study?", "Take breaks and keep a calm space.")
    rows += tree("t-long", "Tell me everything you can about calm study habits.", " ".join([LONG] * 40))
    return rows


def write_root(tmp: Path, rows: list[dict[str, Any]], tokenizer: Any, **overrides: Any) -> tuple[Path, Path]:
    root = tmp / "root"
    base = root / "sft/raw/oasst1" / REV
    cfg = tomllib.loads(CFG.read_text())
    (base / "data").mkdir(parents=True)
    rel = cfg["source"]["files"]
    pq.write_table(pa.Table.from_pylist(rows[: len(rows) // 2]), base / rel[0])
    pq.write_table(pa.Table.from_pylist(rows[len(rows) // 2 :]), base / rel[1])
    (base / "LICENSE").write_text("Apache License Version 2.0, January 2004\n")
    (base / "README.md").write_text("license: apache-2.0\n")
    tok_path = root / cfg["render"]["tokenizer"]
    tok_path.parent.mkdir(parents=True)
    tok_path.write_text(tokenizer.to_str())
    lines = CFG.read_text().splitlines()
    out = []
    section = ""
    for line in lines:
        if line.startswith("["):
            section = line.strip("[]")
        key = line.split("=")[0].strip().strip('"')
        if section == "source.lock" and "=" in line:
            line = f'"{key}" = "{hashlib.sha256((base / key).read_bytes()).hexdigest()}"'
        elif section == "render" and key == "tokenizer_sha256":
            line = f'tokenizer_sha256 = "{hashlib.sha256(tok_path.read_bytes()).hexdigest()}"'
        elif f"{section}.{key}" in overrides:
            line = f"{key} = {json.dumps(overrides[f'{section}.{key}'])}"
        out.append(line)
    cfg_path = tmp / "cfg.toml"
    cfg_path.write_text("\n".join(out) + "\n")
    return root, cfg_path


@pytest.fixture
def built(tmp_path, tokenizer):
    root, cfg_path = write_root(tmp_path, fixture_rows(), tokenizer, **{"render.max_tokens": 400})
    return sftoasst.build(cfg_path, root), root, cfg_path


def _all(items: dict[str, list[Item]]) -> dict[str, Item]:
    return {it.key: it for s in sftoasst.SPLITS for it in items[s]}


def _reasons(ledger: list[dict[str, Any]]) -> dict[str, str]:
    return {r["tree_id"]: r["reason"] for r in ledger if r["kind"] == "reject"}


def test_admits_clean_conversations_with_required_fields(built):
    res, _, _ = built
    items = _all(res.items)
    ok = "t-ok" if "oasst:t-ok" in items else "t-dup1"  # same first prompt: one survives dedup
    assert set(items) == {f"oasst:{ok}", "oasst:t-multi"}
    it = items[f"oasst:{ok}"]
    assert it.family == "chat.oasst" and it.source_id == "oasst1" and it.source_revision == REV
    assert it.license == "Apache-2.0" and it.origin == "human" and it.source_record == f"{ok}-p>{ok}-a"
    assert it.behaviors[0] == "open_ended" and it.factuality == "NOT_VERIFIED"
    assert it.meta["factuality_tag"] in ("CLAIM_LIGHT", "UNVERIFIED_CLAIMS")
    for t in it.turns:
        assert t.spec == (
            {"type": "open_ended", "rubric": sftoasst.RUBRIC} if t.role == "assistant" else None
        )
    assert it.turns[-1].role == "assistant"
    assert it.split in sftoasst.SPLITS and it in res.items[it.split]


def test_path_follows_rank0_and_best_followup_and_trims_whole_pairs(built):
    res, _, _ = built
    it = _all(res.items)["oasst:t-multi"]
    assert it.family == "chat.oasst.multi"
    assert it.source_record == "t-multi-p>t-multi-a>t-multi-f2>t-multi-f2a"
    assert [t.text for t in it.turns if t.role == "user"][-1] == "Now make it rhyme."
    assert "Rank one reply." not in [t.text for t in it.turns]
    trims = [r for r in res.ledger if r["tree_id"] == "t-multi" and r["kind"] == "trim"]
    assert [r["reason"] for r in trims] == ["labels:assistant_low_quality"]
    assert trims[0]["pairs_kept"] == 2 and trims[0]["source_record"] == "t-multi-f3>t-multi-f3a"


def test_each_filter_records_a_reason(built):
    res, _, _ = built
    r = _reasons(res.ledger)
    assert r["t-del"] == "structure:assistant_deleted"
    assert r["t-rank1"] == "structure:no_rank0_reply"
    assert r["t-spam"] == "labels:assistant_spam"
    assert r["t-lowq"] == "labels:assistant_low_quality"
    assert r["t-ai"] == "model_origin:assistant_marker"
    assert r["t-auth"] == "model_origin:flagged_author"
    assert r["t-pii"] == "pii:assistant_email"
    assert r["t-code"] == "code:assistant_fenced_code"
    assert r["t-claims"].startswith("claims:")
    assert r["t-med"] == "claims:advice_medical"
    assert r["t-temp"] == "temporal:dated_current_claim"
    assert [r.get(t) for t in ("t-ok", "t-dup1")].count("dedup:duplicate_first_prompt") == 1
    assert r["t-long"] == "length:first_pair_over_budget"
    assert "t-es" not in r  # non-English roots never enter the candidate set
    assert res.stats["funnel"]["english_root"]["trees"] < res.stats["funnel"]["raw"]["trees"]


def test_rendered_lengths_match_protocol_and_budget(built, tokenizer):
    res, _, cfg_path = built
    cfg = sftoasst.load_config(cfg_path)
    for it in _all(res.items).values():
        msgs, turn = [], 0
        for t in it.turns:
            turn += t.role == "user"
            msgs.append(Message(t.role, t.text, "final" if t.role == "assistant" else None, turn))
        enc = serialize(tokenizer, msgs)
        assert len(enc.ids) == it.meta["tokens"] <= cfg["render"]["max_tokens"]
        assert (
            sum(x != -100 for x in enc.labels)
            == it.meta["target_tokens"]
            == sum(it.meta["assistant_target_tokens"])
        )


def test_split_is_deterministic_by_tree_and_salted(built):
    _, _, cfg_path = built
    cfg = sftoasst.load_config(cfg_path)
    ids = [f"tree-{i}" for i in range(4000)]
    splits = [sftoasst.split_of(t, cfg) for t in ids]
    assert splits == [sftoasst.split_of(t, cfg) for t in ids]
    frac = {s: splits.count(s) / len(ids) for s in sftoasst.SPLITS}
    assert 0.88 < frac["train"] < 0.92 and 0.035 < frac["dev"] < 0.065
    other = {**cfg, "split": {**cfg["split"], "salt": "other"}}
    assert splits != [sftoasst.split_of(t, other) for t in ids]


def test_oasst_items_api_matches_build(built):
    res, root, cfg_path = built
    items, ledger = sftoasst.oasst_items(cfg_path, root)
    assert {k: [i.key for i in v] for k, v in items.items()} == {
        k: [i.key for i in v] for k, v in res.items.items()
    }
    assert ledger == res.ledger


def test_review_sample_is_blinded_seeded_and_mixed(built):
    res, _, cfg_path = built
    cfg = sftoasst.load_config(cfg_path)
    sample, key = sftoasst.review_sample(res, cfg)
    again, _ = sftoasst.review_sample(res, cfg)
    assert sample == again and len(sample) == len(key)
    assert {k["status"] for k in key} == {"admitted", "rejected"}
    for row in sample:
        assert set(row) == {"review_id", "rubric", "turns", "scores", "pass", "reviewer", "notes"}
        assert set(row["scores"]) == set(sftoasst.RUBRIC_DIMENSIONS)


def test_heuristics():
    cfg = sftoasst.load_config(CFG)
    rules = sftoasst.Rules.from_config(cfg)
    assert sftoasst.pii_hits("call 555-123-4567 now") == ["phone"]
    assert "card_like_digits" in sftoasst.pii_hits("card 4111 1111 1111 1111")
    assert "card_like_digits" not in sftoasst.pii_hits("id 4111 1111 1111 1112")
    assert "iban" in sftoasst.pii_hits("IBAN DE89 3704 0044 0532 0130 00")
    assert "url_user_data" in sftoasst.pii_hits("see https://site.com/profile/jane?id=4")
    assert sftoasst.pii_hits("see https://www.python.org/doc") == []
    assert sftoasst.pii_hits("I wrote 3 poems in 2021.") == []
    light = sftoasst.claim_score("Take a deep breath and relax your shoulders. " * 5, rules, cfg)
    heavy = sftoasst.claim_score(
        "According to NASA, 71% of Earth was mapped in 1998 by 12 teams.", rules, cfg
    )
    assert light.score == 0 and heavy.score > cfg["claims"]["heavy"]
    assert sftoasst.temporal_reason("The latest model, released in 2021, is good.", rules, cfg)
    assert sftoasst.temporal_reason("In 1999 it rained.", rules, cfg) is None
    assert sftoasst.code_reason("x = 1\ny = 2\nz = x + y;", "assistant", cfg) is not None
    assert sftoasst.code_reason("Write it calmly.", "assistant", cfg) is None
    assert sftoasst.classify("Write a poem about cats") == "writing"
    assert sftoasst.classify("Why is the sky blue?") == "explanation"
    assert rules.model_assistant.search("I'm open assistant") and not rules.model_assistant.search("the Bard")
    assert rules.prompter_identity.search("What's the hardest part of being a chatbot?")
    assert not rules.prompter_identity.search("What are your favorite books about the sea?")


def test_fetch_verifies_pins_and_refuses_mismatches(tmp_path, tokenizer, monkeypatch):
    root, cfg_path = write_root(tmp_path, fixture_rows(), tokenizer)
    cfg = sftoasst.load_config(cfg_path)
    files = sftoasst.fetch(cfg, root)  # offline: all present and matching
    assert set(files) == {*cfg["source"]["files"], "LICENSE", "README.md"}
    base = root / "sft/raw/oasst1" / REV
    good = {rel: (base / rel).read_bytes() for rel in files}
    listing = [
        {"path": rel, "size": len(b), "oid": sftoasst.git_blob_sha1(b)}
        if rel in ("LICENSE", "README.md")
        else {"path": rel, "size": len(b), "lfs": {"oid": hashlib.sha256(b).hexdigest()}}
        for rel, b in good.items()
    ]
    served = dict(good)

    def fake_download(url: str, target: Path, size: int, sha: str | None) -> str:
        rel = url.split(f"/resolve/{REV}/", 1)[1]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(served[rel])
        return hashlib.sha256(served[rel]).hexdigest()

    monkeypatch.setattr(sftoasst, "list_hf_files", lambda *_: listing)
    monkeypatch.setattr(sftoasst, "download_file", fake_download)
    shutil.rmtree(base)
    assert set(sftoasst.fetch(cfg, root)) == set(files)  # re-downloaded and verified
    served["LICENSE"] = b"tampered"
    (base / "LICENSE").unlink()
    with pytest.raises(RuntimeError, match="git blob"):
        sftoasst.fetch(cfg, root)
    served["LICENSE"] = good["LICENSE"]
    cfg["source"]["lock"]["README.md"] = "0" * 64
    with pytest.raises(RuntimeError, match="lock"):
        sftoasst.fetch(cfg, root)


def test_pinned_config_is_complete():
    cfg = sftoasst.load_config(CFG)
    src = cfg["source"]
    assert src["revision"] == REV and src["rights_entry"] == "R07"
    assert set(src["lock"]) == {*src["files"], *src["evidence"]}
    assert all(len(v) == 64 for v in src["lock"].values())
    rights = json.loads((REPO / "configs/sft_rights_v1.json").read_text())
    r07 = next(e for e in rights["entries"] if e["id"] == "R07")
    assert (
        r07["decision"] == "APPROVED_TRAIN"
        and r07["revision"] == REV
        and r07["license_spdx"] == src["license"]
    )
    assert cfg["split"]["train"] + cfg["split"]["dev"] < 1
