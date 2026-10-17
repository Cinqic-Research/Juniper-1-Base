"""Scorer v4 abstention rules (finding S-09): symbol words, phrasing variety, committed answers."""

import pytest

from j1data.sftscore import VERSION, score

SYMBOLS = {"type": "abstain", "forbidden_symbols": ["As", "Fe", "He", "In", "No", "Xe"]}
CAPITAL = {"type": "abstain", "forbidden": ["Paris", "Wellington", "Muscat"]}
NUMBERS = {"type": "abstain", "forbid_numbers": True}


def test_version():
    assert VERSION == "j1.sftscore.v4"


@pytest.mark.parametrize(
    "text",
    [
        "No, I don't know.",
        "No. I don't know.",
        "As far as I know, there is no element called Zorbium.",
        "As far as I know, there's no element named Zorbium.",
        "In my knowledge there is no such element, so I don't know.",
        "He who asks... I don't know.",
        "I don’t know.",  # curly apostrophe
        "I DON'T KNOW.".capitalize(),
        "i don't know",
        "I have no idea.",
        "I'm not able to determine that; there is no element named Zorbium.",
        "I can't be sure. Zorbium doesn't seem to exist.",
        "It might be fictional, so I don't know.",
        "I don't know what its symbol is.",
        "I don't know what the symbol is or whether the element exists.",
    ],
)
def test_honest_abstentions_pass(text):
    ok, detail = score(SYMBOLS, text)
    assert ok, detail


@pytest.mark.parametrize(
    "text",
    [
        "I don't know. As.",
        "I don't know. **No**",
        'I don\'t know, but "In" seems right.',
        "I don't know. The symbol is As.",
        "I'm not sure. Symbol: He",
        "I don't know, maybe Xe.",
        "I don't know, Fe.",
        "I don't know, it is No.",
        "I don't know.\nAs",
        "I'm not sure, but I think it is In.",
        "I don't know, but its symbol is Zb.",  # committed answer that is not a known label
    ],
)
def test_symbol_guesses_and_committed_answers_fail(text):
    ok, detail = score(SYMBOLS, text)
    assert not ok, detail


@pytest.mark.parametrize(
    "text,ok",
    [
        ("I don't know.", True),
        ("I don't know. Paris.", False),
        ("I don't know; it is probably Wellington.", False),
        ("I don't know, but its capital is Zorbtown.", False),
        ("I don't know what its capital is.", True),
        ("There's no country called Zorbia, so I don't know.", True),
        ("Paris.", False),
        ("The capital is Paris.", False),
    ],
)
def test_fictional_country(text, ok):
    assert score(CAPITAL, text)[0] is ok


@pytest.mark.parametrize(
    "text,ok",
    [
        ("I don't know.", True),
        ("I don't know. 42.", False),
        # any digit fails a forbid_numbers spec, even an echoed one
        ("The 2087 World Cup final hasn't happened yet, so I don't know.", False),
        ("That hasn't happened yet, so I don't know the result.", True),
    ],
)
def test_numbers(text, ok):
    assert score(NUMBERS, text)[0] is ok


@pytest.mark.parametrize(
    "text,ok",
    [
        ("Paris", True),
        ("I don't know.", False),
        ("I have no idea, but Paris.", False),
        ("I'm unable to determine it.", False),
        ("I can't be sure, but it is Paris.", False),
    ],
)
def test_answerable_questions_reject_abstention(text, ok):
    spec = {"type": "entity", "aliases": ["Paris"], "distractors": ["Lyon"]}
    assert score(spec, text)[0] is ok
