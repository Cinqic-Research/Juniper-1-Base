"""Teacher rationale verification (distillation protocol v2)."""

import pytest

from j1data.teacherverify import arm_targets, check_rationale, normalize


def test_latex_is_normalized():
    assert normalize(r"\frac{3}{4} \times 8 = 6") == "((3)/(4)) * 8 = 6"
    assert normalize(r"$1,200 \div 4 = 300$") == "$1200 / 4 = 300$"


@pytest.mark.parametrize(
    "text",
    [
        "She buys 3 * 4 = 12 apples.",
        r"Total: \(12 \times 5 = 60\) dollars",
        "12 + 8 = 20\n20 / 4 = 5",
        "3 * (4 + 2) = 18 = 9 * 2",
        "10 / 3 = 3.33",
        r"\frac{1}{2} \cdot 40 = 20",
        "Earnings: $15 * 4 = $60",
        "Total = 3 + 4 = 7",  # a label before the chain is not part of the statement
    ],
)
def test_correct_rationales_verify(text):
    r = check_rationale(text)
    assert r.verified and r.checked, r.as_dict()


@pytest.mark.parametrize(
    "text",
    ["3 * 4 = 13", "12 + 8 = 20\n20 / 4 = 6", "3 * 4 = 12 = 11"],
)
def test_wrong_steps_are_found(text):
    r = check_rationale(text)
    assert r.wrong and not r.verified


@pytest.mark.parametrize(
    "text",
    [
        "Let x be the cost. 2x + 3 = 11",
        "x = 5",
        "20% of 50 = 10",
        "cost = price * n",
        "50 = 10% of 500",
        "3^x = 9",
    ],
)
def test_unverifiable_steps_reject_instead_of_being_skipped(text):
    r = check_rationale(text)
    assert r.unverifiable and not r.verified, r.as_dict()


def test_arm_targets():
    good = arm_targets("She has 3 * 4 = 12 apples.\n#### 12", "12")
    assert good["A1"] == "#### 12"
    assert good["A2"] == "She has 3 * 4 = 12 apples.\n#### 12"
    assert good["A3"] == "3 * 4 = 12\n#### 12"
    algebra = arm_targets("Let x = 4, so 3x = 12.\n#### 12", "12")
    assert algebra["A1"] == "#### 12" and algebra["A2"] is None and algebra["A3"] is None
    prose = arm_targets("She has twelve apples.\n#### 12", "12")
    assert prose["A2"] is not None and prose["A3"] is None
    long = arm_targets(" ".join(["word"] * 70) + "\n#### 12", "12")
    assert long["A2"] is None


@pytest.mark.parametrize(
    "text",
    [
        "60 + 25 = 85 .",
        "- 11 + 9 = 20",
        "$200 + $20 =  $220",
        "4 * $65 = $260",
        "* 2 * 86 = 172,",
        "1. 12 + 6 + 3 =  21 .",
    ],
)
def test_v11_layout_normalization_verifies_true_equalities(text):
    r = check_rationale(text)
    assert r.verified and r.checked and not r.wrong, r.as_dict()


@pytest.mark.parametrize("text", ["60 + 25 = 86 .", "- 11 + 9 = 21", "1. 12 + 6 = 19."])
def test_v11_layout_normalization_still_finds_false_equalities(text):
    r = check_rationale(text)
    assert r.wrong and not r.verified, r.as_dict()


def test_v11_minus_signs_and_step_numbers_stay_arithmetic():
    assert check_rationale("-5 + 3 = -2").verified
    assert check_rationale("2.5 + 1 = 3.5").verified
    assert check_rationale("12 + 3 = 15.").verified
    assert check_rationale("2000 g / 1000 = 2 kg .").unverifiable
