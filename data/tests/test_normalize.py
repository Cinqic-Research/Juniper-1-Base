import json

import pytest

from j1data.normalize import normalize_stored, signature_text

PRESERVED = {
    "prose": "The color of the sky is blue; it isn't green.",
    "python": (
        "def area(r: float) -> float:\n    if r < 0:\n        raise ValueError('r')\n"
        "    return 3.14159 * r ** 2"
    ),
    "leading_indent": "    indented_first_line = True\n        deeper()",
    "tabs": "\tkey:\tvalue\n\t\tnested",
    "equation": "E = mc² and ∫₀^∞ e^{-x²} dx = √π / 2, with α ≤ β ≠ γ",
    "latex": r"\frac{\partial u}{\partial t} = \alpha \nabla^2 u",
    "json": json.dumps({"name": "juniper", "values": [1, 2.5, None], "nested": {"ok": True}}, indent=2),
    "yaml": "server:\n  host: 0.0.0.0\n  ports:\n    - 8080\n    - 8443\n",
    "xml": '<?xml version="1.0"?>\n<root a="1"><child>text &amp; more</child></root>',
    "shell": "$ export PATH=\"$HOME/.local/bin:$PATH\" && ls -la /usr/local | grep -E '^d'",
    "paths": r"C:\Program Files\Tool\bin and /var/log/nginx/access.log",
    "url": "https://example.org/api/v2/items?id=42&sort=desc#section-3",
    "api": "numpy.linalg.norm(x, ord=2) vs torch.nn.functional.softmax(logits, dim=-1)",
    "unicode": "naïve café — 東京 😀 ﬁ ligature and full-width ＡＢＣ",
}


@pytest.mark.parametrize("name", sorted(PRESERVED))
def test_stored_text_preserves_meaningful_content(name: str) -> None:
    text = PRESERVED[name].rstrip("\n")
    assert normalize_stored(text).text == text


def test_line_endings_bom_and_controls() -> None:
    result = normalize_stored("\ufeffline1\r\nline2\rline3\x00\x07\x1b[0m end\x0c")
    assert result.text == "line1\nline2\nline3[0m end"
    assert result.removed_controls == 4


def test_unpaired_surrogates_are_replaced() -> None:
    raw = "ok" + "\ud800" + "fine"
    result = normalize_stored(raw)
    assert result.text == "ok\ufffdfine"
    assert result.replaced_surrogates == 1
    result.text.encode("utf-8")


def test_outer_blank_lines_trimmed_but_first_indent_kept() -> None:
    assert normalize_stored("\n\n   \n    x = 1\n\n  \n").text == "    x = 1"


def test_signature_folds_case_accents_width_and_punctuation() -> None:
    assert signature_text("Naïve CAFÉ, ＡＢＣ!") == signature_text("naive cafe abc")
    assert signature_text("x² + y²") == "x2 y2"
    assert signature_text("ﬁle") == "file"


def test_signature_never_changes_stored_text() -> None:
    text = PRESERVED["equation"]
    signature_text(text)
    assert normalize_stored(text).text == text
