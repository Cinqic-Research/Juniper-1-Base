from j1data import domains
from j1data.ngrams import word_hashes
from j1data.normalize import signature_words


def label(text: str, kind: str = "prose") -> str:
    return domains.domain_label(word_hashes(signature_words(text)), kind)


def test_domain_labels_follow_the_lexicon() -> None:
    assert (
        label("The theorem gives the derivative of the polynomial; the integral of the function follows.")
        == "mathematics"
    )
    assert (
        label("Install the library with pip, import the class, and call the function from your Python code.")
        == "computing"
    )
    assert (
        label(
            "The cell membrane controls which protein and enzyme molecules enter; "
            "DNA in the gene encodes them."
        )
        == "life_science"
    )
    assert label("We went to the beach on Saturday and had a lovely lunch with friends.") == "general"
    assert label("anything", kind="code") == "code"


def test_spelling_counts() -> None:
    hashes = word_hashes(
        signature_words("The colour of the centre; the color of the center; organize behaviour.")
    )
    assert domains.spelling_counts(hashes) == (3, 3)


def test_formats_present() -> None:
    text = "# Title\n\n```python\nprint(1)\n```\n\n$ pip install numpy\n\n| a | b |\n|---|---|\n"
    found = domains.formats_present(text)
    assert {"markdown", "shell", "table"} <= set(found)
    assert "json" in domains.formats_present('{\n  "key": 1\n}')
