import pytest

from juniper1.tokenizer import (
    CONTROL_TOKENS,
    FIRST_LEARNED_ID,
    N_CONTROL,
    VOCAB_SIZE,
    Juniper1Tokenizer,
    TokenizerError,
)

FIXTURES = [
    "The quick brown fox jumps over the lazy dog.",
    "def f(x):\n    return x ** 2  # tab\tindented\n",
    "    four-space indent\n\t\ttwo tabs\n",
    "∀ε>0 ∃δ>0: |x−a|<δ ⇒ |f(x)−f(a)|<ε; ∇·E = ρ/ε₀",
    "naïve café — “quoted” … 😀👍🏽 中文 日本語",
    "",
    " ",
    "\n\n\n",
    "123456789 + 987654321 = 1111111110; 3.14159",
    "<|endoftext|><|user|><|assistant|><|system|><|reserved_12|>",
    "text with <|user|> inside and <|eot|> at the end<|eot|>",
    "\u0000\u0001 control bytes \x7f",
]


@pytest.fixture(scope="module")
def tok():
    return Juniper1Tokenizer()


def test_layout(tok):
    assert tok.vocab_size == VOCAB_SIZE == 24576
    assert len(CONTROL_TOKENS) == N_CONTROL == 64
    assert FIRST_LEARNED_ID == 320
    assert tok.control_id("<|endoftext|>") == 0 and tok.control_id("<|user|>") == 3 and tok.control_id("<|eot|>") == 11


@pytest.mark.parametrize("text", FIXTURES)
def test_roundtrip_and_no_control_ids_from_text(tok, text):
    ids = tok.encode(text)
    assert all(0 <= i < VOCAB_SIZE for i in ids)
    assert all(i >= N_CONTROL for i in ids), "literal control spellings must never become control IDs"
    assert tok.decode(ids) == text


def test_control_ids_only_via_api(tok):
    ids = [tok.control_id("<|user|>")] + tok.encode("hi") + [tok.control_id("<|eot|>")]
    assert ids[0] == 3 and ids[-1] == 11
    assert tok.decode_text(ids) == "hi"
    assert tok.decode(ids) == "<|user|>hi<|eot|>"


def test_byte_fallback_for_rare_scripts(tok):
    ids = tok.encode("😀")
    assert ids and all(64 <= i < 320 for i in ids), "emoji is encoded by byte fallback"
    assert tok.decode(ids) == "😀"


def test_add_eos(tok):
    assert tok.encode("x", add_eos=True)[-1] == 0


def test_bare_loader_hazard_is_documented():
    """Loading tokenizer.json without the flag maps a literal control spelling to a control ID."""
    from tokenizers import Tokenizer

    from juniper1.tokenizer import DEFAULT_PATH

    bare = Tokenizer.from_file(str(DEFAULT_PATH))
    assert 3 in bare.encode("<|user|>", add_special_tokens=False).ids
    assert 3 not in Juniper1Tokenizer().encode("<|user|>")


def test_wrong_bytes_rejected(tmp_path):
    p = tmp_path / "tokenizer.json"
    p.write_text("{}")
    with pytest.raises(TokenizerError):
        Juniper1Tokenizer(p)
