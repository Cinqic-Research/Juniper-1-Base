"""The frozen tokenizer must ship inside the installed package with the recorded bytes."""
import hashlib
from importlib import resources

from juniper1.tokenizer import TOKENIZER_JSON_SHA256, Juniper1Tokenizer


def test_tokenizer_is_package_data():
    data = resources.files("juniper1").joinpath("tokenizer.json").read_bytes()
    assert hashlib.sha256(data).hexdigest() == TOKENIZER_JSON_SHA256
    assert Juniper1Tokenizer().encode("ok")
