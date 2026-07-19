"""
Purpose: Protects legacy legacy behaviour.
Required because: A failing test in this module identifies a production contract or migration expectation that must be reviewed before merge.
"""

from __future__ import annotations

from neural_ir.tokenizer import tokenize
from neural_ir.vocab import Vocabulary


def test_vocabulary_build_save_load_encode(tmp_path) -> None:
    vocab = Vocabulary(min_freq=1)
    vocab.build([tokenize("Top 5 customers by sales"), tokenize("sales amount customer")])
    ids = vocab.encode(tokenize("top customers unknown_token"), max_len=6)

    assert len(ids) == 6
    assert vocab.decode(ids)[0] == "<CLS>"

    path = tmp_path / "vocab.json"
    vocab.save(str(path))
    loaded = Vocabulary.load(str(path))

    assert loaded.token_to_id == vocab.token_to_id
    assert loaded.encode(["top"], max_len=4)[1] == vocab.token_to_id["top"]
