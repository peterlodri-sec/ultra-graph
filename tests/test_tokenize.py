import numpy as np

from ultragraph import ByteTokenizer


def test_byte_roundtrip_ascii_and_unicode():
    tok = ByteTokenizer()
    assert tok.vocab_size == 256
    for s in ["hello", "ultra graph world", "café ☕ ✓ 三 — 1.58 bits", ""]:
        ids = tok.encode(s)
        assert ids.dtype == np.int64
        if ids.size:
            assert ids.min() >= 0 and ids.max() <= 255
        assert tok.decode(ids) == s


def test_ids_are_utf8_bytes():
    tok = ByteTokenizer()
    assert list(tok.encode("A")) == [65]
    assert list(tok.encode("€")) == [0xE2, 0x82, 0xAC]  # 3 UTF-8 bytes


def test_decode_accepts_2d():
    tok = ByteTokenizer()
    ids = tok.encode("hi").reshape(1, -1)
    assert tok.decode(ids) == "hi"
