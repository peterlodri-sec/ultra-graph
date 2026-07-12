"""A byte-level tokenizer — lossless for any UTF-8 text, vocab of 256.

The simplest honest tokenizer: a token is a byte. No training, no vocab file, no
out-of-vocabulary. Pairs with a 256-row embedding.
"""

import numpy as np


class ByteTokenizer:
    """UTF-8 byte tokenizer. ``encode`` -> int64 ids in [0, 255]; ``decode`` -> str."""

    vocab_size = 256

    def encode(self, text: str) -> np.ndarray:
        return np.frombuffer(text.encode("utf-8"), dtype=np.uint8).astype(np.int64)

    def decode(self, ids) -> str:
        return bytes(int(i) & 0xFF for i in np.asarray(ids).reshape(-1)).decode("utf-8", errors="replace")
