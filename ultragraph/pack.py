"""Dense ternary bit-packing — the true 1.58-bit storage.

The default byte-graph stores one ternary weight per byte (8 bits) to keep the
"1 edge = 1 byte" contract. When you actually want the density, pack 5 ternary
values into a single byte via base-3 (3**5 = 243 < 256): 8 bits / 5 weights =
1.6 bits per weight, a 5x shrink and within a whisker of the log2(3) ~ 1.58 limit.
"""

import numpy as np

_WEIGHTS = np.array([1, 3, 9, 27, 81], dtype=np.uint16)  # base-3 place values


def pack_ternary(q: np.ndarray) -> np.ndarray:
    """Pack a ternary array (values in {-1,0,+1}) into bytes, 5 values per byte.

    The input is flattened; recover shape/length with ``unpack_ternary`` + reshape.
    Returns a uint8 array of length ``ceil(q.size / 5)``.
    """
    d = (np.asarray(q).reshape(-1).astype(np.int16) + 1).astype(np.uint16)  # {-1,0,1} -> {0,1,2}
    if np.any((d < 0) | (d > 2)):
        raise ValueError("pack_ternary expects values in {-1, 0, +1}")
    pad = (-d.size) % 5
    if pad:
        d = np.concatenate([d, np.zeros(pad, dtype=np.uint16)])
    return (d.reshape(-1, 5) * _WEIGHTS).sum(axis=1).astype(np.uint8)


def unpack_ternary(packed: np.ndarray, n: int) -> np.ndarray:
    """Inverse of :func:`pack_ternary`; returns the first ``n`` values as int8."""
    b = np.asarray(packed).astype(np.uint16).copy()
    out = np.empty(b.size * 5, dtype=np.int8)
    for i in range(5):
        out[i::5] = (b % 3).astype(np.int8) - 1
        b //= 3
    return out[:n]


def packed_bits_per_weight() -> float:
    """Storage cost of the packed format, in bits per ternary weight (= 1.6)."""
    return 8.0 / 5.0
