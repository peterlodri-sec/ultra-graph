"""Quantization primitives (BitNet b1.58 style).

Weights are ternary {-1, 0, +1}; activations are int8. Each quantized value fits
in one byte, honoring the ultragraph byte contract.
"""

import numpy as np

EPS = 1e-5


def quantize_weight_ternary(w: np.ndarray) -> tuple[np.ndarray, float]:
    """Absmean ternary quantization.

    Returns (q, scale) with q in {-1, 0, +1} as int8 and scale a python float such
    that ``q * scale`` approximates ``w``.
    """
    w = np.asarray(w, dtype=np.float32)
    m = float(np.mean(np.abs(w))) if w.size else 0.0
    if not np.isfinite(m):
        m = 0.0
    scale = m + EPS
    q = np.clip(np.round(np.nan_to_num(w / scale)), -1, 1).astype(np.int8)
    return q, scale


def quantize_act_int8(x: np.ndarray) -> tuple[np.ndarray, float]:
    """Absmax int8 activation quantization.

    Returns (q, scale) with q in [-127, 127] as int8 and scale a python float such
    that ``q * scale`` approximates ``x``.
    """
    x = np.asarray(x, dtype=np.float32)
    m = float(np.max(np.abs(x))) if x.size else 0.0
    if not np.isfinite(m):
        m = 0.0
    scale = m / 127.0 + EPS
    q = np.clip(np.round(np.nan_to_num(x / scale)), -127, 127).astype(np.int8)
    return q, scale


def dequant(q: np.ndarray, scale: float) -> np.ndarray:
    """Inverse of the quantizers: reconstruct a float32 array."""
    return np.asarray(q, dtype=np.float32) * np.float32(scale)
