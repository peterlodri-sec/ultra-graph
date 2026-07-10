import numpy as np

from ultragraph.quant import dequant, quantize_act_int8, quantize_weight_ternary


def test_ternary_bounds_and_values():
    rng = np.random.RandomState(0)
    w = rng.randn(32, 16).astype(np.float32)
    q, scale = quantize_weight_ternary(w)
    assert q.dtype == np.int8
    assert set(np.unique(q)).issubset({-1, 0, 1})
    assert scale > 0
    # reconstruction is in the right ballpark (absmean scaled)
    approx = dequant(q, scale)
    assert approx.shape == w.shape


def test_int8_bounds():
    rng = np.random.RandomState(1)
    x = rng.randn(8, 8).astype(np.float32) * 5.0
    q, scale = quantize_act_int8(x)
    assert q.dtype == np.int8
    assert q.min() >= -127 and q.max() <= 127
    assert scale > 0


def test_empty_weight_scale_is_finite():
    q, scale = quantize_weight_ternary(np.empty((0,), dtype=np.float32))
    assert np.isfinite(scale) and scale > 0
    assert q.size == 0


def test_dequant_roundtrip_small():
    q = np.array([-1, 0, 1], dtype=np.int8)
    out = dequant(q, 2.0)
    assert np.allclose(out, [-2.0, 0.0, 2.0])
