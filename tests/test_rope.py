import numpy as np

from ultragraph.autograd import Tensor
from ultragraph.nn import RoPE


def test_rope_preserves_norm():
    # A rotation is orthogonal, so per-token vector norms are unchanged.
    rope = RoPE(dim=8, max_len=16)
    x = Tensor(np.random.RandomState(0).randn(1, 2, 5, 8).astype(np.float32))  # [B,H,T,dh]
    y = rope(x)
    assert y.shape == x.shape
    nx = np.linalg.norm(x.data, axis=-1)
    ny = np.linalg.norm(y.data, axis=-1)
    assert np.allclose(nx, ny, atol=1e-4)


def test_rope_relative_position():
    # RoPE encodes relative position: <RoPE(q,m), RoPE(k,n)> depends on m-n only.
    rope = RoPE(dim=16, max_len=64)
    rng = np.random.RandomState(1)
    q = rng.randn(16).astype(np.float32)
    k = rng.randn(16).astype(np.float32)

    def dot(m, n):
        qm = rope(Tensor(q.reshape(1, 1, 16)), offset=m).data.reshape(-1)
        kn = rope(Tensor(k.reshape(1, 1, 16)), offset=n).data.reshape(-1)
        return float(qm @ kn)

    assert abs(dot(5, 2) - dot(10, 7)) < 1e-3   # same gap (3) -> same score
    assert abs(dot(5, 2) - dot(6, 2)) > 1e-3    # different gap -> different score


def test_rope_offset_matches_slice():
    # Applying RoPE with an offset equals slicing a longer full-length application.
    rope = RoPE(dim=8, max_len=32)
    x = Tensor(np.random.RandomState(2).randn(1, 1, 3, 8).astype(np.float32))
    full = np.zeros((1, 1, 6, 8), dtype=np.float32)
    full[:, :, 3:] = x.data
    y_full = rope(Tensor(full)).data[:, :, 3:]
    y_off = rope(x, offset=3).data
    assert np.allclose(y_full, y_off, atol=1e-5)
