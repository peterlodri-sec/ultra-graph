import numpy as np

from ultragraph.autograd import Tensor, cat


def test_cat_forward_values():
    a = Tensor(np.arange(6, dtype=np.float32).reshape(2, 3))
    b = Tensor(np.arange(4, dtype=np.float32).reshape(2, 2))
    out = cat([a, b], axis=-1)
    assert out.shape == (2, 5)
    assert np.allclose(out.data, np.concatenate([a.data, b.data], axis=-1))


def test_cat_backward_splits_grad():
    rng = np.random.RandomState(0)
    A = rng.randn(3, 2).astype(np.float32)
    B = rng.randn(3, 4).astype(np.float32)
    C = rng.randn(3, 6).astype(np.float32)  # weights on the concatenated result

    a = Tensor(A.copy(), requires_grad=True)
    b = Tensor(B.copy(), requires_grad=True)
    loss = (cat([a, b], axis=-1) * Tensor(C)).sum()
    loss.backward()

    # d/da = C[:, :2], d/db = C[:, 2:]
    assert np.allclose(a.grad, C[:, :2])
    assert np.allclose(b.grad, C[:, 2:])


def test_cat_axis0():
    a = Tensor(np.ones((2, 3), dtype=np.float32), requires_grad=True)
    b = Tensor(np.ones((1, 3), dtype=np.float32), requires_grad=True)
    out = cat([a, b], axis=0)
    assert out.shape == (3, 3)
    out.sum().backward()
    assert np.allclose(a.grad, 1.0) and np.allclose(b.grad, 1.0)
