import numpy as np

from ultragraph.autograd import Tensor, ternary_linear


def _numeric_grad(forward_np, arrays, wrt, eps=1e-3):
    """Central finite-difference gradient of scalar forward_np(arrays) wrt arrays[wrt]."""
    a = arrays[wrt]
    grad = np.zeros_like(a)
    flat = a.reshape(-1)
    gflat = grad.reshape(-1)
    for i in range(flat.size):
        orig = flat[i]
        flat[i] = orig + eps
        lp = float(forward_np(arrays))
        flat[i] = orig - eps
        lm = float(forward_np(arrays))
        flat[i] = orig
        gflat[i] = (lp - lm) / (2 * eps)
    return grad


def test_add_mul_matmul_relu_grads():
    rng = np.random.RandomState(0)
    A = rng.randn(3, 4).astype(np.float32)
    B = rng.randn(4, 2).astype(np.float32)

    def fwd_np(arrs):
        a, b = arrs
        return np.maximum(a @ b, 0).sum()

    def fwd_t(arrs):
        a = Tensor(arrs[0], requires_grad=True)
        b = Tensor(arrs[1], requires_grad=True)
        loss = (a @ b).relu().sum()
        loss.backward()
        return a, b

    a_t, b_t = fwd_t([A.copy(), B.copy()])
    ga = _numeric_grad(fwd_np, [A.copy(), B.copy()], 0)
    gb = _numeric_grad(fwd_np, [A.copy(), B.copy()], 1)
    assert np.allclose(a_t.grad, ga, atol=1e-2)
    assert np.allclose(b_t.grad, gb, atol=1e-2)


def test_softmax_grad():
    rng = np.random.RandomState(2)
    Z = rng.randn(4, 5).astype(np.float32)
    C = rng.randn(4, 5).astype(np.float32)

    def fwd_np(arrs):
        z, c = arrs
        zz = z - z.max(axis=-1, keepdims=True)
        e = np.exp(zz)
        p = e / e.sum(axis=-1, keepdims=True)
        return (p * c).sum()

    z_t = Tensor(Z.copy(), requires_grad=True)
    c_t = Tensor(C.copy(), requires_grad=True)
    loss = (z_t.softmax(axis=-1) * c_t).sum()
    loss.backward()
    gz = _numeric_grad(fwd_np, [Z.copy(), C.copy()], 0)
    assert np.allclose(z_t.grad, gz, atol=1e-2)


def test_cross_entropy_grad():
    rng = np.random.RandomState(3)
    Z = rng.randn(6, 4).astype(np.float32)
    targets = np.array([0, 1, 2, 3, 0, 1])

    def fwd_np(arrs):
        z = arrs[0]
        zz = z - z.max(axis=-1, keepdims=True)
        e = np.exp(zz)
        p = e / e.sum(axis=-1, keepdims=True)
        n = z.shape[0]
        return -np.log(p[np.arange(n), targets] + 1e-12).mean()

    z_t = Tensor(Z.copy(), requires_grad=True)
    loss = z_t.cross_entropy(targets)
    loss.backward()
    gz = _numeric_grad(fwd_np, [Z.copy()], 0)
    assert np.allclose(z_t.grad, gz, atol=1e-2)


def test_transpose_grad():
    rng = np.random.RandomState(5)
    A = rng.randn(3, 5).astype(np.float32)
    B = rng.randn(3, 4).astype(np.float32)  # loss = sum(A.T @ B) : A.T is [5,3]

    def fwd_np(arrs):
        a, b = arrs
        return (a.T @ b).sum()

    a_t = Tensor(A.copy(), requires_grad=True)
    b_t = Tensor(B.copy(), requires_grad=True)
    (a_t.transpose() @ b_t).sum().backward()
    ga = _numeric_grad(fwd_np, [A.copy(), B.copy()], 0)
    assert np.allclose(a_t.grad, ga, atol=1e-2)
    # .T property mirrors transpose()
    assert np.array_equal(Tensor(A).T.data, A.T)


def test_ternary_linear_ste_matches_surrogate():
    """STE: grads should match the *unquantized* linear surrogate y = x @ W.T + b."""
    rng = np.random.RandomState(4)
    X = rng.randn(5, 6).astype(np.float32)
    W = rng.randn(3, 6).astype(np.float32)
    b = rng.randn(3).astype(np.float32)

    def fwd_np(arrs):
        x, w, bb = arrs
        return (x @ w.T + bb).sum()

    x_t = Tensor(X.copy(), requires_grad=True)
    w_t = Tensor(W.copy(), requires_grad=True)
    b_t = Tensor(b.copy(), requires_grad=True)
    y = ternary_linear(x_t, w_t, b_t)
    y.sum().backward()

    gw = _numeric_grad(fwd_np, [X.copy(), W.copy(), b.copy()], 1)
    gx = _numeric_grad(fwd_np, [X.copy(), W.copy(), b.copy()], 0)
    gb = _numeric_grad(fwd_np, [X.copy(), W.copy(), b.copy()], 2)
    assert np.allclose(w_t.grad, gw, atol=1e-2)
    assert np.allclose(x_t.grad, gx, atol=1e-2)
    assert np.allclose(b_t.grad, gb, atol=1e-2)


def test_batched_matmul_grad():
    rng = np.random.RandomState(6)
    A = rng.randn(2, 3, 4).astype(np.float32)
    B = rng.randn(2, 4, 5).astype(np.float32)

    def fwd_np(arrs):
        return (arrs[0] @ arrs[1]).sum()

    a_t = Tensor(A.copy(), requires_grad=True)
    b_t = Tensor(B.copy(), requires_grad=True)
    (a_t @ b_t).sum().backward()
    assert np.allclose(a_t.grad, _numeric_grad(fwd_np, [A.copy(), B.copy()], 0), atol=1e-2)
    assert np.allclose(b_t.grad, _numeric_grad(fwd_np, [A.copy(), B.copy()], 1), atol=1e-2)


def test_reshape_and_swapaxes_grad():
    rng = np.random.RandomState(7)
    A = rng.randn(2, 3, 4).astype(np.float32)
    C = rng.randn(2, 4, 3).astype(np.float32)

    def fwd_np(arrs):
        a, c = arrs
        return (np.swapaxes(a, 1, 2) * c).sum()

    a_t = Tensor(A.copy(), requires_grad=True)
    c_t = Tensor(C.copy(), requires_grad=True)
    (a_t.swapaxes(1, 2) * c_t).sum().backward()
    assert np.allclose(a_t.grad, _numeric_grad(fwd_np, [A.copy(), C.copy()], 0), atol=1e-2)

    # reshape round-trips gradient
    b_t = Tensor(A.copy(), requires_grad=True)
    b_t.reshape(6, 4).sum().backward()
    assert np.allclose(b_t.grad, np.ones_like(A), atol=1e-5)


def test_mean_axis_sqrt_div_grads():
    rng = np.random.RandomState(8)
    A = rng.randn(3, 5).astype(np.float32)

    def mean_np(arrs):
        return (arrs[0].mean(axis=1) * np.arange(1, 4)).sum()

    a_t = Tensor(A.copy(), requires_grad=True)
    (a_t.mean(axis=1) * Tensor(np.arange(1, 4).astype(np.float32))).sum().backward()
    assert np.allclose(a_t.grad, _numeric_grad(mean_np, [A.copy()], 0), atol=1e-2)

    P = np.abs(rng.randn(3, 5).astype(np.float32)) + 0.5

    def sqrt_np(arrs):
        return np.sqrt(arrs[0]).sum()

    p_t = Tensor(P.copy(), requires_grad=True)
    p_t.sqrt().sum().backward()
    assert np.allclose(p_t.grad, _numeric_grad(sqrt_np, [P.copy()], 0), atol=1e-2)

    X = rng.randn(3, 5).astype(np.float32)
    D = np.abs(rng.randn(3, 1).astype(np.float32)) + 0.5

    def div_np(arrs):
        return (arrs[0] / arrs[1]).sum()

    x_t = Tensor(X.copy(), requires_grad=True)
    d_t = Tensor(D.copy(), requires_grad=True)
    (x_t / d_t).sum().backward()
    assert np.allclose(x_t.grad, _numeric_grad(div_np, [X.copy(), D.copy()], 0), atol=1e-2)
    assert np.allclose(d_t.grad, _numeric_grad(div_np, [X.copy(), D.copy()], 1), atol=1e-2)


def test_ternary_linear_3d_ste():
    rng = np.random.RandomState(9)
    X = rng.randn(2, 3, 4).astype(np.float32)
    W = rng.randn(5, 4).astype(np.float32)
    b = rng.randn(5).astype(np.float32)

    def fwd_np(arrs):
        x, w, bb = arrs
        return (x @ w.T + bb).sum()

    x_t = Tensor(X.copy(), requires_grad=True)
    w_t = Tensor(W.copy(), requires_grad=True)
    b_t = Tensor(b.copy(), requires_grad=True)
    y = ternary_linear(x_t, w_t, b_t)
    assert y.shape == (2, 3, 5)
    y.sum().backward()
    assert np.allclose(w_t.grad, _numeric_grad(fwd_np, [X.copy(), W.copy(), b.copy()], 1), atol=1e-2)
    assert np.allclose(x_t.grad, _numeric_grad(fwd_np, [X.copy(), W.copy(), b.copy()], 0), atol=1e-2)


def test_getitem_slice_grad():
    rng = np.random.RandomState(10)
    A = rng.randn(4, 5).astype(np.float32)
    C = rng.randn(4, 2).astype(np.float32)

    def fwd_np(arrs):
        a, c = arrs
        return (a[:, 1:3] * c).sum()

    a_t = Tensor(A.copy(), requires_grad=True)
    c_t = Tensor(C.copy(), requires_grad=True)
    (a_t[:, 1:3] * c_t).sum().backward()
    assert np.allclose(a_t.grad, _numeric_grad(fwd_np, [A.copy(), C.copy()], 0), atol=1e-2)
    assert np.all(a_t.grad[:, [0, 3, 4]] == 0)  # untouched columns get no grad


def test_sum_axis_grad():
    rng = np.random.RandomState(11)
    A = rng.randn(3, 4).astype(np.float32)
    C = rng.randn(3).astype(np.float32)

    def fwd_np(arrs):
        return (arrs[0].sum(axis=1) * arrs[1]).sum()

    a_t = Tensor(A.copy(), requires_grad=True)
    c_t = Tensor(C.copy(), requires_grad=True)
    (a_t.sum(axis=1) * c_t).sum().backward()
    assert np.allclose(a_t.grad, _numeric_grad(fwd_np, [A.copy(), C.copy()], 0), atol=1e-2)


def test_activation_grads():
    rng = np.random.RandomState(12)
    X = rng.randn(3, 4).astype(np.float32)
    fns = [
        ("exp", np.exp),
        ("tanh", np.tanh),
        ("sigmoid", lambda a: 1.0 / (1.0 + np.exp(-a))),
        ("gelu", lambda a: 0.5 * a * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (a + 0.044715 * a ** 3)))),
        ("silu", lambda a: a / (1.0 + np.exp(-a))),
    ]
    for name, np_fn in fns:
        def fwd_np(arrs, f=np_fn):
            return f(arrs[0]).sum()

        x = Tensor(X.copy(), requires_grad=True)
        getattr(x, name)().sum().backward()
        assert np.allclose(x.grad, _numeric_grad(fwd_np, [X.copy()], 0), atol=1e-2), name
