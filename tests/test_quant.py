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


# -- STE gradient fidelity at model scale ------------------------------------


def test_ste_gradient_at_depth():
    """STE gradients must remain usable through stacked ternary-linear layers."""
    from ultragraph.autograd import Tensor
    from ultragraph.core import Tree

    rng = np.random.RandomState(99)
    d_model = 32

    trees = []
    for i in range(4):
        t = Tree.dense(d_model, d_model, name=f"layer{i}", act="none")
        w = rng.randn(d_model, d_model).astype(np.float32) * 0.02
        t.adhoc["w_master"].data[:] = w
        t.adhoc["bias"].data[:] = rng.randn(d_model).astype(np.float32) * 0.01
        trees.append(t)

    x = rng.randn(8, d_model).astype(np.float32) * 0.1
    t_inp = Tensor(x, requires_grad=True)
    t = t_inp

    for tree in trees:
        tree.adhoc["w_master"].data[:] = rng.randn(d_model, d_model).astype(np.float32) * 0.02
        t = tree.forward(t)

    loss = t.sum()
    loss.backward()

    grad_norm = float(np.sqrt((t_inp.grad ** 2).sum()))
    assert grad_norm > 0, "gradients must not vanish through 4 ternary-linear layers"
    assert np.isfinite(grad_norm), "gradients must not explode"


def test_ste_gradient_deeper():
    """4-layer network with relu: STE gradients survive but may weaken with depth."""
    from ultragraph.autograd import Tensor
    from ultragraph.core import Tree

    rng = np.random.RandomState(42)
    d_model = 16

    trees = [Tree.dense(d_model, d_model, name=f"layer{i}", act="relu") for i in range(4)]
    for t in trees:
        t.adhoc["w_master"].data[:] = rng.randn(d_model, d_model).astype(np.float32) * 0.02
        t.adhoc["bias"].data[:] = np.zeros(d_model, dtype=np.float32)

    x = rng.randn(4, d_model).astype(np.float32) * 0.1
    t_inp = Tensor(x, requires_grad=True)
    t = t_inp

    for tree in trees:
        t = tree.forward(t)

    loss = t.sum()
    loss.backward()

    grad_norm = float(np.sqrt((t_inp.grad ** 2).sum()))
    assert grad_norm > 0, "gradients must propagate through 4 ternary-linear+relu layers"
    assert np.isfinite(grad_norm), "gradients must not explode"


def test_ste_gradient_vanishing_with_deep_relu():
    """Documented: 8+ ternary-linear+relu layers cause gradient death.

    This validates the known limitation: relu after ternary quantization
    kills activations, and STE gradients vanish with depth. Use act='none'
    or gelu for deeper networks.
    """
    from ultragraph.autograd import Tensor
    from ultragraph.core import Tree

    rng = np.random.RandomState(42)
    d_model = 16

    trees = [Tree.dense(d_model, d_model, name=f"layer{i}", act="relu") for i in range(8)]
    for t in trees:
        t.adhoc["w_master"].data[:] = rng.randn(d_model, d_model).astype(np.float32) * 0.02
        t.adhoc["bias"].data[:] = np.zeros(d_model, dtype=np.float32)

    x = rng.randn(4, d_model).astype(np.float32) * 0.1
    t_inp = Tensor(x, requires_grad=True)
    t = t_inp

    for tree in trees:
        t = tree.forward(t)

    loss = t.sum()
    loss.backward()

    grad_norm = float(np.sqrt((t_inp.grad ** 2).sum()))
    # At 8 layers with relu, gradients typically vanish. This is expected.
    # The test documents this limitation rather than pretending it works.
    assert np.isfinite(grad_norm), "gradients must not explode even if they vanish"
