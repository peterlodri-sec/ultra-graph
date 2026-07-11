import numpy as np
import pytest

from ultragraph import GPT, Adam, Mesh


def _experts(n=3):
    np.random.seed(0)
    return [GPT(vocab=16, d_model=16, n_layers=1, n_heads=2, max_len=32) for _ in range(n)]


def test_mesh_forward_shape():
    m = Mesh(_experts(3), vocab=16)
    ids = np.array([[1, 2, 3, 4]], dtype=np.int64)
    out = m(ids)
    assert out.shape == (1, 4, 16)
    # 1-D input works too
    assert m(np.array([1, 2, 3])).shape == (3, 16)


def test_mesh_gate_is_convex_combination():
    m = Mesh(_experts(3), vocab=16)
    ids = np.array([[2, 5, 7, 1]], dtype=np.int64)
    gate = m._gate(ids)
    assert gate.shape == (1, 3)
    assert np.allclose(gate.data.sum(axis=-1), 1.0, atol=1e-5)
    assert (gate.data >= 0).all()


def test_mesh_topk_routing_zeros_the_rest():
    m = Mesh(_experts(4), vocab=16, top_k=2)
    gate = m._gate(np.array([[1, 2, 3, 4, 5]], dtype=np.int64)).data
    assert (gate > 0).sum(axis=-1)[0] == 2          # only 2 experts active
    assert np.allclose(gate.sum(axis=-1), 1.0)      # still normalized


def test_mesh_parameters_include_router_and_experts():
    experts = _experts(2)
    m = Mesh(experts, vocab=16)
    total = m.n_params()
    experts_only = sum(e.n_params() for e in experts)
    assert total > experts_only                     # router + router-embedding add params


def test_mesh_rejects_bad_top_k():
    with pytest.raises(ValueError):
        Mesh(_experts(3), vocab=16, top_k=0)          # 0 would zero all logits
    with pytest.raises(ValueError):
        Mesh(_experts(3), vocab=16, top_k=5)          # > n_experts


def test_mesh_overfits_toy_sequence():
    m = Mesh(_experts(2), vocab=16)
    opt = Adam(m, lr=0.02, clip=1.0)
    seq = np.array([[2, 7, 1, 8, 2, 8, 1, 8]], dtype=np.int64)
    x, y = seq[:, :-1], seq[:, 1:]
    first = None
    for _ in range(80):
        loss = m(x).cross_entropy(y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if first is None:
            first = float(loss.data)
    assert float(loss.data) < first * 0.6
