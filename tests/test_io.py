import os
import tempfile

import numpy as np

from ultragraph.autograd import Tensor
from ultragraph.core import Embedding, Tree, UltraGraph
from ultragraph.io import load, save
from ultragraph.nn import mlp


def test_save_load_byte_exact():
    np.random.seed(0)
    ug = mlp([4, 6, 3])
    x = Tensor(np.random.randn(2, 4).astype(np.float32))
    out_before = ug.forward(x).data.copy()

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "model.ug")
        save(ug, path)
        ug2 = load(path)

    # byte-exact on the compact state
    for t1, t2 in zip(ug.trees, ug2.trees):
        assert np.array_equal(t1.nodes, t2.nodes)
        assert np.array_equal(t1.wq, t2.wq)
        assert t1.w_scale == t2.w_scale
        assert np.array_equal(t1.adhoc["bias"].data, t2.adhoc["bias"].data)

    # reloaded model reproduces the forward output
    out_after = ug2.forward(x).data
    assert np.allclose(out_before, out_after)


def test_save_load_sparse_and_embedding():
    ug = UltraGraph("mixed")
    t = Tree(3, "sparse")
    t[0] >> t[1]
    t[1] >> t[2]
    t[0] = 5
    ug.add(t)
    ug.register(Embedding(7, 4, "emb"))
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "m.ug")
        save(ug, path)
        ug2 = load(path)
    t2 = ug2.trees[0]
    assert t2.edge_src.tolist() == [0, 1]
    assert t2.edge_dst.tolist() == [1, 2]
    assert t2.nodes[0] == 5
    assert ug2.modules[0].vocab == 7 and ug2.modules[0].dim == 4


def test_save_load_embedding_index_with_interleaved_module():
    """Non-Embedding modules interleaved must not break embedding array indexing."""

    class _Dummy:
        def parameters(self):
            return []

    ug = UltraGraph("interleaved")
    ug.register(Embedding(5, 3, "e0"))
    ug.register(_Dummy())
    ug.register(Embedding(9, 2, "e1"))
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "m.ug")
        save(ug, path)
        ug2 = load(path)
    embs = [m for m in ug2.modules if isinstance(m, Embedding)]
    assert [(m.vocab, m.dim) for m in embs] == [(5, 3), (9, 2)]
