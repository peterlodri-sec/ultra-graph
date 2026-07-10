import numpy as np

from ultragraph import Tensor, Tree, mlp, viz


def test_tree_svg_dense_and_sparse():
    ug = mlp([4, 6, 3])
    ug.forward(Tensor(np.random.randn(2, 4).astype(np.float32)))
    s = viz.tree_svg(ug.trees[0])
    assert s.startswith("<svg") and s.rstrip().endswith("</svg>")

    t = Tree(3, "g")
    t[0] >> t[1]
    s2 = viz.tree_svg(t)
    assert s2.startswith("<svg")


def test_ultragraph_svg():
    ug = mlp([4, 6, 3])
    s = viz.ultragraph_svg(ug)
    assert s.startswith("<svg") and s.rstrip().endswith("</svg>")


def test_byte_heatmap_svg_1d_and_2d():
    a = np.array([-128, -1, 0, 1, 127], dtype=np.int8)
    s = viz.byte_heatmap_svg(a)
    assert s.startswith("<svg")
    b = np.arange(-6, 6, dtype=np.int8).reshape(3, 4)
    s2 = viz.byte_heatmap_svg(b)
    assert s2.startswith("<svg")


def test_repr_svg_hooks():
    ug = mlp([4, 6, 3])
    assert ug._repr_svg_().startswith("<svg")
    assert ug.trees[0]._repr_svg_().startswith("<svg")
