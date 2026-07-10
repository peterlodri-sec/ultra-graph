import numpy as np

from ultragraph.core import NodeRef, Tree, UltraEdge, UltraGraph


def test_node_byte_get_set_and_container_dunders():
    t = Tree(4, name="g")
    assert len(t) == 4
    assert 2 in t
    assert 9 not in t
    t[2] = 7
    assert t.nodes[2] == 7
    assert isinstance(t[2], NodeRef)
    assert t[2].value == 7
    t[2].value = -3
    assert t.nodes[2] == -3
    refs = list(t)
    assert len(refs) == 4 and all(isinstance(r, NodeRef) for r in refs)


def test_byte_clipping():
    t = Tree(2, name="g")
    t[0] = 500  # clip to 127
    t[1] = -500  # clip to -128
    assert t.nodes[0] == 127
    assert t.nodes[1] == -128


def test_micro_edge_via_rshift():
    t = Tree(4, name="g")
    e = t[0] >> t[1]
    assert (e.src, e.dst) == (0, 1)
    assert t.edge_src.tolist() == [0]
    assert t.edge_dst.tolist() == [1]
    assert t.edge_val.tolist() == [0]


def test_micro_edge_cross_tree_rejected():
    a, b = Tree(2, "a"), Tree(2, "b")
    try:
        _ = a[0] >> b[1]
        raised = False
    except ValueError:
        raised = True
    assert raised


def test_ultra_edge_via_rshift():
    ug = UltraGraph()
    a = ug.add(Tree.dense(3, 4, "a"))
    b = ug.add(Tree.dense(4, 2, "b", act="none"))
    ue = a >> b
    assert isinstance(ue, UltraEdge)
    assert ue.kind == "plain"
    assert ue in ug.ultra_edges
    ue2 = a.wire(b, kind="residual")
    assert ue2.kind == "residual"
    assert len(ug.ultra_edges) == 2


def test_wire_requires_owner():
    a, b = Tree.dense(3, 4, "a"), Tree.dense(4, 2, "b")
    try:
        _ = a >> b
        raised = False
    except ValueError:
        raised = True
    assert raised


def test_forward_ignores_insertion_order():
    """forward() must topo-sort by ultra-edges, not rely on add() order."""
    import numpy as np

    from ultragraph.autograd import Tensor

    ug = UltraGraph()
    # add the OUTPUT tree first, then the input tree -- reverse of data flow
    b = ug.add(Tree.dense(6, 3, "b", act="none"))
    a = ug.add(Tree.dense(4, 6, "a", act="relu"))
    a >> b  # data flows a -> b, but b was added first
    out = ug.forward(Tensor(np.random.randn(2, 4).astype(np.float32)))
    assert out.shape == (2, 3)


def test_forward_detects_cycle():
    import numpy as np

    from ultragraph.autograd import Tensor

    ug = UltraGraph()
    a = ug.add(Tree.dense(4, 4, "a", act="none"))
    b = ug.add(Tree.dense(4, 4, "b", act="none"))
    a >> b
    b >> a  # cycle
    try:
        ug.forward(Tensor(np.random.randn(2, 4).astype(np.float32)))
        raised = False
    except ValueError:
        raised = True
    assert raised


def test_dense_tree_has_byte_weights():
    t = Tree.dense(5, 3, "lin")
    assert t.wq.shape == (3, 5)
    assert t.wq.dtype == np.int8
    assert set(np.unique(t.wq)).issubset({-1, 0, 1})
    assert t.w_scale > 0
