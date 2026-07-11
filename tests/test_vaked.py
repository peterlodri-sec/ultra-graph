import numpy as np

from ultragraph import compile_vaked, lower_graph


def test_lower_graph_to_bytegraph():
    nodes = [{"id": "a", "kind": "stream"}, {"id": "b", "kind": "node"}, {"id": "c", "kind": "index"}]
    edges = [
        {"src": "a", "dst": "b", "label": "contains"},
        {"src": "b", "dst": "c", "label": "depends_on"},
        {"src": "x", "dst": "a", "label": "dangling"},  # unknown src -> skipped
    ]
    t = lower_graph(nodes, edges)
    assert len(t) == 3
    assert t.edge_src.tolist() == [0, 1]
    assert t.edge_dst.tolist() == [1, 2]
    assert t.adhoc["vaked_ids"] == ["a", "b", "c"]
    assert t.nodes.dtype == np.int8
    # kind codes are deterministic across runs
    assert np.array_equal(t.nodes, lower_graph(nodes, edges).nodes)


def test_lower_graph_duck_typed_objects():
    class N:
        def __init__(self, i, k):
            self.id, self.kind = i, k

    class E:
        def __init__(self, s, d, label):
            self.src, self.dst, self.label = s, d, label

    t = lower_graph([N("n1", "decl"), N("n2", "node")], [E("n1", "n2", "contains")])
    assert len(t) == 2 and t.edge_src.tolist() == [0] and t.edge_dst.tolist() == [1]


def test_lower_graph_none_ids_dont_collide():
    # a node without an id must not capture edges missing src/dst (None collision)
    nodes = [{"kind": "orphan"}, {"id": "b", "kind": "node"}]
    edges = [{"dst": "b", "label": "x"}]  # missing src -> skipped, not routed to the id-less node
    t = lower_graph(nodes, edges)
    assert len(t) == 2
    assert t.edge_src.tolist() == []


def test_compile_vaked_requires_importable_vakedc():
    # vakedc is optional (vendored, currently de-indented upstream) -> clean ImportError
    try:
        compile_vaked("stream x { }")
        raised = False
    except ImportError:
        raised = True
    assert raised
