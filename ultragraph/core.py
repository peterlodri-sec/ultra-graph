"""The byte-graph data model and dunder-driven API.

Three levels:
  * node / edge   -- 1 byte each (int8 activation / ternary weight)
  * tree          -- a whole graph == one net/module
  * ultra-edge    -- typed wiring between trees; the set of trees + ultra-edges
                     is the ultra-graph (the full model).

``>>`` is overloaded by operand type: ``node >> node`` builds a micro-edge inside a
tree; ``tree >> tree`` builds an ultra-edge between trees.
"""

from typing import Iterator

import numpy as np

from .autograd import Tensor, ternary_forward, ternary_linear
from .quant import quantize_act_int8, quantize_weight_ternary


def _clip_byte(v) -> np.int8:
    return np.int8(int(np.clip(round(float(v)), -128, 127)))


class Edge:
    """A directed micro-edge inside a sparse tree. Value is a ternary weight byte."""

    __slots__ = ("src", "dst", "value")

    def __init__(self, src: int, dst: int, value: int = 0):
        self.src = int(src)
        self.dst = int(dst)
        self.value = int(value)

    def __repr__(self):
        return f"Edge({self.src}->{self.dst}, {self.value:+d})"


class NodeRef:
    """Lightweight handle for node ``index`` in ``tree`` (no per-node object stored)."""

    __slots__ = ("tree", "index")

    def __init__(self, tree: "Tree", index: int):
        self.tree = tree
        self.index = int(index)

    @property
    def value(self) -> int:
        return int(self.tree.nodes[self.index])

    @value.setter
    def value(self, v):
        self.tree.nodes[self.index] = _clip_byte(v)

    def __rshift__(self, other: "NodeRef") -> Edge:
        if not isinstance(other, NodeRef):
            return NotImplemented
        if other.tree is not self.tree:
            raise ValueError("micro-edges connect nodes within the same tree")
        return self.tree.add_edge(self.index, other.index)

    def __repr__(self):
        return f"NodeRef({self.tree.name}[{self.index}]={self.value:+d})"


class UltraEdge:
    """A typed connection between two trees. kind in {'plain', 'residual'}."""

    __slots__ = ("src", "dst", "kind")

    def __init__(self, src: "Tree", dst: "Tree", kind: str = "plain"):
        self.src = src
        self.dst = dst
        self.kind = kind

    def __repr__(self):
        return f"UltraEdge({self.src.name} ==={self.kind}=> {self.dst.name})"


class Tree:
    """A whole graph == one module. Dense (Linear) or sparse (general graph)."""

    def __init__(self, n_nodes: int, name: str = "tree"):
        # generic / sparse constructor
        self.kind = "sparse"
        self.name = name
        self.n_nodes = int(n_nodes)
        self.nodes = np.zeros(self.n_nodes, dtype=np.int8)
        self._esrc: list[int] = []
        self._edst: list[int] = []
        self._eval: list[int] = []
        self.adhoc: dict = {"name": name}
        self._owner: "UltraGraph | None" = None
        # dense-only fields (unset for sparse)
        self.in_dim = None
        self.out_dim = None
        self.act = "none"
        self.wq: np.ndarray | None = None
        self.w_scale: float = 1.0
        self._wq_stale: bool = False

    def _invalidate_wq(self) -> None:
        """Mark the cached ternary weight matrix as stale.

        Called by optimizers before updating fp32 masters. The next
        ``requantize()`` call clears this flag.
        """
        self._wq_stale = True

    # -- constructors ---------------------------------------------------------
    @classmethod
    def dense(cls, in_dim: int, out_dim: int, name: str = "linear", act: str = "relu") -> "Tree":
        t = cls(out_dim, name=name)
        t.kind = "dense"
        t.in_dim = int(in_dim)
        t.out_dim = int(out_dim)
        t.act = act
        w = (np.random.randn(out_dim, in_dim).astype(np.float32) * np.float32(1.0 / np.sqrt(in_dim)))
        t.adhoc["w_master"] = Tensor(w, requires_grad=True)
        t.adhoc["bias"] = Tensor(np.zeros(out_dim, dtype=np.float32), requires_grad=True)
        t.requantize()
        return t

    # -- sparse edge building -------------------------------------------------
    def add_edge(self, src: int, dst: int, value: int = 0) -> Edge:
        if self.kind != "sparse":
            raise ValueError("micro-edges are only supported on sparse trees")
        self._esrc.append(int(src))
        self._edst.append(int(dst))
        self._eval.append(int(value))
        return Edge(src, dst, value)

    @property
    def edge_src(self) -> np.ndarray:
        return np.array(self._esrc, dtype=np.int32)

    @property
    def edge_dst(self) -> np.ndarray:
        return np.array(self._edst, dtype=np.int32)

    @property
    def edge_val(self) -> np.ndarray:
        return np.array(self._eval, dtype=np.int8)

    # -- dense compute --------------------------------------------------------
    @property
    def deployed(self) -> bool:
        """True for a dense tree with no fp32 master — inference-only, runs straight
        from the stored ternary bytes (see ``GPT.load_deployed``)."""
        return self.kind == "dense" and self.adhoc.get("w_master") is None

    def requantize(self) -> None:
        """Refresh the ternary weight bytes from the fp32 master (dense only)."""
        if self.kind != "dense" or self.adhoc.get("w_master") is None:
            return
        wq, scale = quantize_weight_ternary(self.adhoc["w_master"].data)
        self.wq = wq
        self.w_scale = scale
        self._wq_stale = False

    def forward(self, x: Tensor) -> Tensor:
        if self.kind != "dense":
            raise NotImplementedError("forward is defined for dense trees in the MVP")
        if self.deployed:
            y = ternary_forward(x, self.wq, self.w_scale, self.adhoc.get("bias"))
        else:
            y = ternary_linear(x, self.adhoc["w_master"], self.adhoc["bias"])
        if self.act == "relu":
            y = y.relu()
        # write node bytes from the int8 activation averaged over all leading (batch/
        # sequence) dims, for inspection/viz
        flat = y.data.reshape(-1, y.data.shape[-1]) if y.data.ndim >= 2 else y.data.reshape(1, -1)
        q, _ = quantize_act_int8(flat.mean(axis=0))
        self.nodes[:] = q[: self.n_nodes]
        return y

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)

    def parameters(self) -> list[Tensor]:
        if self.kind == "dense" and not self.deployed:
            return [self.adhoc["w_master"], self.adhoc["bias"]]
        return []

    # -- dunder graph API -----------------------------------------------------
    def __len__(self) -> int:
        return self.n_nodes

    def __getitem__(self, i: int) -> NodeRef:
        if not (0 <= i < self.n_nodes):
            raise IndexError(i)
        return NodeRef(self, i)

    def __setitem__(self, i: int, v) -> None:
        self.nodes[i] = _clip_byte(v)

    def __iter__(self) -> Iterator[NodeRef]:
        return (NodeRef(self, i) for i in range(self.n_nodes))

    def __contains__(self, i) -> bool:
        return isinstance(i, int) and 0 <= i < self.n_nodes

    def __rshift__(self, other: "Tree") -> UltraEdge:
        if not isinstance(other, Tree):
            return NotImplemented
        owner = self._owner or other._owner
        if owner is None:
            raise ValueError("both trees must be added to an UltraGraph before wiring")
        return owner.wire(self, other, "plain")

    def wire(self, other: "Tree", kind: str = "plain") -> UltraEdge:
        owner = self._owner or other._owner
        if owner is None:
            raise ValueError("both trees must be added to an UltraGraph before wiring")
        return owner.wire(self, other, kind)

    def _repr_svg_(self):
        from . import viz
        return viz.tree_svg(self)

    def __repr__(self):
        return f"Tree({self.kind} {self.name!r}, nodes={self.n_nodes})"


class Embedding:
    """A small fp32 lookup table (kept full precision -- it is tiny)."""

    def __init__(self, vocab: int, dim: int, name: str = "embed"):
        self.name = name
        self.vocab = int(vocab)
        self.dim = int(dim)
        table = np.random.randn(vocab, dim).astype(np.float32) * np.float32(0.1)
        self.table = Tensor(table, requires_grad=True)

    def __call__(self, ids: np.ndarray) -> Tensor:
        ids = np.asarray(ids, dtype=np.int64)
        out = Tensor(self.table.data[ids], requires_grad=self.table.requires_grad, _prev=(self.table,))

        def _backward():
            np.add.at(self.table.grad, ids, out.grad)

        if out.requires_grad:
            out._backward = _backward
        return out

    def parameters(self) -> list[Tensor]:
        return [self.table]


class UltraGraph:
    """Top-level model: trees wired by ultra-edges, plus optional extra modules."""

    def __init__(self, name: str = "ultragraph"):
        self.name = name
        self.trees: list[Tree] = []
        self.ultra_edges: list[UltraEdge] = []
        self.modules: list = []  # extra param-bearing modules (e.g. Embedding)

    def add(self, tree: Tree) -> Tree:
        tree._owner = self
        self.trees.append(tree)
        return tree

    def register(self, module) -> object:
        self.modules.append(module)
        return module

    def wire(self, a: Tree, b: Tree, kind: str = "plain") -> UltraEdge:
        ue = UltraEdge(a, b, kind)
        self.ultra_edges.append(ue)
        return ue

    def forward(self, x: Tensor) -> Tensor:
        if not self.trees:
            return x
        incoming: dict[int, list[UltraEdge]] = {id(t): [] for t in self.trees}
        for e in self.ultra_edges:
            incoming[id(e.dst)].append(e)

        outputs: dict[int, Tensor] = {}
        order: list[Tree] = []  # topological finalization order
        remaining = list(self.trees)
        while remaining:
            progressed = False
            still: list[Tree] = []
            for tree in remaining:
                edges = incoming[id(tree)]
                if not all(id(e.src) in outputs for e in edges):
                    still.append(tree)
                    continue
                plain_srcs = [e.src for e in edges if e.kind == "plain"]
                if plain_srcs:
                    inp = outputs[id(plain_srcs[0])]
                    for s in plain_srcs[1:]:
                        inp = inp + outputs[id(s)]
                else:
                    inp = x
                out = tree.forward(inp)
                for e in edges:
                    if e.kind == "residual":
                        out = out + outputs[id(e.src)]
                outputs[id(tree)] = out
                order.append(tree)
                progressed = True
            if not progressed:
                raise ValueError("cycle detected in ultra-graph wiring")
            remaining = still
        # model output = the sink (a tree with no outgoing plain edge); if several,
        # the last one reached in topological order.
        plain_src_ids = {id(e.src) for e in self.ultra_edges if e.kind == "plain"}
        sinks = [t for t in order if id(t) not in plain_src_ids]
        result_tree = sinks[-1] if sinks else order[-1]
        return outputs[id(result_tree)]

    def parameters(self) -> list[Tensor]:
        params: list[Tensor] = []
        for m in self.modules:
            params.extend(m.parameters())
        for t in self.trees:
            params.extend(t.parameters())
        return params

    def requantize(self) -> None:
        for m in self.modules:
            r = getattr(m, "requantize", None)
            if callable(r):
                r()
        for t in self.trees:
            t.requantize()

    def _repr_svg_(self):
        from . import viz
        return viz.ultragraph_svg(self)

    def __repr__(self):
        return f"UltraGraph({self.name!r}, trees={len(self.trees)}, ultra_edges={len(self.ultra_edges)})"
