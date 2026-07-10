"""Layer patterns expressed as trees. Layers are configs over the byte-graph."""
from __future__ import annotations

from .core import Tree, UltraGraph


def linear_tree(in_dim: int, out_dim: int, name: str = "linear", act: str = "relu") -> Tree:
    """A dense ternary Linear layer as a tree."""
    return Tree.dense(in_dim, out_dim, name=name, act=act)


def mlp(dims: list[int], name: str = "mlp") -> UltraGraph:
    """Stack dense ternary linear trees, wired plain, relu between; last layer linear.

    ``dims`` is [in, hidden1, ..., out].
    """
    if len(dims) < 2:
        raise ValueError("mlp needs at least an input and output dim")
    ug = UltraGraph(name)
    prev = None
    for i in range(len(dims) - 1):
        is_last = i == len(dims) - 2
        t = linear_tree(dims[i], dims[i + 1], name=f"linear{i}", act="none" if is_last else "relu")
        ug.add(t)
        if prev is not None:
            prev >> t  # plain ultra-edge
        prev = t
    return ug
