"""Render ultragraph structures to SVG.

Builds a dense ``mlp`` (an ``UltraGraph`` of ternary linear trees) and a small
sparse ``Tree`` wired with a couple of micro-edges (``t[0] >> t[1]``), runs one
forward pass to populate the node bytes, then renders three SVGs via the ``viz``
module and writes them into ``./out/``.

Run:
    uv run python examples/render_viz.py
"""
from __future__ import annotations

import os

import numpy as np

from ultragraph import Tensor, Tree, mlp, viz


def main() -> None:
    out_dir = os.path.abspath("out")
    os.makedirs(out_dir, exist_ok=True)

    # dense net: forward pass populates each tree's node bytes for viz
    net = mlp([4, 8, 3])
    x = Tensor(np.random.randn(2, 4).astype(np.float32))
    net.forward(x)

    # sparse tree with a couple of micro-edges
    t = Tree(4, "sparse")
    t[0] = 120
    t[1] = -80
    t[2] = 40
    t[3] = -10
    t[0] >> t[1]
    t[1] >> t[2]

    tree_path = os.path.join(out_dir, "tree.svg")
    ultra_path = os.path.join(out_dir, "ultragraph.svg")
    heatmap_path = os.path.join(out_dir, "heatmap.svg")

    with open(tree_path, "w") as f:
        f.write(viz.tree_svg(t))
    with open(ultra_path, "w") as f:
        f.write(viz.ultragraph_svg(net))
    with open(heatmap_path, "w") as f:
        f.write(viz.byte_heatmap_svg(net.trees[0].nodes))

    for p in (tree_path, ultra_path, heatmap_path):
        print("wrote", p)


if __name__ == "__main__":
    main()
