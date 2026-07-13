"""Interactive tree viewer — self-contained HTML widget.

Produces a standalone HTML document with an interactive canvas view of a Tree.
Pan, zoom, hover for node details. No server, no dependencies — paste anywhere.
"""

from __future__ import annotations

from ultragraph.vizard.core.scene import scene_from_tree
from ultragraph.vizard.renderer.html5 import render_html


def tree_widget(tree, width: int = 700, height: int = 500) -> str:
    """Render a Tree as a self-contained interactive HTML widget.

    Returns a complete HTML string. Save to file or embed in an iframe.
    """
    scene = scene_from_tree(tree)
    scene.width = width
    scene.height = height
    return render_html(scene, title=f"{tree.name} — {tree.kind} | {int(tree.n_nodes)} nodes")


def tree_to_file(tree, path: str):
    """Render a Tree to an interactive HTML file."""
    scene = scene_from_tree(tree)
    scene.width = 800
    scene.height = 600
    with open(path, "w") as f:
        f.write(render_html(
            scene,
            title=f"{tree.name} — {tree.kind} | {int(tree.n_nodes)} nodes",
        ))
