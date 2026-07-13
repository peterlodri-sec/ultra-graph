"""ug-vizard — pure visualization sub-module for ultra-graph.

Zero-dependency interactive visualisation: scene graph, HTML5 Canvas renderer,
widgets, and animation engine. Everything is self-contained — no matplotlib,
no JavaScript frameworks, no npm. Output is pure HTML/CSS/JS strings that
work offline in any browser.

Public API
----------
* ``view.graph(path)``        — 3D interactive ultra-graph explorer
* ``animate.forward_pass(...)`` — forward-pass flythrough as HTML/GIF
* ``dashboard.watch(...)``     — live training dashboard
* ``widget.tree(tree)``        — self-contained interactive tree viewer
* ``widget.heatmap(arr)``      — interactive byte heatmap
"""

from .anim import animate_forward_pass
from .core.scene import Camera, Scene
from .renderer.html5 import render_html, render_to_file
from .widgets.heatmap import heatmap_to_file, heatmap_widget
from .widgets.tree import tree_to_file, tree_widget

__all__ = [
    "Scene",
    "Camera",
    "render_html",
    "render_to_file",
    "tree_widget",
    "tree_to_file",
    "heatmap_widget",
    "heatmap_to_file",
    "animate_forward_pass",
]
