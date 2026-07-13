"""ug-vizard — pure visualization sub-module for ultra-graph.

Zero-dependency interactive visualisation: scene graph, HTML5 Canvas renderer,
widgets, shader system, animation engine, and frame export. Everything is
self-contained — no matplotlib, no JavaScript frameworks, no npm. Output is
pure HTML/CSS/JS strings that work offline in any browser.

Public API
----------
* ``render_html(scene)``          — interactive HTML5 Canvas scene
* ``tree_widget(tree)``            — self-contained interactive tree viewer
* ``heatmap_widget(arr)``          — interactive byte heatmap
* ``animate_forward_pass(ug)``     — orbit flythrough as HTML animation
* ``export_gif(animation, path)``  — render animation as animated GIF
* ``export_mp4(animation, path)``  — render animation as MP4 video
* ``heatmap_shader()`` / ``sign_shader()`` / ... — programmable visual shaders
"""

from .anim import animate_forward_pass
from .core.scene import Camera, Scene
from .core.shader import (
    EdgeShader,
    NodeShader,
    activation_history_shader,
    apply_shaders,
    edge_weight_shader,
    gradient_magnitude_shader,
    heatmap_shader,
    sign_shader,
)
from .export import export_frame, export_gif, export_mp4
from .renderer.html5 import render_html, render_to_file
from .widgets.heatmap import heatmap_to_file, heatmap_widget
from .widgets.tree import tree_to_file, tree_widget

__all__ = [
    "Scene",
    "Camera",
    "NodeShader",
    "EdgeShader",
    "heatmap_shader",
    "sign_shader",
    "gradient_magnitude_shader",
    "activation_history_shader",
    "edge_weight_shader",
    "apply_shaders",
    "render_html",
    "render_to_file",
    "tree_widget",
    "tree_to_file",
    "heatmap_widget",
    "heatmap_to_file",
    "animate_forward_pass",
    "export_gif",
    "export_mp4",
    "export_frame",
]
