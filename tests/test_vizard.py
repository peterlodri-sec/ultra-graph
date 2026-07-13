"""Tests for ug-vizard — pure visualization sub-module.

Tests cover: scene graph, HTML5 renderer, widgets, shaders, animation, export.
Optional dependency tests (pillow, imageio) are skipped when not installed.
"""

from __future__ import annotations

import numpy as np
import pytest

from ultragraph.core import Tree, UltraGraph
from ultragraph.vizard import (
    Camera,
    EdgeShader,
    NodeShader,
    Scene,
    activation_history_shader,
    animate_forward_pass,
    apply_shaders,
    edge_weight_shader,
    gradient_magnitude_shader,
    heatmap_shader,
    heatmap_widget,
    render_html,
    sign_shader,
    tree_widget,
)
from ultragraph.vizard.core.scene import Label, VisualEdge, VisualNode, scene_from_tree, scene_from_ultragraph

# -- scene graph -------------------------------------------------------------


def test_scene_empty():
    s = Scene()
    assert s.nodes == []
    assert s.edges == []
    assert s.labels == []
    assert s.width == 800
    assert s.height == 600


def test_camera():
    cam = Camera()
    assert cam.zoom == 1.0
    cam.dolly(-10)
    assert cam.z >= 90


def test_scene_from_tree_sparse():
    t = Tree(4, name="test")
    t.nodes[0] = -50
    t.nodes[1] = 0
    t.nodes[2] = 50
    t.nodes[3] = -128

    scene = scene_from_tree(t)
    assert len(scene.nodes) == 4
    assert scene.nodes[0].byte_value == -50
    assert scene.nodes[1].byte_value == 0
    assert scene.nodes[2].byte_value == 50
    assert scene.title == "test"


def test_scene_from_tree_dense():
    t = Tree.dense(4, 3, name="linear")
    scene = scene_from_tree(t)
    assert len(scene.nodes) >= 3
    assert scene.title == "linear"


def test_scene_from_ultragraph():
    ug = UltraGraph("test-graph")
    a = Tree.dense(4, 8, name="encoder")
    b = Tree.dense(8, 4, name="decoder")
    ug.add(a)
    ug.add(b)
    ug.wire(a, b, "plain")

    scene = scene_from_ultragraph(ug)
    assert len(scene.nodes) == 2
    assert len(scene.edges) == 1
    assert len(scene.labels) >= 2


def test_scene_bounds():
    s = Scene()
    s.add_node(VisualNode(id="a", x=10, y=20))
    s.add_node(VisualNode(id="b", x=100, y=200))
    min_x, min_y, max_x, max_y = s.bounds()
    assert min_x == 10
    assert min_y == 20
    assert max_x == 100
    assert max_y == 200


def test_scene_bounds_empty():
    s = Scene()
    min_x, min_y, max_x, max_y = s.bounds()
    assert min_x == 0
    assert max_x == s.width


def test_scene_add_label():
    s = Scene()
    s.add_label(Label(id="l1", x=10, y=10, text="hello", bold=True, anchor="middle"))
    assert len(s.labels) == 1
    assert s.labels[0].bold


# -- HTML5 renderer -----------------------------------------------------------


def test_render_html_returns_string():
    t = Tree(4, name="viz-test")
    scene = scene_from_tree(t)
    html = render_html(scene, title="Test Viz")
    assert isinstance(html, str)
    assert "<!DOCTYPE html>" in html
    assert "<canvas" in html
    assert "</html>" in html


# -- widgets ------------------------------------------------------------------


def test_tree_widget_returns_string():
    t = Tree(9, name="widget-test")
    html = tree_widget(t)
    assert isinstance(html, str)
    assert "<!DOCTYPE html>" in html
    assert "widget-test" in html


def test_heatmap_widget_1d():
    arr = np.array([0, -50, 50, -128, 127, 0, 10, -10], dtype=np.int8)
    html = heatmap_widget(arr, title="test heatmap")
    assert isinstance(html, str)
    assert "<!DOCTYPE html>" in html
    assert "test heatmap" in html


def test_heatmap_widget_2d():
    arr = np.array([[0, -1], [1, 0]], dtype=np.int8)
    html = heatmap_widget(arr, cell=20)
    assert isinstance(html, str)
    assert "<!DOCTYPE html>" in html


def test_heatmap_widget_invalid_ndim():
    arr = np.zeros((2, 2, 2), dtype=np.int8)
    with pytest.raises(ValueError):
        heatmap_widget(arr)


# -- shaders ------------------------------------------------------------------


def test_heatmap_shader():
    shader = heatmap_shader()
    assert isinstance(shader, NodeShader)
    n = VisualNode(id="n", x=0, y=0, byte_value=50)
    shader.apply(n)
    assert n.color != (235, 235, 235)
    assert n.color[0] > 200  # reddish for positive


def test_heatmap_shader_zero():
    shader = heatmap_shader()
    n = VisualNode(id="n", x=0, y=0, byte_value=0)
    shader.apply(n)
    assert n.color == (235, 235, 235)


def test_sign_shader():
    shader = sign_shader()
    n_pos = VisualNode(id="p", x=0, y=0, byte_value=100)
    n_neg = VisualNode(id="n", x=0, y=0, byte_value=-100)
    n_zero = VisualNode(id="z", x=0, y=0, byte_value=0)
    shader.apply(n_pos)
    shader.apply(n_neg)
    shader.apply(n_zero)
    assert n_pos.color[0] > 200
    assert n_neg.color[2] > 200
    assert n_zero.color == (180, 180, 180)


def test_gradient_magnitude_shader():
    shader = gradient_magnitude_shader(min_radius=2, max_radius=30)
    n_big = VisualNode(id="b", x=0, y=0, byte_value=128)
    n_small = VisualNode(id="s", x=0, y=0, byte_value=1)
    shader.apply(n_big)
    shader.apply(n_small)
    assert n_big.radius > n_small.radius


def test_activation_history_shader():
    shader = activation_history_shader()
    n = VisualNode(id="n", x=0, y=0, byte_value=0, metadata={"activations": [-10, 0, 120]})
    shader.apply(n)
    assert n.opacity > 0.5


def test_edge_weight_shader():
    shader = edge_weight_shader()
    assert isinstance(shader, EdgeShader)
    e = VisualEdge(id="e", src_id="a", dst_id="b", weight=1)
    shader.apply(e)
    assert e.color[0] > 200


def test_apply_shaders_on_scene():
    s = Scene()
    s.add_node(VisualNode(id="n", x=0, y=0, byte_value=-128))
    s.add_edge(VisualEdge(id="e", src_id="n", dst_id="n", weight=-1))
    apply_shaders(s, node_shader=heatmap_shader(), edge_shader=edge_weight_shader())
    assert s.nodes[0].color[2] > 200  # blue for negative
    assert s.edges[0].color[2] > 200  # blue for negative weight


# -- animation ----------------------------------------------------------------


def test_animate_forward_pass():
    ug = UltraGraph("anim-test")
    a = Tree.dense(2, 4, name="a")
    b = Tree.dense(4, 2, name="b")
    ug.add(a)
    ug.add(b)
    ug.wire(a, b, "plain")

    html = animate_forward_pass(ug, duration=3.0, frames=2)
    assert isinstance(html, str)
    assert "<!DOCTYPE html>" in html
    assert "keyframes" in html


# -- export (pillow-dependent, skipped if not installed) ----------------------


def test_export_frame_creates_file(tmp_path):
    _ = pytest.importorskip("PIL")
    from ultragraph.vizard.export import export_frame

    t = Tree(9, name="frame-test")
    scene = scene_from_tree(t)
    path = tmp_path / "frame.png"
    export_frame(scene, str(path))
    assert path.exists()
    assert path.stat().st_size > 100


def test_export_gif_creates_animated_gif(tmp_path):
    pytest.importorskip("PIL")
    from ultragraph.vizard.anim import Animation
    from ultragraph.vizard.export import export_gif

    t = Tree(9, name="gif-test")
    scene = scene_from_tree(t)
    anim = Animation(scene=scene, duration=1.0, loop=False)
    anim.add_keyframe(Camera(x=0, y=0, zoom=1), 0.0)
    anim.add_keyframe(Camera(x=20, y=10, zoom=1.5), 1.0)

    path = tmp_path / "anim.gif"
    export_gif(anim, str(path), fps=4)
    assert path.exists()
    assert path.stat().st_size > 100


def test_export_mp4_skipped_without_imageio():
    try:
        import imageio  # noqa: F401
    except ImportError:
        pytest.skip("imageio not installed")
    from ultragraph.vizard.export import export_mp4
    assert callable(export_mp4)


# -- WebGL 3D viewer ----------------------------------------------------------


def test_render_webgl_returns_html():
    t = Tree(9, name="gl-test")
    scene = scene_from_tree(t)
    from ultragraph.vizard.renderer.webgl import render_webgl
    html = render_webgl(scene, title="3D Test")
    assert isinstance(html, str)
    assert "<!DOCTYPE html>" in html
    assert "webgl" in html.lower()


# -- dashboard ----------------------------------------------------------------


def test_dashboard_create():
    from ultragraph.vizard.dashboard import Dashboard
    dash = Dashboard(port=18765)
    assert dash.port == 18765
    assert dash._server is None


def test_dashboard_push_no_clients():
    from ultragraph.vizard.dashboard import Dashboard
    dash = Dashboard(port=18766)
    dash.push("loss", 0.5)


def test_dashboard_context_manager():
    from ultragraph.vizard.dashboard import Dashboard
    with Dashboard(port=18767) as dash:
        dash.push("loss", 1.0)
