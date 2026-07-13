"""Tests for ug-vizard — pure visualization sub-module."""

from __future__ import annotations

import numpy as np

from ultragraph.core import Tree, UltraGraph
from ultragraph.vizard import (
    Camera,
    Scene,
    animate_forward_pass,
    heatmap_widget,
    render_html,
    tree_widget,
)
from ultragraph.vizard.core.scene import scene_from_tree, scene_from_ultragraph


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


def test_render_html_returns_string():
    t = Tree(4, name="viz-test")
    scene = scene_from_tree(t)
    html = render_html(scene, title="Test Viz")
    assert isinstance(html, str)
    assert "<!DOCTYPE html>" in html
    assert "<canvas" in html
    assert "</html>" in html


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
    try:
        heatmap_widget(arr)
        assert False, "should have raised"
    except ValueError:
        pass


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


def test_scene_bounds():
    s = Scene()
    from ultragraph.vizard.core.scene import VisualNode
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
