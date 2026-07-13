"""Programmable shader system for ug-vizard.

Shaders are callables that transform visual properties of nodes and edges
based on their data. Like GLSL shaders but in pure Python — they take a
visual primitive, read its metadata, and return modified display properties.

Built-in presets:
- ``heatmap`` — color by byte value (blue→gray→red)
- ``gradient_magnitude`` — size by absolute byte value
- ``activation_history`` — opacity by recency in activation ring buffer
- ``sign`` — color by sign: red (+), blue (-), gray (0)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from ultragraph.vizard.core.scene import VisualEdge, VisualNode


def _byte_rgb(v: int) -> tuple[int, int, int]:
    """Map byte [-128, 127] to (r, g, b)."""
    gray = (235, 235, 235)
    neg = (40, 90, 220)
    pos = (220, 55, 45)
    if v == 0:
        return gray
    if v < 0:
        target = neg
        t = min(1.0, -v / 128.0)
    else:
        target = pos
        t = min(1.0, v / 127.0)
    return tuple(int(round(g + (h - g) * t)) for g, h in zip(gray, target))


@dataclass
class NodeShader:
    """Transforms VisualNode display properties."""

    name: str = ""
    _fn: Callable[[VisualNode], dict] | None = None

    def apply(self, node: VisualNode):
        if self._fn:
            updates = self._fn(node)
            for k, v in updates.items():
                setattr(node, k, v)


@dataclass
class EdgeShader:
    """Transforms VisualEdge display properties."""

    name: str = ""
    _fn: Callable[[VisualEdge], dict] | None = None

    def apply(self, edge: VisualEdge):
        if self._fn:
            updates = self._fn(edge)
            for k, v in updates.items():
                setattr(edge, k, v)


# -- built-in presets --------------------------------------------------------


def heatmap_shader() -> NodeShader:
    """Color nodes by byte value: blue (negative) → gray (zero) → red (positive)."""

    def fn(n: VisualNode) -> dict:
        return {"color": _byte_rgb(n.byte_value)}

    return NodeShader(name="heatmap", _fn=fn)


def gradient_magnitude_shader(min_radius: float = 4, max_radius: float = 20) -> NodeShader:
    """Size nodes by absolute byte value. Strong activations → larger circles."""

    def fn(n: VisualNode) -> dict:
        mag = abs(n.byte_value)
        frac = min(1.0, mag / 128.0)
        return {"radius": min_radius + (max_radius - min_radius) * frac}

    return NodeShader(name="gradient-magnitude", _fn=fn)


def sign_shader() -> NodeShader:
    """Color by sign only: red for positive, blue for negative, gray for zero."""

    def fn(n: VisualNode) -> dict:
        v = n.byte_value
        if v > 0:
            return {"color": (220, 55, 45)}
        elif v < 0:
            return {"color": (40, 90, 220)}
        return {"color": (180, 180, 180)}

    return NodeShader(name="sign", _fn=fn)


def activation_history_shader(history_key: str = "activations") -> NodeShader:
    """Fade opacity based on recency in activation ring buffer.

    Reads ``node.metadata[history_key]`` — a list of activation values
    where index 0 is oldest and -1 is most recent. Recent activations
    are opaque; stale ones fade toward transparent.
    """

    def fn(n: VisualNode) -> dict:
        history = n.metadata.get(history_key, [])
        if not history:
            return {"opacity": 0.3}
        recent = abs(history[-1]) if history else 0
        mag = min(1.0, recent / 128.0)
        return {"opacity": 0.15 + 0.85 * mag}

    return NodeShader(name="activation-history", _fn=fn)


def edge_weight_shader() -> EdgeShader:
    """Color edges by weight sign. Strong weights → thicker, more opaque."""

    def fn(e: VisualEdge) -> dict:
        w = e.weight
        if w > 0:
            color = (220, 55, 45)
        elif w < 0:
            color = (40, 90, 220)
        else:
            color = (160, 160, 170)
        mag = min(1.0, abs(w))
        return {
            "color": color,
            "width": 0.5 + 3.0 * mag,
            "opacity": 0.3 + 0.7 * mag,
        }

    return EdgeShader(name="edge-weight", _fn=fn)


# -- composite ---------------------------------------------------------------


def apply_shaders(
    scene,
    node_shader: NodeShader | None = None,
    edge_shader: EdgeShader | None = None,
):
    """Apply shaders to every node and edge in a scene."""
    if node_shader:
        for n in scene.nodes:
            node_shader.apply(n)
    if edge_shader:
        for e in scene.edges:
            edge_shader.apply(e)
