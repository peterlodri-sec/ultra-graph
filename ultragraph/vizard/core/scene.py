"""Scene graph — the visual model of a byte-graph.

A Scene holds a collection of visual nodes and edges (not to be confused
with ultragraph nodes/edges — these are visual primitives). The scene graph
is renderer-agnostic: the same scene can be rendered as SVG, HTML5 Canvas,
or WebGL without changes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal


@dataclass
class Camera:
    """Viewpoint for 2D/3D scene rendering."""

    x: float = 0.0
    y: float = 0.0
    z: float = 100.0
    zoom: float = 1.0
    rotation_x: float = 0.0
    rotation_y: float = 0.0
    rotation_z: float = 0.0

    def orbit(self, dx: float = 0.0, dy: float = 0.0):
        self.rotation_y += dx * 0.01
        self.rotation_x += dy * 0.01
        self.rotation_x = max(-math.pi / 2, min(math.pi / 2, self.rotation_x))

    def dolly(self, delta: float):
        self.z = max(1.0, self.z + delta)

    def pan(self, dx: float = 0.0, dy: float = 0.0):
        self.x += dx
        self.y += dy


@dataclass
class VisualNode:
    """A visual circle/point in the scene, backed by a byte value."""

    id: str
    x: float
    y: float
    z: float = 0.0
    radius: float = 10.0
    byte_value: int = 0
    label: str = ""
    color: tuple[int, int, int] = (100, 100, 220)
    opacity: float = 1.0
    metadata: dict = field(default_factory=dict)


@dataclass
class VisualEdge:
    """A visual line/curve connecting two visual nodes."""

    id: str
    src_id: str
    dst_id: str
    weight: int = 0
    kind: Literal["plain", "residual", "dashed"] = "plain"
    color: tuple[int, int, int] = (120, 120, 140)
    width: float = 1.5
    opacity: float = 0.7
    metadata: dict = field(default_factory=dict)


@dataclass
class Label:
    """Floating text annotation."""

    id: str
    x: float
    y: float
    text: str
    font_size: int = 12
    color: tuple[int, int, int] = (50, 50, 50)
    bold: bool = False
    anchor: Literal["start", "middle", "end"] = "start"


@dataclass
class Scene:
    """A renderer-agnostic scene graph.

    Holds all visual primitives and a camera. Renderers consume the scene
    and produce output in their target format.
    """

    nodes: list[VisualNode] = field(default_factory=list)
    edges: list[VisualEdge] = field(default_factory=list)
    labels: list[Label] = field(default_factory=list)
    camera: Camera = field(default_factory=Camera)
    width: int = 800
    height: int = 600
    background: tuple[int, int, int] = (255, 255, 255)
    title: str = ""

    def add_node(self, node: VisualNode):
        self.nodes.append(node)

    def add_edge(self, edge: VisualEdge):
        self.edges.append(edge)

    def add_label(self, label: Label):
        self.labels.append(label)

    def bounds(self) -> tuple[float, float, float, float]:
        """Compute bounding box of all nodes. Returns (min_x, min_y, max_x, max_y)."""
        if not self.nodes:
            return (0, 0, self.width, self.height)
        xs = [n.x for n in self.nodes]
        ys = [n.y for n in self.nodes]
        return (min(xs), min(ys), max(xs), max(ys))


def scene_from_tree(tree) -> Scene:
    """Build a scene from an ultra-graph Tree.

    Maps each tree node to a VisualNode, each micro-edge to a VisualEdge.
    Byte values drive colors: negative→blue, zero→gray, positive→red.
    """
    from ultragraph.viz.svg import _byte_rgb

    n = int(tree.n_nodes)
    if n == 0:
        return Scene(title=tree.name)

    cols = max(1, int(math.ceil(math.sqrt(n))))
    step = 40
    pad = 40

    scene = Scene(width=800, height=600, title=tree.name)
    nodes_arr = tree.nodes

    for i in range(min(n, 256)):
        row, col = divmod(i, cols)
        vx = pad + col * step
        vy = pad + row * step
        val = int(nodes_arr[i])
        scene.add_node(
            VisualNode(
                id=f"n{i}",
                x=vx,
                y=vy,
                byte_value=val,
                label=str(val),
                color=_byte_rgb(val),
                radius=14,
            )
        )

    if tree.kind == "sparse":
        esrc = tree.edge_src
        edst = tree.edge_dst
        evals = tree.edge_val
        for k in range(min(len(esrc), 256)):
            s, d = int(esrc[k]), int(edst[k])
            w = int(evals[k])
            scene.add_edge(
                VisualEdge(
                    id=f"e{k}",
                    src_id=f"n{s}",
                    dst_id=f"n{d}",
                    weight=w,
                    kind="plain",
                    width=1.5,
                )
            )

    return scene


def scene_from_ultragraph(ug) -> Scene:
    """Build a scene from an UltraGraph.

    Each tree becomes a VisualNode (box), each ultra-edge a VisualEdge.
    """
    trees = list(ug.trees)
    n = len(trees)
    if n == 0:
        return Scene(title=ug.name)

    box_w = 120
    gap = 40
    pad = 40
    step = box_w + gap

    scene = Scene(width=max(400, pad * 2 + n * step), height=300, title=ug.name)

    for i, t in enumerate(trees):
        vx = pad + i * step
        vy = 80
        scene.add_node(
            VisualNode(
                id=f"tree{i}",
                x=vx + box_w / 2,
                y=vy + 30,
                radius=28,
                byte_value=0,
                label=t.name,
                color=(220, 225, 240),
            )
        )
        scene.add_label(
            Label(
                id=f"label{i}",
                x=vx + box_w / 2,
                y=vy + 70,
                text=f"{t.kind} | {int(t.n_nodes)} nodes",
                font_size=10,
                anchor="middle",
            )
        )

    index = {id(t): i for i, t in enumerate(trees)}
    for e in ug.ultra_edges:
        si = index.get(id(e.src))
        di = index.get(id(e.dst))
        if si is not None and di is not None:
            scene.add_edge(
                VisualEdge(
                    id=f"ue{si}{di}",
                    src_id=f"tree{si}",
                    dst_id=f"tree{di}",
                    kind=e.kind,
                    width=3.0,
                )
            )

    return scene
