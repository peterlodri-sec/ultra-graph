"""Bridge: a lowered Vaked workflow becomes a ``SpacetimeGraph``, then a ``UGMZ`` world.

``vakedc lower`` turns a ``.vaked`` source into an artifact tree; the piece that
matters here is ``gen/workflow/wf.json`` -- the workflow's steps and edges::

    {"on": "e:created", "maxDepth": 3,
     "steps": [{"name": "A", "agent": "field.a"}, ...],
     "edges": [{"from": "A", "to": "B"}, ...]}

That is already a labeled graph, so the bridge is a *mapping*, not a rewrite:

* each step becomes a node, in declaration order, at position ``(i * spacing, 0, 0)``
  with ``spacing`` in raw Q16.16 (one world-unit apart);
* the node's int8 state is the step's ``state`` field when present, otherwise a
  deterministic 8-bit code of its ``agent`` -- an *encoding*, not a claim about
  what the agent means;
* each edge becomes a ternary causal link with delay ``dt`` (default 1).

The resulting world is emitted with :func:`ultragraph.zerocpu.emit.emit_ugm` and
ticked by the Rust ``world`` binary. Nothing here invents semantics the workflow
does not carry; it only fixes the shape.

``lower_vaked`` shells out to the canonical ``vakedc`` CLI. The copy vendored in
this repo is de-indented upstream, so by default it is *not* used: point
``VAKED_ROOT`` at a working vaked checkout (``python -m vakedc`` from its root).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from .emit import emit_ugm
from .fixed import ONE
from .spacetime import SpacetimeGraph

__all__ = [
    "WorkflowError",
    "workflow_to_graph",
    "workflow_meta",
    "lower_vaked",
    "vaked_to_ugm",
]


class WorkflowError(ValueError):
    """The lowered workflow JSON is not a graph we can map."""


def _label_code(text: str) -> int:
    """Deterministic int8 code for a label (same rule as ``ultragraph.vaked._code``)."""
    if not text:
        return 0
    h = 0
    for ch in text:
        h = (h * 31 + ord(ch)) & 0xFFFFFFFF
    return (h % 251) - 125


def _require(wf: dict, key: str, kind: type) -> object:
    if not isinstance(wf, dict):
        raise WorkflowError(f"workflow must be an object, got {type(wf).__name__}")
    if key not in wf:
        raise WorkflowError(f"workflow is missing {key!r}")
    value = wf[key]
    if not isinstance(value, kind):
        raise WorkflowError(f"{key!r} must be {kind.__name__}, got {type(value).__name__}")
    return value


def workflow_meta(wf: dict) -> dict:
    """The non-topological fields we do not encode in the world, kept for provenance."""
    return {
        "on": wf.get("on"),
        "maxDepth": wf.get("maxDepth"),
        "depth": wf.get("depth"),
        "log": wf.get("log"),
    }


def workflow_to_graph(
    wf: dict,
    *,
    spacing: int = ONE,
    name: str | None = None,
) -> SpacetimeGraph:
    """Map a lowered workflow (``gen/workflow/wf.json``) to a spacetime world."""
    steps = _require(wf, "steps", list)
    edges = _require(wf, "edges", list)
    if not steps:
        raise WorkflowError("workflow has no steps")

    graph = SpacetimeGraph(name or str(wf.get("workflow") or "vaked-workflow"))
    index: dict[str, int] = {}
    for i, step in enumerate(steps):
        if not isinstance(step, dict) or "name" not in step:
            raise WorkflowError(f"step #{i} needs a 'name'")
        label = str(step["name"])
        if label in index:
            raise WorkflowError(f"duplicate step name {label!r}")
        state = step.get("state")
        if state is None:
            state = _label_code(str(step.get("agent", "")))
        index[label] = graph.add_node(int(state), (i * spacing, 0, 0))

    for i, edge in enumerate(edges):
        if not isinstance(edge, dict):
            raise WorkflowError(f"edge #{i} must be an object")
        src, dst = edge.get("from"), edge.get("to")
        if src not in index or dst not in index:
            raise WorkflowError(f"edge #{i} references unknown step: {src!r} -> {dst!r}")
        weight = int(edge.get("weight", 1))
        if weight not in (-1, 0, 1):
            raise WorkflowError(f"edge #{i} weight must be ternary, got {weight}")
        dt = int(edge.get("dt", 1))
        if dt < 0:
            raise WorkflowError(f"edge #{i} dt must be >= 0, got {dt}")
        graph.add_edge(index[src], index[dst], weight, dt)

    return graph


def _vaked_python_root() -> str | None:
    """Where to run ``python -m vakedc`` from: ``$VAKED_ROOT`` if it has a vakedc."""
    root = os.environ.get("VAKED_ROOT")
    if root and (Path(root) / "vakedc" / "__init__.py").is_file():
        return root
    return None


def lower_vaked(source, out_dir=None, *, timeout: int = 180) -> dict:
    """Run ``python -m vakedc lower`` on ``source`` and return the workflow JSON.

    Requires a working ``vakedc`` (see module docstring). Raises
    :class:`WorkflowError` when the compiler is unavailable or emits no workflow.
    """
    root = _vaked_python_root()
    if root is None:
        raise WorkflowError(
            "no usable vakedc: set VAKED_ROOT to a vaked checkout "
            "(the copy vendored in this repo is stored de-indented upstream)"
        )
    source = Path(source).resolve()
    if not source.is_file():
        raise WorkflowError(f"no such .vaked source: {source}")
    tmp = None
    if out_dir is None:
        tmp = tempfile.mkdtemp(prefix="vaked-lower-")
        out_dir = tmp
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "vakedc", "lower", str(source), "--out", str(out_dir)],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        if proc.returncode != 0:
            tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-3:]
            raise WorkflowError(f"vakedc lower failed ({proc.returncode}): " + " | ".join(tail))
        wf_path = Path(out_dir) / "gen" / "workflow" / "wf.json"
        if not wf_path.is_file():
            raise WorkflowError(f"vakedc lower produced no workflow JSON under {out_dir}")
        return json.loads(wf_path.read_text(encoding="utf-8"))
    finally:
        if tmp is not None:
            import shutil

            shutil.rmtree(tmp, ignore_errors=True)


def vaked_to_ugm(source, ugmz_path, *, out_dir=None) -> dict:
    """``.vaked`` -> lowered workflow -> ``SpacetimeGraph`` -> ``.ugmz`` artifact.

    Returns the workflow metadata plus the artifact path and world shape.
    """
    wf = lower_vaked(source, out_dir)
    graph = workflow_to_graph(wf, name=Path(source).stem)
    path = emit_ugm(graph, ugmz_path)
    return {
        "artifact": path,
        "meta": workflow_meta(wf),
        "nodes": graph.n_nodes,
        "edges": graph.n_edges,
        "tick": graph.tick_count,
    }
