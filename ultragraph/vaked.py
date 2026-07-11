"""Optional bridge to the vaked language — lower a vaked graph into an ultragraph.

`lower_graph(nodes, edges)` turns any labeled property graph (nodes with an ``id``
and optional ``kind``; edges with ``src``/``dst``/``label``) into a sparse
ultragraph :class:`~ultragraph.core.Tree` — the byte-graph representation, where
node bytes encode the node kind and edge bytes encode the edge label.

`compile_vaked(source)` runs the (vendored, optional) vakedc compiler front-end to
produce that graph first. It is optional: it requires an importable ``vakedc``
(vendored at the repo root). ultragraph itself never imports vakedc at module load.
"""
from __future__ import annotations

from .core import Tree


def _get(o, key, default=None):
    return o.get(key, default) if isinstance(o, dict) else getattr(o, key, default)


def _code(s: str) -> int:
    """Deterministic small signed-byte code for a string label/kind (int8 range)."""
    if not s:
        return 0
    h = 0
    for ch in s:
        h = (h * 31 + ord(ch)) & 0xFFFFFFFF
    return (h % 251) - 125


def lower_graph(nodes, edges, name: str = "vaked") -> Tree:
    """Lower a labeled property graph to a sparse ultragraph byte-graph ``Tree``.

    ``nodes``: iterable of items exposing ``id`` (and optionally ``kind``).
    ``edges``: iterable exposing ``src``, ``dst`` (node ids) and optionally ``label``.
    Edges referencing unknown node ids are skipped. The node ids, kinds and edge
    labels are stashed in ``tree.adhoc`` for inspection.
    """
    nodes = list(nodes)
    ids = [_get(n, "id") for n in nodes]
    index = {nid: i for i, nid in enumerate(ids) if nid is not None}
    tree = Tree(len(nodes), name=name)
    kinds = [str(_get(n, "kind", "")) for n in nodes]
    for i, kind in enumerate(kinds):
        tree[i] = _code(kind)
    labels = []
    for e in edges:
        s = index.get(_get(e, "src", _get(e, "source")))
        d = index.get(_get(e, "dst", _get(e, "target")))
        if s is None or d is None:
            continue
        lab = str(_get(e, "label", ""))
        tree.add_edge(s, d, _code(lab))
        labels.append(lab)
    tree.adhoc.update({"vaked_ids": ids, "vaked_kinds": kinds, "vaked_labels": labels})
    return tree


def compile_vaked(source: str, filename: str = "<vaked>") -> Tree:
    """Compile vaked ``source`` to an ultragraph byte-graph via the vendored vakedc
    front-end (lex → parse → resolve), then :func:`lower_graph`.

    Optional: requires an importable ``vakedc`` (vendored at the repo root). Raises
    ``ImportError`` with guidance when vakedc is unavailable or de-indented upstream.
    """
    try:
        from vakedc import parse_string
    except Exception as ex:  # noqa: BLE001 - optional dep; may be de-indented upstream
        raise ImportError(
            "vaked support needs an importable `vakedc` (vendored at the repo root; "
            "some upstream files are stored de-indented and must be fixed first): " + repr(ex)
        ) from ex
    graph = parse_string(source, filename)  # tokenize -> parse -> resolve -> LPG
    return lower_graph(graph.nodes, graph.edges, name=filename)
