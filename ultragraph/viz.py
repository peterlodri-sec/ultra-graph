"""Pure-stdlib SVG visualisation for byte-graphs.

No plotting dependencies: every public function returns a self-contained SVG
string (it starts with ``<svg`` and ends with ``</svg>``) built purely from
string formatting, so the output is deterministic and free of randomness.
``numpy`` is used only to read the int8 arrays that live on a ``Tree``.

Public API
----------
* ``tree_svg(tree)``           -- micro view of a single ``Tree``.
* ``ultragraph_svg(ug)``       -- macro view of a whole ``UltraGraph``.
* ``byte_heatmap_svg(arr)``    -- grid heatmap of a 1-D/2-D int8 array.

Colour convention for a byte value ``v`` in ``[-128, 127]``:
negative -> blue, zero -> light gray, positive -> red (linear on intensity).
"""
from __future__ import annotations

import math

import numpy as np

_GRAY = (235, 235, 235)          # zero
_NEG = (40, 90, 220)             # most-negative byte (blue)
_POS = (220, 55, 45)             # most-positive byte (red)

_KIND_COLOR = {"plain": "#4a6fa5", "residual": "#d08a2e"}
_KIND_DEFAULT = "#8a8f98"


# -- small helpers ----------------------------------------------------------
def _esc(s) -> str:
    """Escape text for safe inclusion in SVG markup."""
    return (
        str(s)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )


def _hexrgb(rgb) -> str:
    r, g, b = (int(max(0, min(255, round(c)))) for c in rgb)
    return f"#{r:02x}{g:02x}{b:02x}"


def _byte_rgb(v) -> tuple:
    """Map a byte value in [-128, 127] to an (r, g, b) tuple."""
    v = int(v)
    if v == 0:
        return _GRAY
    target, t = (_NEG, min(1.0, -v / 128.0)) if v < 0 else (_POS, min(1.0, v / 127.0))
    return tuple(round(g + (h - g) * t) for g, h in zip(_GRAY, target))


def _byte_color(v) -> str:
    return _hexrgb(_byte_rgb(v))


def _text_on(rgb) -> str:
    """Pick a readable text colour for a filled cell/circle."""
    lum = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
    return "#ffffff" if lum < 140 else "#222222"


def _sign_color(v) -> str:
    if v > 0:
        return _hexrgb(_POS)
    if v < 0:
        return _hexrgb(_NEG)
    return "#9aa0a6"


def _svg_open(w, h) -> str:
    w, h = int(math.ceil(w)), int(math.ceil(h))
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
        f'viewBox="0 0 {w} {h}" font-family="sans-serif" font-size="11">'
    )


def _bg(w, h) -> str:
    return f'<rect x="0" y="0" width="{int(math.ceil(w))}" height="{int(math.ceil(h))}" fill="#ffffff"/>'


def _heatmap_cells(a2d, x0, y0, cell, gap):
    """Emit ``<rect>`` cells for a 2-D array; return (parts, width, height)."""
    a2d = np.asarray(a2d)
    rows = int(a2d.shape[0]) if a2d.ndim >= 1 else 0
    cols = int(a2d.shape[1]) if a2d.ndim >= 2 else 0
    parts = []
    for r in range(rows):
        row = a2d[r]
        for c in range(cols):
            x = x0 + c * (cell + gap)
            y = y0 + r * (cell + gap)
            parts.append(
                f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" '
                f'fill="{_byte_color(int(row[c]))}"/>'
            )
    w = cols * (cell + gap) - gap if cols else 0
    h = rows * (cell + gap) - gap if rows else 0
    return parts, w, h


def _subsample(a, max_r, max_c):
    """Evenly downsample a 2-D array to at most ``max_r`` x ``max_c`` cells."""
    r = int(a.shape[0]) if a.ndim >= 1 else 0
    c = int(a.shape[1]) if a.ndim >= 2 else 0
    if r == 0 or c == 0:
        return np.asarray(a)
    ri = np.unique(np.linspace(0, r - 1, min(r, max_r)).round().astype(int))
    ci = np.unique(np.linspace(0, c - 1, min(c, max_c)).round().astype(int))
    return np.asarray(a)[np.ix_(ri, ci)]


# -- public API -------------------------------------------------------------
def byte_heatmap_svg(arr, cols=None, cell=14, gap=1, pad=6) -> str:
    """Render a 1-D or 2-D int8 array as a grid of byte-coloured cells."""
    a = np.asarray(arr)
    if a.ndim == 1:
        n = int(a.shape[0])
        if cols is None:
            cols = max(1, int(math.ceil(math.sqrt(n)))) if n else 1
        cols = max(1, int(cols))
        rows = int(math.ceil(n / cols)) if n else 0
        flat = a.astype(np.int64)
        grid = np.zeros((rows, cols), dtype=np.int64)
        mask = np.zeros((rows, cols), dtype=bool)
        for i in range(n):
            grid[i // cols, i % cols] = flat[i]
            mask[i // cols, i % cols] = True
    elif a.ndim == 2:
        rows, cols = int(a.shape[0]), int(a.shape[1])
        grid = a.astype(np.int64)
        mask = np.ones((rows, cols), dtype=bool)
    else:
        raise ValueError("byte_heatmap_svg expects a 1-D or 2-D array")

    w = pad * 2 + (cols * (cell + gap) - gap if cols else 0)
    h = pad * 2 + (rows * (cell + gap) - gap if rows else 0)

    parts = [_svg_open(w, h), _bg(w, h)]
    for r in range(rows):
        for c in range(cols):
            if not mask[r, c]:
                continue
            x = pad + c * (cell + gap)
            y = pad + r * (cell + gap)
            parts.append(
                f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" '
                f'fill="{_byte_color(int(grid[r, c]))}"/>'
            )
    parts.append("</svg>")
    return "\n".join(parts)


def tree_svg(tree, node_r=13, node_gap=16, pad=12, header_h=28, max_nodes=256) -> str:
    """Micro view of one tree: node circles coloured by byte value, plus a
    dense weight-matrix heatmap or sparse sign-coloured edges."""
    n = int(tree.n_nodes)
    draw_n = min(n, max_nodes)
    nodes = np.asarray(tree.nodes).astype(np.int64)
    cols = max(1, int(math.ceil(math.sqrt(draw_n)))) if draw_n else 1
    rows = int(math.ceil(draw_n / cols)) if draw_n else 0
    step = 2 * node_r + node_gap
    top = header_h + pad

    def node_xy(i):
        r, c = divmod(i, cols)
        return pad + node_r + c * step, top + node_r + r * step

    nodes_right = pad + (cols * step - node_gap) if draw_n else pad
    nodes_bottom = top + (rows * step - node_gap) if draw_n else top

    body = []

    # sparse: edges drawn under the circles, coloured by sign of the weight
    if tree.kind == "sparse":
        esrc = np.asarray(tree.edge_src)
        edst = np.asarray(tree.edge_dst)
        evals = np.asarray(tree.edge_val)
        for k in range(min(len(esrc), 512)):
            s, d = int(esrc[k]), int(edst[k])
            if s >= draw_n or d >= draw_n:
                continue
            x1, y1 = node_xy(s)
            x2, y2 = node_xy(d)
            body.append(
                f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
                f'stroke="{_sign_color(int(evals[k]))}" stroke-width="1.5" opacity="0.7"/>'
            )

    # node circles
    for i in range(draw_n):
        cx, cy = node_xy(i)
        rgb = _byte_rgb(nodes[i])
        body.append(
            f'<circle cx="{cx}" cy="{cy}" r="{node_r}" fill="{_hexrgb(rgb)}" '
            f'stroke="#555" stroke-width="0.8"/>'
        )
        body.append(
            f'<text x="{cx}" y="{cy + 3.5}" text-anchor="middle" '
            f'fill="{_text_on(rgb)}" font-size="9">{int(nodes[i])}</text>'
        )

    right, bottom = nodes_right, nodes_bottom

    # dense: summarise the ternary weight matrix as a (sub-sampled) heatmap
    if tree.kind == "dense" and getattr(tree, "wq", None) is not None:
        wq = np.asarray(tree.wq)
        sub = _subsample(wq, 20, 28)
        label_y = nodes_bottom + 16
        hm_y = label_y + 8
        cells, hw, hh = _heatmap_cells(sub, pad, hm_y, 8, 1)
        body.append(
            f'<text x="{pad}" y="{label_y}" fill="#333">'
            f'wq {int(wq.shape[0])}x{int(wq.shape[1]) if wq.ndim > 1 else 1} '
            f'{_esc("scale=%s" % format(float(tree.w_scale), ".3g"))}</text>'
        )
        body.extend(cells)
        right = max(right, pad + hw)
        bottom = hm_y + hh

    if draw_n < n:
        body.append(
            f'<text x="{pad}" y="{bottom + 14}" fill="#777" font-size="10">'
            f'(+{n - draw_n} more nodes)</text>'
        )
        bottom += 18

    title = f"{tree.name}  |  {tree.kind} | {n} nodes"
    w = max(right, pad + 7 * len(title)) + pad
    h = bottom + pad
    parts = [
        _svg_open(w, h),
        _bg(w, h),
        f'<text x="{pad}" y="18" fill="#111" font-size="13" '
        f'font-weight="bold">{_esc(title)}</text>',
    ]
    parts.extend(body)
    parts.append("</svg>")
    return "\n".join(parts)


def ultragraph_svg(ug, box_w=128, box_h=64, gap_x=64, pad=16) -> str:
    """Macro view: each tree is a labelled box, each ultra-edge a triple-line
    ``===`` connector coloured by kind, with a legend."""
    trees = list(ug.trees)
    n = len(trees)
    title_h, legend_h = 24, 26
    box_y = title_h + legend_h + pad
    index = {id(t): i for i, t in enumerate(trees)}

    def box_x(i):
        return pad + i * (box_w + gap_x)

    body = []

    # ultra-edges (drawn behind the boxes so endpoints tuck under the borders)
    for e in ug.ultra_edges:
        i = index.get(id(e.src))
        j = index.get(id(e.dst))
        if i is None or j is None:
            continue
        color = _KIND_COLOR.get(e.kind, _KIND_DEFAULT)
        if j >= i:
            x1, x2 = box_x(i) + box_w, box_x(j)
        else:
            x1, x2 = box_x(i), box_x(j) + box_w
        ym = box_y + box_h / 2
        for dy in (-4, 0, 4):
            body.append(
                f'<line x1="{x1}" y1="{ym + dy}" x2="{x2}" y2="{ym + dy}" '
                f'stroke="{color}" stroke-width="2" opacity="0.85"/>'
            )

    # tree boxes
    for i, t in enumerate(trees):
        x = box_x(i)
        body.append(
            f'<rect x="{x}" y="{box_y}" width="{box_w}" height="{box_h}" rx="8" '
            f'fill="#eef2f7" stroke="#334" stroke-width="1.5"/>'
        )
        body.append(
            f'<text x="{x + box_w / 2}" y="{box_y + box_h / 2 - 2}" '
            f'text-anchor="middle" fill="#111" font-size="12" '
            f'font-weight="bold">{_esc(t.name)}</text>'
        )
        body.append(
            f'<text x="{x + box_w / 2}" y="{box_y + box_h / 2 + 14}" '
            f'text-anchor="middle" fill="#555" font-size="10">'
            f'{_esc(t.kind)} | {int(t.n_nodes)}</text>'
        )

    # legend
    legend_y = title_h + 6
    lx = pad
    for kind in ("plain", "residual"):
        body.append(
            f'<rect x="{lx}" y="{legend_y}" width="16" height="10" '
            f'fill="{_KIND_COLOR[kind]}"/>'
        )
        body.append(
            f'<text x="{lx + 22}" y="{legend_y + 9}" fill="#333" '
            f'font-size="10">{kind}</text>'
        )
        lx += 90

    content_right = (pad + n * box_w + (n - 1) * gap_x) if n else 2 * pad
    title = f"{ug.name}  |  {n} trees, {len(ug.ultra_edges)} ultra-edges"
    w = max(content_right, lx, pad + 7 * len(title)) + pad
    h = box_y + box_h + pad

    parts = [
        _svg_open(w, h),
        _bg(w, h),
        f'<text x="{pad}" y="17" fill="#111" font-size="14" '
        f'font-weight="bold">{_esc(title)}</text>',
    ]
    parts.extend(body)
    parts.append("</svg>")
    return "\n".join(parts)
