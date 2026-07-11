"""ultragraph.viz.mpl — optional matplotlib (PNG) rendering backend.

matplotlib is imported lazily inside each function; this module imports fine
without it (the pure-SVG backend in ultragraph.viz.svg is always available).
"""
from __future__ import annotations

import math

import numpy as np

# -- matplotlib (PNG) backend -----------------------------------------------
_MPL_ERR = "matplotlib is required for PNG rendering: uv sync --extra viz"
_MPL_KIND_COLOR = {"plain": "#4a6fa5", "residual": "#d08a2e"}
_MPL_KIND_DEFAULT = "#8a8f98"


def _finish_png(plt, fig, path):
    """Save ``fig`` to ``path`` (or return it) and close when saved."""
    if path is None:
        return fig
    fig.savefig(path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def tree_png(tree, path=None):
    """PNG micro view of one ``Tree`` via matplotlib.

    Dense trees render their ternary weight matrix as a diverging heatmap;
    sparse trees scatter their nodes with sign-coloured edge lines.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError(_MPL_ERR)

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(6, 5))
    fig.patch.set_facecolor("#12141a")
    ax.set_facecolor("#12141a")

    if tree.kind == "dense":
        wq = np.asarray(tree.wq)
        if wq.ndim == 1:
            wq = wq.reshape(1, -1)
        im = ax.imshow(wq, cmap="coolwarm", vmin=-1, vmax=1,
                       aspect="auto", interpolation="nearest")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xlabel("in")
        ax.set_ylabel("out")
    else:
        n = int(tree.n_nodes)
        cols = max(1, int(math.ceil(math.sqrt(n)))) if n else 1
        idx = np.arange(n)
        xs = (idx % cols).astype(float)
        ys = (idx // cols).astype(float)
        esrc = np.asarray(tree.edge_src)
        edst = np.asarray(tree.edge_dst)
        evals = np.asarray(tree.edge_val)
        for k in range(len(esrc)):
            s, d = int(esrc[k]), int(edst[k])
            if s >= n or d >= n:
                continue
            v = int(evals[k])
            c = "#d0554d" if v > 0 else ("#4a6fda" if v < 0 else "#9aa0a6")
            ax.annotate(
                "", xy=(xs[d], ys[d]), xytext=(xs[s], ys[s]),
                arrowprops=dict(arrowstyle="->", color=c, alpha=0.7, lw=1.3),
            )
        nodevals = np.asarray(tree.nodes).astype(float)
        sc = ax.scatter(xs, ys, c=nodevals, cmap="coolwarm",
                        vmin=-128, vmax=127, s=120, edgecolors="#cccccc",
                        linewidths=0.6, zorder=3)
        fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        ax.invert_yaxis()
        ax.set_aspect("equal")

    ax.set_title(str(tree.name))
    fig.tight_layout()
    return _finish_png(plt, fig, path)


def ultragraph_png(ug, path=None):
    """PNG macro view of an ``UltraGraph``: labelled tree nodes left-to-right
    with ultra-edges drawn as arrows coloured by kind, plus a legend."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except ImportError:
        raise ImportError(_MPL_ERR)

    plt.style.use("dark_background")
    trees = list(ug.trees)
    n = len(trees)
    fig, ax = plt.subplots(figsize=(max(6, 2 * n + 2), 4))
    fig.patch.set_facecolor("#12141a")
    ax.set_facecolor("#12141a")

    index = {id(t): i for i, t in enumerate(trees)}
    xs = [float(i) for i in range(n)]
    y = 0.0

    for e in ug.ultra_edges:
        i = index.get(id(e.src))
        j = index.get(id(e.dst))
        if i is None or j is None:
            continue
        c = _MPL_KIND_COLOR.get(e.kind, _MPL_KIND_DEFAULT)
        ax.annotate(
            "", xy=(xs[j], y), xytext=(xs[i], y),
            arrowprops=dict(arrowstyle="->", color=c, lw=2, alpha=0.85,
                            connectionstyle="arc3,rad=0.25"),
        )

    for i, t in enumerate(trees):
        ax.scatter([xs[i]], [y], s=1400, c="#2a2f3a",
                   edgecolors="#8899aa", linewidths=1.5, zorder=3)
        ax.text(xs[i], y, str(t.name), ha="center", va="center",
                color="#ffffff", fontsize=10, fontweight="bold", zorder=4)
        ax.text(xs[i], y - 0.35, f"{t.kind} | {int(t.n_nodes)}",
                ha="center", va="center", color="#aab", fontsize=8, zorder=4)

    handles = [
        Line2D([0], [0], color=_MPL_KIND_COLOR["plain"], lw=2, label="plain"),
        Line2D([0], [0], color=_MPL_KIND_COLOR["residual"], lw=2,
               label="residual"),
    ]
    ax.legend(handles=handles, loc="upper right", frameon=False)

    ax.set_title(f"{ug.name}  |  {n} trees, {len(ug.ultra_edges)} ultra-edges")
    ax.set_xlim(-0.7, (n - 1) + 0.7 if n else 0.7)
    ax.set_ylim(-1.0, 1.0)
    ax.axis("off")
    fig.tight_layout()
    return _finish_png(plt, fig, path)


def byte_heatmap_png(arr, path=None):
    """PNG heatmap of a 1-D or 2-D int8 array; 1-D is reshaped to a
    near-square grid before ``imshow``."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError(_MPL_ERR)

    a = np.asarray(arr)
    if a.ndim == 1:
        n = int(a.shape[0])
        cols = max(1, int(math.ceil(math.sqrt(n)))) if n else 1
        rows = int(math.ceil(n / cols)) if n else 0
        grid = np.zeros(rows * cols, dtype=np.int64)
        grid[:n] = a.astype(np.int64)
        grid = grid.reshape(rows, cols) if rows else grid.reshape(0, cols)
    elif a.ndim == 2:
        grid = a.astype(np.int64)
    else:
        raise ValueError("byte_heatmap_png expects a 1-D or 2-D array")

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(6, 5))
    fig.patch.set_facecolor("#12141a")
    ax.set_facecolor("#12141a")
    im = ax.imshow(grid, cmap="coolwarm", vmin=-128, vmax=127,
                   aspect="auto", interpolation="nearest")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return _finish_png(plt, fig, path)
