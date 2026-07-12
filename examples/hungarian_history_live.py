"""Hungarian history as an ultra-graph — built *live* from hu.wikipedia.

Seeds a BFS with ~10 real Hungarian-history article titles, fetches them (and a
capped set of the pages they link to) via ``ultragraph.wiki``, and wires a single
sparse ``Tree`` whose micro-edges are the inter-article links. Renders a themed
circular node-link SVG reading titles from ``tree.adhoc["labels"]``.

    uv run --extra wiki python examples/hungarian_history_live.py

Writes `examples/data/hungarian_history_live.svg`. Page records are cached under
`examples/data/.wikicache/`, so a second run is offline and instant.
"""

import math
import os

from ultragraph.wiki import build_wiki_graph

OUT = os.path.join(os.path.dirname(__file__), "data")

# pocoo palette: dark bg, cyan nodes, faint edges
BG, NODE, EDGE, TEXT, FAINT = "#070b16", "#00d4ff", "#1b2740", "#c8d4e8", "#5a6b8c"

SEEDS = [
    "Honfoglalás",
    "Árpád (honfoglaló)",
    "I. István magyar király",
    "Tatárjárás",
    "Mohácsi csata",
    "Az 1848–49-es forradalom és szabadságharc",
    "Osztrák–Magyar Monarchia",
    "Trianoni békeszerződés",
    "Az 1956-os forradalom",
    "Rendszerváltás Magyarországon",
]


def _esc(s) -> str:
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def wiki_svg(tree, size: int = 1400) -> str:
    """A force-free circular node-link view of the live wiki tree."""
    labels = tree.adhoc["labels"]
    n = len(labels)
    cx = cy = size / 2
    radius = size / 2 - 260
    pos = []
    for i in range(n):
        ang = 2 * math.pi * i / max(1, n) - math.pi / 2
        pos.append((cx + radius * math.cos(ang), cy + radius * math.sin(ang), math.cos(ang)))

    n_edges = len(tree.edge_src)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {size} {size}" '
        f'width="{size}" height="{size}" font-family="ui-monospace,Menlo,monospace">',
        f'<rect width="{size}" height="{size}" fill="{BG}"/>',
        f'<text x="{cx:.0f}" y="46" text-anchor="middle" fill="{TEXT}" font-size="26" '
        f'font-weight="700">Magyar történelem — live from hu.wikipedia</text>',
        f'<text x="{cx:.0f}" y="74" text-anchor="middle" fill="{FAINT}" font-size="14">'
        f'{n} nodes, {n_edges} micro-edges — built by ultragraph.wiki</text>',
    ]

    esrc, edst = tree.edge_src, tree.edge_dst
    for k in range(n_edges):
        a, b = int(esrc[k]), int(edst[k])
        x1, y1, _ = pos[a]
        x2, y2, _ = pos[b]
        parts.append(
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke="{EDGE}" stroke-width="1" opacity="0.55"/>'
        )

    for i, (title, _summary) in enumerate(labels):
        x, y, cosang = pos[i]
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="16" fill="{NODE}" opacity="0.16"/>')
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6" fill="{NODE}"/>')
        anchor = "start" if cosang >= 0 else "end"
        tx = x + (14 if cosang >= 0 else -14)
        parts.append(
            f'<text x="{tx:.1f}" y="{y + 4:.1f}" text-anchor="{anchor}" '
            f'fill="{TEXT}" font-size="12.5">{_esc(title)}</text>'
        )

    parts.append("</svg>")
    return "".join(parts)


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    print(f"building wiki graph from {len(SEEDS)} Hungarian-history seeds (hu.wikipedia)...")
    ug = build_wiki_graph(SEEDS, lang="hu", max_pages=40)
    tree = ug.trees[0]
    n_nodes, n_edges = tree.n_nodes, len(tree.edge_src)
    print(f"\nultra-graph '{ug.name}': {len(ug.trees)} tree, {n_nodes} nodes, {n_edges} micro-edges")

    if not n_nodes:
        print("no pages fetched — network likely blocked; code path is intact, cache is empty.")
        return

    print("sample titles:")
    for title, _ in tree.adhoc["labels"][:5]:
        print(f"  - {title}")
    path = os.path.join(OUT, "hungarian_history_live.svg")
    with open(path, "w", encoding="utf-8") as f:
        f.write(wiki_svg(tree))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
