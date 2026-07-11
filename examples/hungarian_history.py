"""Hungarian history as an ultra-graph — the whole arc, from Scythia to the EU.

A semi-static knowledge graph built with the ultragraph data model: each **era** is
a sparse ``Tree`` (its nodes = people / events / places, carried in the ad-hoc side
store), micro-edges wire relations *within* an era, and **ultra-edges** (``===``) wire
the eras chronologically into one ``UltraGraph`` — the whole of Hungarian history as a
single byte-graph.

    uv run python examples/hungarian_history.py

Writes `examples/data/hungarian_history.svg` (a themed timeline) and
`hungarian_history_macro.svg` (the library's own ultra-graph view).
"""
from __future__ import annotations

import os

from ultragraph import Tree, UltraGraph
from ultragraph.viz import ultragraph_svg

OUT = os.path.join(os.path.dirname(__file__), "data")

# kind -> palette (person=gold, event=cyan, place=green, idea/law/work=purple)
COL = {"person": "#e6c34d", "event": "#00d4ff", "place": "#00e660", "idea": "#b48bff"}
BG, TEXT, FAINT, RAIL = "#070b16", "#c8d4e8", "#5a6b8c", "#1b2740"

# (title, span, [(label, kind, year)], [(i, j, relation)])
ERAS = [
    ("Eredet — Origins", "before 895", [
        ("Szkítia (Scythia)", "place", None), ("Levédia", "place", None), ("Etelköz", "place", None),
        ("Ügek", "person", None), ("Álmos", "person", 819), ("hét vezér (hetumoger)", "idea", None),
    ], [(0, 1, "migrate"), (1, 2, "migrate"), (3, 4, "father of"), (4, 5, "leads")]),
    ("Honfoglalás — Conquest", "895–900", [
        ("Árpád", "person", 895), ("Vereckei-hágó", "place", 895), ("vérszerződés (blood oath)", "idea", None),
        ("Pannónia", "place", 900), ("Kárpát-medence", "place", 895),
    ], [(0, 2, "sworn by"), (0, 1, "crosses"), (1, 4, "into"), (4, 3, "settles")]),
    ("Kalandozások — Raids", "900–955", [
        ("nyugati kalandozások", "event", 910), ("Augsburg / Lechfeld", "event", 955),
        ("Bulcsú", "person", 955), ("vereség (defeat)", "idea", 955),
    ], [(0, 1, "ends at"), (2, 1, "led"), (1, 3, "brings")]),
    ("Államalapítás — Founding", "972–1038", [
        ("Géza", "person", 972), ("Koppány", "person", 997), ("István I (Szent István)", "person", 1000),
        ("koronázás 1000", "event", 1000), ("kereszténység", "idea", 1000),
    ], [(0, 2, "father of"), (2, 1, "defeats"), (2, 3, "crowned"), (3, 4, "adopts")]),
    ("Árpád-ház — Árpád dynasty", "1038–1301", [
        ("László I (Szent)", "person", 1077), ("Könyves Kálmán", "person", 1095),
        ("Aranybulla (Golden Bull)", "idea", 1222), ("tatárjárás (Mongols)", "event", 1241),
        ("IV. Béla", "person", 1241), ("III. András", "person", 1301),
    ], [(0, 1, "succeeded by"), (1, 2, "leads to"), (2, 3, "before"), (4, 3, "rebuilds after"), (4, 5, "line ends")]),
    ("Anjou & Luxemburg", "1301–1437", [
        ("Károly Róbert", "person", 1308), ("Nagy Lajos", "person", 1342),
        ("Luxemburgi Zsigmond", "person", 1387),
    ], [(0, 1, "father of"), (1, 2, "succeeded by")]),
    ("Hunyadiak — Hunyadis", "1443–1490", [
        ("Hunyadi János", "person", 1443), ("Nándorfehérvár 1456", "event", 1456),
        ("déli harangszó (noon bells)", "idea", 1456), ("Mátyás király (Corvinus)", "person", 1458),
        ("Bibliotheca Corviniana", "idea", 1470),
    ], [(0, 1, "defends"), (1, 2, "commemorated"), (0, 3, "father of"), (3, 4, "founds")]),
    ("Török kor — Ottoman", "1526–1699", [
        ("Mohács 1526", "event", 1526), ("II. Lajos", "person", 1526), ("három részre szakadás", "idea", 1541),
        ("Eger 1552 (Dobó)", "event", 1552), ("Buda 1541→1686", "place", 1686),
    ], [(1, 0, "dies at"), (0, 2, "leads to"), (2, 3, "resists"), (2, 4, "capital lost/retaken")]),
    ("Habsburg & szabadságharc", "1699–1867", [
        ("Rákóczi Ferenc II", "person", 1703), ("1848 forradalom", "event", 1848),
        ("Kossuth Lajos", "person", 1848), ("Petőfi Sándor", "person", 1848),
        ("Széchenyi István", "person", 1848), ("Világos 1849", "event", 1849),
    ], [(0, 1, "precedes"), (2, 1, "leads"), (3, 1, "voices"), (4, 1, "reforms"), (1, 5, "ends at")]),
    ("Kiegyezés — Dual Monarchy", "1867–1918", [
        ("Osztrák–Magyar Monarchia", "idea", 1867), ("Ferenc József", "person", 1867),
        ("Millennium 1896", "event", 1896),
    ], [(1, 0, "seals"), (0, 2, "celebrates")]),
    ("Trianon & világháborúk", "1918–1945", [
        ("I. világháború", "event", 1914), ("Trianon 1920", "event", 1920),
        ("Horthy Miklós", "person", 1920), ("II. világháború", "event", 1939),
    ], [(0, 1, "leads to"), (1, 2, "under"), (2, 3, "into")]),
    ("Népköztársaság → 1989", "1945–1989", [
        ("1956 forradalom", "event", 1956), ("Nagy Imre", "person", 1956),
        ("Kádár János", "person", 1956), ("rendszerváltás 1989", "event", 1989),
    ], [(1, 0, "leads"), (2, 0, "crushes"), (0, 3, "echoes in")]),
    ("Modern — Republic", "1990–", [
        ("Harmadik Köztársaság", "idea", 1989), ("NATO 1999", "event", 1999), ("EU 2004", "event", 2004),
    ], [(0, 1, "joins"), (1, 2, "then")]),
]

# thematic cross-era ultra-edges (residual): src era index -> dst era index
RESIDUAL = [(0, 3, "Árpád line → the crown"), (1, 4, "conquest → dynasty"), (6, 7, "Hunyadi → Mohács")]


def build_history() -> UltraGraph:
    """The whole arc as one UltraGraph: a Tree per era, ultra-edges between eras."""
    ug = UltraGraph("hungarian_history")
    trees = []
    for key, span, nodes, edges in ERAS:
        t = Tree(len(nodes), name=key)
        t.adhoc["title"] = key
        t.adhoc["span"] = span
        t.adhoc["labels"] = nodes                       # ad-hoc side store: the extra data
        t.adhoc["rels"] = {}
        for i, (_, _, year) in enumerate(nodes):
            t.nodes[i] = max(-120, min(120, ((year or 800) - 1000) // 10))  # a byte per node
        for a, b, rel in edges:
            t.add_edge(a, b, 1)                         # a micro-edge (one byte)
            t.adhoc["rels"][(a, b)] = rel
        ug.add(t)
        trees.append(t)
    for prev, cur in zip(trees, trees[1:]):
        prev >> cur                                     # chronological ultra-edge (plain)
    for i, j, _ in RESIDUAL:
        ug.wire(trees[i], trees[j], "residual")
    return ug


def _esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def history_svg(ug: UltraGraph) -> str:
    """A themed vertical-timeline view (oldest at top), reading the ad-hoc labels."""
    band, gutter, x0, w = 148, 250, 250, 1180
    h = 70 + len(ug.trees) * band
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" '
         f'font-family="ui-monospace,Menlo,monospace"><rect width="{w}" height="{h}" fill="{BG}"/>']
    s.append(f'<text x="40" y="44" fill="{TEXT}" font-size="26" font-weight="700">'
             f'Magyar történelem — the whole arc as one ultra-graph</text>')
    s.append(f'<line x1="{gutter-40}" y1="70" x2="{gutter-40}" y2="{h-40}" stroke="{RAIL}" stroke-width="2"/>')

    def dot(x, y, col, r=6):
        halo = f'<circle cx="{x}" cy="{y}" r="{r*2:.0f}" fill="{col}" opacity="0.18"/>'
        return halo + f'<circle cx="{x}" cy="{y}" r="{r}" fill="{col}"/>'

    centers = []  # (x,y) of each era's rail node
    for i, t in enumerate(ug.trees):
        cy = 70 + i * band + band // 2
        rail_x = gutter - 40
        centers.append((rail_x, cy))
        s.append(dot(rail_x, cy, "#b48bff", 7))
        s.append(f'<text x="40" y="{cy-6}" fill="{TEXT}" font-size="16" font-weight="700">{_esc(t.adhoc["title"])}</text>')
        s.append(f'<text x="40" y="{cy+16}" fill="{FAINT}" font-size="13">{_esc(t.adhoc["span"])}</text>')
        labels = t.adhoc["labels"]
        n = len(labels)
        step = min(230, (w - x0 - 40) // max(1, n))
        pos = []
        for j, (label, kind, year) in enumerate(labels):
            nx = x0 + j * step + 12
            ny = cy - 10 + (18 if j % 2 else -18)
            pos.append((nx, ny))
        # intra-era micro-edges
        for (a, b) in t.adhoc["rels"]:
            (ax, ay), (bx, by) = pos[a], pos[b]
            s.append(f'<line x1="{ax}" y1="{ay}" x2="{bx}" y2="{by}" stroke="{RAIL}" stroke-width="1.4" opacity="0.9"/>')
        for j, (label, kind, year) in enumerate(labels):
            nx, ny = pos[j]
            col = COL.get(kind, "#00d4ff")
            s.append(dot(nx, ny, col, 5))
            s.append(f'<text x="{nx+14}" y="{ny+4}" fill="{TEXT}" font-size="12.5">{_esc(label)}</text>')
            if year:
                yx = nx + 14 + int(len(label) * 7.35) + 6   # monospace: place year after the label
                s.append(f'<text x="{yx}" y="{ny+4}" fill="{FAINT}" font-size="12.5">{year}</text>')
    # chronological ultra-edges along the rail (green arrows)
    for (x1, y1), (x2, y2) in zip(centers, centers[1:]):
        s.append(f'<line x1="{x1}" y1="{y1+9}" x2="{x2}" y2="{y2-9}" stroke="#00e660" stroke-width="2" opacity="0.7"/>')
    # residual ultra-edges (thematic) — dashed brackets on a secondary rail
    for i, j, _ in RESIDUAL:
        (x1, y1), (x2, y2) = centers[i], centers[j]
        rx = 232
        s.append(f'<path d="M {x1+8} {y1} L {rx} {y1} L {rx} {y2} L {x2+8} {y2}" fill="none" '
                 f'stroke="#b48bff" stroke-width="1.5" opacity="0.55" stroke-dasharray="5 4"/>')
    # legend
    lx = w - 250
    for k, (kind, col) in enumerate([("person", COL["person"]), ("event", COL["event"]),
                                     ("place", COL["place"]), ("idea/law", COL["idea"])]):
        s.append(dot(lx + k * 62, 40, col, 4))
        s.append(f'<text x="{lx+k*62+8}" y="44" fill="{FAINT}" font-size="11">{kind}</text>')
    s.append("</svg>")
    return "".join(s)


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    ug = build_history()
    n_nodes = sum(t.n_nodes for t in ug.trees)
    n_micro = sum(len(t._eval) for t in ug.trees)
    print(f"ultra-graph '{ug.name}': {len(ug.trees)} era-trees, {n_nodes} nodes, "
          f"{n_micro} micro-edges, {len(ug.ultra_edges)} ultra-edges")
    with open(os.path.join(OUT, "hungarian_history.svg"), "w", encoding="utf-8") as f:
        f.write(history_svg(ug))
    with open(os.path.join(OUT, "hungarian_history_macro.svg"), "w", encoding="utf-8") as f:
        f.write(ultragraph_svg(ug))
    print("wrote hungarian_history.svg (timeline) + hungarian_history_macro.svg (library view)")


if __name__ == "__main__":
    main()
