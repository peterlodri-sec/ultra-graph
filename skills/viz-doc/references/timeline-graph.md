# Timeline Graph

Build a curated knowledge-graph timeline from a set of entities, rendered as an ultra-graph.

## What Success Looks Like

An SVG timeline showing entities as nodes connected by edges, rendered as a sparse `Tree` in an `UltraGraph` — either from curated data (`examples/hungarian_history.py` pattern) or live from Wikipedia (`examples/hungarian_history_live.py` pattern).

## Two approaches

### Curated timeline

Define entities (people, events, eras) with explicit relationships:

```python
from ultragraph import Tree, UltraGraph

ug = UltraGraph("timeline")
tree = Tree(n_nodes=59, name="history")
# node bytes encode the era/kind, edges encode relationships
# micro-edges: tree[era_node] >> tree[event_node]
# node byte value = era code (e.g. 1=honfoglalás, 2=királyság, ...)
```

See `examples/hungarian_history.py` for the full pattern: 13 eras, 59 nodes, themed SVG timeline via `viz.ultragraph_svg()`.

### Live from Wikipedia

Use `ultragraph.wiki.build_wiki_graph()` to scrape a Wikipedia, building a sparse Tree dynamically:

```python
from ultragraph.wiki import build_wiki_graph

ug = build_wiki_graph(["Honfoglalás"], lang="hu", max_pages=40)
# Every fetched page = one node
# Micro-edge where one page links to another in the set
# Node byte = out-degree (number of intra-set links)
```

See `examples/hungarian_history_live.py` for the live version. Requires `wiki` extra (`uv sync --extra wiki`).

## Output

Both approaches produce an `UltraGraph` that renders via `viz.ultragraph_svg()` (macro view of eras/entities connected by ultra-edges) or `viz.tree_svg()` (micro view of individual nodes + edges).

```python
svg = viz.ultragraph_svg(ug)
with open("timeline.svg", "w") as f:
    f.write(svg)
```
