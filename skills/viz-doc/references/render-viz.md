# Render Visualization

Render SVG or PNG visualizations of ultragraph trees, ultra-graphs, and weight matrices.

## What Success Looks Like

An SVG string (or PNG file) that shows the structure of a model — either as a micro-view of one Tree's weight matrix or nodes, or as a macro-view of the UltraGraph's wiring.

## Available renderers

| Function | Output | Dependency | Shows |
|----------|--------|------------|-------|
| `viz.tree_svg(tree)` | SVG string | stdlib only | Node circles + weight heatmap (dense) or edge lines (sparse) |
| `viz.ultragraph_svg(ug)` | SVG string | stdlib only | Tree boxes with ultra-edge triple-lines by kind |
| `viz.byte_heatmap_svg(arr)` | SVG string | stdlib only | 1D/2D byte array as colored grid |
| `viz.tree_png(tree, path)` | PNG file | matplotlib (viz extra) | Same as tree_svg but raster |
| `viz.ultragraph_png(ug, path)` | PNG file | matplotlib (viz extra) | Same as ultragraph_svg but raster |
| `viz.byte_heatmap_png(arr, path)` | PNG file | matplotlib (viz extra) | Same as byte_heatmap_svg but raster |

## Color conventions

- Negative byte values → blue (`rgb(40,90,220)`)
- Zero → light gray (`rgb(235,235,235)`)
- Positive byte values → red (`rgb(220,55,45)`)
- Ultra-edge kinds: plain → `#4a6fa5`, residual → `#d08a2e`

## Usage patterns

```python
from ultragraph import GPT, viz

model = GPT(vocab=256, d_model=128, n_layers=2, n_heads=4)
# After forward pass (populates node bytes):
# svg = viz.tree_svg(model.blocks[0].attn.wq)  # weight matrix
# svg = viz.ultragraph_svg(ug)                   # full model wiring

# Save to file
with open("out/model.svg", "w") as f:
    f.write(svg)

# PNG (requires matplotlib)
# viz.tree_png(tree, "out/tree.png")
```

## Auto-display in notebooks

`Tree._repr_svg_()` and `UltraGraph._repr_svg_()` return SVG for automatic rendering in Jupyter/IPython.

## Performance notes

- Dense weight matrices larger than 20×28 cells are subsampled in SVG output
- Sparse trees with >512 edges are truncated (the first 512 are shown)
- Trees with >256 nodes get a "(+N more nodes)" note
