# Compare Checkpoints

Compare two trained model checkpoints — architecture, weight statistics, and inference output.

## What Success Looks Like

A side-by-side comparison report at `skills/reports/compare-{a}-vs-{b}.md` that surfaces differences in architecture, weight distributions, sparsity, and generation quality.

## What to compare

| Dimension | What to look at |
|-----------|-----------------|
| Architecture | Hyperparameters (d_model, n_layers, n_heads, etc.) |
| Parameter count | `model.n_params()` for each |
| Weight statistics | Sparsity %, positive/negative ratio, scale factors per layer |
| Deployed size | File size of `.q.npz` on disk |
| Sample output | Generate from the same prompt at same temperature |
| Loss history | If training logs available, compare loss curves |

## Comparison table format

```markdown
| Metric | Model A | Model B |
|--------|---------|---------|
| d_model | 128 | 64 |
| n_layers | 4 | 2 |
| Params | 858K | 215K |
| Deployed size | 334 KB | 128 KB |
| Sparsity (avg) | 42% | 38% |
| Final loss | 1.85 | 2.10 |
```

## Weight visualization

Use `viz.byte_heatmap_svg()` or `viz.byte_heatmap_png()` to render weight matrices from both models side by side for visual comparison:

```python
from ultragraph import viz

for name, model in [("A", model_a), ("B", model_b)]:
    trees = model._dense_trees()
    for i, t in enumerate(trees):
        svg = viz.byte_heatmap_svg(t.wq)
        # Name each file: compare_A_t0_wq.svg
```
