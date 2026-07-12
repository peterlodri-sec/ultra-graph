# Wire Tree

Design and wire Tree and UltraGraph networks — the macro level of the byte-graph.

## What Success Looks Like

A working `UltraGraph` or standalone set of `Tree`s wired by ultra-edges, whose `forward()` produces correct-shaped outputs and whose `parameters()` and `requantize()` are plumbed through all modules.

## The three levels

| Level | Primitive | How to create |
|-------|-----------|---------------|
| Micro | `Tree(n_nodes)` / `Tree.dense(in, out)` | `tree[0] >> tree[1]` (micro-edge inside sparse tree) |
| Meso | Tree as a module | `Tree.dense(in_dim, out_dim, name, act)` |
| Macro | UltraGraph | `tree >> tree` (plain), `ug.wire(a, b, "residual")` |

## Wiring patterns

```python
# Plain: output of src → input of dst
a >> b  # or: ug.wire(a, b, "plain")

# Residual: out = tree.forward(inp) + src_output
ug.wire(a, b, "residual")

# Multiple inputs sum at the dst
a_in >> c  # c receives a_in + b_in
b_in >> c
```

## Module protocol

Every module should expose:
- `parameters() -> list[Tensor]` — fp32 master weights and biases
- `requantize()` — refresh ternary byte buffers from fp32 masters
- Forward via `__call__` or explicit `forward()`

A Tree with `adhoc["w_master"] = None` triggers deployed/inference-only path — `Tree.forward` runs from the stored ternary bytes directly.

## Topological order

`UltraGraph.forward()` sorts trees topologically, wires inputs from plain ultra-edges, applies residual adds, and returns the sink tree's output (last tree with no outgoing plain edge). Cycles raise `ValueError`.
