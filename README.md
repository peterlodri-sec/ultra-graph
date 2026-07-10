# ultragraph

A pure-Python (+ numpy) **byte-graph that is a 1-bit (ternary) LLM**.

Three levels:

| level | unit | storage |
|-------|------|---------|
| micro | **node** / **edge** | **1 byte each** — `int8` activation / ternary weight `{-1,0,+1}` |
| meso  | **tree** | a whole graph == one net/module (a Linear/MLP block) |
| macro | **ultra-edge** (`===`) | typed wiring between trees → the **ultra-graph** = the model |

Weights are ternary (BitNet b1.58 style); activations are int8. Full-precision
"master" weights live in an ad-hoc side store during training; the byte buffers are
the deployed state. Training uses a straight-through estimator (STE).

## Install

```sh
uv sync          # Python >=3.14, numpy, pytest
```

## Dunder API

`>>` is overloaded by operand type:

```python
import numpy as np
from ultragraph import Tree, UltraGraph, Tensor, mlp, SGD

# micro-edges inside a sparse tree
g = Tree(4, "g")
g[0] >> g[1]        # node >> node  -> micro-edge
g[2] = 7            # set a node byte
print(len(g), 2 in g, list(g))

# ultra-edges between trees
ug = UltraGraph()
a = ug.add(Tree.dense(8, 16, "a"))
b = ug.add(Tree.dense(16, 4, "b", act="none"))
a >> b              # tree >> tree -> ultra-edge (plain)
a.wire(b, "residual")
```

## Train a tiny ternary net

```python
ug = mlp([4, 16, 2])                 # dense ternary linear trees wired plain
opt = SGD(ug, lr=0.3, momentum=0.9)
x = Tensor(np.random.randn(32, 4).astype("float32"))
for _ in range(300):
    loss = ug.forward(x).cross_entropy(y)
    opt.zero_grad(); loss.backward(); opt.step()   # step() re-quantizes weights
```

See `examples/char_lm.py` for an end-to-end char-level ternary language model.

## Tasks

```sh
just test        # pytest
just test-fast   # dependency-free runner (stdlib + numpy)
just demo        # char-LM end-to-end
just viz         # render example SVGs
```

## Layout

```
ultragraph/quant.py     ternary + int8 quantization, STE
ultragraph/autograd.py  numpy autograd tape; ternary_linear (STE)
ultragraph/core.py      Node/Edge/Tree/UltraEdge/UltraGraph + dunder API
ultragraph/nn.py        layer patterns as trees (linear_tree, mlp)
ultragraph/optim.py     SGD over fp32 masters, re-quantizes after step
ultragraph/viz.py       pure-SVG micro / macro / byte-heatmap views
ultragraph/io.py        byte-exact save / load
```

Design spec: `docs/superpowers/specs/2026-07-10-ultragraph-design.md`.
