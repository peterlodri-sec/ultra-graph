# ultragraph

[![CI](https://github.com/peterlodri-sec/ultra-graph/actions/workflows/ci.yml/badge.svg)](https://github.com/peterlodri-sec/ultra-graph/actions/workflows/ci.yml)
[![python](https://img.shields.io/badge/python-3.11%2B-00d4ff)](pyproject.toml)
[![license](https://img.shields.io/badge/license-MIT-b48bff)](LICENSE)

A pure-Python (+ numpy) **byte-graph that is a 1-bit (ternary) LLM**.

> **genesis** `251e6ea` · themed after [pocoo.vaked.dev](https://pocoo.vaked.dev)

![ultragraph architecture — micro (node/edge, 1 byte each) → meso (tree) → macro (ultra-graph)](assets/architecture.png)

Three levels:

| level | unit | storage |
|-------|------|---------|
| micro | **node** / **edge** | **1 byte each** — `int8` activation / ternary weight `{-1,0,+1}` |
| meso  | **tree** | a whole graph == one net/module (a Linear/MLP block) |
| macro | **ultra-edge** (`===`) | typed wiring between trees → the **ultra-graph** = the model |

Weights are ternary (BitNet b1.58 style); activations are int8. Full-precision
"master" weights live in an ad-hoc side store during training; the byte buffers are
the deployed state. Training uses a straight-through estimator (STE).

## Illustrations

Real outputs from a trained ternary mini-GPT — regenerate with `uv run python assets/make_figures.py`:

| ultra-graph | causal attention | ternary weight bytes |
|:---:|:---:|:---:|
| ![architecture](assets/fig_architecture.png) | ![attention](assets/fig_attention.png) | ![weights](assets/fig_ternary_weights.png) |

Left: the model as an **ultra-graph** — trees wired by ultra-edges (`===`), with residual skips.
Middle: **real** causal self-attention weights (lower-triangular → no peeking at the future).
Right: a trained query projection's weight bytes, each ∈ {−1, 0, +1}.

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

See `examples/char_lm.py` (MLP LM), `examples/transformer_lm.py` (single-head attention),
and `examples/mini_gpt.py` (batched **multi-head** attention + **RMSNorm** + **Adam**) for
end-to-end char-level ternary language models.

```python
from ultragraph import Embedding, MultiHeadAttention, RMSNorm, linear_tree, Adam
# pre-norm transformer block over a [B, T, d_model] sequence:
#   x = x + mha(norm1(x));  x = x + ff2(ff1(norm2(x)))
```

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
ultragraph/nn.py        linear_tree, mlp, Attention, MultiHeadAttention, RMSNorm, LayerNorm
ultragraph/optim.py     SGD + Adam over fp32 masters (grad clip), re-quantize after step
ultragraph/viz.py       pure-SVG + optional matplotlib (micro / macro / byte-heatmap)
ultragraph/io.py        byte-exact save / load
```

Design spec: `docs/superpowers/specs/2026-07-10-ultragraph-design.md`.
Graph-theory reading list (Erdős classics): [`docs/references.md`](docs/references.md).

## Install from source

```sh
git clone https://github.com/peterlodri-sec/ultra-graph
cd ultra-graph
uv sync
just test
```

## License

MIT — see [LICENSE](LICENSE).
