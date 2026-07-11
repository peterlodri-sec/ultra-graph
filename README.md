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
pip install ultragraph-1bit    # then: import ultragraph
# or from source (Python >=3.11):
uv sync
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
`examples/mini_gpt.py` (batched **multi-head** attention + **RMSNorm** + **Adam**), and
`examples/gpt_lm.py` (the whole stack: **`ByteTokenizer` → `GPT` → train → stream**) for
end-to-end char/byte-level ternary language models.

```python
from ultragraph import Embedding, MultiHeadAttention, RMSNorm, linear_tree, Adam
# pre-norm transformer block over a [B, T, d_model] sequence:
#   x = x + mha(norm1(x));  x = x + ff2(ff1(norm2(x)))
```

## A whole ternary GPT

```python
from ultragraph import GPT

m = GPT(vocab=256, d_model=128, n_layers=4, n_heads=4, max_len=256)  # RoPE + KV-cache
logits = m(ids)                       # ids [B, T] -> logits [B, T, vocab]
out = m.generate([72, 105], n_new=64, temperature=0.8, top_k=40, top_p=0.9,
                 repetition_penalty=1.3, stop=10, seed=0)   # stop on newline byte

for tok in m.generate([72, 105], n_new=64, temperature=0.8, stream=True):
    print(tok, end=" ", flush=True)   # token-by-token
m.save("gpt.npz")                     # fp32 masters; reload onto the same architecture

# a true 1-bit-on-disk checkpoint: bit-packed ternary bytes, no fp32 masters.
m.save_deployed("gpt.q.npz")          # ~10x smaller, inference-only
deployed = GPT.load_deployed("gpt.q.npz")   # byte-exact logits, runs from the trits
```

The deployed checkpoint stores weights at their true **~1.6 bits/weight** density (5
ternary values per byte) plus the tiny fp32 pieces (embedding, norm gains, biases).
On an 858k-param model that's **3.4 MB → 334 KB**, and `deployed(ids)` gives logits
identical to the trained model — `Tree.forward` runs straight from the stored bytes.

`generate` decodes with a per-layer **KV-cache**; since activations are quantized
per token, a cached step is byte-for-byte the full-forward result at that position.
Positions come from **RoPE** (rotary embeddings) — relative, and `offset`-aware so
they line up across cached steps.

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
ultragraph/autograd.py  numpy autograd tape; ternary_linear (STE); exp/tanh/sigmoid/gelu/silu
ultragraph/core.py      Node/Edge/Tree/UltraEdge/UltraGraph + dunder API
ultragraph/nn.py        linear_tree, mlp, Attention, MultiHeadAttention, RoPE, RMSNorm, LayerNorm, LearnedPositionalEmbedding, MoE, Dropout, Sequential
ultragraph/model.py     TransformerBlock + GPT (embedding + RoPE + pre-norm blocks + ternary head) with cached .generate()
ultragraph/optim.py     SGD + Adam (grad clip, weight decay) + CosineSchedule, re-quantize after step
ultragraph/pack.py      dense ternary bit-packing (5 values/byte, ~1.58-bit)
ultragraph/tokenize.py  byte-level tokenizer (ByteTokenizer, vocab 256)
ultragraph/vaked.py      optional vaked lowering (lower_graph, compile_vaked via vendored vakedc)
ultragraph/viz/         svg.py (pure-SVG) + mpl.py (optional matplotlib) — micro / macro / byte-heatmap
ultragraph/io.py        byte-exact save / load (optional packed weights); save_params/load_params
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
