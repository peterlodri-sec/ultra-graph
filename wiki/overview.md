# ultra-graph: Project Overview

**Package:** `ultragraph-1bit` (PyPI)
**License:** MIT
**Stack:** Pure Python + numpy (no PyTorch, no JAX)

## What Is ultra-graph?

ultra-graph is a **byte-graph that is a 1-bit (ternary) LLM**. It implements:

- **Custom autograd engine** (reverse-mode tape over numpy float32)
- **BitNet b1.58 quantization** (ternary weights `{-1, 0, +1}`, STE)
- **5-ternary-per-byte packing** (1.6 bits/weight, base-3 encoding)
- **Full GPT architecture** (RoPE, RMSNorm, MHA, KV-cache, MoE)
- **Sparse/dense tree storage** (flexible connectivity patterns)

## Three-Level Architecture

### Micro: Node/Edge (1 byte each)

- **Nodes:** `int8` activations in flat numpy arrays
- **Edges:** `int8` ternary weights `{-1, 0, +1}` in sparse src/dst/val or dense `wq` matrix
- **No per-node Python objects** — `tree[i]` returns transient `NodeRef` proxy
- **Micro-edges:** `node >> node` creates edges inside sparse trees

### Meso: Tree (one net/module)

| Flavor | Use | Storage |
|--------|-----|---------|
| **dense** | Linear/MLP layers | `wq: int8[out, in]` flat matrix |
| **sparse** | General graphs, knowledge graphs | `_esrc/_edst/_eval: list[int]` |

- **`.adhoc` dict** for fp32 master weights, bias, metadata
- **Deployed state:** just byte buffers (no fp32 masters)
- **`Tree.forward()`** defined for dense trees in MVP

### Macro: UltraEdge (`===`) and UltraGraph

- **Plain ultra-edge:** `tree >> tree` (output of src → input of dst)
- **Residual skip:** `out = dst.forward(src_output) + src_output`
- **`UltraGraph.forward()`:** topological sort, wire inputs, handle residuals, identify sink

## Key Features

### Custom Autograd

- Home-grown reverse-mode tape over numpy float32
- Op granularity is **tensor** not scalar
- Numpy broadcasting with `_unbroadcast()` helper
- `Tensor._child()` method creates new tensors + `_backward` closures

### Quantization (BitNet b1.58)

- **Weight quantization:** absmean scale, `q = clip(round(w/scale), -1, 1)`
- **Activation quantization:** absmax scale
- **STE (straight-through estimator):** forward uses quantized, backward passes through full-precision
- **5-ternary-per-byte packing:** base-3 encoding (3^5 = 243 < 256)

### Optimizers

- **SGD** and **Adam** operate on fp32 master weights
- **Gradient accumulation:** accumulate grads over N steps, apply one averaged update
- **`CosineSchedule`:** warmup + cosine decay

### GPT Model

- **TransformerBlock:** MHA + MLP with RoPE, RMSNorm
- **KV-cache:** per-layer dicts for cached activations
- **RoPE `offset`:** keeps positions aligned during generation

## Development Commands

| Action | Command |
|--------|---------|
| Install dev deps | `uv sync --extra viz` |
| Full test suite | `uv run pytest` |
| Fast tests | `python tests/run_all.py` |
| Lint | `uv run ruff check ultragraph tests examples assets` |
| Build | `uv build` |

## Research Foundations

- **BitNet b1.58:** [BitNet: Scaling 1-bit Transformers for Large Language Models](https://arxiv.org/abs/2310.11453)
- **LLM Wiki pattern:** [Andrej Karpathy](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f)
- **Straight-Through Estimator:** [Bengio et al. (2013)](https://arxiv.org/abs/1308.3432)

---

**Wiki status:** [[index]] | **Hot cache:** [[hot]] | **Session log:** [[log]]
