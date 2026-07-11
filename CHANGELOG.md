# Changelog

All notable changes to this project. Format follows
[Keep a Changelog](https://keepachangelog.com/); this project uses
[Semantic Versioning](https://semver.org/).

## [0.6.0] — 2026-07-11

### Added
- **`nn.Sequential`** — chain modules/callables; aggregates parameters, propagates
  `train()`/`eval()` (e.g. to Dropout), and re-quantizes ternary submodules.
- **Callable `Tree`** — `tree(x)` == `tree.forward(x)`, so trees compose in Sequential.
- **`io.save_params` / `io.load_params`** — persist the fp32 parameters of any module
  list (Attention, MoE, RMSNorm/LayerNorm, Embedding, Tree), closing the gap where
  graph save/load did not serialize those modules.
- **Optional vaked lowering** (`ultragraph.vaked`) — `lower_graph(nodes, edges)`
  lowers any labeled property graph into a sparse byte-graph `Tree`; `compile_vaked`
  bridges the vendored vakedc front-end (optional — requires an importable `vakedc`).

## [0.5.0] — 2026-07-11

### Added
- **Activations** — `Tensor.exp` / `tanh` / `sigmoid` autograd ops, plus
  `Tensor.gelu()` and `Tensor.silu()` (composed from primitives).
- **Cosine LR schedule** — `optim.CosineSchedule(opt, total_steps, warmup=…)`:
  linear warmup then cosine decay; call `.step()` once per optimizer step.

## [0.4.0] — 2026-07-11

### Added
- **Top-k routing for `nn.MoE`** — `MoE(..., top_k=k)` keeps only the top-k experts
  per token (gates renormalized); the full soft mixture stays the default.
- **`nn.Dropout`** — inverted dropout (rescales by `1/(1-p)`); passthrough when
  `training=False`.
- **Weight decay** — `weight_decay=` on `SGD` and `Adam` (L2).
- **Autograd** — `Tensor.sum(axis=, keepdims=)`.

## [0.3.0] — 2026-07-11

### Added
- **Mixture of experts** (`nn.MoE`) — a soft mixture of ternary-MLP experts with a
  learned router: `output = Σ_e softmax(router(x))_e · expert_e(x)`. Router and
  experts are dense ternary trees; fully differentiable.
- **Byte-level tokenizer** (`ByteTokenizer`) — lossless UTF-8 byte tokenization,
  vocab 256, no training or OOV. `examples/moe_lm.py` demos a byte-level MoE LM.
- **Autograd slicing** — `Tensor.__getitem__` (basic indexing) with a scatter
  backward, enabling per-expert gating.

## [0.2.1] — 2026-07-11

### Fixed
- `ultragraph.__version__` now derives from installed package metadata
  (`importlib.metadata`) instead of a hardcoded string, so it can no longer drift
  from the released version.

## [0.2.0] — 2026-07-11

### Added
- **Dense ternary bit-packing** (`pack.py`) — `pack_ternary` / `unpack_ternary`
  store 5 ternary values per byte (base-3), the true ~1.58-bit density (1.6
  bits/weight, a 5× shrink over one-byte-per-weight). `io.save(..., packed=True)`
  writes dense weights packed; `load` unpacks transparently (byte-exact).
- **Learned positional embeddings** (`nn.LearnedPositionalEmbedding`) — an fp32
  position table added to a sequence, with gradients flowing only to used
  positions.

## [0.1.0] — 2026-07-11

The byte-graph that is a 1-bit (ternary) LLM.

### Added
- **Core** (`core.py`) — `Node`/`Edge`/`Tree`/`UltraEdge`/`UltraGraph`, the byte
  buffers, the ad-hoc side store, and the dunder API (`node >> node` micro-edge,
  `tree >> tree` ultra-edge, `len`/`iter`/`in`, `_repr_svg_`).
- **Quantization** (`quant.py`) — ternary weights (absmean) + int8 activations
  (absmax), with the small-`eps` / finite guards.
- **Autograd** (`autograd.py`) — n-D reverse-mode tape: `+ - * / @`, `transpose`,
  `swapaxes`, `reshape`, `relu`, `sqrt`, `sum`, `mean(axis)`, `softmax`,
  `cross_entropy`; and `ternary_linear` with a straight-through estimator and
  per-token int8 activation quantization.
- **nn** (`nn.py`) — `linear_tree`, `mlp`, single-head `Attention`, batched
  `MultiHeadAttention` (causal), `RMSNorm`, `LayerNorm`.
- **Optimizers** (`optim.py`) — `SGD` and `Adam`, both with global-norm gradient
  clipping; re-quantize the ternary buffers after each step.
- **Visualization** (`viz.py`) — pure-SVG micro/macro/byte-heatmap views plus an
  optional matplotlib backend (`[viz]` extra).
- **I/O** (`io.py`) — byte-exact save/load.
- **Examples** — `char_lm.py`, `transformer_lm.py`, `mini_gpt.py` (batched
  multi-head transformer that memorizes a toy corpus), `render_viz.py`,
  `make_figures.py`.
- **Tests** — numeric-gradient checks for every op, an attention causality proof,
  save/load byte-exactness, and end-to-end training; a dependency-free runner.
- **Tooling & docs** — `ruff` (config in `pyproject.toml`), GitHub Actions CI
  (`ruff` + `pytest` on Python 3.11–3.13), `CONTRIBUTING.md`, `CHANGELOG.md`, and
  `docs/references.md` (an Erdős graph-theory reading list).

[0.6.0]: https://github.com/peterlodri-sec/ultra-graph/releases/tag/v0.6.0
[0.5.0]: https://github.com/peterlodri-sec/ultra-graph/releases/tag/v0.5.0
[0.4.0]: https://github.com/peterlodri-sec/ultra-graph/releases/tag/v0.4.0
[0.3.0]: https://github.com/peterlodri-sec/ultra-graph/releases/tag/v0.3.0
[0.2.1]: https://github.com/peterlodri-sec/ultra-graph/releases/tag/v0.2.1
[0.2.0]: https://github.com/peterlodri-sec/ultra-graph/releases/tag/v0.2.0
[0.1.0]: https://github.com/peterlodri-sec/ultra-graph/releases/tag/v0.1.0
