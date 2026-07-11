# Changelog

All notable changes to this project. Format follows
[Keep a Changelog](https://keepachangelog.com/); this project uses
[Semantic Versioning](https://semver.org/).

## [0.11.0] — 2026-07-11

### Added
- **`model.Mesh`** — a graph of minds: a learned soft (or top-k) mixture of *full*
  models. Each expert maps token ids to logits; a small ternary router mixes them
  per sequence (`logits = Σ_e gate(ids)_e · expert_e(ids)`). `nn.MoE`'s routing
  lifted to whole networks — fully differentiable, router + experts train together.

### Fixed
- `Mesh` and `nn.MoE` now reject `top_k` outside `[1, n_experts]` (a non-positive
  `top_k` previously zeroed the whole mixture silently).

## [0.10.0] — 2026-07-11

### Added
- **Deployed inference from ternary bytes** — a dense `Tree` with no fp32 master now
  runs `forward` straight from its stored ternary weight bytes (`autograd.ternary_forward`,
  shared numerics with `ternary_linear` → byte-exact). `Tree.deployed` reports the mode.
- **`GPT.save_deployed` / `GPT.load_deployed`** — an inference-only checkpoint holding
  the bit-packed ternary weights (~1.6 bits/weight) plus the small fp32 pieces
  (embedding, norm gains, biases). **~10× smaller** than the fp32-master `save`
  (e.g. 3.4 MB → 334 KB on an 858k-param model) and byte-exact at inference.
- **`GPT.n_params()`** — trainable scalar count.

### Changed
- `Tree.parameters()` / `requantize()` no-op cleanly for a master-less (deployed) tree.

## [0.9.0] — 2026-07-11

### Added
- **`stop` tokens** — `GPT.generate(..., stop=id_or_ids)` halts as soon as a stop
  token is produced (emitted, then generation ends).
- **Repetition penalty** — `GPT.generate(..., repetition_penalty=…)`: the CTRL rule
  over the running history (`> 1` discourages already-seen tokens, `< 1` encourages).
- **`examples/gpt_lm.py`** — the whole stack end to end: `ByteTokenizer` → `GPT`
  (RoPE + KV-cache) → Adam → re-quantized ternary → streaming generation and a
  save/load round-trip, on a byte-level toy corpus.

## [0.8.0] — 2026-07-11

### Added
- **`GPT.save` / `GPT.load`** — persist and restore a model's fp32 parameters
  (byte-exact inference after re-quantize). Rebuild a `GPT` with the same
  hyper-parameters, then `.load(path)`.
- **Streaming generation** — `GPT.generate(..., stream=True)` returns a generator
  that yields each new token id as it is produced (drives token-by-token demos).
- **Nucleus sampling** — `GPT.generate(..., top_p=…)`: keep the smallest set of
  tokens whose cumulative probability reaches `top_p`, then renormalize. Composes
  with `top_k` and `temperature`.

## [0.7.0] — 2026-07-11

### Added
- **`nn.RoPE`** — rotary positional embeddings (no learned params). Encodes
  *relative* position in the attention dot-product and takes an `offset=` so it
  composes with a KV-cache; drop it into `MultiHeadAttention(..., rope=RoPE(...))`.
- **KV-cache** — `MultiHeadAttention(x, cache={'k':…,'v':…})` returns
  `(out, new_cache)` for incremental decoding. Because activation quant is
  per-token, a cached step is byte-for-byte the full-forward result at that
  position (see `tests/test_gpt.py`).
- **`autograd.cat`** — concatenate `Tensor`s along an axis; grad splits back to
  each input. The primitive behind RoPE's `rotate_half` and the KV-cache.
- **`model.GPT` / `model.TransformerBlock`** — a ternary GPT (token embedding +
  shared RoPE + N pre-norm blocks + ternary head) with `.generate(prompt, n_new,
  temperature=, top_k=, seed=)` autoregressive sampling over the KV-cache.

## [0.6.1] — 2026-07-11

### Changed
- Split `viz.py` (491 loc) into a `viz/` package — `viz/svg.py` (pure-SVG) +
  `viz/mpl.py` (optional matplotlib). Public API (`ultragraph.viz.*`) unchanged.
  Every module is now ≤ ~340 loc (readability/maintainability).

### Fixed
- Re-indented the vendored `vakedc` `parser.py`/`graph.py`/`emit.py` so they compile.
  Documented that upstream vakedc is de-indented **and lossy** (missing content), so
  `compile_vaked` still needs the original vakedc; `lower_graph` works standalone.

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

[0.11.0]: https://github.com/peterlodri-sec/ultra-graph/releases/tag/v0.11.0
[0.10.0]: https://github.com/peterlodri-sec/ultra-graph/releases/tag/v0.10.0
[0.9.0]: https://github.com/peterlodri-sec/ultra-graph/releases/tag/v0.9.0
[0.8.0]: https://github.com/peterlodri-sec/ultra-graph/releases/tag/v0.8.0
[0.7.0]: https://github.com/peterlodri-sec/ultra-graph/releases/tag/v0.7.0
[0.6.1]: https://github.com/peterlodri-sec/ultra-graph/releases/tag/v0.6.1
[0.6.0]: https://github.com/peterlodri-sec/ultra-graph/releases/tag/v0.6.0
[0.5.0]: https://github.com/peterlodri-sec/ultra-graph/releases/tag/v0.5.0
[0.4.0]: https://github.com/peterlodri-sec/ultra-graph/releases/tag/v0.4.0
[0.3.0]: https://github.com/peterlodri-sec/ultra-graph/releases/tag/v0.3.0
[0.2.1]: https://github.com/peterlodri-sec/ultra-graph/releases/tag/v0.2.1
[0.2.0]: https://github.com/peterlodri-sec/ultra-graph/releases/tag/v0.2.0
[0.1.0]: https://github.com/peterlodri-sec/ultra-graph/releases/tag/v0.1.0
