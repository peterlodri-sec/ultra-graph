# AGENTS.md — ultragraph

A pure-Python (+ numpy) **byte-graph that is a 1-bit (ternary) LLM**. MIT-licensed, package name `ultragraph-1bit` on PyPI.

## Essential commands

| Action | Command |
|--------|---------|
| Install dev deps | `uv sync --extra viz` |
| Full test suite | `uv run pytest` (or `just test`) |
| Fast tests (no pytest) | `just test-fast` (`python tests/run_all.py`) |
| Lint | `uv run ruff check ultragraph tests examples assets` |
| Typecheck | none (no mypy/pyright in project) |
| Syntax check | `just check` (`python -m compileall ultragraph tests examples`) |
| Build | `uv build` |
| CI | `.github/workflows/ci.yml` — `ruff check` + `pytest` on 3.13 & 3.14 (incl. free-threading `3.14t`) |

PyPI publish uses **trusted publishing (OIDC)** — no API token. Triggered by GitHub Release or `workflow_dispatch` (`.github/workflows/publish.yml`).

The `justfile` also has `demo` (runs `examples/char_lm.py`) and `viz` (renders SVG examples).

## Project map

```
ultragraph/           # package (hatchling build, shipped via PyPI)
  core.py             # Tree, UltraGraph, UltraEdge, NodeRef, Embedding
  autograd.py         # Tensor (custom autograd tape over numpy float32)
  nn.py               # layer patterns: Linear/MLP, Attention, MHA, RMSNorm, LayerNorm, RoPE, MoE, Sequential
  model.py            # GPT, TransformerBlock, Mesh (mixture of full models)
  optim.py            # SGD, Adam, CosineSchedule
  quant.py            # BitNet b1.58: quantize_weight_ternary, quantize_act_int8, dequant
  pack.py             # 5-ternary-per-byte base-3 packing (1.6 bits/weight)
  io.py               # save/load (npz + JSON meta), save_params/load_params
  tokenize.py         # ByteTokenizer (byte-level, vocab=256)
  vaked.py            # bridge to vaked language (lower_graph, compile_vaked)
  wiki.py             # MediaWiki client + BFS graph builder (optional wiki extra)
  viz/
    __init__.py       # auto-dispatch to SVG (stdlib) or matplotlib (optional)
    svg.py            # pure-SVG rendering (always available)
    mpl.py             # matplotlib PNG backend (viz extra)
  __init__.py         # public API surface
examples/             # end-to-end demos and training scripts
  char_lm.py          # tiny char-level MLP LM
  transformer_lm.py   # single-head attention LM
  mini_gpt.py         # multi-head GPT with RoPE, RMSNorm, Adam
  gpt_lm.py           # full ByteTokenizer → GPT → train → stream
  mesh_lm.py          # mixture-of-GPT-experts (Mesh) with gradient accumulation
  anonymus_lm.py      # Latin (Gesta Hungarorum) byte-level GPT
  hungarian_lm.py     # Hungarian byte-level GPT (resumable training)
  enrich_corpus.py    # non-LLM Wikipedia fact enrichment for corpus
  fetch_gesta.py      # fetch Gesta Hungarorum text
  fetch_hungarian.py  # fetch Hungarian public-domain text
  hungarian_history.py / hungarian_history_live.py  # knowledge graph demos
  render_viz.py       # render example SVGs to ./out
  data/               # training data, cached Wikipedia pages, checkpoints
tests/
  run_all.py          # dependency-free runner (imports each test module directly)
  test_*.py           # one file per subsystem
vakedc/               # vendored vaked compiler (optional, not imported at module load)
mcp_server/           # MCP/SSE server (optional mcp extra)
assets/               # architecture diagram PNGs + make_figures.py to regenerate them
```

## Architecture — three-level byte-graph

### Micro: node / edge (1 byte each)

- **Nodes** = `int8` activations stored in a flat numpy array per tree.
- **Edges** = `int8` ternary weights `{-1, 0, +1}` stored in parallel src/dst/val arrays (sparse) or as a full `wq` matrix (dense).
- No per-node Python objects. `tree[i]` returns a transient `NodeRef` proxy.
- `node >> node` creates a micro-edge inside a sparse tree.

### Meso: Tree (one net/module)

Two storage flavors:

| Flavor | Use | Storage |
|--------|-----|---------|
| **dense** (chosen via `Tree.dense(in_dim, out_dim)`) | Linear/MLP layers | `wq: int8[out, in]` flat weight matrix; full connectivity |
| **sparse** (default `Tree(n)`) | General graphs, knowledge graphs | `_esrc/_edst/_eval: list[int]` for edges |

A Tree has an `.adhoc` dict for everything > 1 byte: fp32 master weights (`w_master`), `bias`, metadata. The deployed state is just the byte buffers.

`Tree.forward()` is only defined for dense trees in the MVP.

### Macro: UltraEdge (`===`) and UltraGraph

- `tree >> tree` creates a **plain** ultra-edge (output of src → input of dst).
- `ug.wire(a, b, "residual")` adds a residual skip: `out = tree.forward(inp) + src_output`.
- `UltraGraph.forward()` performs topological sort, wires inputs, handles residual adds, and identifies the sink tree for the return value.

## Autograd (custom, no PyTorch dependency)

`ultragraph.autograd.Tensor` is a home-grown reverse-mode tape over numpy float32 arrays. Key design:

- Op granularity is **tensor** not scalar.
- Numpy broadcasting is supported with `_unbroadcast()` helper to reduce grads back to the original shape.
- `Tensor._child()` method creates new tensors and sets up `_backward` closures.
- `backward()` is called on a scalar loss tensor, walks `_prev` DAG in topological order.
- `ternary_linear()` is a special op: forward quantizes weights (STE), backward passes grads through the **unquantized** surrogate.

Most elementwise ops (add, mul, neg, sub, truediv, relu, exp, tanh, sigmoid, sqrt, gelu, silu) follow the same `_child()` + closure pattern. Check `test_autograd.py` for the standard pattern when adding new ops: analytic backward vs central finite differences.

## Quantization (BitNet b1.58)

- **Weight quantization**: absmean scale, `q = clip(round(w/scale), -1, 1)` → `{-1,0,+1}` stored as `int8`.
- **Activation quantization**: absmax scale, `q = clip(round(x/scale), -127, 127)` → `int8`.
- **STE (straight-through estimator)**: forward uses quantized weights; backward passes gradients through the full-precision surrogate as if no quantization happened.
- **5-ternary-per-byte packing**: `pack_ternary()` / `unpack_ternary()` use base-3 (3^5 = 243 < 256) to store 5 ternary weights per byte (1.6 bits/weight).

## Optimizers (`optim.py`)

- `SGD` and `Adam` operate on the **fp32 master weights** in each Tree's ad-hoc store.
- After every `step()`, calls `target.requantize()` to refresh the byte buffers from the updated masters.
- Both support **gradient accumulation**: `accum_steps > 1` means grads accumulate over N `step()` calls, then apply one averaged update and auto-zero gradients. With accumulation, do **not** call `zero_grad()` in the micro-batch loop — the optimizer handles it.
- `CosineSchedule` wraps any optimizer for warmup + cosine decay.

## Conventions & gotchas

- **`ruff` config**: `line-length = 110`, lint rules `F`, `E7`, `E9`, `I` only. That's all. No type checker.
- **Imports**: stdlib + numpy only at module level. Optional deps (`matplotlib`, `mediawiki`, `mcp`) are imported lazily inside functions. The `vakedc` vendored compiler is never imported at wheel load time.
- **`from __future__ import annotations`** in every file (string annotations for forward refs).
- **`np.random.RandomState` with explicit seeds** in tests, never `np.random.seed()` (seed is per-test). Exception: `test_e2e.py` uses `np.random.seed()` for a simple overfitting demo.
- **Byte contract**: one node = one `int8`. `_clip_byte()` handles rounding + clamping. Edge weights are *ternary* (`{-1,0,+1}`). The contract is about semantic values, not storage — adjacency indices (`_esrc`/`_edst`) are separate structural overhead.
- **`Tensor.grad`** is initialized as zeros (not None) — always. Check `_unbroadcast()` behavior when adding ops that involve broadcasting.
- **`Deployed` mode**: `Tree.adhoc["w_master"] = None` triggers inference-only path (`ternary_forward` instead of `ternary_linear`). Saved via `GPT.save_deployed()` → bit-packed, no fp32 masters. `GPT.load_deployed()` rebuilds with `w_master = None`.
- **KV-cache**: per-layer dicts `{"k": Tensor, "v": Tensor}`. Because activations are quantized per-token, a cached step is byte-exact equivalent to full re-forward at that position. RoPE `offset` keeps positions lining up.
- **`.T` property** on `Tensor` calls `.transpose()`, which swaps the last two axes (not full transpose). For n-D arrays, use `.swapaxes(a, b)`.
- **`Mesh`** = mixture of full GPT models (not MoE layers). A small ternary router network produces per-sequence mixing weights. Each expert keeps its own KV-cache during generation.

## Testing patterns

- **`test_autograd.py`**: New ops validated with `_numeric_grad()` (central finite differences, eps=1e-3) against analytic backward.
- **Quantized paths**: STE tests check that grads match the **unquantized linear surrogate** `y = x @ W.T + b`.
- **E2E tests** (`test_e2e.py`): short training runs that verify loss decreases + sampling works. Uses `np.random.seed(0)` and small models.
- **Fast runner** (`tests/run_all.py`): imports test modules directly and discovers `test_*` functions. No pytest dependency. Useful when matplotlib isn't installed (test_viz is skipped if matplotlib import fails).
- **Viz tests**: conditional on matplotlib being importable (`test_viz.py` catches `ImportError` and skips gracefully).

## Save/load persistence

| Method | Storage | Use case |
|--------|---------|----------|
| `GPT.save()` / `.load()` | fp32 masters | Training checkpoints (re-quantizes on load) |
| `GPT.save_deployed()` / `GPT.load_deployed()` | Bit-packed ternary bytes + tiny fp32 pieces (~1.6 bits/weight) | Inference-only, ~5x smaller |
| `io.save(ug, ...)` / `io.load()` | Full UltraGraph (trees + ultra-edges + optional masters) | General graph persistence |
| `save_params` / `load_params` | Flat list of parameter tensors | Arbitrary module collections |

Deployed checkpoints store hyperparameters in a JSON `__meta__` array inside the npz, enabling `GPT.load_deployed()` to reconstruct the architecture.
