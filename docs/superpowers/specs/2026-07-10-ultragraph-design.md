# ultragraph — design spec (2026-07-10)

A pure-Python (+ numpy) graph library whose byte-graph **is** a 1-bit (ternary) LLM
runtime. Dunder-driven API, a visualization module, and a three-level hierarchy:
**node/edge (micro, 1 byte each) → tree (a whole net/module) → ultra-edge wiring
between trees (macro) = the ultra-graph.**

## 1. Goals & non-goals

Goals (MVP):
- Ultra-compact storage: **1 byte per node** (int8 activation), **1 byte per edge**
  (ternary weight {−1,0,+1} stored value-constrained in a byte).
- Autograd over the byte-graph, tensor/op granularity (numpy), BitNet-b1.58-style
  quantization with straight-through estimator (STE).
- Dunder API: `tree[i]`, `node >> node` (micro-edge), `tree >> tree` (ultra-edge),
  `len`, `iter`, `in`, `_repr_svg_`.
- `nn` layer patterns built as trees (Linear/MLP), wired into an ultra-graph.
- SGD optimizer over full-precision "master" weights held in an ad-hoc side store.
- Pure-SVG visualization (micro/macro/byte-heatmap).
- Save/load byte-exact.
- End-to-end demo: tiny char-level ternary LM that trains and samples.

Non-goals (deferred): attention tree, Adam, matplotlib/graphviz backends, dense
bit-packing (5-ternary/byte), MoE routing ultra-edges.

## 2. Data model & the byte contract

- **Node** — one unit. Payload = 1 byte `int8` (quantized activation). Addressed by
  index within its tree. No per-node Python object stored; `tree[i]` returns a
  lightweight `NodeRef(tree, i)` proxy created on demand.
- **Edge** — directed connection between two nodes in the same tree. Payload = 1 byte.
  Ternary mode: value ∈ {−1,0,+1} stored in an int8.
- **"1 byte" = the semantic value/weight.** Adjacency bookkeeping (src/dst indices) is
  separate structural overhead, not counted in the contract.
- **Tree** — a whole graph = one net/module. Two storage flavors:
  - **dense** (Linear): weights as ternary matrix `wq: int8[out, in]` (E = out·in edges,
    1 byte each), full connectivity, no explicit adjacency arrays. `nodes: int8[out]`
    = last output activation.
  - **sparse** (general graph): `edge_src: int32[E]`, `edge_dst: int32[E]`,
    `edge_val: int8[E]`, `nodes: int8[N]`.
- **UltraEdge** — typed connection between two trees. `kind ∈ {plain, residual}` (MVP).
- **UltraGraph** — top-level model: `trees: list[Tree]`, `ultra_edges: list[UltraEdge]`.
- **Ad-hoc store** — per-tree `dict` holding anything > 1 byte: fp32 master weights,
  grads, per-tree scales, biases, names/tags. Byte buffers = hot/deployable state;
  ad-hoc = rich/training state.

## 3. Quantization (`ultragraph/quant.py`)

BitNet b1.58 style.
- `quantize_weight_ternary(w: np.float32[...]) -> (q: int8 in {-1,0,1}, scale: float)`
  absmean scale: `scale = mean(|w|) + eps`; `q = clip(round(w/scale), -1, 1)`.
- `quantize_act_int8(x: np.float32[...]) -> (q: int8, scale: float)`
  absmax: `scale = max(|x|)/127 + eps`; `q = clip(round(x/scale), -127, 127)`.
- `dequant(q, scale) -> float32` = `q.astype(float32) * scale`.

## 4. Autograd (`ultragraph/autograd.py`)

Reverse-mode tape at tensor/op granularity.
- `class Tensor`: `data: np.float32 ndarray`, `grad: ndarray`, `_backward: callable`,
  `_prev: set[Tensor]`, `requires_grad: bool`.
- Ops (each records a backward closure): `__add__`, `__sub__`, `__neg__`, `__mul__`,
  `__matmul__`, `relu()`, `sum()`, `mean()`, `softmax(axis=-1)`,
  `cross_entropy(targets)`.
- `backward()`: build topo order over `_prev`, seed grad=1 (scalar), propagate.
- **`ternary_linear(x: Tensor, master_w: Tensor, bias: Tensor|None) -> Tensor`** — the
  crux op tying quantization into autograd:
  - Forward: quantize `master_w` → ternary (scale `sw`), quantize `x.data` → int8
    (scale `sx`), integer matmul `acc = xq @ wq.T`, dequant `y = acc*sx*sw (+bias)`.
    Also returns the int8 activations so the caller (Tree) can write node bytes.
  - Backward (STE, quantization treated as identity): `dW_master += dY.T @ x`,
    `dx += dY @ master_w`, `dbias += sum(dY, axis=0)`.

## 5. Core (`ultragraph/core.py`)

- `class NodeRef`: `.tree`, `.index`; `.value` get/set node byte;
  `__rshift__(other: NodeRef) -> Edge` adds a sparse micro-edge; `__repr__`.
- `class Edge`: `src:int`, `dst:int`, `value:int` (ternary).
- `class Tree`:
  - construct dense: `Tree.dense(in_dim, out_dim, name)` — allocates fp32 master weight
    in ad-hoc, initializes byte buffers via requantize.
  - construct sparse: `Tree(n_nodes, name)`.
  - buffers: `nodes: int8[N]`; dense: `wq: int8[out,in]`, `w_scale: float`;
    sparse: `edge_src/edge_dst/edge_val`.
  - ad-hoc: `self.adhoc: dict` (keys: `"w_master"`, `"w_grad"`, `"bias"`, `"bias_grad"`,
    `"name"`, ...).
  - dunders: `__len__`→N, `__getitem__(i)`→NodeRef, `__setitem__(i, v)`→set byte,
    `__iter__`→NodeRefs, `__contains__(i)`, `__rshift__(other: Tree)`→UltraEdge(plain),
    `_repr_svg_`.
  - methods: `requantize()` (master→wq,scale), `params()`→list of (master, grad name)
    for optimizer, `forward(x: Tensor)->Tensor` (dense: ternary_linear then relu unless
    output layer; writes node bytes from returned int8 acts).
  - `.wire(other, kind="plain"|"residual")` → UltraEdge (registers with owner ug).
- `class UltraEdge`: `src: Tree`, `dst: Tree`, `kind: str`.
- `class UltraGraph`: `add(tree)`, `wire(a, b, kind)`, `trees`, `ultra_edges`,
  `forward(x)` (topo over trees via ultra-edges; residual adds skip; plain feeds
  output→next input), `params()` (aggregate), `_repr_svg_`.

## 6. nn (`ultragraph/nn.py`)

- `linear_tree(in_dim, out_dim, name, act="relu"|"none") -> Tree`.
- `mlp(dims: list[int], ug=None) -> UltraGraph` — stacks dense linear trees wired plain,
  relu between, last layer act="none".
- Embedding for char-LM: `Embedding(vocab, dim)` (fp32 lookup table in ad-hoc; not
  ternary — small) producing a Tensor; `unembed` = a linear_tree to vocab logits.

## 7. optim (`ultragraph/optim.py`)

- `class SGD(params, lr, momentum=0.0)`: `params` = list of Tree (or objects exposing
  master/grad). `step()` updates fp32 masters from grads, then calls `requantize()` on
  each dense tree. `zero_grad()` clears grads.

## 8. viz (`ultragraph/viz.py`)

Pure-string SVG (stdlib only).
- `tree_svg(tree) -> str` (micro): nodes as circles colored by byte value; dense edges
  sampled/summarized; sparse edges drawn with sign color.
- `ultragraph_svg(ug) -> str` (macro): trees as boxes, ultra-edges as `===` connectors
  colored by kind.
- `byte_heatmap_svg(arr: np.ndarray) -> str`: grid of cells colored by byte value.
- `Tree._repr_svg_` / `UltraGraph._repr_svg_` delegate here.
- Contract viz relies on: `tree.n_nodes`, `tree.nodes` (int8 ndarray), `tree.name`,
  `tree.kind` ("dense"/"sparse"), dense: `tree.wq`; sparse: `tree.edge_src/dst/val`;
  `ug.trees`, `ug.ultra_edges` with `.src/.dst/.kind` (src/dst are Tree, compared by
  identity → index via `ug.trees.index`).

## 9. io (`ultragraph/io.py`)

- `save(ug, path)`: write a single binary + json header capturing per-tree kind, name,
  shapes, `nodes` bytes, dense `wq`+`w_scale` (+ bias), sparse edge arrays, and
  ultra-edge list. Byte-exact for the compact state (fp32 masters optionally included).
- `load(path) -> UltraGraph`.

## 10. Testing (`tests/`, dependency-free + pytest-compatible)

- `test_quant.py`: ternary/int8 round-trip bounds; dequant.
- `test_autograd.py`: numeric-gradient check for add/mul/matmul/relu/softmax/CE;
  STE gradient of `ternary_linear` vs numeric grad on master weights.
- `test_core.py`: dunder edge-building (`node>>node`, `tree>>tree`), NodeRef byte
  get/set, `len/iter/in`.
- `test_io.py`: save/load byte-exactness.
- `test_e2e.py`: MLP tree overfits a toy dataset (loss decreases); 2 trees + residual
  ultra-edge forward runs; tiny char-LM trains a few steps and loss drops.
- `tests/run_all.py`: dependency-free runner (discovers `test_*` functions).

## 11. Module layout (MVP = flat files, not subpackages)

```
ultragraph/__init__.py   public exports
ultragraph/quant.py
ultragraph/autograd.py
ultragraph/core.py
ultragraph/nn.py
ultragraph/optim.py
ultragraph/viz.py
ultragraph/io.py
tests/*.py
examples/char_lm.py
pyproject.toml
README.md
```

## 12. Decisions ratified by user

1. Byte-graph **is** the ternary LLM (Approach: tensor-tree, byte layout underneath).
2. Generic ternary autograd engine; layers = graph patterns/configs.
3. numpy allowed for math core.
4. Ternary stored 1-per-byte (literal contract), bit-packing deferred.
5. `>>` overloaded: `node>>node` micro-edge, `tree>>tree` ultra-edge.
6. Attention deferred; MLP char-LM is the MVP end-to-end.
