# Tree (ultragraph/core.py)

**Path:** `ultragraph/core.py`
**Type:** Core data structure
**Represents:** Meso-level net/module (one Tree = one neural net/module)

## Overview

Tree is the meso-level storage pattern in ultra-graph's three-level byte-graph architecture. It stores weights and performs forward passes, with two storage flavors:

| Flavor | Use Case | Storage Pattern |
|--------|----------|-----------------|
| **dense** | Linear/MLP layers | `wq: int8[out, in]` flat weight matrix |
| **sparse** | General graphs | `_esrc/_edst/_eval: list[int]` parallel arrays |

## Constructor

```python
Tree(n: int)                    # Sparse tree with n nodes
Tree.dense(in_dim, out_dim)     # Dense tree for linear layers
```

## Storage

### Sparse Trees (default)

- **`_esrc`**: list[int] — source node indices
- **`_edst`**: list[int] — destination node indices  
- **`_eval`**: list[int] — ternary edge values `{-1, 0, +1}`

### Dense Trees

- **`wq`**: `int8[out_dim, in_dim]` — flat weight matrix (full connectivity)

### Ad-hoc Store (`.adhoc` dict)

Holds everything > 1 byte:
- **`w_master`**: fp32 master weights (for training)
- **`bias`**: optional bias vector
- **Metadata**: scale factors, quantization parameters

**Deployed mode:** `adhoc["w_master"] = None` triggers inference-only path.

## Integration with UltraGraph

Trees are wired together via **UltraEdges** (`===`):

```python
ug = UltraGraph()
a = Tree.dense(128, 256)
b = Tree.dense(256, 512)

ug += a
ug += b
ug.wire(a, b)  # Plain ultra-edge
ug.wire(a, b, "residual")  # Residual skip
```

## Training vs Inference

### Training Mode

- `adhoc["w_master"]` contains fp32 master weights
- Forward pass uses `ternary_linear()` with STE
- Optimizer updates masters, then calls `requantize()`

### Inference Mode (Deployed)

- `adhoc["w_master"] = None`
- Forward pass uses `ternary_forward()` directly on quantized weights
- Weights can be bit-packed (5 ternary per byte)

## Related Components

- [[ultra_graph]] — Macro-level wiring of trees
- [[tensor]] — Autograd tensor wrapper
- [[three_level_byte_graph_architecture]] — Overall architecture

---

**Source:** `ultragraph/core.py`, `AGENTS.md`
