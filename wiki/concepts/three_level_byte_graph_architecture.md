# Three-Level Byte-Graph Architecture

**Category:** Architecture
**Status:** Implemented
**Source:** `ultragraph/core.py`, `AGENTS.md`

## Overview

ultra-graph uses a **three-level byte-graph architecture** where all computation happens with byte-level values:
- **1 byte per node** (`int8` activation)
- **1 byte per edge** (ternary weight `{-1, 0, +1}` stored as `int8`)

This enables extreme memory efficiency (~1.6 bits/weight with packing) while maintaining full training capability through custom autograd.

## Architecture Levels

### Level 1: Micro (Node / Edge)

**Granularity:** Individual neurons and synapses

| Component | Storage | Description |
|-----------|---------|-------------|
| **Node** | `int8` in flat numpy array | Activation value (rounded + clamped) |
| **Edge** | `int8` ternary `{-1, 0, +1}` | Weight in sparse src/dst/val or dense matrix |

**Key properties:**
- No per-node Python objects (memory efficient)
- `tree[i]` returns transient `NodeRef` proxy
- `node >> node` creates micro-edge inside sparse tree

### Level 2: Meso (Tree)

**Granularity:** One neural net / one module

A Tree is a collection of nodes and edges with two storage flavors:

| Flavor | Use Case | Storage |
|--------|----------|---------|
| **dense** | Linear/MLP layers | `wq: int8[out, in]` flat matrix |
| **sparse** | General graphs | `_esrc/_edst/_eval: list[int]` |

**Ad-hoc store (`.adhoc` dict):** Holds fp32 master weights, bias, scale factors.

### Level 3: Macro (UltraEdge / UltraGraph)

**Granularity:** Full model architecture

- **UltraEdge:** Connection between trees (`tree >> tree`)
- **UltraGraph:** Container with topological sort, residual wiring

## Memory Efficiency

| Storage Mode | Bits/Weight | Description |
|--------------|-------------|-------------|
| **Raw ternary** | 8 bits (1 byte) | `int8` storage |
| **Bit-packed** | 1.6 bits | 5 ternary weights per byte (base-3) |
| **fp32 baseline** | 32 bits | Standard float32 |

**Packing ratio:** ~20x smaller than fp32

## Related Concepts

- [[straight_through_estimator]] — STE for training quantized networks
- [[tensor]] — Autograd engine
- [[tree]] — Meso-level storage

---

**Source:** `AGENTS.md` (Architecture section)
