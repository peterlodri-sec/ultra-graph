# Tensor (ultragraph/autograd.py)

**Path:** `ultragraph/autograd.py`
**Type:** Core autograd engine
**Represents:** Custom reverse-mode automatic differentiation over numpy float32

## Overview

`Tensor` is ultra-graph's home-grown autograd engine, implementing reverse-mode automatic differentiation on top of numpy. Unlike PyTorch or JAX, it has **no external dependencies** beyond numpy.

Key design decisions:
- **Op granularity is tensor, not scalar** (efficient for large arrays)
- **Numpy broadcasting supported** with `_unbroadcast()` helper
- **`Tensor._child()` method** creates new tensors and sets up `_backward` closures
- **`grad` initialized as zeros** (not None) — always

## Constructor

```python
Tensor(data: ndarray, requires_grad: bool = True, _children: tuple = ())
```

### Attributes

- **`.data`**: underlying numpy array
- **`.grad`**: gradient accumulator (same shape as data, initialized to zeros)
- **`._prev`**: set of parent tensors in computational graph
- **`._backward`**: closure that accumulates gradient into `.grad`

## Core API

### `backward()`

Called on a scalar loss tensor. Walks the computational graph in topological order and accumulates gradients.

```python
loss = ...  # scalar Tensor
loss.backward()  # Populates .grad for all leaf tensors
```

## Supported Operations

### Elementwise Ops

| Op | Method | Backward |
|----|--------|----------|
| **add/sub/mul/div** | `+`, `-`, `*`, `/` | Standard rules |
| **relu** | `x.relu()` | `grad * (x.data > 0)` |
| **exp/tanh/sigmoid** | `x.exp()`, etc. | Analytic derivatives |
| **gelu/silu** | `x.gelu()`, `x.silu()` | Analytic derivatives |

### Matrix Operations

| Op | Method | Description |
|----|--------|-------------|
| **transpose** | `x.transpose()` / `x.T` | Swaps last two axes |
| **matmul** | `x @ y` | Matrix multiplication |
| **ternary_linear** | `ternary_linear(x, w, b)` | Quantized linear with STE |

### Reduction Operations

| Op | Method | Description |
|----|--------|-------------|
| **sum/mean** | `x.sum()`, `x.mean()` | Sum/mean of all elements |

## Special Op: `ternary_linear()`

The quantized linear layer implements the **straight-through estimator (STE)**:

```python
def ternary_linear(x: Tensor, wq: Tensor, b: Tensor | None = None) -> Tensor:
    """
    Forward: quantize weights to {-1, 0, +1}, compute linear
    Backward: pass gradients through unquantized surrogate
    """
```

**Key property:** Gradients match the **unquantized linear surrogate**.

## Example Usage

```python
from ultragraph.autograd import Tensor
import numpy as np

# Create tensors
x = Tensor(np.array([1.0, 2.0, 3.0]))
w = Tensor(np.random.randn(3, 3))

# Forward pass
out = (x @ w).relu()
loss = out.mean()

# Backward pass
loss.backward()

# Access gradients
print(x.grad)  # dL/dx
print(w.grad)  # dL/dw
```

## Testing Pattern

New operations are validated with central finite differences:

```python
def _numeric_grad(f, x, eps=1e-3):
    """Central finite difference gradient approximation"""
    # Compare to analytic backward
```

## Differences from PyTorch

| Feature | Tensor (ultragraph) | PyTorch |
|---------|---------------------|---------|
| **Backend** | numpy float32 | Custom C++/CUDA |
| **Dependencies** | numpy only | None (bundled) |
| **Size** | ~500 lines | Millions of lines |

## Related Components

- [[straight_through_estimator]] — STE theory
- [[tree]] — Uses Tensor for forward/backward
- [[three_level_byte_graph_architecture]] — Overall architecture

---

**Source:** `ultragraph/autograd.py`, `AGENTS.md`
