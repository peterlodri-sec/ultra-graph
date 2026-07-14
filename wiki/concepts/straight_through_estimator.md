# Straight-Through Estimator (STE)

**Category:** Training technique
**Status:** Implemented
**Source:** `ultragraph/quant.py`, `ultragraph/autograd.py`

## Overview

The **Straight-Through Estimator (STE)** enables gradient-based optimization through non-differentiable quantization operations by using a **surrogate gradient** during backpropagation.

In ultra-graph, STE is the core mechanism that allows training of **ternary weight networks** (`{-1, 0, +1}`) using standard gradient descent.

## Mathematical Foundation

### Forward Pass (Quantized)

```python
# Weight quantization (BitNet b1.58)
scale = absmean(w)
q = clip(round(w / scale), -1, 1)  # Ternary: {-1, 0, +1}
y = x @ q.T + b  # Uses quantized weights
```

### Backward Pass (Surrogate)

```python
# Straight-through estimator:
# Pass gradient through as if no quantization happened
dL/dw = dL/dy * x  # Gradient of unquantized linear surrogate
```

## Implementation in ultra-graph

### `ternary_linear()` Function

The core STE implementation:

```python
def ternary_linear(x: Tensor, wq: Tensor, b: Tensor | None = None) -> Tensor:
    """
    Forward: quantize weights -> compute
    Backward: pass gradient through unquantized surrogate
    """
```

**Key property:** Gradients match the **unquantized linear surrogate**, not the quantized forward.

## Why STE Works

### The Problem

Quantization is non-differentiable:
- `round()` has zero gradient almost everywhere
- True derivative: `d/dw round(w) = 0` (except at discontinuities)
- Standard backprop would fail: gradients vanish

### The Solution

STE uses a **biased but useful** gradient estimate:
- **Forward:** Use quantized weights (discrete computation)
- **Backward:** Pretend quantization is identity (continuous surrogate)
- **Result:** Gradients flow, weights update, quantization adapts

## Testing STE

From `test_autograd.py`:

```python
def test_ste_gradient():
    """Verify STE backward matches unquantized surrogate"""
    # Compare analytic gradient to numeric gradient of surrogate
    assert np.allclose(w.grad, numeric_grad, atol=1e-3)
```

## Related Concepts

- [[three_level_byte_graph_architecture]] — Byte-graph architecture
- [[tensor]] — Autograd engine implementing STE
- [[tree]] — Uses STE for quantized forward pass

## References

- **BitNet b1.58:** [BitNet: Scaling 1-bit Transformers for Large Language Models](https://arxiv.org/abs/2310.11453)
- **Original STE:** Bengio et al. (2013) [Estimating or Propagating Gradients Through Stochastic Neurons](https://arxiv.org/abs/1308.3432)

---

**Source:** `ultragraph/quant.py`, `ultragraph/autograd.py`, `AGENTS.md`
