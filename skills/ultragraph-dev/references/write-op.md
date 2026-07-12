# Write Op

Write a new autograd operation for `ultragraph.autograd.Tensor` with analytic backward and a numeric-gradient test.

## What Success Looks Like

A new method on `Tensor` (or standalone function in `autograd.py`) that:
- Composes from existing primitives OR implements a raw op with correct `_backward`
- Has a `test_<op>_grad` in `tests/test_autograd.py` that passes `_numeric_grad()` (central finite differences, eps=1e-3) against the analytic backward at atol=1e-2
- For ternary-specific ops: STE test checks grads match the unquantized surrogate `y = x @ W.T + b`

## Pattern for a new elementwise op

```python
def my_op(self):
    d = _compute(self.data)
    out = self._child(d, (self,), None)
    def _backward():
        self.grad += _grad_formula(self.data, out.grad)
    out._backward = _backward
    return out
```

## Pattern for a new op with multiple parents

```python
def my_binary_op(self, other):
    other = other if isinstance(other, Tensor) else Tensor(other)
    out = self._child(self.data OP other.data, (self, other), None)
    def _backward():
        self.grad += _unbroadcast(SOME_FN, self.data.shape)
        other.grad += _unbroadcast(OTHER_FN, other.data.shape)
    out._backward = _backward
    return out
```

## Key details

- `_unbroadcast(grad, shape)` reduces gradient back to the original shape after numpy broadcasting — essential for any op whose inputs can broadcast.
- `Tensor.grad` is always initialized as zeros, not None.
- `Tensor._child(data, parents, backward)` sets `requires_grad` from any parent, creates the output tensor, and wires `_backward` only when grad is needed.
- `.T` on Tensor swaps the last two axes only (not full transpose). For n-D axis swaps use `.swapaxes(a, b)`.
- Export the new op in `ultragraph/__init__.py` `__all__` if it should be public.
