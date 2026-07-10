"""Minimal reverse-mode autograd over numpy arrays (float32).

Tape granularity is the tensor op, not the scalar. The special op
``ternary_linear`` folds BitNet-style quantization into the forward pass while
using a straight-through estimator (STE) on the backward pass, so gradients flow
to the full-precision master weights.
"""
from __future__ import annotations

from typing import Callable, Iterable

import numpy as np

from .quant import quantize_act_int8, quantize_weight_ternary


def _unbroadcast(grad: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """Reduce ``grad`` back to ``shape`` after numpy broadcasting."""
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for i, dim in enumerate(shape):
        if dim == 1 and grad.shape[i] != 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad.reshape(shape)


class Tensor:
    """A float32 array node in the autograd tape."""

    def __init__(self, data, requires_grad: bool = False, _prev: Iterable["Tensor"] = ()):  # noqa: D401
        self.data = np.asarray(data, dtype=np.float32)
        self.requires_grad = requires_grad
        self.grad = np.zeros_like(self.data)
        self._backward: Callable[[], None] = lambda: None
        self._prev = set(_prev)

    # -- construction helpers -------------------------------------------------
    @property
    def shape(self):
        return self.data.shape

    def __repr__(self):
        return f"Tensor(shape={self.data.shape}, requires_grad={self.requires_grad})"

    def _child(self, data, parents, backward) -> "Tensor":
        req = any(p.requires_grad for p in parents)
        out = Tensor(data, requires_grad=req, _prev=parents)
        if req:
            out._backward = backward
        return out

    # -- elementwise ops ------------------------------------------------------
    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = self._child(self.data + other.data, (self, other), None)

        def _backward():
            self.grad += _unbroadcast(out.grad, self.data.shape)
            other.grad += _unbroadcast(out.grad, other.data.shape)

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = self._child(self.data * other.data, (self, other), None)

        def _backward():
            self.grad += _unbroadcast(out.grad * other.data, self.data.shape)
            other.grad += _unbroadcast(out.grad * self.data, other.data.shape)

        out._backward = _backward
        return out

    def __neg__(self):
        out = self._child(-self.data, (self,), None)

        def _backward():
            self.grad += -out.grad

        out._backward = _backward
        return out

    def __sub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return self + (-other)

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = self._child(self.data @ other.data, (self, other), None)

        def _backward():
            self.grad += out.grad @ other.data.T
            other.grad += self.data.T @ out.grad

        out._backward = _backward
        return out

    def relu(self):
        out = self._child(np.maximum(self.data, 0.0), (self,), None)

        def _backward():
            self.grad += (self.data > 0).astype(np.float32) * out.grad

        out._backward = _backward
        return out

    def sum(self):
        out = self._child(np.array(self.data.sum(), dtype=np.float32), (self,), None)

        def _backward():
            self.grad += np.ones_like(self.data) * out.grad

        out._backward = _backward
        return out

    def mean(self):
        out = self._child(np.array(self.data.mean(), dtype=np.float32), (self,), None)
        n = self.data.size

        def _backward():
            self.grad += (np.ones_like(self.data) / n) * out.grad

        out._backward = _backward
        return out

    def softmax(self, axis: int = -1):
        z = self.data - self.data.max(axis=axis, keepdims=True)
        e = np.exp(z)
        p = e / e.sum(axis=axis, keepdims=True)
        out = self._child(p, (self,), None)

        def _backward():
            # dz = p * (g - sum(g * p))
            g = out.grad
            s = (g * p).sum(axis=axis, keepdims=True)
            self.grad += p * (g - s)

        out._backward = _backward
        return out

    def cross_entropy(self, targets: np.ndarray):
        """Softmax cross-entropy. ``self`` is logits [batch, C]; targets int [batch]."""
        targets = np.asarray(targets, dtype=np.int64)
        z = self.data - self.data.max(axis=-1, keepdims=True)
        e = np.exp(z)
        p = e / e.sum(axis=-1, keepdims=True)
        n = self.data.shape[0]
        logp = np.log(p[np.arange(n), targets] + 1e-12)
        loss = -float(logp.mean())
        out = self._child(np.array(loss, dtype=np.float32), (self,), None)

        def _backward():
            d = p.copy()
            d[np.arange(n), targets] -= 1.0
            d /= n
            self.grad += d * out.grad

        out._backward = _backward
        return out

    # -- backward -------------------------------------------------------------
    def backward(self):
        topo: list[Tensor] = []
        seen = set()

        def build(t: "Tensor"):
            if t not in seen:
                seen.add(t)
                for parent in t._prev:
                    build(parent)
                topo.append(t)

        build(self)
        self.grad = np.ones_like(self.data)
        for t in reversed(topo):
            t._backward()


def ternary_linear(x: Tensor, master_w: Tensor, bias: Tensor | None = None) -> Tensor:
    """Quantized linear layer: y = quant(x) @ quant(W).T + b.

    Forward uses ternary weights and int8 activations (the deployed byte state).
    Backward is a straight-through estimator: gradients treat the quantization as
    identity and flow to the full-precision ``master_w`` and ``bias``.

    Shapes: x [batch, in], master_w [out, in], bias [out] -> y [batch, out].
    """
    wq, sw = quantize_weight_ternary(master_w.data)
    xq, sx = quantize_act_int8(x.data)
    acc = xq.astype(np.int32) @ wq.T.astype(np.int32)
    y = acc.astype(np.float32) * np.float32(sx * sw)
    if bias is not None:
        y = y + bias.data

    parents = [x, master_w] + ([bias] if bias is not None else [])
    req = any(p.requires_grad for p in parents)
    out = Tensor(y, requires_grad=req, _prev=parents)

    if req:
        def _backward():
            dy = out.grad  # [batch, out]
            x.grad += dy @ master_w.data          # STE: identity through quant
            master_w.grad += dy.T @ x.data
            if bias is not None:
                bias.grad += dy.sum(axis=0)

        out._backward = _backward
    return out
