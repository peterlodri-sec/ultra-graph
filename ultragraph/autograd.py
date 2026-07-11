"""Minimal reverse-mode autograd over numpy arrays (float32).

Tape granularity is the tensor op, not the scalar. Ops support n-D arrays with
numpy broadcasting so batched / multi-head layers work; 2-D usage is a special
case with identical behavior. The special op ``ternary_linear`` folds
BitNet-style quantization into the forward pass while using a straight-through
estimator (STE) on the backward pass.
"""
from __future__ import annotations

from typing import Callable, Iterable

import numpy as np

from .quant import EPS, quantize_weight_ternary


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

    def __init__(self, data, requires_grad: bool = False, _prev: Iterable["Tensor"] = ()):
        self.data = np.asarray(data, dtype=np.float32)
        self.requires_grad = requires_grad
        self.grad = np.zeros_like(self.data)
        self._backward: Callable[[], None] = lambda: None
        self._prev = set(_prev)

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

    def __truediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = self._child(self.data / other.data, (self, other), None)

        def _backward():
            self.grad += _unbroadcast(out.grad / other.data, self.data.shape)
            other.grad += _unbroadcast(-out.grad * self.data / (other.data ** 2), other.data.shape)

        out._backward = _backward
        return out

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = self._child(self.data @ other.data, (self, other), None)

        def _backward():
            ga = out.grad @ np.swapaxes(other.data, -1, -2)
            gb = np.swapaxes(self.data, -1, -2) @ out.grad
            self.grad += _unbroadcast(ga, self.data.shape)
            other.grad += _unbroadcast(gb, other.data.shape)

        out._backward = _backward
        return out

    def relu(self):
        out = self._child(np.maximum(self.data, 0.0), (self,), None)

        def _backward():
            self.grad += (self.data > 0).astype(np.float32) * out.grad

        out._backward = _backward
        return out

    def sqrt(self):
        d = np.sqrt(self.data)
        out = self._child(d, (self,), None)

        def _backward():
            self.grad += (0.5 / (d + 1e-12)) * out.grad

        out._backward = _backward
        return out

    def transpose(self):
        out = self._child(self.data.T, (self,), None)

        def _backward():
            self.grad += out.grad.T

        out._backward = _backward
        return out

    @property
    def T(self):
        return self.transpose()

    def swapaxes(self, a: int, b: int):
        out = self._child(np.swapaxes(self.data, a, b), (self,), None)

        def _backward():
            self.grad += np.swapaxes(out.grad, a, b)

        out._backward = _backward
        return out

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        out = self._child(self.data.reshape(shape), (self,), None)

        def _backward():
            self.grad += out.grad.reshape(self.data.shape)

        out._backward = _backward
        return out

    def __getitem__(self, key):
        # Basic indexing (slices / ints / Ellipsis); grad scatters back to the slice.
        out = self._child(self.data[key], (self,), None)

        def _backward():
            self.grad[key] += out.grad

        out._backward = _backward
        return out

    def sum(self):
        out = self._child(np.array(self.data.sum(), dtype=np.float32), (self,), None)

        def _backward():
            self.grad += np.ones_like(self.data) * out.grad

        out._backward = _backward
        return out

    def mean(self, axis: int | None = None, keepdims: bool = False):
        out_data = np.asarray(self.data.mean(axis=axis, keepdims=keepdims), dtype=np.float32)
        out = self._child(out_data, (self,), None)
        cnt = self.data.size if axis is None else self.data.shape[axis]

        def _backward():
            g = out.grad
            if axis is not None and not keepdims:
                g = np.expand_dims(g, axis)
            self.grad += np.ones_like(self.data) * (g / cnt)

        out._backward = _backward
        return out

    def softmax(self, axis: int = -1):
        z = self.data - self.data.max(axis=axis, keepdims=True)
        e = np.exp(z)
        p = e / e.sum(axis=axis, keepdims=True)
        out = self._child(p, (self,), None)

        def _backward():
            g = out.grad
            s = (g * p).sum(axis=axis, keepdims=True)
            self.grad += p * (g - s)

        out._backward = _backward
        return out

    def cross_entropy(self, targets: np.ndarray):
        """Softmax cross-entropy. ``self`` is logits [..., C]; targets int [...]."""
        targets = np.asarray(targets, dtype=np.int64).reshape(-1)
        logits2 = self.data.reshape(-1, self.data.shape[-1])
        z = logits2 - logits2.max(axis=-1, keepdims=True)
        e = np.exp(z)
        p = e / e.sum(axis=-1, keepdims=True)
        m = logits2.shape[0]
        logp = np.log(p[np.arange(m), targets] + 1e-12)
        loss = -float(logp.mean())
        out = self._child(np.array(loss, dtype=np.float32), (self,), None)

        def _backward():
            d = p.copy()
            d[np.arange(m), targets] -= 1.0
            d /= m
            self.grad += d.reshape(self.data.shape) * out.grad

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
    """Quantized linear layer: y = quant(x) @ quant(W).T + b, over n-D x.

    Forward uses ternary weights and per-token int8 activations (the deployed byte
    state). Backward is a straight-through estimator: quantization and the per-row
    activation scale are treated as identity, so gradients flow to the fp32
    ``master_w`` / ``bias`` as if y = x @ W.T + b.

    Shapes: x [..., in], master_w [out, in], bias [out] -> y [..., out].
    """
    wq, sw = quantize_weight_ternary(master_w.data)
    xd = np.asarray(x.data, dtype=np.float32)
    amax = np.max(np.abs(xd), axis=-1, keepdims=True)
    amax = np.where(np.isfinite(amax), amax, 0.0)
    sx = amax / 127.0 + EPS  # [..., 1]
    xq = np.clip(np.round(np.nan_to_num(xd / sx)), -127, 127).astype(np.int8)
    acc = xq.astype(np.int32) @ wq.T.astype(np.int32)  # [..., out]
    y = acc.astype(np.float32) * sx.astype(np.float32) * np.float32(sw)
    if bias is not None:
        y = y + bias.data

    parents = [x, master_w] + ([bias] if bias is not None else [])
    req = any(p.requires_grad for p in parents)
    out = Tensor(y, requires_grad=req, _prev=parents)

    if req:
        def _backward():
            dy = out.grad  # [..., out]
            x.grad += dy @ master_w.data  # STE: identity through quant + scale
            dy2 = dy.reshape(-1, dy.shape[-1])
            x2 = x.data.reshape(-1, x.data.shape[-1])
            master_w.grad += dy2.T @ x2
            if bias is not None:
                bias.grad += dy2.sum(axis=0)

        out._backward = _backward
    return out
