"""Layer patterns expressed as trees. Layers are configs over the byte-graph."""
from __future__ import annotations

import math

import numpy as np

from .autograd import Tensor
from .core import Tree, UltraGraph


def linear_tree(in_dim: int, out_dim: int, name: str = "linear", act: str = "relu") -> Tree:
    """A dense ternary Linear layer as a tree."""
    return Tree.dense(in_dim, out_dim, name=name, act=act)


def mlp(dims: list[int], name: str = "mlp") -> UltraGraph:
    """Stack dense ternary linear trees, wired plain, relu between; last layer linear.

    ``dims`` is [in, hidden1, ..., out].
    """
    if len(dims) < 2:
        raise ValueError("mlp needs at least an input and output dim")
    ug = UltraGraph(name)
    prev = None
    for i in range(len(dims) - 1):
        is_last = i == len(dims) - 2
        t = linear_tree(dims[i], dims[i + 1], name=f"linear{i}", act="none" if is_last else "relu")
        ug.add(t)
        if prev is not None:
            prev >> t  # plain ultra-edge
        prev = t
    return ug


class Attention:
    """Single-head ternary self-attention over a [T, d_model] sequence.

    Q,K,V,O are dense ternary linear trees; scaled dot-product attention with
    optional causal masking. Plugs into the module protocol (parameters /
    requantize) so an optimizer treats it like any other module.
    """
    def __init__(self, d_model, d_head=None, causal=True, name="attn"):
        self.d_model = int(d_model)
        self.d_head = int(d_head) if d_head else int(d_model)
        self.causal = bool(causal)
        self.name = name
        self.wq = Tree.dense(self.d_model, self.d_head, f"{name}.q", act="none")
        self.wk = Tree.dense(self.d_model, self.d_head, f"{name}.k", act="none")
        self.wv = Tree.dense(self.d_model, self.d_head, f"{name}.v", act="none")
        self.wo = Tree.dense(self.d_head, self.d_model, f"{name}.o", act="none")

    def __call__(self, x):            # x: Tensor [T, d_model] -> Tensor [T, d_model]
        q = self.wq.forward(x)        # [T, d_head]
        k = self.wk.forward(x)
        v = self.wv.forward(x)
        scores = (q @ k.transpose()) * (1.0 / math.sqrt(self.d_head))   # [T, T]
        if self.causal:
            T = x.shape[0]
            mask = np.triu(np.full((T, T), -1e9, dtype=np.float32), k=1)
            scores = scores + Tensor(mask)     # additive causal mask (no grad)
        attn = scores.softmax(axis=-1)          # [T, T]
        ctx = attn @ v                          # [T, d_head]
        return self.wo.forward(ctx)             # [T, d_model]

    def parameters(self):
        ps = []
        for t in (self.wq, self.wk, self.wv, self.wo):
            ps.extend(t.parameters())
        return ps

    def requantize(self):
        for t in (self.wq, self.wk, self.wv, self.wo):
            t.requantize()


class MultiHeadAttention:
    def __init__(self, d_model, n_heads, causal=True, name="mha"):
        assert d_model % n_heads == 0, "n_heads must divide d_model"
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.d_head = d_model // n_heads
        self.causal = bool(causal)
        self.name = name
        self.wq = Tree.dense(d_model, d_model, f"{name}.q", act="none")
        self.wk = Tree.dense(d_model, d_model, f"{name}.k", act="none")
        self.wv = Tree.dense(d_model, d_model, f"{name}.v", act="none")
        self.wo = Tree.dense(d_model, d_model, f"{name}.o", act="none")

    def __call__(self, x):
        two_d = (len(x.shape) == 2)
        if two_d:
            T, d = x.shape
            x = x.reshape(1, T, d)
        B, T, d = x.shape
        H, dh = self.n_heads, self.d_head
        q = self.wq.forward(x).reshape(B, T, H, dh).swapaxes(1, 2)   # [B,H,T,dh]
        k = self.wk.forward(x).reshape(B, T, H, dh).swapaxes(1, 2)
        v = self.wv.forward(x).reshape(B, T, H, dh).swapaxes(1, 2)
        scores = (q @ k.swapaxes(-1, -2)) * (1.0 / math.sqrt(dh))    # [B,H,T,T]
        if self.causal:
            mask = np.triu(np.full((T, T), -1e9, dtype=np.float32), k=1)
            scores = scores + Tensor(mask)                          # broadcasts over [B,H,·,·]
        attn = scores.softmax(axis=-1)
        ctx = (attn @ v).swapaxes(1, 2).reshape(B, T, d)            # merge heads
        out = self.wo.forward(ctx)                                  # [B,T,d]
        if two_d:
            out = out.reshape(T, d)
        return out

    def parameters(self):
        ps = []
        for t in (self.wq, self.wk, self.wv, self.wo):
            ps.extend(t.parameters())
        return ps

    def requantize(self):
        for t in (self.wq, self.wk, self.wv, self.wo):
            t.requantize()


class RMSNorm:
    def __init__(self, dim, eps=1e-5, name="rmsnorm"):
        self.dim = int(dim)
        self.eps = float(eps)
        self.name = name
        self.gain = Tensor(np.ones(dim, dtype=np.float32), requires_grad=True)

    def __call__(self, x):
        ms = (x * x).mean(axis=-1, keepdims=True)      # [...,1]
        return (x / (ms + self.eps).sqrt()) * self.gain

    def parameters(self):
        return [self.gain]


class LayerNorm:
    def __init__(self, dim, eps=1e-5, name="layernorm"):
        self.dim = int(dim)
        self.eps = float(eps)
        self.name = name
        self.gain = Tensor(np.ones(dim, dtype=np.float32), requires_grad=True)
        self.bias = Tensor(np.zeros(dim, dtype=np.float32), requires_grad=True)

    def __call__(self, x):
        mu = x.mean(axis=-1, keepdims=True)
        xc = x - mu
        var = (xc * xc).mean(axis=-1, keepdims=True)
        return (xc / (var + self.eps).sqrt()) * self.gain + self.bias

    def parameters(self):
        return [self.gain, self.bias]
