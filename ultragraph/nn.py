"""Layer patterns expressed as trees. Layers are configs over the byte-graph."""

import math

import numpy as np

from .autograd import Tensor, cat
from .core import Tree, UltraGraph


def linear_tree(in_dim: int, out_dim: int, name: str = "linear", act: str = "relu") -> Tree:
    """A dense ternary Linear layer as a tree."""
    return Tree.dense(in_dim, out_dim, name=name, act=act)


class Linear:
    """A dense ternary linear layer. Wraps ``Tree.dense`` with a clean API.

    Example:
        layer = nn.Linear(128, 256)
        y = layer(x)  # forward with STE, activation quantized to int8
    """

    def __init__(self, in_dim: int, out_dim: int, act: str = "relu", name: str = "linear"):
        self.tree = Tree.dense(in_dim, out_dim, name=name, act=act)

    def __call__(self, x):
        return self.tree.forward(x)

    @property
    def w_master(self):
        return self.tree.adhoc["w_master"]

    @property
    def bias(self):
        return self.tree.adhoc["bias"]

    def parameters(self):
        return self.tree.parameters()

    def requantize(self):
        self.tree.requantize()


class Residual:
    """Skip-connection wrapper: ``y = layer(x) + x``. Drops in anywhere.

    Example:
        block = Sequential(Linear(64, 64), Residual(Linear(64, 64)))
    """

    def __init__(self, module):
        self.module = module

    def __call__(self, x):
        return self.module(x) + x

    def parameters(self):
        if hasattr(self.module, "parameters"):
            return self.module.parameters()
        return []

    def requantize(self):
        if callable(getattr(self.module, "requantize", None)):
            self.module.requantize()


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
    def __init__(self, d_model, n_heads, causal=True, rope=None, name="mha"):
        assert d_model % n_heads == 0, "n_heads must divide d_model"
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.d_head = d_model // n_heads
        self.causal = bool(causal)
        self.rope = rope
        if rope is not None:
            assert self.d_head % 2 == 0, "d_head must be even for RoPE"
        self.name = name
        self.wq = Tree.dense(d_model, d_model, f"{name}.q", act="none")
        self.wk = Tree.dense(d_model, d_model, f"{name}.k", act="none")
        self.wv = Tree.dense(d_model, d_model, f"{name}.v", act="none")
        self.wo = Tree.dense(d_model, d_model, f"{name}.o", act="none")

    def __call__(self, x, cache=None):
        """x: [T, d] or [B, T, d]. With ``cache`` (a {'k','v'} dict) does
        incremental decoding: returns ``(out, new_cache)`` and attends over the
        cached keys/values plus the new tokens. Without a cache: returns ``out``."""
        two_d = (len(x.shape) == 2)
        if two_d:
            T, d = x.shape
            x = x.reshape(1, T, d)
        B, T, d = x.shape
        H, dh = self.n_heads, self.d_head
        q = self.wq.forward(x).reshape(B, T, H, dh).swapaxes(1, 2)   # [B,H,T,dh]
        k = self.wk.forward(x).reshape(B, T, H, dh).swapaxes(1, 2)
        v = self.wv.forward(x).reshape(B, T, H, dh).swapaxes(1, 2)
        offset = 0
        if cache is not None and cache.get("k") is not None:
            offset = cache["k"].shape[-2]                            # past key length
        if self.rope is not None:
            q = self.rope(q, offset=offset)
            k = self.rope(k, offset=offset)
        new_cache = None
        if cache is not None:
            if cache.get("k") is not None:
                k = cat([cache["k"], k], axis=-2)                    # prepend past K/V
                v = cat([cache["v"], v], axis=-2)
            new_cache = {"k": k, "v": v}
        Tk = k.shape[-2]
        scores = (q @ k.swapaxes(-1, -2)) * (1.0 / math.sqrt(dh))    # [B,H,T,Tk]
        if self.causal:
            qpos = np.arange(offset, offset + T)[:, None]           # abs query positions
            kpos = np.arange(Tk)[None, :]
            mask = np.where(kpos <= qpos, 0.0, -1e9).astype(np.float32)  # [T, Tk]
            scores = scores + Tensor(mask)                          # broadcasts over [B,H,·,·]
        attn = scores.softmax(axis=-1)
        ctx = (attn @ v).swapaxes(1, 2).reshape(B, T, d)            # merge heads
        out = self.wo.forward(ctx)                                  # [B,T,d]
        if two_d:
            out = out.reshape(T, d)
        return (out, new_cache) if cache is not None else out

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


class LearnedPositionalEmbedding:
    """Learnable position vectors added to a [..., T, dim] sequence (fp32, not ternary)."""

    def __init__(self, max_len, dim, name="pos"):
        self.max_len = int(max_len)
        self.dim = int(dim)
        self.name = name
        self.table = Tensor(np.random.randn(self.max_len, dim).astype(np.float32) * 0.02, requires_grad=True)

    def __call__(self, x):
        t = x.shape[-2]
        if t > self.max_len:
            raise ValueError(f"sequence length {t} exceeds max_len {self.max_len}")
        tbl = self.table
        pos = Tensor(tbl.data[:t], requires_grad=tbl.requires_grad, _prev=(tbl,))

        def _backward():
            tbl.grad[:t] += pos.grad

        if pos.requires_grad:
            pos._backward = _backward
        return x + pos  # broadcasts [t, dim] over any leading dims

    def parameters(self):
        return [self.table]


class RoPE:
    """Rotary positional embedding (RoPE). Rotates pairs of feature dims by a
    position-dependent angle, applied to a ``[..., T, dim]`` tensor (``dim`` even).

    No learnable parameters. Encodes *relative* position directly in the
    attention dot-product, so it generalizes past the primed length and plugs
    into a KV-cache via the ``offset`` (absolute start position).
    """

    def __init__(self, dim, max_len=4096, base=10000.0, name="rope"):
        assert dim % 2 == 0, "RoPE dim must be even"
        self.dim = int(dim)
        self.max_len = int(max_len)
        self.name = name
        half = self.dim // 2
        inv_freq = 1.0 / (base ** (np.arange(half, dtype=np.float32) * 2.0 / self.dim))
        pos = np.arange(self.max_len, dtype=np.float32)
        freqs = np.outer(pos, inv_freq)                       # [max_len, half]
        emb = np.concatenate([freqs, freqs], axis=-1)         # [max_len, dim]
        self._cos = np.cos(emb).astype(np.float32)
        self._sin = np.sin(emb).astype(np.float32)

    def __call__(self, x, offset=0):
        t = x.shape[-2]
        if offset + t > self.max_len:
            raise ValueError(f"RoPE positions {offset}+{t} exceed max_len {self.max_len}")
        cos = Tensor(self._cos[offset:offset + t])            # [T, dim], broadcasts
        sin = Tensor(self._sin[offset:offset + t])
        return x * cos + self._rotate_half(x) * sin

    @staticmethod
    def _rotate_half(x):
        h = x.shape[-1] // 2
        return cat([-x[..., h:], x[..., :h]], axis=-1)

    def parameters(self):
        return []


class MoE:
    """Soft mixture of ternary-MLP experts with a learned router.

    ``output = sum_e softmax(router(x))_e * expert_e(x)``, over the last axis. The
    router and every expert are dense ternary trees, so the whole block is still
    a byte-graph. Fully differentiable (soft routing over all experts).
    """

    def __init__(self, dim, n_experts=4, hidden=None, top_k=None, name="moe"):
        self.dim = int(dim)
        self.n_experts = int(n_experts)
        if top_k is None:
            self.top_k = self.n_experts
        else:
            self.top_k = int(top_k)
            if not (1 <= self.top_k <= self.n_experts):
                raise ValueError(f"top_k must be in [1, {self.n_experts}], got {top_k}")
        self.name = name
        h = int(hidden) if hidden else 4 * dim
        self.router = Tree.dense(dim, self.n_experts, f"{name}.router", act="none")
        self.experts_in = [Tree.dense(dim, h, f"{name}.e{i}.in", act="relu") for i in range(self.n_experts)]
        self.experts_out = [Tree.dense(h, dim, f"{name}.e{i}.out", act="none") for i in range(self.n_experts)]

    def __call__(self, x):
        gate = self.router.forward(x).softmax(axis=-1)  # [..., n_experts]
        if self.top_k < self.n_experts:
            # keep only the top-k experts per token, then renormalize the gates
            g = gate.data
            idx = np.argpartition(-g, self.top_k - 1, axis=-1)[..., : self.top_k]
            mask = np.zeros_like(g)
            np.put_along_axis(mask, idx, 1.0, axis=-1)
            gate = gate * Tensor(mask)
            gate = gate / (gate.sum(axis=-1, keepdims=True) + 1e-9)
        out = None
        for i in range(self.n_experts):
            y = self.experts_out[i].forward(self.experts_in[i].forward(x))  # [..., dim]
            term = y * gate[..., i : i + 1]  # broadcast [...,1] over [...,dim]
            out = term if out is None else out + term
        return out

    def parameters(self):
        ps = list(self.router.parameters())
        for t in self.experts_in + self.experts_out:
            ps.extend(t.parameters())
        return ps

    def requantize(self):
        self.router.requantize()
        for t in self.experts_in + self.experts_out:
            t.requantize()


class Dropout:
    """Inverted dropout. When ``training`` is True, zeroes a fraction ``p`` of the
    activations and rescales the rest by ``1/(1-p)``; when False it is a passthrough.
    """

    def __init__(self, p=0.1, name="dropout"):
        self.p = float(p)
        self.name = name
        self.training = True

    def __call__(self, x):
        if not self.training or self.p <= 0.0:
            return x
        keep = 1.0 - self.p
        mask = (np.random.rand(*x.shape) < keep).astype(np.float32) / keep
        return x * Tensor(mask)

    def parameters(self):
        return []


class Sequential:
    """Chain modules/callables: each output feeds the next. Aggregates parameters,
    propagates train/eval (e.g. to Dropout), and re-quantizes ternary submodules."""

    def __init__(self, *modules, name="sequential"):
        self.modules = list(modules)
        self.name = name

    def __call__(self, x):
        for m in self.modules:
            x = m(x)
        return x

    def parameters(self):
        ps = []
        for m in self.modules:
            if hasattr(m, "parameters"):
                ps.extend(m.parameters())
        return ps

    def requantize(self):
        for m in self.modules:
            r = getattr(m, "requantize", None)
            if callable(r):
                r()

    def train(self, mode: bool = True):
        for m in self.modules:
            match m:
                case _ if callable(getattr(m, "train", None)):
                    m.train(mode)  # recurse into nested Sequential
                case _ if hasattr(m, "training"):
                    m.training = mode
        return self

    def eval(self):
        return self.train(False)
