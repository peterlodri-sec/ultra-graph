"""Composed ternary transformer: pre-norm blocks and a ``GPT`` with generation.

``GPT`` wires an embedding, a shared :class:`~ultragraph.nn.RoPE`, N pre-norm
:class:`TransformerBlock` s, a final norm, and a ternary output head. Every
weight matrix is a dense ternary :class:`~ultragraph.core.Tree`; the whole model
is one byte-graph. ``generate`` decodes autoregressively with a per-layer
KV-cache — and because activation quantization is per token, a cached step is
byte-for-byte the same as the full forward pass at that position.
"""
from __future__ import annotations

import numpy as np

from .core import Embedding, Tree
from .nn import MultiHeadAttention, RMSNorm, RoPE


class TransformerBlock:
    """Pre-norm block: ``x = x + attn(norm1(x)); x = x + ff(norm2(x))``."""

    def __init__(self, d_model, n_heads, mlp_ratio=4, rope=None, name="block"):
        self.name = name
        self.norm1 = RMSNorm(d_model, name=f"{name}.n1")
        self.attn = MultiHeadAttention(d_model, n_heads, causal=True, rope=rope, name=f"{name}.attn")
        self.norm2 = RMSNorm(d_model, name=f"{name}.n2")
        h = d_model * int(mlp_ratio)
        self.ff1 = Tree.dense(d_model, h, f"{name}.ff1", act="relu")
        self.ff2 = Tree.dense(h, d_model, f"{name}.ff2", act="none")

    def __call__(self, x, cache=None):
        if cache is not None:
            a, new_cache = self.attn(self.norm1(x), cache=cache)
            x = x + a
        else:
            x = x + self.attn(self.norm1(x))
        x = x + self.ff2.forward(self.ff1.forward(self.norm2(x)))
        return (x, new_cache) if cache is not None else x

    def parameters(self):
        ps = list(self.norm1.parameters()) + list(self.attn.parameters())
        ps += list(self.norm2.parameters()) + list(self.ff1.parameters()) + list(self.ff2.parameters())
        return ps

    def requantize(self):
        self.attn.requantize()
        self.ff1.requantize()
        self.ff2.requantize()


class GPT:
    """A ternary GPT: token embedding + RoPE + N pre-norm blocks + ternary head."""

    def __init__(self, vocab, d_model, n_layers, n_heads, max_len=512, mlp_ratio=4, name="gpt"):
        assert (d_model // n_heads) % 2 == 0, "d_head (d_model // n_heads) must be even for RoPE"
        self.vocab = int(vocab)
        self.d_model = int(d_model)
        self.n_layers = int(n_layers)
        self.n_heads = int(n_heads)
        self.max_len = int(max_len)
        self.name = name
        self.embed = Embedding(vocab, d_model, name=f"{name}.embed")
        self.rope = RoPE(d_model // n_heads, max_len=max_len, name=f"{name}.rope")
        self.blocks = [
            TransformerBlock(d_model, n_heads, mlp_ratio, rope=self.rope, name=f"{name}.b{i}")
            for i in range(self.n_layers)
        ]
        self.norm = RMSNorm(d_model, name=f"{name}.nf")
        self.head = Tree.dense(d_model, vocab, f"{name}.head", act="none")

    def __call__(self, ids):
        """ids: int array ``[T]`` or ``[B, T]`` -> logits ``[..., T, vocab]``."""
        x = self.embed(np.asarray(ids))
        for b in self.blocks:
            x = b(x)
        return self.head.forward(self.norm(x))

    def parameters(self):
        ps = list(self.embed.parameters())
        for b in self.blocks:
            ps += b.parameters()
        ps += list(self.norm.parameters()) + list(self.head.parameters())
        return ps

    def requantize(self):
        for b in self.blocks:
            b.requantize()
        self.head.requantize()

    # -- generation -----------------------------------------------------------
    def generate(self, prompt, n_new, temperature=1.0, top_k=None, seed=None):
        """Autoregressive decode with a per-layer KV-cache. Returns ``prompt + n_new``
        token ids. ``temperature <= 0`` is greedy (argmax)."""
        rng = np.random.default_rng(seed)
        ids = [int(t) for t in np.asarray(prompt, dtype=np.int64).reshape(-1)]
        if not ids:
            raise ValueError("prompt must be non-empty")
        caches = [{"k": None, "v": None} for _ in self.blocks]
        logits = self._decode(np.array(ids, dtype=np.int64)[None, :], caches)  # prime
        for _ in range(int(n_new)):
            nxt = self._sample(logits[0, -1], temperature, top_k, rng)
            ids.append(nxt)
            logits = self._decode(np.array([[nxt]], dtype=np.int64), caches)
        return ids

    def _decode(self, ids, caches):
        x = self.embed(ids)
        for b, c in zip(self.blocks, caches):
            x, nc = b(x, cache=c)
            c["k"], c["v"] = nc["k"], nc["v"]
        return self.head.forward(self.norm(x)).data  # [B, T, vocab] fp32

    @staticmethod
    def _sample(logits, temperature, top_k, rng):
        logits = logits.astype(np.float64)
        if temperature <= 0:
            return int(np.argmax(logits))
        logits = logits / float(temperature)
        if top_k:
            k = min(int(top_k), logits.shape[-1])
            logits = np.where(logits < np.sort(logits)[-k], -np.inf, logits)
        z = logits - logits.max()
        p = np.exp(z)
        p /= p.sum()
        return int(rng.choice(len(p), p=p))
