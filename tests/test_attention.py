"""Tests for the ternary self-attention block (``ultragraph.nn.Attention``)."""

import numpy as np

from ultragraph import SGD, Embedding, Tensor, UltraGraph, linear_tree
from ultragraph.nn import Attention


def test_attention_forward_shape_finite():
    attn = Attention(8, causal=True)
    np.random.seed(0)
    x = Tensor(np.random.randn(5, 8).astype(np.float32))
    out = attn(x)
    assert out.shape == (5, 8)
    assert np.isfinite(out.data).all()


def test_attention_causality():
    np.random.seed(1)
    attn = Attention(8, causal=True)
    X = np.random.randn(6, 8).astype(np.float32)
    out1 = attn(Tensor(X.copy())).data.copy()
    X2 = X.copy()
    X2[5] += 3.0  # change only the LAST position
    out2 = attn(Tensor(X2)).data
    # earlier positions must be unchanged (no future -> past leak)
    assert np.allclose(out1[:5], out2[:5], atol=1e-4)
    # the changed position did affect its own output (sanity)
    assert not np.allclose(out1[5], out2[5], atol=1e-4)


def test_transformer_lm_trains():
    np.random.seed(0)
    text = "hello ultra graph world "
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    vocab = len(chars)
    d_model = 16
    ids = np.array([stoi[c] for c in text], dtype=np.int64)
    inputs, targets = ids[:-1], ids[1:]

    emb = Embedding(vocab, d_model, "emb")
    attn = Attention(d_model, causal=True)
    ff1 = linear_tree(d_model, 4 * d_model, "ff1", act="relu")
    ff2 = linear_tree(4 * d_model, d_model, "ff2", act="none")
    unembed = linear_tree(d_model, vocab, "unembed", act="none")

    model = UltraGraph("transformer")
    model.register(emb)
    model.register(attn)
    model.add(ff1)
    model.add(ff2)
    model.add(unembed)

    opt = SGD(model, lr=0.05, momentum=0.9, clip=1.0)

    def forward(idx):
        x = emb(idx)                # [T, d_model]
        x = x + attn(x)             # residual attention
        h = ff2.forward(ff1.forward(x))
        x = x + h                   # residual mlp
        return unembed.forward(x)   # [T, vocab]

    losses = []
    for _ in range(400):
        logits = forward(inputs)
        loss = logits.cross_entropy(targets)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss.data))

    assert min(losses[-10:]) < 0.8 * losses[0], (losses[0], min(losses[-10:]))
