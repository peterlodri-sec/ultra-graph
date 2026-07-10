"""End-to-end batched mini-GPT on the ultragraph byte-graph.

A pre-norm transformer block (RMSNorm -> multi-head causal attention -> residual,
RMSNorm -> ternary MLP -> residual) trained with Adam over the fp32 masters, then
re-quantized to ternary. Runs under `uv run python examples/mini_gpt.py`.
"""
from __future__ import annotations

import numpy as np

from ultragraph import (
    Adam,
    Embedding,
    MultiHeadAttention,
    RMSNorm,
    Tensor,
    UltraGraph,
    linear_tree,
)


def main() -> None:
    np.random.seed(0)
    text = "hello ultra graph world " * 4
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}
    vocab = len(chars)
    d_model, n_heads, T = 24, 4, 8

    ids = np.array([stoi[c] for c in text], dtype=np.int64)
    seqs = np.stack([ids[i : i + T + 1] for i in range(len(ids) - T)])
    inputs, targets = seqs[:, :-1], seqs[:, 1:]
    B = inputs.shape[0]

    emb = Embedding(vocab, d_model, "emb")
    n1 = RMSNorm(d_model, name="n1")
    mha = MultiHeadAttention(d_model, n_heads, causal=True)
    n2 = RMSNorm(d_model, name="n2")
    ff1 = linear_tree(d_model, 4 * d_model, "ff1", act="relu")
    ff2 = linear_tree(4 * d_model, d_model, "ff2", act="none")
    unembed = linear_tree(d_model, vocab, "unembed", act="none")

    model = UltraGraph("mini_gpt")
    for m in (emb, n1, mha, n2):
        model.register(m)
    for t in (ff1, ff2, unembed):
        model.add(t)
    opt = Adam(model, lr=0.02, clip=1.0)

    def block(x):
        x = x + mha(n1(x))
        x = x + ff2.forward(ff1.forward(n2(x)))
        return x

    def forward_batched(idx):
        b = idx.shape[0]
        x = emb(idx.reshape(-1)).reshape(b, idx.shape[1], d_model)
        return unembed.forward(block(x))

    for step in range(300):
        logits = forward_batched(inputs)
        loss = logits.cross_entropy(targets)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 50 == 0:
            print(f"step {step:4d}  loss {float(loss.data):.4f}")

    # autoregressive greedy sampling from a seed context
    ctx = [stoi[c] for c in "hello ul"][:T]
    out = "".join(itos[i] for i in ctx)
    for _ in range(40):
        idx = np.array(ctx[-T:], dtype=np.int64).reshape(1, T)
        x = emb(idx.reshape(-1)).reshape(1, T, d_model)
        logits = unembed.forward(block(x))
        nxt = int(np.argmax(logits.data[0, -1]))
        ctx.append(nxt)
        out += itos[nxt]
    print("sample:", repr(out))


if __name__ == "__main__":
    main()
