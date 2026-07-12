"""Tiny 1-layer ternary transformer language model demo.

Trains a residual attention + MLP block on a short char corpus, then greedily
samples a few characters. Run with: ``uv run python examples/transformer_lm.py``.
"""

import numpy as np

from ultragraph import SGD, Embedding, UltraGraph, linear_tree
from ultragraph.nn import Attention


def main():
    np.random.seed(0)
    text = "hello ultra graph world "
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}
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

    for step in range(400):
        logits = forward(inputs)
        loss = logits.cross_entropy(targets)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 50 == 0:
            print(f"step {step:4d}  loss {float(loss.data):.4f}")

    # greedy sample: start from the first char, argmax next-token 30 times
    cur = ids[:1].copy()
    out_chars = [itos[int(cur[0])]]
    for _ in range(30):
        logits = forward(cur)
        nxt = int(np.argmax(logits.data[-1]))
        out_chars.append(itos[nxt])
        cur = np.append(cur, np.int64(nxt))
    print("sample:", "".join(out_chars))


if __name__ == "__main__":
    main()
