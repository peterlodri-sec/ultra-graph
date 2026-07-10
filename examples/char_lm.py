"""Tiny char-level ternary language model demo (ultragraph public API).

Trains a minuscule char-level LM on a short corpus using an ``Embedding`` feeding
a ternary-linear ``mlp`` (an ``UltraGraph`` of dense trees). Prints the training
loss decreasing, then greedily samples a short continuation.

Run:
    uv run python examples/char_lm.py
"""
from __future__ import annotations

import numpy as np

from ultragraph import Embedding, SGD, Tensor, mlp


def main() -> None:
    text = "hello ultra graph world "
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}
    vocab = len(chars)
    dim = 8

    ids = np.array([stoi[c] for c in text], dtype=np.int64)
    inputs, targets = ids[:-1], ids[1:]

    emb = Embedding(vocab, dim, "emb")
    net = mlp([dim, 16, vocab])   # UltraGraph of dense ternary linear trees
    net.register(emb)             # emb params train + are in net.parameters()
    opt = SGD(net, lr=0.1, momentum=0.9)

    for step in range(400):
        e = emb(inputs)               # -> Tensor [n, dim]
        logits = net.forward(e)       # -> Tensor [n, vocab]
        loss = logits.cross_entropy(targets)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 50 == 0:
            print(f"step {step:4d}  loss {float(loss.data):.3f}")

    # greedy sample:
    cur = stoi[chars[0]]
    out = chars[0]
    for _ in range(30):
        e = emb(np.array([cur]))
        logits = net.forward(e)
        cur = int(np.argmax(logits.data[0]))
        out += itos[cur]
    print("sample:", repr(out))


if __name__ == "__main__":
    main()
