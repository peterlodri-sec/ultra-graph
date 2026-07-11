"""Byte-level mixture-of-experts LM on the ultragraph byte-graph.

Embedding -> RMSNorm -> ternary MoE (residual) -> unembed, over UTF-8 bytes.
Run: uv run python examples/moe_lm.py
"""
from __future__ import annotations

import numpy as np

from ultragraph import Adam, ByteTokenizer, Embedding, MoE, RMSNorm, UltraGraph, linear_tree


def main() -> None:
    np.random.seed(0)
    tok = ByteTokenizer()
    text = "the loop is already here. minus one, zero, plus one. " * 4
    ids = tok.encode(text)
    T, d_model = 12, 24
    seqs = np.stack([ids[i : i + T + 1] for i in range(len(ids) - T)])
    inputs, targets = seqs[:, :-1], seqs[:, 1:]

    emb = Embedding(tok.vocab_size, d_model, "emb")
    n1 = RMSNorm(d_model, name="n1")
    moe = MoE(d_model, n_experts=4, hidden=64)
    unembed = linear_tree(d_model, tok.vocab_size, "unembed", act="none")
    model = UltraGraph("moe_lm")
    for m in (emb, n1, moe):
        model.register(m)
    model.add(unembed)
    opt = Adam(model, lr=0.02, clip=1.0)

    def fwd(idx):
        x = emb(idx.reshape(-1)).reshape(idx.shape[0], idx.shape[1], d_model)
        x = x + moe(n1(x))
        return unembed.forward(x)

    for step in range(300):
        loss = fwd(inputs).cross_entropy(targets)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 50 == 0:
            print(f"step {step:4d}  loss {float(loss.data):.4f}")

    ctx = list(tok.encode("the ")[-T:])
    ctx = [32] * (T - len(ctx)) + ctx
    out = []
    for _ in range(60):
        idx = np.array(ctx[-T:], dtype=np.int64).reshape(1, T)
        nxt = int(np.argmax(fwd(idx).data[0, -1]))
        ctx.append(nxt)
        out.append(nxt)
    print("sample:", repr(tok.decode(np.array(out))))


if __name__ == "__main__":
    main()
