"""The first 1-bit Hungarian LLM — a ternary GPT trained on Anonymus.

Trains a byte-level ternary `GPT` (RoPE + KV-cache) on the *Gesta Hungarorum* of
Anonymus (c. 1200), the founding chronicle of the Hungarian nation. Weights are
ternary {-1,0,+1}; the whole model is a byte-graph. Fetch the corpus first with
`examples/fetch_gesta.py`, then:

    uv run python examples/anonymus_lm.py            # ~1500 steps
    STEPS=3000 uv run python examples/anonymus_lm.py # longer / better samples

Writes a deployed (bit-packed, ~1.6 bits/weight) checkpoint next to the corpus.
"""

import os

import numpy as np

from ultragraph import GPT, Adam, ByteTokenizer, CosineSchedule

DATA = os.path.join(os.path.dirname(__file__), "data", "gesta_hungarorum.txt")
CKPT = os.path.join(os.path.dirname(__file__), "data", "anonymus.gpt.npz")


def sample(model, tok, seed, n=200, temperature=0.85, top_p=0.9, seed_rng=0):
    ids = tok.encode(seed)
    out = model.generate(ids, n_new=n, temperature=temperature, top_p=top_p, seed=seed_rng)
    return tok.decode(out)


def main() -> None:
    np.random.seed(0)
    steps = int(os.environ.get("STEPS", "1500"))
    T, batch = 64, 16

    tok = ByteTokenizer()
    with open(DATA, encoding="utf-8") as f:
        text = f.read()
    data = tok.encode(text)
    print(f"corpus: {len(text):,} chars / {len(data):,} bytes")

    model = GPT(vocab=256, d_model=96, n_layers=3, n_heads=4, max_len=512)  # room to decode past T
    print(f"model: {model.n_params():,} trainable params (ternary weights + fp32 embed/norms)")
    opt = Adam(model, lr=3e-3, clip=1.0)
    sched = CosineSchedule(opt, total_steps=steps, warmup=steps // 20)

    rng = np.random.default_rng(0)
    for step in range(1, steps + 1):
        starts = rng.integers(0, len(data) - T - 1, size=batch)
        xb = np.stack([data[s : s + T] for s in starts])
        yb = np.stack([data[s + 1 : s + T + 1] for s in starts])
        loss = model(xb).cross_entropy(yb)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if step % 100 == 0 or step == 1:
            print(f"step {step:5d}/{steps}  loss {float(loss.data):.4f}  lr {opt.lr:.5f}")
        if step % 500 == 0:
            print("  sample:", repr(sample(model, tok, "P. dictus magister ", n=120))[:220])

    model.save_deployed(CKPT)
    kb = os.path.getsize(CKPT) / 1024
    print(f"\nsaved deployed (1-bit) checkpoint: {CKPT}  ({kb:.0f} KB)")
    for seed in ("Almus dux ", "In terra scithica ", "Arpad "):
        print(f"\n=== {seed!r} ===")
        print(sample(model, tok, seed, n=280, temperature=0.8, top_p=0.9))


if __name__ == "__main__":
    main()
