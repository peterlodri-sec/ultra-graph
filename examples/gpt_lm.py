"""End-to-end byte-level ternary GPT — the whole stack in one file.

`ByteTokenizer` (vocab 256) -> `GPT` (embedding + RoPE + pre-norm blocks + ternary
head) -> Adam over the fp32 masters -> re-quantized ternary weights -> KV-cached
streaming generation. Save/load round-trips the trained model. Runs under
`uv run python examples/gpt_lm.py`.
"""

import tempfile

import numpy as np

from ultragraph import GPT, Adam, ByteTokenizer


def main() -> None:
    np.random.seed(0)
    tok = ByteTokenizer()
    text = "the graph is the model. the model is the graph. " * 8
    ids = tok.encode(text)                      # int64 bytes in [0, 255]

    T = 16
    seqs = np.stack([ids[i : i + T + 1] for i in range(len(ids) - T)])
    inputs, targets = seqs[:, :-1], seqs[:, 1:]

    model = GPT(vocab=256, d_model=48, n_layers=2, n_heads=4, max_len=64)
    opt = Adam(model, lr=0.01, clip=1.0)

    for step in range(400):
        loss = model(inputs).cross_entropy(targets)
        opt.zero_grad()
        loss.backward()
        opt.step()                              # re-quantizes ternary weights
        if step % 50 == 0:
            print(f"step {step:4d}  loss {float(loss.data):.4f}")

    prompt = tok.encode("the graph")
    print("\ngreedy:  ", repr(tok.decode(model.generate(prompt, n_new=48, temperature=0.0))))

    print("stream:   ", end="", flush=True)
    out = list(prompt)
    for t in model.generate(prompt, n_new=48, temperature=0.7, top_p=0.9,
                            repetition_penalty=1.3, seed=0, stream=True):
        out.append(t)
    print(repr(tok.decode(out)))

    # save / load round-trips the trained model (byte-exact inference)
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
        path = f.name
    model.save(path)
    reloaded = GPT(vocab=256, d_model=48, n_layers=2, n_heads=4, max_len=64).load(path)
    a = model.generate(prompt, n_new=32, temperature=0.0)
    b = reloaded.generate(prompt, n_new=32, temperature=0.0)
    print("\nsave/load identical greedy:", a == b)


if __name__ == "__main__":
    main()
