"""A mesh of ternary minds — a learned mixture of full GPTs.

Trains a `Mesh` of small byte-level `GPT` experts on a toy corpus using Adam with
**gradient accumulation** (learning on a budget: sum grads over micro-batches, step
once), inspects the router gate, then decodes jointly with `Mesh.generate` and shows
batched generation from a single expert. Runs under `uv run python examples/mesh_lm.py`.
"""

import numpy as np

from ultragraph import GPT, Adam, ByteTokenizer, Mesh


def main() -> None:
    np.random.seed(0)
    tok = ByteTokenizer()
    text = "the mesh is the model. the model is the mesh. " * 8
    ids = tok.encode(text)

    T = 16
    seqs = np.stack([ids[i : i + T + 1] for i in range(len(ids) - T)])
    inputs, targets = seqs[:, :-1], seqs[:, 1:]

    experts = [GPT(vocab=256, d_model=32, n_layers=1, n_heads=4, max_len=64) for _ in range(3)]
    mesh = Mesh(experts, vocab=256, top_k=2)
    print(f"mesh: {mesh.n_experts} experts, {mesh.n_params():,} trainable params")

    # gradient accumulation: 2 micro-batches per optimizer update
    opt = Adam(mesh, lr=0.01, clip=1.0, accum_steps=2)
    micro = np.array_split(np.arange(inputs.shape[0]), 2)
    for step in range(200):
        for mb in micro:
            loss = mesh(inputs[mb]).cross_entropy(targets[mb])
            loss.backward()
            opt.step()   # applies (and zeros) only every 2nd call
        if step % 40 == 0:
            print(f"step {step:4d}  loss {float(loss.data):.4f}")

    gate = mesh._gate(tok.encode("the mesh")[None, :]).data[0]
    print("\nrouter gate over experts:", np.round(gate, 3))

    prompt = tok.encode("the mesh")
    out = mesh.generate(prompt, n_new=40, temperature=0.0)  # joint greedy decode
    print("mesh greedy:", repr(tok.decode(out)))

    # batched generation from one expert (equal-length prompts, per-sequence sampling)
    prompts = np.stack([tok.encode("the mesh"), tok.encode("the mode")])
    batch = experts[0].generate_batch(prompts, n_new=24, temperature=0.0)
    for row in batch:
        print("batch:", repr(tok.decode(row)))


if __name__ == "__main__":
    main()
