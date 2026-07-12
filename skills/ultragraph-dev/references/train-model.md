# Train Model

Train, resume, evaluate, or deploy a ternary GPT/Mesh/MLP model.

## What Success Looks Like

A loss curve that converges, a generation sample that shows learned patterns, and a deployed bit-packed checkpoint ready for inference.

## Available model types

| Model | Use case | Key features |
|-------|----------|-------------|
| `GPT(vocab, d_model, n_layers, n_heads)` | Byte-level language model | RoPE + KV-cache, generate, save/load, save_deployed/load_deployed |
| `Mesh(experts, vocab, d_router)` | Mixture of full GPT models | Learned router, joint generate, per-expert KV-cache |
| `mlp(dims)` | Simple feed-forward | Stacked dense ternary trees with ultra-edges |

## Training setup

```python
from ultragraph import GPT, Adam, CosineSchedule, ByteTokenizer

model = GPT(vocab=256, d_model=128, n_layers=4, n_heads=4, max_len=256)
opt = Adam(model, lr=1e-3)
sched = CosineSchedule(opt, total_steps=5000, warmup=200)

for step in range(5000):
    logits = model(ids)
    loss = logits.cross_entropy(targets)
    opt.zero_grad()
    loss.backward()
    opt.step()
    sched.step()
```

## Resume training

The `save()`/`load()` checkpoint stores fp32 masters + model config. Load onto the same architecture and continue:

```python
model = GPT(vocab=256, d_model=128, n_layers=4, n_heads=4)
model.load("checkpoint.npz")  # re-quantizes automatically
# ... training loop continues
```

Use `TOTAL`/`STEPS` env vars pattern from `examples/hungarian_lm.py` to cope with wall-clock caps — save periodically, resume from latest.

## Gradient accumulation

```python
opt = Adam(model, lr=1e-3, accum_steps=4)  # DO NOT call zero_grad() in loop
```

## Deploy

```python
model.save_deployed("model.q.npz")   # bit-packed, ~1.6 bits/weight
deployed = GPT.load_deployed("model.q.npz")  # runs from ternary bytes
```

## Evaluation

- Check `model.n_params()` for total trainable scalar count
- Check loss convergence: `min(losses[-10:]) < 0.5 * losses[0]` is a reasonable bar
- Sample with `model.generate(prompt, n_new=64, temperature=0.8, top_k=40, top_p=0.9)`
