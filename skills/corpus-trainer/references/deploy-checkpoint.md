# Deploy Checkpoint

Take a trained fp32-master checkpoint and produce a bit-packed deployed checkpoint for inference.

## What Success Looks Like

A `.npz` file at ~1.6 bits/weight (typically 5-10x smaller than the training checkpoint) that loads with `GPT.load_deployed()` and produces byte-identical logits to the trained model.

## How to deploy

```python
from ultragraph import GPT

# First, have the trained model in memory
model.save_deployed("examples/data/model.q.npz")  # bit-packed ternary bytes only

# Inference from deployed — no fp32 masters needed
deployed = GPT.load_deployed("examples/data/model.q.npz")
logits = deployed(ids)  # runs from ternary bytes, same logits as trained model
```

## What `save_deployed` stores

| Component | Storage | Size |
|-----------|---------|------|
| Ternary weight matrices | Bit-packed via `pack_ternary()` (5 values/byte) | ~1.6 bits/weight |
| Embedding table | fp32 (small — 256 × d_model) | Tiny |
| Norm gains | fp32 (one per RMSNorm) | Negligible |
| Biases | fp32 (one per dense tree) | Negligible |
| Hyperparameters | JSON in `__meta__` array | ~100 bytes |

## Verify deployment

```python
# Generate same output on trained vs deployed
trained_out = model.generate(prompt, n_new=32)
deployed_out = deployed.generate(prompt, n_new=32)
assert trained_out == deployed_out  # byte-exact
```

## Loading deployed

`GPT.load_deployed()` reads the JSON `__meta__` to reconstruct the architecture, so you don't need to remember the hyperparameters:
```python
m = GPT.load_deployed("model.q.npz")  # infers vocab, d_model, n_layers, etc.
```
