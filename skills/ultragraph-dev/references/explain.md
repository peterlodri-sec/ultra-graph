# Explain

Explain byte-graph internals — the concepts that make ultragraph work.

## Key concepts to explain

### The byte contract
One node = one `int8` activation. One edge weight = one ternary value in {-1,0,+1} stored as `int8`. Semantic values fit in one byte; adjacency bookkeeping (src/dst indices) is structural overhead and doesn't count against the contract.

### Three-level hierarchy
Micro (node/edge, 1 byte each) → Meso (Tree = one net/module) → Macro (UltraGraph = model). Trees can be **dense** (full weight matrix, for linear layers) or **sparse** (explicit edge lists, for knowledge graphs).

### Autograd tape
Home-grown reverse-mode autograd over numpy float32. Op granularity is tensor, not scalar. `Tensor._child()` creates nodes, `_backward` closures implement the chain rule, and `backward()` walks the `_prev` DAG in topological order.

### STE (straight-through estimator)
Forward: `ternary_linear` quantizes weights to {-1,0,+1} via absmean scaling. Backward: gradients pass through the **unquantized** surrogate `y = x @ W.T + b` as if no quantization happened. This is why training works — the master weights stay fp32 and the byte buffers track them.

### Deployed inference
When `Tree.adhoc["w_master"] = None`, forward runs from the ternary weight bytes directly (`ternary_forward` instead of `ternary_linear`). The deployed checkpoint stores weights bit-packed at ~1.6 bits/weight (5 ternary values per byte via base-3 encoding).

### KV-cache
Per-layer dicts `{"k": Tensor, "v": Tensor}` that concatenate past keys/values. Because activations are quantized per-token, a cached step is byte-for-byte identical to the full re-forward at that position. RoPE's `offset` parameter keeps absolute positions lining up across cached steps.

### Mesh vs MoE
`nn.MoE` routes within a layer (expert MLPs over the last dimension). `model.Mesh` routes between whole models (each expert is a full GPT with its own KV-cache). The router is a small ternary dense tree that reads a pooled token embedding and produces per-sequence mixing weights.
