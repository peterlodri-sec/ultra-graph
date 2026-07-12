# Train LM

Train or resume-train a byte-level ternary GPT language model on a corpus.

## What Success Looks Like

A trained model checkpoint (`.npz` with fp32 masters) that shows a decreasing loss curve and produces recognizable text in the target language when sampled.

## Architecture choices

| Hyperparameter | Typical range | Notes |
|---------------|---------------|-------|
| `d_model` | 64-256 | Larger = more capacity, slower |
| `n_layers` | 2-6 | Deeper = more abstraction |
| `n_heads` | 2-8 | Must divide d_model; d_head must be even for RoPE |
| `max_len` | 128-512 | Context window |
| `mlp_ratio` | 4 | Hidden dim = d_model * ratio |

## Training loop (from examples)

```python
from ultragraph import GPT, ByteTokenizer, Adam, CosineSchedule

tok = ByteTokenizer()
model = GPT(vocab=256, d_model=128, n_layers=4, n_heads=4, max_len=256)
opt = Adam(model, lr=3e-4)
sched = CosineSchedule(opt, total_steps=10_000, warmup=500)

ids = tok.encode(corpus_text)
for step in range(10_000):
    # chunk, forward, backward, step, schedule
```

## Resume training pattern

Training can hit wall-clock caps. The save/load pattern handles this:

```python
import os
model = GPT(vocab=256, d_model=128, n_layers=4, n_heads=4)
total = int(os.environ.get("TOTAL", "10000"))
steps_file = "checkpoint.npz"
if os.path.exists(steps_file):
    model.load(steps_file)  # re-quantizes, continues
```

See `examples/hungarian_lm.py` for the full resume-training pattern with `TOTAL`/`STEPS` env vars and periodic save.

## Existing examples to reference

- `examples/char_lm.py` — simplest: char-level MLP LM
- `examples/gpt_lm.py` — full ByteTokenizer → GPT → train → stream
- `examples/hungarian_lm.py` — resumable training with periodic checkpoints
- `examples/anonymus_lm.py` — Latin chronicle GPT
