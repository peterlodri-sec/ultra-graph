# Model Card

Generate a markdown model card report from a trained checkpoint.

## What Success Looks Like

A markdown document at `skills/reports/{model-name}-model-card.md` containing: architecture summary, hyperparameters, training stats, weight statistics, sample output, and deployment info.

## Sections to include

### Architecture
- Model type (GPT, Mesh, MLP, UltraGraph)
- Hyperparameters: vocab, d_model, n_layers, n_heads, max_len, mlp_ratio
- Total parameter count (`model.n_params()`)
- Deployment size (size of `.q.npz` on disk)

### Training
- Loss curve: initial loss, final loss, best loss
- Training steps completed
- Optimizer and schedule used
- Corpus size (bytes, tokens)

### Weight statistics
For each dense layer:
- Sparsity (% of zero weights)
- Positive/negative ratio
- Weight scale factors

### Sample output
- 2-3 generated samples at different temperatures
- Greedy (temp=0) and creative (temp=0.8) examples

### Deployment
- Checkpoint file size and bit density
- `GPT.load_deployed()` usage example
- Byte-exact verification

## Usage

```python
from ultragraph import GPT, ByteTokenizer

model = GPT.load_deployed("model.q.npz")
tok = ByteTokenizer()

# Collect stats
n_params = model.n_params()
sample = tok.decode(model.generate(tok.encode("The "), n_new=64, temperature=0.8))

# Write report to skills/reports/
```
