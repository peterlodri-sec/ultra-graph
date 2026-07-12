"""A 1-bit Hungarian-language LLM — a ternary GPT trained on Hungarian literature.

Unlike `anonymus_lm.py` (medieval Latin), this trains on a genuinely **Hungarian**
public-domain corpus (Arany János and 19th-c. prose, via `fetch_hungarian.py`). The
byte tokenizer handles the accents (á é í ó ö ő ú ü ű) losslessly. Weights are ternary
{-1,0,+1}; the trained model deploys to a bit-packed checkpoint.

    uv run python examples/fetch_hungarian.py       # build the corpus first
    uv run python examples/hungarian_lm.py          # ~3000 steps
    STEPS=6000 uv run python examples/hungarian_lm.py
"""

import json
import os

import numpy as np

from ultragraph import GPT, Adam, ByteTokenizer, CosineSchedule

_D = os.path.join(os.path.dirname(__file__), "data")
DATA = os.path.join(_D, "hungarian_corpus.txt")
CKPT = os.environ.get("HU_CKPT", os.path.join(_D, "hungarian.gpt.npz"))            # deployed ternary
TRAIN_CKPT = os.environ.get("HU_TRAIN_CKPT", os.path.join(_D, "hungarian.train.npz"))  # fp32 masters (resume)
STATE = os.environ.get("HU_STATE", os.path.join(_D, "hungarian.train.json"))       # {steps_done}
SEEDS = ["A magyar ", "Egyszer volt, hol nem volt, ", "Az ember ", "Buda "]


def sample(model, tok, seed, n=200, temperature=0.8, top_p=0.9, rng=0):
    out = model.generate(tok.encode(seed), n_new=n, temperature=temperature,
                         top_p=top_p, repetition_penalty=1.2, seed=rng)
    return tok.decode(out)


def _save_all(model, step):
    model.save(TRAIN_CKPT)        # fp32 masters -> resumable
    model.save_deployed(CKPT)     # ternary bytes -> deployable
    with open(STATE, "w") as f:
        json.dump({"steps_done": step}, f)


def main() -> None:
    np.random.seed(0)
    total = int(os.environ.get("TOTAL", "6000"))   # target step count across all segments
    seg = int(os.environ.get("STEPS", "2200"))     # steps to run this segment
    T, batch = 96, 16

    tok = ByteTokenizer()
    with open(DATA, encoding="utf-8") as f:
        text = f.read()
    data = tok.encode(text)
    print(f"corpus: {len(text):,} chars / {len(data):,} bytes (Hungarian)")

    model = GPT(vocab=256, d_model=128, n_layers=4, n_heads=4, max_len=512)
    done = 0
    if os.path.exists(TRAIN_CKPT) and os.path.exists(STATE):
        model.load(TRAIN_CKPT)                     # resume: warm-start the fp32 masters
        done = int(json.load(open(STATE)).get("steps_done", 0))
        print(f"resumed from step {done}/{total}")
    print(f"model: {model.n_params():,} trainable params")

    opt = Adam(model, lr=3e-3, clip=1.0)
    sched = CosineSchedule(opt, total_steps=total, warmup=total // 20)
    sched.t = done                                 # continue the LR schedule across segments

    target = min(done + seg, total)
    rng = np.random.default_rng(done)              # vary the data order per segment
    for step in range(done + 1, target + 1):
        starts = rng.integers(0, len(data) - T - 1, size=batch)
        xb = np.stack([data[s : s + T] for s in starts])
        yb = np.stack([data[s + 1 : s + T + 1] for s in starts])
        loss = model(xb).cross_entropy(yb)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if step % 100 == 0 or step == done + 1:
            print(f"step {step:5d}/{total}  loss {float(loss.data):.4f}  lr {opt.lr:.5f}")
        if step % 500 == 0:
            print("  sample:", repr(sample(model, tok, "A magyar ", n=120))[:230])
            _save_all(model, step)
            print(f"  (checkpoint saved @ step {step})")

    _save_all(model, target)
    print(f"\nsegment done: {target}/{total} steps  ->  {CKPT} ({os.path.getsize(CKPT)/1024:.0f} KB)")
    if target < total:
        print(f"run again to continue — resumes automatically from step {target}")
    for seed in SEEDS:
        print(f"\n=== {seed!r} ===")
        print(sample(model, tok, seed, n=280, temperature=0.8, top_p=0.9))


if __name__ == "__main__":
    main()
