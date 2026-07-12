"""Generate themed illustrations for the README from a REAL trained model.

Palette matches pocoo.vaked.dev: deep navy background with cyan / green / purple
accents. Figures are produced from an actual mini-GPT forward pass (ternary
weights, causal attention matrix) plus an architecture diagram.

Run: uv run python assets/make_figures.py
"""

import math
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from ultragraph import Adam, Embedding, MultiHeadAttention, RMSNorm, Tensor, UltraGraph, linear_tree

BG = "#070b16"
PANEL = "#0b1120"
CYAN = "#00d4ff"
GREEN = "#00e660"
PURPLE = "#b48bff"
FG = "#c8d3e6"
MUTED = "#4a5a80"

OUT = os.path.join(os.path.dirname(__file__))

# purple -> navy -> cyan, for ternary weights in {-1, 0, +1}
TERNARY_CMAP = LinearSegmentedColormap.from_list("ug_ternary", [PURPLE, "#0e1730", CYAN])
# navy -> cyan, sequential, for attention weights in [0, 1]
ATTN_CMAP = LinearSegmentedColormap.from_list("ug_attn", ["#0a1024", "#0f5c78", CYAN])


def _style(ax):
    ax.set_facecolor(BG)
    for s in ax.spines.values():
        s.set_color(MUTED)
    ax.tick_params(colors=FG, labelsize=8)
    ax.xaxis.label.set_color(FG)
    ax.yaxis.label.set_color(FG)
    ax.title.set_color(FG)


def _save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print("wrote", path)
    return path


def train_model():
    np.random.seed(0)
    text = "hello ultra graph world " * 4
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    vocab = len(chars)
    d_model, n_heads, T = 24, 4, 8
    ids = np.array([stoi[c] for c in text], dtype=np.int64)
    seqs = np.stack([ids[i : i + T + 1] for i in range(len(ids) - T)])
    inputs, targets = seqs[:, :-1], seqs[:, 1:]
    B = inputs.shape[0]

    emb = Embedding(vocab, d_model, "emb")
    n1 = RMSNorm(d_model, name="n1")
    mha = MultiHeadAttention(d_model, n_heads, causal=True)
    n2 = RMSNorm(d_model, name="n2")
    ff1 = linear_tree(d_model, 4 * d_model, "ff1", act="relu")
    ff2 = linear_tree(4 * d_model, d_model, "ff2", act="none")
    unembed = linear_tree(d_model, vocab, "unembed", act="none")
    model = UltraGraph("mini_gpt")
    for m in (emb, n1, mha, n2):
        model.register(m)
    for t in (ff1, ff2, unembed):
        model.add(t)
    opt = Adam(model, lr=0.02, clip=1.0)

    def forward(idx):
        x = emb(idx.reshape(-1)).reshape(B, idx.shape[1], d_model)
        x = x + mha(n1(x))
        x = x + ff2.forward(ff1.forward(n2(x)))
        return unembed.forward(x)

    for _ in range(300):
        loss = forward(inputs).cross_entropy(targets)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return dict(mha=mha, n1=n1, emb=emb, inputs=inputs, T=T, d_model=d_model, n_heads=n_heads, B=B)


def fig_ternary_weights(m):
    """Real ternary weight matrix from the trained attention query projection."""
    wq = m["mha"].wq.wq  # int8 {-1,0,1}, shape [d_model, d_model]
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    fig.patch.set_facecolor(BG)
    im = ax.imshow(wq, cmap=TERNARY_CMAP, vmin=-1, vmax=1, interpolation="nearest")
    _style(ax)
    ax.set_title("ternary weight bytes  ·  W_q  ∈ {-1, 0, +1}", fontsize=10)
    ax.set_xlabel("in")
    ax.set_ylabel("out")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, ticks=[-1, 0, 1])
    cb.ax.yaxis.set_tick_params(color=FG)
    plt.setp(plt.getp(cb.ax.axes, "yticklabels"), color=FG)
    return _save(fig, "fig_ternary_weights.png")


def fig_attention(m):
    """Real causal attention weights (head 0) from a forward pass."""
    mha, n1, emb = m["mha"], m["n1"], m["emb"]
    T, d, H = m["T"], m["d_model"], m["n_heads"]
    dh = d // H
    x = emb(m["inputs"][0]).reshape(1, T, d)
    xn = n1(x)
    q = mha.wq.forward(xn).reshape(1, T, H, dh).swapaxes(1, 2)
    k = mha.wk.forward(xn).reshape(1, T, H, dh).swapaxes(1, 2)
    scores = (q @ k.swapaxes(-1, -2)) * (1.0 / math.sqrt(dh))
    mask = np.triu(np.full((T, T), -1e9, dtype=np.float32), k=1)
    attn = (scores + Tensor(mask)).softmax(axis=-1).data[0, 0]  # [T, T]

    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    fig.patch.set_facecolor(BG)
    im = ax.imshow(attn, cmap=ATTN_CMAP, vmin=0, vmax=1, interpolation="nearest")
    _style(ax)
    ax.set_title("causal self-attention  ·  softmax(QKᵀ/√d)", fontsize=10)
    ax.set_xlabel("key position")
    ax.set_ylabel("query position")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.yaxis.set_tick_params(color=FG)
    plt.setp(plt.getp(cb.ax.axes, "yticklabels"), color=FG)
    return _save(fig, "fig_attention.png")


def fig_architecture():
    """Ultra-graph view of the mini-GPT block: trees wired by ultra-edges."""
    fig, ax = plt.subplots(figsize=(5.2, 6.4))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 13)
    ax.axis("off")

    def box(y, label, color, sub=""):
        b = FancyBboxPatch((2.6, y), 4.8, 1.0, boxstyle="round,pad=0.08,rounding_size=0.18",
                           linewidth=1.6, edgecolor=color, facecolor=PANEL)
        ax.add_patch(b)
        ax.text(5.0, y + 0.62, label, ha="center", va="center", color=FG, fontsize=10, fontweight="bold")
        if sub:
            ax.text(5.0, y + 0.28, sub, ha="center", va="center", color=MUTED, fontsize=7.5)
        return y

    layers = [
        (11.4, "Embedding", GREEN, "vocab → d_model"),
        (9.6, "RMSNorm", PURPLE, ""),
        (8.2, "MultiHeadAttention", CYAN, "ternary Q·K·V·O, causal"),
        (6.4, "RMSNorm", PURPLE, ""),
        (5.0, "MLP (ternary)", CYAN, "ff1 relu · ff2"),
        (3.0, "Unembed", GREEN, "d_model → vocab"),
    ]
    for y, lab, col, sub in layers:
        box(y, lab, col, sub)

    def arrow(y0, y1, color=FG, curve=0.0, lw=1.4):
        a = FancyArrowPatch((5.0, y0), (5.0, y1), connectionstyle=f"arc3,rad={curve}",
                            arrowstyle="-|>", mutation_scale=12, color=color, lw=lw)
        ax.add_patch(a)

    arrow(11.4, 10.6)      # emb -> norm
    arrow(9.6, 9.2)        # norm -> attn
    arrow(8.2, 7.4)        # attn -> norm2
    arrow(6.4, 6.0)        # norm2 -> mlp
    arrow(5.0, 4.0)        # mlp -> unembed
    # residual ultra-edges (curved, on the right)
    for y0, y1, c in [(11.4, 8.2, CYAN), (8.2, 5.0, CYAN)]:
        a = FancyArrowPatch((7.4, y0 + 0.5), (7.4, y1 + 0.5), connectionstyle="arc3,rad=-0.5",
                            arrowstyle="-|>", mutation_scale=11, color=c, lw=1.2, linestyle=(0, (4, 2)))
        ax.add_patch(a)
        ax.text(9.0, (y0 + y1) / 2 + 0.5, "===", ha="center", va="center", color=c, fontsize=9, rotation=90)

    ax.text(5.0, 12.5, "ultra-graph  ·  trees wired by ultra-edges (===)", ha="center",
            color=FG, fontsize=11, fontweight="bold")
    ax.text(5.0, 2.2, "every weight = 1 ternary byte  ·  every activation = 1 int8 byte",
            ha="center", color=MUTED, fontsize=8)
    return _save(fig, "fig_architecture.png")


def main():
    m = train_model()
    fig_ternary_weights(m)
    fig_attention(m)
    fig_architecture()


if __name__ == "__main__":
    main()
