import numpy as np

from ultragraph import (
    Adam,
    Embedding,
    LayerNorm,
    MultiHeadAttention,
    RMSNorm,
    Tensor,
    UltraGraph,
    linear_tree,
    mlp,
)


def test_mha_shape_batched_and_2d():
    np.random.seed(0)
    mha = MultiHeadAttention(16, n_heads=4, causal=True)
    x3 = Tensor(np.random.randn(2, 5, 16).astype(np.float32))
    out3 = mha(x3)
    assert out3.shape == (2, 5, 16) and np.isfinite(out3.data).all()
    x2 = Tensor(np.random.randn(5, 16).astype(np.float32))
    assert mha(x2).shape == (5, 16)


def test_mha_heads_must_divide():
    try:
        MultiHeadAttention(16, n_heads=5)
        raised = False
    except AssertionError:
        raised = True
    assert raised


def test_mha_causality_and_batch_independence():
    np.random.seed(1)
    mha = MultiHeadAttention(16, n_heads=4, causal=True)
    X = np.random.randn(2, 6, 16).astype(np.float32)
    out1 = mha(Tensor(X.copy())).data.copy()
    X2 = X.copy()
    X2[1, 3:] += 3.0  # perturb batch 1, positions >=3 only
    out2 = mha(Tensor(X2)).data
    # batch 0 fully unchanged (batch independence)
    assert np.allclose(out1[0], out2[0], atol=1e-4)
    # batch 1 positions < 3 unchanged (causality)
    assert np.allclose(out1[1, :3], out2[1, :3], atol=1e-4)
    # a perturbed position DID change (sanity)
    assert not np.allclose(out1[1, 5], out2[1, 5], atol=1e-4)


def test_rmsnorm_property_and_grad():
    np.random.seed(2)
    x = Tensor((np.random.randn(4, 8) * 5).astype(np.float32))
    norm = RMSNorm(8)
    out = norm(x)
    rms = np.sqrt((out.data ** 2).mean(axis=-1))
    assert np.allclose(rms, 1.0, atol=1e-2)
    out.sum().backward()
    assert np.abs(norm.gain.grad).sum() > 0


def test_layernorm_property_and_grad():
    np.random.seed(3)
    x = Tensor((np.random.randn(4, 8) * 5 + 2).astype(np.float32))
    norm = LayerNorm(8)
    out = norm(x)
    assert np.allclose(out.data.mean(axis=-1), 0.0, atol=1e-4)
    assert np.allclose(out.data.var(axis=-1), 1.0, atol=1e-2)
    out.sum().backward()
    assert np.abs(norm.gain.grad).sum() > 0
    assert np.abs(norm.bias.grad).sum() > 0


def test_positional_embedding_shape_and_grad():
    from ultragraph import LearnedPositionalEmbedding

    np.random.seed(0)
    pos = LearnedPositionalEmbedding(16, 8)
    out = pos(Tensor(np.zeros((2, 5, 8), dtype=np.float32)))
    assert out.shape == (2, 5, 8)
    # zero input -> output is the (position-dependent) table, so positions differ
    assert not np.allclose(out.data[:, 0], out.data[:, 1])
    out.sum().backward()
    assert np.abs(pos.table.grad[:5]).sum() > 0     # used positions get gradient
    assert np.all(pos.table.grad[5:] == 0)          # unused positions do not


def test_moe_shape_and_gradients():
    from ultragraph import MoE

    np.random.seed(0)
    moe = MoE(8, n_experts=3, hidden=16)
    x = Tensor(np.random.randn(2, 5, 8).astype(np.float32))
    out = moe(x)
    assert out.shape == (2, 5, 8) and np.isfinite(out.data).all()
    out.sum().backward()
    # gradient reaches both the router and the experts
    assert np.abs(moe.router.parameters()[0].grad).sum() > 0
    assert np.abs(moe.experts_in[0].parameters()[0].grad).sum() > 0
    assert np.abs(moe.experts_out[1].parameters()[0].grad).sum() > 0


def test_adam_overfits_toy_classification():
    np.random.seed(0)
    n = 32
    X = np.random.randn(n, 4).astype(np.float32)
    y = (X[:, 0] > 0).astype(np.int64)
    ug = mlp([4, 16, 2])
    opt = Adam(ug, lr=0.02, clip=1.0)
    losses = []
    xt = Tensor(X)
    for _ in range(300):
        loss = ug.forward(xt).cross_entropy(y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss.data))
    assert min(losses[-10:]) < 0.5 * losses[0], (losses[0], min(losses[-10:]))


def test_mini_transformer_batched_trains():
    """Batched pre-norm transformer block (RMSNorm + multi-head attn + MLP) trains."""
    np.random.seed(0)
    text = "hello ultra graph world " * 3
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    vocab = len(chars)
    d_model, n_heads, T = 16, 4, 8

    ids = np.array([stoi[c] for c in text], dtype=np.int64)
    seqs = [ids[i : i + T + 1] for i in range(len(ids) - T)]
    seqs = np.stack(seqs)  # [B, T+1]
    inputs, targets = seqs[:, :-1], seqs[:, 1:]  # [B, T]
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
        x = emb(idx.reshape(-1)).reshape(B, T, d_model)
        x = x + mha(n1(x))
        x = x + ff2.forward(ff1.forward(n2(x)))
        return unembed.forward(x)  # [B, T, vocab]

    losses = []
    for _ in range(200):
        logits = forward(inputs)
        loss = logits.cross_entropy(targets)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss.data))
    assert min(losses[-10:]) < 0.6 * losses[0], (losses[0], min(losses[-10:]))
