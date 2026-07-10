import numpy as np

from ultragraph.autograd import Tensor
from ultragraph.core import Embedding, Tree, UltraGraph
from ultragraph.nn import mlp
from ultragraph.optim import SGD


def test_mlp_overfits_toy_classification():
    np.random.seed(0)
    n = 32
    X = np.random.randn(n, 4).astype(np.float32)
    y = (X[:, 0] > 0).astype(np.int64)  # trivially separable, 2 classes

    ug = mlp([4, 16, 2])
    opt = SGD(ug, lr=0.3, momentum=0.9)

    losses = []
    xt = Tensor(X)
    for _ in range(300):
        logits = ug.forward(xt)
        loss = logits.cross_entropy(y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss.data))

    assert min(losses[-10:]) < 0.5 * losses[0], (losses[0], min(losses[-10:]))


def test_two_trees_residual_forward():
    np.random.seed(1)
    ug = UltraGraph("res")
    a = ug.add(Tree.dense(4, 4, "a", act="relu"))
    b = ug.add(Tree.dense(4, 4, "b", act="none"))
    a >> b  # plain: b's input is a's output
    ug.wire(a, b, "residual")  # residual: add a's output to b's output
    x = Tensor(np.random.randn(3, 4).astype(np.float32))
    out = ug.forward(x)
    assert out.shape == (3, 4)
    assert np.isfinite(out.data).all()


def test_tiny_char_lm_trains_and_samples():
    np.random.seed(0)
    text = "hello ultra graph world "
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}
    vocab = len(chars)
    dim = 8

    ids = np.array([stoi[c] for c in text], dtype=np.int64)
    inputs = ids[:-1]
    targets = ids[1:]

    emb = Embedding(vocab, dim, "emb")
    net = mlp([dim, 16, vocab])
    net.register(emb)
    opt = SGD(net, lr=0.1, momentum=0.9)

    losses = []
    for _ in range(400):
        e = emb(inputs)
        logits = net.forward(e)
        loss = logits.cross_entropy(targets)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss.data))

    assert min(losses[-10:]) < 0.7 * losses[0], (losses[0], min(losses[-10:]))

    # sampling runs and returns the requested length
    cur = stoi["h"]
    out = "h"
    for _ in range(10):
        e = emb(np.array([cur]))
        logits = net.forward(e)
        cur = int(np.argmax(logits.data[0]))
        out += itos[cur]
    assert len(out) == 11
