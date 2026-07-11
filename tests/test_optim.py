import numpy as np

from ultragraph import GPT, SGD


def _model(seed=0):
    np.random.seed(seed)
    return GPT(vocab=16, d_model=16, n_layers=1, n_heads=2, max_len=32)


def test_grad_accum_equals_full_batch_step():
    # Averaging grads over two equal micro-batches, then one SGD update, must equal a
    # single full-batch update (CE is a mean over tokens; halves are equal size).
    seq = np.array([[2, 7, 1, 8, 2, 8, 1, 8]], dtype=np.int64)
    # widen to a batch of 4 so we can split into two equal halves
    batch = np.repeat(seq, 4, axis=0)
    x, y = batch[:, :-1], batch[:, 1:]

    full = _model(0)
    ofull = SGD(full, lr=0.2)  # no momentum -> update is a pure function of the grad
    ofull.zero_grad()
    full(x).cross_entropy(y).backward()
    ofull.step()

    acc = _model(0)
    oacc = SGD(acc, lr=0.2, accum_steps=2)
    for half in (slice(0, 2), slice(2, 4)):
        acc(x[half]).cross_entropy(y[half]).backward()
        oacc.step()  # applies (averaged) + zeros on the 2nd call

    fp = {id(p): p for p in full.parameters()}
    for pa, pf in zip(acc.parameters(), full.parameters()):
        assert np.allclose(pa.data, pf.data, atol=1e-5)
    assert fp  # sanity


def test_accum_defers_update_until_window_closes():
    m = _model(0)
    opt = SGD(m, lr=0.5, accum_steps=3)
    before = m.head.adhoc["w_master"].data.copy()
    x = np.array([[1, 2, 3, 4]], dtype=np.int64)
    for i in range(2):  # only two of three micro-steps
        opt.zero_grad() if i == 0 else None
        m(x[:, :-1]).cross_entropy(x[:, 1:]).backward()
        opt.step()
    assert np.allclose(m.head.adhoc["w_master"].data, before)  # no update yet
    m(x[:, :-1]).cross_entropy(x[:, 1:]).backward()
    opt.step()  # third -> applies
    assert not np.allclose(m.head.adhoc["w_master"].data, before)
