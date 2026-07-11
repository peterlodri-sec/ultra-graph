import numpy as np

from ultragraph import GPT, SGD


def _tiny_gpt():
    np.random.seed(0)
    return GPT(vocab=16, d_model=16, n_layers=2, n_heads=2, max_len=32)


def test_gpt_forward_shape():
    m = _tiny_gpt()
    ids = np.array([[1, 2, 3, 4]], dtype=np.int64)
    logits = m(ids)
    assert logits.shape == (1, 4, 16)


def test_gpt_kv_cache_matches_full_forward():
    # Per-token activation quant => a cached incremental decode is identical to
    # the full parallel forward at every position (fp32 rounding tolerance).
    m = _tiny_gpt()
    seq = [3, 1, 4, 1, 5, 9, 2]
    full = m(np.array([seq], dtype=np.int64)).data[0]        # [T, vocab]

    caches = [{"k": None, "v": None} for _ in m.blocks]
    got = []
    for t in seq:
        step = m._decode(np.array([[t]], dtype=np.int64), caches)  # [1,1,vocab]
        got.append(step[0, -1])
    got = np.stack(got)
    assert np.allclose(full, got, atol=1e-3)


def test_gpt_generate_length_and_greedy_determinism():
    m = _tiny_gpt()
    out = m.generate([1, 2, 3], n_new=5, temperature=0.0)
    assert len(out) == 8 and out[:3] == [1, 2, 3]
    # greedy is deterministic
    assert m.generate([1, 2, 3], n_new=5, temperature=0.0) == out


def test_gpt_overfits_toy_sequence():
    m = _tiny_gpt()
    opt = SGD(m, lr=0.5, momentum=0.9, clip=1.0)
    seq = np.array([[2, 7, 1, 8, 2, 8, 1, 8]], dtype=np.int64)
    x, y = seq[:, :-1], seq[:, 1:]
    first = None
    for _ in range(120):
        loss = m(x).cross_entropy(y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if first is None:
            first = float(loss.data)
    assert float(loss.data) < first * 0.5   # loss at least halves
