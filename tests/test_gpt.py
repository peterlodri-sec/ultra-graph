import os
import tempfile

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


def test_gpt_save_load_roundtrip():
    m = _tiny_gpt()
    ids = np.array([[2, 5, 7, 1, 3]], dtype=np.int64)
    before = m(ids).data
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "gpt.npz")
        m.save(p)
        m2 = GPT(vocab=16, d_model=16, n_layers=2, n_heads=2, max_len=32)  # fresh weights
        assert not np.allclose(m2(ids).data, before)                       # differ before load
        m2.load(p)
        assert np.allclose(m2(ids).data, before, atol=1e-5)                # identical after


def test_load_model_reconstructs_nondefault_arch():
    # load_model must rebuild ANY architecture from embedded hp — not the old
    # hardcoded GPT(256, 128, 2, 2). This shape differs in every hyper-parameter.
    m = GPT(vocab=16, d_model=16, n_layers=1, n_heads=4, max_len=16)
    ids = np.array([[2, 5, 7, 1, 3]], dtype=np.int64)
    before = m(ids).data
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "gpt.npz")
        m.save(p)
        m2 = GPT.load_model(p)                       # no pre-built shape supplied
        assert (m2.vocab, m2.d_model, m2.n_layers, m2.n_heads) == (16, 16, 1, 4)
        assert np.allclose(m2(ids).data, before, atol=1e-5)


def test_load_model_auto_detects_deployed():
    m = _tiny_gpt()
    ids = np.array([[2, 5, 7, 1, 3]], dtype=np.int64)
    ref = m(ids).data
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "dep.npz")
        m.save_deployed(p)
        dm = GPT.load_model(p)                        # auto-detect -> deployed path
        assert all(t.deployed for t in dm._dense_trees())
        assert np.allclose(dm(ids).data, ref, atol=1e-5)


def test_load_model_legacy_without_meta_raises():
    # A checkpoint with no architecture metadata cannot be reshaped safely
    # (n_heads/max_len are not recoverable) -> loud error, never silent mis-shape.
    import pytest

    from ultragraph.io import save_params
    m = _tiny_gpt()
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "legacy.npz")
        save_params([m], p)                          # no meta= -> legacy layout
        with pytest.raises(ValueError, match="no architecture metadata"):
            GPT.load_model(p)


def test_gpt_deployed_is_byte_exact_and_smaller():
    m = _tiny_gpt()
    ids = np.array([[2, 5, 7, 1, 3]], dtype=np.int64)
    ref = m(ids).data
    with tempfile.TemporaryDirectory() as d:
        full = os.path.join(d, "full.npz")
        dep = os.path.join(d, "dep.npz")
        m.save(full)                      # fp32 masters
        m.save_deployed(dep, packed=True)  # ternary bytes, packed
        dm = GPT.load_deployed(dep)
        assert all(t.deployed for t in dm._dense_trees())   # no masters -> deployed path
        assert np.allclose(dm(ids).data, ref, atol=1e-5)    # byte-exact vs trained
        assert os.path.getsize(dep) < os.path.getsize(full) * 0.5  # much smaller


def test_gpt_deployed_generation_matches():
    m = _tiny_gpt()
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "dep.npz")
        m.save_deployed(p)
        dm = GPT.load_deployed(p)
        assert dm.generate([1, 2, 3], n_new=8, temperature=0.0) == \
            m.generate([1, 2, 3], n_new=8, temperature=0.0)


def test_gpt_generate_batch_matches_single():
    m = _tiny_gpt()
    prompts = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64)
    batch = m.generate_batch(prompts, n_new=5, temperature=0.0)
    assert len(batch) == 2 and all(len(r) == 8 for r in batch)
    # each row is independent -> matches single-sequence greedy on that prompt
    assert batch[0] == m.generate([1, 2, 3], n_new=5, temperature=0.0)
    assert batch[1] == m.generate([4, 5, 6], n_new=5, temperature=0.0)


def test_gpt_generate_batch_stop_per_sequence():
    m = _tiny_gpt()
    full = m.generate([1, 2, 3], n_new=6, temperature=0.0)
    sid = full[3]
    batch = m.generate_batch(np.array([[1, 2, 3], [1, 2, 3]]), n_new=6, temperature=0.0, stop=sid)
    assert all(r[-1] == sid and len(r) == 4 for r in batch)   # both halt on the stop token


def test_gpt_stream_matches_batch():
    m = _tiny_gpt()
    ref = m.generate([4, 2, 1], n_new=6, temperature=0.0)
    streamed = list(m.generate([4, 2, 1], n_new=6, temperature=0.0, stream=True))
    assert streamed == ref[3:]          # stream yields only the new tokens
    assert len(streamed) == 6


def test_gpt_top_p_sampling_runs_and_is_seeded():
    m = _tiny_gpt()
    a = m.generate([1, 2], n_new=8, temperature=1.0, top_p=0.9, seed=0)
    b = m.generate([1, 2], n_new=8, temperature=1.0, top_p=0.9, seed=0)
    assert a == b and len(a) == 10      # same seed -> deterministic
    assert all(0 <= t < 16 for t in a)


def test_gpt_stop_token_halts():
    m = _tiny_gpt()
    # find what greedy would produce, then use its first new token as a stop id
    full = m.generate([1, 2, 3], n_new=6, temperature=0.0)
    stop_id = full[3]
    stopped = m.generate([1, 2, 3], n_new=6, temperature=0.0, stop=stop_id)
    assert stopped[-1] == stop_id            # halts right after emitting the stop token
    assert len(stopped) == 4                 # prompt(3) + 1 generated (the stop)


def test_gpt_repetition_penalty_changes_output():
    m = _tiny_gpt()
    base = m.generate([5, 5, 5], n_new=10, temperature=0.7, seed=1)
    pen = m.generate([5, 5, 5], n_new=10, temperature=0.7, seed=1, repetition_penalty=2.0)
    assert base != pen                       # penalty perturbs the distribution
    assert len(pen) == 13


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
