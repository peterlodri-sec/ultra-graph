import os
import tempfile

import numpy as np

from ultragraph import Tensor, mlp, pack_ternary, unpack_ternary
from ultragraph.io import load, save


def test_pack_roundtrip_various_sizes():
    rng = np.random.RandomState(0)
    for n in [0, 1, 4, 5, 6, 17, 100, 1234]:
        q = rng.choice([-1, 0, 1], size=n).astype(np.int8)
        packed = pack_ternary(q)
        assert packed.dtype == np.uint8
        assert len(packed) == (n + 4) // 5  # ceil(n/5), 5 values per byte
        assert np.array_equal(unpack_ternary(packed, n), q)


def test_pack_2d_and_smaller():
    rng = np.random.RandomState(1)
    q = rng.choice([-1, 0, 1], size=(7, 9)).astype(np.int8)
    packed = pack_ternary(q)
    back = unpack_ternary(packed, q.size).reshape(q.shape)
    assert np.array_equal(back, q)
    assert len(packed) == (q.size + 4) // 5  # ~5x fewer bytes than raw int8


def test_pack_rejects_nonternary():
    try:
        pack_ternary(np.array([2], dtype=np.int8))
        raised = False
    except ValueError:
        raised = True
    assert raised


def test_io_packed_roundtrip_byte_exact_and_smaller():
    np.random.seed(0)
    ug = mlp([64, 128, 32])  # large enough that packing savings dominate .npz overhead
    ug.forward(Tensor(np.random.randn(2, 64).astype(np.float32)))
    with tempfile.TemporaryDirectory() as d:
        raw_path = os.path.join(d, "raw.ug")
        packed_path = os.path.join(d, "packed.ug")
        save(ug, raw_path, include_masters=False)
        save(ug, packed_path, include_masters=False, packed=True)
        raw = load(raw_path)
        packed = load(packed_path)
        for t_raw, t_packed in zip(raw.trees, packed.trees):
            assert np.array_equal(t_raw.wq, t_packed.wq)  # exact ternary weights recovered
        assert os.path.getsize(packed_path) < os.path.getsize(raw_path)
