"""Tests for the .ugm binary format."""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from ultragraph.ugm import (
    UGMFile,
    UGMHistorySegment,
    UGMMetadata,
    UGMTree,
    UGMUltraEdge,
    from_ultragraph,
    load_ugm,
    save_ugm,
)


def _make_module() -> UGMFile:
    """A minimal 2-tree .ugm: 4→8→2, relu then none."""
    t0 = UGMTree(kind=0, act=1, in_dim=4, out_dim=8, name="linear0", w_scale=1.0,
                 wq=np.random.randint(-1, 2, (8, 4)).astype(np.int8),
                 bias=np.zeros(8, dtype=np.float32))
    t1 = UGMTree(kind=0, act=0, in_dim=8, out_dim=2, name="linear1", w_scale=1.0,
                 wq=np.random.randint(-1, 2, (2, 8)).astype(np.int8),
                 bias=np.zeros(2, dtype=np.float32))
    ue = UGMUltraEdge(src_idx=0, dst_idx=1, kind=0)
    return UGMFile(trees=[t0, t1], ultra_edges=[ue])


def test_save_load_roundtrip():
    module = _make_module()
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "test.ugm"
        save_ugm(path, module)
        assert path.exists()
        assert path.stat().st_size > 0

        loaded = load_ugm(str(path))
        assert len(loaded.trees) == 2
        assert len(loaded.ultra_edges) == 1
        assert loaded.trees[0].name == "linear0"
        assert loaded.trees[0].in_dim == 4
        assert loaded.trees[0].out_dim == 8
        assert loaded.trees[0].kind == 0
        assert loaded.ultra_edges[0].src_idx == 0
        assert loaded.ultra_edges[0].dst_idx == 1
        # Weight data roundtrips
        assert np.array_equal(loaded.trees[0].wq, module.trees[0].wq)
        assert np.array_equal(loaded.trees[1].wq, module.trees[1].wq)


def test_forward_pass():
    module = _make_module()
    x = np.random.randn(2, 4).astype(np.float32)
    out = module.run(x)
    assert out.shape == (2, 2)
    assert np.isfinite(out).all()


def test_sink_detection():
    module = _make_module()
    assert module.sink_idx == 1  # last tree, no outgoing edges


def test_optional_segments():
    t0 = _make_module().trees[0]
    module = UGMFile(
        trees=[t0],
        ultra_edges=[],
        history=UGMHistorySegment(n_nodes=1, depth=10,
                                  buffer=np.zeros((1, 10), dtype=np.int8)),
        metadata=UGMMetadata(data={"author": "test", "lang": "swahili"}),
    )
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "test.ugm"
        save_ugm(path, module)
        loaded = load_ugm(str(path))
        assert loaded.history is not None
        assert loaded.history.depth == 10
        assert loaded.metadata is not None
        assert loaded.metadata.data["author"] == "test"
        assert loaded.metadata.data["lang"] == "swahili"


def test_from_ultragraph_gpt():
    """Verify we can convert a GPT to .ugm and run it."""
    from ultragraph import GPT
    model = GPT(vocab=256, d_model=8, n_layers=2, n_heads=2, max_len=64)
    with tempfile.TemporaryDirectory() as d:
        ckpt = Path(d) / "model.q.npz"
        model.save_deployed(str(ckpt))
        m2 = GPT.load_deployed(str(ckpt))
        module = from_ultragraph(m2)
        assert len(module.trees) == 13  # 6 per block * 2 + 1 head
        # Run forward with norm-shaped input
        x = np.random.randn(1, 8).astype(np.float32)
        out = module.run(x)
        # Output should be [1, vocab=256] from the head tree
        assert out.shape == (1, 256)
        assert np.isfinite(out).all()
