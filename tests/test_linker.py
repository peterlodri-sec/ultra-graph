"""Tests for the .ugm model linker (P3)."""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from ultragraph.linker import (
    link_parallel,
    link_sequential,
)
from ultragraph.ugm import (
    UGMFile,
    UGMTree,
    UGMUltraEdge,
    load_ugm,
    save_ugm,
)


def _make_mod(in_dim: int, hid_dim: int, out_dim: int, name: str) -> UGMFile:
    """Create a 2-tree dense module."""
    t0 = UGMTree(kind=0, act=1, in_dim=in_dim, out_dim=hid_dim, name=f"{name}_0",
                 w_scale=1.0,
                 wq=np.random.randint(-1, 2, (hid_dim, in_dim)).astype(np.int8),
                 bias=np.zeros(hid_dim, dtype=np.float32))
    t1 = UGMTree(kind=0, act=0, in_dim=hid_dim, out_dim=out_dim, name=f"{name}_1",
                 w_scale=1.0,
                 wq=np.random.randint(-1, 2, (out_dim, hid_dim)).astype(np.int8),
                 bias=np.zeros(out_dim, dtype=np.float32))
    return UGMFile(trees=[t0, t1], ultra_edges=[UGMUltraEdge(src_idx=0, dst_idx=1, kind=0)])


def test_link_sequential():
    """Sequential link: output of mod_a → input of mod_b."""
    a = _make_mod(4, 8, 6, "a")
    b = _make_mod(6, 10, 2, "b")
    merged = link_sequential([a, b])
    assert len(merged.trees) == 4
    assert len(merged.ultra_edges) == 3  # 2 internal + 1 cross
    x = np.random.randn(1, 4).astype(np.float32)
    out = merged.run(x)
    assert out.shape == (1, 2)
    assert np.isfinite(out).all()


def test_link_sequential_roundtrip():
    """Sequential-linked module survives save/load."""
    a = _make_mod(4, 8, 6, "a")
    b = _make_mod(6, 10, 2, "b")
    merged = link_sequential([a, b])
    x = np.random.randn(1, 4).astype(np.float32)
    expected = merged.run(x)
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "linked.ugm"
        save_ugm(p, merged)
        loaded = load_ugm(str(p))
        result = loaded.run(x)
        assert np.allclose(expected, result, atol=1e-4)


def test_link_sequential_dim_mismatch():
    """Raises ValueError on incompatible dimensions."""
    a = _make_mod(4, 8, 6, "a")
    b = _make_mod(8, 10, 2, "b")  # b expects in_dim=8, a sink is 6
    import pytest
    with pytest.raises(ValueError, match="Dimension mismatch"):
        link_sequential([a, b])


def test_link_parallel_sum():
    """Parallel sum: outputs are elementwise summed."""
    a = _make_mod(4, 8, 8, "a")
    b = _make_mod(4, 8, 8, "b")
    merged = link_parallel([a, b], mode="sum")
    x = np.random.randn(1, 4).astype(np.float32)
    out = merged.run(x)
    expected = a.run(x) + b.run(x)
    assert np.allclose(out, expected, atol=1e-4)


def test_link_parallel_three_way():
    """Parallel sum with 3 modules."""
    a = _make_mod(4, 8, 8, "a")
    b = _make_mod(4, 8, 8, "b")
    c = _make_mod(4, 8, 8, "c")
    merged = link_parallel([a, b, c], mode="sum")
    x = np.random.randn(1, 4).astype(np.float32)
    out = merged.run(x)
    expected = a.run(x) + b.run(x) + c.run(x)
    assert np.allclose(out, expected, atol=1e-4)


def test_link_parallel_different_hidden():
    """Modules with different hidden dims but same output dim can be merged."""
    a = _make_mod(4, 8, 6, "a")
    b = _make_mod(4, 10, 6, "b")
    merged = link_parallel([a, b], mode="sum")
    x = np.random.randn(1, 4).astype(np.float32)
    out = merged.run(x)
    expected = a.run(x) + b.run(x)
    assert np.allclose(out, expected, atol=1e-4)


def test_link_parallel_output_dim_mismatch():
    """Raises ValueError when sink dimensions differ."""
    a = _make_mod(4, 8, 6, "a")
    b = _make_mod(4, 8, 8, "b")
    import pytest
    with pytest.raises(ValueError, match="sink out_dim"):
        link_parallel([a, b], mode="sum")


def test_cli_link_command():
    """`ug link` works via the CLI entry point."""
    import sys

    import ultragraph.cli

    a = _make_mod(4, 8, 6, "a")
    b = _make_mod(6, 10, 2, "b")

    with tempfile.TemporaryDirectory() as d:
        p_a = Path(d) / "a.ugm"
        p_b = Path(d) / "b.ugm"
        out = Path(d) / "out.ugm"
        save_ugm(p_a, a)
        save_ugm(p_b, b)

        sys.argv = ["ug", "link", str(p_a), str(p_b), "--output", str(out)]
        try:
            ultragraph.cli.main()
        except SystemExit:
            pass
        assert out.exists()
        loaded = load_ugm(str(out))
        x = np.random.randn(1, 4).astype(np.float32)
        result = loaded.run(x)
        assert result.shape == (1, 2)


def test_link_sequential_to_wasm():
    """Sequential-linked module can be compiled to WASM."""
    from ultragraph.wasm import generate_wat

    a = _make_mod(4, 8, 4, "a")
    b = _make_mod(4, 8, 4, "b")
    merged = link_sequential([a, b])
    wat = generate_wat(merged)
    assert "(func (export \"run\")" in wat
    assert wat.count("(") == wat.count(")")
