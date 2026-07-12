"""Tests for RP2040 C code generation and history segment."""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from ultragraph.rp2040 import generate_c, save_rp2040
from ultragraph.ugm import (
    UGMFile,
    UGMTree,
    UGMUltraEdge,
    load_ugm,
    save_ugm,
)


def _make_module() -> UGMFile:
    t0 = UGMTree(kind=0, act=1, in_dim=4, out_dim=8, name="linear0", w_scale=1.0,
                 wq=np.random.randint(-1, 2, (8, 4)).astype(np.int8),
                 bias=np.zeros(8, dtype=np.float32))
    t1 = UGMTree(kind=0, act=0, in_dim=8, out_dim=2, name="linear1", w_scale=1.0,
                 wq=np.random.randint(-1, 2, (2, 8)).astype(np.int8),
                 bias=np.zeros(2, dtype=np.float32))
    return UGMFile(trees=[t0, t1], ultra_edges=[UGMUltraEdge(src_idx=0, dst_idx=1, kind=0)])


def test_rp2040_generates_valid_c():
    module = _make_module()
    c_code = generate_c(module)
    assert "void model_forward" in c_code
    assert "int8_t t0_wq" in c_code
    assert "int8_t t1_wq" in c_code
    assert "float t0_bias" in c_code
    assert "tree0_forward" in c_code
    assert "tree1_forward" in c_code
    assert "#ifdef RP2040" in c_code


def test_rp2040_save_writes_file():
    module = _make_module()
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "model.c"
        save_rp2040(module, path)
        assert path.exists()
        content = path.read_text()
        assert "model_forward" in content
        assert "int8_t" in content


def test_rp2040_activations():
    module = _make_module()
    c_code = generate_c(module)
    # relu activation (act=1) should produce max(0, x) check
    assert "if (out[o] < 0.0f)" in c_code


def test_cli_history_record_and_inspect():
    """Test recording history and inspecting via CLI."""
    from argparse import Namespace

    from ultragraph.cli import _record_history, cmd_history

    module = _make_module()
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "test.ugm"
        save_ugm(path, module)

        # Record history
        loaded = load_ugm(str(path))
        x = np.random.randn(1, 4).astype(np.float32)
        _record_history(loaded, x)
        save_ugm(path, loaded)

        # Verify history saved
        loaded2 = load_ugm(str(path))
        assert loaded2.history is not None
        assert loaded2.history.buffer is not None

        # Inspect (just verify it runs without crash)
        import io
        import sys
        captured = io.StringIO()
        old = sys.stdout
        try:
            sys.stdout = captured
            cmd_history(Namespace(file=str(path), clear=False))
        finally:
            sys.stdout = old
        output = captured.getvalue()
        assert "test.ugm" in output


def test_cli_history_clear():
    """Test clearing history segment."""
    from argparse import Namespace

    from ultragraph.cli import _record_history, cmd_history

    module = _make_module()
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "test.ugm"
        save_ugm(path, module)

        loaded = load_ugm(str(path))
        x = np.random.randn(1, 4).astype(np.float32)
        _record_history(loaded, x)
        save_ugm(path, loaded)

        # Clear
        cmd_history(Namespace(file=str(path), clear=True))

        # Verify cleared
        loaded2 = load_ugm(str(path))
        assert loaded2.history is None