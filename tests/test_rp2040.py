"""Tests for RP2040 C code generation and history segment."""

import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest

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


def _make_wide_chain_module() -> UGMFile:
    """3-tree plain chain whose middle tree is wider than the first input
    and the last output (exercises buffer sizing + aliasing across trees)."""
    t0 = UGMTree(kind=0, act=1, in_dim=4, out_dim=10, name="l0", w_scale=0.05,
                 wq=np.random.default_rng(1).integers(-3, 4, (10, 4)).astype(np.int8),
                 bias=np.random.default_rng(2).standard_normal(10).astype(np.float32))
    t1 = UGMTree(kind=0, act=1, in_dim=10, out_dim=8, name="l1", w_scale=0.05,
                 wq=np.random.default_rng(3).integers(-3, 4, (8, 10)).astype(np.int8),
                 bias=np.random.default_rng(4).standard_normal(8).astype(np.float32))
    t2 = UGMTree(kind=0, act=0, in_dim=8, out_dim=3, name="l2", w_scale=0.05,
                 wq=np.random.default_rng(5).integers(-3, 4, (3, 8)).astype(np.int8),
                 bias=np.random.default_rng(6).standard_normal(3).astype(np.float32))
    return UGMFile(trees=[t0, t1, t2], ultra_edges=[
        UGMUltraEdge(src_idx=0, dst_idx=1, kind=0),
        UGMUltraEdge(src_idx=1, dst_idx=2, kind=0),
    ])


def test_rp2040_no_buffer_aliasing():
    """Regression: chained trees must not run in-place (tree(buf, buf))."""
    c_code = generate_c(_make_wide_chain_module())
    # The specific aliasing bug that corrupted every tree after the first:
    assert "tree1_forward(buf, buf)" not in c_code
    assert "buf, buf)" not in c_code
    # No forward call may pass the same array as both source and destination.
    calls = re.findall(r"tree\d+_forward\((\w+),\s*(\w+)\)", c_code)
    assert calls, "expected at least one tree forward call"
    for src, dst in calls:
        assert src != dst, f"aliasing call detected: tree_forward({src}, {dst})"
    # Ping-pong buffers present; first tree reads the caller's input.
    assert "buf_a" in c_code and "buf_b" in c_code
    assert re.search(r"tree\d+_forward\(input,\s*buf_a\)", c_code)


def test_rp2040_buffer_sized_for_widest_tree():
    """Regression: buffers must fit the widest tree, not just first/last."""
    c_code = generate_c(_make_wide_chain_module())
    sizes = [int(n) for n in re.findall(r"buf_[ab]\[(\d+)\]", c_code)]
    assert sizes, "no ping-pong buffers declared"
    # Middle tree emits 10 floats; a max(in_dim=4, out_dim=3)=4 buffer overflows.
    assert min(sizes) >= 10, f"buffer too small for widest tree: {sizes}"


def test_rp2040_rejects_residual_edges():
    """Residual edges are unsupported and must fail loud, not emit wrong C."""
    module = _make_module()
    module.ultra_edges.append(UGMUltraEdge(src_idx=0, dst_idx=1, kind=1))
    with pytest.raises(NotImplementedError):
        generate_c(module)


def test_rp2040_rejects_merge_edges():
    """Merge (two sources into one tree) is unsupported and must fail loud."""
    t0 = UGMTree(kind=0, act=0, in_dim=4, out_dim=3, name="a", w_scale=1.0,
                 wq=np.zeros((3, 4), np.int8), bias=np.zeros(3, np.float32))
    t1 = UGMTree(kind=0, act=0, in_dim=4, out_dim=3, name="b", w_scale=1.0,
                 wq=np.zeros((3, 4), np.int8), bias=np.zeros(3, np.float32))
    t2 = UGMTree(kind=0, act=0, in_dim=3, out_dim=2, name="c", w_scale=1.0,
                 wq=np.zeros((2, 3), np.int8), bias=np.zeros(2, np.float32))
    module = UGMFile(trees=[t0, t1, t2], ultra_edges=[
        UGMUltraEdge(src_idx=0, dst_idx=2, kind=0),
        UGMUltraEdge(src_idx=1, dst_idx=2, kind=0),
    ])
    with pytest.raises(NotImplementedError):
        generate_c(module)


def test_rp2040_numerical_parity():
    """Compile the generated C and check it matches the Python interpreter.

    This is the strongest guard for the aliasing/sizing fixes: if trees ran
    in-place or overflowed, the compiled output would diverge from run().
    """
    gcc = shutil.which("gcc") or shutil.which("cc") or shutil.which("clang")
    if gcc is None:
        pytest.skip("no C compiler available")

    module = _make_wide_chain_module()
    c_src = generate_c(module, name="model")

    rng = np.random.default_rng(7)
    x = rng.standard_normal((1, 4)).astype(np.float32)
    expected = module.run(x)[0]

    inits = ", ".join(f"{float(v):.9g}" for v in x[0])
    driver = (
        "\n#include <stdio.h>\n"
        "int main(void) {\n"
        f"  float input[4] = {{{inits}}};\n"
        "  float output[3];\n"
        "  model_forward(input, output);\n"
        "  for (int i = 0; i < 3; i++) printf(\"%.7g\\n\", output[i]);\n"
        "  return 0;\n"
        "}\n"
    )
    with tempfile.TemporaryDirectory() as d:
        src_path = Path(d) / "model.c"
        src_path.write_text(c_src + driver)
        bin_path = Path(d) / "prog"
        subprocess.run([gcc, str(src_path), "-lm", "-o", str(bin_path)], check=True)
        result = subprocess.run([str(bin_path)], capture_output=True, text=True, check=True)

    got = np.array([float(tok) for tok in result.stdout.split()], dtype=np.float32)
    np.testing.assert_allclose(got, expected, rtol=1e-3, atol=1e-3)


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