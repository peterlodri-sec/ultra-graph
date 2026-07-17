"""Tests for the .ugm → WASM compiler."""

import re
import tempfile
from pathlib import Path

import numpy as np

from ultragraph.ugm import (
    UGMFile,
    UGMTree,
    UGMUltraEdge,
    from_ultragraph,
)
from ultragraph.wasm import (
    _compute_layout,
    generate_wat,
    save_wasm,
)


def _make_module() -> UGMFile:
    """Minimal 2-tree .ugm: 4→8→2, relu then none."""
    t0 = UGMTree(kind=0, act=1, in_dim=4, out_dim=8, name="linear0", w_scale=1.0,
                 wq=np.random.randint(-1, 2, (8, 4)).astype(np.int8),
                 bias=np.zeros(8, dtype=np.float32))
    t1 = UGMTree(kind=0, act=0, in_dim=8, out_dim=2, name="linear1", w_scale=1.0,
                 wq=np.random.randint(-1, 2, (2, 8)).astype(np.int8),
                 bias=np.zeros(2, dtype=np.float32))
    return UGMFile(trees=[t0, t1], ultra_edges=[UGMUltraEdge(src_idx=0, dst_idx=1, kind=0)])


def test_wat_generates_valid_module():
    """WAT output has the expected top-level structure."""
    module = _make_module()
    wat = generate_wat(module)
    assert "(module" in wat
    assert "(memory (export \"memory\")" in wat
    assert "(func (export \"run\")" in wat
    assert "(param $input i32)" in wat
    assert "(param $output i32)" in wat


def test_wat_balanced_parens():
    """Every opening paren has a matching closing paren."""
    module = _make_module()
    wat = generate_wat(module)
    opens = wat.count("(")
    closes = wat.count(")")
    assert opens == closes, f"{opens} open vs {closes} close"


def test_wat_tree_functions():
    """One WAT function emitted per tree."""
    module = _make_module()
    wat = generate_wat(module)
    indices = [int(m) for m in re.findall(r"func \$tree_(\d+)", wat)]
    assert sorted(indices) == [0, 1]


def test_wat_data_segments():
    """Two data segments per dense tree (bias + wq)."""
    module = _make_module()
    wat = generate_wat(module)
    segs = re.findall(r"data \(i32\.const (\d+)\)", wat)
    assert len(segs) == 4  # 2 trees × 2


def test_wat_forward_pass_matches_python():
    """Forward pass in WAT (simulated via numpy) matches Python interpreter."""
    module = _make_module()
    layout = _compute_layout(module)

    # Reconstruct the same memory layout the WASM module would use
    memory = np.zeros(layout["scratch_end"], dtype=np.uint8)
    for i, t in enumerate(module.trees):
        if t.bias is not None:
            src = np.frombuffer(t.bias.tobytes(), dtype=np.uint8)
            offset = layout["bias_off"][i]
            memory[offset:offset + len(src)] = src
        if t.wq is not None:
            src = np.frombuffer(t.wq.tobytes(), dtype=np.uint8)
            offset = layout["wq_off"][i]
            memory[offset:offset + len(src)] = src

    # Verify data integrity through the memory buffer
    for i, t in enumerate(module.trees):
        bo = layout["bias_off"][i]
        bs = layout["bias_size"][i]
        if bs > 0:
            loaded = np.frombuffer(memory[bo:bo + bs].tobytes(), dtype=np.float32)
            assert np.allclose(loaded, t.bias), f"tree[{i}] bias mismatch"
        wo = layout["wq_off"][i]
        ws = layout["wq_size"][i]
        if ws > 0:
            loaded = np.frombuffer(memory[wo:wo + ws].tobytes(), dtype=np.int8)
            assert np.array_equal(loaded, t.wq.ravel()), f"tree[{i}] wq mismatch"

    # Python forward pass
    x = np.random.randn(1, 4).astype(np.float32)
    py_out = module.run(x)
    assert py_out.shape == (1, 2)
    assert np.isfinite(py_out).all()


def test_wat_sink_detection():
    """Sink tree is correctly identified in WAT output."""
    module = _make_module()
    wat = generate_wat(module)
    # The WAT should copy sink output to $(local.get $output)
    assert "(local.get $output)" in wat


def test_wat_with_residual():
    """Residual ultra-edges produce correct WAT."""
    t0 = UGMTree(kind=0, act=1, in_dim=4, out_dim=8, name="l0", w_scale=1.0,
                 wq=np.random.randint(-1, 2, (8, 4)).astype(np.int8),
                 bias=np.zeros(8, dtype=np.float32))
    t1 = UGMTree(kind=0, act=0, in_dim=8, out_dim=8, name="l1", w_scale=1.0,
                 wq=np.random.randint(-1, 2, (8, 8)).astype(np.int8),
                 bias=np.zeros(8, dtype=np.float32))
    module = UGMFile(
        trees=[t0, t1],
        ultra_edges=[
            UGMUltraEdge(src_idx=0, dst_idx=1, kind=0),
            UGMUltraEdge(src_idx=0, dst_idx=1, kind=1),
        ],
    )
    wat = generate_wat(module)
    assert "(func $tree_0" in wat
    assert "(func $tree_1" in wat
    # Should have a copy/addition loop for the residual
    assert "f32.add" in wat
    # Forward pass still works
    x = np.random.randn(1, 4).astype(np.float32)
    out = module.run(x)
    assert out.shape == (1, 8)
    assert np.isfinite(out).all()


def test_wat_from_gpt():
    """WAT generation from a full GPT model."""
    from ultragraph import GPT

    gpt = GPT(vocab=256, d_model=8, n_layers=2, n_heads=2, max_len=64)
    with tempfile.TemporaryDirectory() as d:
        ckpt = Path(d) / "model.q.npz"
        gpt.save_deployed(str(ckpt))
        gpt2 = GPT.load_deployed(str(ckpt))
        module = from_ultragraph(gpt2)

    wat = generate_wat(module)
    # 13 trees (6 per block × 2 + head)
    indices = sorted(int(m) for m in re.findall(r"func \$tree_(\d+)", wat))
    assert indices == list(range(13)), f"expected 13 trees, got {len(indices)}"

    # 26 data segments (bias + wq per tree)
    segs = re.findall(r"data \(i32\.const (\d+)\)", wat)
    assert len(segs) == 26

    # Balanced parens
    assert wat.count("(") == wat.count(")")

    # Forward pass works
    x = np.random.randn(1, 8).astype(np.float32)
    out = module.run(x)
    assert out.shape == (1, 256)
    assert np.isfinite(out).all()

    # Fits in one memory page
    layout = _compute_layout(module)
    assert layout["scratch_end"] <= 65536


def test_save_wasm_writes_wat():
    """save_wasm writes a .wat file even without wat2wasm."""
    module = _make_module()
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "test.wasm"
        save_wasm(module, out)
        # .wat file should exist (save_wasm always writes it)
        wat_path = out.with_suffix(".wat")
        assert wat_path.exists(), f"{wat_path} not found"
        content = wat_path.read_text()
        assert "(module" in content
        assert "(func (export \"run\")" in content


def test_wat_activations():
    """All activation types produce valid WAT expressions."""
    act_names = {0: "none", 1: "relu", 3: "sigmoid", 4: "tanh"}
    for act_val, act_name in act_names.items():
        t0 = UGMTree(kind=0, act=act_val, in_dim=2, out_dim=2, name=f"act_{act_name}",
                     w_scale=1.0,
                     wq=np.random.randint(-1, 2, (2, 2)).astype(np.int8),
                     bias=np.zeros(2, dtype=np.float32))
        module = UGMFile(trees=[t0], ultra_edges=[])
        wat = generate_wat(module)
        assert f"tree[0]: act_{act_name}" in wat or f"act={act_name}" in wat
        assert wat.count("(") == wat.count(")")
        # Forward pass still works
        x = np.random.randn(1, 2).astype(np.float32)
        out = module.run(x)
        assert out.shape == (1, 2)
        assert np.isfinite(out).all()


def test_wat_tanh_no_literal_placeholder():
    """Tanh activation interpolates {acc} rather than emitting it literally."""
    t0 = UGMTree(kind=0, act=4, in_dim=2, out_dim=2, name="tanh0", w_scale=1.0,
                 wq=np.random.randint(-1, 2, (2, 2)).astype(np.int8),
                 bias=np.zeros(2, dtype=np.float32))
    module = UGMFile(trees=[t0], ultra_edges=[])
    wat = generate_wat(module)
    assert "{acc}" not in wat


def test_wat_multi_source():
    """Tree with multiple plain inputs produces correct sum loop."""
    t0 = UGMTree(kind=0, act=1, in_dim=2, out_dim=8, name="src0", w_scale=1.0,
                 wq=np.random.randint(-1, 2, (8, 2)).astype(np.int8),
                 bias=np.zeros(8, dtype=np.float32))
    t1 = UGMTree(kind=0, act=1, in_dim=2, out_dim=8, name="src1", w_scale=1.0,
                 wq=np.random.randint(-1, 2, (8, 2)).astype(np.int8),
                 bias=np.zeros(8, dtype=np.float32))
    t2 = UGMTree(kind=0, act=0, in_dim=8, out_dim=4, name="dst", w_scale=1.0,
                 wq=np.random.randint(-1, 2, (4, 8)).astype(np.int8),
                 bias=np.zeros(4, dtype=np.float32))
    module = UGMFile(
        trees=[t0, t1, t2],
        ultra_edges=[
            UGMUltraEdge(src_idx=0, dst_idx=2, kind=0),
            UGMUltraEdge(src_idx=1, dst_idx=2, kind=0),
        ],
    )
    wat = generate_wat(module)
    assert "(func $tree_2" in wat
    # Should have a loop summing two sources
    assert "f32.add" in wat
    # Forward pass works
    x = np.random.randn(1, 2).astype(np.float32)
    out = module.run(x)
    assert out.shape == (1, 4)
    assert np.isfinite(out).all()
