"""Minimal tests for the `ug` CLI commands."""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import numpy as np

from ultragraph.cli import cmd_breed, cmd_new, cmd_practice, cmd_pull, cmd_report


def _fake_checkpoint(path: Path, with_meta: bool = True) -> None:
    """Write a minimal deployed-style .npz for testing.

    Matches a 2-layer GPT(d_model=8, n_heads=2, mlp_ratio=4).
    13 dense trees total, 5 norm gains, 1 embedding.
    """
    arrays = {}
    shapes = [(8, 8)] * 4 + [(32, 8), (8, 32)]   # block 0
    shapes += [(8, 8)] * 4 + [(32, 8), (8, 32)]    # block 1
    shapes.append((256, 8))                          # head
    for i, (out_d, in_d) in enumerate(shapes):
        arrays[f"t{i}_wq"] = np.zeros((out_d, in_d), dtype=np.int8)
        arrays[f"t{i}_bias"] = np.zeros(out_d, dtype=np.float32)
    arrays["f_embed"] = np.zeros((256, 8), dtype=np.float32)
    for name in ("n1_0", "n2_0", "n1_1", "n2_1", "nf"):
        arrays[f"f_{name}"] = np.ones(8, dtype=np.float32)
    if with_meta:
        hp = {"vocab": 256, "d_model": 8, "n_layers": 2, "n_heads": 2, "max_len": 64, "mlp_ratio": 4}
        tree_meta = [{"shape": list(s), "w_scale": 1.0, "packed": False} for s in shapes]
        meta = {"hp": hp, "packed": False, "trees": tree_meta}
        arrays["__meta__"] = np.array(json.dumps(meta))
    np.savez(path, **arrays)


def _make_args(**kw):
    from argparse import Namespace
    return Namespace(**kw)


def test_new_scaffolds_project():
    with tempfile.TemporaryDirectory() as d:
        orig = Path.cwd()
        try:
            os.chdir(d)
            cmd_new(_make_args(name="mylm", lang="swahili"))
            assert Path("mylm/corpus").is_dir()
            assert Path("mylm/checkpoints").is_dir()
            assert Path("mylm/train.yml").is_file()
            assert Path("mylm/justfile").is_file()
            assert "swahili" in Path("mylm/train.yml").read_text()
        finally:
            os.chdir(orig)


def test_breed_averages_checkpoints():
    with tempfile.TemporaryDirectory() as d:
        a = Path(d) / "a.npz"
        b = Path(d) / "b.npz"
        _fake_checkpoint(a)
        _fake_checkpoint(b)
        out = Path(d) / "out.npz"
        cmd_breed(_make_args(model_a=str(a), model_b=str(b), ratio=0.5, output=str(out)))
        assert out.exists()
        data = np.load(out, allow_pickle=False)
        assert "__meta__" in data.files
        meta = json.loads(str(data["__meta__"]))
        assert meta["hp"]["parents"] == ["a.npz", "b.npz"]


def test_report_generates_html():
    with tempfile.TemporaryDirectory() as d:
        ckpt = Path(d) / "model.q.npz"
        _fake_checkpoint(ckpt)
        cmd_report(_make_args(checkpoint=str(ckpt)))
        html = ckpt.with_suffix(".html")
        assert html.exists()
        assert "ultragraph Report" in html.read_text()
        assert "Parameters" in html.read_text()


def test_practice_runs():
    import io
    import sys
    captured = io.StringIO()
    old = sys.stdout
    try:
        sys.stdout = captured
        cmd_practice(_make_args(steps=10, corpus=None))
    finally:
        sys.stdout = old
    out = captured.getvalue()
    assert "step" in out and "loss" in out and "complete" in out


def test_pull_download_fails_gracefully():
    import io
    import sys
    captured = io.StringIO()
    old = sys.stdout, sys.stderr
    try:
        sys.stdout = sys.stderr = captured
        cmd_pull(_make_args(name="nonexistent", url="https://0.0.0.0/nope.npz", dir="/tmp"))
    except SystemExit:
        pass
    finally:
        sys.stdout, sys.stderr = old
