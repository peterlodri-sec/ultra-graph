"""Tests for hub, spar, taste CLI commands."""
from __future__ import annotations

import json
import os
import tempfile
from argparse import Namespace
from pathlib import Path

import numpy as np

from ultragraph.cli import _search_local, cmd_spar, cmd_taste


def _fake_checkpoint(path: Path) -> None:
    arrays = {}
    shapes = [(8, 8)] * 4 + [(32, 8), (8, 32)]
    shapes += [(8, 8)] * 4 + [(32, 8), (8, 32)]
    shapes.append((256, 8))
    for i, (out_d, in_d) in enumerate(shapes):
        arrays[f"t{i}_wq"] = np.zeros((out_d, in_d), dtype=np.int8)
        arrays[f"t{i}_bias"] = np.zeros(out_d, dtype=np.float32)
    arrays["f_embed"] = np.zeros((256, 8), dtype=np.float32)
    for name in ("n1_0", "n2_0", "n1_1", "n2_1", "nf"):
        arrays[f"f_{name}"] = np.ones(8, dtype=np.float32)
    hp = {"vocab": 256, "d_model": 8, "n_layers": 2, "n_heads": 2, "max_len": 64, "mlp_ratio": 4}
    tree_meta = [{"shape": list(s), "w_scale": 1.0, "packed": False} for s in shapes]
    meta = {"hp": hp, "packed": False, "trees": tree_meta}
    arrays["__meta__"] = np.array(json.dumps(meta))
    np.savez(path, **arrays)


def test_spar_runs():
    with tempfile.TemporaryDirectory() as d:
        a = Path(d) / "a.npz"
        b = Path(d) / "b.npz"
        _fake_checkpoint(a)
        _fake_checkpoint(b)
        import io
        import sys
        captured = io.StringIO()
        old = sys.stdout
        try:
            sys.stdout = captured
            cmd_spar(Namespace(model_a=str(a), model_b=str(b), prompt="Hi", n=8))
        finally:
            sys.stdout = old
        out = captured.getvalue()
        assert "Sparring match" in out
        assert "Model A" in out
        assert "Model B" in out


def test_taste_runs():
    with tempfile.TemporaryDirectory() as d:
        ckpt = Path(d) / "model.npz"
        _fake_checkpoint(ckpt)
        import io
        import sys
        captured = io.StringIO()
        old = sys.stdout
        try:
            sys.stdout = captured
            cmd_taste(Namespace(model=str(ckpt), prompt="The", n=16))
        finally:
            sys.stdout = old
        out = captured.getvalue()
        assert "Tasting flight" in out
        assert "temp=" in out
        assert "greedy" in out


def test_search_local_no_files():
    with tempfile.TemporaryDirectory() as d:
        orig = Path.cwd()
        try:
            os.chdir(d)
            import io
            import sys
            captured = io.StringIO()
            old = sys.stdout
            try:
                sys.stdout = captured
                _search_local()
            finally:
                sys.stdout = old
            assert "No local" in captured.getvalue()
        finally:
            os.chdir(orig)


def test_search_local_find_files():
    with tempfile.TemporaryDirectory() as d:
        orig = Path.cwd()
        try:
            os.chdir(d)
            Path("model1.npz").write_bytes(b"\x00" * 100)
            Path("model2.npz").write_bytes(b"\x00" * 200)
            import io
            import sys
            captured = io.StringIO()
            old = sys.stdout
            try:
                sys.stdout = captured
                _search_local()
            finally:
                sys.stdout = old
            assert "model1.npz" in captured.getvalue()
            assert "model2.npz" in captured.getvalue()
        finally:
            os.chdir(orig)