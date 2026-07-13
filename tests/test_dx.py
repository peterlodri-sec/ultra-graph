"""Tests for model_config, command, match/case dispatch, TypeVarTuple, and SSE streaming."""
from __future__ import annotations

import io
import sys
from unittest.mock import patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# @ug.model
# ---------------------------------------------------------------------------
class TestModelConfig:
    def test_basic_model_creation(self):
        """@ug.model wraps a config dataclass and builds a GPT."""
        from ultragraph.model_config import model

        @model
        class TinyLM:
            vocab: int = 256
            d_model: int = 32
            n_layers: int = 1
            n_heads: int = 2
            max_len: int = 64
            mlp_ratio: int = 4
            name: str = "tiny"

        cfg = TinyLM()
        assert cfg.vocab == 256
        assert cfg.d_model == 32
        gpt = cfg()
        from ultragraph.model import GPT
        assert isinstance(gpt, GPT)
        assert gpt.vocab == 256
        assert gpt.name == "tiny"

    def test_model_config_overrides(self):
        """Fields can be overridden at construction."""
        from ultragraph.model_config import model

        @model
        class MedLM:
            vocab: int = 256
            d_model: int = 64
            n_layers: int = 2
            n_heads: int = 2

        cfg = MedLM(d_model=128, n_layers=3)
        assert cfg.d_model == 128
        assert cfg.n_layers == 3
        assert cfg.vocab == 256

    def test_model_config_invalid_override(self):
        """Passing an unknown field raises TypeError."""
        from ultragraph.model_config import model

        @model
        class SmallLM:
            vocab: int = 256
            d_model: int = 32

        with pytest.raises(TypeError, match="no field"):
            SmallLM(unknown_field=123)

    def test_model_config_no_default(self):
        """Fields without a default are supported and required."""
        from ultragraph.model_config import model

        @model
        class RequiredLM:
            vocab: int
            d_model: int = 32

        cfg = RequiredLM(vocab=256)
        assert cfg.vocab == 256
        assert cfg.d_model == 32

        with pytest.raises(TypeError, match="missing.*required"):
            RequiredLM()


# ---------------------------------------------------------------------------
# @ug.command
# ---------------------------------------------------------------------------
class TestCommand:
    def test_command_registration(self):
        """@command registers a function."""
        from ultragraph.command import _registry, command

        _registry.clear()

        @command("test-cmd")
        def my_cmd(name: str = "default", count: int = 1):
            """Run a test command."""
            return f"{name}:{count}"

        assert "test-cmd" in _registry
        assert "Run a test command" in _registry["test-cmd"]["help"]
        _registry.clear()

    def test_command_dispatch(self):
        """run() dispatches to the correct command."""
        from ultragraph.command import _registry, command, run

        _registry.clear()
        results = []

        @command("hello")
        def hello(name: str = "world"):
            """Say hello."""
            results.append(f"hello {name}")

        run(["hello", "--name", "crush"])
        assert results == ["hello crush"]

        results.clear()
        run(["hello"])
        assert results == ["hello world"]
        _registry.clear()

    def test_command_help_available(self):
        """Missing command prints help."""
        from ultragraph.command import _registry, command, run

        _registry.clear()

        @command("list")
        def list_cmd():
            """List items."""

        buf = io.StringIO()
        with patch.object(sys, "stdout", buf):
            run([])
        output = buf.getvalue()
        assert "list" in output
        _registry.clear()

    def test_command_bool_flag(self):
        """Bool params become flags."""
        from ultragraph.command import _registry, command, run

        _registry.clear()
        results = []

        @command("train")
        def train(quiet: bool = False, steps: int = 10):
            """Train the model."""
            results.append((quiet, steps))

        run(["train", "--quiet"])
        assert results == [(True, 10)]

        results.clear()
        run(["train", "--steps", "20"])
        assert results == [(False, 20)]
        _registry.clear()

    def test_command_bool_default_true(self):
        """Bool params with default=True can be turned off with --no-*."""
        from ultragraph.command import _registry, command, run

        _registry.clear()
        results = []

        @command("run-val")
        def run_val(validate: bool = True):
            """Run validation."""
            results.append(validate)

        run(["run-val", "--no-validate"])
        assert results == [False]

        results.clear()
        run(["run-val"])
        assert results == [True]
        _registry.clear()

    def test_command_bool_required(self):
        """Required bool params must be provided and accept either flag."""
        from ultragraph.command import _registry, command, run

        _registry.clear()
        results = []

        @command("toggle")
        def toggle(enable: bool):
            """Toggle state."""
            results.append(enable)

        run(["toggle", "--enable"])
        assert results == [True]

        results.clear()
        run(["toggle", "--no-enable"])
        assert results == [False]

        _registry.clear()


# ---------------------------------------------------------------------------
# match/case dispatch
# ---------------------------------------------------------------------------
class TestMatchCase:
    def test_reshape_with_tuple(self):
        """Tensor.reshape accepts a single tuple argument."""
        from ultragraph import tensor
        x = tensor(np.arange(6).reshape(2, 3).astype(np.float32))
        y = x.reshape((3, 2))
        assert y.shape == (3, 2)

    def test_reshape_with_list(self):
        """Tensor.reshape accepts a single list argument."""
        from ultragraph import tensor
        x = tensor(np.arange(6).reshape(2, 3).astype(np.float32))
        y = x.reshape([6])
        assert y.shape == (6,)

    def test_reshape_with_args(self):
        """Tensor.reshape works with multiple args."""
        from ultragraph import tensor
        x = tensor(np.arange(6).reshape(2, 3).astype(np.float32))
        y = x.reshape(6)
        assert y.shape == (6,)

    def test_tree_forward_deployed_match(self):
        """Tree.forward uses match/case for deployed state."""
        from ultragraph.autograd import Tensor
        from ultragraph.core import Tree

        # Set up in deployed mode by setting w_master to None
        t = Tree.dense(4, 4, "test")
        t.adhoc["w_master"] = None  # triggers deployed path
        t.adhoc["bias"] = Tensor(np.zeros(4, dtype=np.float32))
        t.wq = np.zeros((4, 4), dtype=np.int8)
        t.w_scale = 1.0

        x = Tensor(np.ones((1, 4), dtype=np.float32))
        y = t.forward(x)
        assert y.shape == (1, 4)

    def test_tree_forward_act_match(self):
        """Tree.forward uses match/case for activation type."""
        from ultragraph.autograd import Tensor
        from ultragraph.core import Tree

        t = Tree.dense(4, 4, "test", act="none")
        w = np.random.randn(4, 4).astype(np.float32)
        t.adhoc["w_master"] = Tensor(w, requires_grad=False)
        t.adhoc["bias"] = Tensor(np.zeros(4, dtype=np.float32))
        t.requantize()
        x = Tensor(np.ones((1, 4), dtype=np.float32))
        y = t.forward(x)
        assert y.shape == (1, 4)

        # Test valid activations
        for act in ["relu", "gelu", "silu", "tanh", "sigmoid"]:
            t.act = act
            y = t.forward(x)
            assert y.shape == (1, 4)

        # Test invalid activation raises ValueError
        t.act = "invalid_act"
        with pytest.raises(ValueError, match="Unsupported activation"):
            t.forward(x)

    def test_sequential_train_match(self):
        """Sequential.train uses match/case for module dispatch."""
        from ultragraph.nn import Dropout, Sequential
        seq = Sequential(Dropout(0.0))
        assert hasattr(seq.modules[0], "training")
        seq.train(True)
        assert seq.modules[0].training is True
        seq.eval()
        assert seq.modules[0].training is False


# ---------------------------------------------------------------------------
# TypeVarTuple generics
# ---------------------------------------------------------------------------
class TestTypeVarTuple:
    def test_shape_typevar_importable(self):
        """Shape TypeVarTuple is importable from autograd."""
        from ultragraph.autograd import Shape
        assert Shape is not None
        assert Shape.__name__ == "Shape"

    def test_tensor_with_shape_hint_works(self):
        """Tensor creation still works with TypeVarTuple imported."""
        from ultragraph import tensor
        x = tensor(np.ones((2, 3, 4)))
        assert x.shape == (2, 3, 4)


# ---------------------------------------------------------------------------
# SSE streaming MCP server
# ---------------------------------------------------------------------------
class TestMCPStreaming:
    def test_stream_tokens_tool_exists(self):
        """stream_tokens is registered as an async tool."""
        pytest.importorskip("mcp")
        import mcp_server.server as srv
        assert srv.stream_tokens is not None

    def test_stream_tokens_is_async(self):
        """stream_tokens is an async function or async generator."""
        import inspect
        pytest.importorskip("mcp")
        import mcp_server.server as srv
        assert inspect.iscoroutinefunction(srv.stream_tokens) or inspect.isasyncgenfunction(srv.stream_tokens)
