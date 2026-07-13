"""ultragraph -- a byte-graph that is a 1-bit (ternary) LLM.

node/edge (1 byte each) -> tree (a whole net) -> ultra-edge wiring -> ultra-graph.
"""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

import numpy as np

from .autograd import Tensor, cat, ternary_linear
from .client import Client
from .command import command
from .command import run as run_cli
from .compile import compile
from .core import Edge, Embedding, NodeRef, Tree, UltraEdge, UltraGraph
from .io import load, load_params, save, save_params
from .model import GPT, Mesh, TransformerBlock
from .model_config import model
from .nn import (
    Attention,
    Dropout,
    LayerNorm,
    LearnedPositionalEmbedding,
    Linear,
    MoE,
    MultiHeadAttention,
    Residual,
    RMSNorm,
    RoPE,
    Sequential,
    linear_tree,
    mlp,
)
from .optim import SGD, Adam, CosineSchedule
from .pack import pack_ternary, unpack_ternary
from .quant import dequant, quantize_act_int8, quantize_weight_ternary
from .tokenize import ByteTokenizer
from .vaked import compile_vaked, lower_graph

try:
    __version__ = _pkg_version("ultragraph-1bit")
except PackageNotFoundError:  # running from a source tree without install
    __version__ = "0.19.0"


def tensor(data, requires_grad=False) -> Tensor:
    """Create a Tensor from array-like data. Wraps numpy arrays invisibly.

    ``ug.tensor([[1,2],[3,4]])`` works like ``np.array`` but returns a
    Tensor that flows through autograd. No need to import ``Tensor`` separately.

    Example:
        x = ug.tensor(np.random.randn(4, 8))
        y = model(x)
    """
    return Tensor(data, requires_grad=requires_grad)


def seed(s: int = 42):
    """Set the global random seed for reproducibility.

    Example:
        ug.seed(42)
        model = ug.GPT(256, 128, 2, 2)
    """
    np.random.seed(s)

__all__ = [
    "tensor",
    "seed",
    "compile",
    "model",
    "command",
    "run_cli",
    "Client",
    "Tensor",
    "ternary_linear",
    "cat",
    "Tree",
    "Edge",
    "NodeRef",
    "UltraEdge",
    "UltraGraph",
    "Embedding",
    "linear_tree",
    "mlp",
    "Attention",
    "MultiHeadAttention",
    "RMSNorm",
    "LayerNorm",
    "LearnedPositionalEmbedding",
    "RoPE",
    "MoE",
    "Dropout",
    "Linear",
    "Residual",
    "Sequential",
    "TransformerBlock",
    "GPT",
    "Mesh",
    "ByteTokenizer",
    "SGD",
    "Adam",
    "CosineSchedule",
    "pack_ternary",
    "unpack_ternary",
    "save",
    "load",
    "save_params",
    "load_params",
    "lower_graph",
    "compile_vaked",
    "quantize_weight_ternary",
    "quantize_act_int8",
    "dequant",
]
