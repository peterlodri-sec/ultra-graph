"""ultragraph -- a byte-graph that is a 1-bit (ternary) LLM.

node/edge (1 byte each) -> tree (a whole net) -> ultra-edge wiring -> ultra-graph.
"""
from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

from .autograd import Tensor, cat, ternary_linear
from .core import Edge, Embedding, NodeRef, Tree, UltraEdge, UltraGraph
from .io import load, load_params, save, save_params
from .model import GPT, Mesh, TransformerBlock
from .nn import (
    Attention,
    Dropout,
    LayerNorm,
    LearnedPositionalEmbedding,
    MoE,
    MultiHeadAttention,
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
    __version__ = "0.16.0"

__all__ = [
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
