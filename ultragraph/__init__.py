"""ultragraph -- a byte-graph that is a 1-bit (ternary) LLM.

node/edge (1 byte each) -> tree (a whole net) -> ultra-edge wiring -> ultra-graph.
"""
from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

from .autograd import Tensor, ternary_linear
from .core import Edge, Embedding, NodeRef, Tree, UltraEdge, UltraGraph
from .nn import (
    Attention,
    LayerNorm,
    LearnedPositionalEmbedding,
    MoE,
    MultiHeadAttention,
    RMSNorm,
    linear_tree,
    mlp,
)
from .optim import SGD, Adam
from .pack import pack_ternary, unpack_ternary
from .quant import dequant, quantize_act_int8, quantize_weight_ternary
from .tokenize import ByteTokenizer

try:
    __version__ = _pkg_version("ultragraph-1bit")
except PackageNotFoundError:  # running from a source tree without install
    __version__ = "0.3.0"

__all__ = [
    "Tensor",
    "ternary_linear",
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
    "MoE",
    "ByteTokenizer",
    "SGD",
    "Adam",
    "pack_ternary",
    "unpack_ternary",
    "quantize_weight_ternary",
    "quantize_act_int8",
    "dequant",
]
