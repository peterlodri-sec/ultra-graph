"""ultragraph -- a byte-graph that is a 1-bit (ternary) LLM.

node/edge (1 byte each) -> tree (a whole net) -> ultra-edge wiring -> ultra-graph.
"""
from __future__ import annotations

from .autograd import Tensor, ternary_linear
from .core import Edge, Embedding, NodeRef, Tree, UltraEdge, UltraGraph
from .nn import Attention, linear_tree, mlp
from .optim import SGD
from .quant import dequant, quantize_act_int8, quantize_weight_ternary

__version__ = "0.1.0"

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
    "SGD",
    "quantize_weight_ternary",
    "quantize_act_int8",
    "dequant",
]
