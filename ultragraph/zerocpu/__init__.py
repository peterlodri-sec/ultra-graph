"""ZeroCPU -- deterministic integer compute for ultragraph.

Three pieces that the float32 trainer and the int8 deployment path leave open:

* :mod:`~ultragraph.zerocpu.fixed` -- Q16.16 arithmetic with symmetric rounding,
  so identical programs produce identical bytes on every machine.
* :mod:`~ultragraph.zerocpu.spacetime` -- a causal world model whose nodes carry
  int8 state and positions, advanced by a simultaneous integer tick.
* :mod:`~ultragraph.zerocpu.evolve` -- p/q prompts: exact-rational mutation
  programs that program ternary weights and wiring reproducibly from a seed.
* :mod:`~ultragraph.zerocpu.emit` -- the compiled ``UGMZ`` world artifact, so the
  model ticks in Rust (``zerocpu::neon``) instead of Python.
* :mod:`~ultragraph.zerocpu.vaked_bridge` -- a lowered Vaked workflow (``wf.json``)
  becomes a world, so a ``.vaked`` source can be compiled straight to ``.ugmz``.
* :mod:`~ultragraph.zerocpu.lod` -- quasi-regular partitioning: the coarse graph
  that keeps a big world model tickable, with its defect reported, not assumed.
"""

from .emit import emit_ugm, from_bytes, load_ugm, to_bytes
from .evolve import Op, PqError, Prompt, Report, apply_pq, evolve, parse_pq
from .fixed import (
    FRAC_BITS,
    ONE,
    fx_add,
    fx_dist,
    fx_div,
    fx_dot,
    fx_matvec,
    fx_mul,
    fx_sqrt,
    fx_sub,
    saturate,
    to_float,
    to_fx,
    to_int,
)
from .lod import Partition, PartitionError, coarsen, density_stats, partition
from .spacetime import INT8_MAX, INT8_MIN, CausalEdge, SpacetimeGraph
from .vaked_bridge import (
    WorkflowError,
    lower_vaked,
    vaked_to_ugm,
    workflow_meta,
    workflow_to_graph,
)

__all__ = [
    "CausalEdge",
    "FRAC_BITS",
    "INT8_MAX",
    "INT8_MIN",
    "ONE",
    "Op",
    "Partition",
    "PartitionError",
    "PqError",
    "Prompt",
    "Report",
    "SpacetimeGraph",
    "WorkflowError",
    "apply_pq",
    "coarsen",
    "density_stats",
    "emit_ugm",
    "evolve",
    "from_bytes",
    "fx_add",
    "fx_dist",
    "fx_div",
    "fx_dot",
    "fx_matvec",
    "fx_mul",
    "fx_sqrt",
    "fx_sub",
    "load_ugm",
    "lower_vaked",
    "parse_pq",
    "partition",
    "saturate",
    "to_bytes",
    "to_float",
    "to_fx",
    "to_int",
    "vaked_to_ugm",
    "workflow_meta",
    "workflow_to_graph",
]
