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
* :mod:`~ultragraph.zerocpu.pqq` -- compile a `p/q` prompt (draws resolved) and
  replay the result without a PRNG, in Python or in Rust.
* :mod:`~ultragraph.zerocpu.hope` -- project a text to integer heat, seed it into
  a world, tick, and collapse the state with an observer.
* :mod:`~ultragraph.zerocpu.trit` -- the Q-delta ternary quantizer and the
  A6P-15A bitplane layout: thresholds one way, popcount-ready planes the other.
* :mod:`~ultragraph.zerocpu.router` -- deterministic integer routing: which
  handler (kernel, batch, model, plugin) a prompt deserves, and why.
* :mod:`~ultragraph.zerocpu.memnet` -- sequence-ordered associative memory that
  merges across nodes without a clock.
* :mod:`~ultragraph.zerocpu.mesh` -- the LAN peer announce frame and its merge
  rule, byte-exact against the Rust twin.
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
from .hope import (
    SIGNAL_STEP,
    Observer,
    Resonance,
    Signal,
    observer,
    resonate,
    seed_signal,
    signal,
)
from .lod import Partition, PartitionError, coarsen, density_stats, partition
from .memnet import MEMNET_SLOTS, MemNet, MemNetEntry, key_id
from .mesh import Peer, merge_peers, pack_announce, parse_announce
from .pqq import (
    PQQ_HEADER,
    PQQ_VERSION,
    Resolved,
    dump_pqq,
    load_pqq,
    resolve,
    run_pqq,
    write_bundle,
)
from .router import (
    ROUTE_BATCH,
    ROUTE_CAVEMAN,
    ROUTE_DENY,
    ROUTE_MODEL,
    ROUTE_PLUGIN,
    explain,
    features,
    route,
    route_name,
)
from .spacetime import INT8_MAX, INT8_MIN, CausalEdge, SpacetimeGraph
from .trit import (
    A6P_LEVELS,
    MAGNITUDE_MAX,
    MAX_LANES,
    bitplanes,
    decay,
    delta_for_level,
    from_planes,
    plane_parity,
    quantize,
    quantize_many,
)
from .vaked_bridge import (
    WorkflowError,
    lower_vaked,
    vaked_to_ugm,
    workflow_meta,
    workflow_to_graph,
)

__all__ = [
    "A6P_LEVELS",
    "CausalEdge",
    "FRAC_BITS",
    "INT8_MAX",
    "INT8_MIN",
    "MAGNITUDE_MAX",
    "MAX_LANES",
    "MEMNET_SLOTS",
    "MemNet",
    "MemNetEntry",
    "ONE",
    "Observer",
    "Op",
    "PQQ_HEADER",
    "PQQ_VERSION",
    "Partition",
    "PartitionError",
    "Peer",
    "PqError",
    "Prompt",
    "ROUTE_BATCH",
    "ROUTE_CAVEMAN",
    "ROUTE_DENY",
    "ROUTE_MODEL",
    "ROUTE_PLUGIN",
    "Report",
    "Resolved",
    "Resonance",
    "SIGNAL_STEP",
    "Signal",
    "SpacetimeGraph",
    "WorkflowError",
    "apply_pq",
    "bitplanes",
    "coarsen",
    "decay",
    "delta_for_level",
    "density_stats",
    "dump_pqq",
    "emit_ugm",
    "evolve",
    "explain",
    "features",
    "from_bytes",
    "from_planes",
    "fx_add",
    "fx_dist",
    "fx_div",
    "fx_dot",
    "fx_matvec",
    "fx_mul",
    "fx_sqrt",
    "fx_sub",
    "key_id",
    "load_pqq",
    "load_ugm",
    "lower_vaked",
    "merge_peers",
    "observer",
    "pack_announce",
    "parse_announce",
    "parse_pq",
    "partition",
    "plane_parity",
    "quantize",
    "quantize_many",
    "resolve",
    "resonate",
    "route",
    "route_name",
    "run_pqq",
    "saturate",
    "seed_signal",
    "signal",
    "to_bytes",
    "to_float",
    "to_fx",
    "to_int",
    "vaked_to_ugm",
    "workflow_meta",
    "workflow_to_graph",
    "write_bundle",
]
