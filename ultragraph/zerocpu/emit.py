"""Compile a :class:`SpacetimeGraph` into the ``UGMZ`` world artifact, and read it back.

``UGMZ`` is the *compiled* form of the world model: a flat, fixed-width,
little-endian file that the Rust ``zerocpu`` runtime (``rust/zerocpu/src/bin/world.rs``)
reads and ticks without a Python process in the loop. ``zerocpu::ternary::dot_i8``
does the per-node accumulation, so on aarch64 the tick runs on the NEON kernel.

It is a sibling of the existing ``.ugm`` byte-graph format (see
:mod:`ultragraph.ugm`), not a replacement: ``.ugm`` stores a sparse ``Tree``,
``.ugmz`` stores a spacetime world (ternary node states, Q16.16 positions,
causal edges, tick count).

Layout v1 -- every field little-endian, every record 24 bytes::

    header 32 bytes: magic "UGMZ" | version u16 | flags u16 | n_nodes u32
                     | n_edges u32 | pos_dim u8 | reserved[7] | tick_count u64
    node   24 bytes: state i8 | pad[3] | pos[pos_dim] i32 | reserved u64
    edge   24 bytes: src u32 | dst u32 | dt u64 | weight i8 | pad[7]

The format is deliberately boring: no floats, no varints, no compression, no
endianness ambiguity. Two runs of ``emit_ugm`` on equal graphs are byte-identical,
which is what makes the artifact diffable and the fingerprint reproducible.
"""

from __future__ import annotations

import struct

from .spacetime import INT8_MAX, INT8_MIN, SpacetimeGraph

__all__ = [
    "MAGIC",
    "VERSION",
    "HEADER_SIZE",
    "NODE_RECORD",
    "EDGE_RECORD",
    "emit_ugm",
    "to_bytes",
    "from_bytes",
    "load_ugm",
    "state_hex",
    "pos_hex",
]

MAGIC = b"UGMZ"
VERSION = 1
HEADER_SIZE = 32
NODE_RECORD = 24
EDGE_RECORD = 24

_HEADER = struct.Struct("<4sHHIIB7xQ")
_NODE = struct.Struct("<b3xiiiQ")
_EDGE = struct.Struct("<IIQb7x")

assert _HEADER.size == HEADER_SIZE
assert _NODE.size == NODE_RECORD
assert _EDGE.size == EDGE_RECORD


def _pos_dim(graph: SpacetimeGraph) -> int:
    """All nodes must share one position width; ``UGMZ`` records are fixed-width."""
    widths = {len(p) for p in graph._pos}
    if not widths:
        return 0
    if len(widths) != 1:
        raise ValueError(f"ragged positions are not encodable, widths={sorted(widths)}")
    dim = widths.pop()
    if dim > 3:
        # the record reserves exactly three i32 slots for position
        raise ValueError(f"at most 3 position components are encodable, got {dim}")
    return dim


def to_bytes(graph: SpacetimeGraph, pad_pos: bool = True) -> bytes:
    """Serialize ``graph`` to ``UGMZ`` v1.

    With ``pad_pos`` the position slots are zero-filled up to three components so a
    reader can always unpack three i32 values; ``pos_dim`` in the header still
    records how many are meaningful.
    """
    dim = _pos_dim(graph)
    if dim == 0 and graph._pos:
        raise ValueError("unreachable: non-empty graph with no position width")
    chunks = [
        _HEADER.pack(MAGIC, VERSION, 0, len(graph._state), len(graph._edges), dim, graph.tick_count)
    ]
    for state, pos in zip(graph._state, graph._pos):
        if not INT8_MIN <= state <= INT8_MAX:
            raise ValueError(f"node state outside int8: {state}")
        xyz = tuple(pos) + (0,) * (3 - len(pos))
        chunks.append(_NODE.pack(state, xyz[0], xyz[1], xyz[2], 0))
    for edge in graph._edges:
        chunks.append(_EDGE.pack(edge.src, edge.dst, edge.dt, edge.weight))
    return b"".join(chunks)


def from_bytes(data: bytes, *, name: str = "spacetime") -> SpacetimeGraph:
    """Parse ``UGMZ`` v1 back into a :class:`SpacetimeGraph`.

    Raises ``ValueError`` on any structural mismatch; never silently truncates.
    """
    if len(data) < HEADER_SIZE:
        raise ValueError(f"truncated header: {len(data)} bytes")
    magic, version, flags, n_nodes, n_edges, dim, ticks = _HEADER.unpack_from(data, 0)
    if magic != MAGIC:
        raise ValueError(f"bad magic {magic!r}, want {MAGIC!r}")
    if version != VERSION:
        raise ValueError(f"unsupported version {version}, want {VERSION}")
    if flags != 0:
        raise ValueError(f"unsupported flags {flags:#06x}")
    if dim > 3:
        raise ValueError(f"pos_dim {dim} exceeds 3")
    want = HEADER_SIZE + n_nodes * NODE_RECORD + n_edges * EDGE_RECORD
    if len(data) != want:
        raise ValueError(f"length {len(data)} != expected {want}")

    graph = SpacetimeGraph(name=name)
    off = HEADER_SIZE
    for _ in range(n_nodes):
        state, x, y, z, _reserved = _NODE.unpack_from(data, off)
        graph.add_node(state, tuple((x, y, z)[:dim]))
        off += NODE_RECORD
    for _ in range(n_edges):
        src, dst, dt, weight = _EDGE.unpack_from(data, off)
        if not (0 <= src < n_nodes and 0 <= dst < n_nodes):
            raise ValueError(f"edge references missing node: {src}->{dst}")
        graph.add_edge(src, dst, weight, dt)
        off += EDGE_RECORD
    graph.tick_count = ticks
    return graph


def emit_ugm(graph: SpacetimeGraph, path) -> str:
    """Write the ``UGMZ`` artifact to ``path`` and return the path as ``str``."""
    blob = to_bytes(graph)
    path = str(path)
    with open(path, "wb") as fh:
        fh.write(blob)
    return path


def load_ugm(path, *, name: str = "spacetime") -> SpacetimeGraph:
    """Read a ``UGMZ`` artifact from disk."""
    with open(path, "rb") as fh:
        return from_bytes(fh.read(), name=name)


def state_hex(graph: SpacetimeGraph) -> str:
    """Node states as two hex digits each, in node order (what ``world`` prints)."""
    return bytes(s & 0xFF for s in graph._state).hex()


def pos_hex(graph: SpacetimeGraph) -> str:
    """Positions as little-endian i32 hex in node order, three components each."""
    out = bytearray()
    for pos in graph._pos:
        xyz = tuple(pos) + (0,) * (3 - len(pos))
        out += struct.pack("<iii", *xyz)
    return bytes(out).hex()
