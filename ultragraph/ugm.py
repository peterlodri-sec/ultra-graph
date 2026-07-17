""".ugm — Ultragraph Module binary format.

The `.ugm` format turns the byte-graph into a standalone executable binary.
A .ugm file bundles tree definitions, ultra-edge wiring, and ternary weight
data into a single file that any runtime can load and execute.

Spec v0.1 (2026-07-12)
-----------------------

Header (32 bytes)::

    offset  size  field
    ------  ----  -----
    0       4     magic           -- b"\\x55\\x47\\x4D\\x01" (UGM1)
    4       2     version         -- uint16, currently 1
    6       2     flags           -- uint16 (bit 0: packed weights)
    8       4     n_trees         -- uint32
    12      4     n_ultra_edges   -- uint32
    16      4     trees_offset    -- uint32, byte offset to tree table
    20      4     ue_offset       -- uint32, byte offset to ultra-edge table
    24      4     weights_offset  -- uint32, byte offset to weight data
    28      4     weights_size    -- uint32, byte count of weight data

Tree table (variable per tree)::

    offset  size  field
    ------  ----  -----
    0       1     kind            -- 0=dense, 1=sparse
    1       1     act             -- 0=none, 1=relu, 2=identity, 3=sigmoid, 4=tanh
    2       4     in_dim          -- uint32 (0 for sparse)
    6       4     out_dim         -- uint32 (n_nodes for dense)
    10      2     name_len        -- uint16
    12      N     name            -- UTF-8 bytes
    12+N    4     w_scale         -- float32
    = 16 + name_len bytes

Ultra-edge table (9 bytes each)::

    0       4     src_idx         -- uint32, index into tree table
    4       4     dst_idx         -- uint32
    8       1     kind            -- 0=plain, 1=residual

Weight data per dense tree::

    - wq: out_dim * in_dim bytes (int8)
    - bias: out_dim * 4 bytes (float32)

Optional segments follow weight data. Each segment::

    0       4     seg_type        -- uint32 (1=history, 2=metadata)
    4       4     seg_len         -- uint32
    8       N     seg_data

Usage::

    from ultragraph.ugm import load_ugm, save_ugm, UGMHeader

    module = load_ugm("model.ugm")         # -> UGMModule
    module.save("model.ugm")               # write back
    module.run(input_array)                # interpret
"""


import json
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO

import numpy as np

# -- constants ----------------------------------------------------------------
_MAGIC = b"\x55\x47\x4D\x01"  # UGM1
_VERSION = 1
_FLAG_PACKED = 1 << 0

KIND_DENSE = 0
KIND_SPARSE = 1

ACT_NONE = 0
ACT_RELU = 1
ACT_IDENTITY = 2
ACT_SIGMOID = 3
ACT_TANH = 4
_ACT_NAMES = {0: "none", 1: "relu", 2: "identity", 3: "sigmoid", 4: "tanh"}

UE_PLAIN = 0
UE_RESIDUAL = 1

SEG_HISTORY = 1
SEG_METADATA = 2


# -- data types ---------------------------------------------------------------
@dataclass(slots=True)
class UGMTree:
    """A tree definition in the .ugm format."""
    kind: int = KIND_DENSE       # 0=dense, 1=sparse
    act: int = ACT_RELU
    in_dim: int = 0
    out_dim: int = 0
    name: str = ""
    w_scale: float = 1.0
    # weight data (loaded from / written to the weights segment)
    wq: np.ndarray | None = None      # int8[out_dim, in_dim] (dense)
    bias: np.ndarray | None = None    # float32[out_dim] (dense)
    # sparse edge data
    esrc: list[int] = field(default_factory=list)
    edst: list[int] = field(default_factory=list)
    eval: list[int] = field(default_factory=list)


@dataclass(slots=True)
class UGMUltraEdge:
    src_idx: int = 0
    dst_idx: int = 0
    kind: int = UE_PLAIN


@dataclass(slots=True)
class UGMHeader:
    version: int = _VERSION
    flags: int = 0
    n_trees: int = 0
    n_ultra_edges: int = 0
    trees_offset: int = 32
    ue_offset: int = 0
    weights_offset: int = 0
    weights_size: int = 0


@dataclass(slots=True)
class UGMHistorySegment:
    """Optional history segment — per-node activation ring buffers."""
    n_nodes: int = 0
    depth: int = 100
    buffer: np.ndarray | None = None  # int8[n_nodes, depth]


@dataclass(slots=True)
class UGMMetadata:
    """Optional metadata segment — JSON blob."""
    data: dict = field(default_factory=dict)


@dataclass(slots=True)
class UGMFile:
    """A complete .ugm module loaded in memory."""
    header: UGMHeader = field(default_factory=UGMHeader)
    trees: list[UGMTree] = field(default_factory=list)
    ultra_edges: list[UGMUltraEdge] = field(default_factory=list)
    history: UGMHistorySegment | None = None
    metadata: UGMMetadata | None = None

    @property
    def sink_idx(self) -> int:
        """Index of the sink tree (no outgoing ultra-edge)."""
        src_set = {e.src_idx for e in self.ultra_edges}
        for i in range(len(self.trees) - 1, -1, -1):
            if i not in src_set:
                return i
        return len(self.trees) - 1

    # -- forward pass (minimal interpreter) -----------------------------------
    def run(self, x: np.ndarray) -> np.ndarray:
        """Interpret the .ugm as a forward pass.

        ``x`` is float32 input ``[B, in_dim]``.
        Returns float32 output from the sink tree.
        """
        outputs: dict[int, np.ndarray] = {}
        remaining = list(range(len(self.trees)))
        incoming: dict[int, list[UGMUltraEdge]] = {i: [] for i in range(len(self.trees))}
        for ue in self.ultra_edges:
            incoming[ue.dst_idx].append(ue)

        while remaining:
            progressed = False
            still = []
            for ti in remaining:
                edges = incoming[ti]
                if not all(e.src_idx in outputs for e in edges):
                    still.append(ti)
                    continue
                tree = self.trees[ti]
                plain_srcs = [e.src_idx for e in edges if e.kind == UE_PLAIN]
                if plain_srcs:
                    inp = outputs[plain_srcs[0]]
                    for s in plain_srcs[1:]:
                        inp = inp + outputs[s]
                else:
                    inp = x
                out = self._forward_tree(tree, inp)
                for e in edges:
                    if e.kind == UE_RESIDUAL:
                        out = out + outputs[e.src_idx]
                outputs[ti] = out
                progressed = True
            if not progressed:
                raise ValueError("cycle detected in .ugm graph")
            remaining = still
        return outputs[self.sink_idx]

    @staticmethod
    def _forward_tree(tree: UGMTree, x: np.ndarray) -> np.ndarray:
        """Run one tree's forward pass."""
        if tree.kind == KIND_DENSE and tree.wq is not None:
            wq = tree.wq.astype(np.float32) * tree.w_scale
            bias = tree.bias if tree.bias is not None else 0
            out = x @ wq.T + bias
            act = _ACT_NAMES.get(tree.act, "none")
            if act == "relu":
                out = np.maximum(out, 0)
            elif act == "sigmoid":
                out = 1.0 / (1.0 + np.exp(-out))
            elif act == "tanh":
                out = np.tanh(out)
            return out.astype(np.float32)
        raise NotImplementedError(f"Tree kind={tree.kind} forward not implemented")


# -- binary serialization -----------------------------------------------------
def _write_tree(fh: BinaryIO, tree: UGMTree) -> None:
    name_bytes = tree.name.encode("utf-8")
    fh.write(struct.pack("<BB", tree.kind, tree.act))
    fh.write(struct.pack("<II", tree.in_dim, tree.out_dim))
    fh.write(struct.pack("<H", len(name_bytes)))
    fh.write(name_bytes)
    fh.write(struct.pack("<f", tree.w_scale))


def _read_tree(fh: BinaryIO) -> UGMTree:
    kind, act = struct.unpack("<BB", fh.read(2))
    in_dim, out_dim = struct.unpack("<II", fh.read(8))
    name_len, = struct.unpack("<H", fh.read(2))
    name = fh.read(name_len).decode("utf-8")
    w_scale, = struct.unpack("<f", fh.read(4))
    return UGMTree(kind=kind, act=act, in_dim=in_dim, out_dim=out_dim,
                   name=name, w_scale=w_scale)


def save_ugm(path: str | Path, module: UGMFile, packed: bool = False) -> None:
    """Serialize a UGMFile to a .ugm binary."""
    path = Path(path)
    flags = _FLAG_PACKED if packed else 0

    # Build tree table + weight data in memory to compute offsets
    tree_data = bytearray()
    weight_data = bytearray()
    weight_sizes: list[int] = []

    for tree in module.trees:
        buf = bytearray()
        name_bytes = tree.name.encode("utf-8")
        buf += struct.pack("<BB", tree.kind, tree.act)
        buf += struct.pack("<II", tree.in_dim, tree.out_dim)
        buf += struct.pack("<H", len(name_bytes))
        buf += name_bytes
        buf += struct.pack("<f", tree.w_scale)
        tree_data += buf

        # Weight data
        if tree.kind == KIND_DENSE and tree.wq is not None:
            w_bytes = tree.wq.tobytes()
            b_bytes = tree.bias.tobytes() if tree.bias is not None else b"\x00" * (tree.out_dim * 4)
            weight_data += w_bytes + b_bytes
            weight_sizes.append(len(w_bytes) + len(b_bytes))

    ue_data = bytearray()
    for ue in module.ultra_edges:
        ue_data += struct.pack("<IIB", ue.src_idx, ue.dst_idx, ue.kind)

    # Optional segments
    seg_data = bytearray()
    if module.history is not None:
        h = module.history
        buf = h.buffer.tobytes() if h.buffer is not None else b""
        seg_data += struct.pack("<II", SEG_HISTORY, len(buf))
        seg_data += buf
    if module.metadata is not None:
        raw = json.dumps(module.metadata.data).encode("utf-8")
        seg_data += struct.pack("<II", SEG_METADATA, len(raw))
        seg_data += raw

    # Layout
    hdr = UGMHeader(flags=flags, n_trees=len(module.trees),
                    n_ultra_edges=len(module.ultra_edges))
    hdr.trees_offset = 32
    hdr.ue_offset = hdr.trees_offset + len(tree_data)
    hdr.weights_offset = hdr.ue_offset + len(ue_data)
    hdr.weights_size = len(weight_data)

    with open(path, "wb") as fh:
        # Header
        fh.write(_MAGIC)
        fh.write(struct.pack("<HHI", hdr.version, hdr.flags, hdr.n_trees))
        fh.write(struct.pack("<I", hdr.n_ultra_edges))
        fh.write(struct.pack("<IIII", hdr.trees_offset, hdr.ue_offset,
                             hdr.weights_offset, hdr.weights_size))
        # Tables
        fh.write(tree_data)
        fh.write(ue_data)
        fh.write(weight_data)
        # Optional segments
        fh.write(seg_data)


def load_ugm(path: str | Path) -> UGMFile:
    """Deserialize a .ugm binary into a UGMFile."""
    path = Path(path)
    with open(path, "rb") as fh:
        magic = fh.read(4)
        if magic != _MAGIC:
            raise ValueError(f"Not a .ugm file: magic={magic.hex()}")
        version, flags, n_trees = struct.unpack("<HHI", fh.read(8))
        if version != _VERSION:
            raise ValueError(f"unsupported .ugm version {version}, expected {_VERSION}")
        n_ue, = struct.unpack("<I", fh.read(4))
        toff, uoff, woff, wsize = struct.unpack("<IIII", fh.read(16))

        # Read trees
        fh.seek(toff)
        trees = []
        for _ in range(n_trees):
            trees.append(_read_tree(fh))

        # Read ultra-edges
        fh.seek(uoff)
        ue_list = []
        for _ in range(n_ue):
            src, dst, kind = struct.unpack("<IIB", fh.read(9))
            ue_list.append(UGMUltraEdge(src_idx=src, dst_idx=dst, kind=kind))

        # Read weight data
        fh.seek(woff)
        for tree in trees:
            if tree.kind == KIND_DENSE:
                n_bytes = tree.out_dim * tree.in_dim
                tree.wq = np.frombuffer(fh.read(n_bytes), dtype=np.int8).reshape(tree.out_dim, tree.in_dim)
                tree.bias = np.frombuffer(fh.read(tree.out_dim * 4), dtype=np.float32)

        # Read optional segments
        remaining = fh.read()
        pos = 0
        history = None
        metadata = None
        while pos + 8 <= len(remaining):
            seg_type, seg_len = struct.unpack("<II", remaining[pos:pos + 8])
            pos += 8
            if pos + seg_len > len(remaining):
                break
            seg_data = remaining[pos:pos + seg_len]
            pos += seg_len
            if seg_type == SEG_HISTORY and seg_len > 0:
                n_nodes = len(trees)  # infer
                if n_nodes > 0 and seg_len % n_nodes == 0:
                    depth = seg_len // n_nodes
                    if depth > 0:
                        buf = np.frombuffer(seg_data, dtype=np.int8).reshape(n_nodes, depth)
                        history = UGMHistorySegment(n_nodes=n_nodes, depth=depth, buffer=buf)
            elif seg_type == SEG_METADATA:
                metadata = UGMMetadata(data=json.loads(seg_data.decode("utf-8")))

    return UGMFile(
        header=UGMHeader(version=version, flags=flags, n_trees=n_trees, n_ultra_edges=n_ue,
                         trees_offset=toff, ue_offset=uoff, weights_offset=woff, weights_size=wsize),
        trees=trees, ultra_edges=ue_list, history=history, metadata=metadata,
    )


# -- conversion helpers -------------------------------------------------------
def from_ultragraph(ug, name: str = "model") -> UGMFile:
    """Convert an ultragraph UltraGraph or GPT to a UGMFile."""
    from .core import UltraGraph  # lazy to avoid circular import
    from .model import GPT

    trees: list[UGMTree] = []
    ue_list: list[UGMUltraEdge] = []
    act_map = {"none": ACT_NONE, "relu": ACT_RELU, "identity": ACT_IDENTITY, "sigmoid": ACT_SIGMOID, "tanh": ACT_TANH}

    if isinstance(ug, GPT):
        dense = ug._dense_trees()
        idx_map = {}
        for i, t in enumerate(dense):
            ugt = UGMTree(kind=KIND_DENSE, act=act_map.get(t.act, ACT_NONE),
                          in_dim=t.in_dim, out_dim=t.out_dim,
                          name=t.name, w_scale=float(t.w_scale),
                          wq=np.asarray(t.wq).copy() if t.wq is not None else None,
                          bias=t.adhoc["bias"].data.copy() if "bias" in t.adhoc else None)
            trees.append(ugt)
            idx_map[id(t)] = i

        # Wire blocks sequentially
        for bi in range(len(dense) - 1):
            ue_list.append(UGMUltraEdge(src_idx=bi, dst_idx=bi + 1, kind=UE_PLAIN))

    elif isinstance(ug, UltraGraph):
        id_map = {id(t): i for i, t in enumerate(ug.trees)}
        for t in ug.trees:
            ugt = UGMTree(kind=KIND_DENSE if t.kind == "dense" else KIND_SPARSE,
                          act=act_map.get(t.act, ACT_NONE),
                          in_dim=t.in_dim or 0, out_dim=t.out_dim or 0,
                          name=t.name, w_scale=float(t.w_scale),
                          wq=np.asarray(t.wq).copy() if t.wq is not None else None)
            trees.append(ugt)
        for ue in ug.ultra_edges:
            ue_list.append(UGMUltraEdge(src_idx=id_map[id(ue.src)], dst_idx=id_map[id(ue.dst)],
                                        kind=UE_PLAIN if ue.kind == "plain" else UE_RESIDUAL))
    else:
        raise TypeError(f"Cannot convert {type(ug).__name__} to .ugm")

    return UGMFile(
        header=UGMHeader(n_trees=len(trees), n_ultra_edges=len(ue_list),
                         trees_offset=32, weights_offset=0),
        trees=trees, ultra_edges=ue_list,
    )


def to_ultragraph(module: UGMFile):
    """Convert a UGMFile back to an ultragraph UltraGraph."""
    from .core import Tree, UltraGraph  # lazy

    ug = UltraGraph("ugm")
    for t in module.trees:
        tree = Tree.dense(t.in_dim, t.out_dim, name=t.name, act=_ACT_NAMES.get(t.act, "none"))
        if t.wq is not None:
            tree.wq = t.wq.copy()
            tree.w_scale = t.w_scale
            tree.adhoc["w_master"] = None  # deployed mode
        ug.add(tree)
    for ue in module.ultra_edges:
        kind = "plain" if ue.kind == UE_PLAIN else "residual"
        ug.wire(ug.trees[ue.src_idx], ug.trees[ue.dst_idx], kind)
    return ug
