"""Byte-exact save/load of an ultra-graph (numpy .npz container + JSON header)."""

import json

import numpy as np

from .autograd import Tensor
from .core import Embedding, Tree, UltraGraph
from .pack import pack_ternary, unpack_ternary


def save(ug: UltraGraph, path: str, include_masters: bool = True, packed: bool = False) -> None:
    """Save an ultra-graph. With ``packed=True``, dense ternary weights are stored
    bit-packed (5 values/byte) for ~5x smaller files; ``load`` unpacks transparently."""
    meta = {"name": ug.name, "trees": [], "ultra_edges": [], "modules": []}
    arrays: dict[str, np.ndarray] = {}

    for idx, t in enumerate(ug.trees):
        entry = {
            "kind": t.kind,
            "name": t.name,
            "n_nodes": t.n_nodes,
            "in_dim": t.in_dim,
            "out_dim": t.out_dim,
            "act": t.act,
            "w_scale": float(t.w_scale),
        }
        meta["trees"].append(entry)
        arrays[f"t{idx}_nodes"] = t.nodes
        if t.kind == "dense":
            if packed:
                arrays[f"t{idx}_wq"] = pack_ternary(t.wq)
                entry["wq_packed"] = True
                entry["wq_shape"] = list(t.wq.shape)
            else:
                arrays[f"t{idx}_wq"] = t.wq
            arrays[f"t{idx}_bias"] = t.adhoc["bias"].data
            if include_masters:
                arrays[f"t{idx}_wmaster"] = t.adhoc["w_master"].data
        else:
            arrays[f"t{idx}_esrc"] = t.edge_src
            arrays[f"t{idx}_edst"] = t.edge_dst
            arrays[f"t{idx}_eval"] = t.edge_val

    tree_index = {id(t): i for i, t in enumerate(ug.trees)}
    for e in ug.ultra_edges:
        meta["ultra_edges"].append(
            {"src": tree_index[id(e.src)], "dst": tree_index[id(e.dst)], "kind": e.kind}
        )

    emb_count = 0
    for m in ug.modules:
        if isinstance(m, Embedding):
            key = f"m{emb_count}_table"
            meta["modules"].append(
                {"type": "embedding", "vocab": m.vocab, "dim": m.dim, "name": m.name, "array": key}
            )
            arrays[key] = m.table.data
            emb_count += 1

    with open(path, "wb") as f:
        np.savez(f, __meta__=np.array(json.dumps(meta)), **arrays)


def load(path: str) -> UltraGraph:
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data["__meta__"]))
        ug = UltraGraph(meta["name"])

        for idx, entry in enumerate(meta["trees"]):
            if entry["kind"] == "dense":
                t = Tree.dense(entry["in_dim"], entry["out_dim"], name=entry["name"], act=entry["act"])
                if entry.get("wq_packed"):
                    shape = tuple(entry["wq_shape"])
                    t.wq = unpack_ternary(data[f"t{idx}_wq"], int(np.prod(shape))).reshape(shape)
                else:
                    t.wq = data[f"t{idx}_wq"]
                t.w_scale = float(entry["w_scale"])
                t.adhoc["bias"] = Tensor(data[f"t{idx}_bias"], requires_grad=True)
                if f"t{idx}_wmaster" in data.files:
                    t.adhoc["w_master"] = Tensor(data[f"t{idx}_wmaster"], requires_grad=True)
            else:
                t = Tree(entry["n_nodes"], name=entry["name"])
                t._esrc = data[f"t{idx}_esrc"].tolist()
                t._edst = data[f"t{idx}_edst"].tolist()
                t._eval = data[f"t{idx}_eval"].tolist()
            t.nodes[:] = data[f"t{idx}_nodes"]
            ug.add(t)

        for me in meta["modules"]:
            if me["type"] == "embedding":
                emb = Embedding(me["vocab"], me["dim"], name=me["name"])
                emb.table = Tensor(data[me["array"]], requires_grad=True)
                ug.register(emb)

        for ue in meta["ultra_edges"]:
            ug.wire(ug.trees[ue["src"]], ug.trees[ue["dst"]], ue["kind"])

        return ug


def save_params(modules, path: str, meta: dict | None = None) -> None:
    """Save the fp32 parameters of an arbitrary list of modules (anything exposing
    ``parameters() -> list[Tensor]``) — covers Attention/MoE/RMSNorm/LayerNorm/
    Embedding/Tree, which the graph save/load does not fully serialize. Restore with
    ``load_params`` onto the same architecture.

    If ``meta`` is given it is stored (JSON) under the ``__meta__`` key so loaders can
    recover architecture; omitted by default so generic callers stay byte-compatible."""
    arrays: dict[str, np.ndarray] = {}
    for i, m in enumerate(modules):
        params = m.parameters() if hasattr(m, "parameters") else []
        for j, p in enumerate(params):
            arrays[f"m{i}_p{j}"] = p.data
    if meta is not None:
        arrays["__meta__"] = np.array(json.dumps(meta))
    with open(path, "wb") as f:
        np.savez(f, **arrays)


def load_params(modules, path: str) -> None:
    """Load parameters saved by ``save_params`` into the same module list (in place),
    then re-quantize any module exposing ``requantize()``."""
    with np.load(path, allow_pickle=False) as data:
        for i, m in enumerate(modules):
            params = m.parameters() if hasattr(m, "parameters") else []
            for j, p in enumerate(params):
                key = f"m{i}_p{j}"
                if key in data.files:
                    p.data[...] = data[key]
    for m in modules:
        r = getattr(m, "requantize", None)
        if callable(r):
            r()
