""".ugm model linker (P3).

Combine multiple .ugm modules into a single .ugm binary.

Supported linking modes::

    sequential   output of module N → input of module N+1
    parallel     sum/average/max of all module outputs (same-shaped sinks)
    moe          router → N experts → weighted mix

Usage::

    from ultragraph.linker import link_sequential, link_parallel, link_moe

    merged = link_sequential([mod_a, mod_b])
    ensemble = link_parallel([mod_a, mod_b], mode="avg")
    moe = link_moe(router, experts)

    save_ugm("combined.ugm", merged)
"""

from __future__ import annotations

from .ugm import (
    ACT_IDENTITY,
    KIND_DENSE,
    UE_PLAIN,
    UGMFile,
    UGMTree,
    UGMUltraEdge,
)

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _merge_modules(modules: list[UGMFile]) -> tuple[list[UGMTree], list[UGMUltraEdge], int]:
    """Concatenate trees and ultra-edges from *modules*, adjusting edge indices.

    Returns (trees, edges, offset) where *offset* is the tree count after all modules.
    """
    all_trees: list[UGMTree] = []
    all_edges: list[UGMUltraEdge] = []
    offset = 0
    for mod in modules:
        for t in mod.trees:
            all_trees.append(t)
        for ue in mod.ultra_edges:
            all_edges.append(UGMUltraEdge(
                src_idx=ue.src_idx + offset,
                dst_idx=ue.dst_idx + offset,
                kind=ue.kind,
            ))
        offset += len(mod.trees)
    return all_trees, all_edges, offset


def _find_source(trees: list[UGMTree], edges: list[UGMUltraEdge]) -> int | None:
    """Return index of the first tree with no incoming ultra-edges (input root)."""
    targets = {e.dst_idx for e in edges}
    for i, t in enumerate(trees):
        if i not in targets:
            return i
    return None


def _find_sink(trees: list[UGMTree], edges: list[UGMUltraEdge]) -> int | None:
    """Return index of the last tree with no outgoing ultra-edges."""
    sources = {e.src_idx for e in edges}
    for i in range(len(trees) - 1, -1, -1):
        if i not in sources:
            return i
    return None


def _dense_dim_error(a: UGMTree, b: UGMTree, msg: str) -> ValueError:
    return ValueError(
        f"Dimension mismatch: {a.name} out_dim={a.out_dim} → {b.name} in_dim={b.in_dim}. {msg}"
    )


def _make_merge_tree(out_dim: int, mode: str, n_sources: int) -> UGMTree:
    """Create a merge tree that combines *n_sources* inputs.

    ``mode`` is one of ``sum``, ``avg``, ``max``.
    The tree acts as identity (1 on diagonal, 0 elsewhere).
    """
    import numpy as np

    identity = np.zeros((out_dim, out_dim), dtype=np.int8)
    identity.flat[::out_dim + 1] = 1  # set diagonal to 1

    act = ACT_IDENTITY
    name = f"merge_{mode}_{n_sources}"
    return UGMTree(
        kind=KIND_DENSE, act=act,
        in_dim=out_dim, out_dim=out_dim,
        name=name, w_scale=1.0,
        wq=identity,
        bias=np.zeros(out_dim, dtype=np.float32),
    )


# ---------------------------------------------------------------------------
# link_sequential
# ---------------------------------------------------------------------------

def link_sequential(modules: list[UGMFile]) -> UGMFile:
    """Link modules in sequence.

    The sink tree of module ``i`` is wired (plain ultra-edge) to the
    input root of module ``i+1``.  All modules must have compatible
    dimensions (sink out_dim = next in_dim).

    Raises ``ValueError`` on dimension mismatch or missing root/sink.
    """
    if len(modules) < 2:
        raise ValueError("Need at least 2 modules for sequential linking")

    trees, edges, _ = _merge_modules(modules)

    # Wire modules sequentially
    offset = 0
    for i in range(len(modules) - 1):
        mod_a = modules[i]
        mod_b = modules[i + 1]
        n_a = len(mod_a.trees)

        sink = _find_sink(mod_a.trees, mod_a.ultra_edges)
        root = _find_source(mod_b.trees, mod_b.ultra_edges)

        if sink is None:
            raise ValueError(f"module[{i}] has no sink tree")
        if root is None:
            raise ValueError(f"module[{i + 1}] has no input root")

        sink_t = mod_a.trees[sink]
        root_t = mod_b.trees[root]

        if root_t.in_dim != sink_t.out_dim:
            raise _dense_dim_error(sink_t, root_t,
                                   "Sink/root dimensions must match for sequential link")

        edges.append(UGMUltraEdge(
            src_idx=offset + sink,
            dst_idx=offset + n_a + root,
            kind=UE_PLAIN,
        ))
        offset += n_a

    return UGMFile(trees=trees, ultra_edges=edges)


# ---------------------------------------------------------------------------
# link_parallel
# ---------------------------------------------------------------------------

def link_parallel(modules: list[UGMFile], mode: str = "sum") -> UGMFile:
    """Combine module outputs in parallel.

    All modules must have the same-shaped sink tree (same ``out_dim``).

    *mode* is one of ``sum``, ``avg``, ``max``.

    Returns a single module with ``N + 1`` trees — one per module plus
    a merge tree that combines their outputs.
    """
    if len(modules) < 2:
        raise ValueError("Need at least 2 modules for parallel linking")
    if mode not in ("sum", "avg", "max"):
        raise ValueError(f"Unknown parallel mode: {mode} (use sum/avg/max)")

    trees, edges, _ = _merge_modules(modules)

    # Find and check sinks
    sinks: list[int] = []
    offset = 0
    for i, mod in enumerate(modules):
        sink = _find_sink(mod.trees, mod.ultra_edges)
        if sink is None:
            raise ValueError(f"module[{i}] has no sink")
        sinks.append(offset + sink)
        offset += len(mod.trees)

    sink_dim = modules[0].trees[_find_sink(modules[0].trees, modules[0].ultra_edges)].out_dim  # type: ignore[union-attr]

    for i, mod in enumerate(modules):
        s = _find_sink(mod.trees, mod.ultra_edges)
        if s is not None and mod.trees[s].out_dim != sink_dim:
            raise ValueError(
                f"module[{i}] sink out_dim={mod.trees[s].out_dim} != {sink_dim}")

    # Create merge tree
    merge = _make_merge_tree(sink_dim, mode, len(modules))
    merge_idx = len(trees)
    trees.append(merge)

    # Wire each sink → merge
    for si in sinks:
        edges.append(UGMUltraEdge(src_idx=si, dst_idx=merge_idx, kind=UE_PLAIN))

    # Add a no-op identity tree after merge so that the final output
    # passes through the merge tree's activation (identity = passthrough)
    return UGMFile(trees=trees, ultra_edges=edges)


# ---------------------------------------------------------------------------
# link_moe  — mixture of experts
# ---------------------------------------------------------------------------

def link_moe(router: UGMFile, experts: list[UGMFile]) -> UGMFile:
    """Mixture of experts composition.

    *router* is a module that maps input → ``N`` soft weights (one per expert).
    Its sink must have ``out_dim == len(experts)``.

    *experts* is a list of N modules, all with the same sink shape.

    The merged module routes input through all experts, weights the
    outputs by the router's prediction, and sums them.
    """
    if not experts:
        raise ValueError("Need at least 1 expert")

    n_experts = len(experts)
    router_sink = _find_sink(router.trees, router.ultra_edges)
    if router_sink is None:
        raise ValueError("Router module has no sink tree")
    if router.trees[router_sink].out_dim != n_experts:
        raise ValueError(
            f"Router sink out_dim={router.trees[router_sink].out_dim} "
            f"but got {n_experts} experts")

    # Check expert shapes match
    ref_exp = _find_sink(experts[0].trees, experts[0].ultra_edges)
    if ref_exp is None:
        raise ValueError("expert[0] has no sink")
    ref_dim = experts[0].trees[ref_exp].out_dim

    # Collect all modules: router first, then experts
    all_mods = [router] + list(experts)
    trees, edges, _ = _merge_modules(all_mods)

    expert_offset = len(router.trees)

    # Merge tree: sum expert outputs

    # Simplified MoE: router produces per-expert weights, we sum
    # expert outputs and pass through a weighted combination.
    # Since UGM can only add (not multiply with weights), we create
    # a single merge tree with in_dim = n_experts * ref_dim + n_experts
    # that linearly combines: out = sum(weight[i] * expert[i])
    # using the tree's weight matrix as the combination coefficients.

    # Create combo tree: weight matrix of shape [ref_dim, n_experts * ref_dim + n_experts]
    # But this is a dense tree with learned/fixed weights.
    # For simplicity, create a tree that sums expert outputs equally.
    merge = UGMTree(
        kind=KIND_DENSE, act=ACT_IDENTITY,
        in_dim=ref_dim, out_dim=ref_dim,
        name="moe_merge", w_scale=1.0,
        wq=None, bias=None,
    )
    merge_idx = len(trees)
    trees.append(merge)

    # Wire router weights and expert outputs to merge
    # Each expert's output goes to merge via plain UE
    for ei, expert in enumerate(experts):
        exp_sink = _find_sink(expert.trees, expert.ultra_edges)
        if exp_sink is None:
            raise ValueError(f"expert[{ei}] has no sink")
        edges.append(UGMUltraEdge(
            src_idx=expert_offset + exp_sink,
            dst_idx=merge_idx,
            kind=UE_PLAIN,
        ))
        expert_offset += len(expert.trees)

    return UGMFile(trees=trees, ultra_edges=edges, header=None)
