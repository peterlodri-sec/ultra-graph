"""Quasi-regular partitioning: the level-of-detail compressor for big worlds.

A world model with 10^5 causal nodes cannot be ticked node-by-node forever. The
classical answer is a coarse graph: partition the vertices into equal blocks and
run the tick on the blocks. Szemerédi's regularity lemma says such a partition
exists for every graph, but the lemma is existential and its proof is not an
algorithm, so nothing here claims to realise it. This module does the honest,
deterministic part and reports two numbers instead of one promise:

``concentration``
    The objective actually optimised: ``sum over block pairs of
    count(a,b)^2 / (|a| * |b|)`` in Q16.16. It is large when edges live inside
    few block pairs (structure) and small when they are spread evenly (noise).
    A greedy local search, one vertex at a time under a hard block-size cap,
    increases it and keeps the best partition seen.

``defect``
    The largest deviation of any block pair's density from the graph's global
    density, in Q16.16. This is the quasi-randomness measure: a *small* defect
    means the coarse blocks look like uniform noise (a good stand-in), a large
    one means structure is being averaged away and the coarse model will lie.

The two disagree on purpose. Optimising structure raises the defect, because
structure is exactly what deviates from the global average. Reporting both keeps
the compressor from selling one as the other.

Everything is integer: counts are ``int``, densities are Q16.16 through
:func:`~ultragraph.zerocpu.fixed._round_div`, so Python and Rust agree bit for bit.
"""

from __future__ import annotations

from dataclasses import dataclass

from .fixed import ONE, _round_div

__all__ = [
    "PartitionError",
    "Partition",
    "partition",
    "density_stats",
    "concentration",
    "coarsen",
]


class PartitionError(ValueError):
    """The requested partition is not well defined for this graph."""


@dataclass(frozen=True)
class Partition:
    """A block partition plus the two measures that describe what it did."""

    blocks: int
    block_of: tuple[int, ...]
    sizes: tuple[int, ...]
    densities: tuple[int, ...]
    global_density: int
    defect: int
    concentration: int
    iterations: int

    @property
    def compression(self) -> tuple[int, int]:
        """``(k, n)`` -- the compression factor as a fraction."""
        return (self.blocks, len(self.block_of))


def _check(n: int, edges, k: int) -> list[tuple[int, int]]:
    if not isinstance(n, int) or isinstance(n, bool):
        raise PartitionError(f"n must be an int, got {type(n).__name__}")
    if n <= 0:
        raise PartitionError(f"n must be positive, got {n}")
    if not isinstance(k, int) or isinstance(k, bool):
        raise PartitionError(f"k must be an int, got {type(k).__name__}")
    if not 1 <= k <= n:
        raise PartitionError(f"need 1 <= k <= n, got k={k}, n={n}")
    out: list[tuple[int, int]] = []
    for i, edge in enumerate(edges):
        try:
            u, v = edge
        except (TypeError, ValueError):
            raise PartitionError(f"edge #{i} is not a pair: {edge!r}") from None
        u, v = int(u), int(v)
        if not (0 <= u < n and 0 <= v < n):
            raise PartitionError(f"edge #{i} out of range: {u}->{v} (n={n})")
        out.append((u, v))
    return out


def _seed_order(n: int, edges) -> list[int]:
    """Deterministic breadth-first vertex order (smallest unvisited root first).

    Seeding blocks from a traversal rather than raw index order matters: index
    order interleaves communities and single-vertex moves can then never repair
    them under a size cap. BFS order keeps neighbourhoods together and is still
    reproducible, which is the whole point.
    """
    adj: list[list[int]] = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    for row in adj:
        row.sort()
    order: list[int] = []
    seen = [False] * n
    for root in range(n):
        if seen[root]:
            continue
        seen[root] = True
        queue = [root]
        head = 0
        while head < len(queue):
            u = queue[head]
            head += 1
            order.append(u)
            for v in adj[u]:
                if not seen[v]:
                    seen[v] = True
                    queue.append(v)
    return order


def _balanced_order(n: int, k: int, order: list[int]) -> list[int]:
    """Cut a vertex order into ``k`` contiguous blocks of near-equal size.

    Contiguous, not round-robin: a traversal order puts neighbours next to each
    other, and round-robin dealing would scatter them across every block again.
    Sizes are ``n//k`` or ``n//k + 1``, so the partition stays balanced.
    """
    base, rem = divmod(n, k)
    block_of = [0] * n
    i = 0
    for b in range(k):
        for _ in range(base + (1 if b < rem else 0)):
            block_of[order[i]] = b
            i += 1
    return block_of


def _sizes(block_of: list[int], k: int) -> list[int]:
    sizes = [0] * k
    for b in block_of:
        sizes[b] += 1
    return sizes


def _counts(n: int, edges, block_of: list[int], k: int) -> list[list[int]]:
    counts = [[0] * k for _ in range(k)]
    for u, v in edges:
        counts[block_of[u]][block_of[v]] += 1
    return counts


def _pair_term(count: int, size_a: int, size_b: int) -> int:
    """``count^2 / (|a| * |b|)`` in Q16.16 -- one pair's contribution."""
    denom = size_a * size_b
    if denom == 0:
        return 0
    return _round_div(count * count * ONE, denom)


def _density(count: int, size_a: int, size_b: int) -> int:
    denom = size_a * size_b
    return _round_div(count * ONE, denom) if denom else 0


def _global_density(n: int, m: int) -> int:
    return _round_div(m * ONE, n * n) if n else 0


def _defect(counts: list[list[int]], sizes: list[int], global_density: int) -> int:
    return max(
        (
            abs(_density(counts[a][b], sizes[a], sizes[b]) - global_density)
            for a in range(len(counts))
            for b in range(len(counts))
        ),
        default=0,
    )


def _concentration(counts: list[list[int]], sizes: list[int]) -> int:
    return sum(
        _pair_term(counts[a][b], sizes[a], sizes[b])
        for a in range(len(counts))
        for b in range(len(counts))
    )


def _stats(n: int, edges, block_of, k: int) -> dict:
    sizes = _sizes(block_of, k)
    counts = _counts(n, edges, block_of, k)
    gd = _global_density(n, len(edges))
    densities = tuple(
        _density(counts[a][b], sizes[a], sizes[b]) for a in range(k) for b in range(k)
    )
    return {
        "blocks": k,
        "sizes": tuple(sizes),
        "counts": counts,
        "densities": densities,
        "global_density": gd,
        "defect": _defect(counts, sizes, gd),
        "concentration": _concentration(counts, sizes),
    }


def density_stats(n: int, edges, block_of) -> dict:
    """Densities of every block pair, the global density, and both measures."""
    block_of = list(block_of)
    if len(block_of) != n:
        raise PartitionError(f"block_of has {len(block_of)} entries, want {n}")
    if any(b < 0 for b in block_of):
        raise PartitionError("block ids must be >= 0")
    k = (max(block_of) + 1) if block_of else 1
    stats = _stats(n, list(edges), block_of, k)
    stats.pop("counts")
    return stats


def concentration(n: int, edges, block_of) -> int:
    """The structure objective for a given partition (Q16.16, higher is better)."""
    block_of = list(block_of)
    if len(block_of) != n:
        raise PartitionError(f"block_of has {len(block_of)} entries, want {n}")
    k = (max(block_of) + 1) if block_of else 1
    return _stats(n, list(edges), block_of, k)["concentration"]


def partition(n: int, edges, k: int, *, epsilon: int = 0, max_iters: int = 8) -> Partition:
    """Partition ``n`` vertices into ``k`` blocks that concentrate the edges.

    ``epsilon`` stops the search early once the *defect* drops to that value
    (raw Q16.16). The default ``0`` means "keep going while structure improves";
    pass a larger value when a merely quasi-random partition is good enough.
    ``max_iters`` bounds the greedy passes. The returned partition is the best
    ``concentration`` seen, so it is never worse than the seed partition on the
    objective it reports.

    The search is deliberately simple: for each vertex, try every other block at
    once (apply, measure, revert), then keep the best strictly improving move.
    That is ``O(passes * n * k^3)`` integer work and no randomness, so Python and
    Rust walk the identical tree of candidate moves.
    """
    edges = _check(n, edges, k)
    if not isinstance(max_iters, int) or max_iters < 0:
        raise PartitionError(f"max_iters must be a non-negative int, got {max_iters!r}")
    if epsilon < 0:
        raise PartitionError(f"epsilon must be >= 0, got {epsilon}")

    block_of = _balanced_order(n, k, _seed_order(n, edges))
    stats = _stats(n, edges, block_of, k)
    best = Partition(
        blocks=k,
        block_of=tuple(block_of),
        sizes=stats["sizes"],
        densities=stats["densities"],
        global_density=stats["global_density"],
        defect=stats["defect"],
        concentration=stats["concentration"],
        iterations=0,
    )
    if max_iters == 0 or best.defect <= epsilon:
        return best

    counts = stats["counts"]
    sizes = list(stats["sizes"])
    current = list(block_of)
    global_density = stats["global_density"]
    capacity = (n + k - 1) // k + 1

    # Counts of one vertex's incident edges per block, recomputed after each move
    # of that vertex only (cheap: O(deg u)).
    out_adj: list[list[int]] = [[] for _ in range(n)]
    in_adj: list[list[int]] = [[] for _ in range(n)]
    for u, v in edges:
        out_adj[u].append(v)
        in_adj[v].append(u)

    def apply(u: int, src: int, dst: int) -> None:
        for v in out_adj[u]:
            counts[src][current[v]] -= 1
            counts[dst][current[v]] += 1
        for w in in_adj[u]:
            counts[current[w]][src] -= 1
            counts[current[w]][dst] += 1
        sizes[src] -= 1
        sizes[dst] += 1
        current[u] = dst

    for it in range(1, max_iters + 1):
        improved = False
        for u in range(n):
            here = current[u]
            target, best_score = here, best.concentration
            for b in range(k):
                if b == here or sizes[b] >= capacity:
                    continue
                apply(u, here, b)
                score = _concentration(counts, sizes)
                apply(u, b, here)
                if score > best_score:
                    target, best_score = b, score
            if target == here:
                continue
            apply(u, here, target)
            improved = True
            best = Partition(
                blocks=k,
                block_of=tuple(current),
                sizes=tuple(sizes),
                densities=tuple(
                    _density(counts[a][b], sizes[a], sizes[b])
                    for a in range(k)
                    for b in range(k)
                ),
                global_density=global_density,
                defect=_defect(counts, sizes, global_density),
                concentration=best_score,
                iterations=it,
            )
            if best.defect <= epsilon:
                return best
        if not improved:
            break
    return best


def coarsen(n: int, edges, block_of) -> tuple[tuple[int, int, int], ...]:
    """Aggregate edges between blocks into weighted super-edges (sorted, deduped).

    Aggregation is exact: the super-edge weights sum to the original edge count,
    so a coarse tick loses resolution, not mass.
    """
    block_of = list(block_of)
    if len(block_of) != n:
        raise PartitionError(f"block_of has {len(block_of)} entries, want {n}")
    agg: dict[tuple[int, int], int] = {}
    for u, v in edges:
        key = (block_of[u], block_of[v])
        agg[key] = agg.get(key, 0) + 1
    return tuple((a, b, w) for (a, b), w in sorted(agg.items()))
