"""A discrete space-time world model carried by a ternary causal graph.

The ultra-graph is already a graph, so the world model does not need a second
representation: nodes carry an int8 state plus a position, edges carry a ternary
weight plus a time delay ``dt >= 0``, and :meth:`SpacetimeGraph.tick` advances
the whole world by one simultaneous integer update.

Properties worth being precise about:

* **Deterministic** -- the update is ``state' = clamp(state + sum(w * src))``
  using Python ints only.  Two runs on different machines produce identical
  bytes; there is no float, no reduction order, no SIMD reassociation.
* **Reversible only on demand** -- saturation at the int8 boundary destroys
  information, so the forward map is not bijective.  ``tick(remember=True)``
  keeps the previous state so :meth:`untick` can restore it exactly; that is
  memory-based reversal, not a claim of physical unitarity.
* **Causal** -- an edge with delay ``dt`` means information from ``src`` reaches
  ``dst`` no earlier than ``dt`` ticks later, which is what :meth:`lightcone`
  computes.
"""

from __future__ import annotations

import hashlib
import heapq
from dataclasses import dataclass

from .fixed import fx_dist, saturate

INT8_MIN = -128
INT8_MAX = 127

__all__ = ["CausalEdge", "SpacetimeGraph", "INT8_MIN", "INT8_MAX"]


@dataclass(frozen=True)
class CausalEdge:
    """One directed, delay-carrying, ternary-weighted causal link."""

    src: int
    dst: int
    weight: int = 1
    dt: int = 1

    def __post_init__(self) -> None:
        if self.weight not in (-1, 0, 1):
            raise ValueError(f"causal weight must be ternary, got {self.weight}")
        if self.dt < 0:
            raise ValueError(f"causal delay must be >= 0, got {self.dt}")


class SpacetimeGraph:
    """Nodes = events, edges = causal links, :meth:`tick` = one integer time step."""

    def __init__(self, name: str = "spacetime") -> None:
        self.name = name
        self.tick_count = 0
        self._state: list[int] = []
        self._pos: list[tuple[int, ...]] = []
        self._edges: list[CausalEdge] = []
        self._past: list[tuple[int, ...]] = []
        self._incoming: list[tuple[int, int, int]] | None = None

    # -- construction ---------------------------------------------------------
    def add_node(self, state: int = 0, pos: tuple[int, ...] = (0, 0, 0)) -> int:
        """Append a node and return its index.  ``pos`` is raw Q16.16 by convention."""
        self._state.append(saturate(state, INT8_MIN, INT8_MAX))
        self._pos.append(tuple(int(c) for c in pos))
        self._incoming = None
        return len(self._state) - 1

    def add_edge(self, src: int, dst: int, weight: int = 1, dt: int = 1) -> CausalEdge:
        edge = CausalEdge(int(src), int(dst), int(weight), int(dt))
        self._check_index(edge.src)
        self._check_index(edge.dst)
        self._edges.append(edge)
        self._incoming = None
        return edge

    def remove_edge(self, src: int, dst: int) -> int:
        """Drop every edge ``src -> dst``; returns how many were removed."""
        before = len(self._edges)
        self._edges = [e for e in self._edges if not (e.src == src and e.dst == dst)]
        self._incoming = None
        return before - len(self._edges)

    def find_edge(self, src: int, dst: int) -> CausalEdge | None:
        for e in self._edges:
            if e.src == src and e.dst == dst:
                return e
        return None

    def set_edge_weight(self, src: int, dst: int, weight: int) -> bool:
        """Rewrite the first ``src -> dst`` weight in place; False if no such edge."""
        if weight not in (-1, 0, 1):
            raise ValueError(f"causal weight must be ternary, got {weight}")
        for i, e in enumerate(self._edges):
            if e.src == src and e.dst == dst:
                self._edges[i] = CausalEdge(e.src, e.dst, weight, e.dt)
                self._incoming = None
                return True
        return False

    # -- introspection --------------------------------------------------------
    @property
    def n_nodes(self) -> int:
        return len(self._state)

    @property
    def n_edges(self) -> int:
        return len(self._edges)

    @property
    def edges(self) -> tuple[CausalEdge, ...]:
        return tuple(self._edges)

    def state(self) -> tuple[int, ...]:
        return tuple(self._state)

    def set_state(self, idx: int, value: int) -> None:
        self._check_index(idx)
        self._state[idx] = saturate(value, INT8_MIN, INT8_MAX)

    def pos_of(self, idx: int) -> tuple[int, ...]:
        self._check_index(idx)
        return self._pos[idx]

    def set_pos(self, idx: int, pos: tuple[int, ...]) -> None:
        self._check_index(idx)
        self._pos[idx] = tuple(int(c) for c in pos)

    def distance(self, a: int, b: int) -> int:
        """Raw Q16.16 Euclidean distance between two node positions."""
        return fx_dist(self.pos_of(a), self.pos_of(b))

    # -- time -----------------------------------------------------------------
    def tick(self, n: int = 1, remember: bool = True) -> tuple[int, ...]:
        """Advance ``n`` simultaneous integer steps and return the new state.

        ``state'[i] = clamp8(state[i] + sum over incoming (w * state[src]))``.
        The write is synchronous: every node reads the pre-tick state, so node
        ordering can never change the result.
        """
        if n < 0:
            raise ValueError(f"tick count must be >= 0, got {n}")
        for _ in range(n):
            if remember:
                self._past.append(tuple(self._state))
            incoming = self._incoming_edges()
            delta = [0] * len(self._state)
            for src, dst, weight in incoming:
                delta[dst] += weight * self._state[src]
            self._state = [
                saturate(s + d, INT8_MIN, INT8_MAX) for s, d in zip(self._state, delta)
            ]
            self.tick_count += 1
        return tuple(self._state)

    def untick(self, n: int = 1) -> tuple[int, ...]:
        """Restore ``n`` remembered states.  Raises if the history is short."""
        if n < 0:
            raise ValueError(f"untick count must be >= 0, got {n}")
        if n > len(self._past):
            raise IndexError(f"history holds {len(self._past)} ticks, cannot untick {n}")
        for _ in range(n):
            self._state = list(self._past.pop())
            self.tick_count -= 1
        return tuple(self._state)

    def lightcone(self, origin: int, max_dt: int | None = None) -> dict[int, int]:
        """Map every causally reachable node to its minimum arrival delay."""
        self._check_index(origin)
        best: dict[int, int] = {origin: 0}
        heap: list[tuple[int, int]] = [(0, origin)]
        while heap:
            t, node = heapq.heappop(heap)
            if t > best.get(node, -1):
                continue
            for e in self._edges:
                if e.src != node:
                    continue
                arrive = t + e.dt
                if max_dt is not None and arrive > max_dt:
                    continue
                if arrive < best.get(e.dst, 1 << 62):
                    best[e.dst] = arrive
                    heapq.heappush(heap, (arrive, e.dst))
        return best

    # -- identity -------------------------------------------------------------
    def fingerprint(self) -> str:
        """SHA-256 over the full world state -- use it to compare two runs."""
        payload = repr((self._state, self._edges, self._pos)).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def __repr__(self) -> str:
        return (
            f"SpacetimeGraph({self.name!r}, nodes={self.n_nodes}, "
            f"edges={self.n_edges}, t={self.tick_count})"
        )

    # -- internals ------------------------------------------------------------
    def _check_index(self, idx: int) -> None:
        if not isinstance(idx, int) or not 0 <= idx < len(self._state):
            raise IndexError(f"node index {idx!r} out of range (n={len(self._state)})")

    def _incoming_edges(self) -> list[tuple[int, int, int]]:
        if self._incoming is None:
            self._incoming = [(e.src, e.dst, e.weight) for e in self._edges]
        return self._incoming
