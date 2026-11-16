"""Hope cascade: a text becomes integer energy, then the world does the rest.

There is no separate cascade engine here, and that is the point.  A signal is
folded into per-node ternary states, and propagation is :meth:`SpacetimeGraph.tick`
-- the same synchronous integer rule the rest of ``zerocpu`` uses.  A "resonance"
is therefore inspectable: seed, tick, read the state.

Two conventions worth stating plainly, because they are conventions and not
physics:

* **bucket rule** -- character ``i`` with codepoint ``o`` lands in lane
  ``(o * 31 + i) % n``.  The salt makes position matter, so a permutation of the
  text is a different signal rather than the same multiset.
* **polarity rule** -- even codepoints push ``+``, odd pull ``-``, each by
  ``SIGNAL_STEP`` of heat.  Any fixed rule would do; what is guaranteed is that
  the same text always yields the same lanes, the same heat and the same trits.
* **centering rule** -- the trit of a lane is the sign of its heat *minus the
  per-lane mean* (floor division, integer only), so what resonates is the lanes
  above the text's own average.  Without this every lane of a long text drifts
  to ``+1`` and the signal carries no shape.

Nothing here is a float, and nothing here classifies meaning.  ``"hope"`` and
``"Shawshank"`` are not special strings: they are inputs like any other, and the
artifact is a state, not a narration.
"""

from __future__ import annotations

from dataclasses import dataclass

from .emit import state_hex
from .spacetime import SpacetimeGraph

__all__ = [
    "Observer",
    "Resonance",
    "SIGNAL_STEP",
    "Signal",
    "observer",
    "resonate",
    "seed_signal",
    "signal",
]

#: One unit of heat per character, in Q16.16.  ``1 << 8`` keeps 256 characters
#: inside the int8 state range before saturation starts to matter.
SIGNAL_STEP = 1 << 8

#: Mixed into the lane index so position, not just content, shapes the signal.
SALT = 31


@dataclass(frozen=True)
class Signal:
    """A text projected onto ``n`` lanes: signed heat plus its ternary sign."""

    heat: tuple[int, ...]
    trits: tuple[int, ...]

    def __len__(self) -> int:
        return len(self.heat)

    def total_heat(self) -> int:
        return sum(self.heat)

    def balance(self) -> int:
        """Sum of trits: how lopsided the signal is, in [-n, +n]."""
        return sum(self.trits)


@dataclass(frozen=True)
class Resonance:
    """The trace of a cascade: what went in, and every state it passed through."""

    signal: Signal
    rounds: int
    trace: tuple[str, ...]
    collapse: int

    def settled(self) -> str:
        return self.trace[-1] if self.trace else ""


def signal(text: str, n: int) -> Signal:
    """Project ``text`` onto ``n`` lanes.  Deterministic; no floats, no keywords."""
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    if not isinstance(text, str):
        raise TypeError(f"signal expects str, got {type(text).__name__}")
    heat = [0] * n
    for i, ch in enumerate(text):
        o = ord(ch)
        heat[(o * SALT + i) % n] += SIGNAL_STEP if (o & 1) == 0 else -SIGNAL_STEP
    mean = sum(heat) // n  # floor, matching ``i64::div_euclid`` on the Rust side
    trits = tuple(
        0 if (h - mean) == 0 else (1 if (h - mean) > 0 else -1) for h in heat
    )
    return Signal(heat=tuple(heat), trits=trits)


def seed_signal(graph: SpacetimeGraph, sig: Signal) -> None:
    """Write a signal's ternary sign into the world.  ``n`` must match exactly."""
    if len(sig) != graph.n_nodes:
        raise ValueError(f"signal has {len(sig)} lanes, world has {graph.n_nodes} nodes")
    for i, trit in enumerate(sig.trits):
        graph.set_state(i, trit)


class Observer:
    """The read side: collapse a world state to one of ``{-1, 0, +1}``."""

    __slots__ = ("_label",)

    def __init__(self, label: str = "") -> None:
        self._label = label

    def of_node(self, graph: SpacetimeGraph, node: int) -> int:
        state = graph.state()[node]
        return 0 if state == 0 else (1 if state > 0 else -1)

    def of_world(self, graph: SpacetimeGraph) -> int:
        total = sum(1 if s > 0 else (-1 if s < 0 else 0) for s in graph.state())
        return 0 if total == 0 else (1 if total > 0 else -1)

    def __repr__(self) -> str:
        return f"Observer({self._label!r})"


def resonate(
    graph: SpacetimeGraph,
    text: str,
    *,
    rounds: int = 1,
    observer: Observer | None = None,
) -> Resonance:
    """Seed the world with ``text``, tick ``rounds`` times, and record the trace."""
    if rounds < 0:
        raise ValueError(f"rounds must be >= 0, got {rounds}")
    sig = signal(text, graph.n_nodes)
    seed_signal(graph, sig)
    trace = [state_hex(graph)]
    for _ in range(rounds):
        graph.tick()
        trace.append(state_hex(graph))
    obs = observer if observer is not None else Observer()
    return Resonance(
        signal=sig,
        rounds=rounds,
        trace=tuple(trace),
        collapse=obs.of_world(graph),
    )


def observer(graph: SpacetimeGraph) -> int:
    """Convenience: collapse this world right now."""
    return Observer().of_world(graph)
