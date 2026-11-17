"""Observer frames, the Mahakala guard, and memories that outlive their clock.

The trainer has one clock; a fleet does not.  A frame here is an *explicit*
mapping from world ticks to local time -- ``local = (tick - origin) * rate >> 16``
-- so every node can print a different "now" for the same world tick and still
agree, byte for byte, on the structure underneath.  That is the whole content of
"the difference of time is not equal to the difference of time from another
observer": with linear rates the *order* of two world ticks is frame
independent, only the labels differ.  Disagreement about order needs a pair that
no signal can connect, and that question is already answered by
:meth:`~ultragraph.zerocpu.spacetime.SpacetimeGraph.lightcone` -- not by a clock.

Three other things this module refuses to fake:

* **No floats.**  Decay is ``v * DECAY_Q16 >> 16`` repeated, integer all the way
  down, and ``elapsed`` past ``MAX_DECAY_STEPS`` collapses to the floor instead
  of looping.  ``exp()`` would drift between platforms; a rational recurrence
  cannot.
* **No keyword sniffing.**  :class:`Mahakala` guards a chain of names the caller
  supplies and reports which tokens it matched; it does not decide that a text
  is sacred because it contains a word.
* **Anchors are explicit.**  :meth:`Witness.remember` takes ``anchor=True``.  A
  monument is something a caller declares, not something a substring detects.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

#: One world tick of proper time, in Q16.16.  A frame at ``RATE_ONE`` keeps time
#: with the world; ``RATE_ONE // 2`` runs at half speed; ``0`` is frozen.
RATE_ONE = 1 << 16

#: ``round(e ** -0.05 * 65536)``: the per-tick variance multiplier, as an integer.
DECAY_Q16 = 62339

#: 0.1 in Q16.16 -- below this a memory is indistinguishable from noise.
VARIANCE_FLOOR = 6554

#: 100.0 in Q16.16 -- an anchor keeps its original weight forever.
ANCHOR_VARIANCE = 100 << 16

#: Upper bound on decay iterations, so no input can spin this loop forever.
MAX_DECAY_STEPS = 4096

#: ``round(1 / ln 2 * 65536)`` -- splits an exponent into a power of two and a
#: remainder in ``[0, ln 2)``, so ``e^-x = 2^-k * e^-r`` stays integer.
INV_LN2_Q16 = 94548
#: ``round(ln 2 * 65536)``.
LN2_Q16 = 45426
#: Past 40 nats ``e^-x`` is below one Q16.16 ulp; the closed form says zero.
PADE_X_MAX = 40 << 16

#: Default chain of names the guard defends.  A caller may pass its own.
MAHAKALA_GUARDIANS = ("mother", "father", "son")

_WORD = re.compile(r"[0-9a-z]+")


def decay(
    variance_q16: int,
    elapsed: int,
    decay_q16: int = DECAY_Q16,
    floor: int = VARIANCE_FLOOR,
) -> int:
    """Apply ``elapsed`` ticks of geometric decay to a Q16.16 variance.

    Integer-only and bounded: at most :data:`MAX_DECAY_STEPS` multiplications are
    performed, and anything longer is treated as fully decayed.  Non-positive
    ``elapsed`` returns the input unchanged, so re-ticking at the same world tick
    is idempotent.
    """
    if elapsed <= 0:
        return variance_q16
    if variance_q16 <= floor:
        return variance_q16
    if elapsed > MAX_DECAY_STEPS:
        return floor
    value = variance_q16
    for _ in range(elapsed):
        value = (value * decay_q16) >> 16
        if value <= floor:
            return floor
    return value


def pade_exp_neg(x_q16: int) -> int:
    """``e^-x`` for Q16.16 ``x``, integer-only, via a ``[3/3]`` Padé approximant.

    ``x`` is split as ``k * ln2 + r`` with ``r`` in ``[0, ln2)``, so
    ``e^-x = 2^-k * P(-r) / P(r)`` with ``P(t) = 120 + 60t + 12t^2 + t^3``.
    No loop, no ``exp()``, no platform-dependent rounding: ``e^-1`` is exactly
    ``24109`` on every machine.

    Dropped from the sketch: the 40-nat guard returned ``0`` but the caller
    treated it as a weight, which is a different zero (absence of evidence vs
    evidence of absence).  Here zero means "below one ulp" and nothing else.
    """
    if not isinstance(x_q16, int) or isinstance(x_q16, bool):
        raise TypeError(f"pade_exp_neg expects int, got {type(x_q16).__name__}")
    if x_q16 <= 0:
        return RATE_ONE
    if x_q16 > PADE_X_MAX:
        return 0
    k = (x_q16 * INV_LN2_Q16) >> 32
    r = x_q16 - k * LN2_Q16
    r2 = (r * r) >> 16
    r3 = (r2 * r) >> 16
    num = 120 * RATE_ONE - 60 * r + 12 * r2 - r3
    den = 120 * RATE_ONE + 60 * r + 12 * r2 + r3
    if den <= 0:
        return 0
    pade = (num << 16) // den
    if k >= 64:
        return 0
    return pade >> k


def decay_x(elapsed: int, half_life_ticks: int) -> int:
    """The scaled decay exponent ``x = elapsed * 65536 / half_life``, clamped.

    Python has arbitrary-precision integers, so the widen-then-divide that the
    Rust side needs for ``|elapsed| > 2**47`` is free here; the clamp and the
    "older than the domain" shortcut are the parts worth sharing. Keeping the
    same name and contract on both sides is what lets the fixtures pin them
    together.
    """
    if half_life_ticks <= 0:
        raise ValueError("half_life_ticks must be positive")
    limit = (PADE_X_MAX >> 16) * half_life_ticks
    if elapsed >= limit:
        return PADE_X_MAX
    return min((elapsed << 16) // half_life_ticks, PADE_X_MAX)


def decay_pade(
    variance_q16: int,
    elapsed: int,
    half_life_ticks: int,
) -> int:
    """Closed-form variance decay: ``e^-(elapsed / half_life)``, Q16.16.

    Unlike :func:`decay`, this is not a recurrence: any ``elapsed`` costs the
    same few multiplications, so a decade-long gap cannot turn into a loop.  Use
    :func:`decay` when you want tick-exact replay, this when you want a
    half-life.
    """
    if half_life_ticks <= 0:
        raise ValueError("half_life_ticks must be positive")
    if elapsed <= 0:
        return variance_q16
    amplitude = pade_exp_neg((elapsed << 16) // half_life_ticks)
    return (variance_q16 * amplitude) >> 16


@dataclass(frozen=True)
class Relativity:
    """Two frames' readings of one world tick, and the gap between them."""

    local_a: int
    local_b: int

    @property
    def delta(self) -> int:
        return self.local_b - self.local_a

    def __str__(self) -> str:
        return f"local_a={self.local_a} local_b={self.local_b} delta={self.delta}"


class ObserverFrame:
    """A local clock: ``local = (world_tick - origin_world) * rate >> 16``."""

    __slots__ = ("id", "local_tick", "origin_world", "rate_q16")

    def __init__(
        self,
        id: str = "",
        origin_world: int = 0,
        rate_q16: int = RATE_ONE,
    ) -> None:
        if rate_q16 < 0:
            raise ValueError("rate_q16 must be non-negative")
        self.id = id
        self.origin_world = origin_world
        self.rate_q16 = rate_q16
        self.local_tick = 0

    def local_time(self, world_tick: int) -> int:
        """This frame's clock reading at ``world_tick`` (floor division, exact)."""
        return ((world_tick - self.origin_world) * self.rate_q16) // RATE_ONE

    def world_of(self, local_tick: int) -> int:
        """The earliest world tick whose reading is ``local_tick``.

        Precisely: the smallest world tick whose reading is at least
        ``local_tick`` (ceiling division, so it works for negative ticks too).
        For a reading that actually happened this is exact.  A fast clock steps
        by more than one local tick per world tick, so some local integers are
        never read at all; for those the answer reads *past* the request, which
        is detectable: ``local_time(world_of(l)) != l`` means ``l`` was never a
        reading.

        That lossiness is why the mesh carries **world ticks, not readings**.
        A peer needs the tick and its own frame; asking it to invert somebody
        else's rounded reading is how two nodes drift apart with no one to
        blame.  A frozen frame (``rate_q16 == 0``) has no earliest tick at all,
        so this refuses rather than inventing one.
        """
        if self.rate_q16 <= 0:
            raise ValueError("a frame with rate_q16 == 0 has no invertible clock")
        return self.origin_world + (local_tick * RATE_ONE + self.rate_q16 - 1) // self.rate_q16

    def observe(self, world_tick: int) -> int:
        """Read the clock and store it as the frame's current local tick."""
        self.local_tick = self.local_time(world_tick)
        return self.local_tick

    def advance(self, ticks: int = 1) -> int:
        """Move the local clock forward without consulting the world."""
        self.local_tick += ticks
        return self.local_tick

    def __repr__(self) -> str:
        return (
            f"ObserverFrame(id={self.id!r}, origin_world={self.origin_world}, "
            f"rate_q16={self.rate_q16}, local_tick={self.local_tick})"
        )


def relativity(a: ObserverFrame, b: ObserverFrame, world_tick: int) -> Relativity:
    """Both frames' readings of ``world_tick``; equality is not required."""
    return Relativity(local_a=a.local_time(world_tick), local_b=b.local_time(world_tick))


class Mahakala:
    """A guard over a chain of names -- fierce, and honest about what it saw."""

    __slots__ = ("guardians",)

    def __init__(self, guardians: tuple[str, ...] = MAHAKALA_GUARDIANS) -> None:
        self.guardians = tuple(name.lower() for name in guardians)

    def raise_guard(self, text: str) -> tuple[str, ...]:
        """Return the guardians present in ``text``, in chain order, de-duplicated.

        Matching is whole-token and case-insensitive; substrings do not count, so
        ``"seasonal"`` does not summon the ``"son"`` guard.
        """
        seen = set(_WORD.findall(text.lower()))
        return tuple(name for name in self.guardians if name in seen)

    def protects(self, text: str) -> bool:
        """``True`` when any guardian is present."""
        return bool(self.raise_guard(text))


@dataclass
class Slot:
    """One remembered thing: its birth tick, its peak weight, and its anchor bit."""

    key: str
    born_tick: int
    variance0_q16: int
    anchor: bool
    seq: int

    def variance_at(self, world_tick: int, decay_q16: int) -> int:
        """Weight at ``world_tick`` -- anchors never move, others decay from birth."""
        if self.anchor:
            return ANCHOR_VARIANCE
        return decay(self.variance0_q16, world_tick - self.born_tick, decay_q16)


class Witness:
    """Memory as a witness keeps it: what was said, when, and what must not fade.

    Weight is recomputed from ``(born_tick, world_tick)`` rather than mutated in
    place, so ``tick`` is idempotent and replaying a timeline cannot accumulate
    rounding drift.
    """

    __slots__ = ("_seq", "_slots", "decay_q16")

    def __init__(self, decay_q16: int = DECAY_Q16) -> None:
        if decay_q16 < 0 or decay_q16 > RATE_ONE:
            raise ValueError("decay_q16 must be within [0, RATE_ONE]")
        self.decay_q16 = decay_q16
        self._slots: dict[str, Slot] = {}
        self._seq = 0

    def remember(
        self,
        key: str,
        world_tick: int,
        *,
        span: int = 0,
        anchor: bool = False,
    ) -> Slot:
        """Store or replace ``key``; ``span`` is the caller's weight estimate.

        ``span`` is a plain integer (typically a character count) scaled into
        Q16.16 and capped at ``10.0``: long inputs are remembered harder, but no
        input outranks an anchor.
        """
        if span < 0:
            raise ValueError("span must be non-negative")
        weight = min(span * VARIANCE_FLOOR, 10 * RATE_ONE)
        if anchor:
            weight = ANCHOR_VARIANCE
        self._seq += 1
        slot = Slot(
            key=key,
            born_tick=world_tick,
            variance0_q16=weight,
            anchor=bool(anchor),
            seq=self._seq,
        )
        self._slots[key] = slot
        return slot

    def tick(self, world_tick: int) -> int:
        """Return how many live (non-anchor) slots are still above the floor."""
        live = 0
        for slot in self._slots.values():
            if slot.anchor or slot.variance_at(world_tick, self.decay_q16) > VARIANCE_FLOOR:
                live += 1
        return live

    def variance(self, key: str, world_tick: int) -> int:
        """Weight of ``key`` at ``world_tick``; ``KeyError`` if never stored."""
        slot = self._slots[key]
        return slot.variance_at(world_tick, self.decay_q16)

    def anchors(self) -> tuple[str, ...]:
        """Every declared monument, sorted so two implementations agree."""
        return tuple(sorted(key for key, slot in self._slots.items() if slot.anchor))

    def strongest(self, world_tick: int, limit: int = 1) -> list[tuple[str, int]]:
        """The heaviest ``limit`` slots at ``world_tick``, ties broken by recency."""
        ranked = sorted(
            (
                (slot.variance_at(world_tick, self.decay_q16), slot.seq, key)
                for key, slot in self._slots.items()
            ),
            reverse=True,
        )
        return [(key, weight) for weight, _seq, key in ranked[: max(0, limit)]]

    def __len__(self) -> int:
        return len(self._slots)

    def __contains__(self, key: object) -> bool:
        return key in self._slots
