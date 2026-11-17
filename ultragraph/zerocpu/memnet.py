"""MemNet: sequence-ordered associative memory that merges without a clock.

Two nodes that have been talking to different people must still agree on what
they know once they meet.  MemNet gets there with one monotonic counter per
node: every write carries a sequence number, and :meth:`MemNet.merge` keeps the
higher rank.  The rank is ``(seq, value)`` -- sequence first, value as a total
tie-break -- which makes merge commutative, associative and idempotent.  No
wall clock, no vector clock, no conflict resolution prompt.

Keys are identified by their 32-bit FNV-1a id.  A hash collision is therefore
treated as the same key; at 256 slots that costs nothing and keeps the wire
format four integers wide.
"""

from __future__ import annotations

from dataclasses import dataclass

MEMNET_SLOTS = 256
MEMNET_VERSION = 1

FNV_OFFSET = 0x811C9DC5
FNV_PRIME = 0x01000193
MASK32 = 0xFFFFFFFF

INT32_MIN = -(1 << 31)
INT32_MAX = (1 << 31) - 1


def key_id(key: str) -> int:
    """32-bit FNV-1a of the UTF-8 bytes of ``key``."""
    if not isinstance(key, str):
        raise TypeError(f"key must be str, got {type(key).__name__}")
    digest = FNV_OFFSET
    for byte in key.encode("utf-8"):
        digest = ((digest ^ byte) * FNV_PRIME) & MASK32
    return digest


@dataclass(frozen=True)
class MemNetEntry:
    """One association: which slot, which key, what value, written when."""

    slot: int
    key_id: int
    value: int
    seq: int

    def rank(self) -> tuple[int, int]:
        """Merge order: sequence dominates, value breaks ties totally."""
        return (self.seq, self.value)

    def to_text(self) -> str:
        return f"{self.slot}:{self.key_id}:{self.value}:{self.seq}"


class MemNet:
    """A mergeable key -> int32 store.  One entry per key id."""

    __slots__ = ("_entries", "_seq")

    def __init__(self) -> None:
        self._entries: dict[int, MemNetEntry] = {}
        self._seq = 0

    def put(self, key: str, value: int) -> MemNetEntry:
        """Store ``value`` under ``key`` and return the entry that was written."""
        if not isinstance(key, str):
            raise TypeError(f"key must be str, got {type(key).__name__}")
        if not key:
            raise ValueError("key must not be empty")
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError(f"value must be int, got {type(value).__name__}")
        if not INT32_MIN <= value <= INT32_MAX:
            raise ValueError(f"value must fit int32, got {value}")
        kid = key_id(key)
        entry = MemNetEntry(
            slot=kid % MEMNET_SLOTS, key_id=kid, value=value, seq=self._seq
        )
        self._seq += 1
        self._entries[kid] = entry
        return entry

    def get(self, key: str) -> int | None:
        entry = self._entries.get(key_id(key))
        return None if entry is None else entry.value

    def entries(self) -> list[MemNetEntry]:
        """Every entry, sorted by ``(slot, key_id, seq)`` -- never dict order."""
        return sorted(self._entries.values(), key=lambda e: (e.slot, e.key_id, e.seq))

    def merge(self, other: "MemNet") -> int:
        """Fold ``other`` in; return how many entries changed.

        Idempotent and order-independent: merging the same peer twice is a
        no-op, and ``a.merge(b)`` leaves the same state as ``b.merge(a)``.
        """
        changed = 0
        for incoming in other._entries.values():
            current = self._entries.get(incoming.key_id)
            if current is None or incoming.rank() > current.rank():
                self._entries[incoming.key_id] = incoming
                changed += 1
        if self._entries:
            highest = max(entry.seq for entry in self._entries.values())
            if self._seq <= highest:
                self._seq = highest + 1
        return changed

    def to_hex(self) -> str:
        """Wire form: ``slot:keyid:value:seq`` entries joined by commas."""
        return ",".join(entry.to_text() for entry in self.entries())

    @classmethod
    def from_hex(cls, text: str) -> "MemNet":
        """Inverse of :meth:`to_hex`.  Raises ``ValueError`` on anything else."""
        net = cls()
        if not isinstance(text, str):
            raise TypeError(f"from_hex expects str, got {type(text).__name__}")
        if not text:
            return net
        for chunk in text.split(","):
            parts = chunk.split(":")
            if len(parts) != 4:
                raise ValueError(f"bad entry {chunk!r}")
            try:
                slot, kid, value, seq = (int(part, 10) for part in parts)
            except ValueError as exc:
                raise ValueError(f"bad entry {chunk!r}") from exc
            if not 0 <= slot < MEMNET_SLOTS:
                raise ValueError(f"slot out of range in {chunk!r}")
            if not 0 <= kid <= MASK32:
                raise ValueError(f"key id out of range in {chunk!r}")
            if not INT32_MIN <= value <= INT32_MAX:
                raise ValueError(f"value out of range in {chunk!r}")
            if seq < 0:
                raise ValueError(f"sequence must be >= 0 in {chunk!r}")
            if kid in net._entries:
                raise ValueError(f"duplicate key id {kid} in {text!r}")
            net._entries[kid] = MemNetEntry(slot=slot, key_id=kid, value=value, seq=seq)
        if net._entries:
            net._seq = max(entry.seq for entry in net._entries.values()) + 1
        return net

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and key_id(key) in self._entries

    def __repr__(self) -> str:
        return f"MemNet({len(self._entries)} entries)"


#: Fixed ring size for :class:`RingMemNet` -- the whole memory fits in L1.
MEMNET_RING = 64


@dataclass
class RingSlot:
    """One ring entry: a signed amplitude, when it was written, and its anchor bit."""

    id: int
    amp_q16: int
    born_tick: int
    anchor: bool

    @property
    def trit(self) -> int:
        """The sign of the amplitude, as ``-1``, ``0`` or ``+1``."""
        return (self.amp_q16 > 0) - (self.amp_q16 < 0)


class RingMemNet:
    """Fixed-capacity, allocation-free MemNet with protected anchors.

    Where :class:`MemNet` answers "who wrote last", this answers "what is left
    after time": amplitudes decay by half-life through the closed-form Padé
    exponential, anchors do not decay at all, and :meth:`consensus` collapses
    the weighted sum to a single trit.

    Two behaviours are deliberately *not* copied from the sketch:

    * a full ring of anchors returns ``None`` instead of silently dropping the
      write while still handing back an id -- a memory that lies about what it
      kept is worse than one that says no;
    * placement scans from the cursor and stops at the first free or non-anchor
      slot, so the eviction order is a function of the writes, not of the ring
      walk that happened to run.
    """

    __slots__ = ("_cursor", "_occupied", "_seq", "_slots")

    def __init__(self) -> None:
        self._slots: list[RingSlot | None] = [None] * MEMNET_RING
        # Indices of the occupied slots, in the order they were first written.
        # Reading a nearly empty ring then walks a handful of entries instead of
        # branching over 64 empty ones.
        self._occupied: list[int] = []
        self._cursor = 0
        self._seq = 0

    def record(self, amp_q16: int, born_tick: int, *, anchor: bool = False) -> int | None:
        """Place a slot and return its id, or ``None`` when every slot is anchored."""
        if not isinstance(amp_q16, int) or isinstance(amp_q16, bool):
            raise TypeError(f"record expects an int amplitude, got {type(amp_q16).__name__}")
        for offset in range(MEMNET_RING):
            index = (self._cursor + offset) % MEMNET_RING
            existing = self._slots[index]
            if existing is None or not existing.anchor:
                if existing is None:
                    self._occupied.append(index)
                self._seq += 1
                self._slots[index] = RingSlot(
                    id=self._seq,
                    amp_q16=amp_q16,
                    born_tick=born_tick,
                    anchor=bool(anchor),
                )
                self._cursor = (index + 1) % MEMNET_RING
                return self._seq
        return None

    def slots(self) -> tuple[RingSlot, ...]:
        """The occupied slots, in ring order."""
        return tuple(slot for slot in self._slots if slot is not None)

    def variance(self, now: int, half_life_ticks: int, rate_q16: int | None = None) -> int:
        """``E_v0[t]``: the weighted sum of every remembered amplitude at ``now``.

        Anchors contribute their full amplitude forever.  Everything else decays
        as ``e^-(elapsed / half_life)`` and is then scaled by the observer's
        ``rate_q16`` -- the ``dT/dt`` term.  Every step is integer.
        """
        from .observer import RATE_ONE, pade_exp_neg

        rate = RATE_ONE if rate_q16 is None else rate_q16
        if half_life_ticks <= 0:
            raise ValueError("half_life_ticks must be positive")
        if rate < 0:
            raise ValueError("rate_q16 must be non-negative")
        total = 0
        for index in self._occupied:
            slot = self._slots[index]
            if slot is None:
                continue
            if slot.anchor:
                amplitude = RATE_ONE
            else:
                elapsed = now - slot.born_tick
                amplitude = (
                    RATE_ONE if elapsed <= 0 else pade_exp_neg((elapsed << 16) // half_life_ticks)
                )
            term = (slot.amp_q16 * amplitude) >> 16
            total += (term * rate) >> 16
        return total

    def variance_world(self, now_world: int, half_life_ticks: int) -> int:
        """``E_v0[t]`` with **no** rate scaling: the number every node agrees on.

        `variance` multiplies by an observer's ``rate_q16`` -- correct for a
        local reading, fatal for a verdict, because two frames with different
        rates would then disagree about the same band at the same world tick and
        no amount of message exchange could reconcile them.  Consensus therefore
        consumes **world ticks only**; a frame may relabel a tick, it may not
        change how much of it has passed.

        Consequence worth stating plainly: there is nothing to synchronise.
        Two nodes that exchange a world tick and a half-life compute the same
        integer on both sides, offline, with no clock authority anywhere.
        """
        from .observer import RATE_ONE

        return self.variance(now_world, half_life_ticks, RATE_ONE)

    def consensus_world(self, now_world: int, half_life_ticks: int) -> int:
        """The settled trit, ``+1`` / ``0`` / ``-1``, from `variance_world`.

        The sign of the weighted sum -- not "is any anchor alive".  An anchor
        that has been outvoted contributes its amplitude and still loses.
        """
        total = self.variance_world(now_world, half_life_ticks)
        return (total > 0) - (total < 0)

    def local_variance(
        self, frame, now_world: int, half_life_ticks: int
    ) -> int:
        """This observer's private reading: `variance` scaled by its own rate.

        Legitimate for display ("time runs fast here"), never for agreement.
        Kept a separate method so the rate cannot be passed by accident to a
        path that then calls itself consensus.
        """
        return self.variance(now_world, half_life_ticks, frame.rate_q16)

    def consensus(self, now: int, half_life_ticks: int, rate_q16: int | None = None) -> int:
        """The settled trit: ``+1``, ``0`` or ``-1`` from the current variance.

        Prefer `consensus_world` when the answer must be the same on every node.
        """
        total = self.variance(now, half_life_ticks, rate_q16)
        return (total > 0) - (total < 0)

    def __len__(self) -> int:
        return len(self._occupied)
