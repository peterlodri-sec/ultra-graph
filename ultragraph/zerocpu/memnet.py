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
