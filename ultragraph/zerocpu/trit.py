"""The ternary quantizer: Q-delta thresholds and the A6P-15A bitplane layout.

Two halves of the same idea.

``quantize`` maps an integer to ``{-1, 0, +1}`` by comparing it against a
threshold ``delta`` -- the coarse "does this matter" decision every Caveman
kernel makes before it does any arithmetic.

``bitplanes`` goes the other way at full resolution: it explodes up to 16
signed magnitudes into fifteen magnitude planes plus one sign plane (A6P-15A),
so a dot product becomes a popcount instead of a multiply.

No floating point is used anywhere.  ``delta`` is a Q16.16 integer and every
operation here is exact.
"""

from __future__ import annotations

from collections.abc import Sequence

#: The three ternary values, as plain ints.  A trit is an ``i8`` on the Rust side.
TRIT_NEG = -1
TRIT_ZERO = 0
TRIT_ONE = 1

#: A6P detail levels.  Level 0 is the coarsest threshold, level 14 the finest.
A6P_LEVELS = 15
#: Fifteen magnitude planes plus one sign plane.
A6P_PLANES = A6P_LEVELS + 1
#: Largest magnitude representable in the magnitude planes.
MAGNITUDE_MAX = (1 << A6P_LEVELS) - 1
#: Every plane is a 16-bit word: at most sixteen lanes per vector.
MAX_LANES = 16
WORD_MASK = 0xFFFF

#: Name of the layout, for transcripts and error messages.
A6P_NAME = f"A6P-{A6P_LEVELS}A"


def delta_for_level(level: int) -> int:
    """Threshold for A6P detail ``level``: ``16384`` at 0 down to ``1`` at 14."""
    if not 0 <= level < A6P_LEVELS:
        raise ValueError(f"level must be in 0..{A6P_LEVELS - 1}, got {level}")
    return 1 << (A6P_LEVELS - 1 - level)


def quantize(value: int, delta: int) -> int:
    """Ternary threshold: ``+1`` at or above ``delta``, ``-1`` at or below ``-delta``.

    ``delta == 0`` degenerates to the plain sign, with ``0`` still mapping to
    ``0`` -- an all-zero threshold must not turn silence into a positive trit.
    """
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"quantize expects int, got {type(value).__name__}")
    if not isinstance(delta, int) or isinstance(delta, bool):
        raise TypeError(f"quantize expects an int delta, got {type(delta).__name__}")
    if delta < 0:
        raise ValueError(f"delta must be >= 0, got {delta}")
    if delta == 0:
        if value == 0:
            return TRIT_ZERO
        return TRIT_ONE if value > 0 else TRIT_NEG
    if value >= delta:
        return TRIT_ONE
    if value <= -delta:
        return TRIT_NEG
    return TRIT_ZERO


def quantize_many(values: Sequence[int], delta: int) -> list[int]:
    """`quantize` over a vector.  Same threshold for every lane."""
    return [quantize(value, delta) for value in values]


def decay(value: int, shift: int) -> int:
    """Shrink toward zero by ``1 - 2**-shift``.

    The integer stand-in for ``v *= exp(-dt)``: ``shift == 1`` halves, ``shift
    == 2`` quarters, and so on.  Values are treated as ``i64``; a shift of 63
    or more decays everything to zero.
    """
    if not isinstance(shift, int) or isinstance(shift, bool):
        raise TypeError(f"decay expects an int shift, got {type(shift).__name__}")
    if shift < 0:
        raise ValueError(f"shift must be >= 0, got {shift}")
    if shift >= 63 or value == 0:
        return 0
    lost = abs(value) >> shift
    return value - (lost if value > 0 else -lost)


def bitplanes(values: Sequence[int]) -> list[int]:
    """Explode up to :data:`MAX_LANES` values into :data:`A6P_PLANES` words.

    Plane ``l`` (0..14) carries bit ``l`` of each magnitude; plane 15 carries
    the sign bit.  Magnitudes saturate at :data:`MAGNITUDE_MAX`, and missing
    lanes are padded with zero, so a short vector and a padded vector are the
    same bytes.
    """
    if len(values) > MAX_LANES:
        raise ValueError(f"at most {MAX_LANES} lanes, got {len(values)}")
    planes = [0] * A6P_PLANES
    for lane, value in enumerate(values):
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError(f"lane {lane} is {type(value).__name__}, expected int")
        magnitude = abs(value)
        if magnitude > MAGNITUDE_MAX:
            magnitude = MAGNITUDE_MAX
        bit = 1 << lane
        for level in range(A6P_LEVELS):
            if (magnitude >> level) & 1:
                planes[level] |= bit
        if value < 0:
            planes[A6P_LEVELS] |= bit
    return planes


def from_planes(planes: Sequence[int]) -> list[int]:
    """Rebuild sixteen signed values from the A6P-15A words."""
    if len(planes) != A6P_PLANES:
        raise ValueError(f"expected {A6P_PLANES} planes, got {len(planes)}")
    words = [word & WORD_MASK for word in planes]
    sign_plane = words[A6P_LEVELS]
    out: list[int] = []
    for lane in range(MAX_LANES):
        bit = 1 << lane
        magnitude = 0
        for level in range(A6P_LEVELS):
            if words[level] & bit:
                magnitude |= 1 << level
        out.append(-magnitude if sign_plane & bit else magnitude)
    return out


def plane_parity(planes: Sequence[int]) -> int:
    """XOR fold of every plane: a 16-bit checksum over the whole vector."""
    parity = 0
    for word in planes:
        parity ^= word & WORD_MASK
    return parity & WORD_MASK
