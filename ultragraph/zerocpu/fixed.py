"""Deterministic Q16.16 fixed-point arithmetic -- the ZeroCPU numeric floor.

Training in ultragraph uses float32; deployed graphs store ternary ``{-1,0,+1}``
weights and int8 activations.  This module covers the third case: code paths
that must be *bit-identical* on every machine, CPU, and Python build.

A value is a plain Python ``int`` holding ``real * 2**16``.  Python ints are
arbitrary precision, so there is no wraparound to reason about; call
:func:`saturate` when a fixed width is required (int8 node bytes, for example).

Rounding is symmetric about zero (see :func:`fx_mul`), so ``-x`` behaves like the
negation of ``x``.  IEEE-754 does not have that property (``0.1 + 0.2 != 0.3``),
which is precisely how two observers running the same program end up
disagreeing about the same universe.
"""

from __future__ import annotations

import math
from fractions import Fraction

FRAC_BITS = 16
ONE = 1 << FRAC_BITS
_HALF = ONE >> 1

__all__ = [
    "FRAC_BITS",
    "ONE",
    "to_fx",
    "to_float",
    "to_int",
    "fx_add",
    "fx_sub",
    "fx_mul",
    "fx_div",
    "fx_dot",
    "fx_matvec",
    "fx_sqrt",
    "fx_dist",
    "saturate",
]


def _round_shift(product: int) -> int:
    """Divide a raw Q32 product by ``2**16`` with round-half-away-from-zero."""
    if product >= 0:
        return (product + _HALF) >> FRAC_BITS
    return -((-product + _HALF) >> FRAC_BITS)


def _round_div(num: int, den: int) -> int:
    """Integer division rounded half away from zero.  ``den`` must be non-zero."""
    if den == 0:
        raise ZeroDivisionError("fixed-point division by zero")
    q, r = divmod(abs(num), abs(den))
    if r * 2 >= abs(den):
        q += 1
    return q if (num < 0) == (den < 0) else -q


def to_fx(value: int | float | str | Fraction) -> int:
    """Convert to a raw Q16.16 integer.

    Accepts ``int`` (exact), ``float`` (its exact binary ratio), ``"p/q"`` or
    decimal strings, and :class:`fractions.Fraction`.  Rejects NaN and inf.
    """
    if isinstance(value, bool):  # bool is an int; keep it explicit
        raise TypeError("to_fx expects a number, not bool")
    if isinstance(value, int):
        return value << FRAC_BITS
    if isinstance(value, str):
        value = Fraction(value)
    if isinstance(value, Fraction):
        return _round_div(value.numerator << FRAC_BITS, value.denominator)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("to_fx rejects NaN and infinity")
        num, den = value.as_integer_ratio()
        return _round_div(num << FRAC_BITS, den)
    raise TypeError(f"to_fx cannot convert {type(value).__name__}")


def to_float(raw: int) -> float:
    """Exact-to-float conversion of a raw Q16.16 value."""
    return int(raw) / ONE


def to_int(raw: int) -> int:
    """Nearest integer to a raw Q16.16 value (round half away from zero)."""
    return _round_div(int(raw), ONE)


def fx_add(a: int, b: int) -> int:
    return int(a) + int(b)


def fx_sub(a: int, b: int) -> int:
    return int(a) - int(b)


def fx_mul(a: int, b: int) -> int:
    """Multiply two raw values, keeping Q16.16 and rounding half away from zero."""
    return _round_shift(int(a) * int(b))


def fx_div(a: int, b: int) -> int:
    """Divide two raw values, keeping Q16.16 and rounding half away from zero."""
    b = int(b)
    if b == 0:
        raise ZeroDivisionError("fixed-point division by zero")
    return _round_div(int(a) << FRAC_BITS, b)


def fx_dot(a, b) -> int:
    """Dot product of two equal-length raw vectors; one rounding at the end.

    Accumulating the full Q32 products before shifting keeps the result exact up
    to a single rounding step instead of one rounding per multiply-add.
    """
    if len(a) != len(b):
        raise ValueError(f"length mismatch: {len(a)} != {len(b)}")
    total = 0
    for x, y in zip(a, b):
        total += int(x) * int(y)
    return _round_shift(total)


def fx_matvec(matrix, vec) -> list[int]:
    """``matrix @ vec`` for raw values; rows may be any sequence (list, ndarray)."""
    return [fx_dot(row, vec) for row in matrix]


def fx_sqrt(value: int) -> int:
    """Square root of a raw value, staying in Q16.16.

    For a non-negative raw ``v = r * 2**16`` the raw root is ``sqrt(r) * 2**16``,
    and ``isqrt(v * 2**16)`` lands on it exactly for perfect squares.
    """
    v = int(value)
    if v < 0:
        raise ValueError("fx_sqrt of a negative value")
    return math.isqrt(v << FRAC_BITS)


def fx_dist(a, b) -> int:
    """Euclidean distance between two raw coordinate tuples, as a raw value."""
    if len(a) != len(b):
        raise ValueError(f"dimension mismatch: {len(a)} != {len(b)}")
    sq = 0
    for x, y in zip(a, b):
        d = int(x) - int(y)
        sq += d * d
    return math.isqrt(sq)


def saturate(value: int, lo: int, hi: int) -> int:
    """Clamp to ``[lo, hi]``.  Deterministic, order-independent, idempotent."""
    if lo > hi:
        raise ValueError(f"empty range [{lo}, {hi}]")
    v = int(value)
    if v < lo:
        return lo
    if v > hi:
        return hi
    return v
